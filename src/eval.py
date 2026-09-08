"""Evaluation script for surface defect segmentation.

Usage:
    python -m src.eval --config configs/train.yaml --checkpoint runs/best.pt

Metrics computed:
  * mIoU (mean Intersection over Union)
  * per-class IoU
  * per-class recall (sensitivity)
  * overall accuracy

JSON output is written to ``results/`` with a timestamp and git SHA.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import yaml

from src.data.unify import CLASSES

logger = logging.getLogger(__name__)

NUM_CLASSES = len(CLASSES) + 1  # 6 defect + background


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return (num_classes, num_classes) confusion matrix.

    Entry [i, j] = count of pixels with true label i predicted as j.
    """
    mask = (targets >= 0) & (targets < num_classes)
    cm = torch.bincount(
        num_classes * targets[mask].long() + preds[mask].long(),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def confusion_to_iou(cm: torch.Tensor) -> torch.Tensor:
    """Per-class IoU from confusion matrix."""
    intersection = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection / union.clamp(min=1)
    return iou


def confusion_to_recall(cm: torch.Tensor) -> torch.Tensor:
    """Per-class recall (sensitivity) from confusion matrix."""
    intersection = torch.diag(cm)
    total = cm.sum(dim=1).clamp(min=1)
    return intersection / total


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.eval()
    total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            output = model(images)
            if isinstance(output, dict):
                logits = output["out"]
            else:
                logits = output

        preds = logits.argmax(dim=1)
        cm = compute_confusion_matrix(preds, masks, NUM_CLASSES)
        total_cm += cm

    iou = confusion_to_iou(total_cm)
    recall = confusion_to_recall(total_cm)
    miou = iou[1:].mean().item()  # exclude background
    overall_acc = total_cm.diag().sum().item() / total_cm.sum().item()

    per_class_iou: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    for i, cname in enumerate(CLASSES, start=1):
        per_class_iou[cname] = round(iou[i].item(), 4)
        per_class_recall[cname] = round(recall[i].item(), 4)

    return {
        "mIoU": round(miou, 4),
        "per_class_iou": per_class_iou,
        "per_class_recall": per_class_recall,
        "overall_accuracy": round(overall_acc, 4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parent.parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "unknown"


def get_gpu_info() -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {"gpu": "N/A", "driver": "N/A"}
    name = torch.cuda.get_device_name(0)
    driver = torch.version.cuda or "N/A"
    return {"gpu": name, "driver": driver}


def write_results(metrics: Dict[str, Any], tag: str = "eval") -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = get_git_sha()
    gpu = get_gpu_info()

    result = {
        "timestamp": ts,
        "git_sha": sha,
        "gpu": gpu["gpu"],
        "driver": gpu["driver"],
        "tag": tag,
        **metrics,
    }

    os.makedirs("results", exist_ok=True)
    fname = f"results/{tag}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Results written to %s", fname)
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str, checkpoint: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(config_path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    input_size = tuple(cfg.get("input_size", [256, 256]))
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model and load weights
    from src.models.build import build_model

    model = build_model(num_classes=cfg["model"]["num_classes"], pretrained_backbone=False)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded checkpoint: %s (epoch %s)", checkpoint, ckpt.get("epoch"))

    # Build eval dataset (uses val split from the same seed)
    data_cfg_path = cfg.get("data_config", "configs/data.yaml")
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    split_seed = data_cfg.get("split_seed", 42)
    from src.data.splits import controlled_split, naive_split
    from src.train import TrainDataset

    # Rebuild sample list (same logic as train.py)
    import csv

    data_root = data_cfg.get("paths", {}).get("severstal", "data/severstal")
    samples: List[Dict[str, Any]] = []
    img_dir = os.path.join(data_root, "train_images")
    csv_path = os.path.join(data_root, "train.csv")

    class_map = {
        "1": "pitted_surface",
        "2": "crazing",
        "3": "scratches",
        "4": "patches",
    }

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["ImageId_ClassId"].rsplit("_", 1)[0]
                cid = row["ImageId_ClassId"].rsplit("_", 1)[1]
                cname = class_map.get(cid)
                if cname is None:
                    continue
                samples.append(
                    {
                        "source": "severstal",
                        "class_name": cname,
                        "group_id": f"severstal/{fname}",
                        "image_rel": f"train_images/{fname}",
                        "mask_rel": f"train_images/{fname}",
                    }
                )

    split_strategy = cfg.get("split_strategy", "controlled")
    if split_strategy == "controlled":
        _, val_samples, _ = controlled_split(samples, split_seed)
    else:
        _, val_samples, _ = naive_split(samples, split_seed)

    val_ds = TrainDataset(val_samples, data_root, input_size, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    amp_enabled = cfg.get("amp", True) and device.type == "cuda"

    # Run evaluation
    metrics = evaluate(model, val_loader, device, amp_enabled)

    logger.info("mIoU: %.4f", metrics["mIoU"])
    logger.info("Overall accuracy: %.4f", metrics["overall_accuracy"])
    for cname in CLASSES:
        iou_val = metrics["per_class_iou"].get(cname, 0.0)
        rec_val = metrics["per_class_recall"].get(cname, 0.0)
        logger.info("  %-20s  IoU=%.4f  recall=%.4f", cname, iou_val, rec_val)

    # Write JSON
    tag = os.path.splitext(os.path.basename(checkpoint))[0]
    write_results(metrics, tag=tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint)
