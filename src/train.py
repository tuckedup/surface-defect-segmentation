"""Training script for surface defect segmentation.

Usage:
    python -m src.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import yaml

from src.data.splits import controlled_split, naive_split
from src.models.build import build_model
from src.data.unify import CLASSES, map_label

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed set to %d", seed)


# ---------------------------------------------------------------------------
# Dataset wrapper for training (returns image tensor + mask tensor)
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """Wraps a list of sample dicts into (image, mask) tensors."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_root: str,
        input_size: tuple,
        augment: bool = False,
    ):
        self.samples = samples
        self.data_root = data_root
        self.input_size = tuple(input_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2

        s = self.samples[idx]
        img_path = os.path.join(self.data_root, s["source"], s["image_rel"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.data_root, s["source"], s["mask_rel"])
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Resize
        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # To float tensors  (C, H, W) and (H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_loss(name: str) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "focal":
        return nn.CrossEntropyLoss()  # placeholder; swap for true focal
    elif name == "dice_ce":
        return nn.CrossEntropyLoss()  # placeholder; swap for dice + CE
    else:
        raise ValueError(f"Unknown loss: {name}")


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            output = model(images)
            if isinstance(output, dict):
                logits = output["out"]
            else:
                logits = output
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

        if log_every > 0 and step % log_every == 0:
            logger.info(
                "  epoch %d  step %d/%d  loss %.4f",
                epoch, step, len(loader), running_loss / n_batches,
            )

    elapsed = time.time() - t0
    avg_loss = running_loss / max(n_batches, 1)
    logger.info("Epoch %d train loss %.4f  (%.1fs)", epoch, avg_loss, elapsed)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            output = model(images)
            if isinstance(output, dict):
                logits = output["out"]
            else:
                logits = output
            loss = criterion(logits, masks)

        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    logger.info("  val loss %.4f", avg_loss)
    return avg_loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any], path: str, is_best: bool = False
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pt")
        torch.save(state, best_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    with open(config_path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Config: %s", config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data -----------------------------------------------------------
    input_size = tuple(cfg.get("input_size", [256, 256]))
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 4)

    data_cfg_path = cfg.get("data_config", "configs/data.yaml")
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    split_seed = data_cfg.get("split_seed", 42)
    data_root = data_cfg.get("paths", {}).get("severstal", "data/severstal")

    # Build sample records from the primary source (Severstal)
    # This is a skeleton; the real version will iterate all eval_sources
    samples: List[Dict[str, Any]] = []
    img_dir = os.path.join(data_root, "train_images")
    csv_path = os.path.join(data_root, "train.csv")

    if os.path.exists(img_dir):
        import csv

        # Map Severstal class IDs -> unified names
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
                            "mask_rel": f"train_images/{fname}",  # placeholder
                        }
                    )
        else:
            # Fallback: just list images
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".jpg"):
                    samples.append(
                        {
                            "source": "severstal",
                            "class_name": "mixed",
                            "group_id": f"severstal/{fname}",
                            "image_rel": f"train_images/{fname}",
                            "mask_rel": f"train_images/{fname}",
                        }
                    )

    if not samples:
        logger.warning("No samples found – placing dummy sample for smoke test")
        samples = [
            {
                "source": "dummy",
                "class_name": "crazing",
                "group_id": "dummy/0",
                "image_rel": "0.jpg",
                "mask_rel": "0.png",
            }
        ]

    logger.info("Total samples: %d", len(samples))

    # Split
    split_strategy = cfg.get("split_strategy", "controlled")
    if split_strategy == "controlled":
        train_samples, val_samples, test_samples = controlled_split(
            samples, split_seed
        )
    else:
        train_samples, val_samples, test_samples = naive_split(
            samples, split_seed
        )

    train_ds = TrainDataset(train_samples, data_root, input_size)
    val_ds = TrainDataset(val_samples, data_root, input_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Model ----------------------------------------------------------
    model = build_model(num_classes=cfg["model"]["num_classes"], pretrained_backbone=cfg["model"].get("pretrained", True)).to(device)
    criterion = build_loss(cfg.get("loss", "cross_entropy"))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg["optim"]["lr"]},
        ],
        weight_decay=cfg["optim"].get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["optim"].get("T_max", 50),
        eta_min=cfg["optim"].get("eta_min", 1e-6),
    )

    amp_enabled = cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # ---- Training loop --------------------------------------------------
    epochs = cfg.get("epochs", 50)
    val_every = cfg.get("val_every", 1)
    log_every = cfg.get("log_every", 50)
    ckpt_dir = cfg.get("checkpoint_dir", "runs")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    logger.info("Starting training for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, amp_enabled, log_every, epoch,
        )
        scheduler.step()

        if epoch % val_every == 0:
            val_loss = validate(model, val_loader, criterion, device, amp_enabled)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "seed": seed,
                    "config": cfg,
                },
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                is_best=is_best,
            )
            logger.info(
                "  checkpoint saved  (best val_loss=%.4f)", best_val_loss
            )

    # Save last
    save_checkpoint(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss if val_every > 0 else None,
            "train_loss": train_loss,
            "seed": seed,
            "config": cfg,
        },
        os.path.join(ckpt_dir, "last.pt"),
    )
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    main(args.config)
