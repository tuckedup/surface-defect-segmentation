"""Run corruption suites over a set of images.

Callable interface that applies all 15 conditions (5 suites × 3 severities)
to a collection of images.  Does NOT run evaluation – just produces
corrupted image sets for Session 4 to consume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.robustness.corruptions import SUITES, SUITENAMES, SEVERITIES


def apply_suite(
    img: np.ndarray,
    suite_name: str,
    severity: int,
) -> np.ndarray:
    """Apply all transforms in a suite sequentially at the given severity.

    Args:
        img: Input image, (H, W, C) uint8.
        suite_name: One of the five suite names.
        severity: 1 (mild), 2 (moderate), or 3 (severe).

    Returns:
        Corrupted image in the same format.
    """
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'.  Choose from {SUITENAMES}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be in {SEVERITIES}, got {severity}")

    result = img.copy()
    for transform_fn in SUITES[suite_name]:
        result = transform_fn(result, severity)
    return result


def run_suites(
    images: Sequence[np.ndarray],
    *,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Apply every (suite, severity) condition to each image.

    Args:
        images: List of (H, W, C) uint8 numpy arrays.
        seed: Random seed for reproducibility.
        output_dir: If set, write corrupted images as PNGs organised by
            ``output_dir/<suite>_sev<severity>/<idx>.png``.

    Returns:
        Nested dict  ``results[suite_name][str(severity)]`` → list of
        corrupted images (one per input image), in the same order as
        *images*.
    """
    np.random.seed(seed)

    results: dict[str, dict[str, list[np.ndarray]]] = {}
    for suite_name in SUITENAMES:
        results[suite_name] = {}
        for sev in SEVERITIES:
            corrupted: list[np.ndarray] = []
            for img in images:
                corrupted.append(apply_suite(img, suite_name, sev))

            results[suite_name][str(sev)] = corrupted

            # Optional: persist to disk
            if output_dir is not None:
                folder = Path(output_dir) / f"{suite_name}_sev{sev}"
                folder.mkdir(parents=True, exist_ok=True)
                for idx, cimg in enumerate(corrupted):
                    cv2.imwrite(str(folder / f"{idx}.png"), cimg)

    return results


def summarise(results: dict) -> dict[str, dict[str, int]]:
    """Return a quick shape summary of the results dict.

    Useful for verifying that every condition produced the expected
    number of outputs without inspecting the images themselves.
    """
    summary: dict[str, dict[str, int]] = {}
    for suite_name, sev_dict in results.items():
        summary[suite_name] = {}
        for sev_key, img_list in sev_dict.items():
            summary[suite_name][sev_key] = len(img_list)
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point (for manual smoke-testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply corruption suites to images in a directory."
    )
    parser.add_argument("input_dir", help="Directory containing PNG/JPG images.")
    parser.add_argument("-o", "--output-dir", default="corrupted_output",
                        help="Where to write corrupted images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=10,
                        help="Cap on images to process (0 = all).")
    args = parser.parse_args()

    # Load images
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in exts
    )
    if args.max_images > 0:
        paths = paths[: args.max_images]

    imgs: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)

    print(f"Loaded {len(imgs)} images from {args.input_dir}")

    results = run_suites(imgs, seed=args.seed, output_dir=args.output_dir)
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    print(f"Corrupted images written to {args.output_dir}/")
