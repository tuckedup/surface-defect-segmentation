"""Train / val / test split utilities.

Two strategies:
  * naive_split  – random shuffle, no grouping or stratification.
  * controlled_split – group-aware, stratified split that prevents source-
    image leakage across splits and balances class representation.

Both accept a list of sample dicts with at least ``source``, ``class_name``,
and ``group_id`` keys, and return three lists (train, val, test).
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_counts(samples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Return {class_name: count} for a list of sample dicts."""
    return dict(Counter(s["class_name"] for s in samples))


def _log_split(name: str, samples: Sequence[Dict[str, Any]]) -> None:
    counts = _class_counts(samples)
    total = len(samples)
    logger.info(
        "%s split: %d samples  |  %s",
        name,
        total,
        ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
    )


def _validate_samples(samples: Sequence[Dict[str, Any]]) -> None:
    if not samples:
        raise ValueError("samples list is empty")
    required = {"source", "class_name", "group_id"}
    first = set(samples[0].keys())
    missing = required - first
    if missing:
        raise ValueError(f"sample dicts missing keys: {missing}")


# ---------------------------------------------------------------------------
# naive_split
# ---------------------------------------------------------------------------

def naive_split(
    samples: Sequence[Dict[str, Any]],
    split_seed: int,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Random shuffle + split.  No grouping, no stratification.

    Parameters
    ----------
    samples : sequence of dicts
        Each dict must contain ``source``, ``class_name``, ``group_id``.
    split_seed : int
        Seed for reproducibility.
    ratios : (train, val, test) floats that sum to 1.0
    """
    _validate_samples(samples)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    rng = random.Random(split_seed)
    indexed = list(samples)
    rng.shuffle(indexed)

    n = len(indexed)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # remainder goes to test
    train = indexed[:n_train]
    val = indexed[n_train : n_train + n_val]
    test = indexed[n_train + n_val :]

    _log_split("train", train)
    _log_split("val", val)
    _log_split("test", test)
    return train, val, test


# ---------------------------------------------------------------------------
# controlled_split
# ---------------------------------------------------------------------------

def controlled_split(
    samples: Sequence[Dict[str, Any]],
    split_seed: int,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group-aware, stratified split.

    1. Group samples by ``group_id`` (source image).
    2. Within each group, bucket by ``class_name``.
    3. Assign each group to a split using stratified sampling at the class
       level so that every class's groups are distributed across train/val/test
       in roughly the target ratios.
    4. Collect individual samples from the assigned groups.

    This guarantees:
      * No single source image leaks across splits.
      * Class representation is approximately balanced.
    """
    _validate_samples(samples)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    # Step 1 – group by group_id, record dominant class per group
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[s["group_id"]].append(s)

    # Dominant class = most frequent class_name inside the group
    group_meta: Dict[str, str] = {}
    for gid, members in groups.items():
        class_counter = Counter(m["class_name"] for m in members)
        group_meta[gid] = class_counter.most_common(1)[0][0]

    # Step 2 – bucket groups by class
    class_to_groups: Dict[str, List[str]] = defaultdict(list)
    for gid, cname in group_meta.items():
        class_to_groups[cname].append(gid)

    # Step 3 – stratified assignment
    rng = random.Random(split_seed)
    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    for cname in sorted(class_to_groups.keys()):
        gids = class_to_groups[cname]
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_groups.extend(gids[:n_train])
        val_groups.extend(gids[n_train : n_train + n_val])
        test_groups.extend(gids[n_train + n_val :])

    def _collect(gids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for gid in gids:
            out.extend(groups[gid])
        return out

    train = _collect(train_groups)
    val = _collect(val_groups)
    test = _collect(test_groups)

    # Sanity: no group overlap
    tg, vg, eg = set(train_groups), set(val_groups), set(test_groups)
    assert tg.isdisjoint(vg), "train/val group overlap"
    assert tg.isdisjoint(eg), "train/test group overlap"
    assert vg.isdisjoint(eg), "val/test group overlap"

    _log_split("train", train)
    _log_split("val", val)
    _log_split("test", test)
    return train, val, test
