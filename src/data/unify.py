from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Unified 6-class space - canonical NEU-DET vocabulary (Song & Yan, 2013)
# 0: background (implied)
# 1: crazing
# 2: inclusion
# 3: patches
# 4: pitted_surface
# 5: rolled-in_scale
# 6: scratches
# ---------------------------------------------------------------------------
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# ---------------------------------------------------------------------------
# Per-source explicit mapping dicts
# Native label -> unified class name.
# Source keys are lowercase snake_case, matching configs/data.yaml paths.
# If a native label is absent from its dict, map_label() returns None -
# callers must drop that sample, not bucket it into a fake "other" class.
# ---------------------------------------------------------------------------

MAPPING_NEU_DET = {
    "crazing":         "crazing",
    "inclusion":       "inclusion",
    "patches":         "patches",
    "pitted_surface":  "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "scratches":       "scratches",
}

MAPPING_DAGM: Optional[Dict[str, str]] = None

MAPPING_SEVERSTAL = {
    "1": "pitted_surface",
    "2": "crazing",
    "3": "scratches",
    "4": "patches",
}

MAPPING_MVTEC = {
    "scratch":       "scratches",
    "crack":         "scratches",
    "contamination": "inclusion",
    "hole":          "pitted_surface",
    "fold":          "other",
}

MAPPING_SD_SALIENCY = {
    "patch":     "patches",
    "inclusion": "inclusion",
    "scratch":   "scratches",
}

SOURCE_MAPPINGS: Dict[str, Optional[Dict[str, str]]] = {
    "neu_det":     MAPPING_NEU_DET,
    "dagm":        MAPPING_DAGM,
    "severstal":   MAPPING_SEVERSTAL,
    "mvtec":       MAPPING_MVTEC,
    "sd_saliency": MAPPING_SD_SALIENCY,
}

MASK_AVAILABILITY: Dict[str, str] = {
    "neu_det":     "bounding_boxes",
    "dagm":        "weak_ellipse",
    "severstal":   "pixel_masks",
    "mvtec":       "pixel_masks",
    "sd_saliency": "pixel_masks",
}

EVAL_SOURCES = ["severstal", "sd_saliency", "synthetic"]


def map_label(source: str, original_label: str) -> Optional[str]:
    mapping = SOURCE_MAPPINGS.get(source)

    if mapping is None:
        return None

    if original_label in mapping:
        result = mapping[original_label]
        return None if result == "other" else result

    if source == "mvtec":
        for key, val in MAPPING_MVTEC.items():
            if key in original_label.lower():
                return None if val == "other" else val

    return None


def get_class_id(class_name: str) -> Optional[int]:
    if class_name not in CLASSES:
        return None
    return CLASSES.index(class_name) + 1
