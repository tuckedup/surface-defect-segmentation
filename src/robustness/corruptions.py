"""Corruption suites for robustness evaluation.

Five suites × three severity levels = 15 test conditions.
Each function accepts an (H, W, C) uint8 numpy array and severity {1,2,3},
returns the corrupted image in the same format.

Albumentations is used where a matching transform exists; otherwise
custom numpy/cv2 code is written explicitly.
"""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A


# ---------------------------------------------------------------------------
# Severity-parameter maps  (mild=1, moderate=2, severe=3)
# ---------------------------------------------------------------------------

_BLUR_GAUSSIAN: dict[int, tuple[int, int]] = {1: (3, 3), 2: (7, 7), 3: (15, 15)}
_BLUR_MOTION: dict[int, int] = {1: 5, 2: 15, 3: 25}

_BRIGHTNESS: dict[int, tuple[float, float]] = {1: (0.9, 1.1), 2: (0.7, 1.3), 3: (0.5, 1.5)}
_CONTRAST: dict[int, tuple[float, float]] = {1: (0.9, 1.1), 2: (0.7, 1.3), 3: (0.5, 1.5)}
_GAMMA: dict[int, tuple[float, float]] = {1: (0.8, 1.2), 2: (0.6, 1.5), 3: (0.4, 2.0)}

_DUST_SPECKLE_VAR: dict[int, float] = {1: 0.02, 2: 0.06, 3: 0.12}
_DUST_PARTICLE_AREA: dict[int, tuple[int, int]] = {1: (2, 5), 2: (5, 10), 3: (10, 20)}
_DUST_PARTICLE_COUNT: dict[int, tuple[int, int]] = {1: (10, 30), 2: (30, 60), 3: (60, 120)}

_OCCL_ERASE_RATIO: dict[int, float] = {1: 0.02, 2: 0.05, 3: 0.10}
_OCCL_COARSE_H: dict[int, tuple[int, int]] = {1: (8, 16), 2: (16, 32), 3: (32, 48)}
_OCCL_COARSE_W: dict[int, tuple[int, int]] = {1: (8, 16), 2: (16, 32), 3: (32, 48)}
_OCCL_COARSE_COUNT: dict[int, tuple[int, int]] = {1: (1, 3), 2: (3, 6), 3: (6, 10)}

_VIEW_PERSP_STRENGTH: dict[int, float] = {1: 0.05, 2: 0.12, 3: 0.22}
_VIEW_ROTATION: dict[int, tuple[float, float]] = {1: (-3, 3), 2: (-8, 8), 3: (-15, 15)}
_VIEW_SCALE: dict[int, tuple[float, float]] = {1: (0.95, 1.05), 2: (0.88, 1.12), 3: (0.80, 1.20)}


# ---------------------------------------------------------------------------
# Suite 1 – Blur
# ---------------------------------------------------------------------------

def gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian blur.  Severity controls kernel size."""
    ksize = _BLUR_GAUSSIAN[severity]
    transform = A.GaussianBlur(blur_limit=ksize, p=1.0)
    return transform(image=img)["image"]


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Apply motion blur.  Severity controls kernel length."""
    length = _BLUR_MOTION[severity]
    transform = A.MotionBlur(blur_limit=length, p=1.0)
    return transform(image=img)["image"]


# ---------------------------------------------------------------------------
# Suite 2 – Lighting
# ---------------------------------------------------------------------------

def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    """Shift brightness.  Severity widens the random range."""
    lo, hi = _BRIGHTNESS[severity]
    transform = A.RandomBrightnessContrast(
        brightness_limit=(lo - 1.0, hi - 1.0), contrast_limit=0, p=1.0
    )
    return transform(image=img)["image"]


def contrast(img: np.ndarray, severity: int) -> np.ndarray:
    """Shift contrast.  Severity widens the random range."""
    lo, hi = _CONTRAST[severity]
    transform = A.RandomBrightnessContrast(
        brightness_limit=0, contrast_limit=(lo - 1.0, hi - 1.0), p=1.0
    )
    return transform(image=img)["image"]


def gamma_shift(img: np.ndarray, severity: int) -> np.ndarray:
    """Apply gamma correction (no albumentations equivalent – custom)."""
    lo, hi = _GAMMA[severity]
    gamma = np.random.uniform(lo, hi)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(img, table)


# ---------------------------------------------------------------------------
# Suite 3 – Dust
# ---------------------------------------------------------------------------

def speckle_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Additive speckle (multiplicative) noise.  Severity controls variance."""
    var = _DUST_SPECKLE_VAR[severity]
    noise = np.random.normal(1.0, var, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def dust_particles(img: np.ndarray, severity: int) -> np.ndarray:
    """Small white occluding particles (no albumentations equivalent)."""
    h, w = img.shape[:2]
    area_lo, area_hi = _DUST_PARTICLE_AREA[severity]
    count_lo, count_hi = _DUST_PARTICLE_COUNT[severity]
    n_particles = np.random.randint(count_lo, count_hi + 1)

    canvas = img.copy()
    for _ in range(n_particles):
        rh = np.random.randint(area_lo, area_hi + 1)
        rw = np.random.randint(area_lo, area_hi + 1)
        ry = np.random.randint(0, max(h - rh, 1))
        rx = np.random.randint(0, max(w - rw, 1))
        canvas[ry : ry + rh, rx : rx + rw] = 255
    return canvas


# ---------------------------------------------------------------------------
# Suite 4 – Occlusion
# ---------------------------------------------------------------------------

def random_erasing(img: np.ndarray, severity: int) -> np.ndarray:
    """Random rectangular erasing to black.  Severity controls area ratio."""
    ratio = _OCCL_ERASE_RATIO[severity]
    h, w = img.shape[:2]
    area = h * w * ratio
    aspect = np.random.uniform(0.3, 3.3)
    eh = int(np.sqrt(area * aspect))
    ew = int(np.sqrt(area / aspect))
    eh = min(eh, h - 1)
    ew = min(ew, w - 1)
    top = np.random.randint(0, h - eh)
    left = np.random.randint(0, w - ew)
    canvas = img.copy()
    canvas[top : top + eh, left : left + ew] = 0
    return canvas


def coarse_dropout(img: np.ndarray, severity: int) -> np.ndarray:
    """Multiple rectangular patches set to black (albumentations CoarseDropout)."""
    h, w = img.shape[:2]
    h_lo, h_hi = _OCCL_COARSE_H[severity]
    w_lo, w_hi = _OCCL_COARSE_W[severity]
    count_lo, count_hi = _OCCL_COARSE_COUNT[severity]

    transform = A.CoarseDropout(
        num_holes_range=(count_lo, count_hi),
        hole_height_range=(h_lo, h_hi),
        hole_width_range=(w_lo, w_hi),
        fill=0,
        p=1.0,
    )
    return transform(image=img)["image"]


# ---------------------------------------------------------------------------
# Suite 5 – Viewpoint
# ---------------------------------------------------------------------------

def perspective_warp(img: np.ndarray, severity: int) -> np.ndarray:
    """Perspective distortion simulating a camera angle change."""
    strength = _VIEW_PERSP_STRENGTH[severity]
    h, w = img.shape[:2]

    def _jitter(val: float) -> float:
        return val * np.random.uniform(-strength, strength)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [_jitter(w), _jitter(h)],
        [w + _jitter(w), _jitter(h)],
        [w + _jitter(w), h + _jitter(h)],
        [_jitter(w), h + _jitter(h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def small_rotation(img: np.ndarray, severity: int) -> np.ndarray:
    """Small in-plane rotation.  Severity widens the angle range."""
    lo, hi = _VIEW_ROTATION[severity]
    angle = np.random.uniform(lo, hi)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def scale_jitter(img: np.ndarray, severity: int) -> np.ndarray:
    """Random scale then center-crop back to original size."""
    lo, hi = _VIEW_SCALE[severity]
    scale = np.random.uniform(lo, hi)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # centre crop (or pad) back to (h, w)
    if scale >= 1.0:
        cy, cx = new_h // 2, new_w // 2
        top = cy - h // 2
        left = cx - w // 2
        result = scaled[top : top + h, left : left + w]
    else:
        result = np.zeros_like(img)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        result[top : top + new_h, left : left + new_w] = scaled
    return result


# ---------------------------------------------------------------------------
# Registry – flat dict keyed by suite_name.severity_index
# ---------------------------------------------------------------------------

SUITES: dict[str, list[callable]] = {
    "blur": [gaussian_blur, motion_blur],
    "lighting": [brightness, contrast, gamma_shift],
    "dust": [speckle_noise, dust_particles],
    "occlusion": [random_erasing, coarse_dropout],
    "viewpoint": [perspective_warp, small_rotation, scale_jitter],
}

SUITENAMES: list[str] = list(SUITES.keys())
SEVERITIES: list[int] = [1, 2, 3]

# 5 suites × 3 severities = 15 conditions (product of suite funcs × severities)
# Actual condition count = sum(len(funcs) for funcs in SUITES.values()) × 3
# For evaluation we apply ALL functions in a suite to every image at each severity.


def get_condition_count() -> int:
    """Return total number of (suite, severity) conditions.

    Each suite has 3 severity levels; a condition is (suite_name, severity).
    All transforms in a suite are applied sequentially for that severity.
    """
    return len(SUITES) * len(SEVERITIES)
