# Surface Defect Segmentation

Multi-source surface defect segmentation, unifying steel and industrial
defect datasets into a single 6-class label space, with a controlled
train/val/test split, TensorRT-optimized inference, and robustness
evaluation under common real-world corruptions.

## Class space

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

## Data sources

| Source | Annotation type | Role |
|---|---|---|
| neu_det | Bounding boxes | Pretraining / detection only - not used for mIoU |
| dagm | Weak elliptical labels, no semantic class names | Pretraining only - excluded from mIoU |
| severstal | Pixel masks (RLE) | Primary - trains and evaluates |
| mvtec | Pixel masks (non-steel objects) | Pretraining only - excluded from mIoU |
| sd_saliency | Pixel masks | Primary - trains and evaluates |

### Known assumption: Severstal class mapping

Severstal's Kaggle release labels defects only as classes 1, 2, 3, 4 with
no official semantic names published by the dataset authors. The mapping
used here (src/data/unify.py, MAPPING_SEVERSTAL) assigns semantic names
based on interpretations used in some published work. This mapping is
UNVERIFIED and should be treated as an assumption, not ground truth.

### Why mvtec is excluded from evaluation

mvtec has real pixel masks but covers non-steel objects (leather, carpet,
bottles). Mapping its defect types onto steel-defect classes (e.g. hole
-> pitted_surface on a bottle) is not semantically valid for a steel
surface defect benchmark, so it's used for pretraining only.

### DAGM 2007 exclusion from evaluation

DAGM 2007 provides only rough elliptical regions marking defect presence,
not pixel-accurate masks. Used for pretraining/pretext tasks only.

## Setup

See requirements.txt for dependencies and scripts/download_data.sh for
dataset download instructions.

## Project status

Work in progress. See results/ for baseline and final metrics as they
are produced.
