\# Surface Defect Segmentation



Multi-source surface defect segmentation, unifying steel and industrial

defect datasets into a single 6-class label space, with a controlled

train/val/test split, TensorRT-optimized inference, and robustness

evaluation under common real-world corruptions.



\## Class space



Unified into the canonical NEU-DET defect vocabulary:



\- crazing

\- inclusion

\- patches

\- pitted\_surface

\- rolled-in\_scale

\- scratches



\## Data sources



| Source | Annotation type | Role |

|---|---|---|

| NEU-DET | Bounding boxes | Pretraining / detection only — not used for mIoU |

| DAGM 2007 | Weak elliptical labels, no semantic class names | Pretraining only — \*\*excluded from mIoU evaluation\*\* |

| Severstal | Pixel masks (RLE) | Primary — trains and evaluates |

| MVTec AD | Pixel masks | Primary — trains and evaluates |

| SD-saliency-900 | Pixel masks | Primary — trains and evaluates |



\### Known assumption: Severstal class mapping



Severstal's Kaggle release labels defects only as classes `1`, `2`, `3`,

`4` with no official semantic names published by the dataset authors.

The mapping used in this project (`src/data/unify.py`, `MAPPING\_SEVERSTAL`)

assigns semantic names to these four classes based on interpretations

used in some published work. \*\*This mapping is unverified\*\* and should

be treated as an assumption, not ground truth, when interpreting results

that include Severstal data.



\### DAGM 2007 exclusion from evaluation



DAGM 2007 provides only rough elliptical regions marking defect presence,

not pixel-accurate masks. It is used for pretraining/pretext tasks only.

`configs/data.yaml: eval\_sources` excludes DAGM to prevent silently

computing mIoU against weak labels.



\## Setup



See `requirements.txt` for dependencies and `scripts/download\_data.sh`

for dataset download instructions.



\## Reproducing results



See `scripts/reproduce.sh` (to be added in a later session).



\## Project status



Work in progress. See `results/` for baseline and final metrics as they

are produced.

