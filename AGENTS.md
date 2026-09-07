# Surface Defect Segmentation

## Rules
- Python 3.11, PyTorch, type hints on public functions
- All hyperparameters in configs/*.yaml - never hardcode
- Every metric writes JSON to results/ with a timestamp and git SHA
- Fixed seeds everywhere; log the seed
- No new dependencies without asking

## Benchmark conditions (never change silently)
- Input: 512x512, batch 1
- 50 warmup + 500 timed iterations
- Report mean FPS and p95 latency in ms
- Record GPU name and driver version in every result file

## Data
6 classes: [fill in after unify.py]
Sources: NEU-DET, DAGM 2007, Severstal, MVTec AD, + synthetic
