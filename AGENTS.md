# Surface Defect Segmentation

See PROJECT_PLAN.md for the full session plan, data plan, and rationale.
Read that file first, then this one for hard rules.

## Rules
- Python 3.11, PyTorch, type hints on public functions
- All hyperparameters in configs/*.yaml - never hardcode
- Every metric writes JSON to results/ with a timestamp and git SHA
- Fixed seeds everywhere; log the seed
- No new dependencies without asking
- Git operations run in plain PowerShell, never through the agent

## Benchmark conditions (FROZEN - never change silently)
- Input: 256x256, batch 4
- Batch size 4 with mixed precision (AMP); gradient accumulation if a
  larger effective batch is needed
- 50 warmup + 500 timed iterations
- Report mean FPS and p95 latency in ms
- Record GPU name and driver version in every result file
- GPU: RTX 4060 Laptop, 8GB VRAM - plug in during benchmarking to avoid
  thermal-throttling noise

## Data
6 classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
Sources: NEU-DET (boxes, pretrain only), DAGM 2007 (weak ellipses, excluded
from eval), Severstal (pixel masks, primary), MVTec AD (pixel masks,
non-steel, pretrain only), SD-saliency-900 (pixel masks, primary), synthetic
(to be built, primary)
eval_sources: severstal, sd_saliency, synthetic
