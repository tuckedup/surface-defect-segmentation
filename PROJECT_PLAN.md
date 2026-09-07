# Surface Defect Segmentation - Project Plan

Save this as PROJECT_PLAN.md in the repo root. It is the single source of
truth for what gets built, in what order, and which artifact proves each
resume bullet.

---

## 0. Target deliverables

Three resume bullets, each backed by a committed artifact and a JSON result
file. Nothing goes on the resume that isn't produced by a script in this repo.

| Bullet | Claim | Proof artifact |
|---|---|---|
| 1 | 18K images, 6 classes, controlled split lifts mIoU 0.68 -> 0.84 | results/baseline.json, results/final.json |
| 2 | ONNX + TensorRT FP16/INT8: 22 -> 61 FPS, p95 45 -> 16 ms | results/latency_*.json |
| 3 | 5 corruption suites, worst-case recall 69% -> 83% | results/robustness_*.json |

Plus: public GitHub repo, README with figures, one-command reproduction.

---

## 1. Ground rules

1. Every number in the README comes from a JSON file in results/. No
   hand-typed metrics. Each JSON records the git SHA, timestamp, seed, GPU
   name, and driver version.
2. Benchmark conditions are frozen in AGENTS.md and never change silently.
3. Prove the export path before training anything. Session 3 runs on a
   randomly-initialised model.
4. One session per phase, fresh context each time. Commit between sessions.
5. Correctness-critical files get human review: splits.py, latency.py,
   corruptions.py, calibrate.py.
6. Git operations run in PowerShell, never through the agent.

---

## 2. Hardware constraints (RTX 4060 Laptop, 8GB)

- Input size: 256x256. Severstal images are natively 1600x256.
- Batch size: 4, with mixed precision (AMP). Gradient accumulation if needed.
- Plug the laptop in during benchmarking.
- INT8 calibration needs a held-out calibration set of ~500 images.

---

## 3. Data plan

### Class space
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

### Sources and their honest roles

| Source | Images | Annotation | Role |
|---|---|---|---|
| Severstal | 12,568 | RLE pixel masks | Primary - train + eval |
| SD-saliency-900 | 900 | pixel masks | Primary - train + eval (3 classes) |
| Synthetic | you generate | exact masks | Primary - fills thin classes |
| NEU-DET | 1,800 | bounding boxes | Detection head only, no mIoU |
| DAGM 2007 | ~17,100 | weak ellipses | Pretraining only, excluded from mIoU |
| MVTec AD | varies | pixel masks | Optional pretraining; non-steel |

eval_sources: [severstal, sd_saliency, synthetic]

### Documented assumptions
- Severstal classes 1-4 have no official semantic names. Unverified mapping.
- DAGM class1-class10 have no semantic defect names. Not mappable, not evaluated.

---

## 4. Session plan

### Session 1 - Data loaders - DONE
### Session 1.5 - Mapping corrections - DONE

### Session 2 - Splits + baseline training
Files: src/data/splits.py, src/train.py, src/eval.py, configs/train.yaml
- naive_split() -> 0.68 baseline
- controlled_split() -> grouped, stratified, no leakage
- train.py -> U-Net or DeepLabV3+, AMP, fixed seed, checkpoints to runs/
- eval.py -> mIoU, per-class IoU, recall, JSON with git SHA
Review splits.py by hand before trusting it.
Output: results/baseline.json, mIoU ~0.68

### Session 3 - Export + benchmark (DO EARLY, can run in parallel with Session 2)
Files: src/export/to_onnx.py, src/export/build_engine.py,
src/export/calibrate.py, src/bench/latency.py
- ONNX export, opset 17, fixed input shape, verify with onnxruntime
- TensorRT FP32 -> FP16 -> INT8
- INT8 needs entropy calibrator over ~500 held-out images
- latency.py: fixed input size, batch size, GPU, driver, 50 warmup +
  500 timed iterations, mean FPS and p95 latency
Run against an untrained checkpoint first, purely to prove the pipeline.
Output: results/latency_pytorch.json, latency_fp16.json, latency_int8.json

### Session 4 - Robustness suites
Files: src/robustness/corruptions.py, src/robustness/run_suites.py
Five suites, three severities each, test-only in this pass:
blur, lighting, dust, occlusion, viewpoint
Output: results/robustness_baseline.json

### Session 5 - Final training
Train on controlled split -> 0.84 mIoU -> results/final.json
Add corruption augmentation, retrain -> 83% worst-condition recall ->
results/robustness_final.json
Real numbers only. Do not backfill.

### Session 6 - C++ inference harness (optional)
cpp/infer.cpp - TensorRT C++ API, ring-buffered input queue

### Session 7 - Documentation and packaging
scripts/reproduce.sh, figures, final README, honest limitations section

---

## 5. Ordering rationale

Session 3 before Session 4 deliberately - export is the only phase that can
fail for reasons you cannot fix (TensorRT version mismatch, unsupported op,
insufficient VRAM). Find the hard blocker first.

---

## 6. Interview-readiness checklist

- [ ] How did you measure p95 latency?
- [ ] How do you know there's no leakage between splits?
- [ ] What does controlled split mean concretely, and why did it help?
- [ ] Which classes did INT8 hurt, and by how much?
- [ ] Why is DAGM excluded from evaluation?
- [ ] Which class has the least data, and what did you do about it?
- [ ] What's the accuracy cost of FP16 vs INT8 on your worst class?
- [ ] What are the limitations of this project?

---

## 7. Cost control

- Free tier: Gemini 3 Flash Preview via Google AI Studio, no billing.
- Rate limits are per-minute, not daily.
- Verify the model selector each session.
- Cheap model for Sessions 1 and 7. Strongest model for Sessions 3 and 4.
- Bring splits.py, latency.py, corruptions.py, calibrate.py to chat for review.

---

## 8. Next three actions

1. Fix the three silent bugs, batch_size 4, input 256x256.
2. Update AGENTS.md with frozen 256x256 benchmark conditions.
3. Commit, then start Session 2 in a fresh OpenCode window.
