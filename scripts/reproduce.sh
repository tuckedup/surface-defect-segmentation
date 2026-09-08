#!/usr/bin/env bash
#
# Reproduce every reported result end to end, in order.
#umbered "Step N:" lines mark each phase (see PROJECT_PLAN.md sessions).
#
# Expected outputs:
#   results/baseline.json      (Session 2, naive split, mIoU ~0.68)
#   results/final.json         (Session 5, controlled split, mIoU ~0.84)
#   results/latency_*.json     (Session 3, mean FPS + p95 latency)
#   results/robustness_*.json  (Session 4, corruption suites)
#
# Notes:
# - src/train.py and src/eval.py take only --config, and the repo ships a
#   single training config (configs/train.yaml) whose `split_strategy` field
#   selects "naive" vs "controlled". Steps 2-3 therefore toggle that field
#   at runtime with sed (backed up and restored; no extra config files).
# - src/eval.py names its output results/<checkpoint-basename>_<ts>.json,
#   so each eval output is copied to the stable results/baseline.json /
#   results/final.json names afterwards.
# - src/robustness/run_suites.py prints its JSON summary to stdout, so Step 6
#   redirects it into results/robustness_<ts>.json.

set -euo pipefail

mkdir -p results

# Safety net: always restore configs/train.yaml if Step 2 is interrupted.
restore_train_config() {
    if [ -f configs/train.yaml.bak ]; then
        mv -f configs/train.yaml.bak configs/train.yaml
    fi
}
trap restore_train_config EXIT

echo "Step 1: Data download"
bash scripts/download_data.sh

echo "Step 2: Baseline training (naive split)"
cp configs/train.yaml configs/train.yaml.bak
sed -i.bak 's/^split_strategy:.*/split_strategy: "naive"/' configs/train.yaml
python -m src.train --config configs/train.yaml
cp runs/best.pt runs/baseline_best.pt
python -m src.eval --config configs/train.yaml --checkpoint runs/baseline_best.pt
cp "$(ls -t results/baseline_best_*.json | head -n 1)" results/baseline.json
mv -f configs/train.yaml.bak configs/train.yaml

echo "Step 3: Controlled-split training"
python -m src.train --config configs/train.yaml
python -m src.eval --config configs/train.yaml --checkpoint runs/best.pt
cp "$(ls -t results/best_*.json | head -n 1)" results/final.json

echo "Step 4: ONNX export + TensorRT engine builds"
python -m src.export.to_onnx
python -m src.export.calibrate
python -m src.export.build_engine

echo "Step 5: Latency benchmark"
python -m src.bench.latency

echo "Step 6: Robustness suite evaluation"
python -m src.robustness.run_suites data/severstal/train_images --seed 42 \
    > "results/robustness_$(date -u +%Y%m%dT%H%M%SZ).json"

echo "Done. Results in results/:"
ls -t results/
