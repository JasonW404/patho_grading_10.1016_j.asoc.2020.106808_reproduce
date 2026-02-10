#!/usr/bin/env bash
set -euo pipefail

# Runs CNN_global (paper) pipeline:
# 1) Prepare global patches within top-N ROIs
# 2) Train CNN_global with K-fold CV (default: 3)
#
# Usage:
#   scripts/run_cnn_global_paper_cv3.sh [run_id]
#
# Optional env vars:
#   MAX_SLIDES (default: 0 meaning all)
#   MAX_PATCHES_PER_SLIDE (default: 200)
#   CV_FOLDS (default: 3)
#   EPOCHS (default: 30)
#   DEVICE (default: npu:0)
#   FOLD_DEVICES (default: npu:0,npu:1,npu:2 when CV_FOLDS=3)
#   PARALLEL_FOLDS (default: 1)
#   EVAL_EVERY (default: 2)
#   ENSEMBLE_STRATEGY (default: sum)
#   SKIP_PREPARE (default: 0; set 1 to reuse existing index.csv)
#   GLOBAL_PAPER_NUM_WORKERS (default: 4)
#   GLOBAL_PAPER_PREFETCH_FACTOR (default: 2)
#   GLOBAL_PAPER_PERSISTENT_WORKERS (default: 1; set 0 to disable)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${1:-20260210_$(date +%H%M)_global_cv3}"
max_slides="${MAX_SLIDES:-0}"
max_patches_per_slide="${MAX_PATCHES_PER_SLIDE:-200}"
cv_folds="${CV_FOLDS:-3}"
epochs="${EPOCHS:-30}"
device="${DEVICE:-npu:0}"
ensemble_strategy="${ENSEMBLE_STRATEGY:-sum}"
parallel_folds="${PARALLEL_FOLDS:-1}"
eval_every="${EVAL_EVERY:-2}"
skip_prepare="${SKIP_PREPARE:-0}"

gp_num_workers="${GLOBAL_PAPER_NUM_WORKERS:-4}"
gp_prefetch_factor="${GLOBAL_PAPER_PREFETCH_FACTOR:-2}"
gp_persistent_workers="${GLOBAL_PAPER_PERSISTENT_WORKERS:-1}"

fold_devices="${FOLD_DEVICES:-}"
if [[ -z "$fold_devices" && "$cv_folds" == "3" ]]; then
  fold_devices="npu:0,npu:1,npu:2"
fi

run_dir="output/mf_cnn/CNN_global/runs/${run_id}"
log_dir="logs/CNN_global/${run_id}"

mkdir -p "$run_dir/global_patches" "$run_dir/models" "$run_dir/metrics" "$log_dir"

roi_report_csv="output/roi_selector/outputs/reports/stage_two_roi_selection.csv"

prepare_log="$log_dir/prepare.log"
train_log="$log_dir/train.log"

extra_train_args=()
if [[ "$parallel_folds" != "0" ]]; then
  extra_train_args+=("--global-paper-parallel-folds")
fi
if [[ -n "$fold_devices" ]]; then
  extra_train_args+=("--global-paper-fold-devices" "$fold_devices")
fi

extra_train_args+=("--global-paper-num-workers" "$gp_num_workers")
extra_train_args+=("--global-paper-prefetch-factor" "$gp_prefetch_factor")
if [[ "$gp_persistent_workers" == "0" ]]; then
  extra_train_args+=("--global-paper-no-persistent-workers")
fi

train_py_cmd=("uv" "run" "python")
if [[ "$device" == npu:* || "$fold_devices" == *npu:* ]]; then
  # NPU runtime requires Ascend env to provide libhccl.so etc.
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  train_py_cmd=("/usr/local/python3.10.0/bin/python")
fi

index_csv="$run_dir/global_patches/index.csv"
if [[ "$skip_prepare" == "1" ]]; then
  echo "SKIP_PREPARE=1: reuse $index_csv" | tee -a "$prepare_log"
  if [[ ! -f "$index_csv" ]]; then
    echo "ERROR: SKIP_PREPARE=1 but missing index_csv: $index_csv" | tee -a "$prepare_log" >&2
    echo "Either set SKIP_PREPARE=0 to regenerate it, or copy an existing prepared index.csv into this run_dir." | tee -a "$prepare_log" >&2
    exit 2
  fi
else
  # Prepare step is CPU-bound; keep using uv env where deps are known-good.
  uv run python -m cabcds.mf_cnn \
    --prepare-global \
    --out-root "$run_dir/global_patches" \
    --index-csv "$index_csv" \
    --roi-report-csv "$roi_report_csv" \
    --roi-top-n 4 \
    --max-slides "$max_slides" \
    --max-patches-per-slide "$max_patches_per_slide" \
    --patch-size 512 \
    --overlap 80 \
    --level 0 \
    2>&1 | tee "$prepare_log"
fi

"${train_py_cmd[@]}" -m cabcds.mf_cnn \
  --train-global-paper \
  --global-paper-index-csv "$index_csv" \
  --global-paper-checkpoint "$run_dir/models/cnn_global_paper.pt" \
  --global-paper-metrics-csv "$run_dir/metrics/global_metrics.csv" \
  --global-paper-cv-folds "$cv_folds" \
  --global-paper-eval-every "$eval_every" \
  "${extra_train_args[@]}" \
  --global-paper-ensemble-strategy "$ensemble_strategy" \
  --global-paper-ensemble-manifest "$run_dir/models/cnn_global_paper.ensemble.json" \
  --global-batch-size 32 \
  --global-paper-epochs "$epochs" \
  --global-lr 1e-4 \
  --global-momentum 0.9 \
  --global-weight-decay 5e-4 \
  --device "$device" \
  2>&1 | tee "$train_log"

echo "Done. run_dir=$run_dir"
