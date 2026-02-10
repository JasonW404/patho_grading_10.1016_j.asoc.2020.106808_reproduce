#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <physical_npu_id> <seg_ckpt> [run_tag]" >&2
  echo "  Outputs under: output/mf_cnn/CNN_det/runs/<timestamp>_<run_tag>/" >&2
  exit 2
fi

PHYS_NPU_ID="$1"
SEG_CKPT="$2"
RUN_TAG="${3:-seg090236_r30}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$PHYS_NPU_ID"
export ASCEND_VISIBLE_DEVICES="$PHYS_NPU_ID"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${TS}_${RUN_TAG}}"
ROOT="output/mf_cnn/CNN_det/runs/${RUN_ID}"

DET_OUT_ROOT="${ROOT}/det_patches"
DET_INDEX_CSV="${DET_OUT_ROOT}/index.csv"
DET_CKPT="${ROOT}/models/cnn_det_paper.pt"
DET_METRICS_CSV="${ROOT}/metrics/det_metrics.csv"
LOG_DIR="logs/CNN_det/${RUN_ID}"
LOG_FILE="${LOG_DIR}/pipeline.log"

mkdir -p "${DET_OUT_ROOT}" "$(dirname "${DET_CKPT}")" "$(dirname "${DET_METRICS_CSV}")" "${LOG_DIR}"

echo "[det] physical_npu_id=${PHYS_NPU_ID} mapped_device=npu:0" | tee -a "${LOG_FILE}"
echo "[det] seg_ckpt=${SEG_CKPT}" | tee -a "${LOG_FILE}"
echo "[det] root=${ROOT}" | tee -a "${LOG_FILE}"

/usr/local/python3.10.0/bin/python -u -m cabcds.mf_cnn \
  --prepare-det \
  --device npu:0 \
  --det-seg-checkpoint "${SEG_CKPT}" \
  --det-out-root "${DET_OUT_ROOT}" \
  --det-index-csv "${DET_INDEX_CSV}" \
  --det-max-tiles 0 \
  --det-max-candidates-per-tile 200 \
  --det-max-neg-per-pos 3 \
  --det-match-radius 30 \
  2>&1 | tee -a "${LOG_FILE}"

/usr/local/python3.10.0/bin/python -u -m cabcds.mf_cnn \
  --train-det-paper \
  --device npu:0 \
  --batch-size 64 \
  --det-paper-tune-epochs 10 \
  --det-paper-final-epochs 5 \
  --det-paper-split-seed 1337 \
  --det-paper-index-csv "${DET_INDEX_CSV}" \
  --det-paper-checkpoint "${DET_CKPT}" \
  --det-paper-metrics-csv "${DET_METRICS_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[det] done. root=${ROOT}" | tee -a "${LOG_FILE}"
