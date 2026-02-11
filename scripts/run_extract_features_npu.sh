#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <physical_npu_id> <train|test> [limit] [out_dir]" >&2
  echo "Example: $0 2 test 3 output/features/smoke_npu" >&2
  exit 2
fi

PHYS_NPU_ID="$1"
SPLIT="$2"
LIMIT="${3:-3}"
OUT_DIR="${4:-output/features/smoke_npu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ascend runtime environment (required for NPU kernels/drivers).
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Map the physical card to process-local npu:0.
export ASCEND_RT_VISIBLE_DEVICES="$PHYS_NPU_ID"
export ASCEND_VISIBLE_DEVICES="$PHYS_NPU_ID"

echo "[extract] physical_npu_id=${PHYS_NPU_ID} mapped_device=npu:0 split=${SPLIT} limit=${LIMIT} out=${OUT_DIR}"

# IMPORTANT:
# - We use system python 3.10 where `torch_npu` is installed.
# - Do NOT use `uv run` here unless your uv env also includes torch_npu.
/usr/local/python3.10.0/bin/python -u scripts/extract_tupac_features.py \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --out-dir "$OUT_DIR" \
  --device npu:0 \
  --npu-id "$PHYS_NPU_ID"
