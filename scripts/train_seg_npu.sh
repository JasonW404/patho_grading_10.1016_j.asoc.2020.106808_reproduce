#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ascend runtime environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p output/mf_cnn

# Use system python where torch_npu is installed.
/usr/local/python3.10.0/bin/python -u -m cabcds.mf_cnn \
  --train-seg \
  --device npu \
  --batch-size 2 \
  --seg-epochs 10 \
  --seg-max-steps 1000 \
  > output/mf_cnn/train_seg_npu.log 2>&1
