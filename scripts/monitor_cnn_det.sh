#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/monitor_cnn_det.sh <RUN_ID> [interval_seconds]
# Example:
#   scripts/monitor_cnn_det.sh 20260209_0855_seg090236_r30_relax_gtforce_hnm_v4 60

RUN_ID=${1:-}
INTERVAL=${2:-60}

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <RUN_ID> [interval_seconds]" >&2
  exit 2
fi

ROOT="/app/code/patho_grading_10.1016_j.asoc.2020.106808_reproduce"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MET="$repo_root/output/mf_cnn/CNN_det/runs/$RUN_ID/metrics/det_metrics.csv"
RUN_DIR="$repo_root/output/mf_cnn/CNN_det/runs/$RUN_ID"
LOG_DIR="$repo_root/logs/CNN_det/$RUN_ID"
OUT="$LOG_DIR/monitor.log"

mkdir -p "$LOG_DIR"

echo "[monitor] start run_id=$RUN_ID interval=${INTERVAL}s" | tee -a "$OUT"

epoch_last=""

while true; do
  ts=$(date '+%F %T')

  if [[ ! -f "$MET" ]]; then
    echo "$ts METRICS_MISSING $MET" | tee -a "$OUT"
    sleep "$INTERVAL"
    continue
  fi

  read -r epoch phase val_ap test_ap < <(python - "$MET" <<'PY'
import csv
import sys

path = sys.argv[1]
last = None
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        last = row

if not last:
    print("", "", "", "")
    raise SystemExit(0)

epoch = str(last.get("epoch") or "")
phase = str(last.get("phase") or "")
val_ap = str(last.get("val_ap") or "")
test_ap = str(last.get("test_ap") or "")
print(epoch, phase, val_ap, test_ap)
PY
  )

  if [[ -z "${epoch}${phase}${val_ap}${test_ap}" ]]; then
    echo "$ts METRICS_EMPTY" | tee -a "$OUT"
    sleep "$INTERVAL"
    continue
  fi

  # Models present?
  model_n=$(find "$RUN_DIR/models" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')

  # Only log when epoch changes, otherwise keep it quiet.
  if [[ "$epoch" != "$epoch_last" ]]; then
    epoch_last="$epoch"
    echo "$ts phase=$phase epoch=$epoch val_ap=$val_ap test_ap=$test_ap models=$model_n" | tee -a "$OUT"
  fi

  # If training ended, you might see a final-phase row + models; we still keep monitoring.
  sleep "$INTERVAL"
done
