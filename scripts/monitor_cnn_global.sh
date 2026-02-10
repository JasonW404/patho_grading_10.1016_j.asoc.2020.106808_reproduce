#!/usr/bin/env bash
set -euo pipefail

# Monitor CNN_global paper CV run (fold1..K).
#
# Usage:
#   scripts/monitor_cnn_global.sh <run_id> [folds]
#
# Examples:
#   scripts/monitor_cnn_global.sh 20260210_0339_global_cv3 3
#   watch -n 10 scripts/monitor_cnn_global.sh 20260210_0339_global_cv3 3

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${1:?run_id required}"
folds="${2:-3}"

run_dir="output/mf_cnn/CNN_global/runs/${run_id}"
metrics_dir="$run_dir/metrics"
models_dir="$run_dir/models"

if [[ ! -d "$run_dir" ]]; then
  echo "Missing run_dir: $run_dir" >&2
  exit 1
fi

printf "RUN=%s\n" "$run_id"

for f in $(seq 1 "$folds"); do
  csv="$metrics_dir/global_metrics_fold${f}.csv"
  log="$metrics_dir/global_metrics_fold${f}.log"
  ckpt="$models_dir/cnn_global_paper_fold${f}.pt"

  printf "\nFOLD %s\n" "$f"
  if [[ -f "$ckpt" ]]; then
    echo "  ckpt: OK ($ckpt)"
  else
    echo "  ckpt: --"
  fi

  if [[ -f "$csv" ]]; then
    # CSV contains quoted fields with commas (e.g. val_slides list), so avoid awk -F','.
    python - "$csv" <<'PY'
import csv
import sys

path = sys.argv[1]
last_epoch = None
last_row = None
final_exists = False

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (row.get("event") or "").strip().lower() == "final":
            final_exists = True
        epoch_s = (row.get("epoch") or "").strip()
        if epoch_s.isdigit():
            last_epoch = int(epoch_s)
            last_row = row

if last_epoch is None or last_row is None:
    print("  csv: present (no epoch rows yet)")
else:
    tl = (last_row.get("train_loss") or "").strip()
    ta = (last_row.get("train_acc") or "").strip()
    vl = (last_row.get("val_loss") or "").strip()
    va = (last_row.get("val_acc") or "").strip()
    print(f"  epoch: {last_epoch} | train_loss={tl} train_acc={ta} | val_loss={vl} val_acc={va}")

print("  final: YES" if final_exists else "  final: no")
PY
  else
    echo "  csv: MISSING ($csv)"
  fi

  if [[ -f "$log" ]]; then
    tail -n 3 "$log" | sed 's/^/  log: /'
  else
    echo "  log: MISSING ($log)"
  fi

done
