#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <out_csv> <csv1> [csv2 ...]" >&2
  echo "Note: pass a quoted glob for shard outputs." >&2
  echo "Example:" >&2
  echo "  $0 output/features/all_test.csv 'output/features/full_run_test/shard_*_of_*/stage_four_hybrid_descriptors.csv'" >&2
  exit 2
fi

OUT_CSV="$1"
CSV_GLOB="$2"

# Expand glob safely.
shopt -s nullglob
FILES=( $CSV_GLOB )
shopt -u nullglob

if (( ${#FILES[@]} == 0 )); then
  echo "Error: no files matched: $CSV_GLOB" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"

# Write header from first file, then append remaining rows.
head -n 1 "${FILES[0]}" > "$OUT_CSV"
for f in "${FILES[@]}"; do
  tail -n +2 "$f" >> "$OUT_CSV"
done

# Optional: stable sort by group (keeps header first). Requires coreutils sort.
TMP_SORTED="$OUT_CSV.tmp"
{ head -n 1 "$OUT_CSV"; tail -n +2 "$OUT_CSV" | sort -t, -k1,1; } > "$TMP_SORTED"
mv "$TMP_SORTED" "$OUT_CSV"

echo "[merge] wrote: $OUT_CSV (rows=$(($(wc -l < "$OUT_CSV") - 1)))"
