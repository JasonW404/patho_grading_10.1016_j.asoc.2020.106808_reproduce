"""Add slide group names to TUPAC16 ground truth.

The original file `dataset/tupac16/train/ground_truth.csv` contains two unnamed columns
(label, score) in fixed slide order (1..500). For training and feature-join safety,
we generate an explicit table with a stable `group` key:

- group: e.g. TUPAC-TR-001
- label: integer class label (as provided)
- score: float value (as provided)

Optionally, this script can compare the produced label table with a descriptor CSV
that contains a `group` column (e.g. extracted 15-D features) and report missing
slides, so you don't accidentally mis-align labels when some feature rows are absent.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path


LOGGER = logging.getLogger("make_ground_truth_with_groups")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add group column to TUPAC16 ground truth.")

    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default="dataset/tupac16/train/ground_truth.csv",
        help="Path to the original ground truth CSV (two columns: label, score).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="dataset/tupac16/train/ground_truth_with_groups.csv",
        help="Output CSV path with columns: group,label,score.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="TUPAC-TR",
        help="Group prefix to generate (default: TUPAC-TR).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based slide index to start naming from (default: 1 -> XXX-001).",
    )

    parser.add_argument(
        "--features-csv",
        type=str,
        default=None,
        help=(
            "Optional descriptor/features CSV to compare against. Must include a 'group' column. "
            "If provided, the script reports which label groups are missing from the features file."
        ),
    )

    return parser.parse_args()


def _format_group(prefix: str, index_1_based: int) -> str:
    return f"{prefix}-{index_1_based:03d}"


def _load_features_groups(features_csv: Path) -> set[str]:
    with features_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None or "group" not in reader.fieldnames:
            raise ValueError(f"Features CSV must include a 'group' column: {features_csv}")

        groups: set[str] = set()
        for row in reader:
            group = row.get("group")
            if not group:
                raise ValueError(f"Found empty 'group' value in features CSV: {features_csv}")
            groups.add(group)
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    ground_truth_csv = Path(args.ground_truth_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")

    rows: list[tuple[str, int, float]] = []
    with ground_truth_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)
        for row_index_zero_based, row in enumerate(reader):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Expected 2 columns (label, score) at line {row_index_zero_based + 1}, got: {row}"
                )

            label = int(row[0])
            score = float(row[1])

            slide_index_1_based = args.start_index + row_index_zero_based
            group = _format_group(args.prefix, slide_index_1_based)
            rows.append((group, label, score))

    LOGGER.info("Loaded %d ground-truth rows", len(rows))

    with out_csv.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["group", "label", "score"])
        writer.writerows(rows)

    LOGGER.info("Wrote: %s", str(out_csv))

    if args.features_csv is not None:
        features_csv = Path(args.features_csv)
        if not features_csv.exists():
            raise FileNotFoundError(f"Features CSV not found: {features_csv}")

        features_groups = _load_features_groups(features_csv)
        label_groups = {group for group, _, _ in rows}

        missing_in_features = sorted(label_groups - features_groups)
        extra_in_features = sorted(features_groups - label_groups)

        LOGGER.info(
            "Features rows: %d | Label rows: %d | Intersection: %d",
            len(features_groups),
            len(label_groups),
            len(features_groups & label_groups),
        )

        if missing_in_features:
            LOGGER.warning(
                "Missing %d groups in features (present in labels, absent in features). First 20: %s",
                len(missing_in_features),
                ",".join(missing_in_features[:20]),
            )
        else:
            LOGGER.info("No label groups are missing from features.")

        if extra_in_features:
            LOGGER.warning(
                "Found %d extra groups in features (not present in labels). First 20: %s",
                len(extra_in_features),
                ",".join(extra_in_features[:20]),
            )


if __name__ == "__main__":
    main()
