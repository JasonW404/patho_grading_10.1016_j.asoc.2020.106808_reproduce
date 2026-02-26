"""Build stage-five (WSI scorer) holdout datasets from stage-four descriptors.

We want stage-five training/testing to mirror the CNN_global holdout split
inside the labeled TUPAC training slides (e.g. 400 train + 100 held-out test).

Inputs
- Stage-four descriptor CSV (typically produced from ROI patches under the
  TUPAC train slides): must contain a `group` column.
- Labels CSV with at least columns: `group,label`.
- CNN_global payload JSON containing `train_slide_ids` and `test_slide_ids`.

Outputs (under --out-dir)
- holdout_train_features.csv
- holdout_test_features.csv
- holdout_train_labels.csv
- holdout_test_labels.csv
- meta.json (data alignment summary)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class HoldoutSplit:
    train_ids: list[str]
    test_ids: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build holdout datasets for stage-five WSI scorer.")

    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Stage-four descriptor CSV (from TUPAC-TR slides).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        required=True,
        help="Labels CSV with columns: group,label.",
    )
    parser.add_argument(
        "--cnn-global-payload-json",
        type=str,
        required=True,
        help="CNN_global metrics payload JSON containing train_slide_ids/test_slide_ids.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/wsi_scorer/holdout_dataset",
        help="Output directory.",
    )

    return parser.parse_args()


def _load_payload_split(payload_path: Path) -> HoldoutSplit:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    train_ids = payload.get("train_slide_ids")
    val_ids = payload.get("val_slide_ids")
    test_ids = payload.get("test_slide_ids")

    if not isinstance(train_ids, list) or not isinstance(test_ids, list):
        raise ValueError(
            "Payload JSON must contain lists: train_slide_ids, test_slide_ids. "
            f"Got keys: {sorted(payload.keys())}"
        )

    train_ids_s = [str(x).strip() for x in train_ids if str(x).strip()]
    test_ids_s = [str(x).strip() for x in test_ids if str(x).strip()]

    # CNN_global per-fold payloads (CV inside 400) store:
    # - train_slide_ids: fold-train (~266)
    # - val_slide_ids: fold-val (~134)
    # Stage-five should train on the full 400-slide pool, so we union them.
    if isinstance(val_ids, list):
        val_ids_s = [str(x).strip() for x in val_ids if str(x).strip()]
        train_ids_s = sorted(set(train_ids_s) | set(val_ids_s))
    return HoldoutSplit(train_ids=train_ids_s, test_ids=test_ids_s)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_by_groups(
    rows: list[dict[str, str]],
    *,
    group_key: str,
    allowed: set[str],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        group = (row.get(group_key) or "").strip()
        if not group:
            continue
        if group in allowed:
            out.append(row)
    return out


def _meta_summary(*, split: HoldoutSplit, features_groups: set[str], labels_groups: set[str]) -> dict[str, Any]:
    train_set = set(split.train_ids)
    test_set = set(split.test_ids)
    overlap = sorted(train_set & test_set)

    return {
        "split": {
            "train_ids": len(split.train_ids),
            "test_ids": len(split.test_ids),
            "overlap": overlap,
        },
        "available": {
            "features_groups": len(features_groups),
            "labels_groups": len(labels_groups),
        },
        "intersection": {
            "train": {
                "features_only": sorted((train_set & features_groups) - labels_groups)[:50],
                "labels_only": sorted((train_set & labels_groups) - features_groups)[:50],
                "usable": len(train_set & features_groups & labels_groups),
            },
            "test": {
                "features_only": sorted((test_set & features_groups) - labels_groups)[:50],
                "labels_only": sorted((test_set & labels_groups) - features_groups)[:50],
                "usable": len(test_set & features_groups & labels_groups),
            },
        },
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_csv = Path(args.features_csv)
    labels_csv = Path(args.labels_csv)
    payload_json = Path(args.cnn_global_payload_json)

    split = _load_payload_split(payload_json)
    train_ids = set(split.train_ids)
    test_ids = set(split.test_ids)

    feat_fieldnames, feat_rows = _read_csv_rows(features_csv)
    if "group" not in feat_fieldnames:
        raise ValueError("features-csv must include a 'group' column")

    lbl_fieldnames, lbl_rows = _read_csv_rows(labels_csv)
    if "group" not in lbl_fieldnames or "label" not in lbl_fieldnames:
        raise ValueError("labels-csv must include columns: group,label")

    features_groups = {(r.get("group") or "").strip() for r in feat_rows if (r.get("group") or "").strip()}
    labels_groups = {(r.get("group") or "").strip() for r in lbl_rows if (r.get("group") or "").strip()}

    # Keep only rows with both features and labels (by group) to avoid misalignment.
    usable_groups = features_groups & labels_groups

    train_allowed = usable_groups & train_ids
    test_allowed = usable_groups & test_ids

    feat_train = _filter_by_groups(feat_rows, group_key="group", allowed=train_allowed)
    feat_test = _filter_by_groups(feat_rows, group_key="group", allowed=test_allowed)
    lbl_train = _filter_by_groups(lbl_rows, group_key="group", allowed=train_allowed)
    lbl_test = _filter_by_groups(lbl_rows, group_key="group", allowed=test_allowed)

    _write_csv(out_dir / "holdout_train_features.csv", feat_fieldnames, feat_train)
    _write_csv(out_dir / "holdout_test_features.csv", feat_fieldnames, feat_test)
    _write_csv(out_dir / "holdout_train_labels.csv", lbl_fieldnames, lbl_train)
    _write_csv(out_dir / "holdout_test_labels.csv", lbl_fieldnames, lbl_test)

    meta = _meta_summary(split=split, features_groups=features_groups, labels_groups=labels_groups)
    meta.update(
        {
            "paths": {
                "features_csv": str(features_csv),
                "labels_csv": str(labels_csv),
                "payload_json": str(payload_json),
            },
            "outputs": {
                "train_features": str(out_dir / "holdout_train_features.csv"),
                "test_features": str(out_dir / "holdout_test_features.csv"),
                "train_labels": str(out_dir / "holdout_train_labels.csv"),
                "test_labels": str(out_dir / "holdout_test_labels.csv"),
            },
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[holdout] wrote datasets. "
        f"train_usable={len(train_allowed)} test_usable={len(test_allowed)} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
