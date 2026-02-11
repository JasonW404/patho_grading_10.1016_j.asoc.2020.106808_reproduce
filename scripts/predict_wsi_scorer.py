"""Run stage-five WSI scorer inference from a trained joblib artifact.

This script loads the sklearn Pipeline saved by `scripts/train_wsi_scorer.py`
and applies it to a stage-four descriptor CSV.

Outputs a CSV with at least:
- group
- predicted_score

Optionally also outputs per-class probabilities when the model supports
`predict_proba`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from cabcds.hybrid_descriptor.descriptor import HYBRID_DESCRIPTOR_FEATURE_NAMES


LOGGER = logging.getLogger("predict_wsi_scorer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict WSI scores from 15-D descriptors.")

    parser.add_argument(
        "--model-joblib",
        type=str,
        required=True,
        help="Path to model artifact produced by scripts/train_wsi_scorer.py (models/wsi_svm.joblib).",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Path to stage-four descriptor CSV (train or test). Must include 'group' and feature columns.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--write-proba",
        action="store_true",
        help="If the model supports predict_proba, also write probability columns.",
    )

    return parser.parse_args()


def _parse_feature_row(row: dict[str, str]) -> np.ndarray:
    if "feature_1" in row:
        values: list[float] = []
        for index in range(1, 16):
            key = f"feature_{index}"
            if key not in row:
                raise ValueError(f"Missing {key} in descriptor row")
            values.append(float(row[key]))
        return np.array(values, dtype=np.float32)

    values_named: list[float] = []
    for key in HYBRID_DESCRIPTOR_FEATURE_NAMES:
        if key not in row:
            raise ValueError(
                "Descriptor CSV missing expected feature column. "
                f"Missing '{key}'. Available keys: {sorted(row.keys())}"
            )
        values_named.append(float(row[key]))
    return np.array(values_named, dtype=np.float32)


def _load_features(features_csv: Path) -> tuple[list[str], np.ndarray]:
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    with features_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Features CSV has no header: {features_csv}")
        if "group" not in reader.fieldnames:
            raise ValueError(f"Features CSV must include 'group' column: {features_csv}")

        groups: list[str] = []
        vectors: list[np.ndarray] = []
        for row in reader:
            group = (row.get("group") or "").strip()
            if not group:
                raise ValueError("Features CSV contains empty 'group'")
            groups.append(group)
            vectors.append(_parse_feature_row(row))

    return groups, np.stack(vectors)


def _load_model(model_joblib: Path) -> Any:
    if not model_joblib.exists():
        raise FileNotFoundError(f"Model joblib not found: {model_joblib}")

    payload = joblib.load(model_joblib)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    model_joblib = Path(args.model_joblib)
    features_csv = Path(args.features_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_joblib)
    groups, x = _load_features(features_csv)

    preds = model.predict(x)

    proba: np.ndarray | None = None
    proba_labels: list[str] | None = None
    if bool(args.write_proba) and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)
            if hasattr(model, "classes_"):
                proba_labels = [str(int(c)) for c in model.classes_.tolist()]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("predict_proba failed, continuing without proba: %s", str(exc))
            proba = None
            proba_labels = None

    with out_csv.open("w", encoding="utf-8", newline="") as file_handle:
        fieldnames = ["group", "predicted_score"]
        if proba is not None:
            if proba_labels is None:
                proba_labels = [str(i) for i in range(proba.shape[1])]
            fieldnames.extend([f"proba_{label}" for label in proba_labels])

        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, group in enumerate(groups):
            row: dict[str, Any] = {"group": group, "predicted_score": int(preds[index])}
            if proba is not None:
                assert proba_labels is not None
                for j, label in enumerate(proba_labels):
                    row[f"proba_{label}"] = f"{float(proba[index, j]):.6f}"
            writer.writerow(row)

    LOGGER.info("Wrote predictions: %s", str(out_csv))


if __name__ == "__main__":
    main()
