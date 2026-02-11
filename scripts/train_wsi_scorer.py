"""Train the stage-five WSI scorer (SVM) from stage-four hybrid descriptors.

Inputs
- Features CSV: produced by stage four (15-D hybrid descriptors). Must contain a
  `group` column and either legacy `feature_1..feature_15` columns or the new
  semantic feature columns.
- Labels CSV: a table with at least `group,label` columns.

This script is intentionally group-keyed to avoid accidental misalignment when
some slides are missing features (e.g. only 498 feature rows out of 500 labels).

Outputs
- CV report CSV (per-fold accuracy + quadratic weighted kappa)
- Trained model artifact (joblib): sklearn Pipeline(scaler+SVC) + metadata
- Meta JSON with run configuration and data alignment summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cabcds.hybrid_descriptor.descriptor import HYBRID_DESCRIPTOR_FEATURE_NAMES


LOGGER = logging.getLogger("train_wsi_scorer")


@dataclass(frozen=True)
class TrainOutputs:
    out_dir: Path
    model_path: Path
    cv_report_path: Path
    meta_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WSI scorer SVM (stage five).")

    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Path to stage-four hybrid descriptor CSV (train split).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        required=True,
        help="Path to labels CSV with at least columns: group,label.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/wsi_scorer",
        help="Output directory for reports and model artifacts.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=10)

    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument(
        "--svm-kernel",
        type=str,
        default="linear",
        choices=("linear", "rbf", "poly", "sigmoid"),
    )
    parser.add_argument(
        "--decision-function-shape",
        type=str,
        default="ovo",
        choices=("ovo", "ovr"),
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=("none", "balanced"),
        help="Use 'balanced' to mitigate class imbalance.",
    )
    parser.add_argument(
        "--grid-c",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of C values for a small grid search, "
            "e.g. '0.01,0.1,1'. If set, the script selects best config by mean quadratic kappa."
        ),
    )
    parser.add_argument(
        "--grid-class-weight",
        type=str,
        default="none",
        choices=("none", "balanced", "both"),
        help="Class-weight options to consider during grid search (requires --grid-c).",
    )

    parser.add_argument(
        "--report-confusion",
        action="store_true",
        help="Log per-fold confusion matrix and prediction distribution.",
    )
    parser.add_argument(
        "--probability",
        action="store_true",
        help="Enable SVC probability estimates (slower).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit to first N groups (after join) for a quick smoke run.",
    )

    return parser.parse_args()


def _resolve_outputs(out_dir: Path) -> TrainOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    report_dir = out_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return TrainOutputs(
        out_dir=out_dir,
        model_path=model_dir / "wsi_svm.joblib",
        cv_report_path=report_dir / "stage_five_wsi_scorer_cv.csv",
        meta_path=out_dir / "meta.json",
    )


def _load_label_map(labels_csv: Path) -> dict[str, int]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    with labels_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {labels_csv}")
        if "group" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                "Labels CSV must have columns 'group' and 'label'. "
                f"Found: {reader.fieldnames}"
            )

        label_map: dict[str, int] = {}
        for row in reader:
            group = (row.get("group") or "").strip()
            label_str = (row.get("label") or "").strip()
            if not group:
                raise ValueError("Labels CSV contains empty 'group'")
            if not label_str:
                raise ValueError(f"Labels CSV contains empty 'label' for group={group}")
            label_map[group] = int(label_str)

    return label_map


def _parse_feature_row(row: dict[str, str]) -> np.ndarray:
    # Legacy format: feature_1..feature_15
    if "feature_1" in row:
        values: list[float] = []
        for index in range(1, 16):
            key = f"feature_{index}"
            if key not in row:
                raise ValueError(f"Missing {key} in descriptor row")
            values.append(float(row[key]))
        return np.array(values, dtype=np.float32)

    # New format: semantic names
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


def _inner_join_by_group(
    feature_groups: list[str],
    feature_matrix: np.ndarray,
    label_map: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    features_by_group = {g: feature_matrix[i] for i, g in enumerate(feature_groups)}

    missing_labels = sorted([g for g in feature_groups if g not in label_map])
    if missing_labels:
        raise ValueError(
            "Found feature groups without labels. First 20: "
            + ",".join(missing_labels[:20])
        )

    common_groups = sorted([g for g in feature_groups if g in label_map])
    x = np.stack([features_by_group[g] for g in common_groups])
    y = np.array([label_map[g] for g in common_groups], dtype=np.int32)

    missing_features = sorted([g for g in label_map.keys() if g not in features_by_group])

    summary: dict[str, Any] = {
        "n_feature_rows": int(len(feature_groups)),
        "n_label_rows": int(len(label_map)),
        "n_joined": int(len(common_groups)),
        "missing_in_features": missing_features,
    }

    return common_groups, x, y, summary


@dataclass(frozen=True)
class ModelSpec:
    """Hyperparameters for the SVM stage."""

    svm_c: float
    svm_kernel: str
    decision_function_shape: str
    class_weight: str
    probability: bool
    seed: int


def _build_model(spec: ModelSpec) -> Pipeline:
    class_weight = None if spec.class_weight == "none" else "balanced"

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(spec.svm_c),
                    kernel=str(spec.svm_kernel),
                    decision_function_shape=str(spec.decision_function_shape),
                    class_weight=class_weight,
                    probability=bool(spec.probability),
                    random_state=int(spec.seed),
                ),
            ),
        ]
    )


def _default_spec(args: argparse.Namespace) -> ModelSpec:
    return ModelSpec(
        svm_c=float(args.svm_c),
        svm_kernel=str(args.svm_kernel),
        decision_function_shape=str(args.decision_function_shape),
        class_weight=str(args.class_weight),
        probability=bool(args.probability),
        seed=int(args.seed),
    )


def _parse_grid(values: str) -> list[float]:
    parsed: list[float] = []
    for token in values.split(","):
        token_stripped = token.strip()
        if not token_stripped:
            continue
        parsed.append(float(token_stripped))
    if not parsed:
        raise ValueError(f"Empty grid specification: {values!r}")
    return parsed


def _evaluate_cv(
    x_all: np.ndarray,
    y_all: np.ndarray,
    cv: StratifiedKFold,
    spec: ModelSpec,
    report_confusion: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    accs: list[float] = []
    kappas: list[float] = []
    fold_confusions: list[list[list[int]]] = []
    fold_pred_distributions: list[dict[str, int]] = []

    labels_sorted = sorted(np.unique(y_all).tolist())

    for fold_index, (train_idx, val_idx) in enumerate(cv.split(x_all, y_all), start=1):
        x_train, x_val = x_all[train_idx], x_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        fold_model = _build_model(spec)
        fold_model.fit(x_train, y_train)

        y_pred = fold_model.predict(x_val)
        acc = float(accuracy_score(y_val, y_pred))
        kappa = float(cohen_kappa_score(y_val, y_pred, weights="quadratic"))

        accs.append(acc)
        kappas.append(kappa)
        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "accuracy": f"{acc:.6f}",
                "kappa_quadratic": f"{kappa:.6f}",
            }
        )

        if report_confusion:
            cm = confusion_matrix(y_val, y_pred, labels=labels_sorted)
            fold_confusions.append(cm.astype(int).tolist())

            pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
            dist = {str(int(k)): int(v) for k, v in zip(pred_unique.tolist(), pred_counts.tolist(), strict=False)}
            fold_pred_distributions.append(dist)
            LOGGER.info("Fold %d pred distribution: %s", fold_index, str(dist))
            LOGGER.info("Fold %d confusion matrix (labels=%s):\n%s", fold_index, str(labels_sorted), str(cm))

    summary: dict[str, Any] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_std": float(np.std(kappas)),
        "fold_confusion_matrices": fold_confusions if report_confusion else None,
        "fold_pred_distributions": fold_pred_distributions if report_confusion else None,
        "labels_sorted": labels_sorted,
    }

    return fold_rows, summary


def _select_best_spec(
    x_all: np.ndarray,
    y_all: np.ndarray,
    cv: StratifiedKFold,
    base_spec: ModelSpec,
    c_grid: list[float],
    grid_class_weight: str,
    report_confusion: bool,
) -> tuple[ModelSpec, dict[str, Any]]:
    if grid_class_weight == "both":
        class_weights = ["none", "balanced"]
    else:
        class_weights = [grid_class_weight]

    best: ModelSpec | None = None
    best_summary: dict[str, Any] | None = None

    for cw in class_weights:
        for c_value in c_grid:
            spec = ModelSpec(
                svm_c=float(c_value),
                svm_kernel=base_spec.svm_kernel,
                decision_function_shape=base_spec.decision_function_shape,
                class_weight=str(cw),
                probability=base_spec.probability,
                seed=base_spec.seed,
            )

            _, summary = _evaluate_cv(x_all, y_all, cv, spec, report_confusion=False)
            LOGGER.info(
                "Grid candidate C=%s class_weight=%s -> kappa=%.4f acc=%.4f",
                str(spec.svm_c),
                spec.class_weight,
                float(summary["kappa_mean"]),
                float(summary["accuracy_mean"]),
            )

            if best is None:
                best, best_summary = spec, summary
                continue

            assert best_summary is not None
            if float(summary["kappa_mean"]) > float(best_summary["kappa_mean"]):
                best, best_summary = spec, summary
            elif float(summary["kappa_mean"]) == float(best_summary["kappa_mean"]):
                if float(summary["accuracy_mean"]) > float(best_summary["accuracy_mean"]):
                    best, best_summary = spec, summary

    assert best is not None and best_summary is not None
    return best, best_summary


def _write_cv_report(
    out_path: Path,
    fold_rows: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "n_train",
        "n_val",
        "accuracy",
        "kappa_quadratic",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    features_csv = Path(args.features_csv)
    labels_csv = Path(args.labels_csv)
    outputs = _resolve_outputs(Path(args.out_dir))

    label_map = _load_label_map(labels_csv)
    feature_groups, feature_matrix = _load_features(features_csv)

    groups, x_all, y_all, join_summary = _inner_join_by_group(feature_groups, feature_matrix, label_map)

    if args.limit is not None:
        limit = int(args.limit)
        groups = groups[:limit]
        x_all = x_all[:limit]
        y_all = y_all[:limit]
        join_summary["limit"] = limit

    unique, counts = np.unique(y_all, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist(), strict=False)}

    LOGGER.info("Joined dataset rows: %d", x_all.shape[0])
    LOGGER.info("Class distribution: %s", str(class_distribution))

    missing_in_features = join_summary.get("missing_in_features") or []
    if missing_in_features:
        LOGGER.warning(
            "Labels exist but features are missing for %d groups. First 20: %s",
            len(missing_in_features),
            ",".join(missing_in_features[:20]),
        )

    cv_folds = int(args.cv_folds)
    if cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if x_all.shape[0] < cv_folds:
        raise ValueError(
            f"Not enough samples ({x_all.shape[0]}) for cv_folds={cv_folds}. Reduce --cv-folds or remove --limit."
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(args.seed))

    base_spec = _default_spec(args)
    selected_spec = base_spec
    grid_summary: dict[str, Any] | None = None

    if args.grid_c is not None:
        c_grid = _parse_grid(str(args.grid_c))
        if args.grid_class_weight == "none":
            grid_cw = "none"
        else:
            grid_cw = str(args.grid_class_weight)
        selected_spec, grid_summary = _select_best_spec(
            x_all,
            y_all,
            cv,
            base_spec,
            c_grid=c_grid,
            grid_class_weight=grid_cw,
            report_confusion=False,
        )
        LOGGER.info(
            "Selected best spec: C=%s class_weight=%s (mean kappa=%.4f, mean acc=%.4f)",
            str(selected_spec.svm_c),
            selected_spec.class_weight,
            float(grid_summary["kappa_mean"]),
            float(grid_summary["accuracy_mean"]),
        )

    fold_rows, cv_summary = _evaluate_cv(
        x_all,
        y_all,
        cv,
        selected_spec,
        report_confusion=bool(args.report_confusion),
    )

    LOGGER.info("CV accuracy: %.4f ± %.4f", float(cv_summary["accuracy_mean"]), float(cv_summary["accuracy_std"]))
    LOGGER.info(
        "CV kappa (quadratic): %.4f ± %.4f",
        float(cv_summary["kappa_mean"]),
        float(cv_summary["kappa_std"]),
    )

    _write_cv_report(outputs.cv_report_path, fold_rows)
    LOGGER.info("Wrote CV report: %s", str(outputs.cv_report_path))

    model = _build_model(selected_spec)
    model.fit(x_all, y_all)

    payload = {
        "model": model,
        "feature_names": list(HYBRID_DESCRIPTOR_FEATURE_NAMES),
        "groups": groups,
        "args": vars(args),
        "selected_spec": {
            "svm_c": selected_spec.svm_c,
            "svm_kernel": selected_spec.svm_kernel,
            "decision_function_shape": selected_spec.decision_function_shape,
            "class_weight": selected_spec.class_weight,
            "probability": selected_spec.probability,
            "seed": selected_spec.seed,
        },
        "join_summary": join_summary,
        "class_distribution": class_distribution,
        "cv": {
            "folds": cv_folds,
            "accuracy_mean": float(cv_summary["accuracy_mean"]),
            "accuracy_std": float(cv_summary["accuracy_std"]),
            "kappa_mean": float(cv_summary["kappa_mean"]),
            "kappa_std": float(cv_summary["kappa_std"]),
            "labels_sorted": cv_summary["labels_sorted"],
            "fold_confusion_matrices": cv_summary["fold_confusion_matrices"],
            "fold_pred_distributions": cv_summary["fold_pred_distributions"],
        },
        "grid": grid_summary,
    }

    joblib.dump(payload, outputs.model_path)
    LOGGER.info("Saved model: %s", str(outputs.model_path))

    meta = {
        "outputs": {"out_dir": str(outputs.out_dir), "model": str(outputs.model_path), "cv_report": str(outputs.cv_report_path)},
        "inputs": {"features_csv": str(features_csv), "labels_csv": str(labels_csv)},
        "args": vars(args),
        "selected_spec": payload["selected_spec"],
        "join_summary": join_summary,
        "class_distribution": class_distribution,
        "cv": payload["cv"],
        "grid": payload["grid"],
    }

    outputs.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote meta: %s", str(outputs.meta_path))


if __name__ == "__main__":
    main()
