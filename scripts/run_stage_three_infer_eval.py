"""Stage-three end-to-end inference + evaluation (WSI -> ROIs -> 15-D -> score).

This script orchestrates the repository's existing stage modules:
1) ROI selection (stage two): `cabcds.roi_selector.inference.RoiSelector`
   - Reads `.svs` using OpenSlide
   - Scans at ~10x and selects Top-4 ROIs per slide
   - Writes ROI patches under `<roi-output-dir>/patches/test/<group>/roi_..png`
2) Hybrid descriptor (stage four): `cabcds.hybrid_descriptor.HybridDescriptorPipeline`
   - Extracts 15-D features per WSI (Table 3 definition)
   - Uses CNN_global 3-fold ensemble with Sum Strategy
3) WSI scorer (stage five): trained sklearn SVM pipeline (joblib)
   - Predicts final WSI grade per WSI
4) Evaluation:
   - Computes Quadratic Weighted Cohen's Kappa and confusion matrix

Outputs (in `--out-dir`):
- `predictions.csv` (group,predicted_score)
- `evaluation_report.txt` (kappa + confusion matrix + data alignment summary)

Notes
- To compute kappa, you must provide `--labels-csv` with columns `group,label`.
  The repository does not ship test labels by default.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from cabcds.hybrid_descriptor.config import HybridDescriptorConfig, HybridDescriptorInferenceConfig
from cabcds.hybrid_descriptor.pipeline import HybridDescriptorPipeline
from cabcds.roi_selector.config import ROISelectorConfig
from cabcds.roi_selector.inference import RoiSelector
from cabcds.wsi_scorer.config import WsiScorerConfig
from cabcds.wsi_scorer.pipeline import WsiScorerPredictor


LOGGER = logging.getLogger("run_stage_three_infer_eval")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage-three end-to-end inference + evaluation.")

    parser.add_argument("--wsi-dir", type=str, default="dataset/tupac16/test")
    parser.add_argument("--out-dir", type=str, default="output/stage_three")
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/stage_three/run.log",
        help="Runtime log file path (only logs go under logs/; other outputs go under out-dir).",
    )

    parser.add_argument("--roi-model", type=str, default="output/roi_selector/models/roi_svm.joblib")
    parser.add_argument("--roi-output-dir", type=str, default="output/roi_selector/outputs")
    parser.add_argument("--roi-top-n", type=int, default=4)
    parser.add_argument("--roi-scan-magnification", type=float, default=10.0)
    parser.add_argument("--roi-overlap", type=float, default=0.1)
    parser.add_argument("--roi-patch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit to first N WSIs.")

    parser.add_argument("--cnn-seg", type=str, required=True, help="Path to CNN_seg checkpoint (.pt/.pth).")
    parser.add_argument("--cnn-det", type=str, required=True, help="Path to CNN_det checkpoint (.pt/.pth).")
    parser.add_argument(
        "--cnn-global-folds",
        type=str,
        required=True,
        help="Comma/semicolon-separated list of 3 CNN_global fold checkpoints.",
    )
    parser.add_argument(
        "--cnn-global-ensemble",
        type=str,
        default="sum",
        choices=("sum", "mean"),
        help="How to combine CNN_global folds over softmax probabilities.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for hybrid descriptor (cpu, cuda:0, npu:0, ...).",
    )
    parser.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help="Optional physical NPU id to bind via ASCEND_VISIBLE_DEVICES.",
    )

    parser.add_argument(
        "--wsi-scorer-model",
        type=str,
        required=True,
        help="Path to trained WSI scorer joblib (models/wsi_svm.joblib).",
    )

    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional: labels CSV with columns group,label to compute kappa.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation even if labels are missing.",
    )

    return parser.parse_args()


def _maybe_bind_npu(npu_id: int | None) -> None:
    """Bind the current process to a physical Ascend NPU id.

    Args:
        npu_id: Physical device id to expose as `npu:0` via ASCEND_VISIBLE_DEVICES.
    """
    if npu_id is None:
        return
    if os.environ.get("ASCEND_VISIBLE_DEVICES") is None:
        os.environ["ASCEND_VISIBLE_DEVICES"] = str(npu_id)
        LOGGER.info("Set ASCEND_VISIBLE_DEVICES=%s", str(npu_id))


def _override_roi_config(base: ROISelectorConfig, args: argparse.Namespace) -> ROISelectorConfig:
    """Create an ROI selector config with CLI overrides."""
    payload = base.model_dump()
    payload.update(
        {
            "infer_wsi_dir": Path(args.wsi_dir),
            "infer_output_dir": Path(args.roi_output_dir),
            "infer_model_path": Path(args.roi_model),
            "infer_top_n": int(args.roi_top_n),
            "infer_scan_magnification": float(args.roi_scan_magnification),
            "infer_overlap_ratio": float(args.roi_overlap),
            "infer_patch_size": int(args.roi_patch_size),
            "infer_max_images": int(args.limit) if args.limit is not None else None,
        }
    )
    return ROISelectorConfig(**payload)


def _parse_folds(value: str) -> str:
    """Normalize a fold list into the repo's expected semicolon-separated format."""
    folds = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if len(folds) != 3:
        raise ValueError(f"Expected exactly 3 cnn-global folds, got {len(folds)}: {folds}")
    return ";".join(folds)


def _list_wsi_groups(wsi_dir: Path, limit: int | None) -> list[str]:
    """List WSI groups (stems) from a directory containing `.svs` files."""
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI dir not found: {wsi_dir}")
    groups = sorted([p.stem for p in wsi_dir.iterdir() if p.is_file() and p.suffix.lower() == ".svs"])
    if limit is not None:
        groups = groups[: int(limit)]
    if not groups:
        raise FileNotFoundError(f"No .svs files found under: {wsi_dir}")
    return groups


def _load_label_map(labels_csv: Path) -> dict[str, int]:
    """Load `group -> label` mapping from a CSV with columns group,label."""
    with labels_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None or "group" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels-csv must have header columns: group,label")

        label_map: dict[str, int] = {}
        for row in reader:
            group = (row.get("group") or "").strip()
            label_str = (row.get("label") or "").strip()
            if not group or not label_str:
                continue
            label_map[group] = int(label_str)
    return label_map


def _write_predictions_csv(path: Path, predictions: dict[str, int]) -> None:
    """Write final stage-three predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["group", "predicted_score"])
        writer.writeheader()
        for group in sorted(predictions.keys()):
            writer.writerow({"group": group, "predicted_score": int(predictions[group])})


def _write_evaluation_report(
    path: Path,
    *,
    groups: list[str],
    predictions: dict[str, int],
    label_map: dict[str, int],
) -> None:
    """Write an evaluation report with QWK and confusion matrix."""
    y_true: list[int] = []
    y_pred: list[int] = []

    missing_labels: list[str] = []
    missing_predictions: list[str] = []

    for group in groups:
        if group not in label_map:
            missing_labels.append(group)
            continue
        if group not in predictions:
            missing_predictions.append(group)
            continue
        y_true.append(int(label_map[group]))
        y_pred.append(int(predictions[group]))

    labels_sorted = [1, 2, 3]
    kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if y_true else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted) if y_true else np.zeros((3, 3), dtype=int)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("Stage-three evaluation report\n")
        file_handle.write("===========================\n\n")
        file_handle.write(f"Evaluated samples: {len(y_true)}\n")
        file_handle.write(f"Quadratic weighted kappa: {kappa:.6f}\n\n")

        file_handle.write("Confusion matrix (rows=true, cols=pred; labels=1,2,3)\n")
        file_handle.write(str(cm) + "\n\n")

        if missing_labels:
            file_handle.write(f"Missing labels for {len(missing_labels)} groups. First 20: {','.join(missing_labels[:20])}\n")
        if missing_predictions:
            file_handle.write(
                f"Missing predictions for {len(missing_predictions)} groups. First 20: {','.join(missing_predictions[:20])}\n"
            )


def main() -> None:
    args = _parse_args()

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    LOGGER.info("Logging to %s", str(log_path))

    _maybe_bind_npu(args.npu_id)

    wsi_dir = Path(args.wsi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ROI selection (writes patches + stage_two report)
    base_roi_cfg = ROISelectorConfig()
    roi_cfg = _override_roi_config(base_roi_cfg, args)
    LOGGER.info("Running ROI selection on %s", str(roi_cfg.infer_wsi_dir))
    selector = RoiSelector(roi_cfg)
    _ = selector.select_rois()

    # 2) Hybrid descriptors
    groups = _list_wsi_groups(wsi_dir, args.limit)
    roi_patches_dir = Path(args.roi_output_dir) / "patches"

    hybrid_cfg = HybridDescriptorConfig()
    infer_cfg = HybridDescriptorInferenceConfig(
        roi_patches_dir=str(roi_patches_dir / "test" if (roi_patches_dir / "test").exists() else roi_patches_dir),
        output_dir=str(out_dir / "hybrid_descriptor"),
        segmentation_model_path=str(Path(args.cnn_seg)),
        detection_model_path=str(Path(args.cnn_det)),
        roi_scoring_model_path=_parse_folds(str(args.cnn_global_folds)),
        roi_scoring_ensemble_strategy=str(args.cnn_global_ensemble),
        device=str(args.device),
    )

    LOGGER.info("Running hybrid descriptor extraction for %d WSIs", len(groups))
    HybridDescriptorPipeline(hybrid_cfg, infer_cfg).run(groups=groups)

    descriptor_csv = Path(infer_cfg.output_dir) / "stage_four_hybrid_descriptors.csv"
    if not descriptor_csv.exists():
        raise FileNotFoundError(f"Descriptor CSV not produced: {descriptor_csv}")

    # 3) WSI scorer predictions
    scorer_cfg = WsiScorerConfig(
        descriptor_csv=descriptor_csv,
        labels_csv=Path("."),
        model_output_path=Path(args.wsi_scorer_model),
        report_dir=out_dir,
        svm_c=0.1,
        svm_kernel="linear",
        cv_folds=10,
        decision_function_shape="ovo",
    )

    predictor = WsiScorerPredictor(scorer_cfg)
    predictions = predictor.predict(descriptor_csv=descriptor_csv)

    predictions_csv = out_dir / "predictions.csv"
    _write_predictions_csv(predictions_csv, predictions)
    LOGGER.info("Wrote predictions.csv: %s", str(predictions_csv))

    # 4) Evaluation
    evaluation_path = out_dir / "evaluation_report.txt"
    if args.labels_csv is None:
        if args.skip_eval:
            evaluation_path.write_text("Evaluation skipped: labels-csv not provided.\n", encoding="utf-8")
            LOGGER.warning("Skipped evaluation: labels-csv not provided")
            return
        raise FileNotFoundError("labels-csv is required for evaluation (or pass --skip-eval)")

    label_map = _load_label_map(Path(args.labels_csv))
    _write_evaluation_report(evaluation_path, groups=groups, predictions=predictions, label_map=label_map)
    LOGGER.info("Wrote evaluation_report.txt: %s", str(evaluation_path))


if __name__ == "__main__":
    main()
