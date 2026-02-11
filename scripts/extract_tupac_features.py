"""Extract 15-dimensional hybrid descriptors for TUPAC slides.

This script is a thin CLI wrapper around `cabcds.hybrid_descriptor.HybridDescriptorPipeline`.
It is intended for NPU-only inference jobs, with optional per-job NPU binding via
`ASCEND_VISIBLE_DEVICES`.

Outputs:
- A CSV at `<out-dir>/stage_four_hybrid_descriptors.csv` with columns:
  group,feature_1..feature_15

Notes:
- The descriptor computation itself is defined in `cabcds/hybrid_descriptor/descriptor.py`.
- ROI patches are expected to exist under:
  `output/roi_selector/outputs/patches/{train|test}/<slide_id>/roi_..png`
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from cabcds.hybrid_descriptor.config import HybridDescriptorConfig, HybridDescriptorInferenceConfig
from cabcds.hybrid_descriptor.pipeline import HybridDescriptorPipeline


LOGGER = logging.getLogger("extract_tupac_features")


@dataclass(frozen=True)
class Checkpoints:
    seg: Path
    det: Path
    global_folds: list[Path]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 15-D hybrid descriptors (stage four).")

    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=("train", "test"),
        help="Which ROI patch split folder to use under roi-patches-dir (train|test).",
    )
    p.add_argument(
        "--slides",
        type=str,
        default=None,
        help="Comma-separated slide ids to process (e.g. TUPAC-TE-001,TUPAC-TE-002).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N slides discovered in the split directory.",
    )

    p.add_argument(
        "--roi-patches-dir",
        type=str,
        default="output/roi_selector/outputs/patches",
        help="Root directory containing ROI patches. Default matches repo output.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="output/features",
        help="Output directory for descriptors CSV.",
    )

    p.add_argument(
        "--cnn-seg-checkpoint",
        type=str,
        default=None,
        help="Path to CNN_seg checkpoint (.pt). If omitted, tries to auto-detect.",
    )
    p.add_argument(
        "--cnn-det-checkpoint",
        type=str,
        default=None,
        help="Path to CNN_det checkpoint (.pt). If omitted, tries to auto-detect.",
    )
    p.add_argument(
        "--cnn-global-folds",
        type=str,
        default=None,
        help="Comma/semicolon-separated list of CNN_global fold checkpoints (.pt). If omitted, tries to auto-detect.",
    )

    p.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help=(
            "Optional physical NPU id to bind via ASCEND_VISIBLE_DEVICES. "
            "If you already exported ASCEND_VISIBLE_DEVICES, you can omit this."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help="Torch device string to use. For Ascend NPU jobs, use npu:0.",
    )

    p.add_argument("--min-blob-area", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--roi-resize", type=int, default=227)
    p.add_argument("--det-resize", type=int, default=227)
    p.add_argument("--det-patch-size", type=int, default=80)

    return p.parse_args()


def _maybe_bind_npu(npu_id: int | None) -> None:
    if npu_id is None:
        return

    # Best-effort: set env var if not already set. This is most reliable when
    # set before importing/initializing torch_npu.
    if os.environ.get("ASCEND_VISIBLE_DEVICES") is None:
        os.environ["ASCEND_VISIBLE_DEVICES"] = str(npu_id)
        LOGGER.info("Set ASCEND_VISIBLE_DEVICES=%s", str(npu_id))
    else:
        LOGGER.info(
            "ASCEND_VISIBLE_DEVICES already set to %s (npu-id=%s ignored)",
            os.environ.get("ASCEND_VISIBLE_DEVICES"),
            str(npu_id),
        )


def _auto_detect_checkpoints() -> Checkpoints:
    seg_dir = Path("output/mf_cnn/CNN_seg/models")
    seg_candidates = sorted(seg_dir.glob("*best.pt")) + sorted(seg_dir.glob("*.pt"))
    if not seg_candidates:
        raise FileNotFoundError(f"No CNN_seg checkpoints found under {seg_dir}")
    seg = seg_candidates[0]

    det_dir = Path("output/mf_cnn/CNN_det/runs")
    det_candidates = sorted(det_dir.glob("**/models/*last.pt")) + sorted(det_dir.glob("**/models/*.pt"))
    if not det_candidates:
        raise FileNotFoundError(f"No CNN_det checkpoints found under {det_dir}")
    det = det_candidates[0]

    global_dir = Path("output/mf_cnn/CNN_global/runs")
    fold_candidates = sorted(global_dir.glob("**/models/cnn_global_paper_fold*.pt"))
    if len(fold_candidates) < 3:
        raise FileNotFoundError(f"Expected at least 3 CNN_global fold checkpoints under {global_dir}")

    # Prefer a single run directory (take the newest run by folder name, then its 3 folds).
    run_dirs = sorted({p.parent for p in fold_candidates})
    run_dir = run_dirs[-1]
    folds = [run_dir / "cnn_global_paper_fold1.pt", run_dir / "cnn_global_paper_fold2.pt", run_dir / "cnn_global_paper_fold3.pt"]
    if not all(p.exists() for p in folds):
        # Fallback: take first three discovered.
        folds = fold_candidates[:3]

    return Checkpoints(seg=seg, det=det, global_folds=folds)


def _select_roi_patch_root(roi_patches_dir: Path, split: str | None) -> Path:
    if split is None:
        return roi_patches_dir
    return roi_patches_dir / split


def _resolve_groups(root_dir: Path, slides_csv: str | None, limit: int | None) -> list[str]:
    if slides_csv:
        groups = [s.strip() for s in slides_csv.split(",") if s.strip()]
        if not groups:
            raise ValueError("--slides provided but empty after parsing")
        return groups

    if not root_dir.exists():
        raise FileNotFoundError(f"ROI patches directory not found: {root_dir}")

    groups = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
    if limit is not None:
        groups = groups[: int(limit)]
    if not groups:
        raise FileNotFoundError(f"No slide folders found under: {root_dir}")
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    _maybe_bind_npu(args.npu_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoints.
    if args.cnn_seg_checkpoint and args.cnn_det_checkpoint and args.cnn_global_folds:
        global_folds = [p.strip() for p in str(args.cnn_global_folds).replace(";", ",").split(",") if p.strip()]
        checkpoints = Checkpoints(
            seg=Path(args.cnn_seg_checkpoint),
            det=Path(args.cnn_det_checkpoint),
            global_folds=[Path(p) for p in global_folds],
        )
    else:
        checkpoints = _auto_detect_checkpoints()

    LOGGER.info("Using CNN_seg: %s", str(checkpoints.seg))
    LOGGER.info("Using CNN_det: %s", str(checkpoints.det))
    LOGGER.info("Using CNN_global folds: %s", ", ".join(str(p) for p in checkpoints.global_folds))

    roi_patches_dir = Path(args.roi_patches_dir)
    roi_root = _select_roi_patch_root(roi_patches_dir, args.split)

    groups = _resolve_groups(roi_root, args.slides, args.limit)
    LOGGER.info("Slides selected: %d", len(groups))

    hybrid_cfg = HybridDescriptorConfig()
    infer_cfg = HybridDescriptorInferenceConfig(
        roi_patches_dir=str(roi_root),
        output_dir=str(out_dir),
        segmentation_model_path=str(checkpoints.seg),
        detection_model_path=str(checkpoints.det),
        roi_scoring_model_path=";".join(str(p) for p in checkpoints.global_folds),
        roi_scoring_ensemble_strategy="sum",
        device=str(args.device),
        min_blob_area=int(args.min_blob_area),
        detection_patch_size=int(args.det_patch_size),
        detection_resize=int(args.det_resize),
        roi_resize=int(args.roi_resize),
        batch_size=int(args.batch_size),
        use_detection=True,
        use_segmentation=True,
    )

    pipeline = HybridDescriptorPipeline(hybrid_cfg, infer_cfg)
    pipeline.run(groups=groups)

    LOGGER.info("Done. Output CSV: %s", str(out_dir / "stage_four_hybrid_descriptors.csv"))


if __name__ == "__main__":
    main()
