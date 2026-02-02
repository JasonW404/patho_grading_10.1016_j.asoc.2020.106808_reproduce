"""Export/verify ROI preview patches from the ROI-Selector stage-two report.

The ROI-Selector inference writes its primary results as a CSV report:
`output/roi_selector/outputs/reports/stage_two_roi_selection.csv`.

Optionally, inference also writes preview patches under:
`output/roi_selector/outputs/patches/<slide_id>/roi_<rank>_x..._y....png`

This module helps:
- verify that already-saved preview patches are 1:1 consistent with the CSV
  (same rank -> same x/y)
- generate missing preview patches for slides already present in the CSV
  (e.g., train WSIs), without re-running the expensive full-slide scan.

Important
---------
These preview patches are *feature-size* patches (default 512x512) resized
from the scan window read at the chosen OpenSlide level.
They are NOT the full 40x ROI crops (which can be saved separately by ROI
inference when `infer_save_full_roi_40x=True`).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import openslide
from PIL import Image

from cabcds.roi_selector.config import load_roi_selector_config
from cabcds.roi_selector.utils.io import save_image_uint8

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoiRow:
    image_path: Path
    rank: int
    x: int
    y: int
    score: float


def _iter_report_rows(report_csv: Path) -> Iterable[RoiRow]:
    report_csv = Path(report_csv)
    with report_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path((row.get("image_path") or "").strip())
            if not str(image_path):
                continue
            try:
                rank = int(float(row.get("rank") or 0))
                x = int(float(row.get("x") or 0))
                y = int(float(row.get("y") or 0))
                score = float(row.get("score") or 0.0)
            except Exception:
                continue
            if rank <= 0:
                continue
            yield RoiRow(image_path=image_path, rank=rank, x=x, y=y, score=score)


_FILENAME_RE = re.compile(r"^roi_(?P<rank>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _parse_patch_filename(path: Path) -> tuple[int, int, int] | None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group("rank")), int(m.group("x")), int(m.group("y"))


def verify_preview_patches_match_report(
    *,
    report_csv: Path,
    patches_root: Path,
    expected_split_prefix: str,
    max_slides: int | None = None,
    prune_extras: bool = False,
    orphans_root: Path | None = None,
    split_subdirs: bool = False,
) -> None:
    """Verify that existing preview patches match the report ranks and coords.

    Args:
        report_csv: Stage-two report CSV.
        patches_root: Root folder containing per-slide patch folders.
        expected_split_prefix: Expected `image_path` prefix, e.g. "dataset/test/".
        max_slides: Optional cap for quicker checks.

    Raises:
        RuntimeError if mismatches are detected.
    """

    report_csv = Path(report_csv)
    patches_root = Path(patches_root)

    if split_subdirs:
        if expected_split_prefix.rstrip("/") == "dataset/test":
            patches_root = patches_root / "test"
        elif expected_split_prefix.rstrip("/") == "dataset/train":
            patches_root = patches_root / "train"
        else:
            raise ValueError(
                f"split_subdirs=True requires expected_split_prefix to be 'dataset/test/' or 'dataset/train/' (got {expected_split_prefix!r})"
            )

    # Map image_path -> rank -> (x,y)
    expected: dict[str, dict[int, tuple[int, int]]] = {}
    for row in _iter_report_rows(report_csv):
        ip = str(row.image_path)
        if expected_split_prefix not in ip:
            continue
        expected.setdefault(ip, {})[int(row.rank)] = (int(row.x), int(row.y))

    slide_dirs = [p for p in patches_root.iterdir() if p.is_dir()]
    slide_dirs.sort(key=lambda p: p.name)

    checked = 0
    mismatches: list[str] = []
    extras_moved = 0
    extras_seen = 0

    for slide_dir in slide_dirs:
        if max_slides is not None and checked >= int(max_slides):
            break

        slide_id = slide_dir.name
        # Expected report uses full path like dataset/test/TUPAC-TE-001.svs
        # but we match by stem.
        candidates = [k for k in expected.keys() if Path(k).stem == slide_id]
        if not candidates:
            # Not all patch dirs must have a report row (but in practice they should).
            continue
        image_key = candidates[0]
        exp_ranks = expected.get(image_key, {})

        found: dict[int, set[tuple[int, int]]] = {}
        files_by_rank: dict[int, list[Path]] = {}
        for file_path in slide_dir.iterdir():
            if not file_path.is_file():
                continue
            parsed = _parse_patch_filename(file_path)
            if parsed is None:
                continue
            r, x, y = parsed
            found.setdefault(int(r), set()).add((int(x), int(y)))
            files_by_rank.setdefault(int(r), []).append(file_path)

        # Verify and (optionally) prune duplicates so the folder becomes 1:1 with the report.
        if prune_extras:
            if orphans_root is None:
                raise ValueError("orphans_root must be provided when prune_extras=True")
            orphans_root = Path(orphans_root)
            orphan_dir = orphans_root / slide_id
            orphan_dir.mkdir(parents=True, exist_ok=True)

        for r, (x, y) in exp_ranks.items():
            if r not in found:
                mismatches.append(f"{slide_id}: missing patch for rank={r}")
                continue

            if (int(x), int(y)) not in found[r]:
                # The slide folder has patches, but not the one corresponding to the report.
                any_xy = next(iter(found[r]))
                mismatches.append(
                    f"{slide_id}: rank={r} missing expected patch(x,y)=({x},{y}); found example={any_xy}"
                )
                continue

            # Prune files for this rank that don't match expected coords.
            if prune_extras:
                rank_files = files_by_rank.get(int(r), [])
                keep_name = f"roi_{int(r):02d}_x{int(x)}_y{int(y)}"
                for fp in rank_files:
                    parsed = _parse_patch_filename(fp)
                    if parsed is None:
                        continue
                    rr, fx, fy = parsed
                    if int(rr) != int(r):
                        continue
                    if (int(fx), int(fy)) == (int(x), int(y)):
                        continue
                    extras_seen += 1
                    target = orphan_dir / fp.name
                    try:
                        fp.rename(target)
                        extras_moved += 1
                    except Exception:
                        # Non-fatal: still counts as extra, but we keep going.
                        continue

        checked += 1

    if mismatches:
        sample = "\n".join(mismatches[:20])
        raise RuntimeError(f"Preview patch verification failed (showing up to 20):\n{sample}")

    logger.info(
        "Preview patch verification OK: checked_slide_dirs=%d extras_seen=%d extras_moved=%d",
        checked,
        extras_seen,
        extras_moved,
    )


def _choose_scan_level(slide: openslide.OpenSlide, *, target_downsample: float) -> int:
    downsamples = [float(d) for d in slide.level_downsamples]
    for level, ds in enumerate(downsamples):
        if ds >= target_downsample:
            return level
    return len(downsamples) - 1


def export_missing_preview_patches_from_report(
    *,
    report_csv: Path,
    out_patches_root: Path,
    only_split_prefix: str,
    only_missing: bool = True,
    max_slides: int | None = None,
    split_subdirs: bool = False,
) -> None:
    """Export preview patches for rows in the report.

    This reconstructs the same patch saved by ROI inference:
    - read region at the chosen scan level using the ROI window size
    - resize to infer_patch_size for feature extraction

    Args:
        report_csv: Stage-two report CSV.
        out_patches_root: Root output folder for preview patches.
        only_split_prefix: Filter report rows by prefix, e.g. "dataset/train/".
        only_missing: Skip writing if the target filename already exists.
        max_slides: Optional cap (unique slides) for quicker runs.
    """

    cfg = load_roi_selector_config()
    report_csv = Path(report_csv)
    out_patches_root = Path(out_patches_root)
    if split_subdirs:
        if only_split_prefix.rstrip("/") == "dataset/test":
            out_patches_root = out_patches_root / "test"
        elif only_split_prefix.rstrip("/") == "dataset/train":
            out_patches_root = out_patches_root / "train"
        else:
            raise ValueError(
                f"split_subdirs=True requires only_split_prefix to be 'dataset/test/' or 'dataset/train/' (got {only_split_prefix!r})"
            )

    # Group rows by slide path.
    grouped: dict[str, list[RoiRow]] = {}
    for row in _iter_report_rows(report_csv):
        ip = str(row.image_path)
        if only_split_prefix not in ip:
            continue
        grouped.setdefault(ip, []).append(row)

    slide_paths = sorted(grouped.keys())
    if max_slides is not None:
        slide_paths = slide_paths[: int(max_slides)]

    feature_size = int(cfg.infer_patch_size)
    scan_mag = float(cfg.infer_scan_magnification)
    roi_size_40x = int(cfg.infer_roi_size_40x)

    logger.info(
        "Exporting ROI preview patches: slides=%d feature_size=%d scan_mag=%.1f roi40=%d out=%s",
        len(slide_paths),
        feature_size,
        scan_mag,
        roi_size_40x,
        str(out_patches_root),
    )

    written = 0
    skipped_existing = 0
    skipped_missing_slide = 0
    skipped_read_fail = 0

    for slide_index, slide_path_str in enumerate(slide_paths, start=1):
        slide_path = Path(slide_path_str)
        if not slide_path.exists():
            logger.warning("Missing slide on disk; skipping: %s", slide_path_str)
            skipped_missing_slide += 1
            continue

        rows = grouped[slide_path_str]
        # Keep rank order deterministic.
        rows.sort(key=lambda r: int(r.rank))

        out_dir = out_patches_root / slide_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        with openslide.OpenSlide(str(slide_path)) as slide:
            base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
            roi_size_level0 = int(round(roi_size_40x * (base_mag / 40.0)))
            roi_size_level0 = max(256, roi_size_level0)

            target_downsample = base_mag / max(1e-6, scan_mag)
            read_level = _choose_scan_level(slide, target_downsample=target_downsample)
            level_downsample = float(slide.level_downsamples[read_level])
            read_size = int(round(roi_size_level0 / level_downsample))
            read_size = max(64, read_size)

            if slide_index == 1 or slide_index % 25 == 0:
                logger.info(
                    "Slide %d/%d %s: base_mag=%.1f level=%d downsample=%.3f roi0=%d read_size=%d",
                    slide_index,
                    len(slide_paths),
                    slide_path.name,
                    base_mag,
                    read_level,
                    level_downsample,
                    roi_size_level0,
                    read_size,
                )

            for row in rows:
                filename = f"roi_{int(row.rank):02d}_x{int(row.x)}_y{int(row.y)}.png"
                out_path = out_dir / filename
                if only_missing and out_path.exists():
                    skipped_existing += 1
                    continue

                try:
                    region = slide.read_region((int(row.x), int(row.y)), read_level, (read_size, read_size)).convert("RGB")
                except Exception as e:
                    logger.warning("read_region failed; skipping %s rank=%d: %s", slide_path.name, int(row.rank), type(e).__name__)
                    skipped_read_fail += 1
                    continue

                if read_size != feature_size:
                    region = region.resize((feature_size, feature_size), Image.Resampling.BILINEAR)

                save_image_uint8(np.asarray(region, dtype=np.uint8), out_path)
                written += 1

    logger.info(
        "Export complete: slides=%d written=%d skipped_existing=%d skipped_missing_slide=%d skipped_read_fail=%d",
        len(slide_paths),
        written,
        skipped_existing,
        skipped_missing_slide,
        skipped_read_fail,
    )


def restructure_patches_into_split_subdirs(*, patches_root: Path) -> None:
    """Move existing per-slide patch folders into patches/{train,test}/.

    This is a filesystem-only operation. It does not touch the report CSV.
    It is meant to clean up the current mixed layout:
      patches/TUPAC-TR-xxx -> patches/train/TUPAC-TR-xxx
      patches/TUPAC-TE-xxx -> patches/test/TUPAC-TE-xxx
    """

    patches_root = Path(patches_root)
    train_root = patches_root / "train"
    test_root = patches_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    for p in sorted([d for d in patches_root.iterdir() if d.is_dir()]):
        name = p.name
        if name in {"train", "test"}:
            continue
        if name.startswith("TUPAC-TR-"):
            dest = train_root / name
        elif name.startswith("TUPAC-TE-"):
            dest = test_root / name
        else:
            skipped += 1
            continue
        if dest.exists():
            # If a dest already exists, do not merge implicitly.
            skipped += 1
            continue
        p.rename(dest)
        moved += 1

    logger.info("Restructure complete: moved=%d skipped=%d", moved, skipped)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_roi_selector_config()

    parser = argparse.ArgumentParser(description="Export/verify ROI preview patches from stage-two ROI report")
    parser.add_argument(
        "--report-csv",
        type=str,
        default=str(cfg.infer_output_dir / "reports" / "stage_two_roi_selection.csv"),
    )
    parser.add_argument(
        "--patches-root",
        type=str,
        default=str(cfg.infer_output_dir / "patches"),
    )
    parser.add_argument(
        "--split-subdirs",
        action="store_true",
        help="Use patches/train and patches/test subdirectories to avoid mixing TR/TE under one folder.",
    )
    parser.add_argument(
        "--restructure-split",
        action="store_true",
        help="Move existing patches/<slide_id> folders into patches/{train,test}/ (based on TUPAC-TR/TE prefix).",
    )
    parser.add_argument(
        "--verify-test",
        action="store_true",
        help="Verify existing test patches match report rows (dataset/test)",
    )
    parser.add_argument(
        "--prune-test-extras",
        action="store_true",
        help=(
            "If set together with --verify-test, move extra/duplicate test preview patches that do not match the report "
            "into output/roi_selector/outputs/patches_orphans/<slide_id>/ so patches become 1:1 with the report."
        ),
    )
    parser.add_argument(
        "--export-train",
        action="store_true",
        help="Export missing train preview patches from report into patches-root (dataset/train)",
    )
    parser.add_argument("--max-slides", type=int, default=0, help="Limit slides (0=all)")
    args = parser.parse_args()

    report_csv = Path(args.report_csv)
    patches_root = Path(args.patches_root)
    max_slides = None if int(args.max_slides) <= 0 else int(args.max_slides)

    if args.verify_test:
        verify_preview_patches_match_report(
            report_csv=report_csv,
            patches_root=patches_root,
            expected_split_prefix="dataset/test/",
            max_slides=max_slides,
            prune_extras=bool(args.prune_test_extras),
            orphans_root=(cfg.infer_output_dir / "patches_orphans") if bool(args.prune_test_extras) else None,
            split_subdirs=bool(args.split_subdirs),
        )

    if args.restructure_split:
        restructure_patches_into_split_subdirs(patches_root=patches_root)

    if args.export_train:
        export_missing_preview_patches_from_report(
            report_csv=report_csv,
            out_patches_root=patches_root,
            only_split_prefix="dataset/train/",
            only_missing=True,
            max_slides=max_slides,
            split_subdirs=bool(args.split_subdirs),
        )


if __name__ == "__main__":
    main()
