"""MF-CNN preprocessing helpers.

This module provides utilities to prepare training data for the MF-CNN stage.
The heaviest operation is extracting `512x512` patches with overlap (default 80)
from TUPAC16 WSIs for `CNN_global`.

We keep this separate from the model definitions and training loop so that
datasets can be prepared once and reused.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoiRect:
    """Axis-aligned ROI rectangle in level-0 coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return int(self.x + self.w)

    @property
    def y2(self) -> int:
        return int(self.y + self.h)


@dataclass(frozen=True)
class GlobalPatchRecord:
    """One extracted patch record for `CNN_global` training."""

    path: Path
    label: int
    slide_id: str
    level: int
    x: int
    y: int


def iter_tupac_train_slides(tupac_train_dir: Path) -> Iterator[Path]:
    """Yield `.svs` slide paths in sorted order."""

    yield from iter_wsi_slides(Path(tupac_train_dir), slide_glob="TUPAC-TR-*.svs")


def iter_wsi_slides(wsi_dir: Path, *, slide_glob: str = "*.svs") -> Iterator[Path]:
    """Yield WSI slide paths in sorted order.

    Args:
        wsi_dir: Directory containing slide files.
        slide_glob: Glob pattern used to select slides (e.g. `*.svs`).

    Yields:
        Slide file paths in lexicographic order.
    """

    for slide in sorted(Path(wsi_dir).glob(str(slide_glob))):
        if slide.is_file():
            yield slide


def compute_simple_tissue_fraction(rgb: np.ndarray) -> float:
    """Compute a cheap tissue fraction estimate.

    White background dominates WSIs. A simple heuristic is to count pixels whose
    brightness is below a threshold.
    """

    gray = rgb.mean(axis=2)
    return float((gray < 230).mean())


def load_roi_rects_from_csv(roi_csv_path: Path) -> list[RoiRect]:
    """Load ROI rectangles from a `*-ROI.csv` file.

    The ROI selector code in this repo writes ROI CSV files without a header
    and with 4 integer columns: x, y, w, h (all in level-0 coordinates).

    Args:
        roi_csv_path: Path to a ROI CSV file.

    Returns:
        List of ROI rectangles. Empty if missing/unreadable.
    """

    roi_csv_path = Path(roi_csv_path)
    if not roi_csv_path.exists():
        return []

    rects: list[RoiRect] = []
    try:
        with roi_csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                x, y, w, h = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3])))
                if w <= 0 or h <= 0:
                    continue
                rects.append(RoiRect(x=x, y=y, w=w, h=h))
    except Exception:
        logger.exception("Failed reading ROI CSV: %s", str(roi_csv_path))
        return []

    return rects


def load_roi_rects_from_stage_two_report(
    *,
    report_csv_path: Path,
    slide_id: str,
    slide_path: Path,
    top_n: int = 4,
    roi_size_40x: int = 5657,
) -> list[RoiRect]:
    """Load ROI rectangles for a slide from ROI-Selector stage-two report.

    The ROI selector inference writes a single CSV containing per-image ROI
    candidates:
        image_path,rank,x,y,score

    The `x,y` columns are the top-left coordinates in *level-0* pixel space.
    The report does not store ROI width/height; per the paper, the ROI window
    is defined as `roi_size_40x` pixels at 40x. We convert that to level-0 size
    using the slide objective power if available.

    Args:
        report_csv_path: Path to `stage_two_roi_selection.csv`.
        slide_id: Slide id, e.g. `TUPAC-TR-001`.
        slide_path: Path to the `.svs` for objective power lookup.
        top_n: Number of ROI windows to keep (paper uses 4).
        roi_size_40x: ROI window size in pixels at 40x.

    Returns:
        List of ROI rectangles (level-0 coords). Empty if none found.
    """

    report_csv_path = Path(report_csv_path)
    if not report_csv_path.exists():
        return []

    if openslide is None:
        raise RuntimeError("openslide-python is not available; cannot read .svs")

    slide_path = Path(slide_path)
    slide_id = str(slide_id)

    # Compute ROI size in level-0 coordinates.
    with openslide.OpenSlide(str(slide_path)) as slide:  # type: ignore[attr-defined]
        base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
    roi_size_level0 = int(round(int(roi_size_40x) * (base_mag / 40.0)))
    roi_size_level0 = max(256, roi_size_level0)

    ranked: list[tuple[int, RoiRect]] = []
    try:
        with report_csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = str(row.get("image_path") or "")
                if slide_id not in image_path:
                    continue
                try:
                    rank = int(float(row.get("rank") or 0))
                except Exception:
                    continue
                if rank <= 0 or rank > int(top_n):
                    continue
                try:
                    x = int(float(row.get("x") or 0))
                    y = int(float(row.get("y") or 0))
                except Exception:
                    continue
                ranked.append((int(rank), RoiRect(x=x, y=y, w=int(roi_size_level0), h=int(roi_size_level0))))
    except Exception:
        logger.exception("Failed reading ROI stage-two report: %s", str(report_csv_path))
        return []

    ranked.sort(key=lambda t: t[0])
    rects = [rect for _, rect in ranked][: int(top_n)]
    return rects


def load_candidate_centers_from_csv(candidate_csv_path: Path) -> list[tuple[int, int]]:
    """Load candidate centers from a simple `x,y` CSV.

    This is intended to be produced by a CNN_seg inference stage: each row is a
    candidate mitosis center in level-0 pixel coordinates.

    Args:
        candidate_csv_path: Path to a CSV with `x,y` per line (no header).

    Returns:
        List of (x, y) centers.
    """

    candidate_csv_path = Path(candidate_csv_path)
    if not candidate_csv_path.exists():
        return []

    centers: list[tuple[int, int]] = []
    try:
        with candidate_csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                centers.append((x, y))
    except Exception:
        logger.exception("Failed reading candidate CSV: %s", str(candidate_csv_path))
        return []

    return centers


def extract_global_patches_from_wsi(
    *,
    slide_path: Path,
    out_dir: Path,
    label: int,
    patch_size: int = 512,
    overlap: int = 80,
    level: int = 0,
    max_patches: int | None = None,
    min_tissue_fraction: float = 0.2,
    skip_existing: bool = True,
    roi_rects: Iterable[RoiRect] | None = None,
    candidate_centers: Iterable[tuple[int, int]] | None = None,
) -> list[GlobalPatchRecord]:
    """Extract overlapping patches from a WSI and write them to disk.

    Args:
        slide_path: Path to the `.svs` file.
        out_dir: Output directory for this slide's patches.
        label: Class label (typically 1/2/3).
        patch_size: Patch size in pixels.
        overlap: Overlap in pixels (stride = patch_size - overlap).
        level: OpenSlide pyramid level to read from (0 = highest resolution).
        max_patches: Optional cap to limit extraction.
        min_tissue_fraction: Skip patches with less tissue than this fraction.

    Returns:
        List of extracted patch records.
    """

    if openslide is None:
        raise RuntimeError("openslide-python is not available; cannot read .svs")

    slide_path = Path(slide_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = int(patch_size - overlap)
    if stride <= 0:
        raise ValueError(f"Invalid overlap={overlap} for patch_size={patch_size}")

    records: list[GlobalPatchRecord] = []
    slide_id = slide_path.stem
    roi_rect_list = list(roi_rects) if roi_rects is not None else []
    candidate_center_list = list(candidate_centers) if candidate_centers is not None else []

    with openslide.OpenSlide(str(slide_path)) as slide:  # type: ignore[attr-defined]
        width, height = slide.level_dimensions[int(level)]

        logger.info(
            "Extracting global patches: slide_id=%s level=%d dims=%dx%d patch=%d stride=%d min_tissue=%.3f max_patches=%s",
            slide_id,
            int(level),
            int(width),
            int(height),
            int(patch_size),
            int(stride),
            float(min_tissue_fraction),
            str(max_patches) if max_patches is not None else "None",
        )

        max_x = max(0, int(width) - int(patch_size))
        max_y = max(0, int(height) - int(patch_size))

        def _candidate_in_any_roi(cx: int, cy: int) -> bool:
            if not roi_rect_list:
                return True
            for rect in roi_rect_list:
                if rect.x <= cx < rect.x2 and rect.y <= cy < rect.y2:
                    return True
            return False

        def _extract_at(x: int, y: int) -> bool:
            rgba = slide.read_region((int(x), int(y)), int(level), (int(patch_size), int(patch_size)))
            rgb = np.asarray(rgba.convert("RGB"))
            if compute_simple_tissue_fraction(rgb) < float(min_tissue_fraction):
                return False

            filename = f"{slide_id}_L{level}_{x}_{y}.png"
            patch_path = out_dir / filename
            if (not skip_existing) or (not patch_path.exists()):
                Image.fromarray(rgb).save(patch_path)
            records.append(
                GlobalPatchRecord(
                    path=patch_path,
                    label=int(label),
                    slide_id=slide_id,
                    level=int(level),
                    x=int(x),
                    y=int(y),
                )
            )
            return True

        def _snap_to_stride(v: int) -> int:
            return int((int(v) // int(stride)) * int(stride))

        # Paper-aligned mode: crop 512x512 patches centered on candidate points
        # (typically prospective mitoses produced by CNN_seg) and optionally
        # constrain to top-N ROIs.
        if candidate_center_list:
            max_keep = int(max_patches) if max_patches is not None else None
            accepted = 0
            visited: set[tuple[int, int]] = set()

            # Deterministic shuffle by slide id to avoid always using the same
            # candidates when a cap is applied.
            rng = np.random.default_rng(abs(hash(slide_id)) % (2**32))
            order = np.arange(len(candidate_center_list))
            rng.shuffle(order)

            for idx in order:
                cx, cy = candidate_center_list[int(idx)]
                if not _candidate_in_any_roi(int(cx), int(cy)):
                    continue

                x = int(cx) - int(patch_size // 2)
                y = int(cy) - int(patch_size // 2)
                x = max(0, min(max_x, int(x)))
                y = max(0, min(max_y, int(y)))

                # Snap for consistent overlap behavior.
                x = _snap_to_stride(int(x))
                y = _snap_to_stride(int(y))
                key = (x, y)
                if key in visited:
                    continue
                visited.add(key)

                if _extract_at(int(x), int(y)):
                    accepted += 1
                    if max_keep is not None and accepted >= int(max_keep):
                        break

            logger.info(
                "Candidate-centered extraction done: slide_id=%s candidates=%d accepted=%d max_patches=%s",
                slide_id,
                len(candidate_center_list),
                int(accepted),
                str(max_patches) if max_patches is not None else "None",
            )
            return records

        # If the caller requests a cap, random sampling is much faster than a
        # full systematic scan (WSIs can be enormous). If ROI rectangles are
        # provided, sample *within* those regions (paper-aligned behavior).
        count = 0
        if max_patches is not None:
            rng = np.random.default_rng(abs(hash(slide_id)) % (2**32))
            visited: set[tuple[int, int]] = set()
            target = int(max_patches)
            # Trial cap prevents rare infinite loops when tissue is extremely sparse.
            max_trials = max(10_000, target * 200)
            trials = 0
            while count < target and trials < max_trials:
                trials += 1

                if roi_rect_list:
                    rect = roi_rect_list[int(rng.integers(0, len(roi_rect_list)))]
                    # If ROI smaller than patch, fall back to its center.
                    if rect.w < patch_size or rect.h < patch_size:
                        x = int(rect.x + rect.w // 2 - patch_size // 2)
                        y = int(rect.y + rect.h // 2 - patch_size // 2)
                    else:
                        # Sample on the stride grid within the ROI.
                        nx = max(1, ((rect.w - patch_size) // stride) + 1)
                        ny = max(1, ((rect.h - patch_size) // stride) + 1)
                        ix = int(rng.integers(0, nx))
                        iy = int(rng.integers(0, ny))
                        x = int(rect.x + ix * stride)
                        y = int(rect.y + iy * stride)
                else:
                    x = int(rng.integers(0, max_x + 1))
                    y = int(rng.integers(0, max_y + 1))

                x = _snap_to_stride(max(0, min(max_x, int(x))))
                y = _snap_to_stride(max(0, min(max_y, int(y))))
                key = (x, y)
                if key in visited:
                    continue
                visited.add(key)

                if _extract_at(x, y):
                    count += 1

            logger.info(
                "Random sampling done: slide_id=%s accepted=%d trials=%d max_trials=%d",
                slide_id,
                int(count),
                int(trials),
                int(max_trials),
            )
            return records

        if roi_rect_list:
            # Systematic scan of stride grid inside ROIs.
            visited: set[tuple[int, int]] = set()
            for rect in roi_rect_list:
                x0 = max(0, min(max_x, _snap_to_stride(rect.x)))
                y0 = max(0, min(max_y, _snap_to_stride(rect.y)))
                x1 = max(0, min(max_x, rect.x2 - patch_size))
                y1 = max(0, min(max_y, rect.y2 - patch_size))
                for y in range(int(y0), int(y1) + 1, int(stride)):
                    for x in range(int(x0), int(x1) + 1, int(stride)):
                        key = (int(x), int(y))
                        if key in visited:
                            continue
                        visited.add(key)
                        if _extract_at(int(x), int(y)):
                            count += 1
        else:
            for y in range(0, max(1, height - patch_size + 1), stride):
                for x in range(0, max(1, width - patch_size + 1), stride):
                    if _extract_at(int(x), int(y)):
                        count += 1

    return records


def write_global_patch_index_csv(records: Iterable[GlobalPatchRecord], out_csv: Path) -> None:
    """Write a `path,label` CSV for `GlobalScoringPatchDataset`."""

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "slide_id", "level", "x", "y"])
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "path": str(rec.path),
                    "label": int(rec.label),
                    "slide_id": rec.slide_id,
                    "level": int(rec.level),
                    "x": int(rec.x),
                    "y": int(rec.y),
                }
            )


def append_global_patch_index_csv(records: Iterable[GlobalPatchRecord], out_csv: Path) -> None:
    """Append records to an existing index CSV (create it if missing)."""

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "slide_id", "level", "x", "y"])
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "path": str(rec.path),
                    "label": int(rec.label),
                    "slide_id": rec.slide_id,
                    "level": int(rec.level),
                    "x": int(rec.x),
                    "y": int(rec.y),
                }
            )


def read_indexed_slide_ids(index_csv: Path) -> set[str]:
    """Return slide_ids already present in an index CSV."""

    index_csv = Path(index_csv)
    if not index_csv.exists():
        return set()

    slide_ids: set[str] = set()
    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row.get("slide_id")
            if slide_id:
                slide_ids.add(slide_id)
    return slide_ids


def read_indexed_paths(index_csv: Path) -> set[str]:
    """Return patch file paths already present in an index CSV."""

    index_csv = Path(index_csv)
    if not index_csv.exists():
        return set()

    paths: set[str] = set()
    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path")
            if path:
                paths.add(path)
    return paths


def slide_done_marker(out_root: Path, slide_id: str) -> Path:
    """Return the marker path used to indicate a slide was fully processed."""

    return Path(out_root) / slide_id / ".done"


def prepare_global_patch_dataset(
    *,
    wsi_dir: Path | None = None,
    slide_glob: str = "TUPAC-TR-*.svs",
    tupac_train_dir: Path | None = None,
    scores_by_slide_id: dict[str, int],
    out_root: Path,
    index_csv: Path,
    patch_size: int = 512,
    overlap: int = 80,
    level: int = 0,
    max_slides: int | None = None,
    max_patches_per_slide: int | None = None,
    max_patches_by_score: dict[int, int] | None = None,
    roi_csv_dir: Path | None = None,
    roi_report_csv: Path | None = None,
    roi_top_n: int = 4,
    roi_size_40x: int = 5657,
    candidate_csv_dir: Path | None = None,
    require_candidates: bool = False,
    require_roi: bool = False,
    min_tissue_fraction: float = 0.2,
    resume: bool = True,
) -> None:
    """Prepare `CNN_global` patch dataset from TUPAC train WSIs.

    This is an I/O heavy operation. The function is designed to support resume:
    if `index_csv` already contains a slide_id, that slide is skipped.

    Args:
        wsi_dir: Directory containing WSI slides.
        slide_glob: Glob pattern selecting slide files within `wsi_dir`.
        tupac_train_dir: Deprecated alias for `wsi_dir` kept for backward compatibility.
        scores_by_slide_id: Mapping `TUPAC-TR-xxx -> score` (1..3).
        out_root: Root directory where per-slide patch folders are created.
        index_csv: Output index CSV (appended incrementally).
        patch_size: Patch size (pixels).
        overlap: Patch overlap (pixels).
        level: OpenSlide pyramid level.
        max_slides: Optional cap on number of slides to process.
        max_patches_per_slide: Optional cap on patches per slide.
        max_patches_by_score: Optional mapping {1: n1, 2: n2, 3: n3} to cap
            patches per slide by slide-level score (class balancing at extraction).
        roi_csv_dir: If provided, and a matching `{slide_id}-ROI.csv` exists,
            patches are extracted within those ROIs.
        roi_report_csv: Optional path to ROI-Selector `stage_two_roi_selection.csv`.
            If provided (and `roi_csv_dir` is not), ROIs are read from the report.
        roi_top_n: Number of ROIs to use per slide when using `roi_report_csv`.
        roi_size_40x: ROI window size at 40x (paper: 5657).
        candidate_csv_dir: Optional directory containing per-slide candidate center CSVs
            named like `{slide_id}.csv` with `x,y` per line (level-0 coords).
            If present, patches are cropped centered on these candidates.
        require_candidates: If true, skip slides without candidate CSVs.
        require_roi: If true, skip slides without a ROI CSV.
        min_tissue_fraction: Skip near-white patches.
        resume: Skip slides already present in index CSV.
    """

    out_root = Path(out_root)
    index_csv = Path(index_csv)
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_wsi_dir = Path(wsi_dir) if wsi_dir is not None else None
    if resolved_wsi_dir is None and tupac_train_dir is not None:
        resolved_wsi_dir = Path(tupac_train_dir)
    if resolved_wsi_dir is None:
        raise ValueError("Must provide wsi_dir (or deprecated tupac_train_dir)")

    existing_paths = read_indexed_paths(index_csv) if resume else set()
    processed = 0

    for slide_path in iter_wsi_slides(resolved_wsi_dir, slide_glob=str(slide_glob)):
        slide_id = slide_path.stem
        if resume and slide_done_marker(out_root, slide_id).exists():
            continue

        if max_slides is not None and processed >= int(max_slides):
            break

        label = scores_by_slide_id.get(slide_id)
        if label is None:
            logger.warning("No score found for slide_id=%s; skipping", slide_id)
            continue

        roi_rects: list[RoiRect] = []
        if roi_csv_dir is not None:
            roi_csv_path = Path(roi_csv_dir) / f"{slide_id}-ROI.csv"
            roi_rects = load_roi_rects_from_csv(roi_csv_path)
            if roi_rects:
                logger.info("Loaded %d ROI rects for slide_id=%s", len(roi_rects), slide_id)
            elif require_roi:
                logger.warning("No ROI CSV for slide_id=%s (expected %s); skipping", slide_id, str(roi_csv_path))
                continue
        elif roi_report_csv is not None:
            roi_rects = load_roi_rects_from_stage_two_report(
                report_csv_path=Path(roi_report_csv),
                slide_id=slide_id,
                slide_path=slide_path,
                top_n=int(roi_top_n),
                roi_size_40x=int(roi_size_40x),
            )
            if roi_rects:
                logger.info("Loaded %d ROI rects from report for slide_id=%s", len(roi_rects), slide_id)
            elif require_roi:
                logger.warning(
                    "No ROI rows in report for slide_id=%s (report=%s); skipping",
                    slide_id,
                    str(roi_report_csv),
                )
                continue

        candidate_centers: list[tuple[int, int]] = []
        if candidate_csv_dir is not None:
            candidate_csv_path = Path(candidate_csv_dir) / f"{slide_id}.csv"
            candidate_centers = load_candidate_centers_from_csv(candidate_csv_path)
            if candidate_centers:
                logger.info(
                    "Loaded %d candidate centers for slide_id=%s",
                    len(candidate_centers),
                    slide_id,
                )
            elif require_candidates:
                logger.warning(
                    "No candidate CSV for slide_id=%s (expected %s); skipping",
                    slide_id,
                    str(candidate_csv_path),
                )
                continue

        slide_max_patches = max_patches_per_slide
        if max_patches_by_score is not None and int(label) in max_patches_by_score:
            slide_max_patches = int(max_patches_by_score[int(label)])

        slide_out_dir = out_root / slide_id
        records = extract_global_patches_from_wsi(
            slide_path=slide_path,
            out_dir=slide_out_dir,
            label=int(label),
            patch_size=int(patch_size),
            overlap=int(overlap),
            level=int(level),
            max_patches=slide_max_patches,
            min_tissue_fraction=float(min_tissue_fraction),
            skip_existing=bool(resume),
            roi_rects=roi_rects if roi_rects else None,
            candidate_centers=candidate_centers if candidate_centers else None,
        )

        # Append only new paths to avoid duplicating CSV rows on resume.
        new_records: list[GlobalPatchRecord] = []
        for rec in records:
            path_str = str(rec.path)
            if resume and path_str in existing_paths:
                continue
            existing_paths.add(path_str)
            new_records.append(rec)
        append_global_patch_index_csv(new_records, index_csv)

        # Mark slide as processed; resume will rely on this to avoid skipping
        # partially-processed slides (e.g., crash mid-slide).
        marker = slide_done_marker(out_root, slide_id)
        marker.write_text("ok\n", encoding="utf-8")
        processed += 1
        logger.info(
            "Prepared CNN_global patches for %s: %d patches (%d new indexed)",
            slide_id,
            len(records),
            len(new_records),
        )

        if max_slides is not None and processed >= int(max_slides):
            break
