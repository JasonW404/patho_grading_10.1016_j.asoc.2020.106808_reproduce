"""Benchmark (holdout) patch selection.

We create a benchmark folder containing a small holdout set of patches that:
- will NOT be used for training
- can be used later for evaluation/QA

Requirements
------------
- Sample ~N patches from positive and negative_generated
- Copy images into benchmark directory
- Write an index CSV with WSI name and patch coordinates (level-0)

Coordinate conventions
----------------------
- Negative patches: filenames encode (x, y) in level-0 pixels as ..._neg_<x>_<y>_...
- Positive patches: most filenames encode (x, y) as ..._roi<idx>_<x>_<y>.png
  For legacy filenames like ..._roi<idx>_p0.png (no coords), we infer coordinates
  from the corresponding ROI CSV using the same sampling logic used during extraction.
"""

from __future__ import annotations

import csv
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ROISelectorConfig
from .utils.create_roi_patches import compute_sampling_params, load_existing_rois

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkItem:
    subset: str  # "positive" | "negative"
    label: int  # 1 for positive, 0 for negative
    wsi_name: str
    x: int
    y: int
    w: int
    h: int
    source_path: Path
    benchmark_path: Path


_POS_XY_RE = re.compile(r"^(?P<wsi>.+)_roi(?P<roi>\d+)_(?P<x>\d+)_(?P<y>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
_POS_P0_RE = re.compile(r"^(?P<wsi>.+)_roi(?P<roi>\d+)_p0\.(png|jpg|jpeg)$", re.IGNORECASE)
_NEG_RE = re.compile(r"^(?P<wsi>.+)_neg_(?P<x>\d+)_(?P<y>\d+)_.*\.(png|jpg|jpeg)$", re.IGNORECASE)


def _find_wsi_path(wsi_name: str, config: ROISelectorConfig) -> Path | None:
    wsi_dir = config.preproc_dataset_dir
    for ext in config.preproc_image_extensions:
        p = wsi_dir / f"{wsi_name}{ext}"
        if p.exists():
            return p
    return None


def _parse_positive_coords(path: Path, config: ROISelectorConfig) -> tuple[str, int, int]:
    name = path.name
    m = _POS_XY_RE.match(name)
    if m:
        return m.group("wsi"), int(m.group("x")), int(m.group("y"))

    m = _POS_P0_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized positive patch filename: {name}")

    wsi_name = m.group("wsi")
    roi_index = int(m.group("roi"))

    # Infer coords using the ROI CSV and the same sampling params used for extraction.
    roi_csv_path = config.roi_csv_dir / f"{wsi_name}-ROI.csv"
    rois = load_existing_rois(roi_csv_path)
    if roi_index >= len(rois):
        raise ValueError(f"ROI index out of range for {roi_csv_path}: {roi_index}")

    rx, ry, rw, rh = rois[roi_index]

    wsi_path = _find_wsi_path(wsi_name, config)
    if wsi_path is None:
        raise FileNotFoundError(f"Cannot find WSI for {wsi_name} in {config.preproc_dataset_dir}")

    import openslide

    with openslide.OpenSlide(str(wsi_path)) as slide:
        read_level, level_downsample, src_patch_size, read_size = compute_sampling_params(slide, config)

    # Same computation as create_positive_roi() center-crop branch.
    cx = rx + rw // 2
    cy = ry + rh // 2
    px = max(0, cx - src_patch_size // 2)
    py = max(0, cy - src_patch_size // 2)
    return wsi_name, int(px), int(py)


def _parse_negative_coords(path: Path) -> tuple[str, int, int]:
    name = path.name
    m = _NEG_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized negative patch filename: {name}")
    return m.group("wsi"), int(m.group("x")), int(m.group("y"))


def _src_patch_size_level0(wsi_name: str, config: ROISelectorConfig, cache: dict[str, int]) -> int:
    if wsi_name in cache:
        return cache[wsi_name]

    wsi_path = _find_wsi_path(wsi_name, config)
    if wsi_path is None:
        raise FileNotFoundError(f"Cannot find WSI for {wsi_name} in {config.preproc_dataset_dir}")

    import openslide

    with openslide.OpenSlide(str(wsi_path)) as slide:
        _, _, src_patch_size, _ = compute_sampling_params(slide, config)

    cache[wsi_name] = int(src_patch_size)
    return int(src_patch_size)


def _read_existing_benchmark_sources(index_csv: Path) -> set[str]:
    if not index_csv.exists():
        return set()
    df = pd.read_csv(index_csv)
    if "source_path" not in df.columns:
        return set()
    return {str(Path(p).expanduser().resolve()) for p in df["source_path"].dropna().astype(str).tolist()}


def prune_benchmark_sources(config: ROISelectorConfig, *, move: bool = True) -> tuple[int, int]:
    """Remove benchmark-selected source patches from training directories.

    By default this *moves* files into an archive directory under train_dataset_dir to avoid
    losing data. It also updates ROI_labelled.csv (if present) to remove rows that reference
    the pruned source paths.

    Returns:
        Tuple of (files_moved_or_deleted, csv_rows_removed)
    """

    bench_dir = config.benchmark_dir
    index_csv = bench_dir / "benchmark_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Benchmark index not found: {index_csv}")

    df = pd.read_csv(index_csv)
    if "source_path" not in df.columns:
        raise ValueError("benchmark_index.csv missing source_path column")

    sources = [Path(p).expanduser().resolve() for p in df["source_path"].dropna().astype(str).tolist()]

    archive_root = config.train_dataset_dir / str(config.benchmark_archive_subdir)
    moved = 0

    for src in sources:
        if not src.exists():
            continue

        subset = "unknown"
        try:
            # Preserve subset based on parent folder name where possible.
            subset = src.parent.name
        except Exception:
            pass

        dst_dir = archive_root / subset
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if move:
            if not dst.exists():
                shutil.move(str(src), str(dst))
            else:
                # If destination exists, avoid overwrite; remove source instead.
                src.unlink(missing_ok=True)
        else:
            src.unlink(missing_ok=True)
        moved += 1

    # Update ROI_labelled.csv to avoid missing paths during training.
    csv_path = config.train_dataset_dir / "ROI_labelled.csv"
    removed_rows = 0
    if csv_path.exists():
        roi_df = pd.read_csv(csv_path).astype({"path": str, "label": int})
        before = len(roi_df)
        exclude_set = {str(p) for p in sources}
        roi_df = roi_df[~roi_df["path"].isin(exclude_set)].reset_index(drop=True)
        removed_rows = before - len(roi_df)
        roi_df.to_csv(csv_path, index=False)

    logger.info(
        "Pruned benchmark sources: moved_or_deleted=%d, removed_rows_from_ROI_labelled=%d",
        moved,
        removed_rows,
    )
    return moved, removed_rows


def _copy_selected(
    *,
    subset: str,
    label: int,
    selected: list[Path],
    out_dir: Path,
    config: ROISelectorConfig,
    src_patch_cache: dict[str, int],
) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in selected:
        src = p.expanduser().resolve()
        if subset == "positive":
            wsi_name, x, y = _parse_positive_coords(src, config)
        else:
            wsi_name, x, y = _parse_negative_coords(src)

        # Patch rectangle in level-0 coordinates.
        side = _src_patch_size_level0(wsi_name, config, src_patch_cache)
        w = side
        h = side

        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

        items.append(
            BenchmarkItem(
                subset=subset,
                label=label,
                wsi_name=wsi_name,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                source_path=src,
                benchmark_path=dst,
            )
        )

    return items


def create_benchmark(
    config: ROISelectorConfig,
    *,
    positive_count: int = 200,
    negative_count: int = 200,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Create a benchmark holdout set from existing training patches.

    Returns:
        Path to the benchmark index CSV.
    """

    bench_dir = config.benchmark_dir
    pos_dir = config.train_dataset_dir / config.train_positive_subdir
    neg_dir = config.train_dataset_dir / config.train_negative_generated_subdir

    if not pos_dir.exists():
        raise FileNotFoundError(f"Positive dir not found: {pos_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"Negative generated dir not found: {neg_dir}")

    index_csv = bench_dir / "benchmark_index.csv"
    if overwrite and bench_dir.exists():
        shutil.rmtree(bench_dir)

    bench_dir.mkdir(parents=True, exist_ok=True)

    existing_sources = _read_existing_benchmark_sources(index_csv)

    pos_all = sorted([p for p in pos_dir.glob("*.png") if str(p.resolve()) not in existing_sources])
    neg_all = sorted([p for p in neg_dir.glob("*.png") if str(p.resolve()) not in existing_sources])

    rng = random.Random(seed)
    rng.shuffle(pos_all)
    rng.shuffle(neg_all)

    pos_sel = pos_all[: min(positive_count, len(pos_all))]
    neg_sel = neg_all[: min(negative_count, len(neg_all))]

    logger.info(
        "Creating benchmark: positive=%d/%d negative=%d/%d -> %s",
        len(pos_sel),
        positive_count,
        len(neg_sel),
        negative_count,
        bench_dir,
    )

    items: list[BenchmarkItem] = []
    src_patch_cache: dict[str, int] = {}
    items += _copy_selected(
        subset="positive",
        label=1,
        selected=pos_sel,
        out_dir=bench_dir / "positive",
        config=config,
        src_patch_cache=src_patch_cache,
    )
    items += _copy_selected(
        subset="negative",
        label=0,
        selected=neg_sel,
        out_dir=bench_dir / "negative",
        config=config,
        src_patch_cache=src_patch_cache,
    )

    # Write/append index CSV
    write_header = not index_csv.exists()
    with index_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["subset", "label", "wsi_name", "x", "y", "w", "h", "source_path", "benchmark_path"])
        for it in items:
            w.writerow(
                [
                    it.subset,
                    it.label,
                    it.wsi_name,
                    it.x,
                    it.y,
                    it.w,
                    it.h,
                    str(it.source_path),
                    str(it.benchmark_path),
                ]
            )

    logger.info("Wrote benchmark index to %s (%d items)", index_csv, len(items))
    return index_csv
