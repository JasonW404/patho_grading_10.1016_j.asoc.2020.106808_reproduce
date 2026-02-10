"""Paper-aligned CNN_det data preparation.

The paper trains CNN_det on 80x80 candidate patches produced by a trained
CNN_seg model. This module implements that behavior for the auxiliary mitosis
zip dataset shipped with this repo.

Workflow:
- Run CNN_seg inference on each auxiliary tile.
- Convert the predicted mitosis mask into connected components (candidate blobs).
- For each blob, crop an 80x80 patch centered on the blob centroid.
- Label the patch positive if the blob contains any pathologist-provided
  centroid (ground truth) for that tile; otherwise label it negative.
- Optionally downsample negatives to keep a roughly balanced dataset.

The output is an `index.csv` plus on-disk PNG patches.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure

from cabcds.mf_cnn.checkpoints import load_cnn_seg_from_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetPatchIndexRow:
    """One row in the prepared CNN_det patch index."""

    path: Path
    label: int
    case_id: str
    tile_id: str
    cx: int
    cy: int
    top: int = 0
    left: int = 0
    area: int = 0
    mean_prob: float = 0.0
    max_prob: float = 0.0
    is_gt_forced: int = 0


def _match_centroid_to_label(
    labeled: np.ndarray,
    *,
    gx_i: int,
    gy_i: int,
    match_radius: int,
) -> int:
    """Match a GT centroid to at most one connected-component label.

    Priority:
      1) If centroid falls inside a component, return that label.
      2) Else, within `match_radius`, pick the label with the nearest labeled
         pixel (ties broken arbitrarily).

    Returns:
        Label id (>0) if matched, otherwise 0.
    """

    h, w = labeled.shape[:2]
    if not (0 <= gy_i < h and 0 <= gx_i < w):
        return 0

    lab0 = int(labeled[gy_i, gx_i])
    if lab0 > 0:
        return lab0

    r = int(match_radius)
    if r <= 0:
        return 0

    y0 = max(0, gy_i - r)
    y1 = min(h, gy_i + r + 1)
    x0 = max(0, gx_i - r)
    x1 = min(w, gx_i + r + 1)
    window = labeled[y0:y1, x0:x1]
    labs = np.unique(window)
    labs = labs[labs > 0]
    if labs.size == 0:
        return 0

    # Centroid position in window coordinates.
    gy_w = gy_i - y0
    gx_w = gx_i - x0

    best_lab = 0
    best_d2 = int(r * r) + 1
    for lab in labs:
        ys, xs = np.nonzero(window == int(lab))
        if ys.size == 0:
            continue
        d2 = int(np.min((ys - gy_w) ** 2 + (xs - gx_w) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_lab = int(lab)

    return int(best_lab) if best_d2 <= int(r * r) else 0


def _iter_tile_members(
    image_zip_parts: Iterable[Path],
    *,
    ground_truth_zip: Path,
) -> list[tuple[str, str, Path]]:
    """List available auxiliary tiles that have matching ground-truth CSVs."""

    ground_truth_zip = Path(ground_truth_zip)
    with ZipFile(ground_truth_zip) as zf:
        gt_members = set(zf.namelist())

    seen: set[str] = set()
    out: list[tuple[str, str, Path]] = []
    for zip_path in image_zip_parts:
        zip_path = Path(zip_path)
        with ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".tif"):
                    continue
                try:
                    case_id, filename = name.split("/", 1)
                except ValueError:
                    continue
                tile_id = Path(filename).stem
                rel = f"{case_id}/{tile_id}.tif"
                if rel in seen:
                    continue
                seen.add(rel)
                gt_rel = f"mitoses_ground_truth/{case_id}/{tile_id}.csv"
                if gt_rel not in gt_members:
                    continue
                out.append((case_id, tile_id, zip_path))

    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _read_rgb_from_zip(zip_path: Path, member: str) -> np.ndarray:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as fp:
            image = Image.open(fp).convert("RGB")
            return np.asarray(image)


def _read_centroids_from_zip(zip_path: Path, member: str) -> list[tuple[int, int]]:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as fp:
            content = fp.read().decode("utf-8", errors="replace")

    content = content.strip()
    if not content:
        return []

    out: list[tuple[int, int]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        x = int(float(parts[0]))
        y = int(float(parts[1]))
        out.append((x, y))
    return out


def _imagenet_normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def prepare_cnn_det_patches_from_aux_zips(
    *,
    image_zip_parts: Iterable[Path],
    ground_truth_zip: Path,
    seg_checkpoint_path: Path,
    out_root: Path,
    index_csv: Path,
    device: torch.device,
    crop_size: int = 80,
    max_tiles: int | None = None,
    max_candidates_per_tile: int = 200,
    max_negatives_per_positive: int = 3,
    match_radius: int = 30,
    add_gt_patches: bool = False,
    gt_patches_missing_only: bool = True,
    min_region_area: int = 1,
    min_region_mean_prob: float = 0.0,
    min_region_max_prob: float = 0.0,
    seed: int = 1337,
    collect_rows: bool = True,
) -> list[DetPatchIndexRow]:
    """Prepare CNN_det training patches using CNN_seg candidates.

    Args:
        image_zip_parts: Auxiliary image zip parts containing `.tif` tiles.
        ground_truth_zip: Zip containing centroid CSVs.
        seg_checkpoint_path: Path to `cnn_seg.pt` checkpoint saved by training.
        out_root: Output folder for generated patches.
        index_csv: Output CSV path.
        device: Torch device to run CNN_seg inference.
        crop_size: Candidate crop size in pixels (paper uses 80).
        max_tiles: Optional cap for quick experiments.
        max_candidates_per_tile: Cap connected components per tile.
        max_negatives_per_positive: Balance by keeping at most this many negatives per positive.
        match_radius: Radius in pixels for matching a GT centroid to a predicted component.
        add_gt_patches: If True, add GT-centered positive patches even when no nearby component exists.
        gt_patches_missing_only: If True, only add GT patches when the centroid didn't match any component.
        min_region_area: Filter out candidate components smaller than this area.
        min_region_mean_prob: Filter out components with mean predicted prob below this.
        min_region_max_prob: Filter out components with max predicted prob below this.
        seed: RNG seed used for negative downsampling.
        collect_rows: If False, avoid keeping rows in memory (still writes CSV).

    Returns:
        List of index rows written.
    """

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load CNN_seg.
    model = load_cnn_seg_from_checkpoint(Path(seg_checkpoint_path), map_location="cpu")
    model.to(device)
    model.eval()

    tiles = _iter_tile_members(tuple(image_zip_parts), ground_truth_zip=Path(ground_truth_zip))
    if max_tiles is not None:
        tiles = tiles[: int(max_tiles)]

    rng = np.random.default_rng(int(seed))
    rows: list[DetPatchIndexRow] = []
    row_count = 0

    logger.info(
        "Preparing CNN_det patches: tiles=%d crop=%d max_cand=%d max_neg_per_pos=%d match_r=%d out=%s",
        len(tiles),
        int(crop_size),
        int(max_candidates_per_tile),
        int(max_negatives_per_positive),
        int(match_radius),
        str(out_root),
    )

    index_header = [
        "path",
        "label",
        "case_id",
        "tile_id",
        "cx",
        "cy",
        "top",
        "left",
        "area",
        "mean_prob",
        "max_prob",
        "is_gt_forced",
    ]

    rows_since_flush = 0
    with index_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(index_header)

        for tile_index, (case_id, tile_id, zip_path) in enumerate(tiles, start=1):
            rel_image = f"{case_id}/{tile_id}.tif"
            rel_gt = f"mitoses_ground_truth/{case_id}/{tile_id}.csv"
            image = _read_rgb_from_zip(zip_path, rel_image)
            centroids = _read_centroids_from_zip(Path(ground_truth_zip), rel_gt)

            h, w = image.shape[:2]
            if h < crop_size or w < crop_size:
                continue

            # Run segmentation on full tile.
            x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = x.to(device)
            x = _imagenet_normalize_batch(x)
            logits = model(x).logits
            if int(logits.shape[1]) == 1:
                prob_pos_t = torch.sigmoid(logits)[0, 0]
                pred = (prob_pos_t > 0.5).to("cpu").numpy().astype(np.uint8)
            else:
                prob_pos_t = F.softmax(logits, dim=1)[0, 1]
                pred = torch.argmax(logits, dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)
            prob_pos = prob_pos_t.to("cpu").numpy().astype(np.float32)

            labeled = measure.label(pred > 0)
            regions = [
                r
                for r in measure.regionprops(labeled, intensity_image=prob_pos)
                if int(r.area) >= int(min_region_area)
                and float(getattr(r, "mean_intensity", 0.0)) >= float(min_region_mean_prob)
                and float(getattr(r, "max_intensity", 0.0)) >= float(min_region_max_prob)
            ]
            if not regions:
                continue

            # Sort biggest-first and cap.
            regions.sort(key=lambda r: float(r.area), reverse=True)
            regions = regions[: int(max_candidates_per_tile)]

        # Determine which labels are positive.
        #
        # The paper labels a candidate blob as positive if it contains a GT
        # centroid. In practice, CNN_seg predictions can be slightly offset
        # from the pathologist centroid, so we allow a small matching radius.
            positive_labels: set[int] = set()
            matched_centroids: list[tuple[int, int, bool]] = []
            for gx, gy in centroids:
                gx_i, gy_i = int(gx), int(gy)
                lab = _match_centroid_to_label(labeled, gx_i=gx_i, gy_i=gy_i, match_radius=int(match_radius))
                matched = bool(lab > 0)
                matched_centroids.append((gx_i, gy_i, matched))
                if matched:
                    positive_labels.add(int(lab))

            positives: list[tuple[np.ndarray, int, int, int, int, int, int, float, float, int]] = []
            negatives: list[tuple[np.ndarray, int, int, int, int, int, int, float, float, int]] = []

            half = int(crop_size) // 2
            for region in regions:
                cy, cx = region.centroid
                cy_i, cx_i = int(round(cy)), int(round(cx))
                top = max(0, min(h - int(crop_size), cy_i - half))
                left = max(0, min(w - int(crop_size), cx_i - half))
                patch = image[top : top + int(crop_size), left : left + int(crop_size)]
                label = 1 if int(region.label) in positive_labels else 0
                area = int(getattr(region, "area", 0))
                mean_prob = float(getattr(region, "mean_intensity", 0.0))
                max_prob = float(getattr(region, "max_intensity", 0.0))
                if label == 1:
                    positives.append(
                        (
                            patch,
                            label,
                            int(cx_i),
                            int(cy_i),
                            int(top),
                            int(left),
                            int(area),
                            float(mean_prob),
                            float(max_prob),
                            0,
                        )
                    )
                else:
                    negatives.append(
                        (
                            patch,
                            label,
                            int(cx_i),
                            int(cy_i),
                            int(top),
                            int(left),
                            int(area),
                            float(mean_prob),
                            float(max_prob),
                            0,
                        )
                    )

            # Optionally add GT-forced positive patches, so CNN_det sees positives even
            # when CNN_seg fails to propose a candidate near the GT centroid.
            if bool(add_gt_patches):
                for gx_i, gy_i, matched in matched_centroids:
                    if bool(gt_patches_missing_only) and bool(matched):
                        continue
                    if not (0 <= gy_i < h and 0 <= gx_i < w):
                        continue
                    top = max(0, min(h - int(crop_size), int(gy_i) - half))
                    left = max(0, min(w - int(crop_size), int(gx_i) - half))
                    patch = image[top : top + int(crop_size), left : left + int(crop_size)]
                    positives.append(
                        (
                            patch,
                            1,
                            int(gx_i),
                            int(gy_i),
                            int(top),
                            int(left),
                            0,
                            0.0,
                            0.0,
                            1,
                        )
                    )

            if not positives and not negatives:
                continue

        # Downsample negatives for balance.
            if positives:
                max_negs = int(max_negatives_per_positive) * len(positives)
                if len(negatives) > max_negs:
                    keep_idx = rng.choice(len(negatives), size=max_negs, replace=False)
                    negatives = [negatives[int(i)] for i in keep_idx]

            selected = positives + negatives

            # Write patches and index rows.
            out_dir = out_root / "patches" / f"{case_id}" / f"{tile_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, (patch, label, cx_i, cy_i, top, left, area, mean_prob, max_prob, is_gt_forced) in enumerate(selected, start=1):
                filename = f"cand_{idx:03d}_x{cx_i}_y{cy_i}_label{label}.png"
                out_path = out_dir / filename
                Image.fromarray(patch).save(out_path)

                row = DetPatchIndexRow(
                    path=out_path,
                    label=int(label),
                    case_id=str(case_id),
                    tile_id=str(tile_id),
                    cx=int(cx_i),
                    cy=int(cy_i),
                    top=int(top),
                    left=int(left),
                    area=int(area),
                    mean_prob=float(mean_prob),
                    max_prob=float(max_prob),
                    is_gt_forced=int(is_gt_forced),
                )
                if bool(collect_rows):
                    rows.append(row)
                row_count += 1
                writer.writerow(
                    [
                        str(row.path),
                        int(row.label),
                        row.case_id,
                        row.tile_id,
                        int(row.cx),
                        int(row.cy),
                        int(row.top),
                        int(row.left),
                        int(row.area),
                        float(row.mean_prob),
                        float(row.max_prob),
                        int(row.is_gt_forced),
                    ]
                )
                rows_since_flush += 1

                if rows_since_flush >= 500:
                    f.flush()
                    rows_since_flush = 0

            if tile_index == 1 or tile_index % 50 == 0:
                logger.info(
                    "det prepare progress: %d/%d tiles, rows=%d (pos_labels=%d)",
                    tile_index,
                    len(tiles),
                    int(row_count),
                    len(positive_labels),
                )

        if rows_since_flush > 0:
            f.flush()

    logger.info("Wrote CNN_det index: %s (rows=%d)", str(index_csv), int(row_count))
    return rows
