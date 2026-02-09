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
    seed: int = 1337,
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
        seed: RNG seed used for negative downsampling.

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

    logger.info(
        "Preparing CNN_det patches: tiles=%d crop=%d max_cand=%d max_neg_per_pos=%d match_r=%d out=%s",
        len(tiles),
        int(crop_size),
        int(max_candidates_per_tile),
        int(max_negatives_per_positive),
        int(match_radius),
        str(out_root),
    )

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
        pred = torch.argmax(logits, dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)

        labeled = measure.label(pred > 0)
        regions = [r for r in measure.regionprops(labeled) if r.area > 0]
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
        for gx, gy in centroids:
            if 0 <= gy < h and 0 <= gx < w:
                gx_i, gy_i = int(gx), int(gy)
                if int(match_radius) <= 0:
                    lab = int(labeled[gy_i, gx_i])
                    if lab > 0:
                        positive_labels.add(lab)
                    continue

                r = int(match_radius)
                y0 = max(0, gy_i - r)
                y1 = min(h, gy_i + r + 1)
                x0 = max(0, gx_i - r)
                x1 = min(w, gx_i + r + 1)
                window = labeled[y0:y1, x0:x1]
                labs = np.unique(window)
                for lab in labs:
                    lab_i = int(lab)
                    if lab_i > 0:
                        positive_labels.add(lab_i)

        positives: list[tuple[np.ndarray, int, int, int]] = []
        negatives: list[tuple[np.ndarray, int, int, int]] = []

        half = int(crop_size) // 2
        for region in regions:
            cy, cx = region.centroid
            cy_i, cx_i = int(round(cy)), int(round(cx))
            top = max(0, min(h - int(crop_size), cy_i - half))
            left = max(0, min(w - int(crop_size), cx_i - half))
            patch = image[top : top + int(crop_size), left : left + int(crop_size)]
            label = 1 if int(region.label) in positive_labels else 0
            if label == 1:
                positives.append((patch, label, int(cx_i), int(cy_i)))
            else:
                negatives.append((patch, label, int(cx_i), int(cy_i)))

        if not positives and not negatives:
            continue

        # Downsample negatives for balance.
        if positives:
            max_negs = int(max_negatives_per_positive) * len(positives)
            if len(negatives) > max_negs:
                keep_idx = rng.choice(len(negatives), size=max_negs, replace=False)
                negatives = [negatives[int(i)] for i in keep_idx]

        selected = positives + negatives

        # Write patches.
        out_dir = out_root / "patches" / f"{case_id}" / f"{tile_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, (patch, label, cx_i, cy_i) in enumerate(selected, start=1):
            filename = f"cand_{idx:03d}_x{cx_i}_y{cy_i}_label{label}.png"
            out_path = out_dir / filename
            Image.fromarray(patch).save(out_path)
            rows.append(
                DetPatchIndexRow(
                    path=out_path,
                    label=int(label),
                    case_id=str(case_id),
                    tile_id=str(tile_id),
                    cx=int(cx_i),
                    cy=int(cy_i),
                )
            )

        if tile_index == 1 or tile_index % 50 == 0:
            logger.info(
                "det prepare progress: %d/%d tiles, rows=%d (pos_labels=%d)",
                tile_index,
                len(tiles),
                len(rows),
                len(positive_labels),
            )

    with index_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "case_id", "tile_id", "cx", "cy"])
        for r in rows:
            writer.writerow([str(r.path), int(r.label), r.case_id, r.tile_id, int(r.cx), int(r.cy)])

    logger.info("Wrote CNN_det index: %s (rows=%d)", str(index_csv), len(rows))
    return rows
