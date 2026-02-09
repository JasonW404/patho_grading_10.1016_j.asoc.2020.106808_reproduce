"""Patch-based datasets and splitting for CNN_seg.

The MF-CNN paper trains CNN_seg on a mitosis auxiliary dataset where each image
is large (either 2000x2000 or 5657x5657). Training uses 512x512 patches with an
80px overlap.

This module provides:
- deterministic patch grid generation
- image-level train/val/test splits to prevent leakage across patches
- a Dataset that yields (image_patch_tensor, mask_patch_tensor)

Masks are generated from centroid point annotations using BR+Otsu nuclei blobs,
consistent with the existing implementation in cabcds.mf_cnn.utils.loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from cabcds.mf_cnn.utils.loader import (
    MitosisSampleKey,
    _centroids_to_mask_via_nuclei_blobs,
    _find_tile_zip,
    _index_external_centroid_pairs,
    _index_mitosis_tiles,
    _read_centroids_from_path,
    _read_centroids_from_zip,
    _read_rgb_from_path,
    _read_rgb_from_zip,
    _to_tensor_rgb,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegImageRef:
    """Reference to one mitosis auxiliary image + its centroid annotations."""

    kind: str  # "zip" or "path"
    image_ref: str | Path
    centroid_ref: str | Path
    zip_parts: tuple[Path, ...] = ()
    ground_truth_zip: Path | None = None

    @property
    def uid(self) -> str:
        if self.kind == "zip":
            return str(self.image_ref)
        return str(Path(self.image_ref))


def compute_patch_grid(*, height: int, width: int, patch_size: int, overlap: int) -> list[tuple[int, int]]:
    """Compute a deterministic sliding-window grid (top, left).

    The stride is `patch_size - overlap`. The last patch is forced to align with
    the image boundary to ensure full coverage.
    """

    patch_size = int(patch_size)
    overlap = int(overlap)
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid stride: patch_size={patch_size} overlap={overlap}")

    max_top = max(0, int(height) - patch_size)
    max_left = max(0, int(width) - patch_size)

    tops: list[int] = list(range(0, max_top + 1, stride))
    lefts: list[int] = list(range(0, max_left + 1, stride))
    if not tops:
        tops = [0]
    if not lefts:
        lefts = [0]
    if tops[-1] != max_top:
        tops.append(max_top)
    if lefts[-1] != max_left:
        lefts.append(max_left)

    coords: list[tuple[int, int]] = []
    for top in tops:
        for left in lefts:
            coords.append((int(top), int(left)))
    return coords


def _read_image_and_centroids(ref: SegImageRef) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if ref.kind == "zip":
        member = str(ref.image_ref)
        assert ref.ground_truth_zip is not None
        zip_path = _find_tile_zip(ref.zip_parts, member)
        image = _read_rgb_from_zip(zip_path, member)
        centroids = _read_centroids_from_zip(ref.ground_truth_zip, str(ref.centroid_ref))
        return image, centroids

    image = _read_rgb_from_path(Path(ref.image_ref))
    centroids = _read_centroids_from_path(Path(ref.centroid_ref))
    return image, centroids


def _read_centroids_only(ref: SegImageRef) -> list[tuple[int, int]]:
    """Read centroid annotations without loading the full image.

    This is used to compute patch-level positivity for sampling.
    """

    if ref.kind == "zip":
        assert ref.ground_truth_zip is not None
        return _read_centroids_from_zip(ref.ground_truth_zip, str(ref.centroid_ref))
    return _read_centroids_from_path(Path(ref.centroid_ref))


def _scan_unzipped_tupac_folder(root: Path) -> list[SegImageRef]:
    """Scan a folder for extracted TUPAC auxiliary dataset files."""
    refs: list[SegImageRef] = []
    
    # Try to locate the ground truth folder
    gt_root_candidates = [
        root / "mitoses_ground_truth",
        root / "auxiliary_dataset_mitoses" / "mitoses_ground_truth"
    ]
    gt_root = next((p for p in gt_root_candidates if p.exists() and p.is_dir()), None)
    
    # Fallback: assume strict structure if not found explicitly
    if not gt_root:
        gt_root = root / "mitoses_ground_truth"

    logger.info("Scanning for unzipped TUPAC data in %s (GT root: %s)", root, gt_root)
    
    count = 0
    # Walk all .tif files. We look for pattern CaseID/TileID.tif
    for img_path in root.rglob("*.tif"):
        # Explicitely exclude test dataset folder if present to avoid ID collisions
        if "mitoses-test-image-data" in img_path.parts:
            continue

        tile_id = img_path.stem
        case_id = img_path.parent.name
        
        # Check GT at gt_root/case_id/tile_id.csv
        gt_path = gt_root / case_id / f"{tile_id}.csv"
        
        if gt_path.exists():
            refs.append(SegImageRef(kind="path", image_ref=img_path, centroid_ref=gt_path))
            count += 1
            
    logger.info("Found %d unzipped image+GT pairs", count)
    return refs


def build_seg_image_refs(
    *,
    image_zip_parts: Sequence[Path],
    ground_truth_zip: Path,
    external_roots: Sequence[Path] | None,
    require_external_pairs: bool = False,
) -> list[SegImageRef]:
    """Build a unified list of image references from zips + optional external datasets."""

    image_zip_parts = tuple(Path(p) for p in image_zip_parts)
    ground_truth_zip = Path(ground_truth_zip)

    # Check if zips actually exist.
    zips_exist = all(p.exists() and p.is_file() for p in image_zip_parts) and ground_truth_zip.exists() and ground_truth_zip.is_file()
    
    refs: list[SegImageRef] = []
    keys: list[MitosisSampleKey] = []

    # Priority: Unzipped folder at dataset/tupac16/auxiliary_dataset_mitoses
    # User requested to prioritise extracted data and the structure is now fixed.
    unzipped_folder = Path("dataset/tupac16/auxiliary_dataset_mitoses")
    if unzipped_folder.exists():
        logger.info("Scanning unzipped folder: %s", unzipped_folder)
        refs.extend(_scan_unzipped_tupac_folder(unzipped_folder))

    if not refs and zips_exist:
        keys = _index_mitosis_tiles(image_zip_parts, ground_truth_zip=ground_truth_zip)
        for k in keys:
            refs.append(
                SegImageRef(
                    kind="zip",
                    image_ref=k.rel_image_path,
                    centroid_ref=k.rel_centroid_path,
                    zip_parts=image_zip_parts,
                    ground_truth_zip=ground_truth_zip,
                )
            )
    elif not refs:
        # Fallback to unzipped dir scanning from root if nothing else worked
        scan_root = ground_truth_zip.parent
        logger.warning(
            "Zip files missing or incomplete and default unzipped folder empty. Scanning root: %s", 
            scan_root
        )
        refs.extend(_scan_unzipped_tupac_folder(scan_root))

    external_total = 0
    for root in (external_roots or ()):  # type: ignore[assignment]
        pairs = _index_external_centroid_pairs(Path(root))
        if require_external_pairs and not pairs:
            raise FileNotFoundError(
                f"External dataset root has no usable image+centroid pairs: {Path(root)}. "
                "Expected extracted images plus centroid CSV/TXT files on disk."
            )
        for image_path, centroid_path in pairs:
            refs.append(SegImageRef(kind="path", image_ref=image_path, centroid_ref=centroid_path))
        external_total += len(pairs)

    logger.info("CNN_seg image refs: zip=%d external=%d total=%d", len(keys), external_total, len(refs))
    return refs


def split_uids(
    uids: Sequence[str],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split unique image IDs into train/val/test sets."""

    train_frac = float(train_frac)
    val_frac = float(val_frac)
    test_frac = float(test_frac)
    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise ValueError("Split fractions must be positive")
    train_frac /= total
    val_frac /= total
    test_frac /= total

    uids_list = list(uids)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uids_list)

    n = len(uids_list)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = max(0, min(n, n_train))
    n_val = max(0, min(n - n_train, n_val))
    n_test = max(0, n - n_train - n_val)

    train = set(uids_list[:n_train])
    val = set(uids_list[n_train : n_train + n_val])
    test = set(uids_list[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


class SegPatchDataset(Dataset[tuple[Tensor, Tensor]]):
    """Patch-grid dataset for CNN_seg.

    Yields:
        (image_tensor_chw, mask_tensor_hw)
    """

    def __init__(
        self,
        *,
        image_refs: Sequence[SegImageRef],
        allowed_uids: set[str],
        patch_size: int,
        overlap: int,
        nuclei_min_area: int,
        centroid_fallback_radius: int,
        augment: bool = False,
        seed: int = 1337,
    ) -> None:
        self.image_refs = [r for r in image_refs if r.uid in allowed_uids]
        self.allowed_uids = set(allowed_uids)
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)
        self.nuclei_min_area = int(nuclei_min_area)
        self.centroid_fallback_radius = int(centroid_fallback_radius)
        self.augment = bool(augment)
        self.seed = int(seed)

        patches: list[tuple[int, int, int]] = []
        patch_has_positive: list[bool] = []
        for idx, ref in enumerate(self.image_refs):
            centroids = _read_centroids_only(ref)
            if ref.kind == "zip":
                member = str(ref.image_ref)
                assert ref.ground_truth_zip is not None
                zip_path = _find_tile_zip(ref.zip_parts, member)
                with ZipFile(zip_path) as z:
                    with z.open(member) as fp:
                        with Image.open(fp) as im:
                            width, height = im.size
            else:
                with Image.open(Path(ref.image_ref)) as im:
                    width, height = im.size

            for top, left in compute_patch_grid(
                height=int(height),
                width=int(width),
                patch_size=self.patch_size,
                overlap=self.overlap,
            ):
                patches.append((idx, top, left))

                # Patch is considered positive if any centroid falls within it.
                # This is a much cheaper signal than computing the full blob mask
                # for every patch, and it correlates strongly with non-empty masks.
                right = int(left) + int(self.patch_size)
                bottom = int(top) + int(self.patch_size)
                has_pos = any(int(left) <= int(x) < right and int(top) <= int(y) < bottom for x, y in centroids)
                patch_has_positive.append(bool(has_pos))

        self._patch_index = patches
        self._patch_has_positive = patch_has_positive
        logger.info(
            "CNN_seg patches: images=%d patches=%d patch=%d overlap=%d",
            len(self.image_refs),
            len(self._patch_index),
            self.patch_size,
            self.overlap,
        )

        pos_count = int(sum(1 for v in self._patch_has_positive if v))
        logger.info(
            "CNN_seg patches: positive_patches=%d/%d (%.2f%%)",
            pos_count,
            len(self._patch_has_positive),
            (100.0 * pos_count / max(1, len(self._patch_has_positive))),
        )

        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: list[str] = []
        # Increase cache size to utilize available RAM (2TB system)
        # 1000 images * 16MB ~= 16GB per worker. With 16 workers, max 256GB usage. Secure.
        self._cache_max = 1000

    def __len__(self) -> int:
        return len(self._patch_index)

    def patch_has_positive(self) -> list[bool]:
        """Return patch-level positivity flags.

        A patch is marked positive if it contains at least one centroid.
        """

        return list(self._patch_has_positive)

    def _get_cached(self, ref: SegImageRef) -> tuple[np.ndarray, np.ndarray]:
        uid = ref.uid
        cached = self._cache.get(uid)
        if cached is not None:
            return cached

        image, centroids = _read_image_and_centroids(ref)
        mask = _centroids_to_mask_via_nuclei_blobs(
            image,
            centroids,
            min_area=self.nuclei_min_area,
            fallback_radius=self.centroid_fallback_radius,
        )

        self._cache[uid] = (image, mask)
        self._cache_order.append(uid)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return image, mask

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img_idx, top, left = self._patch_index[int(index)]
        ref = self.image_refs[int(img_idx)]
        image, mask = self._get_cached(ref)

        ps = self.patch_size
        image_patch = image[top : top + ps, left : left + ps]
        mask_patch = mask[top : top + ps, left : left + ps]

        if self.augment:
            rng = np.random.default_rng(self.seed + int(index))
            if float(rng.random()) < 0.5:
                image_patch = image_patch[:, ::-1].copy()
                mask_patch = mask_patch[:, ::-1].copy()
            if float(rng.random()) < 0.5:
                image_patch = image_patch[::-1, :].copy()
                mask_patch = mask_patch[::-1, :].copy()

        image_tensor = _to_tensor_rgb(image_patch)
        mask_tensor = torch.from_numpy(mask_patch.astype(np.int64))
        return image_tensor, mask_tensor
