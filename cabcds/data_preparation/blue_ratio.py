"""Blue-Ratio transformation and blob extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import filters, measure


@dataclass(frozen=True)
class BlobStats:
    """Summary statistics for detected blobs.

    Attributes:
        blob_count: Number of blobs after filtering.
        average_area: Average blob area in pixels.
    """

    blob_count: int
    average_area: float


def compute_blue_ratio(image: np.ndarray) -> np.ndarray:
    """Compute the blue-ratio image from an RGB array.

    Args:
        image: RGB image array with shape (H, W, 3).

    Returns:
        Blue-ratio image as float32 array.
    """

    image = image.astype(np.float32)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    denominator_rg = 1.0 + red + green
    denominator_rgb = 1.0 + red + green + blue
    blue_ratio = (100.0 * blue / denominator_rg) * (256.0 / denominator_rgb)
    return blue_ratio


def otsu_mask(blue_ratio: np.ndarray) -> np.ndarray:
    """Generate a binary mask using Otsu thresholding.

    Args:
        blue_ratio: Blue-ratio image.

    Returns:
        Boolean mask where True indicates potential blobs.
    """

    threshold = filters.threshold_otsu(blue_ratio)
    return blue_ratio > threshold


def compute_blob_stats(mask: np.ndarray, min_area: int) -> BlobStats:
    """Compute blob statistics from a binary mask.

    Args:
        mask: Binary mask of candidate blobs.
        min_area: Minimum area (in pixels) to keep a blob.

    Returns:
        BlobStats with count and average area.
    """

    labeled = measure.label(mask)
    regions = [region for region in measure.regionprops(labeled) if region.area >= min_area]
    if not regions:
        return BlobStats(blob_count=0, average_area=0.0)

    areas = np.array([region.area for region in regions], dtype=np.float32)
    return BlobStats(blob_count=len(areas), average_area=float(areas.mean()))
