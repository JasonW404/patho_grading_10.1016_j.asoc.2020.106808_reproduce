"""Feature extraction for ROI selector."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage import color, feature, measure

from ..config import RoiSelectorFeatureConfig
from .blue_ratio import compute_blue_ratio, compute_blob_stats, otsu_mask


def is_patch_too_white(patch: np.ndarray, config: RoiSelectorFeatureConfig) -> bool:
    """Determine if a patch should be filtered due to excessive white pixels.

    Args:
        patch: RGB patch array.
        config: Feature configuration.

    Returns:
        True if patch is mostly white, False otherwise.
    """

    gray = color.rgb2gray(patch)
    white_ratio = float(np.mean(gray >= config.white_pixel_threshold))
    return white_ratio >= config.max_white_ratio


def extract_patch_features(patch: np.ndarray, config: RoiSelectorFeatureConfig) -> np.ndarray:
    """Extract ROI selector features from a patch.

    Args:
        patch: RGB patch array.
        config: Feature configuration.

    Returns:
        1D feature vector.
    """

    color_features = _compute_color_histograms(patch, config.color_bins)
    texture_features = _compute_lbp_histogram(patch, config)
    cell_count = compute_cell_count(patch, config.min_blob_area)
    return np.concatenate([color_features, texture_features, np.array([cell_count], dtype=np.float32)])


def _compute_color_histograms(patch: np.ndarray, bins: int) -> np.ndarray:
    """Compute normalized color histograms for R and HSV channels.

    Args:
        patch: RGB patch array.
        bins: Number of histogram bins per channel.

    Returns:
        Normalized histogram vector.
    """

    red_channel = patch[..., 0].astype(np.float32)
    hsv = color.rgb2hsv(patch)

    histograms = [
        _normalized_histogram(red_channel, bins, (0.0, 255.0)),
        _normalized_histogram(hsv[..., 0], bins, (0.0, 1.0)),
        _normalized_histogram(hsv[..., 1], bins, (0.0, 1.0)),
        _normalized_histogram(hsv[..., 2], bins, (0.0, 1.0)),
    ]
    return np.concatenate(histograms)


def _normalized_histogram(channel: np.ndarray, bins: int, value_range: Tuple[float, float]) -> np.ndarray:
    """Compute a normalized histogram for a channel.

    Args:
        channel: 2D channel array.
        bins: Number of bins.
        value_range: Value range for histogram.

    Returns:
        Normalized histogram vector.
    """

    hist, _ = np.histogram(channel, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return hist / total


def _compute_lbp_histogram(patch: np.ndarray, config: RoiSelectorFeatureConfig) -> np.ndarray:
    """Compute LBP histogram for texture features.

    Args:
        patch: RGB patch array.
        config: Feature configuration.

    Returns:
        L2-normalized LBP histogram.
    """

    gray = color.rgb2gray(patch)
    # Convert to integers to avoid warning: "Applying `local_binary_pattern` to floating-point images..."
    gray = (gray * 255).astype(np.uint8)
    
    lbp = feature.local_binary_pattern(
        gray,
        P=config.lbp_neighbors,
        R=config.lbp_radius,
        method=config.lbp_method,
    )
    n_bins = _lbp_bin_count(config)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm <= 0:
        return np.zeros_like(hist)
    return hist / norm


def _lbp_bin_count(config: RoiSelectorFeatureConfig) -> int:
    """Compute the fixed number of LBP bins.

    Args:
        config: Feature configuration.

    Returns:
        Number of bins for the LBP histogram.
    """

    if config.lbp_method == "uniform":
        return config.lbp_neighbors + 2
    return 2 ** config.lbp_neighbors


def compute_cell_count(patch: np.ndarray, min_blob_area: int) -> float:
    """Compute the cell count using blue-ratio and Otsu segmentation.

    Args:
        patch: RGB patch array.
        min_blob_area: Minimum blob area threshold.

    Returns:
        Estimated cell count.
    """
    stats = get_tissue_stats(patch, min_blob_area)
    return float(stats[0])


def get_tissue_stats(patch: np.ndarray, min_blob_area: int) -> Tuple[int, float, float]:
    """Compute tissue statistics (count, area ratio, and potential nuclear area).

    Args:
        patch: RGB patch array.
        min_blob_area: Minimum blob area threshold.

    Returns:
        Tuple of (blob_count, foreground_ratio, dark_pixel_ratio).
    """
    blue_ratio = compute_blue_ratio(patch)
    mask = otsu_mask(blue_ratio)
    
    # Calculate stats using skimage.measure
    labeled = measure.label(mask)
    regions = [region for region in measure.regionprops(labeled) if region.area >= min_blob_area]
    
    blob_count = len(regions)
    if mask.size > 0:
        # Sum areas of valid blobs only
        valid_area = sum(r.area for r in regions)
        ratio = valid_area / mask.size
    else:
        ratio = 0.0
    
    # Additional check: Dark pixel ratio
    # Cells are usually darker than stroma. A high dark pixel ratio with low cell count
    # indicates the Blue Ratio algorithm failed to segment clearly visible cells.
    # Threshold 0.70 corresponds to pixel value ~178.
    gray = color.rgb2gray(patch)
    dark_pixel_ratio = float(np.mean(gray < 0.70))
        
    return blob_count, ratio, dark_pixel_ratio
