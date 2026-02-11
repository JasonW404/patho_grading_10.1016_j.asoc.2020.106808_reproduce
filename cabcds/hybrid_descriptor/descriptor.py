"""Hybrid descriptor computation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cabcds.hybrid_descriptor.config import HybridDescriptorConfig


@dataclass(frozen=True)
class RoiMetrics:
    """Per-ROI metrics required for hybrid descriptor.

    Attributes:
        blob_count: Number of blobs in the ROI.
        average_blob_area: Average blob area in pixels.
        mitosis_count: Number of mitoses detected in the ROI.
        roi_score: ROI score predicted by CNN_global.
    """

    blob_count: int
    average_blob_area: float
    mitosis_count: int
    roi_score: float


class HybridDescriptorBuilder:
    """Build the 15-dimensional hybrid descriptor."""

    def __init__(self, config: HybridDescriptorConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self, roi_metrics: list[RoiMetrics]) -> np.ndarray:
        """Compute the hybrid descriptor from per-ROI metrics.

        Args:
            roi_metrics: List of ROI metric entries.

        Returns:
            1D numpy array with 15 features.
        """

        if not roi_metrics:
            raise ValueError("roi_metrics cannot be empty.")

        blob_counts = np.array([metric.blob_count for metric in roi_metrics], dtype=np.float32)
        blob_areas = np.array([metric.average_blob_area for metric in roi_metrics], dtype=np.float32)
        mitosis_counts = np.array([metric.mitosis_count for metric in roi_metrics], dtype=np.float32)
        roi_scores = np.array([metric.roi_score for metric in roi_metrics], dtype=np.float32)

        avg_blob_area = float(blob_areas.mean())
        max_blobs = float(blob_counts.max())
        min_blobs = float(blob_counts.min())
        avg_blobs = float(blob_counts.mean())
        sd_blobs = float(blob_counts.std(ddof=0))

        max_mitoses = float(mitosis_counts.max())
        min_mitoses = float(mitosis_counts.min())
        avg_mitoses = float(mitosis_counts.mean())
        sd_mitoses = float(mitosis_counts.std(ddof=0))

        br_max = float(_br_score(max_mitoses, self.config))
        br_min = float(_br_score(min_mitoses, self.config))
        br_avg = float(_br_score(avg_mitoses, self.config))

        ratio_avg = _safe_divide(avg_mitoses, avg_blobs)
        ratio_max = _safe_divide(max_mitoses, max_blobs)

        roi_score = _aggregate_roi_score(roi_scores, self.config.roi_score_strategy)

        descriptor = np.array(
            [
                avg_blob_area,
                max_blobs,
                min_blobs,
                avg_blobs,
                sd_blobs,
                max_mitoses,
                min_mitoses,
                avg_mitoses,
                sd_mitoses,
                br_max,
                br_min,
                br_avg,
                ratio_avg,
                ratio_max,
                roi_score,
            ],
            dtype=np.float32,
        )

        self.logger.info("Hybrid descriptor built with %d ROIs.", len(roi_metrics))
        return descriptor


# Public feature names in the exact order produced by `HybridDescriptorBuilder.build()`.
HYBRID_DESCRIPTOR_FEATURE_NAMES: tuple[str, ...] = (
    "avg_blob_area",
    "blob_count_max",
    "blob_count_min",
    "blob_count_mean",
    "blob_count_sd",
    "mitosis_count_max",
    "mitosis_count_min",
    "mitosis_count_mean",
    "mitosis_count_sd",
    "br_score_max",
    "br_score_min",
    "br_score_mean",
    "mitosis_per_blob_mean",
    "mitosis_per_blob_max",
    "roi_score",
)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two scalars.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.

    Returns:
        Division result or 0.0 if denominator is zero.
    """

    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _br_score(mitosis_count: float, config: HybridDescriptorConfig) -> int:
    """Map mitosis counts to Bloom-Richardson score.

    Args:
        mitosis_count: Mitosis count.
        config: Hybrid descriptor configuration.

    Returns:
        BR score (1, 2, or 3).
    """

    if mitosis_count < config.br_threshold_low:
        return 1
    if mitosis_count <= config.br_threshold_high:
        return 2
    return 3


def _aggregate_roi_score(roi_scores: np.ndarray, strategy: str) -> float:
    """Aggregate ROI scores.

    Args:
        roi_scores: Array of ROI scores.
        strategy: Aggregation strategy (mean or max).

    Returns:
        Aggregated ROI score.
    """

    if strategy == "max":
        return float(roi_scores.max())
    if strategy == "mean":
        return float(roi_scores.mean())

    raise ValueError(f"Unsupported roi_score_strategy: {strategy}")
