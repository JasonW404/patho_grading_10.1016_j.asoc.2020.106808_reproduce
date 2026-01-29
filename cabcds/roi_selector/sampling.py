"""Patch sampling utilities for ROI selection.

This module serves two roles:
- Stage 1 data preparation (positive/negative patch generation for SVM training)
- Stage 2 inference-time patch sampling (sliding-window over an image array)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import load_roi_selector_config
from .config import ROISelectorConfig
from .utils.create_roi_patches import create_positive_roi, create_negative_roi, generate_labelled_csv


@dataclass(frozen=True)
class PatchSample:
    """A sampled patch with its top-left coordinates."""

    x: int
    y: int
    patch: np.ndarray


class SlidingWindowSampler:
    """Generate candidate patches using a sliding window.

    The ROI selector uses these candidates to compute features and score them.
    """

    def __init__(self, config: ROISelectorConfig) -> None:
        self.config = config

    def iter_patches(self, image: np.ndarray) -> Iterator[PatchSample]:
        """Iterate over patch candidates in an image.

        Args:
            image: RGB image array with shape (H, W, 3).

        Yields:
            PatchSample objects.
        """

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

        height, width, _ = image.shape
        patch_size = int(self.config.infer_patch_size)

        # Step size derived from overlap ratio.
        overlap = float(self.config.infer_overlap_ratio)
        step = int(round(patch_size * (1.0 - overlap)))
        step = max(1, step)

        # Exclude borders, but clamp if the loaded image is small (e.g., SVS center crop).
        border = int(self.config.infer_exclude_border)
        if width <= 2 * border + patch_size or height <= 2 * border + patch_size:
            border = 0

        x_end = width - border - patch_size
        y_end = height - border - patch_size
        if x_end < border or y_end < border:
            return

        for y in range(border, y_end + 1, step):
            for x in range(border, x_end + 1, step):
                patch = image[y : y + patch_size, x : x + patch_size, :]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue
                yield PatchSample(x=x, y=y, patch=patch)


def main():
    """Run the sampling pipeline for training data generation."""
    config = load_roi_selector_config()
    create_positive_roi(config)
    create_negative_roi(config)
    generate_labelled_csv(config)


if __name__ == "__main__":
    main()
