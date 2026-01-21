"""Patch sampling utilities for ROI selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from cabcds.roi_selector.config import RoiSelectorInferenceConfig


@dataclass(frozen=True)
class PatchSample:
    """Sampled patch information.

    Attributes:
        x: Top-left x-coordinate.
        y: Top-left y-coordinate.
        patch: RGB patch array.
    """

    x: int
    y: int
    patch: np.ndarray


class SlidingWindowSampler:
    """Generate overlapping patches from a WSI image."""

    def __init__(self, config: RoiSelectorInferenceConfig) -> None:
        self.config = config

    def iter_patches(self, image: np.ndarray) -> Generator[PatchSample, None, None]:
        """Yield patches using a sliding window strategy.

        Args:
            image: RGB image array.

        Yields:
            PatchSample objects.
        """

        height, width = image.shape[:2]
        patch_size = self.config.patch_size
        step = max(int(patch_size * (1.0 - self.config.overlap_ratio)), 1)
        border = self.config.exclude_border

        x_start = border
        y_start = border
        x_end = max(width - border, x_start)
        y_end = max(height - border, y_start)

        for y in range(y_start, y_end - patch_size + 1, step):
            for x in range(x_start, x_end - patch_size + 1, step):
                patch = image[y : y + patch_size, x : x + patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue
                yield PatchSample(x=x, y=y, patch=patch)
