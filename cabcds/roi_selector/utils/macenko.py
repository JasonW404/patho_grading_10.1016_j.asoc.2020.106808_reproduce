"""Stain normalization using the Macenko method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MacenkoNormalizer:
    """Apply Macenko stain normalization to histology images.

    This class implements the stain normalization method described by Macenko et al. (2009).
    The algorithm normalizes the appearance of histology images by estimating stain vectors
    and concentration maps.

    Attributes:
        io_max: Maximum intensity used for optical density conversion.
        alpha: Percentile for stain angle selection.
        beta: OD threshold to exclude transparent pixels.

    The algorithm consists of the following steps:
    1.  Convert RGB image to Optical Density (OD) space.
    2.  Filter out background pixels with low OD values (below `beta`).
    3.  Compute Singular Value Decomposition (SVD) on the OD pixels to find the plane
        of stain vectors.
    4.  Project OD pixels onto the plane and calculate angles.
    5.  Estimate stain vectors using robust extremes (percentiles defined by `alpha`)
        of the angular distribution.
    6.  Deconvolve the image to get stain concentrations.
    7.  Normalize concentrations based on the 99th percentile and reconstruct the image.
    """

    io_max: float = 255.0
    alpha: float = 0.1
    beta: float = 0.15

    stain_matrix: np.ndarray | None = None
    max_concentration: np.ndarray | None = None

    def fit(self, reference_image: np.ndarray) -> None:
        """Fit stain vectors from a reference image.

        Args:
            reference_image: Reference RGB image array.
        """

        stain_matrix, concentrations = self._estimate_stain_matrix(reference_image)
        self.stain_matrix = stain_matrix
        self.max_concentration = np.percentile(concentrations, 99, axis=1)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image using the fitted stain matrix.

        Args:
            image: RGB image array.

        Returns:
            Normalized RGB image array.

        Raises:
            ValueError: If the normalizer has not been fitted.
        """

        if self.stain_matrix is None or self.max_concentration is None:
            raise ValueError("MacenkoNormalizer must be fitted before calling transform().")

        stain_matrix, concentrations = self._estimate_stain_matrix(image)
        scale = self.max_concentration / (np.percentile(concentrations, 99, axis=1) + 1e-8)
        normalized_concentrations = concentrations * scale[:, None]
        normalized_od = self.stain_matrix @ normalized_concentrations
        return self._od_to_rgb(normalized_od, image.shape)

    def _estimate_stain_matrix(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate stain matrix and concentrations for an image.

        Args:
            image: RGB image array.

        Returns:
            Tuple containing stain matrix and concentration matrix.
        """

        optical_density = self._rgb_to_od(image)
        optical_density = optical_density.reshape((-1, 3))
        filtered = optical_density[np.any(optical_density > self.beta, axis=1)]
        if filtered.shape[0] < 10:
            filtered = optical_density

        _, _, v_transpose = np.linalg.svd(filtered, full_matrices=False)
        principal_components = v_transpose[:2].T

        projected = filtered @ principal_components
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        v1 = principal_components @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v2 = principal_components @ np.array([np.cos(max_angle), np.sin(max_angle)])

        stain_matrix = np.stack([v1, v2], axis=1)
        stain_matrix = self._normalize_stain_matrix(stain_matrix)

        concentrations = np.linalg.lstsq(stain_matrix, optical_density.T, rcond=None)[0]
        return stain_matrix, concentrations

    def _normalize_stain_matrix(self, stain_matrix: np.ndarray) -> np.ndarray:
        """Normalize and order stain vectors.

        Args:
            stain_matrix: Raw stain matrix.

        Returns:
            Normalized stain matrix.
        """

        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0, keepdims=True) + 1e-8)
        if stain_matrix[0, 0] < stain_matrix[0, 1]:
            stain_matrix = stain_matrix[:, ::-1]
        return stain_matrix

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to optical density (OD).

        Args:
            image: RGB image array.

        Returns:
            Optical density array.
        """

        image = image.astype(np.float32)
        image = np.clip(image, 1, self.io_max)
        return -np.log(image / self.io_max)

    def _od_to_rgb(self, od: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        """Convert optical density back to RGB.

        Args:
            od: Optical density matrix.
            shape: Target image shape.

        Returns:
            RGB image array.
        """

        rgb = self.io_max * np.exp(-od)
        rgb = np.clip(rgb.T.reshape(shape), 0, self.io_max).astype(np.uint8)
        return rgb
