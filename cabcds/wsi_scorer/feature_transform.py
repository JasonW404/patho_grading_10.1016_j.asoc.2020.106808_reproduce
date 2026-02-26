"""Feature transforms for stage-five WSI scorer experiments.

This module exists to keep sklearn transformers importable so that joblib
artifacts can be reliably loaded across scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAblationTransformer(BaseEstimator, TransformerMixin):
    """Feature transformation for fast ablation studies.

    This transformer supports (1) scaling a set of columns and (2) selecting a
    subset of columns.

    Notes:
        - Indices are 0-based and refer to the input feature matrix.
        - The transform is stateless; fit() validates index ranges.

    Args:
        keep_indices: Optional list of column indices to keep (and reorder).
        scale_indices: Optional list of column indices to multiply by scale_factor.
        scale_factor: Multiplicative factor for scale_indices.
    """

    def __init__(
        self,
        *,
        keep_indices: list[int] | None = None,
        scale_indices: list[int] | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self.keep_indices = keep_indices
        self.scale_indices = scale_indices
        self.scale_factor = float(scale_factor)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "FeatureAblationTransformer":
        del y
        x_2d = np.asarray(x)
        if x_2d.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={x_2d.shape}")
        n_features = int(x_2d.shape[1])

        def _check_indices(indices: list[int], label: str) -> None:
            if any(i < 0 or i >= n_features for i in indices):
                raise ValueError(
                    f"{label} contains out-of-range indices for n_features={n_features}: {indices}"
                )

        if self.scale_indices is not None:
            _check_indices(list(self.scale_indices), "scale_indices")
        if self.keep_indices is not None:
            _check_indices(list(self.keep_indices), "keep_indices")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_2d = np.asarray(x)
        if x_2d.ndim == 1:
            x_2d = x_2d.reshape(1, -1)
        if x_2d.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={x_2d.shape}")

        out = x_2d
        if self.scale_indices is not None and self.scale_factor != 1.0:
            out = out.copy()
            out[:, self.scale_indices] = out[:, self.scale_indices] * self.scale_factor

        if self.keep_indices is not None:
            out = out[:, self.keep_indices]

        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:  # noqa: ANN401
        if input_features is None:
            return np.array([], dtype=object)
        input_features_arr = np.asarray(input_features, dtype=object)
        if self.keep_indices is None:
            return input_features_arr
        return input_features_arr[self.keep_indices]
