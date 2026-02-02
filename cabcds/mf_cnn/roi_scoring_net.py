"""Compatibility wrappers for MF-CNN ROI scoring network.

Historically, some modules import `RoiScoringNet` from here.
The underlying architecture is AlexNet-based and matches `CNNGlobal`.
"""

from __future__ import annotations

from .cnn import CNNGlobal


class RoiScoringNet(CNNGlobal):
    """AlexNet-based 3-class ROI scoring network."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)


__all__ = ["RoiScoringNet"]
