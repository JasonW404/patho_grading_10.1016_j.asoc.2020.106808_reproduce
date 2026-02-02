"""Compatibility wrappers for MF-CNN segmentation network.

Some parts of the repo (e.g. `cabcds.hybrid_descriptor`) historically imported
`CNNSeg` from `cabcds.mf_cnn.mitosis_segmentation_net`.

The canonical implementation now lives in `cabcds.mf_cnn.cnn`.
"""

from __future__ import annotations

from .cnn import CNNSeg, SegmentationOutput

__all__ = ["CNNSeg", "SegmentationOutput"]
