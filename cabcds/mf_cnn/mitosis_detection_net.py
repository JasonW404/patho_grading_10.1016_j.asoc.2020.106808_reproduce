"""Compatibility wrappers for MF-CNN detection network.

The canonical implementation lives in `cabcds.mf_cnn.cnn`.
"""

from __future__ import annotations

from .cnn import CNNDet

__all__ = ["CNNDet"]
