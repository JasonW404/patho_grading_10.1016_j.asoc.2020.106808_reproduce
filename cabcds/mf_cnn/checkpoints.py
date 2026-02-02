"""Checkpoint loading utilities for MF-CNN.

This repo has evolved the `CNNSeg` implementation. In particular, the FCN head
(bottleneck) changed from a lightweight 1x1-conv approximation to a more
FCN-like 7x7/1x1 head.

A CNN_seg checkpoint is therefore not guaranteed to load into the *current*
`CNNSeg` class.

This module provides a small compatibility layer that auto-detects the
checkpoint variant and instantiates the matching model class.
"""

from __future__ import annotations

from pathlib import Path

import pickle

import torch

from cabcds.mf_cnn.cnn import CNNSeg, CNNSegLegacy


def _extract_model_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        state = payload.get("model_state")
        if isinstance(state, dict):
            return state
        state = payload.get("state_dict")
        if isinstance(state, dict):
            return state
    raise ValueError("Unexpected checkpoint format (expected dict with model_state/state_dict)")


def _torch_load_compat(*, path: Path, map_location: str | torch.device) -> object:
    """Load a torch checkpoint across PyTorch versions.

    PyTorch 2.6 changed `torch.load` default `weights_only` to True. Older
    checkpoints that include non-tensor objects (e.g., `pathlib.Path` inside a
    config dict) may fail to load with `weights_only=True`.

    For checkpoints produced locally by this repo (trusted source), we retry
    with `weights_only=False` when needed.
    """

    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as e:
        # Retry for trusted local checkpoints.
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older torch versions do not support `weights_only`.
            raise e


def load_cnn_seg_from_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a CNN_seg model from a checkpoint, handling legacy variants.

    Args:
        checkpoint_path: Path to a saved checkpoint.
        map_location: torch.load map_location.

    Returns:
        A CNNSeg (new) or CNNSegLegacy (old) instance with weights loaded.
    """

    payload = _torch_load_compat(path=Path(checkpoint_path), map_location=map_location)
    state = _extract_model_state(payload)

    # Variant detection by key pattern:
    # - Legacy used a 5-layer Sequential head with indices 0/2/4.
    # - New uses a 7-layer Sequential head with indices 0/3/6.
    if any(k.startswith("fc_block.6.") or k.startswith("fc_block.3.") for k in state):
        num_classes = int(state["fc_block.6.weight"].shape[0])
        model: torch.nn.Module = CNNSeg(num_classes=num_classes, pretrained=False)
    elif any(k.startswith("fc_block.4.") or k.startswith("fc_block.2.") for k in state):
        num_classes = int(state["fc_block.4.weight"].shape[0])
        model = CNNSegLegacy(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError("Could not determine CNNSeg variant from checkpoint keys")

    model.load_state_dict(state, strict=True)
    return model
