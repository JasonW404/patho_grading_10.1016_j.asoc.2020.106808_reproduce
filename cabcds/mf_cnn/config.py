"""Configuration models for MF-CNN stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class MfcCnnConfig(BaseSettings):
    """Configuration for MF-CNN stage.

    Attributes:
        output_dir: Base output directory for MF-CNN artifacts.
        seg_num_classes: Number of classes for segmentation output.
        det_num_classes: Number of classes for detection output.
        global_num_classes: Number of classes for ROI scoring output.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    class Config:
        env_prefix = "CABCDS_MFCNN_"
        env_nested_delimiter = "__"
        frozen = True

    output_dir: Path = Field(default=Path("data/mf_cnn"))
    seg_num_classes: int = Field(default=2, ge=2)
    det_num_classes: int = Field(default=2, ge=2)
    global_num_classes: int = Field(default=3, ge=2)
    pretrained: bool = Field(default=True)


def load_mfc_cnn_config() -> MfcCnnConfig:
    """Load MF-CNN configuration.

    Returns:
        MfcCnnConfig instance.
    """

    return MfcCnnConfig()
