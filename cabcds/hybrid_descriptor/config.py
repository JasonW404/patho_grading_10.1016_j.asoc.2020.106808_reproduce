"""Configuration models for hybrid descriptor stage."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class HybridDescriptorConfig(BaseSettings):
    """Configuration for hybrid descriptor aggregation.

    Attributes:
        br_threshold_low: Lower threshold for BR score mapping.
        br_threshold_high: Upper threshold for BR score mapping.
        roi_score_strategy: Aggregation strategy for ROI scores.
    """

    class Config:
        env_prefix = "CABCDS_HYBRID_"
        env_nested_delimiter = "__"
        frozen = True

    br_threshold_low: float = Field(default=8.0, ge=0.0)
    br_threshold_high: float = Field(default=14.0, ge=0.0)
    roi_score_strategy: str = Field(default="mean")


class HybridDescriptorInferenceConfig(BaseSettings):
    """Configuration for building hybrid descriptors from ROI patches.

    Attributes:
        roi_patches_dir: Directory containing ROI patch folders per WSI.
        output_dir: Directory for descriptor outputs.
        segmentation_model_path: Path to CNN_seg model weights.
        detection_model_path: Path to CNN_det model weights.
        roi_scoring_model_path: Path to CNN_global model weights.
        device: Torch device string.
        image_extensions: Allowed image extensions.
        min_blob_area: Minimum blob area for segmentation blobs.
        detection_patch_size: Crop size around blob centroid for detection.
        detection_resize: Resize size for detection network.
        roi_resize: Resize size for ROI scoring network.
        batch_size: Batch size for ROI scoring.
        use_detection: Whether to run CNN_det for mitosis counting.
        use_segmentation: Whether to run CNN_seg for blob detection.
    """

    class Config:
        env_prefix = "CABCDS_HYBRID_INFER_"
        env_nested_delimiter = "__"
        frozen = True

    roi_patches_dir: str = Field(default="data/roi_selector/outputs/patches")
    output_dir: str = Field(default="data/hybrid_descriptor")
    segmentation_model_path: str | None = Field(default=None)
    detection_model_path: str | None = Field(default=None)
    roi_scoring_model_path: str | None = Field(default=None)
    device: str = Field(default="cpu")
    image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    min_blob_area: int = Field(default=10, ge=1)
    detection_patch_size: int = Field(default=80, ge=16)
    detection_resize: int = Field(default=227, ge=16)
    roi_resize: int = Field(default=227, ge=16)
    batch_size: int = Field(default=8, ge=1)
    use_detection: bool = Field(default=True)
    use_segmentation: bool = Field(default=True)


def load_hybrid_descriptor_config() -> HybridDescriptorConfig:
    """Load hybrid descriptor configuration.

    Returns:
        HybridDescriptorConfig instance.
    """

    return HybridDescriptorConfig()


def load_hybrid_descriptor_inference_config() -> HybridDescriptorInferenceConfig:
    """Load hybrid descriptor inference configuration.

    Returns:
        HybridDescriptorInferenceConfig instance.
    """

    return HybridDescriptorInferenceConfig()
