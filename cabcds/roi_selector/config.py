"""Configuration models for stage two ROI selection."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RoiSelectorFeatureConfig(BaseModel):
    """Feature extraction settings for ROI selection.

    Attributes:
        color_bins: Number of histogram bins per color channel.
        lbp_neighbors: Number of neighbors for LBP.
        lbp_radius: Radius for LBP.
        lbp_method: LBP method string.
        min_blob_area: Minimum blob area for cell counting.
        white_pixel_threshold: Grayscale threshold for white pixel detection.
        max_white_ratio: Maximum ratio of white pixels allowed in a patch.
    """

    color_bins: int = Field(default=20, ge=2)
    lbp_neighbors: int = Field(default=8, ge=1)
    lbp_radius: int = Field(default=1, ge=1)
    lbp_method: str = Field(default="uniform")
    min_blob_area: int = Field(default=10, ge=1)
    white_pixel_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_white_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class RoiSelectorTrainingConfig(BaseSettings):
    """Configuration for training the ROI selector SVM.

    Attributes:
        dataset_dir: Directory containing ROI training patches.
        positive_subdir: Subdirectory name for positive patches.
        negative_subdir: Subdirectory name for negative patches.
        model_output_path: Output path for the trained SVM model.
        image_extensions: Supported image extensions.
        svm_c: Regularization strength for linear SVM.
        feature_config: Feature extraction settings.
    """

    class Config:
        env_prefix = "CABCDS_ROI_TRAIN_"
        env_nested_delimiter = "__"
        frozen = True

    dataset_dir: Path = Field(default=Path("data/roi_selector/training"))
    positive_subdir: str = Field(default="positive")
    negative_subdir: str = Field(default="negative")
    model_output_path: Path = Field(default=Path("data/roi_selector/models/roi_svm.joblib"))
    image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    svm_c: float = Field(default=1.0, gt=0.0)
    feature_config: RoiSelectorFeatureConfig = Field(default_factory=RoiSelectorFeatureConfig)


class RoiSelectorInferenceConfig(BaseSettings):
    """Configuration for ROI selection over WSI images.

    Attributes:
        wsi_dir: Directory containing WSI images.
        output_dir: Output directory for ROI results.
        model_path: Path to trained SVM model.
        image_extensions: Supported image extensions.
        patch_size: Patch size in pixels.
        overlap_ratio: Overlap ratio for sliding window.
        exclude_border: Border size to skip (in pixels).
        top_n: Number of top ROIs to keep.
        save_patches: Whether to save selected ROI patches.
        feature_config: Feature extraction settings.
    """

    class Config:
        env_prefix = "CABCDS_ROI_INFER_"
        env_nested_delimiter = "__"
        frozen = True

    wsi_dir: Path = Field(default=Path("dataset"))
    output_dir: Path = Field(default=Path("data/roi_selector/outputs"))
    model_path: Path = Field(default=Path("data/roi_selector/models/roi_svm.joblib"))
    image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    patch_size: int = Field(default=512, ge=64)
    overlap_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    exclude_border: int = Field(default=1000, ge=0)
    top_n: int = Field(default=4, ge=1)
    save_patches: bool = Field(default=True)
    feature_config: RoiSelectorFeatureConfig = Field(default_factory=RoiSelectorFeatureConfig)


def load_roi_selector_training_config() -> RoiSelectorTrainingConfig:
    """Load ROI selector training configuration.

    Returns:
        RoiSelectorTrainingConfig instance.
    """

    return RoiSelectorTrainingConfig()


def load_roi_selector_inference_config() -> RoiSelectorInferenceConfig:
    """Load ROI selector inference configuration.

    Returns:
        RoiSelectorInferenceConfig instance.
    """

    return RoiSelectorInferenceConfig()
