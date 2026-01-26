"""Configuration models for ROI selector (incorporating stage one data preparation)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from cabcds.config import Config


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


class ROISelectorConfig(BaseSettings):
    """Unified configuration for ROI Selector (Data Prep, Training, Inference)."""

    class Config:
        env_prefix = "CABCDS_ROI_"
        env_nested_delimiter = "__"
        frozen = True

    # --- Data Loader ---
    preproc_dataset_dir: Path = Field(
        default=Path("dataset/train"),
        description="Directory containing raw WSI-derived images for preprocessing."
    )
    preproc_output_dir: Path = Field(
        default=Path("output/roi_selector/preprocessed"),
        description="Directory to store normalized images and masks."
    )
    preproc_reference_image_path: Path | None = Field(
        default=None,
        description="Optional path to a reference image for stain normalization."
    )
    preproc_image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"),
        description="Allowed image extensions for preprocessing."
    )
    preproc_min_blob_area: int = Field(
        default=10, 
        description="Minimum blob area (in pixels) to keep after Otsu thresholding."
    )
    preproc_overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing preprocessing outputs."
    )
    preproc_max_images: int | None = Field(
        default=None,
        description="Optional cap on the number of images to preprocess."
    )
    
    # --- Data Generation ---
    roi_csv_dir: Path = Field(
        default=Path("dataset/auxiliary_dataset_roi"),
        description="Directory containing ROI CSV files."
    )
    sampling_magnification: float = Field(
        default=40.0,
        description="Magnification to sample patches at (e.g., 40.0, 10.0, 20.0)."
    )

    # --- Training ---
    train_dataset_dir: Path = Field(
        default=Path("output/roi_selector/training"),
        description="Directory containing ROI training patches."
    )
    train_positive_subdir: str = Field(default="positive")
    train_negative_subdir: str = Field(default="negative")
    train_model_output_path: Path = Field(default=Path("output/roi_selector/models/roi_svm.joblib"))
    train_image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs")
    )
    train_svm_c: float = Field(default=1.0, gt=0.0)

    # --- Inference ---
    infer_wsi_dir: Path = Field(
        default=Path("dataset/test"),
        description="Directory containing WSI images for inference (test)."
    )
    infer_output_dir: Path = Field(
        default=Path("output/roi_selector/outputs"),
        description="Output directory for ROI results."
    )
    infer_model_path: Path = Field(default=Path("output/roi_selector/models/roi_svm.joblib"))
    infer_image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs")
    )
    infer_patch_size: int = Field(default=512, ge=64)
    infer_overlap_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    infer_exclude_border: int = Field(default=1000, ge=0)
    infer_top_n: int = Field(default=4, ge=1)
    infer_save_patches: bool = Field(default=True)
    infer_max_images: int | None = Field(default=None)

    # --- Shared Feature Config ---
    feature_config: RoiSelectorFeatureConfig = Field(default_factory=RoiSelectorFeatureConfig)

    # --- Initialization (Directory Creation) ---
    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.preproc_output_dir.mkdir(parents=True, exist_ok=True)
        self.infer_output_dir.mkdir(parents=True, exist_ok=True)


def load_roi_selector_config() -> ROISelectorConfig:
    """Load unified ROI selector configuration.

    Returns:
        ROISelectorConfig instance.
    """
    global_config = Config()
    config = ROISelectorConfig()

    if global_config.debug:
        debug_limit = global_config.debug_max_images
        
        # Apply debug limit to both preprocessing and inference
        updates = {}
        if config.preproc_max_images is None or config.preproc_max_images > debug_limit:
            updates["preproc_max_images"] = debug_limit
        if config.infer_max_images is None or config.infer_max_images > debug_limit:
            updates["infer_max_images"] = debug_limit
            
        if updates:
             config = config.model_copy(update=updates)

    return config
