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
    train_negative_generated_subdir: str = Field(
        default="negative_generated",
        description=(
            "Directory name under train_dataset_dir for newly generated negative patches. "
            "Kept separate from train_negative_subdir to avoid overwriting manually labelled negatives."
        ),
    )
    train_model_output_path: Path = Field(default=Path("output/roi_selector/models/roi_svm.joblib"))
    train_image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs")
    )
    train_svm_c: float = Field(default=1.0, gt=0.0)

    # --- Benchmark (Holdout patches, excluded from training) ---
    benchmark_dir: Path = Field(
        default=Path("output/roi_selector/benchmark"),
        description="Directory to store benchmark/holdout patches and their index CSV.",
    )
    benchmark_exclude_from_training: bool = Field(
        default=True,
        description="If true, automatically exclude benchmark source patches from SVM training.",
    )
    benchmark_archive_subdir: str = Field(
        default="holdout_removed",
        description=(
            "Subdirectory under train_dataset_dir where benchmark source patches are moved when pruning. "
            "This avoids deleting data while ensuring training directories no longer contain holdout samples."
        ),
    )

    neg_avoid_benchmark: bool = Field(
        default=True,
        description=(
            "If true, negative sampling avoids generating patches that overlap benchmark rectangles for the same WSI."
        ),
    )

    # --- Negative Sampling (Training Data) ---
    neg_total_target_count: int = Field(
        default=600,
        ge=1,
        description="Target number of negative patches to generate.",
    )
    neg_max_workers: int = Field(
        default=16,
        ge=1,
        description="Max processes for negative sampling (kept modest to reduce WSI I/O contention).",
    )
    neg_attempts_per_slide: int = Field(
        default=500,
        ge=1,
        description="How many random samples to try per opened slide before closing it.",
    )
    neg_attempts_multiplier: int = Field(
        default=5000,
        ge=1,
        description="Total attempts is max(target_count * multiplier, 10000).",
    )
    neg_max_cell_count: int = Field(
        default=20,
        ge=0,
        description="Negative acceptance threshold: blob count must be below this.",
    )
    neg_max_cell_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Negative acceptance threshold: segmented nuclear area ratio must be below this.",
    )
    neg_max_dark_ratio: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Negative acceptance threshold: dark pixel ratio must be below this.",
    )

    neg_clean_generated_output_dir: bool = Field(
        default=False,
        description=(
            "If true, remove the generated negative output directory before sampling. "
            "This never touches train_negative_subdir (the manually-labelled negatives)."
        ),
    )

    # --- Negative Filter (Deep Learning Assisted) ---
    neg_filter_dl_enabled: bool = Field(
        default=False,
        description=(
            "If true, apply a deep-learning negative-filter model to reject suspicious patches during negative sampling."
        ),
    )
    neg_filter_dl_model_path: Path = Field(
        default=Path("output/roi_selector/models/neg_filter_dl.pt"),
        description="Path to a trained deep-learning negative-filter model (saved by torch).",
    )
    neg_filter_dl_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Reject patch if DL P(suspicious) >= threshold.",
    )
    neg_filter_dl_input_size: int = Field(
        default=224,
        ge=64,
        description="Input size (pixels) used by the DL negative filter model.",
    )
    neg_filter_dl_epochs: int = Field(
        default=8,
        ge=1,
        description="Training epochs for DL negative filter.",
    )
    neg_filter_dl_batch_size: int = Field(
        default=32,
        ge=1,
        description="Training batch size for DL negative filter.",
    )
    neg_filter_dl_lr: float = Field(
        default=1e-3,
        gt=0.0,
        description="Training learning rate for DL negative filter.",
    )

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
    infer_scan_magnification: float = Field(
        default=10.0,
        description=(
            "Magnification used to scan WSIs during ROI selection (paper uses 10x). "
            "This controls the OpenSlide level used for sliding-window scanning."
        ),
    )
    infer_roi_size_40x: int = Field(
        default=5657,
        ge=256,
        description=(
            "ROI side length (pixels) at 40x corresponding to ~10 HPF (paper uses 5657). "
            "Used to derive the scanning window size at infer_scan_magnification."
        ),
    )
    infer_save_full_roi_40x: bool = Field(
        default=False,
        description=(
            "If true, also save the full ROI crop at level-0 resolution sized according to infer_roi_size_40x. "
            "This can be large on disk."
        ),
    )
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

    # Auto-detect alternate dataset layouts.
    # New layout (this repo commonly uses):
    #   dataset/tupac16/{train,test,auxiliary_dataset_roi,auxiliary_dataset_mitoses}
    # Old layout:
    #   dataset/{train,test,auxiliary_dataset_roi,auxiliary_dataset_mitoses}
    #
    # We only override when the configured path does not exist, so explicit env
    # overrides (CABCDS_ROI_*) still take priority.
    base = Path("dataset")
    tupac16_root = base / "tupac16"
    if tupac16_root.exists() and tupac16_root.is_dir():
        candidate_train = tupac16_root / "train"
        candidate_test = tupac16_root / "test"
        candidate_roi = tupac16_root / "auxiliary_dataset_roi"

        overrides: dict[str, object] = {}
        if not config.preproc_dataset_dir.exists() and candidate_train.exists():
            overrides["preproc_dataset_dir"] = candidate_train
        if not config.infer_wsi_dir.exists() and candidate_test.exists():
            overrides["infer_wsi_dir"] = candidate_test
        if not config.roi_csv_dir.exists() and candidate_roi.exists():
            overrides["roi_csv_dir"] = candidate_roi

        if overrides:
            config = config.model_copy(update=overrides)

    return config
