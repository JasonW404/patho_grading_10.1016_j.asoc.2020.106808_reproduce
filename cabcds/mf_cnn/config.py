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

    # --- Dataset locations (TUPAC16 layout defaults) ---
    dataset_dir: Path = Field(default=Path("dataset"))
    tupac_train_dir: Path = Field(default=Path("dataset/train"))
    tupac_test_dir: Path = Field(default=Path("dataset/test"))
    tupac_train_ground_truth_csv: Path = Field(default=Path("dataset/train/ground_truth.csv"))
    tupac_aux_roi_dir: Path = Field(default=Path("dataset/auxiliary_dataset_roi"))
    tupac_aux_mitoses_dir: Path = Field(default=Path("dataset/auxiliary_dataset_mitoses"))

    # External datasets (optional). These are used in the paper to improve
    # `CNN_seg` generalization, but they are not required to run a baseline.
    external_dataset_dir: Path = Field(default=Path("dataset/external"))
    mitos12_dir: Path = Field(default=Path("dataset/external/mitos12"))
    mitos14_dir: Path = Field(default=Path("dataset/external/mitos14"))
    use_external_mitosis_datasets: bool = Field(default=False)

    # Mitosis auxiliary dataset zips (as included in this repo's dataset folder)
    mitoses_image_zip_parts: tuple[str, ...] = Field(
        default=(
            "mitoses_image_data/mitoses_image_data_part_1.zip",
            "mitoses_image_data/mitoses_image_data_part_2.zip",
            "mitoses_image_data/mitoses_image_data_part_3.zip",
        )
    )
    mitoses_ground_truth_zip: str = Field(default="mitoses_ground_truth.zip")

    # --- Patch geometry / model IO ---
    seg_crop_size: int = Field(default=512, ge=32)
    det_crop_size: int = Field(default=80, ge=16)
    global_patch_size: int = Field(default=512, ge=32)
    patch_overlap: int = Field(default=80, ge=0)
    alexnet_input_size: int = Field(default=227, ge=32)

    # --- Mask generation (centroids -> pixel labels via BR+Otsu nuclei blobs) ---
    nuclei_min_area: int = Field(default=10, ge=1)

    # If a GT centroid does not fall inside any BR+Otsu blob, we still need a
    # positive region for supervised segmentation. We fall back to drawing a
    # small disk around the centroid.
    centroid_fallback_radius: int = Field(default=6, ge=0)

    # --- Sampling ---
    seed: int = Field(default=1337)
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
