"""Configuration models for WSI scoring stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class WsiScorerConfig(BaseSettings):
    """Configuration for WSI scoring.

    Attributes:
        descriptor_csv: Path to the hybrid descriptor CSV.
        labels_csv: Path to CSV with group-to-label mapping.
        model_output_path: Output path for the trained SVM model.
        report_dir: Directory to store training reports.
        svm_c: Regularization strength for SVM.
        cv_folds: Number of cross-validation folds.
        decision_function_shape: SVM decision function shape.
    """

    class Config:
        env_prefix = "CABCDS_WSI_"
        env_nested_delimiter = "__"
        frozen = True

    descriptor_csv: Path = Field(default=Path("data/hybrid_descriptor/stage_four_hybrid_descriptors.csv"))
    labels_csv: Path = Field(default=Path("data/wsi_scorer/labels.csv"))
    model_output_path: Path = Field(default=Path("data/wsi_scorer/models/wsi_svm.joblib"))
    report_dir: Path = Field(default=Path("data/wsi_scorer/reports"))
    svm_c: float = Field(default=0.1, gt=0.0)
    svm_kernel: str = Field(default="rbf")
    cv_folds: int = Field(default=10, ge=2)
    decision_function_shape: str = Field(default="ovo")


def load_wsi_scorer_config() -> WsiScorerConfig:
    """Load WSI scorer configuration.

    Returns:
        WsiScorerConfig instance.
    """

    return WsiScorerConfig()
