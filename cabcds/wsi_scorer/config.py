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

    config = WsiScorerConfig()
    repo_root = _find_repo_root(Path.cwd())

    overrides: dict[str, Path] = {}

    if not config.descriptor_csv.exists():
        detected_descriptor = _auto_detect_descriptor_csv(repo_root)
        if detected_descriptor is not None:
            overrides["descriptor_csv"] = detected_descriptor

    if not config.labels_csv.exists():
        detected_labels = _auto_detect_labels_csv(repo_root)
        if detected_labels is not None:
            overrides["labels_csv"] = detected_labels

    if not overrides:
        return config

    payload = config.model_dump()
    payload.update(overrides)
    return WsiScorerConfig(**payload)


def _find_repo_root(start: Path) -> Path:
    """Find repository root by walking up until `pyproject.toml` is found.

    Args:
        start: Starting directory.

    Returns:
        Path to the repo root when found, otherwise the resolved start directory.
    """

    current = start.resolve()
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start.resolve()


def _auto_detect_descriptor_csv(repo_root: Path) -> Path | None:
    """Detect a usable hybrid descriptor CSV within the repo.

    Preference order:
    1) `output/features/**/stage_four_hybrid_descriptors.csv`
    2) `output/features/**/all_train_stage_four.csv`

    Args:
        repo_root: Repository root directory.

    Returns:
        Path to the most recently modified candidate, or None if none exist.
    """

    candidates: list[Path] = []
    features_dir = repo_root / "output" / "features"
    if features_dir.exists():
        candidates.extend(features_dir.rglob("stage_four_hybrid_descriptors.csv"))
        candidates.extend(features_dir.rglob("all_train_stage_four.csv"))
    return _pick_most_recent_file(candidates)


def _auto_detect_labels_csv(repo_root: Path) -> Path | None:
    """Detect a labels CSV with `group` and `label` columns.

    Args:
        repo_root: Repository root directory.

    Returns:
        Path to a labels CSV, or None if none exist.
    """

    tupac_labels = repo_root / "dataset" / "tupac16" / "train" / "ground_truth_with_groups.csv"
    if tupac_labels.exists():
        return tupac_labels
    return None


def _pick_most_recent_file(candidates: list[Path]) -> Path | None:
    """Pick the most recently modified file among candidates.

    Args:
        candidates: Candidate file paths.

    Returns:
        Most recently modified path, or None when no candidates exist.
    """

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)
