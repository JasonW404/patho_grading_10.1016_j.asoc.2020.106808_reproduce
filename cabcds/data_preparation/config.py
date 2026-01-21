"""Configuration models for stage one data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StageOneConfig(BaseSettings):
    """Configuration for the stage one preprocessing pipeline.

    Attributes:
        dataset_dir: Directory containing raw WSI-derived images.
        output_dir: Directory to store normalized images and masks.
        reference_image_path: Optional path to a reference image for stain normalization.
        image_extensions: Allowed image extensions to scan.
        min_blob_area: Minimum blob area (in pixels) to keep after Otsu thresholding.
        overwrite: Whether to overwrite existing outputs.
        max_images: Optional cap on the number of images to process (for quick tests).
    """

    class Config:
        env_prefix = "CABCDS_STAGE1_"
        env_nested_delimiter = "__"
        frozen = True

    dataset_dir: Path = Field(default=Path("dataset"))
    output_dir: Path = Field(default=Path("data/preprocessed"))
    reference_image_path: Path | None = Field(default=None)
    image_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    min_blob_area: int = Field(default=10, ge=1)
    overwrite: bool = Field(default=False)
    max_images: int | None = Field(default=None, ge=1)

    def as_extension_set(self) -> set[str]:
        """Return lowercase extension set for filtering files.

        Returns:
            Lowercased extension set.
        """

        return {ext.lower() for ext in self.image_extensions}

    def image_extensions_iterable(self) -> Iterable[str]:
        """Return extensions as an iterable for reporting.

        Returns:
            Iterable of extension strings.
        """

        return tuple(self.image_extensions)

    @field_validator("image_extensions", mode="before")
    @classmethod
    def _coerce_extensions(cls, value: object) -> tuple[str, ...]:
        """Ensure extensions are stored as a tuple.

        Args:
            value: Raw extensions input.

        Returns:
            Tuple of extensions.
        """

        if value is None:
            return (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (str(value),)


def load_stage_one_config() -> StageOneConfig:
    """Load stage one configuration from defaults and environment variables.

    Returns:
        StageOneConfig instance.
    """

    return StageOneConfig()
