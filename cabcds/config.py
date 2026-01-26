"""Global configuration for the CABCDS reproduction project."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Global configuration settings.

    Attributes:
        debug: Enable debug mode. When True, processing is limited to a small number of images (default 5).
        debug_max_images: Number of images to process in debug mode.
    """

    debug: bool = Field(default=False)
    debug_max_images: int = Field(default=5, ge=1)

    dataset_dir: Path = Field(default=Path("dataset"))

    artifact_dir: Path = Field(default=Path("artifact"))
    temp_dir: Path = Field(default=Path("temp"))
    output_dir: Path = Field(default=Path("output"))

    class Config:
        env_prefix = "CABCDS_"
        extra = "ignore"

config = Config()