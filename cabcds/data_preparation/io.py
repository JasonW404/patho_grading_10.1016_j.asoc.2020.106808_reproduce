"""I/O helpers for stage one data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def list_image_files(dataset_dir: Path, extensions: set[str]) -> list[Path]:
    """List image files under a dataset directory.

    Args:
        dataset_dir: Root directory containing images.
        extensions: Lowercase extensions to include.

    Returns:
        Sorted list of image file paths.
    """

    image_files: list[Path] = []
    for file_path in dataset_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)
    return sorted(image_files)


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a uint8 RGB array with shape (H, W, 3).
    """

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    return np.array(rgb_image, dtype=np.uint8)


def ensure_parent_dir(path: Path) -> None:
    """Ensure that the parent directory exists.

    Args:
        path: Output file path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def save_image_uint8(image: np.ndarray, output_path: Path) -> None:
    """Save a uint8 image array to disk.

    Args:
        image: Image array with dtype uint8.
        output_path: Output file path.
    """

    ensure_parent_dir(output_path)
    Image.fromarray(image).save(output_path)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary mask as a uint8 image.

    Args:
        mask: Boolean or binary mask.
        output_path: Output file path.
    """

    mask_uint8 = (mask.astype(np.uint8) * 255)
    save_image_uint8(mask_uint8, output_path)


def compute_output_path(
    input_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    suffix: str,
) -> Path:
    """Compute a mirrored output path for a processed image.

    Args:
        input_path: Original image path.
        dataset_dir: Dataset root directory.
        output_dir: Output root directory.
        suffix: Suffix to append to the filename stem.

    Returns:
        Output file path with the suffix inserted before extension.
    """

    relative_path = input_path.relative_to(dataset_dir)
    new_name = f"{relative_path.stem}_{suffix}{relative_path.suffix}"
    return output_dir / relative_path.with_name(new_name)


def compute_report_path(output_dir: Path, name: str) -> Path:
    """Compute a report path under the output directory.

    Args:
        output_dir: Output root directory.
        name: Base filename for the report.

    Returns:
        Path to the report file.
    """

    return output_dir / "reports" / name


def iter_parent_dirs(paths: Iterable[Path]) -> set[Path]:
    """Return parent directories for a set of paths.

    Args:
        paths: Iterable of file paths.

    Returns:
        Set of unique parent directories.
    """

    return {path.parent for path in paths}
