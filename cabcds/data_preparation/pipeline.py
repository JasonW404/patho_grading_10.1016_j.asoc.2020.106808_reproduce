"""Stage one preprocessing pipeline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cabcds.data_preparation.blue_ratio import BlobStats, compute_blob_stats, compute_blue_ratio, otsu_mask
from cabcds.data_preparation.config import StageOneConfig
from cabcds.data_preparation.io import (
    compute_output_path,
    compute_report_path,
    list_image_files,
    load_rgb_image,
    save_image_uint8,
    save_mask,
)
from cabcds.data_preparation.macenko import MacenkoNormalizer


@dataclass
class StageOneResult:
    """Result record for a processed image.

    Attributes:
        image_path: Input image path.
        normalized_path: Output path for the normalized image.
        blue_ratio_path: Output path for the blue-ratio visualization.
        mask_path: Output path for the blob mask.
        blob_stats: Blob statistics for the image.
    """

    image_path: Path
    normalized_path: Path
    blue_ratio_path: Path
    mask_path: Path
    blob_stats: BlobStats


class StageOnePreprocessor:
    """Run stage one data preparation steps.

    Args:
        config: Stage one configuration object.
    """

    def __init__(self, config: StageOneConfig) -> None:
        self.config = config
        self.normalizer = MacenkoNormalizer()

    def run(self) -> list[StageOneResult]:
        """Execute the preprocessing pipeline.

        Returns:
            List of results for each processed image.
        """

        image_files = list_image_files(self.config.dataset_dir, self.config.as_extension_set())
        if self.config.max_images is not None:
            image_files = image_files[: self.config.max_images]

        if not image_files:
            raise FileNotFoundError(
                "No image files found. Check dataset path and extensions: "
                f"{', '.join(self.config.image_extensions_iterable())}"
            )

        reference_image = self._load_reference_image(image_files)
        self.normalizer.fit(reference_image)

        results: list[StageOneResult] = []
        for image_path in tqdm(image_files, desc="Stage 1 preprocessing"):
            results.append(self._process_single(image_path))

        self._write_report(results)
        return results

    def _load_reference_image(self, image_files: list[Path]) -> np.ndarray:
        """Load the reference image for normalization.

        Args:
            image_files: List of available image files.

        Returns:
            Reference image array.
        """

        if self.config.reference_image_path is not None:
            return load_rgb_image(self.config.reference_image_path)
        return load_rgb_image(image_files[0])

    def _process_single(self, image_path: Path) -> StageOneResult:
        """Process a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            StageOneResult for the image.
        """

        normalized_path = compute_output_path(
            image_path,
            self.config.dataset_dir,
            self.config.output_dir / "normalized",
            "normalized",
        )
        blue_ratio_path = compute_output_path(
            image_path,
            self.config.dataset_dir,
            self.config.output_dir / "blue_ratio",
            "blue_ratio",
        )
        mask_path = compute_output_path(
            image_path,
            self.config.dataset_dir,
            self.config.output_dir / "masks",
            "mask",
        )

        if not self.config.overwrite and normalized_path.exists() and mask_path.exists():
            return StageOneResult(
                image_path=image_path,
                normalized_path=normalized_path,
                blue_ratio_path=blue_ratio_path,
                mask_path=mask_path,
                blob_stats=BlobStats(blob_count=0, average_area=0.0),
            )

        image = load_rgb_image(image_path)
        normalized = self.normalizer.transform(image)

        blue_ratio = compute_blue_ratio(normalized)
        mask = otsu_mask(blue_ratio)
        blob_stats = compute_blob_stats(mask, self.config.min_blob_area)

        save_image_uint8(normalized, normalized_path)
        save_image_uint8(_normalize_to_uint8(blue_ratio), blue_ratio_path)
        save_mask(mask, mask_path)

        return StageOneResult(
            image_path=image_path,
            normalized_path=normalized_path,
            blue_ratio_path=blue_ratio_path,
            mask_path=mask_path,
            blob_stats=blob_stats,
        )

    def _write_report(self, results: list[StageOneResult]) -> None:
        """Write a CSV report for processed images.

        Args:
            results: List of stage one results.
        """

        report_path = compute_report_path(self.config.output_dir, "stage_one_summary.csv")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        header = "image_path,normalized_path,blue_ratio_path,mask_path,blob_count,average_blob_area\n"
        with report_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(header)
            for result in results:
                file_handle.write(
                    f"{result.image_path},"
                    f"{result.normalized_path},"
                    f"{result.blue_ratio_path},"
                    f"{result.mask_path},"
                    f"{result.blob_stats.blob_count},"
                    f"{result.blob_stats.average_area:.4f}\n"
                )


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize a float image to uint8.

    Args:
        image: Input float image.

    Returns:
        Uint8 image scaled to 0-255.
    """

    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) / (max_val - min_val)
    return (scaled * 255.0).astype(np.uint8)
