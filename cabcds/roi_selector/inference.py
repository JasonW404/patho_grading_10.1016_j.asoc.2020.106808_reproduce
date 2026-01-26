"""ROI selection inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from ..data_loader.io import compute_report_path, list_image_files, load_rgb_image, save_image_uint8
from .config import (
    RoiSelectorFeatureConfig,
    ROISelectorConfig, 
    load_roi_selector_config,
)
from .utils.features import extract_patch_features, is_patch_too_white
from .sampling import PatchSample, SlidingWindowSampler


@dataclass(frozen=True)
class RoiCandidate:
    """Scored ROI candidate.

    Attributes:
        x: Top-left x-coordinate.
        y: Top-left y-coordinate.
        score: SVM decision score.
        patch: RGB patch array.
    """

    x: int
    y: int
    score: float
    patch: np.ndarray


@dataclass(frozen=True)
class RoiSelectionResult:
    """ROI selection result for a WSI image.

    Attributes:
        image_path: Path to the WSI image.
        candidates: Selected ROI candidates.
    """

    image_path: Path
    candidates: tuple[RoiCandidate, ...]


class RoiSelector:
    """Select ROIs from WSI images using a trained SVM."""

    def __init__(self, config: ROISelectorConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"cabcds.{self.__class__.__name__}")
        self.model, self.feature_config = self._load_model(config.infer_model_path)
        self.sampler = SlidingWindowSampler(config)

    def select_rois(self) -> list[RoiSelectionResult]:
        """Run ROI selection over all WSI images.

        Returns:
            List of ROI selection results.
        """

        image_files = list_image_files(self.config.infer_wsi_dir, {ext.lower() for ext in self.config.infer_image_extensions})
        if self.config.infer_max_images is not None:
            self.logger.info(f"Limiting to {self.config.infer_max_images} images.")
            image_files = image_files[: self.config.infer_max_images]

        if not image_files:
            raise FileNotFoundError("No WSI images found for ROI selection.")

        results: list[RoiSelectionResult] = []
        for image_path in image_files:
            result = self._select_for_image(image_path)
            results.append(result)

        self._write_report(results)
        return results

    def _select_for_image(self, image_path: Path) -> RoiSelectionResult:
        """Select ROIs for a single image.

        Args:
            image_path: Path to the WSI image.

        Returns:
            ROI selection result.
        """

        image = load_rgb_image(image_path)
        candidates = list(self._score_candidates(image))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = tuple(candidates[: self.config.infer_top_n])

        if self.config.infer_save_patches:
            self._save_patches(image_path, selected)

        self.logger.info(
            "Selected %d ROIs for %s",
            len(selected),
            image_path.name,
        )
        return RoiSelectionResult(image_path=image_path, candidates=selected)

    def _score_candidates(self, image: np.ndarray) -> Iterable[RoiCandidate]:
        """Score candidate patches for a single image.

        Args:
            image: RGB image array.

        Yields:
            RoiCandidate entries.
        """

        for sample in self.sampler.iter_patches(image):
            if is_patch_too_white(sample.patch, self.feature_config):
                continue
            features = extract_patch_features(sample.patch, self.feature_config)
            score = float(self.model.decision_function(features.reshape(1, -1))[0])
            yield RoiCandidate(x=sample.x, y=sample.y, score=score, patch=sample.patch)

    def _save_patches(self, image_path: Path, candidates: tuple[RoiCandidate, ...]) -> None:
        """Save selected ROI patches.

        Args:
            image_path: Input image path.
            candidates: Selected ROI candidates.
        """

        output_dir = self.config.infer_output_dir / "patches" / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, candidate in enumerate(candidates, start=1):
            filename = f"roi_{index:02d}_x{candidate.x}_y{candidate.y}.png"
            save_image_uint8(candidate.patch, output_dir / filename)

    def _write_report(self, results: list[RoiSelectionResult]) -> None:
        """Write a CSV report for ROI selection.

        Args:
            results: ROI selection results.
        """

        report_path = compute_report_path(self.config.infer_output_dir, "stage_two_roi_selection.csv")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        header = "image_path,rank,x,y,score\n"
        with report_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(header)
            for result in results:
                for rank, candidate in enumerate(result.candidates, start=1):
                    file_handle.write(
                        f"{result.image_path},{rank},{candidate.x},{candidate.y},{candidate.score:.6f}\n"
                    )

    def _load_model(self, model_path: Path) -> tuple[object, RoiSelectorFeatureConfig]:
        """Load the trained ROI selector model.

        Args:
            model_path: Path to the model file.

        Returns:
            Tuple of (model, feature_config).
        """

        payload = joblib.load(model_path)
        model = payload["model"]
        feature_config = RoiSelectorFeatureConfig.model_validate(payload.get("feature_config", {}))
        return model, feature_config


def main() -> None:
    """Run ROI inference standalone."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = load_roi_selector_config()
    selector = RoiSelector(config)
    selector.select_rois()


if __name__ == "__main__":
    main()

