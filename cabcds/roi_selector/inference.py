"""ROI selection inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

import openslide
from PIL import Image

from .utils.io import compute_report_path, list_image_files, load_rgb_image, save_image_uint8
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

        report_path = compute_report_path(self.config.infer_output_dir, "stage_two_roi_selection.csv")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        processed: set[str] = set()
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    # Skip header
                    _ = f.readline()
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # CSV columns: image_path,rank,x,y,score
                        image_path_str = line.split(",", 1)[0]
                        if image_path_str:
                            processed.add(image_path_str)
            except Exception:
                processed = set()

        mode = "a" if report_path.exists() and report_path.stat().st_size > 0 else "w"

        results: list[RoiSelectionResult] = []
        # Write incrementally so partial progress is preserved if interrupted.
        with report_path.open(mode, encoding="utf-8") as file_handle:
            if mode == "w":
                file_handle.write("image_path,rank,x,y,score\n")
                file_handle.flush()

            for image_path in image_files:
                if processed and str(image_path) in processed:
                    continue

                result = self._select_for_image(image_path)
                results.append(result)

                for rank, candidate in enumerate(result.candidates, start=1):
                    file_handle.write(
                        f"{result.image_path},{rank},{candidate.x},{candidate.y},{candidate.score:.6f}\n"
                    )
                file_handle.flush()

        return results

    def _select_for_image(self, image_path: Path) -> RoiSelectionResult:
        """Select ROIs for a single image.

        Args:
            image_path: Path to the WSI image.

        Returns:
            ROI selection result.
        """

        if image_path.suffix.lower() == ".svs":
            selected = self._select_for_wsi(image_path)
        else:
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

    def _select_for_wsi(self, image_path: Path) -> tuple[RoiCandidate, ...]:
        """Select ROIs for a WSI via global low-magnification scanning.

        Paper-aligned behavior:
        - Scan the whole slide at ~10x (infer_scan_magnification)
        - Use a window corresponding to 5657x5657 at 40x (infer_roi_size_40x)
        - Keep top-N windows by SVM decision score

        Notes:
        - We compute features on a resized patch of size infer_patch_size to keep compute bounded.
        - Coordinates are returned in level-0 (base) pixel space.
        """

        top_n = int(self.config.infer_top_n)
        feature_size = int(self.config.infer_patch_size)
        scan_mag = float(self.config.infer_scan_magnification)
        roi_size_40x = int(self.config.infer_roi_size_40x)
        overlap = float(self.config.infer_overlap_ratio)

        with openslide.OpenSlide(str(image_path)) as slide:
            base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
            # ROI size at level-0 pixels (base magnification), scaled from the paper's 40x definition.
            roi_size_level0 = int(round(roi_size_40x * (base_mag / 40.0)))
            roi_size_level0 = max(256, roi_size_level0)

            # Choose a scan level that is NOT higher resolution than requested.
            # OpenSlide's get_best_level_for_downsample() picks the closest, which can be level 0
            # even when a lower-resolution level exists. For global scanning, we prefer
            # downsample >= target_downsample to reduce I/O.
            target_downsample = base_mag / max(1e-6, scan_mag)

            def _choose_scan_level(target: float) -> int:
                downsamples = [float(d) for d in slide.level_downsamples]
                # Smallest level whose downsample is >= target.
                for level, ds in enumerate(downsamples):
                    if ds >= target:
                        return level
                return len(downsamples) - 1

            read_level = _choose_scan_level(target_downsample)
            level_downsample = float(slide.level_downsamples[read_level])
            read_size = int(round(roi_size_level0 / level_downsample))
            read_size = max(64, read_size)

            step = int(round(read_size * (1.0 - overlap)))
            step = max(1, step)

            # Convert step at read_level into a step in level-0 coordinates.
            step0 = int(round(step * level_downsample))
            step0 = max(1, step0)

            width0, height0 = slide.dimensions
            x_end0 = width0 - roi_size_level0
            y_end0 = height0 - roi_size_level0
            if x_end0 <= 0 or y_end0 <= 0:
                return tuple()

            self.logger.info(
                "WSI scan %s: base_mag=%.1f scan_mag=%.1f level=%d downsample=%.3f roi40=%d roi0=%d read_size=%d step0=%d",
                image_path.name,
                base_mag,
                scan_mag,
                read_level,
                level_downsample,
                roi_size_40x,
                roi_size_level0,
                read_size,
                step0,
            )

            # Keep top-N using min-heaps: (score, tie_breaker, candidate)
            # - `heap`: candidates that pass the white/background filter (normal behavior)
            # - `white_heap`: fallback candidates from "too-white" patches (only used if `heap` is empty)
            heap: list[tuple[float, int, RoiCandidate]] = []
            white_heap: list[tuple[float, int, RoiCandidate]] = []
            tie_breaker = 0

            # Iterate in level-0 coordinates; convert to the chosen read_level implicitly via OpenSlide.
            total_rows = (y_end0 // step0) + 1
            row_index = 0
            for y0 in range(0, y_end0 + 1, step0):
                row_index += 1
                if row_index == 1 or row_index % 20 == 0:
                    self.logger.info(
                        "WSI %s progress: row %d/%d (y0=%d)",
                        image_path.name,
                        row_index,
                        total_rows,
                        y0,
                    )

                for x0 in range(0, x_end0 + 1, step0):
                    try:
                        region = slide.read_region((x0, y0), read_level, (read_size, read_size)).convert("RGB")
                    except Exception:
                        continue

                    patch = np.asarray(region, dtype=np.uint8)
                    too_white = is_patch_too_white(patch, self.feature_config)

                    if read_size != feature_size:
                        region_feat = region.resize((feature_size, feature_size), Image.Resampling.BILINEAR)
                        patch_feat = np.asarray(region_feat, dtype=np.uint8)
                    else:
                        patch_feat = patch

                    features = extract_patch_features(patch_feat, self.feature_config)
                    score = float(self.model.decision_function(features.reshape(1, -1))[0])
                    candidate = RoiCandidate(x=int(x0), y=int(y0), score=score, patch=patch_feat)

                    tie_breaker += 1
                    target_heap = white_heap if too_white else heap
                    if len(target_heap) < top_n:
                        heappush(target_heap, (score, tie_breaker, candidate))
                    else:
                        if score > target_heap[0][0]:
                            heappop(target_heap)
                            heappush(target_heap, (score, tie_breaker, candidate))

            selected = [c for _, _, c in sorted(heap, key=lambda t: t[0], reverse=True)]
            if not selected and white_heap:
                selected = [c for _, _, c in sorted(white_heap, key=lambda t: t[0], reverse=True)]
                self.logger.warning(
                    "WSI %s selected 0 ROIs after white-filter; falling back to top-%d white patches",
                    image_path.name,
                    len(selected),
                )

            if selected:
                self.logger.info(
                    "WSI %s top-%d scores: %s",
                    image_path.name,
                    len(selected),
                    ", ".join(f"{c.score:.3f}@({c.x},{c.y})" for c in selected),
                )

            # Optionally also save the full ROI crop at level 0.
            # (Preview patches are saved by _select_for_image to avoid double-writing.)
            if bool(self.config.infer_save_patches) and bool(self.config.infer_save_full_roi_40x) and selected:
                full_root = self.config.infer_output_dir / "patches_full_40x"
                if (full_root / "train").exists() or (full_root / "test").exists():
                    split_name = "test" if self.config.infer_wsi_dir.name == "test" else "train"
                    out_dir = full_root / split_name / image_path.stem
                else:
                    out_dir = full_root / image_path.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                for index, cand in enumerate(selected, start=1):
                    try:
                        full = slide.read_region((cand.x, cand.y), 0, (roi_size_level0, roi_size_level0)).convert("RGB")
                        full.save(out_dir / f"roi_{index:02d}_x{cand.x}_y{cand.y}.jpg", quality=90)
                    except Exception:
                        continue

            return tuple(selected)

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

        patches_root = self.config.infer_output_dir / "patches"
        # If the user has split the patches directory into train/test, keep writing there.
        # This preserves backward compatibility with the original flat layout.
        if (patches_root / "train").exists() or (patches_root / "test").exists():
            split_name = "test" if self.config.infer_wsi_dir.name == "test" else "train"
            output_dir = patches_root / split_name / image_path.stem
        else:
            output_dir = patches_root / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, candidate in enumerate(candidates, start=1):
            filename = f"roi_{index:02d}_x{candidate.x}_y{candidate.y}.png"
            save_image_uint8(candidate.patch, output_dir / filename)

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

