"""Training pipeline for ROI selector SVM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cabcds.data_preparation.io import list_image_files, load_rgb_image
from cabcds.roi_selector.config import RoiSelectorTrainingConfig
from cabcds.roi_selector.features import extract_patch_features, is_patch_too_white


@dataclass(frozen=True)
class RoiTrainingSample:
    """Training sample metadata.

    Attributes:
        image_path: Path to the image file.
        label: Integer label (1 for positive, 0 for negative).
        features: Feature vector.
    """

    image_path: Path
    label: int
    features: np.ndarray


class RoiSelectorTrainer:
    """Train a linear SVM for ROI selection."""

    def __init__(self, config: RoiSelectorTrainingConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self) -> Pipeline:
        """Train the ROI selector.

        Returns:
            Trained sklearn Pipeline.
        """

        samples = self._load_samples()
        if not samples:
            raise FileNotFoundError("No training samples found for ROI selector.")

        features = np.stack([sample.features for sample in samples])
        labels = np.array([sample.label for sample in samples], dtype=np.int32)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(C=self.config.svm_c, max_iter=5000)),
            ]
        )
        model.fit(features, labels)

        self._save_model(model)
        self.logger.info("ROI selector training completed with %d samples.", len(samples))
        return model

    def _load_samples(self) -> list[RoiTrainingSample]:
        """Load training samples from positive/negative directories.

        Returns:
            List of training samples.
        """

        dataset_dir = self.config.dataset_dir
        positives = self._collect_samples(dataset_dir / self.config.positive_subdir, label=1)
        negatives = self._collect_samples(dataset_dir / self.config.negative_subdir, label=0)
        return positives + negatives

    def _collect_samples(self, root: Path, label: int) -> list[RoiTrainingSample]:
        """Collect samples under a directory.

        Args:
            root: Directory containing images.
            label: Label for the samples.

        Returns:
            List of RoiTrainingSample objects.
        """

        extensions = {ext.lower() for ext in self.config.image_extensions}
        files = list_image_files(root, extensions) if root.exists() else []
        samples: list[RoiTrainingSample] = []

        for image_path in files:
            image = load_rgb_image(image_path)
            if is_patch_too_white(image, self.config.feature_config):
                continue
            features = extract_patch_features(image, self.config.feature_config)
            samples.append(RoiTrainingSample(image_path=image_path, label=label, features=features))
        return samples

    def _save_model(self, model: Pipeline) -> None:
        """Save the trained model to disk.

        Args:
            model: Trained sklearn pipeline.
        """

        output_path = self.config.model_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_config": self.config.feature_config.model_dump(),
            },
            output_path,
        )
        self.logger.info("Saved ROI selector model to %s", output_path)
