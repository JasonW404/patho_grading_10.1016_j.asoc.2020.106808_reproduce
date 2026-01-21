"""Training and inference utilities for WSI scoring."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cabcds.wsi_scorer.config import WsiScorerConfig


@dataclass(frozen=True)
class WsiSample:
    """WSI sample metadata.

    Attributes:
        group: Group identifier (WSI name).
        features: Feature vector.
        label: Integer label.
    """

    group: str
    features: np.ndarray
    label: int


class WsiScorerTrainer:
    """Train a multi-class SVM for WSI scoring."""

    def __init__(self, config: WsiScorerConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self) -> Pipeline:
        """Train the SVM with cross-validation.

        Returns:
            Trained sklearn Pipeline.
        """

        samples = _load_samples(self.config.descriptor_csv, self.config.labels_csv)
        if not samples:
            raise FileNotFoundError("No WSI samples available for training.")

        features = np.stack([sample.features for sample in samples])
        labels = np.array([sample.label for sample in samples], dtype=np.int32)

        model = _build_model(self.config)
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, features, labels, cv=cv)

        self.logger.info("WSI scorer CV accuracy: %.4f ± %.4f", scores.mean(), scores.std())
        self._write_report(scores)

        model.fit(features, labels)
        self._save_model(model)
        return model

    def _write_report(self, scores: np.ndarray) -> None:
        """Write a cross-validation report.

        Args:
            scores: CV accuracy scores.
        """

        report_dir = self.config.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "stage_five_wsi_scorer_cv.csv"

        with report_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write("fold,accuracy\n")
            for idx, score in enumerate(scores, start=1):
                file_handle.write(f"{idx},{score:.6f}\n")

    def _save_model(self, model: Pipeline) -> None:
        """Save trained model to disk.

        Args:
            model: Trained sklearn pipeline.
        """

        output_path = self.config.model_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "config": self.config.model_dump(),
            },
            output_path,
        )
        self.logger.info("Saved WSI scorer model to %s", output_path)


class WsiScorerPredictor:
    """Predict WSI scores from hybrid descriptors."""

    def __init__(self, config: WsiScorerConfig, model: Pipeline | None = None) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model or self._load_model()

    def predict(self, descriptor_csv: Path | None = None) -> dict[str, int]:
        """Predict WSI scores for the provided descriptor file.

        Args:
            descriptor_csv: Optional descriptor CSV path.

        Returns:
            Mapping from group name to predicted score.
        """

        descriptor_path = descriptor_csv or self.config.descriptor_csv
        groups, features = _load_feature_table(descriptor_path)
        predictions = self.model.predict(features)

        results = {group: int(pred) for group, pred in zip(groups, predictions, strict=False)}
        self._write_predictions(results)
        return results

    def _write_predictions(self, results: dict[str, int]) -> None:
        """Write predictions to CSV.

        Args:
            results: Mapping of group name to predicted score.
        """

        output_dir = self.config.report_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stage_five_wsi_predictions.csv"

        with output_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write("group,predicted_score\n")
            for group, score in results.items():
                file_handle.write(f"{group},{score}\n")

    def _load_model(self) -> Pipeline:
        """Load model from disk.

        Returns:
            Trained sklearn Pipeline.
        """

        payload = joblib.load(self.config.model_output_path)
        return payload["model"]


def _build_model(config: WsiScorerConfig) -> Pipeline:
    """Build the SVM model pipeline.

    Args:
        config: WSI scorer configuration.

    Returns:
        sklearn Pipeline.
    """

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=config.svm_c,
                    kernel="linear",
                    decision_function_shape=config.decision_function_shape,
                ),
            ),
        ]
    )


def _load_samples(descriptor_csv: Path, labels_csv: Path) -> list[WsiSample]:
    """Load WSI samples from descriptor and label tables.

    Args:
        descriptor_csv: Path to descriptors.
        labels_csv: Path to group labels.

    Returns:
        List of WsiSample entries.
    """

    groups, features = _load_feature_table(descriptor_csv)
    label_map = _load_label_map(labels_csv)

    samples: list[WsiSample] = []
    for group, feature in zip(groups, features, strict=False):
        if group not in label_map:
            raise ValueError(f"Missing label for group: {group}")
        samples.append(WsiSample(group=group, features=feature, label=label_map[group]))
    return samples


def _load_feature_table(descriptor_csv: Path) -> tuple[list[str], np.ndarray]:
    """Load hybrid descriptor CSV.

    Args:
        descriptor_csv: Path to descriptor CSV.

    Returns:
        Tuple of (groups, feature matrix).
    """

    if not descriptor_csv.exists():
        raise FileNotFoundError(f"Descriptor CSV not found: {descriptor_csv}")

    with descriptor_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        groups: list[str] = []
        features: list[np.ndarray] = []
        for row in reader:
            group = row.get("group")
            if group is None:
                raise ValueError("Descriptor CSV must include 'group' column.")
            vector = _parse_feature_row(row)
            groups.append(group)
            features.append(vector)

    return groups, np.stack(features)


def _parse_feature_row(row: dict[str, str]) -> np.ndarray:
    """Parse a descriptor CSV row into a feature vector.

    Args:
        row: CSV row mapping.

    Returns:
        Feature vector.
    """

    values: list[float] = []
    for index in range(1, 16):
        key = f"feature_{index}"
        if key not in row:
            raise ValueError(f"Missing {key} in descriptor row.")
        values.append(float(row[key]))
    return np.array(values, dtype=np.float32)


def _load_label_map(labels_csv: Path) -> dict[str, int]:
    """Load group-to-label mapping.

    Args:
        labels_csv: Path to labels CSV.

    Returns:
        Mapping of group to integer label.
    """

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    label_map: dict[str, int] = {}
    with labels_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            group = row.get("group")
            label = row.get("label")
            if group is None or label is None:
                raise ValueError("Labels CSV must include 'group' and 'label' columns.")
            label_map[group] = int(label)
    return label_map
