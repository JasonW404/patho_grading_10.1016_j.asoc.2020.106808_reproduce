"""Training pipeline for ROI selector SVM."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from .utils.loader import ROISelectorDataLoader
from .config import ROISelectorConfig, load_roi_selector_config


class RoiSelectorTrainer:
    """Train a linear SVM for ROI selection."""

    def __init__(self, config: ROISelectorConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"cabcds.{self.__class__.__name__}")

    def train(self) -> Pipeline:
        """Train the ROI selector.

        Returns:
            Trained sklearn Pipeline.
        """
        self.logger.info("Initializing Data Loader...")
        loader = ROISelectorDataLoader(self.config)
        
        # Get DataLoader (will generate data if needed)
        dl = loader.get_dataloader(batch_size=32, shuffle=True, num_workers=2, max_positive=148, max_negative=544)

        self.logger.info("Loading samples from DataLoader...")
        all_features = []
        all_labels = []

        # Iterate to collect all data for SVM
        for features, labels in tqdm(dl, desc="Loading Batches"):
            if features.ndim > 1: # Handle batch
                all_features.append(features.numpy())
                all_labels.append(labels.numpy())
        
        if not all_features:
            raise ValueError("No training data found or loaded.")

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"Loaded {len(X)} samples. Positive: {np.sum(y==1)}, Negative: {np.sum(y==0)}")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(C=self.config.train_svm_c, max_iter=5000)),
            ]
        )
        
        # 10-fold Cross Validation as per paper
        self.logger.info("Running 10-fold Cross-Validation...")
        try:
            scores = cross_val_score(model, X, y, cv=10)
            self.logger.info(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        except Exception as e:
            self.logger.warning(f"CV failed (possibly too few samples): {e}")

        self.logger.info("Training final model on all data...")
        model.fit(X, y)

        self._save_model(model)
        self.logger.info("ROI selector training completed.")
        return model

    def _save_model(self, model: Pipeline) -> None:
        """Save the trained model to disk.

        Args:
            model: Trained sklearn pipeline.
        """

        output_path = self.config.train_model_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_config": self.config.feature_config.model_dump(),
            },
            output_path,
        )
        self.logger.info("Saved ROI selector model to %s", output_path)


def main() -> None:
    """Run ROI training standalone."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = load_roi_selector_config()
    trainer = RoiSelectorTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

