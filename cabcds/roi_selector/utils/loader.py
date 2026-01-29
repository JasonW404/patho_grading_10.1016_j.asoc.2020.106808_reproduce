"""ROI Selector Data Loader.

Note: This module depends on torch and is only needed for SVM training.
"""

from __future__ import annotations

from functools import partial
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import ROISelectorConfig, RoiSelectorFeatureConfig
from .create_roi_patches import create_positive_roi, create_negative_roi, generate_labelled_csv
from .features import extract_patch_features
# from cabcds.roi_selector.data_loader.io import load_rgb_image
# Actually we need an image loader function
from PIL import Image

def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

logger = logging.getLogger(__name__)


def _feature_dim(feature_config: RoiSelectorFeatureConfig) -> int:
    """Compute the fixed feature dimensionality for ROI selector."""

    color_dim = int(feature_config.color_bins) * 4
    if feature_config.lbp_method == "uniform":
        lbp_bins = int(feature_config.lbp_neighbors) + 2
    else:
        lbp_bins = 2 ** int(feature_config.lbp_neighbors)
    return int(color_dim + lbp_bins + 1)


def _collate_skip_invalid(batch: list[tuple[torch.Tensor, torch.Tensor]], *, feature_dim: int):
    """Collate function that drops invalid samples.

    Invalid means label < 0 or feature shape mismatch.
    Returns empty tensors if the entire batch is invalid.
    """

    valid_features: list[torch.Tensor] = []
    valid_labels: list[torch.Tensor] = []

    for features, label in batch:
        try:
            if int(label.item()) < 0:
                continue
        except Exception:
            continue
        if int(features.numel()) != int(feature_dim):
            continue
        valid_features.append(features)
        valid_labels.append(label)

    if not valid_features:
        return (
            torch.empty((0, int(feature_dim)), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    return torch.stack(valid_features, dim=0), torch.stack(valid_labels, dim=0).view(-1)

class ROISelectorDataset(Dataset):
    """PyTorch Dataset for ROI Selection."""
    
    def __init__(
        self,
        csv_path: Path,
        feature_config: RoiSelectorFeatureConfig,
        *,
        max_positive: int | None = None,
        max_negative: int | None = None,
        exclude_paths: set[str] | None = None,
        exclude_prefixes: Iterable[str] | None = None,
    ):
        self.feature_config = feature_config
        self._feature_dim = _feature_dim(feature_config)
        self.samples = []
        
        df = pd.read_csv(csv_path).astype({'path': str, 'label': int})

        if exclude_prefixes:
            prefixes = [p for p in exclude_prefixes if p]
            if prefixes:
                mask = np.ones(len(df), dtype=bool)
                paths = df["path"].astype(str).tolist()
                for i, p in enumerate(paths):
                    for prefix in prefixes:
                        if p.startswith(prefix):
                            mask[i] = False
                            break
                df = df[mask].reset_index(drop=True)

        if exclude_paths:
            df = df[~df["path"].isin(exclude_paths)].reset_index(drop=True)
        
        if max_positive is not None or max_negative is not None:
            positive_df = df[df['label'] == 1]
            negative_df = df[df['label'] == 0]
            
            if max_positive is not None:
                positive_df = positive_df.head(max_positive)
            if max_negative is not None:
                negative_df = negative_df.head(max_negative)
            
            df = pd.concat([positive_df, negative_df], ignore_index=True)
        
        self.samples = list(df.itertuples(index=False, name=None))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path_str, label = self.samples[idx]
        image_path = Path(path_str)
        try:
            image = load_rgb_image(image_path)
            features = extract_patch_features(image, self.feature_config)
            return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading sample {path_str}: {e}")
            # Return fixed-size dummy feature vector with invalid label; collate_fn will drop it.
            return torch.zeros(int(self._feature_dim), dtype=torch.float32), torch.tensor(-1, dtype=torch.long)

class ROISelectorDataLoader:
    """Wrapper to prepare data and provide PyTorch DataLoader."""
    
    def __init__(self, config: ROISelectorConfig):
        self.config = config
    
    def generate_dataset(self):
        """Invoke the sampling pipeline to generate patches."""
        # Check if CSV exists, if so skip?
        csv_path = self.config.train_dataset_dir / "ROI_labelled.csv"
        if csv_path.exists():
            logger.info("Labelled CSV exists, skipping generation.")
            return

        logger.info("Generating ROI dataset...")
        create_positive_roi(self.config)
        create_negative_roi(self.config)
        generate_labelled_csv(self.config)

    def _load_benchmark_exclusions(self) -> set[str]:
        if not bool(self.config.benchmark_exclude_from_training):
            return set()
        index_csv = self.config.benchmark_dir / "benchmark_index.csv"
        if not index_csv.exists():
            return set()
        try:
            df = pd.read_csv(index_csv)
        except Exception as e:
            logger.warning("Failed to read benchmark index %s: %s", index_csv, e)
            return set()
        if "source_path" not in df.columns:
            return set()
        return {str(Path(p).expanduser().resolve()) for p in df["source_path"].dropna().astype(str).tolist()}

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, max_positive: int | None = None, max_negative: int | None = None) -> DataLoader:
        csv_path = self.config.train_dataset_dir / "ROI_labelled.csv"
        if not csv_path.exists():
            self.generate_dataset()

        exclude_paths = self._load_benchmark_exclusions()

        # Never use legacy negatives for SVM training (reserved for early CNN/QA).
        legacy_negative_prefix = str((self.config.train_dataset_dir / self.config.train_negative_subdir).resolve()) + "/"
        dataset = ROISelectorDataset(
            csv_path,
            self.config.feature_config,
            max_positive=max_positive,
            max_negative=max_negative,
            exclude_paths=exclude_paths,
            exclude_prefixes=[legacy_negative_prefix],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(_collate_skip_invalid, feature_dim=_feature_dim(self.config.feature_config)),
        )


