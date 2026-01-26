"""ROI Selector Data Loader."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
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

class ROISelectorDataset(Dataset):
    """PyTorch Dataset for ROI Selection."""
    
    def __init__(self, csv_path: Path, feature_config: RoiSelectorFeatureConfig):
        self.feature_config = feature_config
        self.samples = []
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header if exists 'path, label'
            if header and header[0] != 'path':
                # No header? reset
                f.seek(0)
            
            for line in reader:
                if len(line) >= 2:
                    self.samples.append((line[0], int(line[1])))
    
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
            # Return zeros or handle error? PyTorch dataloader doesn't like None usually.
            # Return dummy
            return torch.zeros(1), torch.tensor(-1)

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

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        csv_path = self.config.train_dataset_dir / "ROI_labelled.csv"
        if not csv_path.exists():
            self.generate_dataset()
            
        dataset = ROISelectorDataset(csv_path, self.config.feature_config)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


