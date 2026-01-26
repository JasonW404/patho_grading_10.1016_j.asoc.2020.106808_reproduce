"""Patch sampling utilities for ROI selection."""

from __future__ import annotations

from .config import load_roi_selector_config
from .utils.create_roi_patches import create_positive_roi, create_negative_roi, generate_labelled_csv


def main():
    """Run the sampling pipeline for training data generation."""
    config = load_roi_selector_config()
    create_positive_roi(config)
    create_negative_roi(config)
    generate_labelled_csv(config)


if __name__ == "__main__":
    main()
