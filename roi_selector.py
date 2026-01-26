#!/usr/bin/env python3
"""
Main entry point for ROI Selector pipeline.
Supports data preparation and training via command line arguments.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the current directory is in the python path
sys.path.append(str(Path(__file__).parent))

from cabcds.roi_selector.config import load_roi_selector_config
from cabcds.roi_selector.utils.create_roi_patches import (
    create_positive_roi,
    create_negative_roi,
    generate_labelled_csv,
)
from cabcds.roi_selector.training import RoiSelectorTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI Selector Pipeline Management")
    parser.add_argument(
        "--prepare", 
        action="store_true", 
        help="Run data preparation: generate positive and negative patches from WSIs."
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Run SVM training using generated patches."
    )
    
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("roi_selector_main")

    # Load configuration
    try:
        config = load_roi_selector_config()
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 1. Data Preparation
    if args.prepare:
        logger.info("=== Starting Data Preparation Phase ===")
        try:
            logger.info("Generating Positive ROIs...")
            create_positive_roi(config)
            
            logger.info("Generating Negative ROIs...")
            create_negative_roi(config)
            
            logger.info("Generating Labelled CSV Index...")
            csv_path = generate_labelled_csv(config)
            logger.info(f"Data preparation completed. Index file: {csv_path}")
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            sys.exit(1)

    # 2. Training
    if args.train:
        logger.info("=== Starting Training Phase ===")
        try:
            trainer = RoiSelectorTrainer(config)
            trainer.train()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

    if not args.prepare and not args.train:
        logger.warning("No action specified. Use --prepare or --train.")
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
