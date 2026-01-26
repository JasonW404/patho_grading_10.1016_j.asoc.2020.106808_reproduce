"""Project entry point for CABCDS reproduction tasks."""

from __future__ import annotations

from cabcds.logging import setup_logger
from cabcds.roi_selector.config import load_roi_selector_config
from cabcds.roi_selector.inference import RoiSelector
from cabcds.roi_selector.training import RoiSelectorTrainer


def main() -> None:
    """Main entry point."""
    logger = setup_logger()
    logger.info("Starting CABCDS reproduction task.")

    # --- Stage 1: ROI Selection ---
    logger.info("--- Stage 1: ROI Selection ---")
    
    config = load_roi_selector_config()
    
    # Train model if needed
    if not config.train_model_output_path.exists():
        logger.info(f"ROI SVM model not found at {config.train_model_output_path}. Starting training...")
        trainer = RoiSelectorTrainer(config)
        trainer.train()
        logger.info("ROI SVM training completed.")
    else:
        logger.info(f"ROI SVM model found at {config.train_model_output_path}. Skipping training.")

    # Run inference
    logger.info("Initializing RoiSelector...")
    selector = RoiSelector(config)

    logger.info("Running ROI selection...")
    selector.select_rois()
    logger.info("Stage 1 pipeline completed successfully.")


if __name__ == "__main__":
	main()
