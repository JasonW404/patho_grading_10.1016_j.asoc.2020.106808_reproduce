"""Project entry point for CABCDS reproduction tasks."""

from __future__ import annotations

from cabcds.logging import setup_logger
from cabcds.roi_selector.config import load_roi_selector_config
from cabcds.roi_selector.inference import RoiSelector
from cabcds.roi_selector.training import RoiSelectorTrainer
from cabcds.hybrid_descriptor.config import load_hybrid_descriptor_config, load_hybrid_descriptor_inference_config
from cabcds.hybrid_descriptor.pipeline import HybridDescriptorPipeline
from cabcds.wsi_scorer.config import load_wsi_scorer_config
from cabcds.wsi_scorer.pipeline import WsiScorerPredictor, WsiScorerTrainer


def main() -> None:
    """Main entry point for end-to-end CABCDS reproduction."""
    logger = setup_logger()
    logger.info("Starting CABCDS end-to-end pipeline.")

    # --- Stage 1: ROI Selection ---
    logger.info("--- Stage 1: ROI Selection ---")
    roi_config = load_roi_selector_config()
    
    # Train ROI SVM model if needed
    if not roi_config.train_model_output_path.exists():
        logger.info(f"ROI SVM model not found at {roi_config.train_model_output_path}. Starting training...")
        trainer = RoiSelectorTrainer(roi_config)
        trainer.train()
        logger.info("ROI SVM training completed.")
    else:
        logger.info(f"ROI SVM model found at {roi_config.train_model_output_path}. Skipping training.")

    # Run inference to extract ROI patches
    logger.info("Initializing RoiSelector...")
    selector = RoiSelector(roi_config)
    logger.info("Running ROI selection (extracting patches)...")
    selector.select_rois()
    logger.info("Stage 1 (ROI Selection) completed.")

    # --- Stage 2 & 3: MF-CNN & Hybrid Descriptor Extraction ---
    # Note: MF-CNN training (Stage 2) is usually done via `python -m cabcds.mf_cnn` 
    # as it requires significant GPU/NPU time and specific paper-aligned hyperparameters.
    # Here we focus on the Inference/Extraction pipeline.
    logger.info("--- Stage 3: Hybrid Descriptor Extraction ---")
    hybrid_config = load_hybrid_descriptor_config()
    hybrid_infer_config = load_hybrid_descriptor_inference_config()
    
    # The pipeline will auto-detect checkpoints for CNN_seg, CNN_det, and CNN_global folds
    # if they are not explicitly provided in the config/env.
    logger.info("Initializing HybridDescriptorPipeline...")
    try:
        pipeline = HybridDescriptorPipeline(hybrid_config, hybrid_infer_config)
        logger.info("Running hybrid descriptor extraction...")
        descriptors = pipeline.run()
        logger.info(f"Extracted descriptors for {len(descriptors)} WSIs.")
    except Exception as e:
        logger.error(f"Failed to run HybridDescriptorPipeline: {e}")
        logger.info("Ensure MF-CNN checkpoints exist in output/mf_cnn/ or specify them via environment variables.")
        return

    # --- Stage 4: WSI Scorer (Final Grading) ---
    logger.info("--- Stage 4: WSI Scorer (Final Grading) ---")
    wsi_config = load_wsi_scorer_config()
    
    # Train WSI SVM if needed
    if not wsi_config.model_output_path.exists():
        logger.info("WSI SVM model not found. Starting training...")
        wsi_trainer = WsiScorerTrainer(wsi_config)
        wsi_trainer.train()
        logger.info("WSI SVM training completed.")
    
    # Run final prediction
    logger.info("Running final WSI scoring/prediction...")
    predictor = WsiScorerPredictor(wsi_config)
    predictions = predictor.predict()
    logger.info(f"Final predictions completed for {len(predictions)} WSIs.")
    logger.info("End-to-end pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
