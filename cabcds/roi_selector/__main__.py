"""Run ROI selector training or inference as a module."""

from __future__ import annotations

import logging

from cabcds.roi_selector.config import load_roi_selector_inference_config, load_roi_selector_training_config
from cabcds.roi_selector.inference import RoiSelector
from cabcds.roi_selector.training import RoiSelectorTrainer


def main() -> None:
    """Entry point for ROI selector tasks."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    train_config = load_roi_selector_training_config()
    trainer = RoiSelectorTrainer(train_config)
    trainer.train()

    infer_config = load_roi_selector_inference_config()
    selector = RoiSelector(infer_config)
    selector.select_rois()


if __name__ == "__main__":
    main()
