"""Run WSI scoring as a module."""

from __future__ import annotations

import logging

from cabcds.wsi_scorer.config import load_wsi_scorer_config
from cabcds.wsi_scorer.pipeline import WsiScorerPredictor, WsiScorerTrainer


def main() -> None:
    """Entry point for WSI scoring tasks."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    config = load_wsi_scorer_config()
    trainer = WsiScorerTrainer(config)
    trainer.train()

    predictor = WsiScorerPredictor(config)
    predictor.predict()


if __name__ == "__main__":
    main()
