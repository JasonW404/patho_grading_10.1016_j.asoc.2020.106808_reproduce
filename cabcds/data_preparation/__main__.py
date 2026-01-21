"""Run stage one preprocessing as a standalone module."""

from __future__ import annotations

import logging

from cabcds.data_preparation.config import load_stage_one_config
from cabcds.data_preparation.pipeline import StageOnePreprocessor


def main() -> None:
    """Run stage one preprocessing with module defaults."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_stage_one_config()
    logger.info("Starting stage one preprocessing.")
    preprocessor = StageOnePreprocessor(config)
    preprocessor.run()
    logger.info("Stage one preprocessing completed.")


if __name__ == "__main__":
    main()
