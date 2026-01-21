"""Run hybrid descriptor pipeline as a module."""

from __future__ import annotations

import logging

from cabcds.hybrid_descriptor.config import (
    load_hybrid_descriptor_config,
    load_hybrid_descriptor_inference_config,
)
from cabcds.hybrid_descriptor.pipeline import HybridDescriptorPipeline


def main() -> None:
    """Entry point for hybrid descriptor generation."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    hybrid_config = load_hybrid_descriptor_config()
    infer_config = load_hybrid_descriptor_inference_config()

    pipeline = HybridDescriptorPipeline(hybrid_config, infer_config)
    pipeline.run()


if __name__ == "__main__":
    main()
