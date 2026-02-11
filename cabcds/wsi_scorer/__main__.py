"""Run WSI scoring as a module.

Supports running training and/or prediction while overriding paths from the
default config. This is useful when artifacts live under `output/...` instead
of the config defaults under `data/...`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cabcds.wsi_scorer.config import WsiScorerConfig, load_wsi_scorer_config
from cabcds.wsi_scorer.pipeline import WsiScorerPredictor, WsiScorerTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI scorer (stage five): train and/or predict.")

    parser.add_argument(
        "--mode",
        type=str,
        default="train-predict",
        choices=("train", "predict", "train-predict"),
        help="Whether to run training, prediction, or both (default: train-predict).",
    )

    parser.add_argument(
        "--train-descriptor-csv",
        type=str,
        default=None,
        help="Optional: descriptor CSV used for training. Overrides config.descriptor_csv.",
    )
    parser.add_argument(
        "--predict-descriptor-csv",
        type=str,
        default=None,
        help="Optional: descriptor CSV used for prediction. If omitted, uses config.descriptor_csv.",
    )

    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional: labels CSV used for training. Overrides config.labels_csv.",
    )

    parser.add_argument(
        "--model-output-path",
        type=str,
        default=None,
        help=(
            "Optional: model joblib path. Used for saving when training, and loading when predicting. "
            "Overrides config.model_output_path."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Optional: directory to write reports/predictions. Overrides config.report_dir.",
    )

    return parser.parse_args()


def _override_config(base: WsiScorerConfig, args: argparse.Namespace) -> WsiScorerConfig:
    overrides: dict[str, object] = {}
    if args.train_descriptor_csv is not None:
        overrides["descriptor_csv"] = Path(args.train_descriptor_csv)
    if args.labels_csv is not None:
        overrides["labels_csv"] = Path(args.labels_csv)
    if args.model_output_path is not None:
        overrides["model_output_path"] = Path(args.model_output_path)
    if args.report_dir is not None:
        overrides["report_dir"] = Path(args.report_dir)

    if not overrides:
        return base

    payload = base.model_dump()
    payload.update(overrides)
    return WsiScorerConfig(**payload)


def main() -> None:
    """Entry point for WSI scoring tasks."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = _parse_args()

    base_config = load_wsi_scorer_config()
    config = _override_config(base_config, args)

    if args.mode in ("train", "train-predict"):
        trainer = WsiScorerTrainer(config)
        trainer.train()

    if args.mode in ("predict", "train-predict"):
        predictor = WsiScorerPredictor(config)
        descriptor_override: Path | None = None
        if args.predict_descriptor_csv is not None:
            descriptor_override = Path(args.predict_descriptor_csv)
        predictor.predict(descriptor_csv=descriptor_override)


if __name__ == "__main__":
    main()
