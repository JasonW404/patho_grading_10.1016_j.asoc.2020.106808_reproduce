#!/usr/bin/env python3
"""
Main entry point for ROI Selector pipeline.
Supports data preparation and training via command line arguments.
"""

import argparse
import logging
import sys
from pathlib import Path

from cabcds.roi_selector.config import load_roi_selector_config
from cabcds.roi_selector.utils.create_roi_patches import (
    create_positive_roi,
    create_negative_roi,
    generate_labelled_csv,
)
from cabcds.roi_selector.negative_filter_dl import train_negative_filter_dl
from cabcds.roi_selector.benchmark import create_benchmark, prune_benchmark_sources
from cabcds.roi_selector.training import RoiSelectorTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROI Selector Pipeline Management",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--prepare", 
        action="store_true", 
        help="Run data preparation: generate positive and negative patches from WSIs."
    )
    parser.add_argument(
        "--prepare-positive", 
        action="store_true", 
        help="Run data preparation: generate only positive patches."
    )
    parser.add_argument(
        "--prepare-negative", 
        action="store_true", 
        help="Run data preparation: generate only negative patches."
    )
    parser.add_argument(
        "--neg-total-target-count",
        type=int,
        default=None,
        help="Override config.neg_total_target_count for negative sampling.",
    )
    parser.add_argument(
        "--train-negative-filter-dl",
        type=str,
        default=None,
        help=(
            "Train a deep-learning negative filter model from a simplified label JSON (id/image_path/label). "
            "Recommended input: output/roi_selector/training/label_studio_exports/negative_raw_simplified.json"
        ),
    )
    parser.add_argument(
        "--use-negative-filter-dl",
        action="store_true",
        help="Enable the trained deep-learning negative filter during --prepare-negative/--prepare.",
    )
    parser.add_argument(
        "--negative-output-subdir",
        type=str,
        default=None,
        help=(
            "Subdirectory under output/roi_selector/training to write newly generated negatives into. "
            "Defaults to config.train_negative_generated_subdir. This does not overwrite the manually labelled negatives."
        ),
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Run SVM training using generated patches."
    )

    parser.add_argument(
        "--create-benchmark",
        action="store_true",
        help=(
            "Create a benchmark (holdout) dataset by sampling patches from positive and negative_generated. "
            "This benchmark will be excluded from training."
        ),
    )
    parser.add_argument(
        "--benchmark-positive-count",
        type=int,
        default=200,
        help="Number of positive patches to include in benchmark (default: 200).",
    )
    parser.add_argument(
        "--benchmark-negative-count",
        type=int,
        default=200,
        help="Number of negative_generated patches to include in benchmark (default: 200).",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=42,
        help="Random seed used to sample benchmark patches (default: 42).",
    )
    parser.add_argument(
        "--benchmark-overwrite",
        action="store_true",
        help="If set, delete and recreate the benchmark directory.",
    )
    parser.add_argument(
        "--prune-benchmark-sources",
        action="store_true",
        help=(
            "Move benchmark-selected source patches out of training directories and update ROI_labelled.csv. "
            "This ensures holdout patches cannot be used for training."
        ),
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

    if args.use_negative_filter_dl:
        config = config.model_copy(update={"neg_filter_dl_enabled": True})

    if args.negative_output_subdir:
        config = config.model_copy(update={"train_negative_generated_subdir": args.negative_output_subdir})

    if args.neg_total_target_count is not None:
        config = config.model_copy(update={"neg_total_target_count": int(args.neg_total_target_count)})

    if args.train_negative_filter_dl:
        labels_json = Path(args.train_negative_filter_dl)
        train_negative_filter_dl(labels_json, config)
        logger.info("DL negative filter training completed successfully.")

    if args.create_benchmark:
        index_csv = create_benchmark(
            config,
            positive_count=int(args.benchmark_positive_count),
            negative_count=int(args.benchmark_negative_count),
            seed=int(args.benchmark_seed),
            overwrite=bool(args.benchmark_overwrite),
        )
        logger.info("Benchmark created: %s", index_csv)

    if args.prune_benchmark_sources:
        moved, removed_rows = prune_benchmark_sources(config, move=True)
        logger.info("Pruned benchmark sources: moved=%d removed_rows=%d", moved, removed_rows)

    ran_aux_action = bool(
        args.train_negative_filter_dl
        or args.create_benchmark
        or args.prune_benchmark_sources
    )
    ran_pipeline_action = bool(args.prepare or args.prepare_positive or args.prepare_negative or args.train)
    if ran_aux_action and not ran_pipeline_action:
        return

    # 1. Data Preparation
    run_positive = args.prepare or args.prepare_positive
    run_negative = args.prepare or args.prepare_negative
    
    if run_positive or run_negative:
        logger.info("=== Starting Data Preparation Phase ===")
        try:
            if run_positive:
                logger.info("Generating Positive ROIs...")
                create_positive_roi(config)
            
            if run_negative:
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

    if not ran_pipeline_action and not ran_aux_action:
        logger.warning("No action specified. Use --prepare or --train.")
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
