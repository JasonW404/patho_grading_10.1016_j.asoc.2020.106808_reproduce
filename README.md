# Pathology grading reproduction (CABCDS)

This repository reproduces parts of the pipeline described in:

- *Automated grading of breast cancer histopathology images using mitosis detection and whole-slide image features* (ASOC 2020, 10.1016/j.asoc.2020.106808)

The most actively maintained component in this repo is the **ROI selector** (traditional ML) that scans WSI images and extracts the top-$k$ candidate Regions of Interest (ROIs) for downstream scoring.

## Setup

This project uses `uv`.

- Create/sync environment: `uv sync`
- Run commands: prefix with `uv run ...`

Example:

- `uv run python roi_selector.py --help`

## Data layout

By default, paths are configured via `pydantic-settings` in [cabcds/roi_selector/config.py](cabcds/roi_selector/config.py).

Expected directories (defaults):

- Training WSIs: `dataset/train`
- Test WSIs: `dataset/test`
- ROI CSV annotations: `dataset/auxiliary_dataset_roi`

Generated outputs (ignored by git):

- Training patches: `output/roi_selector/training/{positive,negative,negative_generated}`
- Trained models: `output/roi_selector/models/`
- Inference outputs: `output/roi_selector/outputs/`
- Benchmark/holdout: `output/roi_selector/benchmark/`

The repo `.gitignore` excludes `dataset/` and `output/`.

### Configuration overrides

All ROI selector settings can be overridden via environment variables prefixed with `CABCDS_ROI_`.

Example (change negative target count):

- `CABCDS_ROI_NEG_TOTAL_TARGET_COUNT=2000 uv run python roi_selector.py --prepare-negative`

Nested fields use `__` as delimiter.

## ROI selector (Stage 1)

### 1) Generate training patches

Generate positive + negative patches and refresh the training index CSV:

- `uv run python roi_selector.py --prepare`

Or run individually:

- `uv run python roi_selector.py --prepare-positive`
- `uv run python roi_selector.py --prepare-negative`

Notes:

- Manually labelled negatives live in `output/roi_selector/training/negative`.
- Newly generated negatives go to `output/roi_selector/training/negative_generated` (so we don’t overwrite manual labels).

#### Optional: DL negative filter (to improve negative quality)

If you trained a DL negative filter (see below), enable it during negative sampling:

- `uv run python roi_selector.py --prepare-negative --use-negative-filter-dl`

### 2) Create a benchmark/holdout (optional but recommended)

Create a holdout set by sampling from the patch directories:

- `uv run python roi_selector.py --create-benchmark --benchmark-positive-count 200 --benchmark-negative-count 200`

To ensure holdout patches are never used for training, prune them out of the training directories (moves files and updates the index):

- `uv run python roi_selector.py --prune-benchmark-sources`

### 3) Train the ROI SVM

Train a Linear SVM on extracted handcrafted features:

- `uv run python roi_selector.py --train`

Model artifact:

- `output/roi_selector/models/roi_svm.joblib`

### 4) Run inference (full-slide scan, top-$k$ ROIs)

Run stage-one ROI selection over the test WSI directory:

- `uv run python main.py`

This uses [cabcds/roi_selector/inference.py](cabcds/roi_selector/inference.py) and writes an incremental report CSV (supports resume):

- `output/roi_selector/outputs/reports/stage_two_roi_selection.csv`

## Label Studio loop (optional)

### Train negative filter (DL)

Train a small CNN binary classifier from a simplified JSON:

- `uv run python roi_selector.py --train-negative-filter-dl output/roi_selector/training/label_studio_exports/negative_raw_simplified.json`

Model artifact:

- `output/roi_selector/models/neg_filter_dl.pt`

## Repo structure

- ROI selector code: [cabcds/roi_selector](cabcds/roi_selector)
- WSI scoring pipeline: [cabcds/wsi_scorer](cabcds/wsi_scorer)
- Hybrid descriptor: [cabcds/hybrid_descriptor](cabcds/hybrid_descriptor)

## Troubleshooting

- If you interrupt inference, re-running `uv run python main.py` will skip already-processed slides based on the existing report CSV.
