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

Preview patches (feature-size) are written under:

- `output/roi_selector/outputs/patches/`

This repo supports a cleaner split layout (recommended):

- `output/roi_selector/outputs/patches/train/TUPAC-TR-xxx/roi_..png`
- `output/roi_selector/outputs/patches/test/TUPAC-TE-xxx/roi_..png`

You can restructure an existing mixed folder using:

- `uv run python -m cabcds.roi_selector.report_export --restructure-split`

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

## MF-CNN (Stage 2, WIP)

This repo includes reference implementations of the MF-CNN submodules:

- `CNN_seg` (VGG16-FCN): mitosis candidate segmentation (trained from centroid annotations)
- `CNN_det` (AlexNet): mitosis / non-mitosis candidate classification
- `CNN_global` (AlexNet): 3-class proliferation score for ROI patches

### Baseline smoke test (no extra downloads)

The baseline uses only the data already present under `dataset/`.

- Run a quick forward+backward smoke test for `CNN_seg` + `CNN_det`:
	- `uv run python -m cabcds.mf_cnn --smoke`

### Train `CNN_seg` (mitosis candidate segmentation)

This trains a VGG16-FCN segmentation model from centroid annotations in the auxiliary mitosis dataset.

- `uv run python -m cabcds.mf_cnn --train-seg --device cpu --batch-size 1 --seg-epochs 1 --seg-max-steps 50`

Checkpoint output:

- `data/mf_cnn/models/cnn_seg.pt`

### Prepare `CNN_det` patches using `CNN_seg` candidates (paper-aligned)

This runs the trained `CNN_seg` on auxiliary tiles, converts predicted masks into connected-component candidates,
and writes a disk-backed patch dataset + index CSV for training `CNN_det`.

- `uv run python -m cabcds.mf_cnn --prepare-det --device cpu --det-seg-checkpoint data/mf_cnn/models/cnn_seg.pt`

Outputs:

- `data/mf_cnn/det_patches/index.csv`

### Train `CNN_det` (mitosis / non-mitosis)

- `uv run python -m cabcds.mf_cnn --train-det --device cpu --batch-size 16 --det-epochs 1 --det-max-steps 200 --det-index-csv data/mf_cnn/det_patches/index.csv`

Checkpoint output:

- `data/mf_cnn/models/cnn_det.pt`

### Prepare `CNN_global` patches (TUPAC train WSIs)

This extracts `512x512` patches with overlap `80` from `dataset/train/*.svs` and writes:

- patches under `data/mf_cnn/global_patches/<slide_id>/*.png`
- an index CSV at `data/mf_cnn/global_patches/index.csv`

Safe-by-default example (process 1 slide, extract 100 patches max):

- `uv run python -m cabcds.mf_cnn --prepare-global --max-slides 1 --max-patches-per-slide 100`

Resume is enabled by default (uses a per-slide `.done` marker). To re-run from scratch:

- delete `data/mf_cnn/global_patches/` or pass `--no-resume`

### Train `CNN_global`

After preparing patches and `index.csv`, run a short training (safe-by-default example):

- `uv run python -m cabcds.mf_cnn --train-global --global-epochs 1 --global-max-steps 10`

Slide-level validation split (recommended to avoid leakage):

- `uv run python -m cabcds.mf_cnn --train-global --global-epochs 1 --global-val-fraction 0.2`

Data augmentation (enabled by default):

- Disable: `uv run python -m cabcds.mf_cnn --train-global --no-global-augment`
- Adjust jitter: `uv run python -m cabcds.mf_cnn --train-global --global-color-jitter 0.05`

Checkpoint output:

- `data/mf_cnn/models/cnn_global.pt`

### Optional external mitosis datasets (placeholders)

The paper improves `CNN_seg` generalization by adding external mitosis datasets
(e.g. MITOS12 / MITOS14 / AMIDA13(+1)). This repo reserves the following paths:

- `dataset/external/mitos12/`
- `dataset/external/mitos14/`

They are **not required** to run the baseline. If you place external centroid-style
tiles on disk, you can enable mixed sampling via:

- `CABCDS_MFCNN_USE_EXTERNAL_MITOSIS_DATASETS=true`

## Troubleshooting

- If you interrupt inference, re-running `uv run python main.py` will skip already-processed slides based on the existing report CSV.
