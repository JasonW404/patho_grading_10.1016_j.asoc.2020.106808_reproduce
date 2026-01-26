"""Functions to create positive and negative ROI patches."""

from __future__ import annotations

import logging
import random
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from ..config import ROISelectorConfig
from .features import is_patch_too_white, compute_cell_count

logger = logging.getLogger(__name__)


def compute_sampling_params(slide: openslide.OpenSlide, config: ROISelectorConfig) -> Tuple[int, float, int, int]:
    """Calculate sampling parameters closer to target magnification."""
    target_mag = config.sampling_magnification
    base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
    target_downsample = base_mag / target_mag
    read_level = slide.get_best_level_for_downsample(target_downsample)
    
    level_downsample = slide.level_downsamples[read_level]
    patch_size = config.infer_patch_size
    
    src_patch_size = int(patch_size * target_downsample)
    read_size = int(src_patch_size / level_downsample)
    
    return read_level, level_downsample, src_patch_size, read_size


def load_existing_rois(csv_path: Path) -> List[Tuple[int, int, int, int]]:
    """Load ROI coordinates from CSV file."""
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path, header=None, names=["x", "y", "w", "h"], usecols=[0, 1, 2, 3])
        return list(df.itertuples(index=False, name=None))
    except Exception:
        return []


def extract_and_resize_patch(
    slide: openslide.OpenSlide, 
    location: Tuple[int, int], 
    level: int, 
    read_size: int, 
    target_size: int
) -> Image.Image:
    """Read region from slide and resize to target patch size."""
    patch = slide.read_region(location, level, (read_size, read_size)).convert('RGB')
    if read_size != target_size:
        patch = patch.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return patch


def _process_positive_wsi(args: Tuple[Path, ROISelectorConfig]) -> int:
    """Process a single WSI for positive ROI generation (worker function)."""
    csv_file, config = args
    
    wsi_dir = config.preproc_dataset_dir
    output_dir = config.train_dataset_dir / config.train_positive_subdir
    patch_size = config.infer_patch_size
    
    wsi_name = csv_file.name.replace("-ROI.csv", "")
    processed_count = 0
    
    # Find WSI
    wsi_path = None
    for ext in config.preproc_image_extensions:
        candidate = wsi_dir / (wsi_name + ext)
        if candidate.exists():
            wsi_path = candidate
            break
    
    if not wsi_path:
        # logger.warning(f"WSI not found for {csv_file.name}") # Avoid spamming logs from workers if common
        return 0
        
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        logger.error(f"Failed to open slide {wsi_path}: {e}")
        return 0
    
    # Determine best level for magnification
    read_level, level_downsample, src_patch_size, read_size = compute_sampling_params(slide, config)

    rois = load_existing_rois(csv_file)
    
    for idx, (rx, ry, rw, rh) in enumerate(rois):
        # Center crop check
        if rw < src_patch_size or rh < src_patch_size:
            cx = rx + rw // 2
            cy = ry + rh // 2
            px = max(0, cx - src_patch_size // 2)
            py = max(0, cy - src_patch_size // 2)
            
            try:
                patch = extract_and_resize_patch(slide, (px, py), read_level, read_size, patch_size)
                patch_np = np.array(patch)
                if not is_patch_too_white(patch_np, config.feature_config):
                        save_path = output_dir / f"{wsi_name}_roi{idx}_p0.png"
                        patch.save(save_path)
                        processed_count += 1
            except Exception:
                pass
            continue

        # Tiling
        step = src_patch_size
        for y in range(ry, ry + rh - src_patch_size + 1, step):
            for x in range(rx, rx + rw - src_patch_size + 1, step):
                try:
                    patch = extract_and_resize_patch(slide, (x, y), read_level, read_size, patch_size)
                    patch_np = np.array(patch)
                    
                    if not is_patch_too_white(patch_np, config.feature_config):
                        save_path = output_dir / f"{wsi_name}_roi{idx}_{x}_{y}.png"
                        patch.save(save_path)
                        processed_count += 1
                except Exception:
                        pass

    slide.close()
    return processed_count


def create_positive_roi(config: ROISelectorConfig) -> None:
    """Generate positive ROI patches from CSV annotations.
    
    Reads ROI CSVs, extracts regions from WSIs, tiles them into patches,
    and saves them to the positive training directory.
    Uses multiprocessing to speed up extraction.
    """
    roi_csv_dir = config.roi_csv_dir
    output_dir = config.train_dataset_dir / config.train_positive_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(roi_csv_dir.glob("*-ROI.csv"))
    logger.info(f"Found {len(csv_files)} ROI CSV files.")
    
    # Determine resources
    total_cpus = multiprocessing.cpu_count()
    max_workers = max(1, total_cpus)
    logger.info(f"Using {max_workers} processed for positive ROI generation.")

    processed_count = 0
    
    # Prepare arguments for workers
    tasks = [(f, config) for f in csv_files]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_process_positive_wsi, tasks), total=len(tasks), desc="Processing Positive ROIs"))
    
    processed_count = sum(results)
    logger.info(f"Generated {processed_count} positive patches.")


def _process_negative_batch(args: Tuple[int, List[Path], ROISelectorConfig, int]) -> int:
    """Generate a batch of negative ROI patches (worker function)."""
    target_count, wsi_files, config, worker_id = args
    generated_count = 0
    
    output_dir = config.train_dataset_dir / config.train_negative_subdir
    patch_size = config.infer_patch_size
    
    # Increase persistency: 1000 attempts per target item
    max_attempts = max(target_count * 1000, 2000)
    attempts_total = 0
    
    while generated_count < target_count and attempts_total < max_attempts:
        wsi_path = random.choice(wsi_files)
        wsi_name = wsi_path.stem
        
        # Check specific constraints for this slide
        roi_csv_path = config.roi_csv_dir / f"{wsi_name}-ROI.csv"
        existing_rois = load_existing_rois(roi_csv_path)
        
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            
            # Determine best level for magnification
            read_level, level_downsample, src_patch_size, read_size = compute_sampling_params(slide, config)
            w, h = slide.dimensions
            
            # Increase tries per slide to reduce IO overhead of opening slides
            for _ in range(50):
                attempts_total += 1
                if attempts_total >= max_attempts:
                    break
                
                # Random coordinates
                border = config.infer_exclude_border
                if w <= 2 * border + src_patch_size or h <= 2 * border + src_patch_size:
                     continue
                
                rx = random.randint(border, w - border - src_patch_size)
                ry = random.randint(border, h - border - src_patch_size)
                
                # Check overlap with Positive ROIs
                overlap = False
                patch_rect = (rx, ry, rx + src_patch_size, ry + src_patch_size)
                
                for px, py, pw, ph in existing_rois:
                     if not (patch_rect[2] <= px or patch_rect[0] >= px + pw or
                             patch_rect[3] <= py or patch_rect[1] >= py + ph):
                         overlap = True
                         break
                
                if overlap:
                    continue
                
                patch = extract_and_resize_patch(slide, (rx, ry), read_level, read_size, patch_size)
                patch_np = np.array(patch)
                
                if is_patch_too_white(patch_np, config.feature_config):
                   continue

                # Check 2: Few or no cells
                cell_count = compute_cell_count(patch_np, config.feature_config.min_blob_area)
                
                # Threshold for "Few or no cells".
                # Keep fairly strict as per paper, but allow slightly more persistence
                if cell_count < 10:
                    save_path = output_dir / f"{wsi_name}_neg_{rx}_{ry}_w{worker_id}_{generated_count}.png"
                    patch.save(save_path)
                    generated_count += 1
                    break 
            
            slide.close()
        except Exception:
            pass
            
    return generated_count


def create_negative_roi(config: ROISelectorConfig) -> None:
    """Generate negative ROI patches.
    
    Randomly selects regions with few cells and high background using multiprocessing.
    """
    total_target_count = 600 # Target based on paper's 544
    
    wsi_dir = config.preproc_dataset_dir
    output_dir = config.train_dataset_dir / config.train_negative_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wsi_files = []
    for ext in config.preproc_image_extensions:
        wsi_files.extend(list(wsi_dir.glob(f"*{ext}")))
    
    if not wsi_files:
        logger.warning("No WSI files found for negative sampling.")
        return

    # Determine resources
    total_cpus = multiprocessing.cpu_count()
    max_workers = max(1, total_cpus)
    
    # Split target count among workers
    base_count = total_target_count // max_workers
    remainder = total_target_count % max_workers
    
    tasks = []
    for i in range(max_workers):
        count = base_count + (1 if i < remainder else 0)
        tasks.append((count, wsi_files, config, i))
    
    logger.info(f"Using {max_workers} processes to generate {total_target_count} negative patches.")
    
    generated_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
         results = list(tqdm(executor.map(_process_negative_batch, tasks), total=len(tasks), desc="Processing Negative ROIs"))
    
    generated_count = sum(results)
    logger.info(f"Generated {generated_count} negative patches.")


def generate_labelled_csv(config: ROISelectorConfig) -> Path:
    """Generate ROI_labelled.csv combining positive and negative samples."""
    pos_dir = config.train_dataset_dir / config.train_positive_subdir
    neg_dir = config.train_dataset_dir / config.train_negative_subdir
    
    rows = []
    
    # Positive = 1
    for p in pos_dir.glob("*.png"):
        rows.append((str(p.absolute()), 1)) # Use absolute path or relative?
        # User requirement says "ROI file name".
        # I'll store the full path to make dataloader easier.
    
    # Negative = 0
    for p in neg_dir.glob("*.png"):
        rows.append((str(p.absolute()), 0))
        
    csv_path = config.train_dataset_dir / "ROI_labelled.csv"
    
    df = pd.DataFrame(rows, columns=["path", "label"])
    df.to_csv(csv_path, index=False)
        
    logger.info(f"Created labelled CSV at {csv_path} with {len(rows)} samples.")
    return csv_path
