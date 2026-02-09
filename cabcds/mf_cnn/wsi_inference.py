"""WSI Inference tools for MF-CNN pipeline.

This module handles running inference (CNN_seg) on Whole Slide Images (WSIs),
specifically within ROIs selected by the first stage.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from skimage import measure

try:
    import openslide
except ImportError:
    openslide = None

from cabcds.mf_cnn.checkpoints import load_cnn_seg_from_checkpoint
from cabcds.mf_cnn.preprocess import load_roi_rects_from_stage_two_report, iter_tupac_train_slides

logger = logging.getLogger(__name__)


def _imagenet_normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def generate_candidates_within_rois(
    *,
    slides_dir: Path,
    roi_report_csv: Path,
    output_dir: Path,
    seg_checkpoint_path: Path,
    device: torch.device,
    threshold: float = 0.5,
    min_area: int = 10,
) -> None:
    """Generate candidate mitosis centroids by running CNN_seg inside ROIs.

    The paper workflow:
    1. Select Top-N ROIs using Low-Mag strategy (Stage 1 & 2).
    2. Run CNN_seg inside these ROIs on the High-Mag (40x) slide.
    3. Threshold the probability map and find connected components.
    4. These centroids become the centers for:
        a) CNN_det candidates (80x80) -> Not fully used for *training* Det in paper (Det is trained on Aux data),
           but used during inference.
        b) CNN_global input patches (512x512).

    This function produces per-slide CSV files in `output_dir` (e.g., `TUPAC-TR-001.csv`),
    each containing `x,y` lines (level-0 coords) for all candidates found in that slide.

    Args:
        slides_dir: Directory containing TUPAC-TR-*.svs files.
        roi_report_csv: Path to `stage_two_roi_selection.csv`.
        output_dir: Directory to store `{slide_id}.csv` files.
        seg_checkpoint_path: Path to `cnn_seg.pt`.
        device: Torch device.
        threshold: Not used if using argmax (binary classification), kept for API compat.
        min_area: Minimum pixel area to consider a blob a candidate.
    """

    slides_dir = Path(slides_dir)
    roi_report_csv = Path(roi_report_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if openslide is None:
        raise RuntimeError("openslide-python is not available")

    # Load CNN_seg
    logger.info("Loading CNN_seg from %s", seg_checkpoint_path)
    model = load_cnn_seg_from_checkpoint(Path(seg_checkpoint_path), map_location="cpu")
    model.to(device)
    model.eval()

    # Iterate slides
    for slide_path in iter_tupac_train_slides(slides_dir):
        slide_id = slide_path.stem
        csv_out_path = output_dir / f"{slide_id}.csv"

        if csv_out_path.exists():
            logger.info("Skipping %s (already exists)", slide_id)
            continue

        # Load ROIs (Top 4)
        rois = load_roi_rects_from_stage_two_report(
            report_csv_path=roi_report_csv,
            slide_id=slide_id,
            slide_path=slide_path,
        )

        if not rois:
            logger.warning("No ROIs found for %s (check report)", slide_id)
            continue

        candidates: list[tuple[int, int]] = []

        with openslide.OpenSlide(str(slide_path)) as slide:
            for roi_idx, rect in enumerate(rois):
                # Check bounds
                if rect.w <= 0 or rect.h <= 0:
                    continue

                # Read ROI region
                try:
                    tile_pil = slide.read_region((rect.x, rect.y), 0, (rect.w, rect.h)).convert("RGB")
                except Exception as e:
                    logger.error("Failed reading ROI %d for %s: %s", roi_idx, slide_id, e)
                    continue
                
                # Simple sliding window approach for the ROI
                window_size = 2000
                stride = 1000 # 50% overlap to handle boundary effects
                
                w, h = tile_pil.size
                tile_np = np.asarray(tile_pil)
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        # Crop window
                        y_end = min(h, y + window_size)
                        x_end = min(w, x + window_size)
                        
                        sub_img = tile_np[y:y_end, x:x_end]
                        sh, sw = sub_img.shape[:2]
                        if sh < 32 or sw < 32: continue
                        
                        # Inference
                        img_tensor = torch.from_numpy(np.ascontiguousarray(sub_img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        img_tensor = img_tensor.to(device)
                        img_tensor = _imagenet_normalize_batch(img_tensor)
                        
                        try:
                            logits = model(img_tensor).logits
                            # Argmax
                            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                            
                            # Connected components
                            labeled = measure.label(pred > 0)
                            regions = [r for r in measure.regionprops(labeled) if r.area >= min_area]
                            
                            for r in regions:
                                cy, cx = r.centroid
                                # Local to Window -> Local to ROI -> Global
                                global_x = rect.x + x + int(cx)
                                global_y = rect.y + y + int(cy)
                                candidates.append((global_x, global_y))
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logger.error("OOM processing window %dx%d on %s", sw, sh, device)
                                torch.cuda.empty_cache()
                            else:
                                raise e

        # Remove duplicate candidates
        unique_candidates = sorted(list(set(candidates)))
        
        logger.info("Found %d candidates in %s", len(unique_candidates), slide_id)
        
        with csv_out_path.open("w", newline="") as f:
            for cx, cy in unique_candidates:
                f.write(f"{cx},{cy}\n")
