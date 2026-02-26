"""MF-CNN utilities entry point.

This module focuses on *smoke checks* so the MF-CNN stack can be validated with
the data already present in this repo (TUPAC16 + auxiliary mitosis zips).

Example:
    `uv run python -m cabcds.mf_cnn --smoke`
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
import re
import signal
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from cabcds.mf_cnn import CNNDet, CNNGlobal, CNNSeg, load_mfc_cnn_config
from cabcds.mf_cnn.utils.seg_patches import SegPatchDataset, build_seg_image_refs, split_uids
from cabcds.mf_cnn.preprocess import prepare_global_patch_dataset
from cabcds.mf_cnn.utils.loader import (
    GlobalScoringPatchDataset,
    GlobalPatchIndexRow,
    build_default_mf_cnn_train_loaders,
    PreparedMitosisDetectionPatchDataset,
    load_slide_labels_from_csv,
    load_tupac_train_scores,
    read_global_patch_index,
    read_det_patch_index,
)
from cabcds.mf_cnn.det_data import prepare_cnn_det_patches_from_aux_zips
from cabcds.mf_cnn.wsi_inference import generate_candidates_within_rois


logger = logging.getLogger(__name__)


def _append_metrics_row(csv_path: Path, row: dict[str, object]) -> None:
    """Append one metrics row to a CSV file.

    Keeps a stable header across calls. If new keys appear in later rows, the
    file is rewritten with an expanded header (metrics CSVs are small).
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing header.
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
    existing_header = [str(h) for h in (existing_header or [])]

    if not existing_header:
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    new_keys = [k for k in row.keys() if k not in existing_header]
    if new_keys:
        with csv_path.open("r", newline="") as f:
            dict_reader = csv.DictReader(f)
            old_rows = list(dict_reader)
        fieldnames = existing_header + new_keys
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old in old_rows:
                writer.writerow(old)
            writer.writerow(row)
        return

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_header)
        writer.writerow(row)


def _best_f1_threshold(*, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute the score threshold that maximizes F1.

    Returns zeros when the curve cannot be computed (e.g. only one class).
    """

    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0 or y_score.size == 0:
        return {"best_thr": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}
    if len(np.unique(y_true)) < 2:
        return {"best_thr": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return {"best_thr": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = (2.0 * precision_t * recall_t) / np.maximum(1e-12, precision_t + recall_t)
    best_idx = int(np.nanargmax(f1))
    return {
        "best_thr": float(thresholds[best_idx]),
        "best_f1": float(f1[best_idx]),
        "best_precision": float(precision_t[best_idx]),
        "best_recall": float(recall_t[best_idx]),
    }


def _qa_det_index(
    *,
    index_csv: Path,
    out_dir: Path,
    radius: int,
    samples: int,
    seed: int,
) -> None:
    """QA a prepared CNN_det `index.csv` against GT centroids.

    Produces:
      - `qa_distances.csv`: per-patch nearest-GT distance.
      - `qa_summary.json`: aggregate stats (pos/neg counts, within-radius rates).
      - `qa_samples.png`: optional grid of patch thumbnails with overlayed center.
    """

    index_csv = Path(index_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_mfc_cnn_config()
    base = cfg.tupac_aux_mitoses_dir
    ground_truth_zip = base / cfg.mitoses_ground_truth_zip
    det_crop_size = int(getattr(cfg, "det_crop_size", 80))

    if not index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {index_csv}")
    if not ground_truth_zip.exists():
        raise FileNotFoundError(f"ground_truth_zip not found: {ground_truth_zip}")

    centroid_cache: dict[tuple[str, str], list[tuple[int, int]]] = {}

    def _read_centroids(case_id: str, tile_id: str) -> list[tuple[int, int]]:
        key = (str(case_id), str(tile_id))
        if key in centroid_cache:
            return centroid_cache[key]
        member = f"mitoses_ground_truth/{case_id}/{tile_id}.csv"
        try:
            with ZipFile(ground_truth_zip) as zf:
                with zf.open(member) as fp:
                    content = fp.read().decode("utf-8", errors="replace").strip()
        except Exception:
            centroid_cache[key] = []
            return []

        out: list[tuple[int, int]] = []
        if content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                except Exception:
                    continue
                out.append((x, y))
        centroid_cache[key] = out
        return out

    rows: list[dict[str, object]] = []
    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        logger.warning("QA det index: empty index_csv=%s", str(index_csv))
        return

    # Compute nearest centroid distance for each patch.
    out_rows: list[dict[str, object]] = []
    pos_d: list[float] = []
    neg_d: list[float] = []
    within_pos = 0
    within_neg = 0
    r2 = float(int(radius) * int(radius))

    # Also compute GT coverage: for each GT centroid, what is the nearest candidate center?
    # (a) any candidate, (b) positive-labeled candidate.
    tile_to_candidates: dict[tuple[str, str], list[tuple[int, int]]] = {}
    tile_to_pos_candidates: dict[tuple[str, str], list[tuple[int, int]]] = {}
    tile_to_windows: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}
    tile_to_pos_windows: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}

    for r in rows:
        case_id = str(r.get("case_id") or "")
        tile_id = str(r.get("tile_id") or "")
        label = int(float(r.get("label") or 0))
        cx = int(float(r.get("cx") or 0))
        cy = int(float(r.get("cy") or 0))
        top = int(float(r.get("top") or 0))
        left = int(float(r.get("left") or 0))
        mean_prob = float(r.get("mean_prob") or 0.0)
        max_prob = float(r.get("max_prob") or 0.0)
        area = int(float(r.get("area") or 0))

        cents = _read_centroids(case_id, tile_id)
        if not cents:
            d2 = float("inf")
        else:
            d2 = float(min((cx - gx) ** 2 + (cy - gy) ** 2 for gx, gy in cents))
        d = float(np.sqrt(d2)) if np.isfinite(d2) else float("inf")

        if label == 1:
            pos_d.append(d)
            if d2 <= r2:
                within_pos += 1
        else:
            neg_d.append(d)
            if d2 <= r2:
                within_neg += 1

        out_rows.append(
            {
                "path": str(r.get("path") or ""),
                "label": int(label),
                "case_id": case_id,
                "tile_id": tile_id,
                "cx": int(cx),
                "cy": int(cy),
                "top": int(top),
                "left": int(left),
                "area": int(area),
                "mean_prob": float(mean_prob),
                "max_prob": float(max_prob),
                "nearest_gt_dist": float(d),
                "nearest_gt_within_r": bool(d2 <= r2),
            }
        )

        key = (case_id, tile_id)
        tile_to_candidates.setdefault(key, []).append((int(cx), int(cy)))
        tile_to_windows.setdefault(key, []).append((int(left), int(top), int(left) + det_crop_size, int(top) + det_crop_size))
        if int(label) == 1:
            tile_to_pos_candidates.setdefault(key, []).append((int(cx), int(cy)))
            tile_to_pos_windows.setdefault(key, []).append((int(left), int(top), int(left) + det_crop_size, int(top) + det_crop_size))


    # GT coverage stats.
    gt_total = 0
    gt_within_any = 0
    gt_within_pos = 0
    gt_within_any_crop = 0
    gt_within_pos_crop = 0
    gt_any_d: list[float] = []
    gt_pos_d: list[float] = []

    gt_rows: list[dict[str, object]] = []
    for (case_id, tile_id), cands in tile_to_candidates.items():
        cents = _read_centroids(case_id, tile_id)
        if not cents:
            continue
        pos_cands = tile_to_pos_candidates.get((case_id, tile_id), [])
        windows = tile_to_windows.get((case_id, tile_id), [])
        pos_windows = tile_to_pos_windows.get((case_id, tile_id), [])

        for gx, gy in cents:
            gt_total += 1
            if cands:
                any_d2 = float(min((gx - cx) ** 2 + (gy - cy) ** 2 for cx, cy in cands))
            else:
                any_d2 = float("inf")
            any_cand_d = float(np.sqrt(any_d2)) if np.isfinite(any_d2) else float("inf")
            gt_any_d.append(any_cand_d)
            within_any = bool(any_d2 <= r2)
            if within_any:
                gt_within_any += 1

            if pos_cands:
                pos_d2 = float(min((gx - cx) ** 2 + (gy - cy) ** 2 for cx, cy in pos_cands))
            else:
                pos_d2 = float("inf")
            pos_cand_d = float(np.sqrt(pos_d2)) if np.isfinite(pos_d2) else float("inf")
            gt_pos_d.append(pos_cand_d)
            within_pos_cand = bool(pos_d2 <= r2)
            if within_pos_cand:
                gt_within_pos += 1

            within_any_crop = bool(any(l <= gx < r and t <= gy < b for l, t, r, b in windows))
            within_pos_crop = bool(any(l <= gx < r and t <= gy < b for l, t, r, b in pos_windows))
            if within_any_crop:
                gt_within_any_crop += 1
            if within_pos_crop:
                gt_within_pos_crop += 1

            gt_rows.append(
                {
                    "case_id": str(case_id),
                    "tile_id": str(tile_id),
                    "gt_x": int(gx),
                    "gt_y": int(gy),
                    "nearest_any_cand_dist": float(any_cand_d),
                    "nearest_any_cand_within_r": bool(within_any),
                    "nearest_pos_cand_dist": float(pos_cand_d),
                    "nearest_pos_cand_within_r": bool(within_pos_cand),
                    "within_any_crop": bool(within_any_crop),
                    "within_pos_crop": bool(within_pos_crop),
                    "n_tile_candidates": int(len(cands)),
                    "n_tile_pos_candidates": int(len(pos_cands)),
                }
            )

    def _pct(p: int, n: int) -> float:
        return float(p / max(1, n))

    def _finite(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr[np.isfinite(arr)]

    pos_d_f = _finite(pos_d)
    neg_d_f = _finite(neg_d)
    gt_any_d_f = _finite(gt_any_d)
    gt_pos_d_f = _finite(gt_pos_d)

    summary = {
        "index_csv": str(index_csv),
        "ground_truth_zip": str(ground_truth_zip),
        "radius": int(radius),
        "n_rows": int(len(out_rows)),
        "n_pos": int(len(pos_d)),
        "n_neg": int(len(neg_d)),
        "gt_total": int(gt_total),
        "gt_within_r_any_candidate": int(gt_within_any),
        "gt_within_r_positive_candidate": int(gt_within_pos),
        "gt_within_r_any_candidate_rate": _pct(gt_within_any, gt_total),
        "gt_within_r_positive_candidate_rate": _pct(gt_within_pos, gt_total),
        "gt_within_any_crop": int(gt_within_any_crop),
        "gt_within_pos_crop": int(gt_within_pos_crop),
        "gt_within_any_crop_rate": _pct(gt_within_any_crop, gt_total),
        "gt_within_pos_crop_rate": _pct(gt_within_pos_crop, gt_total),
        "pos_within_r": int(within_pos),
        "neg_within_r": int(within_neg),
        "pos_within_r_rate": _pct(within_pos, len(pos_d)),
        "neg_within_r_rate": _pct(within_neg, len(neg_d)),
        "pos_dist_median": float(np.median(pos_d_f)) if pos_d_f.size else float("nan"),
        "neg_dist_median": float(np.median(neg_d_f)) if neg_d_f.size else float("nan"),
        "pos_dist_p90": float(np.percentile(pos_d_f, 90)) if pos_d_f.size else float("nan"),
        "neg_dist_p10": float(np.percentile(neg_d_f, 10)) if neg_d_f.size else float("nan"),
        "gt_any_cand_dist_median": float(np.median(gt_any_d_f)) if gt_any_d_f.size else float("nan"),
        "gt_any_cand_dist_p90": float(np.percentile(gt_any_d_f, 90)) if gt_any_d_f.size else float("nan"),
        "gt_pos_cand_dist_median": float(np.median(gt_pos_d_f)) if gt_pos_d_f.size else float("nan"),
        "gt_pos_cand_dist_p90": float(np.percentile(gt_pos_d_f, 90)) if gt_pos_d_f.size else float("nan"),
    }

    (out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "qa_distances.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    if gt_rows:
        with (out_dir / "qa_gt_coverage.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(gt_rows)

    logger.info(
        "QA det index done: rows=%d pos=%d neg=%d pos_within_r=%.3f neg_within_r=%.3f out=%s",
        int(summary["n_rows"]),
        int(summary["n_pos"]),
        int(summary["n_neg"]),
        float(summary["pos_within_r_rate"]),
        float(summary["neg_within_r_rate"]),
        str(out_dir),
    )

    # Optional sample grid.
    if int(samples) <= 0:
        return

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    rng = random.Random(int(seed))
    pos_rows = [r for r in out_rows if int(r["label"]) == 1]
    neg_rows = [r for r in out_rows if int(r["label"]) == 0]
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)

    half = int(samples) // 2
    chosen = pos_rows[:half] + neg_rows[: max(0, int(samples) - half)]
    if not chosen:
        return

    thumb = 96
    cols = int(np.ceil(np.sqrt(len(chosen))))
    rows_n = int(np.ceil(len(chosen) / max(1, cols)))
    canvas = Image.new("RGB", (cols * thumb, rows_n * thumb), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, r in enumerate(chosen):
        path = Path(str(r["path"]))
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue

        # Estimate candidate center in patch coordinates.
        # With the new index fields, center is (cx-left, cy-top). Otherwise, default to patch center.
        cx = int(r.get("cx") or 0)
        cy = int(r.get("cy") or 0)
        left = int(r.get("left") or 0)
        top = int(r.get("top") or 0)
        px = cx - left
        py = cy - top
        if px <= 0 and py <= 0:
            px = im.size[0] // 2
            py = im.size[1] // 2

        # Overlay center cross.
        d = ImageDraw.Draw(im)
        d.line([(px - 4, py), (px + 4, py)], fill=(255, 0, 0), width=1)
        d.line([(px, py - 4), (px, py + 4)], fill=(255, 0, 0), width=1)

        im_t = im.resize((thumb, thumb))
        x0 = (i % cols) * thumb
        y0 = (i // cols) * thumb
        canvas.paste(im_t, (x0, y0))

        # Small label marker.
        lab = int(r["label"])
        color = (0, 255, 0) if lab == 1 else (255, 255, 0)
        draw.rectangle([x0, y0, x0 + 8, y0 + 8], fill=color)

    canvas.save(out_dir / "qa_samples.png")


@torch.no_grad()
def _eval_seg_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    require_positive: bool,
    log_every: int = 0,
    auc_max_pixels: int = 2_000_000,
) -> dict[str, float]:
    """Evaluate CNN_seg on a loader.

    Metrics are computed for the positive class (mitosis=1) using pixel-level
    confusion counts.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()

    tp = fp = fn = tn = 0
    correct = 0
    total = 0
    loss_sum = 0.0
    kept = 0
    seen = 0
    
    # Store predictions for AUC.
    # AUC computed on *all pixels* can be extremely large and slow (and may appear
    # to "hang" with no logs). We cap sampling by default.
    all_y_true: list[np.ndarray] = []
    all_y_score: list[np.ndarray] = []
    collected_auc_pixels = 0

    for images, masks in loader:
        seen += 1
        if int(max_batches) > 0 and seen > int(max_batches):
            break
        if bool(require_positive) and int((masks == 1).sum()) == 0:
            continue

        if int(log_every) > 0 and (seen == 1 or seen % int(log_every) == 0):
            logger.info(
                "CNN_seg(paper) eval progress: seen=%d kept=%d max_batches=%d require_positive=%s",
                int(seen),
                int(kept),
                int(max_batches),
                str(bool(require_positive)),
            )

        images = images.to(device)
        masks = masks.to(device)
        images = _imagenet_normalize_batch(images)

        out = model(images)
        loss = criterion(out.logits, masks)
        loss_sum += float(loss.detach().cpu())
        
        # Collect for AUC (sampled).
        # auc_max_pixels:
        # - 0 disables AUC collection.
        # - >0 caps total sampled pixels across the eval loop.
        # - <0 means no cap.
        if int(auc_max_pixels) != 0:
            probs = F.softmax(out.logits, dim=1)[:, 1, :, :]  # Prob class 1
            y_true_flat = masks.detach().cpu().numpy().ravel()
            y_score_flat = probs.detach().cpu().numpy().ravel()

            if int(auc_max_pixels) > 0:
                remaining = int(auc_max_pixels) - int(collected_auc_pixels)
                if remaining > 0:
                    if y_true_flat.size > remaining:
                        stride = max(1, int(y_true_flat.size // remaining))
                        y_true_flat = y_true_flat[::stride][:remaining]
                        y_score_flat = y_score_flat[::stride][:remaining]
                    collected_auc_pixels += int(y_true_flat.size)
                    all_y_true.append(y_true_flat)
                    all_y_score.append(y_score_flat)
            else:
                all_y_true.append(y_true_flat)
                all_y_score.append(y_score_flat)

        pred = torch.argmax(out.logits, dim=1)
        correct += int((pred == masks).sum().detach().cpu())
        total += int(masks.numel())

        pred_pos = pred == 1
        gt_pos = masks == 1
        tp += int((pred_pos & gt_pos).sum().detach().cpu())
        fp += int((pred_pos & (~gt_pos)).sum().detach().cpu())
        fn += int(((~pred_pos) & gt_pos).sum().detach().cpu())
        tn += int(((~pred_pos) & (~gt_pos)).sum().detach().cpu())

        kept += 1

    avg_loss = float(loss_sum / max(1, kept))
    acc = float(correct / max(1, total))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    dice = float((2 * tp) / max(1, 2 * tp + fp + fn))
    iou = float(tp / max(1, tp + fp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-12, precision + recall))
    
    auc = 0.0
    if all_y_true:
        try:
            y_true_flat = np.concatenate(all_y_true)
            y_score_flat = np.concatenate(all_y_score)
            # Only compute AUC if we have both classes
            if len(np.unique(y_true_flat)) > 1:
                auc = float(roc_auc_score(y_true_flat, y_score_flat))
        except Exception:
            pass

    model.train()
    return {
        "batches_seen": float(seen),
        "batches_kept": float(kept),
        "loss": float(avg_loss),
        "pixel_acc": float(acc),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _train_seg_paper(
    *,
    checkpoint_path: Path,
    metrics_csv: Path,
    batch_size: int,
    tune_epochs: int,
    final_epochs: int,
    max_steps_per_epoch: int,
    device: str,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    split_seed: int,
    eval_max_batches: int,
    eval_every: int,
    eval_log_every: int,
    eval_auc_max_pixels: int,
    require_positive_metrics: bool,
    early_stop_patience: int,
    patch_size: int,
    overlap: int,
) -> None:
    """Paper-aligned CNN_seg training: 60/20/20 image split, 512/overlap80 patches.

    Requirements enforced here:
    - Uses TUPAC auxiliary mitosis dataset + external MITOS12 and MITOS14.
    - Random image-level split: 60% train / 20% val / 20% test.
    - Train on val for tuning; then merge val into train for a final phase.
    - Uses test split for early stopping and final evaluation.
    - Hyperparameters are fixed to paper values: SGD lr=1e-4, momentum=0.9, weight_decay=5e-4.
    - Always uses ImageNet pretrained weights.
    """

    if int(patch_size) != 512 or int(overlap) != 80:
        raise ValueError("Paper-aligned CNN_seg requires patch_size=512 and overlap=80")

    cfg = load_mfc_cnn_config()
    if not bool(cfg.use_external_mitosis_datasets):
        logger.warning(
            "Paper-aligned CNN_seg typically requires external MITOS12+MITOS14. "
            "To use them, set this env var to 1."
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(metrics_csv)

    last_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_last{checkpoint_path.suffix}")
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")

    stop_requested = False
    stop_signal: str | None = None

    def _request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_signal
        stop_requested = True
        try:
            stop_signal = signal.Signals(signum).name
        except Exception:
            stop_signal = str(signum)
        logger.warning(
            "CNN_seg(paper) received %s; will save '%s' and stop at next safe point",
            stop_signal,
            str(last_checkpoint_path),
        )

    base = cfg.tupac_aux_mitoses_dir
    image_zips = [base / rel for rel in cfg.mitoses_image_zip_parts]
    ground_truth_zip = base / cfg.mitoses_ground_truth_zip

    # Updated: Only include external datasets if they exist on disk.
    # User specifically requested excluding MITOS12 if not present/desired.
    # For MITOS14, we specifically target the 'train' subdirectory if available.
    external_roots: list[Path] = []
    if cfg.use_external_mitosis_datasets:
        mitos14_train = cfg.mitos14_dir / "train"
        if mitos14_train.exists():
             external_roots.append(mitos14_train)
        elif cfg.mitos14_dir.exists():
            external_roots.append(cfg.mitos14_dir)
            
        if cfg.mitos12_dir.exists():
            external_roots.append(cfg.mitos12_dir)


    image_refs = build_seg_image_refs(
        image_zip_parts=image_zips,
        ground_truth_zip=ground_truth_zip,
        external_roots=external_roots,
        require_external_pairs=True,
    )
    uids = sorted({r.uid for r in image_refs})
    train_uids, val_uids, test_uids = split_uids(
        uids,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=int(split_seed),
    )
    logger.info(
        "CNN_seg paper split (images): total=%d train=%d val=%d test=%d seed=%d",
        len(uids),
        len(train_uids),
        len(val_uids),
        len(test_uids),
        int(split_seed),
    )

    train_ds = SegPatchDataset(
        image_refs=image_refs,
        allowed_uids=train_uids,
        patch_size=512,
        overlap=80,
        nuclei_min_area=int(cfg.nuclei_min_area),
        centroid_fallback_radius=int(cfg.centroid_fallback_radius),
        augment=True,
        seed=int(cfg.seed),
    )
    val_ds = SegPatchDataset(
        image_refs=image_refs,
        allowed_uids=val_uids,
        patch_size=512,
        overlap=80,
        nuclei_min_area=int(cfg.nuclei_min_area),
        centroid_fallback_radius=int(cfg.centroid_fallback_radius),
        augment=False,
        seed=int(cfg.seed),
    )
    test_ds = SegPatchDataset(
        image_refs=image_refs,
        allowed_uids=test_uids,
        patch_size=512,
        overlap=80,
        nuclei_min_area=int(cfg.nuclei_min_area),
        centroid_fallback_radius=int(cfg.centroid_fallback_radius),
        augment=False,
        seed=int(cfg.seed),
    )

    # Increase num_workers for faster data loading.
    # NOTE: With sliding-window patches, positives are extremely sparse.
    # If we rely on pure shuffle, many batches contain zero positive pixels,
    # which makes the model collapse to predicting background.
    num_workers = 16

    pos_flags = train_ds.patch_has_positive()
    n_pos = int(sum(1 for v in pos_flags if v))
    n_neg = int(len(pos_flags) - n_pos)
    if n_pos > 0 and n_neg > 0:
        pos_weight = float(n_neg / n_pos)
        weights = torch.tensor([pos_weight if v else 1.0 for v in pos_flags], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        logger.info(
            "CNN_seg(paper) train sampling: balanced (pos=%d neg=%d pos_weight=%.3f)",
            n_pos,
            n_neg,
            float(pos_weight),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        logger.warning(
            "CNN_seg(paper) train sampling: cannot balance (pos=%d neg=%d); using shuffle",
            n_pos,
            n_neg,
        )
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=num_workers, pin_memory=True)

    device_t = _resolve_torch_device(device)
    model = CNNSeg(num_classes=cfg.seg_num_classes, pretrained=True).to(device_t)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4,
    )
    
    # Use weighted loss to prevent collapse to background (Low Recall)
    # and encourage learning positive features.
    if cfg.seg_num_classes == 2:
        # Handle extreme foreground/background imbalance (mitosis is rare).
        # Allow override via env var so users can tune when using oversampling.
        # Example: CABCDS_MFCNN_SEG_POS_WEIGHT=100
        try:
            pos_w = float(os.getenv("CABCDS_MFCNN_SEG_POS_WEIGHT", "500"))
        except ValueError:
            pos_w = 500.0
        pos_w = float(max(1.0, pos_w))
        class_weights = torch.tensor([1.0, pos_w]).to(device_t)
        logging.info("Using Weighted CrossEntropyLoss: %s", str(class_weights))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    global_step = 0

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def _restore_signal_handlers() -> None:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        signal.signal(signal.SIGINT, previous_sigint_handler)

    def _serialize_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in state_dict.items()}

    def _save_checkpoint(
        *,
        path: Path,
        phase: str,
        epoch: int,
        score: float | None,
        state_override: dict[str, torch.Tensor] | None = None,
    ) -> None:
        payload = {
            "model": "CNNSeg",
            "paper_aligned": True,
            "patch_size": 512,
            "overlap": 80,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": int(split_seed)},
            "external_roots": [str(p) for p in external_roots],
            "optimizer": {"name": "SGD", "lr": 1e-4, "momentum": 0.9, "weight_decay": 5e-4},
            "pretrained": True,
            "phase": str(phase),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "score": (float(score) if score is not None else None),
            "state_dict": _serialize_state_dict_to_cpu(state_override if state_override is not None else model.state_dict()),
            "config": cfg.model_dump(mode="json"),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved CNN_seg(paper) checkpoint: %s", str(path))

    logger.info(
        "CNN_seg(paper) periodic checkpoints enabled: last=%s best=%s",
        str(last_checkpoint_path),
        str(best_checkpoint_path),
    )

    def _score_from(test_all: dict[str, float], test_pos: dict[str, float]) -> float:
        if bool(require_positive_metrics):
            return float(test_pos["dice"])
        return float(test_all["dice"])

    def _run_one_epoch(*, loader: DataLoader) -> dict[str, float]:
        nonlocal global_step
        running_loss = 0.0
        steps = 0
        
        # Accumulators for training metrics
        tp = fp = fn = tn = 0
        correct = 0
        total_pixels = 0
        
        for images, masks in loader:
            if bool(stop_requested):
                break
            images = images.to(device_t)
            masks = masks.to(device_t)
            images = _imagenet_normalize_batch(images)

            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out.logits, masks)
            loss.backward()
            optimizer.step()

            current_loss = float(loss.detach().cpu())
            running_loss += current_loss
            steps += 1
            global_step += 1
            
            # Compute online metrics (no_grad)
            with torch.no_grad():
                pred = torch.argmax(out.logits, dim=1)
                correct += int((pred == masks).sum().detach().cpu())
                total_pixels += int(masks.numel())

                pred_pos = pred == 1
                gt_pos = masks == 1
                tp += int((pred_pos & gt_pos).sum().detach().cpu())
                fp += int((pred_pos & (~gt_pos)).sum().detach().cpu())
                fn += int(((~pred_pos) & gt_pos).sum().detach().cpu())
                tn += int(((~pred_pos) & (~gt_pos)).sum().detach().cpu())

            if steps == 1 or steps % 20 == 0:
                current_dice = (2 * tp) / max(1, 2 * tp + fp + fn)
                cur_recall = tp / max(1, tp + fn)
                cur_prec = tp / max(1, tp + fp)
                logger.info(
                    "CNN_seg(paper) train step=%d loss=%.4f dice=%.6f recall=%.4f prec=%.4f (accum)", 
                    steps, current_loss, float(current_dice), float(cur_recall), float(cur_prec)
                )
                
            if int(max_steps_per_epoch) > 0 and steps >= int(max_steps_per_epoch):
                break
        
        avg_loss = float(running_loss / max(1, steps))
        dice = float((2 * tp) / max(1, 2 * tp + fp + fn))
        iou = float(tp / max(1, tp + fp + fn))
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        pixel_acc = float(correct / max(1, total_pixels))
        
        return {
            "loss": avg_loss,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "pixel_acc": pixel_acc
        }

    for epoch in range(int(tune_epochs)):
        model.train()
        train_metrics = _run_one_epoch(loader=train_loader)
        train_loss = train_metrics["loss"]

        if bool(stop_requested):
            logger.warning("CNN_seg(paper) stop requested during tune epoch=%d; saving last and exiting", int(epoch + 1))
            _save_checkpoint(path=last_checkpoint_path, phase="tune", epoch=int(epoch + 1), score=None)
            _restore_signal_handlers()
            return

        do_eval = int(eval_every) <= 1 or ((epoch + 1) % int(eval_every) == 0)
        if not bool(do_eval):
            logger.info(
                "CNN_seg(paper) tune epoch=%d skip eval (eval_every=%d)",
                int(epoch + 1),
                int(eval_every),
            )
            _save_checkpoint(path=last_checkpoint_path, phase="tune", epoch=int(epoch + 1), score=None)
            if bool(stop_requested):
                _restore_signal_handlers()
                return
            continue

        val_all = _eval_seg_loader(
            model=model,
            loader=val_loader,
            device=device_t,
            max_batches=int(eval_max_batches),
            require_positive=False,
            log_every=int(eval_log_every),
            auc_max_pixels=int(eval_auc_max_pixels),
        )
        test_all = _eval_seg_loader(
            model=model,
            loader=test_loader,
            device=device_t,
            max_batches=int(eval_max_batches),
            require_positive=False,
            log_every=int(eval_log_every),
            auc_max_pixels=int(eval_auc_max_pixels),
        )

        if bool(require_positive_metrics):
            val_pos = _eval_seg_loader(
                model=model,
                loader=val_loader,
                device=device_t,
                max_batches=int(eval_max_batches),
                require_positive=True,
                log_every=int(eval_log_every),
                auc_max_pixels=int(eval_auc_max_pixels),
            )
            test_pos = _eval_seg_loader(
                model=model,
                loader=test_loader,
                device=device_t,
                max_batches=int(eval_max_batches),
                require_positive=True,
                log_every=int(eval_log_every),
                auc_max_pixels=int(eval_auc_max_pixels),
            )
        else:
            # Keep schema stable while skipping extra passes.
            val_pos = {"dice": float("nan")}
            test_pos = {"dice": float("nan")}

        score = _score_from(test_all, test_pos)
        logger.info(
            "CNN_seg(paper) tune epoch=%d train_loss=%.4f train_dice=%.4f val_dice=%.4f val_acc=%.4f val_auc=%.4f test_dice=%.4f test_acc=%.4f test_auc=%.4f score=%.4f",
            epoch + 1,
            float(train_loss),
            float(train_metrics["dice"]),
            float(val_all["dice"]),
            float(val_all["pixel_acc"]),
            float(val_all["auc"]),
            float(test_all["dice"]),
            float(test_all["pixel_acc"]),
            float(test_all["auc"]),
            float(score),
        )

        _append_metrics_row(
            metrics_csv,
            {
                "phase": "tune",
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "train_dice": float(train_metrics["dice"]),
                "train_iou": float(train_metrics["iou"]),
                "train_precision": float(train_metrics["precision"]),
                "train_recall": float(train_metrics["recall"]),
                "train_acc": float(train_metrics["pixel_acc"]),
                "val_loss": float(val_all["loss"]),
                "val_dice": float(val_all["dice"]),
                "val_iou": float(val_all["iou"]),
                "val_precision": float(val_all["precision"]),
                "val_recall": float(val_all["recall"]),
                "val_acc": float(val_all["pixel_acc"]),
                "val_auc": float(val_all["auc"]),
                "val_pos_dice": float(val_pos["dice"]),
                "test_loss": float(test_all["loss"]),
                "test_dice": float(test_all["dice"]),
                "test_iou": float(test_all["iou"]),
                "test_precision": float(test_all["precision"]),
                "test_recall": float(test_all["recall"]),
                "test_acc": float(test_all["pixel_acc"]),
                "test_auc": float(test_all["auc"]),
                "test_pos_dice": float(test_pos["dice"]),
                "score": float(score),
            },
        )

        _save_checkpoint(path=last_checkpoint_path, phase="tune", epoch=int(epoch + 1), score=float(score))

        if best_score is None or float(score) > float(best_score):
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            _save_checkpoint(
                path=best_checkpoint_path,
                phase="tune",
                epoch=int(epoch + 1),
                score=float(best_score),
                state_override=best_state,
            )
        else:
            no_improve += 1

        if bool(stop_requested):
            logger.warning("CNN_seg(paper) stop requested after tune epoch=%d; saved last and exiting", int(epoch + 1))
            _restore_signal_handlers()
            return

        if int(early_stop_patience) > 0 and no_improve >= int(early_stop_patience):
            logger.info("CNN_seg(paper) early stop in tune phase (patience=%d)", int(early_stop_patience))
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if bool(stop_requested):
        logger.warning(
            "CNN_seg(paper) stopping due to %s; leaving final checkpoint '%s' untouched",
            str(stop_signal),
            str(checkpoint_path),
        )
        _restore_signal_handlers()
        return

    if int(final_epochs) > 0:
        merged_uids = set(train_uids) | set(val_uids)
        merged_ds = SegPatchDataset(
            image_refs=image_refs,
            allowed_uids=merged_uids,
            patch_size=512,
            overlap=80,
            nuclei_min_area=int(cfg.nuclei_min_area),
            centroid_fallback_radius=int(cfg.centroid_fallback_radius),
            augment=True,
            seed=int(cfg.seed),
        )
        merged_pos_flags = merged_ds.patch_has_positive()
        merged_n_pos = int(sum(1 for v in merged_pos_flags if v))
        merged_n_neg = int(len(merged_pos_flags) - merged_n_pos)
        if merged_n_pos > 0 and merged_n_neg > 0:
            merged_pos_weight = float(merged_n_neg / merged_n_pos)
            merged_weights = torch.tensor(
                [merged_pos_weight if v else 1.0 for v in merged_pos_flags],
                dtype=torch.double,
            )
            merged_sampler = WeightedRandomSampler(
                merged_weights,
                num_samples=len(merged_weights),
                replacement=True,
            )
            logger.info(
                "CNN_seg(paper) final sampling: balanced (pos=%d neg=%d pos_weight=%.3f)",
                merged_n_pos,
                merged_n_neg,
                float(merged_pos_weight),
            )
            merged_loader = DataLoader(
                merged_ds,
                batch_size=int(batch_size),
                sampler=merged_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            logger.warning(
                "CNN_seg(paper) final sampling: cannot balance (pos=%d neg=%d); using shuffle",
                merged_n_pos,
                merged_n_neg,
            )
            merged_loader = DataLoader(merged_ds, batch_size=int(batch_size), shuffle=True, num_workers=num_workers, pin_memory=True)
        logger.info("CNN_seg(paper) final phase: merge val into train (images=%d)", len(merged_uids))

        no_improve = 0
        for epoch in range(int(final_epochs)):
            model.train()
            train_metrics = _run_one_epoch(loader=merged_loader)
            train_loss = train_metrics["loss"]

            if bool(stop_requested):
                logger.warning("CNN_seg(paper) stop requested during final epoch=%d; saving last and exiting", int(epoch + 1))
                _save_checkpoint(path=last_checkpoint_path, phase="final", epoch=int(epoch + 1), score=None)
                _restore_signal_handlers()
                return

            do_eval = int(eval_every) <= 1 or ((epoch + 1) % int(eval_every) == 0)
            if not bool(do_eval):
                logger.info(
                    "CNN_seg(paper) final epoch=%d skip eval (eval_every=%d)",
                    int(epoch + 1),
                    int(eval_every),
                )
                _save_checkpoint(path=last_checkpoint_path, phase="final", epoch=int(epoch + 1), score=None)
                if bool(stop_requested):
                    _restore_signal_handlers()
                    return
                continue

            test_all = _eval_seg_loader(
                model=model,
                loader=test_loader,
                device=device_t,
                max_batches=int(eval_max_batches),
                require_positive=False,
                log_every=int(eval_log_every),
                auc_max_pixels=int(eval_auc_max_pixels),
            )

            if bool(require_positive_metrics):
                test_pos = _eval_seg_loader(
                    model=model,
                    loader=test_loader,
                    device=device_t,
                    max_batches=int(eval_max_batches),
                    require_positive=True,
                    log_every=int(eval_log_every),
                    auc_max_pixels=int(eval_auc_max_pixels),
                )
            else:
                test_pos = {"dice": float("nan")}
            score = _score_from(test_all, test_pos)
            logger.info(
                "CNN_seg(paper) final epoch=%d train_loss=%.4f train_dice=%.4f test_dice=%.4f test_acc=%.4f test_auc=%.4f test_pos_dice=%.4f score=%.4f",
                epoch + 1,
                float(train_loss),
                float(train_metrics["dice"]),
                float(test_all["dice"]),
                float(test_all["pixel_acc"]),
                float(test_all["auc"]),
                float(test_pos["dice"]),
                float(score),
            )
            _append_metrics_row(
                metrics_csv,
                {
                    "phase": "final",
                    "epoch": int(epoch + 1),
                    "global_step": int(global_step),
                    "train_loss": float(train_loss),
                    "train_dice": float(train_metrics["dice"]),
                    "train_iou": float(train_metrics["iou"]),
                    "train_precision": float(train_metrics["precision"]),
                    "train_recall": float(train_metrics["recall"]),
                    "train_acc": float(train_metrics["pixel_acc"]),
                    "test_loss": float(test_all["loss"]),
                    "test_dice": float(test_all["dice"]),
                    "test_iou": float(test_all["iou"]),
                    "test_precision": float(test_all["precision"]),
                    "test_recall": float(test_all["recall"]),
                    "test_acc": float(test_all["pixel_acc"]),
                    "test_auc": float(test_all["auc"]),
                    "test_pos_dice": float(test_pos["dice"]),
                    "score": float(score),
                },
            )

            _save_checkpoint(path=last_checkpoint_path, phase="final", epoch=int(epoch + 1), score=float(score))

            if best_score is None or float(score) > float(best_score):
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                _save_checkpoint(
                    path=best_checkpoint_path,
                    phase="final",
                    epoch=int(epoch + 1),
                    score=float(best_score),
                    state_override=best_state,
                )
            else:
                no_improve += 1

            if int(early_stop_patience) > 0 and no_improve >= int(early_stop_patience):
                logger.info("CNN_seg(paper) early stop in final phase (patience=%d)", int(early_stop_patience))
                break

            if bool(stop_requested):
                logger.warning("CNN_seg(paper) stop requested after final epoch=%d; saved last and exiting", int(epoch + 1))
                _restore_signal_handlers()
                return

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": "CNNSeg",
            "paper_aligned": True,
            "patch_size": 512,
            "overlap": 80,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": int(split_seed)},
            "external_roots": [str(p) for p in external_roots],
            "optimizer": {"name": "SGD", "lr": 1e-4, "momentum": 0.9, "weight_decay": 5e-4},
            "pretrained": True,
            "state_dict": model.state_dict(),
            "config": cfg.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    logger.info("Saved CNN_seg(paper) checkpoint: %s", str(checkpoint_path))

    _restore_signal_handlers()


@torch.no_grad()
def _eval_det_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    threshold: float | None = None,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    kept = 0
    
    all_y_true = []
    all_y_score = []

    for images, labels in loader:
        if max_batches > 0 and kept >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        images = _imagenet_normalize_batch(images)

        out = model(images)
        loss = criterion(out, labels)
        loss_sum += float(loss.detach().cpu())

        # For AUC (class 1 prob)
        probs = F.softmax(out, dim=1)[:, 1]
        all_y_score.append(probs.detach().cpu().numpy())
        all_y_true.append(labels.detach().cpu().numpy())

        preds = torch.argmax(out, dim=1)
        correct += int((preds == labels).sum().cpu())
        total += int(labels.numel())
        
        pred_pos = preds == 1
        gt_pos = labels == 1
        tp += int((pred_pos & gt_pos).sum().cpu())
        fp += int((pred_pos & (~gt_pos)).sum().cpu())
        fn += int(((~pred_pos) & gt_pos).sum().cpu())
        tn += int(((~pred_pos) & (~gt_pos)).sum().cpu())

        kept += 1

    avg_loss = float(loss_sum / max(1, kept))
    acc = float(correct / max(1, total))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2 * precision * recall) / max(1e-12, precision + recall))

    auc = 0.0
    ap = 0.0
    best_f1 = 0.0
    best_thr = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_tp = best_fp = best_fn = best_tn = 0.0

    thr_used = float(threshold) if threshold is not None else float("nan")
    thr_precision = 0.0
    thr_recall = 0.0
    thr_f1 = 0.0
    thr_tp = thr_fp = thr_fn = thr_tn = 0.0

    if all_y_true:
        try:
            y_true_flat = np.concatenate(all_y_true)
            y_score_flat = np.concatenate(all_y_score)
            if len(np.unique(y_true_flat)) > 1:
                auc = float(roc_auc_score(y_true_flat, y_score_flat))
                ap = float(average_precision_score(y_true_flat, y_score_flat))

                best = _best_f1_threshold(y_true=y_true_flat, y_score=y_score_flat)
                best_f1 = float(best["best_f1"])
                best_thr = float(best["best_thr"])
                best_precision = float(best["best_precision"])
                best_recall = float(best["best_recall"])

                preds_best = (y_score_flat >= best_thr).astype(np.int64)
                best_tp = float(np.sum((preds_best == 1) & (y_true_flat == 1)))
                best_fp = float(np.sum((preds_best == 1) & (y_true_flat == 0)))
                best_fn = float(np.sum((preds_best == 0) & (y_true_flat == 1)))
                best_tn = float(np.sum((preds_best == 0) & (y_true_flat == 0)))

                if threshold is not None:
                    preds_thr = (y_score_flat >= float(threshold)).astype(np.int64)
                    thr_tp = float(np.sum((preds_thr == 1) & (y_true_flat == 1)))
                    thr_fp = float(np.sum((preds_thr == 1) & (y_true_flat == 0)))
                    thr_fn = float(np.sum((preds_thr == 0) & (y_true_flat == 1)))
                    thr_tn = float(np.sum((preds_thr == 0) & (y_true_flat == 0)))
                    thr_precision = float(thr_tp / max(1e-12, thr_tp + thr_fp))
                    thr_recall = float(thr_tp / max(1e-12, thr_tp + thr_fn))
                    thr_f1 = float((2 * thr_precision * thr_recall) / max(1e-12, thr_precision + thr_recall))
        except Exception:
            pass

    model.train()
    return {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_f1": float(best_f1),
        "best_thr": float(best_thr),
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
        "best_tp": float(best_tp),
        "best_fp": float(best_fp),
        "best_fn": float(best_fn),
        "best_tn": float(best_tn),
        "thr_used": float(thr_used),
        "thr_precision": float(thr_precision),
        "thr_recall": float(thr_recall),
        "thr_f1": float(thr_f1),
        "thr_tp": float(thr_tp),
        "thr_fp": float(thr_fp),
        "thr_fn": float(thr_fn),
        "thr_tn": float(thr_tn),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _train_det_paper(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    metrics_csv: Path,
    batch_size: int,
    tune_epochs: int,
    final_epochs: int,
    device: str,
    split_seed: int,
    early_stop_patience: int = 5,
    eval_every: int = 1,
    eval_max_batches: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    balanced_sampler: bool = True,
    class_weight_mode: str = "auto",
    pos_weight: float = 0.0,
    focal_gamma: float = 0.0,
    select_metric: str = "ap",
    hnm_enabled: bool = False,
    hnm_start_epoch: int = 2,
    hnm_every: int = 1,
    hnm_topk: int = 2000,
    hnm_boost: float = 3.0,
    hnm_max_batches: int = 0,
    resume: bool = False,
) -> None:
    """Paper-aligned CNN_det training: 60/20/20 case split.

    - Reads 80x80 candidate patch index.
    - Splits by case_id.
    - Tuning phase: Train on 60%, Val on 20% (save best).
    - Final phase: Train on 80% (60+20), monitor on 20% test (early stop).
    - Model: AlexNet (pretrained), resized inputs to 227x227.
    - Hyperparameters: SGD lr=0.0001, momentum=0.9, weight_decay=0.0005.

    Notes:
        `eval_every` allows skipping val/test evaluation to speed up training.
        Default is 1 (evaluate every epoch, paper-aligned).
    """

    index_csv = Path(index_csv)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_last.pt")
    tune_best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_tune_best.pt")
    final_best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_final_best.pt")

    metrics_csv = Path(metrics_csv)
    _append_metrics_row(metrics_csv, {"event": "start", "device": device})

    cfg = load_mfc_cnn_config()

    stop_requested = False

    def _request_stop(signum: int, _frame: object | None) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.warning(
            "Stop requested (signal=%s); will checkpoint and exit at next safe point.",
            int(signum),
        )

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _request_stop)
        except Exception:
            # Best-effort; some environments disallow signal handlers.
            pass

    def _save_det_checkpoint(
        path: Path,
        *,
        phase: str,
        epoch: int,
        best_score: float | None,
        best_val_thr: float,
        best_test_score: float | None,
        no_improve: int,
        best_state_dict: dict[str, torch.Tensor] | None,
        include_optimizer: bool,
        global_step: int,
    ) -> None:
        payload: dict[str, object] = {
            "model": "CNNDet",
            "paper_aligned": True,
            "phase": str(phase),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_score": float(best_score) if best_score is not None else None,
            "best_val_thr": float(best_val_thr),
            "best_test_score": float(best_test_score) if best_test_score is not None else None,
            "no_improve": int(no_improve),
            "best_state_dict": (
                {k: v.detach().cpu() for k, v in best_state_dict.items()}
                if best_state_dict is not None
                else None
            ),
            "config": cfg.model_dump(mode="json"),
        }
        if bool(include_optimizer):
            payload["optimizer_state"] = optimizer.state_dict()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved CNN_det(paper) checkpoint: %s", str(path))

    # Load and split
    rows = read_det_patch_index(index_csv)
    case_ids = sorted(list(set(r.case_id for r in rows if r.case_id)))
    rng = random.Random(int(split_seed))
    rng.shuffle(case_ids)

    n_total = len(case_ids)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    
    train_cases = set(case_ids[:n_train])
    val_cases = set(case_ids[n_train:n_train+n_val])
    test_cases = set(case_ids[n_train+n_val:])
    
    train_rows = [r for r in rows if r.case_id in train_cases]
    # Do not evaluate on GT-forced patches; keep val/test reflecting pure CNN_seg candidate proposals.
    val_rows = [r for r in rows if r.case_id in val_cases and int(getattr(r, "is_gt_forced", 0)) == 0]
    test_rows = [r for r in rows if r.case_id in test_cases and int(getattr(r, "is_gt_forced", 0)) == 0]
    
    logger.info(
        "CNN_det split (cases): total=%d train=%d val=%d test=%d seed=%d",
        n_total, len(train_cases), len(val_cases), len(test_cases), int(split_seed)
    )
    logger.info(
        "CNN_det split (patches): total=%d train=%d val=%d test=%d",
        len(rows), len(train_rows), len(val_rows), len(test_rows)
    )

    # Datasets with normalization=False because we normalize in loop
    train_ds = PreparedMitosisDetectionPatchDataset(rows=train_rows, output_size=227, normalize_imagenet=False)
    val_ds = PreparedMitosisDetectionPatchDataset(rows=val_rows, output_size=227, normalize_imagenet=False)
    test_ds = PreparedMitosisDetectionPatchDataset(rows=test_rows, output_size=227, normalize_imagenet=False)

    num_workers_i = max(0, int(num_workers))
    loader_kwargs: dict[str, object] = {
        "num_workers": int(num_workers_i),
    }
    if int(num_workers_i) > 0:
        # Reduce process spawn overhead across epochs.
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    device_t = _resolve_torch_device(device)

    # Imbalance handling: optional balanced sampling + optional loss weighting.
    train_labels = [int(r.label) for r in train_rows]
    n_pos = int(sum(1 for y in train_labels if y == 1))
    n_neg = int(len(train_labels) - n_pos)
    logger.info(
        "CNN_det class balance (train split): pos=%d neg=%d pos_rate=%.4f",
        int(n_pos),
        int(n_neg),
        float(n_pos / max(1, len(train_labels))),
    )

    select_metric_i = str(select_metric).lower().strip()
    if select_metric_i not in {"ap", "auc", "best_f1", "acc"}:
        raise ValueError("--det-paper-select-metric must be one of: ap, auc, best_f1, acc")

    use_sampler = bool(balanced_sampler) or bool(hnm_enabled)
    if use_sampler:
        # Base weights: class balancing (optional) + room for HNM boosting.
        if bool(balanced_sampler) and n_pos > 0 and n_neg > 0:
            w_pos = (len(train_labels) / max(1, n_pos))
            w_neg = (len(train_labels) / max(1, n_neg))
        else:
            w_pos = 1.0
            w_neg = 1.0

        base_weights = torch.tensor([w_pos if y == 1 else w_neg for y in train_labels], dtype=torch.double)
        weights_train = base_weights.clone()

        def _make_train_loader(*, weights: torch.Tensor) -> DataLoader:
            sampler_local = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler_local, shuffle=False, **loader_kwargs)

        train_loader = _make_train_loader(weights=weights_train)
        logger.info(
            "CNN_det using sampler (balanced=%s hnm=%s w_pos=%.3f w_neg=%.3f)",
            str(bool(balanced_sampler)),
            str(bool(hnm_enabled)),
            float(w_pos),
            float(w_neg),
        )
    else:
        base_weights = torch.tensor([], dtype=torch.double)
        weights_train = torch.tensor([], dtype=torch.double)
        _make_train_loader = None  # type: ignore[assignment]
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
    model = CNNDet(num_classes=2, pretrained=True).to(device_t)

    # Paper hypers: LR=0.0001, Momentum=0.9, WD=0.0005
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    class_weight_mode_i = str(class_weight_mode).lower().strip()
    if class_weight_mode_i not in {"none", "auto", "manual"}:
        raise ValueError("--det-paper-class-weight-mode must be one of: none, auto, manual")

    class_weights_t: torch.Tensor | None = None
    if class_weight_mode_i == "manual":
        if float(pos_weight) <= 0:
            raise ValueError("--det-paper-pos-weight must be > 0 when --det-paper-class-weight-mode=manual")
        class_weights_t = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32, device=device_t)
        logger.info("CNN_det using manual class weights: w0=1.0 w1=%.3f", float(pos_weight))
    elif class_weight_mode_i == "auto":
        if n_pos > 0 and n_neg > 0:
            w1 = float(n_neg / max(1, n_pos))
            class_weights_t = torch.tensor([1.0, w1], dtype=torch.float32, device=device_t)
            logger.info("CNN_det using auto class weights: w0=1.0 w1=%.3f (neg/pos)", float(w1))
        else:
            logger.warning("CNN_det auto class weights disabled (pos=%d neg=%d)", int(n_pos), int(n_neg))

    focal_gamma_i = float(focal_gamma)
    if focal_gamma_i < 0:
        raise ValueError("--det-paper-focal-gamma must be >= 0")
    if focal_gamma_i > 0:
        logger.info("CNN_det using focal loss (gamma=%.3f)", float(focal_gamma_i))

    def _criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if focal_gamma_i <= 0:
            return F.cross_entropy(logits, labels, weight=class_weights_t)
        ce = F.cross_entropy(logits, labels, weight=class_weights_t, reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        loss = ((1.0 - pt).clamp(min=0.0, max=1.0) ** focal_gamma_i) * ce
        return loss.mean()

    best_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_val_thr = 0.5
    global_step = 0

    resume_phase: str | None = None
    resume_epoch_done: int = 0
    best_test_score: float | None = None
    no_improve = 0

    if bool(resume):
        resume_path: Path | None = None
        for p in (last_checkpoint_path, checkpoint_path):
            try:
                if Path(p).exists():
                    resume_path = Path(p)
                    break
            except Exception:
                continue
        if resume_path is not None:
            logger.info("Resuming CNN_det(paper) from checkpoint: %s", str(resume_path))
            payload = torch.load(resume_path, map_location="cpu")
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            opt_state = payload.get("optimizer_state")
            if isinstance(opt_state, dict):
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as e:
                    logger.warning("Failed to restore optimizer state (continuing): %s", str(e))

            resume_phase = str(payload.get("phase") or "tune")
            resume_epoch_done = int(payload.get("epoch") or 0)
            global_step = int(payload.get("global_step") or 0)
            best_score_raw = payload.get("best_score")
            best_score = float(best_score_raw) if best_score_raw is not None else None
            best_val_thr = float(payload.get("best_val_thr") or 0.5)
            best_test_raw = payload.get("best_test_score")
            best_test_score = float(best_test_raw) if best_test_raw is not None else None
            no_improve = int(payload.get("no_improve") or 0)

            best_state_raw = payload.get("best_state_dict")
            if isinstance(best_state_raw, dict):
                try:
                    best_state = {str(k): v for k, v in best_state_raw.items() if isinstance(v, torch.Tensor)}
                except Exception:
                    best_state = None

    current_phase = "tune"
    current_epoch = 0

    def _run_one_epoch(*, loader: DataLoader) -> dict[str, float]:
        nonlocal global_step
        running_loss = 0.0
        steps = 0
        correct = 0
        total = 0
        tp = fp = fn = tn = 0

        for images, labels in loader:
            if bool(stop_requested):
                break
            images = images.to(device_t)
            labels = labels.to(device_t)
            images = _imagenet_normalize_batch(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = _criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            steps += 1
            global_step += 1
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().cpu())
                total += int(labels.numel())
                pred_pos = preds == 1
                gt_pos = labels == 1
                tp += int((pred_pos & gt_pos).sum().cpu())
                fp += int((pred_pos & (~gt_pos)).sum().cpu())
                fn += int(((~pred_pos) & gt_pos).sum().cpu())
                tn += int(((~pred_pos) & (~gt_pos)).sum().cpu())
            
            if steps == 1 or steps % 50 == 0:
                logger.info("CNN_det train step=%d loss=%.4f acc=%.3f", steps, float(loss), float(correct/max(1, total)))

        avg = running_loss / max(1, steps)
        acc = float(correct / max(1, total))
        dice = float((2 * tp) / max(1, 2 * tp + fp + fn))
        return {"loss": avg, "acc": acc, "dice": dice}

    def _metric_score(metrics: dict[str, float]) -> float:
        if select_metric_i == "best_f1":
            return float(metrics.get("best_f1", 0.0))
        return float(metrics.get(select_metric_i, 0.0))

    def _update_hnm_weights(*, ds: PreparedMitosisDetectionPatchDataset, labels_list: list[int], base_w: torch.Tensor) -> torch.Tensor:
        if not bool(hnm_enabled):
            return base_w
        if int(hnm_topk) <= 0 or float(hnm_boost) <= 1.0:
            return base_w

        score_loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
        hard: list[tuple[float, int]] = []
        seen_batches = 0
        seen = 0
        model.eval()
        with torch.no_grad():
            for images, labels in score_loader:
                if int(hnm_max_batches) > 0 and int(seen_batches) >= int(hnm_max_batches):
                    break
                seen_batches += 1

                images = images.to(device_t)
                labels_t = labels.to(device_t)
                images = _imagenet_normalize_batch(images)
                logits = model(images)
                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                labels_np = labels_t.detach().cpu().numpy().astype(np.int64)

                for i in range(int(labels_np.size)):
                    if int(labels_np[i]) == 0:
                        hard.append((float(probs[i]), int(seen + i)))
                seen += int(labels_np.size)

        model.train()
        if not hard:
            return base_w

        hard.sort(key=lambda t: float(t[0]), reverse=True)
        topk = min(int(hnm_topk), len(hard))
        hard_idx = [int(idx) for _, idx in hard[:topk] if 0 <= int(idx) < len(labels_list) and int(labels_list[int(idx)]) == 0]
        if not hard_idx:
            return base_w

        out_w = base_w.clone()
        out_w[torch.tensor(hard_idx, dtype=torch.long)] *= float(hnm_boost)
        logger.info("CNN_det HNM updated: boosted %d hard negatives (topk=%d boost=%.2f)", len(hard_idx), int(topk), float(hnm_boost))
        return out_w

    eval_every_i = max(1, int(eval_every))
    logger.info(
        "Starting Tuning Phase (Train on 60%%, Val on 20%%); eval_every=%d eval_max_batches=%d num_workers=%d",
        int(eval_every_i),
        int(eval_max_batches),
        int(num_workers_i),
    )
    tune_start = 0
    if str(resume_phase or "").lower().strip() == "tune" and int(resume_epoch_done) > 0:
        tune_start = min(int(resume_epoch_done), int(tune_epochs))
        logger.info("Resuming tuning phase from epoch %d", int(tune_start) + 1)

    try:
        for epoch in range(int(tune_start), int(tune_epochs)):
            if bool(stop_requested):
                break

            current_phase = "tune"
            current_epoch = int(epoch + 1)

            model.train()
            train_m = _run_one_epoch(loader=train_loader)

            if bool(stop_requested):
                _save_det_checkpoint(
                    last_checkpoint_path,
                    phase="tune",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=True,
                    global_step=int(global_step),
                )
                break

            if (epoch + 1) % int(eval_every_i) != 0:
                logger.info(
                    "CNN_det tune epoch=%d train_loss=%.4f train_acc=%.3f (val skipped; eval_every=%d)",
                    epoch + 1,
                    train_m["loss"],
                    train_m["acc"],
                    int(eval_every_i),
                )
                _append_metrics_row(
                    metrics_csv,
                    {
                        "phase": "tune",
                        "epoch": epoch + 1,
                        "train_loss": train_m["loss"],
                        "val_loss": float("nan"),
                        "val_acc": float("nan"),
                        "val_auc": float("nan"),
                        "val_ap": float("nan"),
                        "val_best_f1": float("nan"),
                        "val_best_thr": float("nan"),
                        "val_skipped": True,
                    },
                )
                _save_det_checkpoint(
                    last_checkpoint_path,
                    phase="tune",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=True,
                    global_step=int(global_step),
                )
                continue

            val_m = _eval_det_loader(
                model=model,
                loader=val_loader,
                device=device_t,
                max_batches=int(eval_max_batches),
            )

            score = _metric_score(val_m)
            logger.info(
                "CNN_det tune epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f val_f1=%.3f val_auc=%.3f val_ap=%.3f val_best_f1=%.3f thr=%.3f",
                epoch + 1,
                train_m["loss"],
                train_m["acc"],
                val_m["loss"],
                val_m["acc"],
                val_m["f1"],
                val_m["auc"],
                val_m.get("ap", 0.0),
                val_m.get("best_f1", 0.0),
                val_m.get("best_thr", 0.5),
            )

            _append_metrics_row(
                metrics_csv,
                {
                    "phase": "tune",
                    "epoch": epoch + 1,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "val_acc": val_m["acc"],
                    "val_auc": val_m["auc"],
                    "val_ap": val_m.get("ap", 0.0),
                    "val_best_f1": val_m.get("best_f1", 0.0),
                    "val_best_thr": val_m.get("best_thr", 0.5),
                    "val_skipped": False,
                },
            )

            if best_score is None or float(score) > float(best_score):
                best_score = score
                best_val_thr = float(val_m.get("best_thr", 0.5))
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                _save_det_checkpoint(
                    tune_best_checkpoint_path,
                    phase="tune",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=False,
                    global_step=int(global_step),
                )

            if (
                bool(hnm_enabled)
                and int(epoch + 1) >= int(hnm_start_epoch)
                and (int(epoch + 1) % max(1, int(hnm_every)) == 0)
            ):
                if use_sampler and _make_train_loader is not None:
                    weights_train = _update_hnm_weights(ds=train_ds, labels_list=train_labels, base_w=base_weights)
                    train_loader = _make_train_loader(weights=weights_train)

            _save_det_checkpoint(
                last_checkpoint_path,
                phase="tune",
                epoch=int(epoch + 1),
                best_score=best_score,
                best_val_thr=float(best_val_thr),
                best_test_score=best_test_score,
                no_improve=int(no_improve),
                best_state_dict=best_state,
                include_optimizer=True,
                global_step=int(global_step),
            )
    except KeyboardInterrupt:
        stop_requested = True
        logger.warning("KeyboardInterrupt received; saving last checkpoint and exiting.")
        _save_det_checkpoint(
            last_checkpoint_path,
            phase=str(current_phase),
            epoch=int(current_epoch),
            best_score=best_score,
            best_val_thr=float(best_val_thr),
            best_test_score=best_test_score,
            no_improve=int(no_improve),
            best_state_dict=best_state,
            include_optimizer=True,
            global_step=int(global_step),
        )
        return
    
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model from tuning phase (%s=%.4f best_val_thr=%.3f)", select_metric_i, float(best_score or 0.0), float(best_val_thr))
    
    if bool(stop_requested):
        logger.info("Stop requested; skipping final phase.")
        return

    logger.info("Starting Final Phase (Train on 80%, Monitor on 20% Test)")
    
    final_rows = train_rows + val_rows
    final_ds = PreparedMitosisDetectionPatchDataset(rows=final_rows, output_size=227, normalize_imagenet=False)
    use_sampler_f = bool(balanced_sampler) or bool(hnm_enabled)
    if bool(use_sampler_f):
        final_labels = [int(r.label) for r in final_rows]
        n_pos_f = int(sum(1 for y in final_labels if y == 1))
        n_neg_f = int(len(final_labels) - n_pos_f)
        if bool(balanced_sampler) and n_pos_f > 0 and n_neg_f > 0:
            w_pos_f = (len(final_labels) / max(1, n_pos_f))
            w_neg_f = (len(final_labels) / max(1, n_neg_f))
        else:
            w_pos_f = 1.0
            w_neg_f = 1.0
        base_weights_f = torch.tensor([w_pos_f if y == 1 else w_neg_f for y in final_labels], dtype=torch.double)
        weights_final = base_weights_f.clone()

        def _make_final_loader(*, weights: torch.Tensor) -> DataLoader:
            sampler_local = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(final_ds, batch_size=int(batch_size), sampler=sampler_local, shuffle=False, **loader_kwargs)

        final_loader = _make_final_loader(weights=weights_final)
    else:
        base_weights_f = torch.tensor([], dtype=torch.double)
        weights_final = torch.tensor([], dtype=torch.double)
        _make_final_loader = None  # type: ignore[assignment]
        final_loader = DataLoader(final_ds, batch_size=int(batch_size), shuffle=True, **loader_kwargs)
    
    # Resume directly into final phase if requested.
    final_start = 0
    if str(resume_phase or "").lower().strip() == "final" and int(resume_epoch_done) > 0:
        final_start = min(int(resume_epoch_done), int(final_epochs))
        logger.info("Resuming final phase from epoch %d", int(final_start) + 1)

    if best_test_score is None:
        best_test_score = 0.0

    try:
        for epoch in range(int(final_start), int(final_epochs)):
            if bool(stop_requested):
                break

            current_phase = "final"
            current_epoch = int(epoch + 1)

            model.train()
            train_m = _run_one_epoch(loader=final_loader)

            if bool(stop_requested):
                _save_det_checkpoint(
                    last_checkpoint_path,
                    phase="final",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=True,
                    global_step=int(global_step),
                )
                break

            if (epoch + 1) % int(eval_every_i) != 0:
                logger.info(
                    "CNN_det final epoch=%d train_loss=%.4f train_acc=%.3f (test skipped; eval_every=%d)",
                    epoch + 1,
                    train_m["loss"],
                    train_m["acc"],
                    int(eval_every_i),
                )
                _append_metrics_row(
                    metrics_csv,
                    {
                        "phase": "final",
                        "epoch": epoch + 1,
                        "train_loss": train_m["loss"],
                        "test_loss": float("nan"),
                        "test_acc": float("nan"),
                        "test_auc": float("nan"),
                        "test_ap": float("nan"),
                        "test_best_f1": float("nan"),
                        "test_best_thr": float("nan"),
                        "test_thr_used": float("nan"),
                        "test_f1_at_val_thr": float("nan"),
                        "test_precision_at_val_thr": float("nan"),
                        "test_recall_at_val_thr": float("nan"),
                        "test_skipped": True,
                    },
                )
                _save_det_checkpoint(
                    last_checkpoint_path,
                    phase="final",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=True,
                    global_step=int(global_step),
                )
                continue

            test_m = _eval_det_loader(
                model=model,
                loader=test_loader,
                device=device_t,
                max_batches=int(eval_max_batches),
                threshold=float(best_val_thr),
            )

            score = _metric_score(test_m)

            logger.info(
                "CNN_det final epoch=%d train_loss=%.4f train_acc=%.3f test_loss=%.4f test_acc=%.3f test_f1=%.3f test_auc=%.3f test_ap=%.3f test_best_f1=%.3f thr=%.3f",
                epoch + 1,
                train_m["loss"],
                train_m["acc"],
                test_m["loss"],
                test_m["acc"],
                test_m["f1"],
                test_m["auc"],
                test_m.get("ap", 0.0),
                test_m.get("best_f1", 0.0),
                test_m.get("best_thr", 0.5),
            )

            _append_metrics_row(
                metrics_csv,
                {
                    "phase": "final",
                    "epoch": epoch + 1,
                    "train_loss": train_m["loss"],
                    "test_loss": test_m["loss"],
                    "test_acc": test_m["acc"],
                    "test_auc": test_m["auc"],
                    "test_ap": test_m.get("ap", 0.0),
                    "test_best_f1": test_m.get("best_f1", 0.0),
                    "test_best_thr": test_m.get("best_thr", 0.5),
                    "test_thr_used": test_m.get("thr_used", float("nan")),
                    "test_f1_at_val_thr": test_m.get("thr_f1", 0.0),
                    "test_precision_at_val_thr": test_m.get("thr_precision", 0.0),
                    "test_recall_at_val_thr": test_m.get("thr_recall", 0.0),
                    "test_skipped": False,
                },
            )

            if float(score) > float(best_test_score):
                best_test_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0

                _save_det_checkpoint(
                    final_best_checkpoint_path,
                    phase="final",
                    epoch=int(epoch + 1),
                    best_score=best_score,
                    best_val_thr=float(best_val_thr),
                    best_test_score=best_test_score,
                    no_improve=int(no_improve),
                    best_state_dict=best_state,
                    include_optimizer=False,
                    global_step=int(global_step),
                )
            else:
                no_improve += 1

            if (
                bool(hnm_enabled)
                and int(epoch + 1) >= int(hnm_start_epoch)
                and (int(epoch + 1) % max(1, int(hnm_every)) == 0)
            ):
                if bool(use_sampler_f) and _make_final_loader is not None:
                    weights_final = _update_hnm_weights(ds=final_ds, labels_list=final_labels, base_w=base_weights_f)
                    final_loader = _make_final_loader(weights=weights_final)

            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                logger.info("Early stopping triggered at final epoch %d", epoch + 1)
                break

            _save_det_checkpoint(
                last_checkpoint_path,
                phase="final",
                epoch=int(epoch + 1),
                best_score=best_score,
                best_val_thr=float(best_val_thr),
                best_test_score=best_test_score,
                no_improve=int(no_improve),
                best_state_dict=best_state,
                include_optimizer=True,
                global_step=int(global_step),
            )
    except KeyboardInterrupt:
        stop_requested = True
        logger.warning("KeyboardInterrupt received; saving last checkpoint and exiting.")
        _save_det_checkpoint(
            last_checkpoint_path,
            phase=str(current_phase),
            epoch=int(current_epoch),
            best_score=best_score,
            best_val_thr=float(best_val_thr),
            best_test_score=best_test_score,
            no_improve=int(no_improve),
            best_state_dict=best_state,
            include_optimizer=True,
            global_step=int(global_step),
        )
        return
             
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": "CNNDet", 
            "paper_aligned": True,
            "state_dict": model.state_dict(),
            "config": cfg.model_dump(mode="json")
        },
        checkpoint_path
    )
    logger.info("Saved CNN_det(paper) checkpoint: %s", str(checkpoint_path))


def _resolve_torch_device(device: str) -> torch.device:
    """Resolve a torch device string including Ascend NPUs.

    Supported values:
    - cpu
    - cuda (NVIDIA)
    - mps (Apple)
    - npu (Huawei Ascend via torch_npu)
    """

    device = str(device).lower().strip()
    
    # Handle any device string starting with npu (e.g. npu, npu:0, npu:1)
    if device.startswith("npu"):
        try:
            # Import registers the `npu` device type in torch.
            import torch_npu  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"Requested device='{device}' but torch_npu could not be imported. "
                "This usually means the Ascend runtime libraries are not on LD_LIBRARY_PATH. "
                "Try: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` in the same shell, then rerun. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
            
        if device == "npu":
             # Legacy behavior: default to device 0
            npu_mod = getattr(torch, "npu", None)
            if npu_mod is not None and hasattr(npu_mod, "set_device"):
                try:
                    npu_mod.set_device(0)
                except Exception:
                    pass
            return torch.device("npu:0")
            
        # If specific NPU requested (e.g. npu:1), let torch.device handle the string
        # but we needed the import above.
        return torch.device(device)

    return torch.device(device)


def _imagenet_normalize_batch(images_bchw: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a batch tensor in BCHW format."""

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images_bchw.dtype, device=images_bchw.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images_bchw.dtype, device=images_bchw.device)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    return (images_bchw - mean) / std


def _run_smoke(*, batch_size: int, seg_length: int, det_length: int, device: str) -> None:
    cfg = load_mfc_cnn_config()
    seg_loader, det_loader = build_default_mf_cnn_train_loaders(
        cfg,
        batch_size=batch_size,
        num_workers=0,
        seg_length=seg_length,
        det_length=det_length,
    )

    device_t = _resolve_torch_device(device)

    seg_model = CNNSeg(num_classes=cfg.seg_num_classes, pretrained=False).to(device_t)
    det_model = CNNDet(num_classes=cfg.det_num_classes, pretrained=False).to(device_t)

    seg_criterion = nn.CrossEntropyLoss()
    det_criterion = nn.CrossEntropyLoss()

    seg_batch = next(iter(seg_loader))
    seg_images, seg_masks = seg_batch[0].to(device_t), seg_batch[1].to(device_t)
    seg_out = seg_model(seg_images)
    seg_loss = seg_criterion(seg_out.logits, seg_masks)
    seg_loss.backward()
    logger.info(
        "CNNSeg smoke ok: images=%s masks=%s loss=%.4f",
        tuple(seg_images.shape),
        tuple(seg_masks.shape),
        float(seg_loss.detach().cpu()),
    )

    det_batch = next(iter(det_loader))
    det_images, det_labels = det_batch[0].to(device_t), det_batch[1].to(device_t)
    det_logits = det_model(det_images)
    det_loss = det_criterion(det_logits, det_labels)
    det_loss.backward()
    logger.info(
        "CNNDet smoke ok: images=%s labels=%s loss=%.4f",
        tuple(det_images.shape),
        tuple(det_labels.shape),
        float(det_loss.detach().cpu()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MF-CNN utilities")
    parser.add_argument("--smoke", action="store_true", help="Run a quick loader/model forward+backward smoke check")

    parser.add_argument(
        "--prepare-global",
        action="store_true",
        help="Extract CNN_global training patches from WSIs and write an index CSV",
    )

    parser.add_argument(
        "--global-run-id",
        type=str,
        default=None,
        help=(
            "Run id used to build default output paths for --prepare-global under "
            "output/mf_cnn/CNN_global/runs/<run_id>/global_patches/. "
            "If omitted, a timestamped id is generated."
        ),
    )

    parser.add_argument(
        "--global-wsi-dir",
        type=str,
        default=None,
        help=(
            "Optional override WSI directory for --prepare-global. "
            "Defaults to config.tupac_train_dir (dataset/tupac16/train)."
        ),
    )
    parser.add_argument(
        "--global-slide-glob",
        type=str,
        default=None,
        help=(
            "Optional glob to select slide files within --global-wsi-dir. "
            "Defaults to 'TUPAC-TR-*.svs' when using the TUPAC train dir; otherwise '*.svs'."
        ),
    )
    parser.add_argument(
        "--global-labels-csv",
        type=str,
        default=None,
        help=(
            "Optional slide-level labels CSV for --prepare-global. "
            "If omitted, uses the TUPAC ground_truth.csv row-index mapping. "
            "This repo also includes dataset/tupac16/train/ground_truth_with_groups.csv (header: group,label,score)."
        ),
    )
    parser.add_argument(
        "--global-label-id-col",
        type=str,
        default=None,
        help="Optional column name in --global-labels-csv containing slide ids (auto-detected if omitted).",
    )
    parser.add_argument(
        "--global-label-col",
        type=str,
        default=None,
        help="Optional column name in --global-labels-csv containing integer class labels (auto-detected if omitted).",
    )
    
    parser.add_argument(
        "--generate-candidates",
        action="store_true",
        help="Run CNN_seg on TUPAC train WSIs (within ROIs) to generate candidate centroids for CNN_global",
    )
    parser.add_argument(
        "--candidates-out-dir",
        type=str,
        default=None,
        help="Output directory for generated candidate CSVs (defaults to output/candidates)",
    )

    parser.add_argument(
        "--train-global",
        action="store_true",
        help="Train CNN_global from a prepared patch index CSV",
    )

    parser.add_argument("--train-seg", action="store_true", help="Train CNN_seg (VGG16-FCN) on mitosis auxiliary zips")
    parser.add_argument(
        "--train-seg-paper",
        action="store_true",
        help=(
            "Paper-aligned CNN_seg training: TUPAC aux + MITOS12+MITOS14, "
            "60/20/20 image split, 512 patches with overlap 80, "
            "SGD lr=1e-4 momentum=0.9 wd=5e-4, test-set early stopping, CSV metrics"
        ),
    )
    parser.add_argument("--train-det", action="store_true", help="Train CNN_det (AlexNet binary) on mitosis auxiliary zips")
    parser.add_argument(
        "--prepare-det",
        action="store_true",
        help="Prepare CNN_det candidate patches using a trained CNN_seg checkpoint (paper-aligned)",
    )
    parser.add_argument("--out-root", type=str, default=None, help="Output root for prepared data (defaults under config.output_dir)")
    parser.add_argument("--index-csv", type=str, default=None, help="Index CSV path (defaults under config.output_dir)")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.2)
    parser.add_argument(
        "--roi-csv-dir",
        type=str,
        default=None,
        help=(
            "Directory containing ROI CSV files named like 'TUPAC-TR-123-ROI.csv' with columns x,y,w,h. "
            "If provided and a matching file exists for a slide, CNN_global patches are extracted inside those ROIs."
        ),
    )
    parser.add_argument(
        "--roi-report-csv",
        type=str,
        default=None,
        help=(
            "Path to ROI-Selector stage-two report CSV (e.g., output/roi_selector/outputs/reports/stage_two_roi_selection.csv). "
            "If provided (and --roi-csv-dir is not), CNN_global patches are extracted inside the top-N ROIs from this report."
        ),
    )
    parser.add_argument(
        "--roi-top-n",
        type=int,
        default=4,
        help="Number of top ROIs to use per slide when using --roi-report-csv (paper uses 4)",
    )
    parser.add_argument(
        "--roi-size-40x",
        type=int,
        default=5657,
        help="ROI window size in pixels at 40x for report-based ROIs (paper uses 5657)",
    )
    parser.add_argument(
        "--candidate-csv-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing per-slide candidate CSVs named like 'TUPAC-TR-123.csv' with 'x,y' per line in level-0 coords. "
            "If provided, CNN_global patches are cropped centered on these candidate points (paper-aligned with CNN_seg candidates)."
        ),
    )
    parser.add_argument(
        "--require-candidates",
        action="store_true",
        help="If set, skip slides that do not have a corresponding candidate CSV",
    )
    parser.add_argument(
        "--require-roi",
        action="store_true",
        help="If set, skip slides that do not have a corresponding ROI CSV",
    )
    parser.add_argument("--max-slides", type=int, default=1, help="Limit number of slides (default 1 to keep it safe)")
    parser.add_argument("--max-patches-per-slide", type=int, default=100)
    parser.add_argument("--max-patches-score1", type=int, default=0, help="Override max patches for score=1 slides (0 disables override)")
    parser.add_argument("--max-patches-score2", type=int, default=0, help="Override max patches for score=2 slides (0 disables override)")
    parser.add_argument("--max-patches-score3", type=int, default=0, help="Override max patches for score=3 slides (0 disables override)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already-indexed slides")

    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-epochs", type=int, default=1)
    parser.add_argument("--global-lr", type=float, default=1e-4)
    parser.add_argument("--global-momentum", type=float, default=0.9)
    parser.add_argument("--global-weight-decay", type=float, default=5e-4)
    parser.add_argument("--global-max-steps", type=int, default=0, help="Limit training steps per epoch (0=unlimited)")
    parser.add_argument("--global-checkpoint", type=str, default=None, help="Checkpoint path (defaults under config.output_dir)")
    parser.add_argument(
        "--global-val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction at slide-level (0 disables validation)",
    )
    parser.add_argument("--global-split-seed", type=int, default=1337)

    parser.add_argument(
        "--no-global-augment",
        action="store_true",
        help="Disable data augmentation for CNN_global training",
    )
    parser.add_argument(
        "--global-color-jitter",
        type=float,
        default=0.1,
        help="Color jitter strength for CNN_global training (0 disables jitter)",
    )
    parser.add_argument(
        "--no-global-balance",
        action="store_true",
        help="Disable class balancing in CNN_global training (WeightedRandomSampler)",
    )

    parser.add_argument("--seg-epochs", type=int, default=1)
    parser.add_argument("--seg-paper-tune-epochs", type=int, default=1)
    parser.add_argument("--seg-paper-final-epochs", type=int, default=0)
    parser.add_argument("--seg-paper-split-seed", type=int, default=1337)
    parser.add_argument(
        "--seg-paper-early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience (in epochs) on test score; 0 disables",
    )
    parser.add_argument(
        "--seg-paper-eval-max-batches",
        type=int,
        default=0,
        help="Limit eval batches for val/test (0=all)",
    )
    parser.add_argument(
        "--seg-paper-eval-every",
        type=int,
        default=1,
        help="Run val/test evaluation every N epochs (1=every epoch)",
    )
    parser.add_argument(
        "--seg-paper-eval-log-every",
        type=int,
        default=200,
        help="Log eval progress every N batches (0 disables)",
    )
    parser.add_argument(
        "--seg-paper-eval-auc-max-pixels",
        type=int,
        default=2_000_000,
        help="Max pixels sampled for AUC during eval (0 disables; <0 means no cap)",
    )
    parser.add_argument(
        "--seg-paper-require-positive-metrics",
        action="store_true",
        help="Use pos-only Dice/IoU (skip empty-GT batches) as the early-stop score",
    )
    parser.add_argument(
        "--seg-paper-max-steps",
        type=int,
        default=0,
        help="Limit training steps per epoch (0=unlimited)",
    )
    parser.add_argument(
        "--seg-paper-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for paper-aligned CNN_seg (defaults under config.output_dir)",
    )
    parser.add_argument(
        "--seg-paper-metrics-csv",
        type=str,
        default=None,
        help="Metrics CSV path (defaults under config.output_dir)",
    )
    parser.add_argument(
        "--seg-paper-patch-size",
        type=int,
        default=512,
        help="Must be 512 for paper-aligned CNN_seg",
    )
    parser.add_argument(
        "--seg-paper-overlap",
        type=int,
        default=80,
        help="Must be 80 for paper-aligned CNN_seg",
    )
    parser.add_argument("--seg-lr", type=float, default=1e-4)
    parser.add_argument("--seg-momentum", type=float, default=0.9)
    parser.add_argument("--seg-weight-decay", type=float, default=5e-4)
    parser.add_argument("--seg-max-steps", type=int, default=0, help="Limit seg training steps per epoch (0=unlimited)")
    parser.add_argument("--seg-checkpoint", type=str, default=None, help="Checkpoint path for CNN_seg")

    parser.add_argument("--det-epochs", type=int, default=1)
    parser.add_argument("--det-lr", type=float, default=1e-4)
    parser.add_argument("--det-momentum", type=float, default=0.9)
    parser.add_argument("--det-weight-decay", type=float, default=5e-4)
    parser.add_argument("--det-max-steps", type=int, default=0, help="Limit det training steps per epoch (0=unlimited)")
    parser.add_argument("--det-checkpoint", type=str, default=None, help="Checkpoint path for CNN_det")
    parser.add_argument(
        "--det-index-csv",
        type=str,
        default=None,
        help="Optional prepared CNN_det patch index CSV (if set, CNN_det trains from disk patches instead of on-the-fly candidates)",
    )
    parser.add_argument(
        "--det-out-root",
        type=str,
        default=None,
        help="Output root for prepared CNN_det patches (defaults under config.output_dir)",
    )
    parser.add_argument(
        "--det-seg-checkpoint",
        type=str,
        default=None,
        help="CNN_seg checkpoint to use when preparing CNN_det patches (defaults to config.output_dir/models/cnn_seg.pt)",
    )
    parser.add_argument("--det-max-tiles", type=int, default=0, help="Limit tiles when preparing det patches (0=all)")
    parser.add_argument("--det-max-candidates-per-tile", type=int, default=200)
    parser.add_argument("--det-max-neg-per-pos", type=int, default=3)
    parser.add_argument(
        "--det-match-radius",
        type=int,
        default=30,
        help=(
            "Radius (pixels) around GT centroid to match predicted candidate components when labeling positives. "
            "0 enforces exact-pixel containment."
        ),
    )

    parser.add_argument(
        "--det-add-gt-patches",
        action="store_true",
        help=(
            "Add GT-forced positive patches centered on pathologist centroids. "
            "This increases GT coverage for CNN_det training when CNN_seg candidates miss GTs. "
            "These rows are marked is_gt_forced=1 in index.csv."
        ),
    )
    parser.add_argument(
        "--det-gt-patches-all",
        action="store_true",
        help=(
            "When used with --det-add-gt-patches, add GT patches for every centroid (default is only for GTs "
            "that are not matched to any candidate within match radius)."
        ),
    )

    parser.add_argument(
        "--det-min-region-area",
        type=int,
        default=1,
        help="Minimum connected-component area for CNN_seg candidates (filters tiny blobs)",
    )
    parser.add_argument(
        "--det-min-region-mean-prob",
        type=float,
        default=0.0,
        help="Minimum mean CNN_seg positive-class probability over a candidate region",
    )
    parser.add_argument(
        "--det-min-region-max-prob",
        type=float,
        default=0.0,
        help="Minimum max CNN_seg positive-class probability within a candidate region",
    )

    parser.add_argument("--qa-det-index", action="store_true", help="QA a prepared CNN_det index.csv against GT centroids")
    parser.add_argument(
        "--qa-det-index-csv",
        type=str,
        default=None,
        help="Index CSV to QA (defaults to config.output_dir/det_patches/index.csv)",
    )
    parser.add_argument(
        "--qa-det-out-dir",
        type=str,
        default=None,
        help="Output folder for QA reports (defaults to <index_csv.parent>/qa)",
    )
    parser.add_argument("--qa-det-radius", type=int, default=30, help="Distance threshold (pixels) for within-radius stats")
    parser.add_argument("--qa-det-samples", type=int, default=64, help="Render this many patch thumbnails into qa_samples.png (0 disables)")
    parser.add_argument("--qa-det-seed", type=int, default=1337)

    parser.add_argument("--train-det-paper", action="store_true", help="Run paper-aligned CNN_det training")
    parser.add_argument(
        "--det-paper-resume",
        action="store_true",
        help="Resume paper-aligned CNN_det training from <checkpoint>_last.pt if present",
    )
    parser.add_argument("--det-paper-tune-epochs", type=int, default=10, help="Epochs for tuning phase")
    parser.add_argument("--det-paper-final-epochs", type=int, default=5, help="Epochs for final phase")
    parser.add_argument("--det-paper-split-seed", type=int, default=1337)
    parser.add_argument(
        "--det-paper-eval-every",
        type=int,
        default=1,
        help="Run val/test evaluation every N epochs during CNN_det paper training (default 1; set 2 to speed up)",
    )
    parser.add_argument(
        "--det-paper-eval-max-batches",
        type=int,
        default=0,
        help="Limit val/test evaluation to at most this many batches (0=all)",
    )
    parser.add_argument(
        "--det-paper-num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers for CNN_det paper training (0 disables multiprocessing)",
    )
    parser.add_argument(
        "--det-paper-prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when num_workers>0 (default 2)",
    )
    parser.add_argument(
        "--det-paper-no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers (may help on some systems)",
    )
    parser.add_argument(
        "--det-paper-no-balanced-sampler",
        action="store_true",
        help="Disable class balancing via WeightedRandomSampler for CNN_det paper training",
    )
    parser.add_argument(
        "--det-paper-class-weight-mode",
        type=str,
        default="auto",
        choices=["none", "auto", "manual"],
        help="Loss class weighting mode for CNN_det paper training (default auto=neg/pos)",
    )
    parser.add_argument(
        "--det-paper-pos-weight",
        type=float,
        default=0.0,
        help="Positive-class weight (w1) when --det-paper-class-weight-mode=manual",
    )
    parser.add_argument(
        "--det-paper-focal-gamma",
        type=float,
        default=0.0,
        help="Use focal loss with this gamma (0 disables; typical 1-2)",
    )
    parser.add_argument(
        "--det-paper-select-metric",
        type=str,
        default="ap",
        choices=["ap", "auc", "best_f1", "acc"],
        help="Metric used to select best model / early-stop: ap|auc|best_f1|acc (default ap)",
    )
    parser.add_argument("--det-paper-hnm", action="store_true", help="Enable hard negative mining (sampler weight boosting)")
    parser.add_argument("--det-paper-hnm-start-epoch", type=int, default=2, help="Start HNM at this epoch (1-based)")
    parser.add_argument("--det-paper-hnm-every", type=int, default=1, help="Update HNM weights every N epochs")
    parser.add_argument("--det-paper-hnm-topk", type=int, default=2000, help="Boost weights for top-K hardest negatives")
    parser.add_argument("--det-paper-hnm-boost", type=float, default=3.0, help="Weight multiplier for hard negatives")
    parser.add_argument(
        "--det-paper-hnm-max-batches",
        type=int,
        default=0,
        help="Limit batches when scoring for HNM (0=all)",
    )
    parser.add_argument("--det-paper-checkpoint", type=str, default=None)
    parser.add_argument("--det-paper-metrics-csv", type=str, default=None)
    parser.add_argument("--det-paper-index-csv", type=str, default=None)

    parser.add_argument("--train-global-paper", action="store_true", help="Run paper-aligned CNN_global training")
    parser.add_argument("--global-paper-epochs", type=int, default=30)
    parser.add_argument("--global-paper-split-seed", type=int, default=1337)
    parser.add_argument(
        "--global-paper-holdout-train-slides",
        type=int,
        default=0,
        help="If >0, sample this many slides (stratified) as the training/CV pool (e.g. 400)",
    )
    parser.add_argument(
        "--global-paper-holdout-test-slides",
        type=int,
        default=0,
        help="If >0, sample this many slides (stratified) as a held-out test set shared across folds (e.g. 100)",
    )
    parser.add_argument(
        "--global-paper-cv-folds",
        type=int,
        default=1,
        help="Number of slide-level CV folds (1 keeps 60/20/20; paper uses 3)",
    )
    parser.add_argument(
        "--global-paper-eval-every",
        type=int,
        default=1,
        help="Evaluate on val every N epochs (default 1; set 2 to eval every two epochs)",
    )
    parser.add_argument(
        "--global-paper-parallel-folds",
        action="store_true",
        help="Run CV folds in parallel (one process per fold; requires enough devices)",
    )
    parser.add_argument(
        "--global-paper-fold-devices",
        type=str,
        default=None,
        help="Comma/semicolon-separated device list for folds (e.g. npu:0,npu:1,npu:2)",
    )
    parser.add_argument(
        "--global-paper-ensemble-strategy",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="Ensemble strategy for fold models at inference (sum|mean; default sum)",
    )
    parser.add_argument(
        "--global-paper-ensemble-manifest",
        type=str,
        default=None,
        help="Optional path to write ensemble manifest JSON (defaults next to checkpoint)",
    )
    parser.add_argument("--global-paper-checkpoint", type=str, default=None)
    parser.add_argument("--global-paper-metrics-csv", type=str, default=None)
    parser.add_argument("--global-paper-index-csv", type=str, default=None)

    parser.add_argument(
        "--global-paper-num-workers",
        type=int,
        default=4,
        help="DataLoader workers per fold (default 4; set 0 for single-process loading)",
    )
    parser.add_argument(
        "--global-paper-prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when num_workers>0 (default 2)",
    )
    parser.add_argument(
        "--global-paper-no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers for CNN_global paper training",
    )

    # Internal: fold worker entrypoint (used to run folds in parallel via subprocess)
    parser.add_argument(
        "--train-global-paper-fold-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--global-paper-worker-payload",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seg-length", type=int, default=10_000)
    parser.add_argument("--det-length", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if bool(getattr(args, "train_global_paper_fold_worker", False)):
        payload_path_s = str(getattr(args, "global_paper_worker_payload", "") or "").strip()
        if not payload_path_s:
            raise ValueError("--global-paper-worker-payload is required when --train-global-paper-fold-worker is set")
        payload_path = Path(payload_path_s)
        if not payload_path.exists():
            raise FileNotFoundError(str(payload_path))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        _train_global_paper_fold_worker(**payload)
        return

    if args.generate_candidates:
        cfg = load_mfc_cnn_config()
        out_root = Path(args.out_root) if args.out_root else cfg.output_dir
        
        candidates_out = Path(args.candidates_out_dir) if args.candidates_out_dir else (out_root / "candidates")
        seg_ckpt = Path(args.det_seg_checkpoint) if args.det_seg_checkpoint else (out_root / "models" / "cnn_seg.pt")
        roi_report = Path(args.roi_report_csv) if args.roi_report_csv else (out_root / "roi_selector" / "outputs" / "reports" / "stage_two_roi_selection.csv")
        
        logger.info("Generating candidates using seg model: %s", seg_ckpt)
        logger.info("Reading ROIs from: %s", roi_report)
        logger.info("Writing candidates to: %s", candidates_out)
        
        generate_candidates_within_rois(
            slides_dir=cfg.tupac_train_dir,
            roi_report_csv=roi_report,
            output_dir=candidates_out,
            seg_checkpoint_path=seg_ckpt,
            device=_resolve_torch_device(args.device),
        )
        return

    if args.smoke:
        _run_smoke(batch_size=args.batch_size, seg_length=args.seg_length, det_length=args.det_length, device=args.device)
        return

    if args.train_seg:
        cfg = load_mfc_cnn_config()
        checkpoint_path = (
            Path(args.seg_checkpoint)
            if args.seg_checkpoint
            else (cfg.output_dir / "models" / "cnn_seg.pt")
        )
        _train_seg(
            checkpoint_path=checkpoint_path,
            batch_size=int(args.batch_size),
            epochs=int(args.seg_epochs),
            lr=float(args.seg_lr),
            momentum=float(args.seg_momentum),
            weight_decay=float(args.seg_weight_decay),
            max_steps=int(args.seg_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            seg_length=int(args.seg_length),
        )
        return

    if args.train_seg_paper:
        cfg = load_mfc_cnn_config()
        checkpoint_path = (
            Path(args.seg_paper_checkpoint)
            if args.seg_paper_checkpoint
            else (cfg.output_dir / "models" / "cnn_seg_paper.pt")
        )
        metrics_csv = (
            Path(args.seg_paper_metrics_csv)
            if args.seg_paper_metrics_csv
            else (cfg.output_dir / "metrics" / "cnn_seg_paper_metrics.csv")
        )
        _train_seg_paper(
            checkpoint_path=checkpoint_path,
            metrics_csv=metrics_csv,
            batch_size=int(args.batch_size),
            tune_epochs=int(args.seg_paper_tune_epochs),
            final_epochs=int(args.seg_paper_final_epochs),
            max_steps_per_epoch=int(args.seg_paper_max_steps),
            device=args.device,
            split_seed=int(args.seg_paper_split_seed),
            eval_max_batches=int(args.seg_paper_eval_max_batches),
            eval_every=int(args.seg_paper_eval_every),
            eval_log_every=int(args.seg_paper_eval_log_every),
            eval_auc_max_pixels=int(args.seg_paper_eval_auc_max_pixels),
            require_positive_metrics=bool(args.seg_paper_require_positive_metrics),
            early_stop_patience=int(args.seg_paper_early_stop_patience),
            patch_size=int(args.seg_paper_patch_size),
            overlap=int(args.seg_paper_overlap),
        )
        return

    if args.train_det_paper:
        cfg = load_mfc_cnn_config()
        checkpoint_path = (
            Path(args.det_paper_checkpoint)
            if args.det_paper_checkpoint
            else (cfg.output_dir / "models" / "cnn_det_paper.pt")
        )
        metrics_csv = (
            Path(args.det_paper_metrics_csv)
            if args.det_paper_metrics_csv
            else (cfg.output_dir / "models" / "det_metrics.csv")
        )
        index_csv = (
            Path(args.det_paper_index_csv)
            if args.det_paper_index_csv 
            else (cfg.output_dir / "det_patches" / "index.csv")
        )
        
        _train_det_paper(
            index_csv=index_csv,
            checkpoint_path=checkpoint_path,
            metrics_csv=metrics_csv,
            batch_size=int(args.batch_size),
            tune_epochs=int(args.det_paper_tune_epochs),
            final_epochs=int(args.det_paper_final_epochs),
            device=args.device,
            split_seed=int(args.det_paper_split_seed),
            eval_every=int(args.det_paper_eval_every),
            eval_max_batches=int(args.det_paper_eval_max_batches),
            num_workers=int(args.det_paper_num_workers),
            prefetch_factor=int(args.det_paper_prefetch_factor),
            persistent_workers=not bool(args.det_paper_no_persistent_workers),
            balanced_sampler=not bool(args.det_paper_no_balanced_sampler),
            class_weight_mode=str(args.det_paper_class_weight_mode),
            pos_weight=float(args.det_paper_pos_weight),
            focal_gamma=float(args.det_paper_focal_gamma),
            select_metric=str(args.det_paper_select_metric),
            hnm_enabled=bool(args.det_paper_hnm),
            hnm_start_epoch=int(args.det_paper_hnm_start_epoch),
            hnm_every=int(args.det_paper_hnm_every),
            hnm_topk=int(args.det_paper_hnm_topk),
            hnm_boost=float(args.det_paper_hnm_boost),
            hnm_max_batches=int(args.det_paper_hnm_max_batches),
            resume=bool(args.det_paper_resume),
        )
        return

    if args.train_global_paper:
        cfg = load_mfc_cnn_config()
        checkpoint_path = (
            Path(args.global_paper_checkpoint)
            if args.global_paper_checkpoint
            else (cfg.output_dir / "models" / "cnn_global_paper.pt")
        )
        metrics_csv = (
            Path(args.global_paper_metrics_csv)
            if args.global_paper_metrics_csv
            else (cfg.output_dir / "models" / "global_metrics.csv")
        )
        index_csv = (
            Path(args.global_paper_index_csv)
            if args.global_paper_index_csv
            else (cfg.output_dir / "global_patches" / "index.csv")
        )

        ensemble_manifest = Path(args.global_paper_ensemble_manifest) if args.global_paper_ensemble_manifest else None

        _train_global_paper(
            index_csv=index_csv,
            checkpoint_path=checkpoint_path,
            metrics_csv=metrics_csv,
            cv_folds=int(args.global_paper_cv_folds),
            eval_every=int(args.global_paper_eval_every),
            parallel_folds=bool(args.global_paper_parallel_folds),
            fold_devices=str(args.global_paper_fold_devices) if args.global_paper_fold_devices else None,
            ensemble_strategy=str(args.global_paper_ensemble_strategy),
            ensemble_manifest=ensemble_manifest,
            batch_size=int(args.global_batch_size),
            num_workers=int(args.global_paper_num_workers),
            prefetch_factor=int(args.global_paper_prefetch_factor),
            persistent_workers=not bool(args.global_paper_no_persistent_workers),
            epochs=int(args.global_paper_epochs),
            lr=float(args.global_lr),
            momentum=float(args.global_momentum),
            weight_decay=float(args.global_weight_decay),
            max_steps=int(args.global_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            split_seed=int(args.global_paper_split_seed),
            holdout_train_slides=int(args.global_paper_holdout_train_slides),
            holdout_test_slides=int(args.global_paper_holdout_test_slides),
            augment=not bool(args.no_global_augment),
            color_jitter=float(args.global_color_jitter),
            balance=not bool(args.no_global_balance),
        )
        return

    if args.train_det:
        cfg = load_mfc_cnn_config()
        checkpoint_path = (
            Path(args.det_checkpoint)
            if args.det_checkpoint
            else (cfg.output_dir / "models" / "cnn_det.pt")
        )
        det_index_csv = Path(args.det_index_csv) if args.det_index_csv else None
        _train_det(
            checkpoint_path=checkpoint_path,
            batch_size=int(args.batch_size),
            epochs=int(args.det_epochs),
            lr=float(args.det_lr),
            momentum=float(args.det_momentum),
            weight_decay=float(args.det_weight_decay),
            max_steps=int(args.det_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            det_length=int(args.det_length),
            det_index_csv=det_index_csv,
        )
        return

    if args.prepare_det:
        cfg = load_mfc_cnn_config()
        det_out_root = Path(args.det_out_root) if args.det_out_root else (cfg.output_dir / "det_patches")
        det_index_csv = Path(args.det_index_csv) if args.det_index_csv else (det_out_root / "index.csv")
        seg_ckpt = (
            Path(args.det_seg_checkpoint)
            if args.det_seg_checkpoint
            else (cfg.output_dir / "models" / "cnn_seg.pt")
        )

        base = cfg.tupac_aux_mitoses_dir
        image_zips = [base / rel for rel in cfg.mitoses_image_zip_parts]
        ground_truth_zip = base / cfg.mitoses_ground_truth_zip

        device_t = _resolve_torch_device(args.device)
        prepare_cnn_det_patches_from_aux_zips(
            image_zip_parts=image_zips,
            ground_truth_zip=ground_truth_zip,
            seg_checkpoint_path=seg_ckpt,
            out_root=det_out_root,
            index_csv=det_index_csv,
            device=device_t,
            crop_size=int(cfg.det_crop_size),
            max_tiles=None if int(args.det_max_tiles) <= 0 else int(args.det_max_tiles),
            max_candidates_per_tile=int(args.det_max_candidates_per_tile),
            max_negatives_per_positive=int(args.det_max_neg_per_pos),
            match_radius=int(args.det_match_radius),
            add_gt_patches=bool(args.det_add_gt_patches),
            gt_patches_missing_only=not bool(args.det_gt_patches_all),
            min_region_area=int(args.det_min_region_area),
            min_region_mean_prob=float(args.det_min_region_mean_prob),
            min_region_max_prob=float(args.det_min_region_max_prob),
            seed=int(cfg.seed),
            collect_rows=False,
        )
        return

    if args.qa_det_index:
        cfg = load_mfc_cnn_config()
        index_csv = (
            Path(args.qa_det_index_csv)
            if args.qa_det_index_csv
            else (cfg.output_dir / "det_patches" / "index.csv")
        )
        out_dir = Path(args.qa_det_out_dir) if args.qa_det_out_dir else (Path(index_csv).parent / "qa")
        _qa_det_index(
            index_csv=Path(index_csv),
            out_dir=Path(out_dir),
            radius=int(args.qa_det_radius),
            samples=int(args.qa_det_samples),
            seed=int(args.qa_det_seed),
        )
        return

    if args.prepare_global:
        cfg = load_mfc_cnn_config()

        if args.out_root is not None or args.index_csv is not None:
            out_root = Path(args.out_root) if args.out_root else (cfg.output_dir / "global_patches")
            index_csv = Path(args.index_csv) if args.index_csv else (Path(out_root) / "index.csv")
        else:
            run_id = str(args.global_run_id) if args.global_run_id else datetime.now().strftime("%Y%m%d_%H%M_prepare_global")
            run_dir = Path("output/mf_cnn/CNN_global/runs") / run_id
            out_root = run_dir / "global_patches"
            index_csv = out_root / "index.csv"

        wsi_dir = Path(args.global_wsi_dir) if args.global_wsi_dir else Path(cfg.tupac_train_dir)
        slide_glob = (
            str(args.global_slide_glob)
            if args.global_slide_glob
            else ("TUPAC-TR-*.svs" if args.global_wsi_dir is None else "*.svs")
        )

        if args.global_labels_csv:
            scores = load_slide_labels_from_csv(
                Path(args.global_labels_csv),
                id_column=(str(args.global_label_id_col) if args.global_label_id_col else None),
                label_column=(str(args.global_label_col) if args.global_label_col else None),
            )
        else:
            scores = load_tupac_train_scores(cfg.tupac_train_ground_truth_csv)

        roi_csv_dir = Path(args.roi_csv_dir) if args.roi_csv_dir else None
        roi_report_csv = Path(args.roi_report_csv) if args.roi_report_csv else None
        if roi_csv_dir is None and roi_report_csv is None:
            default_report = Path("output/roi_selector/outputs/reports/stage_two_roi_selection.csv")
            if default_report.exists():
                roi_report_csv = default_report
                logger.info("Using default ROI report: %s", str(default_report))
        candidate_csv_dir = Path(args.candidate_csv_dir) if args.candidate_csv_dir else None
        max_by_score: dict[int, int] | None = None
        if any(int(v) > 0 for v in (args.max_patches_score1, args.max_patches_score2, args.max_patches_score3)):
            max_by_score = {}
            if int(args.max_patches_score1) > 0:
                max_by_score[1] = int(args.max_patches_score1)
            if int(args.max_patches_score2) > 0:
                max_by_score[2] = int(args.max_patches_score2)
            if int(args.max_patches_score3) > 0:
                max_by_score[3] = int(args.max_patches_score3)

        prepare_global_patch_dataset(
            wsi_dir=wsi_dir,
            slide_glob=slide_glob,
            scores_by_slide_id=scores,
            out_root=out_root,
            index_csv=index_csv,
            patch_size=int(args.patch_size),
            overlap=int(args.overlap),
            level=int(args.level),
            max_slides=None if args.max_slides <= 0 else int(args.max_slides),
            max_patches_per_slide=None if args.max_patches_per_slide <= 0 else int(args.max_patches_per_slide),
            max_patches_by_score=max_by_score,
            roi_csv_dir=roi_csv_dir,
            roi_report_csv=roi_report_csv,
            roi_top_n=int(args.roi_top_n),
            roi_size_40x=int(args.roi_size_40x),
            candidate_csv_dir=candidate_csv_dir,
            require_candidates=bool(args.require_candidates),
            require_roi=bool(args.require_roi),
            min_tissue_fraction=float(args.min_tissue_fraction),
            resume=not bool(args.no_resume),
        )
        logger.info("Done. index_csv=%s", str(index_csv))
        return

    if args.train_global:
        cfg = load_mfc_cnn_config()
        index_csv = Path(args.index_csv) if args.index_csv else (cfg.output_dir / "global_patches" / "index.csv")
        checkpoint_path = (
            Path(args.global_checkpoint)
            if args.global_checkpoint
            else (cfg.output_dir / "models" / "cnn_global.pt")
        )
        _train_global(
            index_csv=index_csv,
            checkpoint_path=checkpoint_path,
            batch_size=int(args.global_batch_size),
            epochs=int(args.global_epochs),
            lr=float(args.global_lr),
            momentum=float(args.global_momentum),
            weight_decay=float(args.global_weight_decay),
            max_steps=int(args.global_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            val_fraction=float(args.global_val_fraction),
            split_seed=int(args.global_split_seed),
            augment=not bool(args.no_global_augment),
            color_jitter=float(args.global_color_jitter),
            balance=not bool(args.no_global_balance),
        )
        return

    parser.print_help()


def _train_global(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    pretrained: bool,
    val_fraction: float,
    split_seed: int,
    augment: bool,
    color_jitter: float,
    balance: bool,
) -> None:
    """Train CNN_global on prepared patches.

    Args:
        index_csv: CSV created by `--prepare-global`.
        checkpoint_path: Where to save the model checkpoint.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: SGD learning rate.
        momentum: SGD momentum.
        weight_decay: SGD weight decay.
        max_steps: Optional cap per epoch (0=unlimited).
        device: cpu/cuda/mps.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_global_patch_index(index_csv)
    train_rows, val_rows = _split_global_rows_by_slide(
        rows,
        val_fraction=float(val_fraction),
        seed=int(split_seed),
    )

    train_transform = _build_global_train_transform(augment=bool(augment), color_jitter=float(color_jitter))
    train_ds = GlobalScoringPatchDataset(rows=train_rows, transform=train_transform, normalize_imagenet=True)

    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if balance:
        # Balance classes by inverse-frequency sampling.
        labels = [int(r.label) for r in train_rows]
        counts: dict[int, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        weights = [1.0 / float(counts[int(lbl)]) for lbl in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info("Global train balance: counts=%s", str(counts))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=0,
    )

    val_loader: DataLoader | None = None
    if val_rows:
        val_ds = GlobalScoringPatchDataset(rows=val_rows, transform=None, normalize_imagenet=True)
        val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    device_t = _resolve_torch_device(device)
    model = CNNGlobal(pretrained=bool(pretrained)).to(device_t)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))

    model.train()
    best_val_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(int(epochs)):
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device_t)
            labels = labels.to(device_t)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())
            preds = logits.argmax(dim=1)
            train_correct += int((preds == labels).sum().detach().cpu())
            train_total += int(labels.numel())

            if max_steps > 0 and step >= int(max_steps):
                break

        train_avg_loss = train_loss_sum / max(1, step)
        train_acc = train_correct / max(1, train_total)

        val_avg_loss: float | None = None
        val_acc: float | None = None
        if val_loader is not None:
            val_avg_loss, val_acc = _eval_global(model=model, loader=val_loader, criterion=criterion, device=device_t)
            logger.info(
                "CNNGlobal epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f steps=%d",
                epoch + 1,
                train_avg_loss,
                train_acc,
                float(val_avg_loss),
                float(val_acc),
                step,
            )
            if best_val_loss is None or float(val_avg_loss) < best_val_loss:
                best_val_loss = float(val_avg_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            logger.info(
                "CNNGlobal epoch=%d train_loss=%.4f train_acc=%.3f steps=%d",
                epoch + 1,
                train_avg_loss,
                train_acc,
                step,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": "CNNGlobal",
            "index_csv": str(index_csv),
            "pretrained": bool(pretrained),
            "val_fraction": float(val_fraction),
            "split_seed": int(split_seed),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    logger.info("Saved CNNGlobal checkpoint: %s", str(checkpoint_path))

def _train_global_paper_fold_worker(
    *,
    index_csv: str,
    train_slide_ids: list[str],
    val_slide_ids: list[str],
    test_slide_ids: list[str],
    checkpoint_path: str,
    metrics_csv: str,
    device: str,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    pretrained: bool,
    augment: bool,
    color_jitter: float,
    balance: bool,
    eval_every: int,
    cv_folds: int,
    ensemble_strategy: str,
    split_payload: dict[str, object],
) -> None:
    """Train one fold/run of CNN_global(paper) given slide splits.

    Designed as a top-level function so it can be launched in a subprocess.
    """

    metrics_csv_p = Path(metrics_csv)
    _append_metrics_row(metrics_csv_p, {"event": "start", "device": str(device), **split_payload})

    rows = read_global_patch_index(Path(index_csv))
    train_slides = set(str(s) for s in train_slide_ids)
    val_slides = set(str(s) for s in val_slide_ids)
    test_slides = set(str(s) for s in test_slide_ids)

    train_rows = [r for r in rows if r.slide_id in train_slides]
    val_rows = [r for r in rows if r.slide_id in val_slides]
    test_rows = [r for r in rows if r.slide_id in test_slides]

    train_transform = _build_global_train_transform(augment=bool(augment), color_jitter=float(color_jitter))
    train_ds = GlobalScoringPatchDataset(rows=train_rows, transform=train_transform, normalize_imagenet=True)
    val_ds = GlobalScoringPatchDataset(rows=val_rows, transform=None, normalize_imagenet=True)
    test_ds = GlobalScoringPatchDataset(rows=test_rows, transform=None, normalize_imagenet=True)

    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if bool(balance):
        labels = [int(r.label) for r in train_rows]
        counts: dict[int, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        weights = [1.0 / float(counts[int(l)]) for l in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info("Global train balance: %s", str(counts))

    num_workers_i = max(0, int(num_workers))
    prefetch_factor_i = max(1, int(prefetch_factor))
    persistent_workers_b = bool(persistent_workers) and num_workers_i > 0

    train_loader_kwargs: dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": int(num_workers_i),
        "persistent_workers": bool(persistent_workers_b),
    }
    eval_loader_kwargs: dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers_i),
        "persistent_workers": bool(persistent_workers_b),
    }
    if num_workers_i > 0:
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor_i)
        eval_loader_kwargs["prefetch_factor"] = int(prefetch_factor_i)

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)

    device_t = _resolve_torch_device(str(device))
    model = CNNGlobal(pretrained=bool(pretrained)).to(device_t)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    eval_every_i = max(1, int(eval_every))
    best_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(int(epochs)):
        epoch_idx = epoch + 1
        model.train()

        train_loss_sum = 0.0
        correct = 0
        total = 0
        steps = 0

        for images, labels in train_loader:
            images = images.to(device_t)
            labels = labels.to(device_t)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().cpu())
            total += int(labels.numel())
            steps += 1
            if int(max_steps) > 0 and steps >= int(max_steps):
                break

        train_loss = train_loss_sum / max(1, steps)
        train_acc = float(correct / max(1, total))

        do_eval = eval_every_i <= 1 or (epoch_idx % eval_every_i == 0) or (epoch_idx == int(epochs))
        val_loss = float("nan")
        val_acc = float("nan")
        test_loss = float("nan")
        test_acc = float("nan")

        if bool(do_eval):
            val_loss, val_acc = _eval_global(model=model, loader=val_loader, criterion=criterion, device=device_t)
            # Only run test at the end to reduce eval overhead.
            if epoch_idx == int(epochs):
                test_loss, test_acc = _eval_global(model=model, loader=test_loader, criterion=criterion, device=device_t)

            logger.info(
                "Global epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
                epoch_idx,
                train_loss,
                train_acc,
                float(val_loss),
                float(val_acc),
            )
        else:
            logger.info(
                "Global epoch=%d train_loss=%.4f train_acc=%.3f (skip eval; eval_every=%d)",
                epoch_idx,
                train_loss,
                train_acc,
                int(eval_every_i),
            )

        _append_metrics_row(
            metrics_csv_p,
            {
                "epoch": int(epoch_idx),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                **split_payload,
            },
        )

        if bool(do_eval) and (not np.isnan(val_acc)) and float(val_acc) > float(best_acc):
            best_acc = float(val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation of the selected model.
    final_val_loss, final_val_acc = _eval_global(model=model, loader=val_loader, criterion=criterion, device=device_t)
    final_test_loss, final_test_acc = _eval_global(model=model, loader=test_loader, criterion=criterion, device=device_t)
    _append_metrics_row(
        metrics_csv_p,
        {
            "event": "final",
            "best_val_acc": float(best_acc),
            "val_loss": float(final_val_loss),
            "val_acc": float(final_val_acc),
            "test_loss": float(final_test_loss),
            "test_acc": float(final_test_acc),
            "device": str(device),
            **split_payload,
        },
    )

    torch.save(
        {
            "model": "CNNGlobal",
            "paper_aligned": True,
            "cv_folds": int(cv_folds),
            "ensemble_strategy": str(ensemble_strategy),
            "eval_every": int(eval_every_i),
            "device": str(device),
            **split_payload,
            "state_dict": model.state_dict(),
        },
        Path(checkpoint_path),
    )
    logger.info(
        "Saved CNN_global(paper) checkpoint: %s (best_val_acc=%.3f)",
        str(checkpoint_path),
        float(best_acc),
    )


def _train_global_paper(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    metrics_csv: Path,
    cv_folds: int,
    eval_every: int,
    parallel_folds: bool,
    fold_devices: str | None,
    ensemble_strategy: str,
    ensemble_manifest: Path | None,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    pretrained: bool,
    split_seed: int,
    holdout_train_slides: int,
    holdout_test_slides: int,
    augment: bool,
    color_jitter: float,
    balance: bool,
) -> None:
    """Train CNN_global using paper-aligned logic.

    - If `cv_folds <= 1`: keep the existing stratified 60/20/20 slide split.
    - If `cv_folds >= 2`: run stratified K-fold cross-validation at slide-level and
      save one best checkpoint per fold. This matches the paper's ensemble setup.
    """

    def _parse_devices(s: str | None) -> list[str]:
        if not s:
            return []
        parts = [p.strip() for p in re.split(r"[;,]", str(s))]
        return [p for p in parts if p]

    index_csv = Path(index_csv)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(metrics_csv)

    cv_folds_i = int(cv_folds)
    if cv_folds_i < 1:
        raise ValueError("--global-paper-cv-folds must be >= 1")

    eval_every_i = int(eval_every)
    if eval_every_i < 1:
        raise ValueError("--global-paper-eval-every must be >= 1")

    ensemble_strategy_i = str(ensemble_strategy).lower().strip()
    if ensemble_strategy_i not in {"sum", "mean"}:
        raise ValueError("--global-paper-ensemble-strategy must be: sum|mean")

    rows = read_global_patch_index(index_csv)

    slide_to_label: dict[str, int] = {}
    for r in rows:
        slide_to_label.setdefault(r.slide_id, int(r.label))

    slide_ids = sorted(list(slide_to_label.keys()))

    def _stratified_sample(
        slides_by_label: dict[int, list[str]],
        *,
        n_total: int,
        rng: random.Random,
    ) -> set[str]:
        if int(n_total) <= 0:
            return set()
        total = sum(len(v) for v in slides_by_label.values())
        if int(n_total) > int(total):
            raise ValueError(f"Requested {n_total} slides but only {total} available")

        # Allocate per label proportional to group sizes.
        desired: dict[int, float] = {}
        base: dict[int, int] = {}
        frac: list[tuple[float, int]] = []
        for lbl, group in slides_by_label.items():
            if not group:
                continue
            d = float(n_total) * (float(len(group)) / float(total))
            k = int(d)
            k = min(k, len(group))
            desired[lbl] = d
            base[lbl] = k
            frac.append((float(d - float(k)), int(lbl)))

        remaining = int(n_total) - sum(int(v) for v in base.values())
        for _, lbl in sorted(frac, key=lambda t: t[0], reverse=True):
            if remaining <= 0:
                break
            cap = len(slides_by_label.get(int(lbl), []))
            if int(base.get(int(lbl), 0)) < int(cap):
                base[int(lbl)] = int(base.get(int(lbl), 0)) + 1
                remaining -= 1

        # If rounding/caps prevented us from reaching n_total, distribute to any label with remaining capacity.
        if remaining > 0:
            labels = sorted(slides_by_label.keys(), key=lambda l: len(slides_by_label[l]), reverse=True)
            while remaining > 0:
                progressed = False
                for lbl in labels:
                    cap = len(slides_by_label.get(int(lbl), []))
                    if int(base.get(int(lbl), 0)) < int(cap):
                        base[int(lbl)] = int(base.get(int(lbl), 0)) + 1
                        remaining -= 1
                        progressed = True
                        if remaining <= 0:
                            break
                if not progressed:
                    break
        if remaining != 0:
            raise RuntimeError("Failed to allocate stratified sample counts")

        picked: set[str] = set()
        for lbl, group in slides_by_label.items():
            group_copy = list(group)
            rng.shuffle(group_copy)
            k = int(base.get(int(lbl), 0))
            picked.update(group_copy[:k])

        if len(picked) != int(n_total):
            # Fallback: trim or fill from remaining slides (still deterministic).
            all_slides = [s for g in slides_by_label.values() for s in g]
            all_slides = sorted(set(all_slides))
            rng.shuffle(all_slides)
            picked = set(list(picked)[: int(n_total)])
            if len(picked) < int(n_total):
                for s in all_slides:
                    if s in picked:
                        continue
                    picked.add(s)
                    if len(picked) == int(n_total):
                        break
        if len(picked) != int(n_total):
            raise RuntimeError(f"Stratified sampling failed: got {len(picked)} != {n_total}")
        return picked

    holdout_train_slides_i = max(0, int(holdout_train_slides))
    holdout_test_slides_i = max(0, int(holdout_test_slides))

    # Optional: reserve a fixed held-out test set, then run CV on a training pool.
    heldout_test: set[str] = set()
    cv_pool: set[str] = set(slide_ids)
    if holdout_test_slides_i > 0 or holdout_train_slides_i > 0:
        by_label_all: dict[int, list[str]] = {}
        for sid, lbl in slide_to_label.items():
            by_label_all.setdefault(int(lbl), []).append(str(sid))

        rng = random.Random(int(split_seed))
        for group in by_label_all.values():
            rng.shuffle(group)

        if holdout_test_slides_i > 0:
            heldout_test = _stratified_sample(by_label_all, n_total=int(holdout_test_slides_i), rng=rng)
            for lbl in list(by_label_all.keys()):
                by_label_all[lbl] = [s for s in by_label_all[lbl] if s not in heldout_test]

        remaining_after_test = set(s for g in by_label_all.values() for s in g)
        if holdout_train_slides_i > 0:
            cv_pool = _stratified_sample(by_label_all, n_total=int(holdout_train_slides_i), rng=rng)
        else:
            cv_pool = set(remaining_after_test)

        if heldout_test:
            logger.info(
                "Global holdout split (slides): total=%d cv_pool=%d test=%d (seed=%d)",
                len(slide_ids),
                len(cv_pool),
                len(heldout_test),
                int(split_seed),
            )

    if cv_folds_i == 1:
        _append_metrics_row(metrics_csv, {"event": "start", "device": str(device), "cv_folds": 1})

        by_label: dict[int, list[str]] = {}
        pool = sorted(list(cv_pool)) if (holdout_test_slides_i > 0 or holdout_train_slides_i > 0) else slide_ids
        for sid in pool:
            by_label.setdefault(int(slide_to_label[str(sid)]), []).append(str(sid))

        rng = random.Random(int(split_seed))
        train_slides: set[str] = set()
        val_slides: set[str] = set()
        test_slides: set[str] = set(heldout_test)

        for group in by_label.values():
            rng.shuffle(group)
            n = len(group)
            n_train = int(0.8 * n)
            n_val = n - n_train
            train_slides.update(group[:n_train])
            val_slides.update(group[n_train:])
        if not test_slides:
            # Backwards-compatible 60/20/20 when no explicit holdout requested.
            train_slides.clear()
            val_slides.clear()
            for group in by_label.values():
                rng.shuffle(group)
                n = len(group)
                n_train = int(0.6 * n)
                n_val = int(0.2 * n)
                train_slides.update(group[:n_train])
                val_slides.update(group[n_train : n_train + n_val])
                test_slides.update(group[n_train + n_val :])

        train_rows = [r for r in rows if r.slide_id in train_slides]
        val_rows = [r for r in rows if r.slide_id in val_slides]
        test_rows = [r for r in rows if r.slide_id in test_slides]

        logger.info(
            "Global paper split (slides): total=%d train=%d val=%d test=%d",
            len(slide_ids),
            len(train_slides),
            len(val_slides),
            len(test_slides),
        )
        logger.info(
            "Global paper split (patches): total=%d train=%d val=%d test=%d",
            len(rows),
            len(train_rows),
            len(val_rows),
            len(test_rows),
        )

        _train_global_paper_fold_worker(
            index_csv=str(index_csv),
            train_slide_ids=sorted(list(train_slides)),
            val_slide_ids=sorted(list(val_slides)),
            test_slide_ids=sorted(list(test_slides)),
            checkpoint_path=str(checkpoint_path),
            metrics_csv=str(metrics_csv),
            device=str(device),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            prefetch_factor=int(prefetch_factor),
            persistent_workers=bool(persistent_workers),
            epochs=int(epochs),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            max_steps=int(max_steps),
            pretrained=bool(pretrained),
            augment=bool(augment),
            color_jitter=float(color_jitter),
            balance=bool(balance),
            eval_every=int(eval_every_i),
            cv_folds=int(cv_folds_i),
            ensemble_strategy=str(ensemble_strategy_i),
            split_payload={
                "cv_folds": 1,
                "split_seed": int(split_seed),
                "split": {"train": "holdout" if bool(heldout_test) else 0.6, "val": "holdout" if bool(heldout_test) else 0.2, "test": "holdout" if bool(heldout_test) else 0.2},
                "holdout": {
                    "train_slides": int(holdout_train_slides_i),
                    "test_slides": int(holdout_test_slides_i),
                },
            },
        )
        return

    # --- K-fold CV (paper-style) ---
    cv_slide_ids = sorted(list(cv_pool))

    by_label: dict[int, list[str]] = {}
    for sid in cv_slide_ids:
        by_label.setdefault(int(slide_to_label[str(sid)]), []).append(str(sid))

    rng = random.Random(int(split_seed))
    for group in by_label.values():
        rng.shuffle(group)

    folds: list[set[str]] = [set() for _ in range(cv_folds_i)]
    for _, group in by_label.items():
        for i, sid in enumerate(group):
            folds[i % cv_folds_i].add(str(sid))

    all_slides = set(cv_slide_ids)
    fold_ckpts: list[str] = []
    fold_metrics: list[str] = []

    devices = _parse_devices(fold_devices)
    if bool(parallel_folds) and not devices:
        # If parallel folds requested but no per-fold devices provided, fall back to
        # a single device list. This will still run folds with limited concurrency.
        devices = [str(device)]

    fold_jobs: list[dict[str, object]] = []

    for fold_idx in range(cv_folds_i):
        val_slides = set(folds[fold_idx])
        train_slides = set(all_slides.difference(val_slides))

        train_rows = [r for r in rows if r.slide_id in train_slides]
        val_rows = [r for r in rows if r.slide_id in val_slides]
        if heldout_test:
            test_rows = [r for r in rows if r.slide_id in heldout_test]
        else:
            # Backwards-compatible CV mode: no separate test; mirror val to keep schema stable.
            test_rows = val_rows

        fold_checkpoint = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_fold{fold_idx + 1}{checkpoint_path.suffix}"
        )
        fold_metrics_csv = metrics_csv.with_name(f"{metrics_csv.stem}_fold{fold_idx + 1}{metrics_csv.suffix}")
        fold_ckpts.append(str(fold_checkpoint))
        fold_metrics.append(str(fold_metrics_csv))

        logger.info(
            "Global CV fold=%d/%d slides: train=%d val=%d (seed=%d)",
            fold_idx + 1,
            cv_folds_i,
            len(train_slides),
            len(val_slides),
            int(split_seed),
        )
        logger.info(
            "Global CV fold=%d/%d patches: train=%d val=%d",
            fold_idx + 1,
            cv_folds_i,
            len(train_rows),
            len(val_rows),
        )

        fold_jobs.append(
            {
                "fold": int(fold_idx + 1),
                "train_slide_ids": sorted(list(train_slides)),
                "val_slide_ids": sorted(list(val_slides)),
                "test_slide_ids": sorted(list(heldout_test)) if heldout_test else sorted(list(val_slides)),
                "checkpoint_path": str(fold_checkpoint),
                "metrics_csv": str(fold_metrics_csv),
                "split_payload": {
                    "cv_folds": int(cv_folds_i),
                    "fold": int(fold_idx + 1),
                    "split_seed": int(split_seed),
                    "val_slides": sorted(list(val_slides)),
                    "holdout": {
                        "train_slides": int(holdout_train_slides_i),
                        "test_slides": int(holdout_test_slides_i),
                    },
                },
            }
        )

    if bool(parallel_folds) and len(devices) > 1:
        import os
        import subprocess
        import sys
        import time

        available_devices: list[str] = list(devices)
        pending = list(fold_jobs)
        # (proc, device, fold, payload_path, log_fp)
        running: list[tuple[subprocess.Popen[bytes], str, int, Path, object]] = []

        logger.info("Global CV: running folds in parallel (devices=%s)", ",".join(devices))

        while pending or running:
            # Launch new jobs if we have free devices.
            while pending and available_devices:
                job = pending.pop(0)
                dev = available_devices.pop(0)
                fold_n = int(job["fold"])  # type: ignore[arg-type]

                payload = {
                    "index_csv": str(index_csv),
                    "train_slide_ids": list(job["train_slide_ids"]),
                    "val_slide_ids": list(job["val_slide_ids"]),
                    "test_slide_ids": list(job["test_slide_ids"]),
                    "checkpoint_path": str(job["checkpoint_path"]),
                    "metrics_csv": str(job["metrics_csv"]),
                    "device": str(dev),
                    "batch_size": int(batch_size),
                    "num_workers": int(num_workers),
                    "prefetch_factor": int(prefetch_factor),
                    "persistent_workers": bool(persistent_workers),
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "momentum": float(momentum),
                    "weight_decay": float(weight_decay),
                    "max_steps": int(max_steps),
                    "pretrained": bool(pretrained),
                    "augment": bool(augment),
                    "color_jitter": float(color_jitter),
                    "balance": bool(balance),
                    "eval_every": int(eval_every_i),
                    "cv_folds": int(cv_folds_i),
                    "ensemble_strategy": str(ensemble_strategy_i),
                    "split_payload": dict(job["split_payload"]),
                }

                payload_path = Path(str(job["metrics_csv"])).with_suffix(".payload.json")
                payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

                fold_log = Path(str(job["metrics_csv"])).with_suffix(".log")
                fold_log.parent.mkdir(parents=True, exist_ok=True)
                log_fp = fold_log.open("w", encoding="utf-8")

                env = dict(os.environ)
                env.setdefault("PYTHONUNBUFFERED", "1")

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cabcds.mf_cnn",
                        "--train-global-paper-fold-worker",
                        "--global-paper-worker-payload",
                        str(payload_path),
                    ],
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                running.append((proc, dev, fold_n, payload_path, log_fp))
                logger.info("Global CV: started fold=%d on device=%s pid=%d log=%s", fold_n, dev, int(proc.pid or -1), str(fold_log))

            # Reap finished.
            still_running: list[tuple[subprocess.Popen[bytes], str, int, Path, object]] = []
            for proc, dev, fold_n, payload_path, log_fp in running:
                rc = proc.poll()
                if rc is None:
                    still_running.append((proc, dev, fold_n, payload_path, log_fp))
                    continue

                try:
                    proc.wait(timeout=0.1)
                except Exception:
                    pass

                try:
                    log_fp.close()
                except Exception:
                    pass

                available_devices.append(dev)
                if int(rc) != 0:
                    raise RuntimeError(f"Global CV fold {fold_n} failed (returncode={rc}) payload={payload_path}")
                logger.info("Global CV: finished fold=%d on device=%s", fold_n, dev)

            running = still_running
            if running and (not available_devices or not pending):
                time.sleep(5)
    else:
        # Sequential (or parallel requested but only one device provided)
        for job in fold_jobs:
            fold_n = int(job["fold"])  # type: ignore[arg-type]
            dev = devices[fold_n - 1] if devices and len(devices) >= fold_n else str(device)
            _train_global_paper_fold_worker(
                index_csv=str(index_csv),
                train_slide_ids=list(job["train_slide_ids"]),
                val_slide_ids=list(job["val_slide_ids"]),
                test_slide_ids=list(job["test_slide_ids"]),
                checkpoint_path=str(job["checkpoint_path"]),
                metrics_csv=str(job["metrics_csv"]),
                device=str(dev),
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                prefetch_factor=int(prefetch_factor),
                persistent_workers=bool(persistent_workers),
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                max_steps=int(max_steps),
                pretrained=bool(pretrained),
                augment=bool(augment),
                color_jitter=float(color_jitter),
                balance=bool(balance),
                eval_every=int(eval_every_i),
                cv_folds=int(cv_folds_i),
                ensemble_strategy=str(ensemble_strategy_i),
                split_payload=dict(job["split_payload"]),
            )

    manifest_path = Path(ensemble_manifest) if ensemble_manifest is not None else checkpoint_path.with_suffix(".ensemble.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "CNNGlobal",
        "paper_aligned": True,
        "cv_folds": int(cv_folds_i),
        "split_seed": int(split_seed),
        "ensemble_strategy": str(ensemble_strategy_i),
        "checkpoints": fold_ckpts,
        "metrics": fold_metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved CNN_global ensemble manifest: %s", str(manifest_path))


def _train_seg(
    *,
    checkpoint_path: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    pretrained: bool,
    seg_length: int,
) -> None:
    """Train CNN_seg on the auxiliary mitosis dataset."""

    cfg = load_mfc_cnn_config()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    seg_loader, _ = build_default_mf_cnn_train_loaders(
        cfg,
        batch_size=int(batch_size),
        num_workers=0,
        seg_length=int(seg_length),
        det_length=1,
    )

    device_t = _resolve_torch_device(device)
    model = CNNSeg(num_classes=cfg.seg_num_classes, pretrained=bool(pretrained)).to(device_t)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    model.train()
    for epoch in range(int(epochs)):
        running = 0.0
        steps = 0
        for batch in seg_loader:
            images, masks = batch[0].to(device_t), batch[1].to(device_t)
            # CNNSeg uses VGG16 backbone (often pretrained), so normalize inputs.
            images = _imagenet_normalize_batch(images)

            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out.logits, masks)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1
            global_step += 1
            if steps == 1 or steps % 20 == 0:
                logger.info("CNN_seg train: epoch=%d step=%d loss=%.4f", epoch + 1, steps, float(loss.detach().cpu()))
            if int(max_steps) > 0 and steps >= int(max_steps):
                break

        avg = running / max(1, steps)
        logger.info("CNN_seg epoch done: epoch=%d avg_loss=%.4f steps=%d", epoch + 1, avg, steps)

    torch.save({"model_state": model.state_dict(), "config": cfg.model_dump(mode="json")}, checkpoint_path)
    logger.info("Saved CNN_seg checkpoint: %s", str(checkpoint_path))


def _train_det(
    *,
    checkpoint_path: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    pretrained: bool,
    det_length: int,
    det_index_csv: Path | None,
) -> None:
    """Train CNN_det on the auxiliary mitosis dataset."""

    cfg = load_mfc_cnn_config()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if det_index_csv is not None:
        det_ds = PreparedMitosisDetectionPatchDataset(
            index_csv=Path(det_index_csv),
            output_size=int(cfg.alexnet_input_size),
            normalize_imagenet=False,  # normalized in training loop
        )
        det_loader = DataLoader(det_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
        logger.info("CNN_det: training from prepared index: %s (rows=%d)", str(det_index_csv), len(det_ds))
    else:
        _, det_loader = build_default_mf_cnn_train_loaders(
            cfg,
            batch_size=int(batch_size),
            num_workers=0,
            seg_length=1,
            det_length=int(det_length),
        )

    device_t = _resolve_torch_device(device)
    model = CNNDet(num_classes=cfg.det_num_classes, pretrained=bool(pretrained)).to(device_t)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(int(epochs)):
        running = 0.0
        steps = 0
        correct = 0
        total = 0
        for batch in det_loader:
            images, labels = batch[0].to(device_t), batch[1].to(device_t)
            # CNN_det uses pretrained AlexNet in the paper; normalize to ImageNet.
            images = _imagenet_normalize_batch(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

            preds = torch.argmax(logits.detach(), dim=1)
            correct += int((preds == labels).sum().cpu())
            total += int(labels.numel())

            if steps == 1 or steps % 50 == 0:
                acc = correct / max(1, total)
                logger.info(
                    "CNN_det train: epoch=%d step=%d loss=%.4f acc=%.3f",
                    epoch + 1,
                    steps,
                    float(loss.detach().cpu()),
                    float(acc),
                )
            if int(max_steps) > 0 and steps >= int(max_steps):
                break

        avg = running / max(1, steps)
        acc = correct / max(1, total)
        logger.info("CNN_det epoch done: epoch=%d avg_loss=%.4f acc=%.3f steps=%d", epoch + 1, avg, float(acc), steps)

    torch.save({"model_state": model.state_dict(), "config": cfg.model_dump(mode="json")}, checkpoint_path)
    logger.info("Saved CNN_det checkpoint: %s", str(checkpoint_path))


def _build_global_train_transform(*, augment: bool, color_jitter: float) -> object | None:
    """Build torchvision transforms for CNN_global training.

    We keep this lightweight and safe. The dataset still resizes to 227x227.
    """

    if not augment:
        return None

    # Keep augmentation in PIL space. Tensor conversion/resize/normalization is
    # handled by `GlobalScoringPatchDataset` to ensure correct ordering.
    ops: list[object] = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ]
    if float(color_jitter) > 0:
        cj = float(color_jitter)
        ops.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.05, cj / 2)))
    return T.Compose(ops)


def _split_global_rows_by_slide(
    rows: list[GlobalPatchIndexRow],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[GlobalPatchIndexRow], list[GlobalPatchIndexRow]]:
    """Split patch rows into train/val by slide_id (stratified by slide label).

    We assume each slide_id has a single score label (1..3).
    """

    if val_fraction <= 0:
        return rows, []

    # slide_id -> label
    slide_to_label: dict[str, int] = {}
    for r in rows:
        slide_to_label.setdefault(r.slide_id, int(r.label))

    by_label: dict[int, list[str]] = {}
    for slide_id, label in slide_to_label.items():
        by_label.setdefault(int(label), []).append(slide_id)

    rng = random.Random(int(seed))
    val_slides: set[str] = set()
    for label, slides in by_label.items():
        rng.shuffle(slides)
        k = max(1, int(round(len(slides) * float(val_fraction)))) if len(slides) > 1 else 0
        val_slides.update(slides[:k])

    train_rows: list[GlobalPatchIndexRow] = []
    val_rows: list[GlobalPatchIndexRow] = []
    for r in rows:
        (val_rows if r.slide_id in val_slides else train_rows).append(r)

    logger.info(
        "Global split: slides_total=%d slides_val=%d rows_total=%d rows_val=%d",
        len(slide_to_label),
        len(val_slides),
        len(rows),
        len(val_rows),
    )
    return train_rows, val_rows


@torch.no_grad()
def _eval_global(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    steps = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += float(loss.detach().cpu())
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().detach().cpu())
        total += int(labels.numel())
        steps += 1

    model.train()
    avg_loss = loss_sum / max(1, steps)
    acc = correct / max(1, total)
    return float(avg_loss), float(acc)


if __name__ == "__main__":
    main()
