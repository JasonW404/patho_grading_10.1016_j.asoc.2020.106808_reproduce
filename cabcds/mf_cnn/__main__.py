"""MF-CNN utilities entry point.

This module focuses on *smoke checks* so the MF-CNN stack can be validated with
the data already present in this repo (TUPAC16 + auxiliary mitosis zips).

Example:
    `uv run python -m cabcds.mf_cnn --smoke`
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import random
import signal

import numpy as np
from sklearn.metrics import roc_auc_score
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
    load_tupac_train_scores,
    read_global_patch_index,
    read_det_patch_index,
)
from cabcds.mf_cnn.det_data import prepare_cnn_det_patches_from_aux_zips
from cabcds.mf_cnn.wsi_inference import generate_candidates_within_rois


logger = logging.getLogger(__name__)


def _append_metrics_row(csv_path: Path, row: dict[str, object]) -> None:
    """Append one metrics row to a CSV file, creating headers if needed."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
            "Running without external datasets enabled (CABCDS_MFCNN_USE_EXTERNAL_MITOSIS_DATASETS=0). "
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
    if all_y_true:
        try:
            y_true_flat = np.concatenate(all_y_true)
            y_score_flat = np.concatenate(all_y_score)
            if len(np.unique(y_true_flat)) > 1:
                auc = float(roc_auc_score(y_true_flat, y_score_flat))
        except Exception:
            pass

    model.train()
    return {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
) -> None:
    """Paper-aligned CNN_det training: 60/20/20 case split.

    - Reads 80x80 candidate patch index.
    - Splits by case_id.
    - Tuning phase: Train on 60%, Val on 20% (save best).
    - Final phase: Train on 80% (60+20), monitor on 20% test (early stop).
    - Model: AlexNet (pretrained), resized inputs to 227x227.
    - Hyperparameters: SGD lr=0.0001, momentum=0.9, weight_decay=0.0005.
    """

    index_csv = Path(index_csv)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(metrics_csv)
    _append_metrics_row(metrics_csv, {"event": "start", "device": device})

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
    val_rows = [r for r in rows if r.case_id in val_cases]
    test_rows = [r for r in rows if r.case_id in test_cases]
    
    logger.info(
        "CNN_det split (cases): total=%d train=%d val=%d test=%d seed=%d",
        n_total, len(train_cases), len(val_cases), len(test_cases), int(split_seed)
    )
    logger.info(
        "CNN_det split (patches): total=%d train=%d val=%d test=%d",
        len(rows), len(train_rows), len(val_rows), len(test_rows)
    )

    cfg = load_mfc_cnn_config()
    
    # Datasets with normalization=False because we normalize in loop
    train_ds = PreparedMitosisDetectionPatchDataset(rows=train_rows, output_size=227, normalize_imagenet=False)
    val_ds = PreparedMitosisDetectionPatchDataset(rows=val_rows, output_size=227, normalize_imagenet=False)
    test_ds = PreparedMitosisDetectionPatchDataset(rows=test_rows, output_size=227, normalize_imagenet=False)
    
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    device_t = _resolve_torch_device(device)
    model = CNNDet(num_classes=2, pretrained=True).to(device_t)

    # Paper hypers: LR=0.0001, Momentum=0.9, WD=0.0005
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    criterion = nn.CrossEntropyLoss()

    best_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0

    def _run_one_epoch(*, loader: DataLoader) -> dict[str, float]:
        nonlocal global_step
        running_loss = 0.0
        steps = 0
        correct = 0
        total = 0
        tp = fp = fn = tn = 0

        for images, labels in loader:
            images = images.to(device_t)
            labels = labels.to(device_t)
            images = _imagenet_normalize_batch(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
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

    logger.info("Starting Tuning Phase (Train on 60%, Val on 20%)")
    for epoch in range(int(tune_epochs)):
        model.train()
        train_m = _run_one_epoch(loader=train_loader)
        val_m = _eval_det_loader(model=model, loader=val_loader, device=device_t)
        
        score = val_m["acc"]
        logger.info(
            "CNN_det tune epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f val_f1=%.3f val_auc=%.3f",
            epoch + 1, train_m["loss"], train_m["acc"], val_m["loss"], val_m["acc"], val_m["f1"], val_m["auc"]
        )
        
        _append_metrics_row(metrics_csv, {
            "phase": "tune",
            "epoch": epoch+1,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_auc": val_m["auc"]
        })

        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model from tuning phase (acc=%.3f)", best_score)
    
    logger.info("Starting Final Phase (Train on 80%, Monitor on 20% Test)")
    
    final_rows = train_rows + val_rows
    final_ds = PreparedMitosisDetectionPatchDataset(rows=final_rows, output_size=227, normalize_imagenet=False)
    final_loader = DataLoader(final_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    
    no_improve = 0
    best_test_score = 0.0
    
    for epoch in range(int(final_epochs)):
        model.train()
        train_m = _run_one_epoch(loader=final_loader)
        test_m = _eval_det_loader(model=model, loader=test_loader, device=device_t)
        
        score = test_m["acc"]
        
        logger.info(
            "CNN_det final epoch=%d train_loss=%.4f train_acc=%.3f test_loss=%.4f test_acc=%.3f test_f1=%.3f test_auc=%.3f",
            epoch + 1, train_m["loss"], train_m["acc"], test_m["loss"], test_m["acc"], test_m["f1"], test_m["auc"]
        )
        
        _append_metrics_row(metrics_csv, {
            "phase": "final",
            "epoch": epoch+1,
            "train_loss": train_m["loss"],
            "test_loss": test_m["loss"],
            "test_acc": test_m["acc"],
            "test_auc": test_m["auc"]
        })
        
        if score > best_test_score:
            best_test_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if early_stop_patience > 0 and no_improve >= early_stop_patience:
             logger.info("Early stopping triggered at final epoch %d", epoch+1)
             break
             
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
        help="Extract CNN_global training patches from TUPAC train WSIs and write an index CSV",
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

    parser.add_argument("--train-det-paper", action="store_true", help="Run paper-aligned CNN_det training")
    parser.add_argument("--det-paper-tune-epochs", type=int, default=10, help="Epochs for tuning phase")
    parser.add_argument("--det-paper-final-epochs", type=int, default=5, help="Epochs for final phase")
    parser.add_argument("--det-paper-split-seed", type=int, default=1337)
    parser.add_argument("--det-paper-checkpoint", type=str, default=None)
    parser.add_argument("--det-paper-metrics-csv", type=str, default=None)
    parser.add_argument("--det-paper-index-csv", type=str, default=None)

    parser.add_argument("--train-global-paper", action="store_true", help="Run paper-aligned CNN_global training")
    parser.add_argument("--global-paper-epochs", type=int, default=30)
    parser.add_argument("--global-paper-split-seed", type=int, default=1337)
    parser.add_argument("--global-paper-checkpoint", type=str, default=None)
    parser.add_argument("--global-paper-metrics-csv", type=str, default=None)
    parser.add_argument("--global-paper-index-csv", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seg-length", type=int, default=10_000)
    parser.add_argument("--det-length", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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
            split_seed=int(args.det_paper_split_seed)
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

        _train_global_paper(
            index_csv=index_csv,
            checkpoint_path=checkpoint_path,
            metrics_csv=metrics_csv,
            batch_size=int(args.global_batch_size),
            epochs=int(args.global_paper_epochs),
            lr=float(args.global_lr),
            momentum=float(args.global_momentum),
            weight_decay=float(args.global_weight_decay),
            max_steps=int(args.global_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            split_seed=int(args.global_paper_split_seed),
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
            seed=int(cfg.seed),
        )
        return

    if args.prepare_global:
        cfg = load_mfc_cnn_config()
        out_root = Path(args.out_root) if args.out_root else (cfg.output_dir / "global_patches")
        index_csv = Path(args.index_csv) if args.index_csv else (cfg.output_dir / "global_patches" / "index.csv")

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
            tupac_train_dir=cfg.tupac_train_dir,
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

        _train_global_paper(
            index_csv=index_csv,
            checkpoint_path=checkpoint_path,
            metrics_csv=metrics_csv,
            batch_size=int(args.global_batch_size),
            epochs=int(args.global_paper_epochs),
            lr=float(args.global_lr),
            momentum=float(args.global_momentum),
            weight_decay=float(args.global_weight_decay),
            max_steps=int(args.global_max_steps),
            device=args.device,
            pretrained=bool(cfg.pretrained),
            split_seed=int(args.global_paper_split_seed),
            augment=not bool(args.no_global_augment),
            color_jitter=float(args.global_color_jitter),
            balance=not bool(args.no_global_balance),
        )
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


def _train_global_paper(
    *,
    index_csv: Path,
    checkpoint_path: Path,
    metrics_csv: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    pretrained: bool,
    split_seed: int,
    augment: bool,
    color_jitter: float,
    balance: bool,
) -> None:
    """Train CNN_global with paper-aligned 60/20/20 slide split.
    
    TUPAC16 Main training set is effectively the entire universe here.
    """
    
    index_csv = Path(index_csv)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(metrics_csv)
    _append_metrics_row(metrics_csv, {"event": "start", "device": device})
    
    rows = read_global_patch_index(index_csv)
    
    # Paper logic: "Split on Slide cases" 60/20/20
    slide_ids = sorted(list(set(r.slide_id for r in rows)))
    # Optional: Stratified by score if score is known per slide
    # We can infer score from patch labels (since all patches from a slide share label)
    slide_to_label = {}
    for r in rows:
        if r.slide_id not in slide_to_label:
            slide_to_label[r.slide_id] = int(r.label)
            
    # Group slides by label
    by_label = {}
    for sid, lbl in slide_to_label.items():
        by_label.setdefault(lbl, []).append(sid)
        
    rng = random.Random(int(split_seed))
    
    train_slides = set()
    val_slides = set()
    test_slides = set()
    
    for lbl, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        # Remaining goes to test (approx 20%)
        
        train_slides.update(group[:n_train])
        val_slides.update(group[n_train:n_train+n_val])
        test_slides.update(group[n_train+n_val:])
        
    train_rows = [r for r in rows if r.slide_id in train_slides]
    val_rows = [r for r in rows if r.slide_id in val_slides]
    test_rows = [r for r in rows if r.slide_id in test_slides]
    
    logger.info(
        "Global paper split (slides): total=%d train=%d val=%d test=%d",
        len(slide_ids), len(train_slides), len(val_slides), len(test_slides)
    )
    logger.info(
        "Global paper split (patches): total=%d train=%d val=%d test=%d",
        len(rows), len(train_rows), len(val_rows), len(test_rows)
    )

    train_transform = _build_global_train_transform(augment=bool(augment), color_jitter=float(color_jitter))
    
    # Datasets
    # Note: GlobalScoringPatchDataset resizes to 227x227
    train_ds = GlobalScoringPatchDataset(rows=train_rows, transform=train_transform, normalize_imagenet=True)
    val_ds = GlobalScoringPatchDataset(rows=val_rows, transform=None, normalize_imagenet=True)
    test_ds = GlobalScoringPatchDataset(rows=test_rows, transform=None, normalize_imagenet=True)
    
    # Sampler for train? (Paper doesn't explicitly mention balancing but usually done)
    # Using 'balance' flag behavior.
    sampler = None
    shuffle = True
    if balance:
        labels = [int(r.label) for r in train_rows] # 1,2,3
        # If labels are 1-based, we can just use them as keys or map to 0-based for unique counting
        # Loader expects 0-based tensors but dataset does the conversion.
        # Here we just need counts.
        counts: dict[int, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        weights = [1.0 / float(counts[int(l)]) for l in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info("Global train balance: %s", str(counts))

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=shuffle, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    device_t = _resolve_torch_device(device)
    model = CNNGlobal(pretrained=bool(pretrained)).to(device_t)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=float(lr), 
        momentum=float(momentum), 
        weight_decay=float(weight_decay)
    )
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
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
            if max_steps > 0 and steps >= int(max_steps):
                break
                
        train_loss = train_loss_sum / max(1, steps)
        train_acc = float(correct / max(1, total))
        
        # Validation
        val_loss, val_acc = _eval_global(model=model, loader=val_loader, criterion=criterion, device=device_t)
        # Test (Monitor)
        test_loss, test_acc = _eval_global(model=model, loader=test_loader, criterion=criterion, device=device_t)
        
        logger.info(
            "Global paper epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f test_acc=%.3f",
            epoch_idx, train_loss, train_acc, val_loss, val_acc, test_acc
        )
        
        _append_metrics_row(metrics_csv, {
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    torch.save(
        {
            "model": "CNNGlobal",
            "paper_aligned": True,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "state_dict": model.state_dict()
        },
        checkpoint_path
    )
    logger.info("Saved CNN_global(paper) checkpoint: %s (best_val_acc=%.3f)", str(checkpoint_path), best_acc)


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
