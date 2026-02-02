"""MF-CNN utilities entry point.

This module focuses on *smoke checks* so the MF-CNN stack can be validated with
the data already present in this repo (TUPAC16 + auxiliary mitosis zips).

Example:
    `uv run python -m cabcds.mf_cnn --smoke`
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from cabcds.mf_cnn import CNNDet, CNNGlobal, CNNSeg, load_mfc_cnn_config
from cabcds.mf_cnn.preprocess import prepare_global_patch_dataset
from cabcds.mf_cnn.utils.loader import (
    GlobalScoringPatchDataset,
    GlobalPatchIndexRow,
    build_default_mf_cnn_train_loaders,
    PreparedMitosisDetectionPatchDataset,
    load_tupac_train_scores,
    read_global_patch_index,
)
from cabcds.mf_cnn.det_data import prepare_cnn_det_patches_from_aux_zips


logger = logging.getLogger(__name__)


def _resolve_torch_device(device: str) -> torch.device:
    """Resolve a torch device string including Ascend NPUs.

    Supported values:
    - cpu
    - cuda (NVIDIA)
    - mps (Apple)
    - npu (Huawei Ascend via torch_npu)
    """

    device = str(device).lower().strip()
    if device == "npu":
        try:
            # Import registers the `npu` device type in torch.
            import torch_npu  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Requested device='npu' but torch_npu could not be imported. "
                "This usually means the Ascend runtime libraries are not on LD_LIBRARY_PATH. "
                "Try: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` in the same shell, then rerun. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        # Most Ascend builds expose `torch.npu`. If present, try to select device 0.
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "set_device"):
            try:
                npu_mod.set_device(0)
            except Exception:
                pass

        return torch.device("npu:0")

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
        "--train-global",
        action="store_true",
        help="Train CNN_global from a prepared patch index CSV",
    )

    parser.add_argument("--train-seg", action="store_true", help="Train CNN_seg (VGG16-FCN) on mitosis auxiliary zips")
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

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seg-length", type=int, default=10_000)
    parser.add_argument("--det-length", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "npu"])
    args = parser.parse_args()

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

    torch.save({"model_state": model.state_dict(), "config": cfg.model_dump()}, checkpoint_path)
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

    torch.save({"model_state": model.state_dict(), "config": cfg.model_dump()}, checkpoint_path)
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
