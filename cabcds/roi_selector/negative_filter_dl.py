"""Deep-learning assisted negative sampling filter.

Goal
----
Train a lightweight CNN to distinguish:
- good negatives (keep) vs
- bad negatives / suspicious (reject)

This model is used during negative patch sampling to bias generated negatives toward
"good_negative" patches.

Input format
------------
We train from a simplified JSON list, where each item looks like:
  {"id": 123, "image_name": "...png", "image_path": "/abs/path.png", "label": "good_negative"}

We intentionally avoid requiring Label Studio's full export schema here.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ROISelectorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DlNegFilterModelBundle:
    model: Any
    device: Any
    input_size: int


def train_negative_filter_dl(labels_json: Path, config: ROISelectorConfig) -> Path:
    """Train a CNN-based negative filter from simplified JSON labels.

    Saves model to config.neg_filter_dl_model_path.

    Args:
        labels_json: Path to simplified label JSON.
        config: ROI selector config.

    Returns:
        Path to saved model.
    """

    # Lazy imports to keep CLI light when not training.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    labels_json = Path(labels_json).expanduser().resolve()
    items = json.loads(labels_json.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("labels_json must be a JSON list")

    x_paths: list[Path] = []
    y: list[int] = []

    # 1) Use manually labelled negatives (good/bad)
    for it in items:
        label = (it.get("label") or "").strip().lower()
        img_path = it.get("image_path")
        if not img_path:
            continue
        p = Path(str(img_path)).expanduser().resolve()
        if not p.exists():
            continue

        if label in {"good", "good_negative"}:
            x_paths.append(p)
            y.append(0)
        elif label in {"bad", "bad_negative"}:
            x_paths.append(p)
            y.append(1)

    # 2) Use positive patches as additional "bad/suspicious" examples.
    # This helps the filter learn to reject tumor-like tissue, improving "good-negative" purity.
    pos_dir = (config.train_dataset_dir / config.train_positive_subdir).expanduser().resolve()
    pos_paths: list[Path] = []
    if pos_dir.exists():
        pos_paths = sorted(pos_dir.glob("*.png")) + sorted(pos_dir.glob("*.jpg")) + sorted(pos_dir.glob("*.jpeg"))
        # Cap positives to keep training time reasonable and avoid overwhelming negatives.
        # Use at most ~2x the number of labelled samples.
        cap = max(200, 2 * len(x_paths))
        if len(pos_paths) > cap:
            rng = random.Random(123)
            rng.shuffle(pos_paths)
            pos_paths = pos_paths[:cap]
        for p in pos_paths:
            if p.exists():
                x_paths.append(p)
                y.append(1)

    if not x_paths:
        raise ValueError("No usable labelled samples found (check image_path and label fields)")

    if len(set(y)) < 2:
        raise ValueError("Need both good and bad labels to train DL negative filter")

    n_good = int(sum(1 for v in y if v == 0))
    n_bad = int(sum(1 for v in y if v == 1))
    logger.info("DL neg-filter training data: good=%d bad=%d (includes %d positive patches as bad)", n_good, n_bad, len(pos_paths))

    # Deterministic split
    rng = random.Random(42)
    idx = list(range(len(x_paths)))
    rng.shuffle(idx)
    split = int(0.85 * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]

    input_size = int(config.neg_filter_dl_input_size)

    class _PatchDs(Dataset):
        def __init__(self, indices: list[int]):
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            j = self.indices[i]
            path = x_paths[j]
            label = y[j]
            img = Image.open(path).convert("RGB")
            img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # Normalize to roughly zero-centered range.
            arr = (arr - 0.5) / 0.5
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
            return t, torch.tensor(label, dtype=torch.float32)

    train_ds = _PatchDs(train_idx)
    val_ds = _PatchDs(val_idx if val_idx else train_idx[: max(1, len(train_idx) // 10)])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.neg_filter_dl_batch_size),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.neg_filter_dl_batch_size),
        shuffle=False,
        num_workers=0,
    )

    class SmallCnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, x):
            x = self.net(x)
            x = x.flatten(1)
            return self.head(x).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCnn().to(device)

    # Handle imbalance: pos_weight = Nneg / Npos for BCEWithLogitsLoss
    n_pos = int(sum(1 for v in y if v == 1))
    n_neg = int(sum(1 for v in y if v == 0))
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=float(config.neg_filter_dl_lr))

    def _eval() -> tuple[float, float]:
        model.eval()
        losses: list[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(float(loss.item()))
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return (float(np.mean(losses)) if losses else 0.0), (correct / max(1, total))

    # Train
    epochs = int(config.neg_filter_dl_epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = _eval()
        logger.info("DL neg-filter epoch %d/%d val_loss=%.4f val_acc=%.3f", epoch, epochs, val_loss, val_acc)

    out_path = config.neg_filter_dl_model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "label_map": {"good_negative": 0, "bad_negative": 1},
        "meta": {"n_samples": len(x_paths), "n_good": n_neg, "n_bad": n_pos, "n_pos_patches": len(pos_paths)},
    }
    torch.save(payload, out_path)
    logger.info("Saved DL negative filter model to %s", out_path)
    return out_path


def load_negative_filter_dl_model(path: Path, *, device: str | None = None) -> DlNegFilterModelBundle:
    """Load a trained DL negative filter model bundle."""

    import torch
    import torch.nn as nn

    path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError as e:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raise e
    input_size = int(payload.get("input_size", 224))

    class SmallCnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, x):
            x = self.net(x)
            x = x.flatten(1)
            return self.head(x).squeeze(1)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCnn().to(dev)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    # Avoid oversubscription when called from multiprocessing workers.
    torch.set_num_threads(1)

    return DlNegFilterModelBundle(model=model, device=dev, input_size=input_size)


def suspicious_probability_dl(bundle: DlNegFilterModelBundle, patch: np.ndarray) -> float:
    """Return P(suspicious) (bad_negative) for a patch."""

    import torch

    img = Image.fromarray(patch.astype(np.uint8), mode="RGB")
    img = img.resize((bundle.input_size, bundle.input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    t = t.to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(t)
        prob = torch.sigmoid(logits)
        return float(prob.item())
