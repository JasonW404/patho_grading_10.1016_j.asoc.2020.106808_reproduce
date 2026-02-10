from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DetRunSummary:
    run_id: str
    metrics_csv: Path
    last_phase: str
    last_epoch: str
    best_val_ap: float
    best_val_epoch: Optional[int]
    best_test_ap: float
    best_test_epoch: Optional[int]


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def summarize_run(run_dir: Path) -> DetRunSummary:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    metrics_csv = run_dir / "metrics" / "det_metrics.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(f"missing metrics: {metrics_csv}")

    best_val_ap = float("-inf")
    best_val_epoch: Optional[int] = None
    best_test_ap = float("-inf")
    best_test_epoch: Optional[int] = None

    last_phase = ""
    last_epoch = ""

    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_phase = str(row.get("phase") or "")
            last_epoch = str(row.get("epoch") or "")

            phase = str(row.get("phase") or "")
            epoch_s = str(row.get("epoch") or "")
            try:
                epoch = int(epoch_s) if epoch_s else None
            except Exception:
                epoch = None

            val_ap = _safe_float(str(row.get("val_ap") or ""))
            test_ap = _safe_float(str(row.get("test_ap") or ""))

            if phase == "tune" and val_ap == val_ap and val_ap > best_val_ap:
                best_val_ap = float(val_ap)
                best_val_epoch = epoch

            if phase == "final" and test_ap == test_ap and test_ap > best_test_ap:
                best_test_ap = float(test_ap)
                best_test_epoch = epoch

    # Normalize -inf if never found
    if best_val_ap == float("-inf"):
        best_val_ap = float("nan")
    if best_test_ap == float("-inf"):
        best_test_ap = float("nan")

    return DetRunSummary(
        run_id=run_id,
        metrics_csv=metrics_csv,
        last_phase=last_phase,
        last_epoch=last_epoch,
        best_val_ap=best_val_ap,
        best_val_epoch=best_val_epoch,
        best_test_ap=best_test_ap,
        best_test_epoch=best_test_epoch,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compare CNN_det runs by metrics/det_metrics.csv")
    p.add_argument("run_dirs", nargs="+", help="One or more run directories under output/mf_cnn/CNN_det/runs/")
    args = p.parse_args()

    summaries = [summarize_run(Path(d)) for d in args.run_dirs]

    for s in summaries:
        print(
            f"{s.run_id} last={s.last_phase}:{s.last_epoch} "
            f"best_val_ap={s.best_val_ap:.4f}@{s.best_val_epoch} "
            f"best_test_ap={s.best_test_ap:.4f}@{s.best_test_epoch} "
            f"metrics={s.metrics_csv}"
        )


if __name__ == "__main__":
    main()
