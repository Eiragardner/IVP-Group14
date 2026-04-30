from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]+')


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint_path: Path
    csv_log_path: Path
    curves_path: Path
    run_name: str
    started_at_utc: str


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def log(self, metrics) -> None:
        fieldnames = [
            "logged_at_utc",
            "run_name",
            "model_name",
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "learning_rate",
        ]

        mode = "a" if self._initialized and self.path.exists() else "w"
        with self.path.open(mode, newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerow(asdict(metrics))
        self._initialized = True


def sanitize_path_component(value: str) -> str:
    cleaned = INVALID_PATH_CHARS.sub("_", value.strip())
    return cleaned or "model"


def resolve_run_paths(output_root: Path, model_name: str) -> RunPaths:
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_path_component(model_name)
    run_name = base_name
    run_dir = output_root / run_name
    version = 0

    while run_dir.exists():
        version += 1
        run_name = f"{base_name} (v{version})"
        run_dir = output_root / run_name

    run_dir.mkdir(parents=True, exist_ok=False)

    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return RunPaths(
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
        csv_log_path=run_dir / "training_log.csv",
        curves_path=run_dir / "training_curves.png",
        run_name=run_name,
        started_at_utc=started_at_utc,
    )


def make_epoch_metrics(run_paths: RunPaths, model_name: str, epoch: int, train_loss: float, val_loss: float, train_accuracy: float, val_accuracy: float, learning_rate: float):
    from .config import EpochMetrics

    return EpochMetrics(
        logged_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_name=run_paths.run_name,
        model_name=model_name,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        learning_rate=learning_rate,
    )


def format_epoch_message(metrics) -> str:
    return (
        f"Epoch {metrics.epoch:03d} | "
        f"train loss {metrics.train_loss:.4f}, train acc {metrics.train_accuracy:.4f} | "
        f"val loss {metrics.val_loss:.4f}, val acc {metrics.val_accuracy:.4f} | "
        f"lr {metrics.learning_rate:.6f}"
    )