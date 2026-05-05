from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataloader.dataset_utils import build_dataloaders
from step2_model import DevanagariCNN, build_loss

from .config import DataConfig, EpochMetrics, TrainingConfig
from .runtime import CsvLogger, format_epoch_message, make_epoch_metrics, resolve_run_paths
from .visualizations import plot_training_curves


@torch.no_grad()
def evaluate(model: DevanagariCNN, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_count += int(labels.size(0))

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def train_one_epoch(
    model: DevanagariCNN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_count += int(labels.size(0))

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def save_checkpoint(
    checkpoint_path: Path,
    model: DevanagariCNN,
    optimizer: Adam,
    epoch: int,
    best_val_loss: float,
    class_names: List[str],
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "data_config": {
            **asdict(data_config),
            "train_dir": str(data_config.train_dir),
        },
        "training_config": {
            **asdict(training_config),
            "output_root": str(training_config.output_root),
            "loss": asdict(training_config.loss),
        },
    }
    torch.save(payload, checkpoint_path)


def train_model(data_config: DataConfig, training_config: TrainingConfig) -> Dict[str, object]:
    setup = build_dataloaders(
        train_dir=data_config.train_dir,
        batch_size=data_config.batch_size,
        val_ratio=data_config.val_ratio,
        seed=data_config.seed,
        num_workers=data_config.num_workers,
        image_size=data_config.image_size,
        use_preprocess_pipeline=data_config.use_preprocess_pipeline,
        use_train_rotation=data_config.use_train_rotation,
        use_train_affine=data_config.use_train_affine,
    )

    run_paths = resolve_run_paths(training_config.output_root, training_config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DevanagariCNN(num_classes=10, input_size=data_config.image_size).to(device)

    criterion = build_loss(training_config.loss)
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    csv_logger = CsvLogger(run_paths.csv_log_path)

    print(f"Training model: {training_config.model_name}")
    print(f"Run directory: {run_paths.run_dir}")
    print("Artifacts: best_model.pt, training_log.csv, training_curves.png")

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=setup.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=setup.val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        metrics = make_epoch_metrics(
            run_paths=run_paths,
            model_name=training_config.model_name,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            learning_rate=current_lr,
        )
        history.append(metrics)
        csv_logger.log(metrics)

        print(format_epoch_message(metrics))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=run_paths.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                class_names=setup.class_names,
                data_config=data_config,
                training_config=training_config,
            )
            print("  best checkpoint updated")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement ({epochs_without_improvement}/{training_config.early_stopping_patience})")

        if epochs_without_improvement >= training_config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    plot_training_curves(history, run_paths.curves_path)
    print("Training complete. Curves and log were saved in the run directory.")
    print(f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "run_dir": run_paths.run_dir,
        "checkpoint_path": run_paths.checkpoint_path,
        "csv_log_path": run_paths.csv_log_path,
        "curves_path": run_paths.curves_path,
    }