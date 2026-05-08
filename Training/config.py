from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from step2_model import LossConfig


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path
    batch_size: int = 64
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    image_size: int = 32
    use_preprocess_pipeline: bool = True
    use_train_rotation: bool = True
    use_train_affine: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "baseline_cnn"
    output_root: Path = Path("Training") / "Trained_models"
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    loss: LossConfig = LossConfig(name="cross_entropy")


@dataclass
class EpochMetrics:
    logged_at_utc: str
    run_name: str
    model_name: str
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float


def build_default_configs(
    project_root: Path,
    *,
    model_name: str = "baseline_cnn",
    batch_size: int = 116,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    image_size: int = 32,
    use_preprocess_pipeline: bool = True,
    use_train_rotation: bool = True,
    use_train_affine: bool = True,
    max_epochs: int = 47,
    learning_rate: float = 0.003858921100781887,
    weight_decay: float = 2.8660953196330396e-05,
    early_stopping_patience: int = 10,
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> tuple[DataConfig, TrainingConfig]:
    data_config = DataConfig(
        train_dir=project_root / "dataset" / "train",
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        image_size=image_size,
        use_preprocess_pipeline=use_preprocess_pipeline,
        use_train_rotation=use_train_rotation,
        use_train_affine=use_train_affine,
    )

    training_config = TrainingConfig(
        model_name=model_name,
        output_root=project_root / "Training" / "Trained_models",
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        loss=LossConfig(name=loss_name, label_smoothing=label_smoothing),
    )

    return data_config, training_config