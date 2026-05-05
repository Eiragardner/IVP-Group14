from __future__ import annotations

from pathlib import Path

from Training.config import build_default_configs
from Training.trainer import train_model


MODEL_NAME = "baseline_cnn"
BATCH_SIZE = 64
VAL_RATIO = 0.15
SEED = 42
NUM_WORKERS = 0
IMAGE_SIZE = 32
USE_PREPROCESS_PIPELINE = True
USE_TRAIN_ROTATION = True
USE_TRAIN_AFFINE = True
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EARLY_STOPPING_PATIENCE = 10
LOSS_NAME = "cross_entropy"
LABEL_SMOOTHING = 0.0


def main() -> None:
    project_root = Path.cwd().resolve()
    data_cfg, train_cfg = build_default_configs(
        project_root=project_root,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        use_preprocess_pipeline=USE_PREPROCESS_PIPELINE,
        use_train_rotation=USE_TRAIN_ROTATION,
        use_train_affine=USE_TRAIN_AFFINE,
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        loss_name=LOSS_NAME,
        label_smoothing=LABEL_SMOOTHING,
    )
    train_model(data_cfg, train_cfg)


if __name__ == "__main__":
    main()
