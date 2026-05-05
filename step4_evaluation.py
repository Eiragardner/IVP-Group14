from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dataloader.dataset_utils import build_dataloaders
from Training.visualizations import collect_predictions, plot_confusion_matrices, plot_one_sample_per_class
from step2_model import DevanagariCNN


class TestCsvDataset(Dataset[Tuple[Tensor, str]]):
    """Test dataset driven by CSV Id values and the validation transform."""

    def __init__(self, test_dir: Path, ids: Sequence[str], transform) -> None:
        self.test_dir = test_dir
        self.ids = list(ids)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        sample_id = self.ids[idx]
        image_path = self.test_dir / f"{sample_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        tensor = self.transform(image)
        if not isinstance(tensor, Tensor):
            raise TypeError("Validation transform must return a torch.Tensor.")
        return tensor, sample_id


def _class_labels(class_names: Sequence[str]) -> List[str]:
    return [str(name) for name in class_names]


def _load_ids_from_test_csv(test_csv_path: Path) -> List[str]:
    ids: List[str] = []
    with test_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row["Id"]).strip()
            if value:
                ids.append(value)
    return ids


@torch.no_grad()
def _create_submission_csv(
    model: torch.nn.Module,
    test_dir: Path,
    test_csv_path: Path,
    output_csv_path: Path,
    transform,
    class_names: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    test_ids = _load_ids_from_test_csv(test_csv_path)
    dataset = TestCsvDataset(test_dir=test_dir, ids=test_ids, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows: List[Tuple[str, str]] = []

    model.eval()
    for images, ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred_indices = logits.argmax(dim=1).cpu().tolist()

        for sample_id, pred_idx in zip(ids, pred_indices):
            rows.append((sample_id, class_names[int(pred_idx)]))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Category"])
        writer.writerows(rows)

    print(f"Submission file written: {output_csv_path}")
    print(f"Rows written: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: Evaluate checkpoint and create submission CSV")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_cnn",
        help="Model run folder under Training/Trained_models (example: baseline_cnn)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside the selected model folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    model_run_dir = project_root / "Training" / "Trained_models" / args.model_name
    checkpoint_path = model_run_dir / args.checkpoint_name
    test_csv_path = project_root / "CSV files" / "test.csv"
    # Final/god submission file: name it after the model (matches sample_submission.csv format)
    output_submission_path = model_run_dir / f"{args.model_name}.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 70)
    print(f"Evaluating model run: {args.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Submission output: {output_submission_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    data_cfg = checkpoint["data_config"]

    setup = build_dataloaders(
        train_dir=Path(data_cfg["train_dir"]),
        batch_size=int(data_cfg["batch_size"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(data_cfg["seed"]),
        num_workers=int(data_cfg["num_workers"]),
        image_size=int(data_cfg.get("image_size", 32)),
        use_preprocess_pipeline=bool(data_cfg.get("use_preprocess_pipeline", True)),
        use_train_rotation=bool(data_cfg.get("use_train_rotation", True)),
        use_train_affine=bool(data_cfg.get("use_train_affine", True)),
    )

    model = DevanagariCNN(num_classes=len(setup.class_names), input_size=int(data_cfg.get("image_size", 32))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred, y_true, _ = collect_predictions(model=model, loader=setup.val_loader, device=device)

    final_val_acc = float((y_pred == y_true).mean())
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    class_labels = _class_labels(setup.class_names)

    print("Per-class accuracy:")
    for class_idx, class_name in enumerate(setup.class_names):
        mask = y_true == class_idx
        class_acc = float((y_pred[mask] == class_idx).mean()) if int(mask.sum()) > 0 else 0.0
        print(f"  Class {class_labels[class_idx]}: {class_acc:.4f}")

    print("\nClassification report:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[class_labels[i] for i in range(len(setup.class_names))],
        digits=4,
        zero_division=0,
    )
    print(report)

    plot_confusion_matrices(y_true=y_true, y_pred=y_pred, class_labels=class_labels)

    plot_one_sample_per_class(
        dataset=setup.val_loader.dataset,
        model=model,
        class_names=setup.class_names,
        class_labels=class_labels,
        mean=setup.mean,
        std=setup.std,
        device=device,
    )

    # Submission helper based on CSV files/test.csv format.
    val_transform = setup.val_loader.dataset.transform
    _create_submission_csv(
        model=model,
        test_dir=project_root / "dataset" / "test",
        test_csv_path=test_csv_path,
        output_csv_path=output_submission_path,
        transform=val_transform,
        class_names=setup.class_names,
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        device=device,
    )


if __name__ == "__main__":
    main()
