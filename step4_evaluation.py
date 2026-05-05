from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dataloader.dataset_utils import build_dataloaders
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


@torch.no_grad()
def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.vstack(all_probs)
    return y_pred, y_true, y_prob


def _plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> None:
    cm_counts = confusion_matrix(y_true, y_pred)
    cm_row_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_row_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()


def _plot_one_sample_per_class(
    dataset,
    model: torch.nn.Module,
    class_names: Sequence[str],
    class_labels: Sequence[str],
    mean: float,
    std: float,
    device: torch.device,
) -> None:
    chosen_indices: Dict[int, int] = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_idx = int(label)
        if label_idx not in chosen_indices:
            chosen_indices[label_idx] = idx
        if len(chosen_indices) == len(class_names):
            break

    ordered_items = sorted(chosen_indices.items(), key=lambda x: x[0])
    if not ordered_items:
        print("No validation samples found to visualize.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for plot_idx, (class_idx, sample_idx) in enumerate(ordered_items):
            image_tensor, true_label = dataset[sample_idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())

            display_image = (image_tensor * std + mean).clamp(0.0, 1.0).cpu().numpy().squeeze(0)

            ax = axes[plot_idx]
            ax.imshow(display_image, cmap="gray")
            ax.axis("off")
            title_color = "red" if pred_idx != int(true_label) else "black"
            ax.set_title(
                f"T:{class_labels[int(true_label)]} P:{class_labels[pred_idx]}",
                color=title_color,
                fontsize=11,
            )

    for remaining in range(len(ordered_items), len(axes)):
        axes[remaining].axis("off")

    fig.suptitle("Validation Samples (One Per Class)", fontsize=14)
    plt.tight_layout()
    plt.show()


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

    y_pred, y_true, _ = _predict_loader(model=model, loader=setup.val_loader, device=device)

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

    _plot_confusion_matrices(y_true=y_true, y_pred=y_pred, class_labels=class_labels)

    _plot_one_sample_per_class(
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

    # Additionally create a submission that follows the sample_submission.csv structure/order
    sample_submission_path = project_root / "CSV files" / "sample_submission.csv"
    sample_output_path = model_run_dir / f"{args.model_name}_submission.csv"
    if sample_submission_path.exists():
        print(f"Creating sample-structured submission by merging full predictions: {sample_output_path}")

        # Read back the full predictions we just wrote into a mapping id->category
        full_pred_path = output_submission_path
        full_map: Dict[str, str] = {}
        if full_pred_path.exists():
            with full_pred_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    full_map[str(row["Id"]).strip()] = str(row["Category"]).strip()

        # Load sample IDs (preserve their order)
        sample_ids = _load_ids_from_test_csv(sample_submission_path)

        # Build output rows containing only the sample IDs (preserving order)
        out_rows: List[Tuple[str, str]] = []
        for sid in sample_ids:
            cat = full_map.get(sid, "")
            out_rows.append((sid, cat))

        # Write sample-only merged file
        sample_output_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Id", "Category"])
            writer.writerows(out_rows)

        print(f"Sample-structured submission (sample-only) written: {sample_output_path}")
        print(f"Rows written: {len(out_rows)}")

    else:
        print(f"Sample submission file not found: {sample_submission_path} — skipping sample-structured output.")


if __name__ == "__main__":
    main()
