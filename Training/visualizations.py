from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch import Tensor
from torch.utils.data import DataLoader

from .config import EpochMetrics


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predictions, true_labels, probabilities) for the entire loader."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels), np.vstack(all_probs)


def plot_per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Bar chart showing accuracy for each digit class."""
    n_classes = len(class_names)
    accs = []
    for i in range(n_classes):
        mask = labels == i
        accs.append(float((preds[mask] == i).mean()) if mask.sum() > 0 else 0.0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, accs, color="steelblue")
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2f}",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Heatmap confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> None:
    """Side-by-side confusion matrices with raw counts and row-normalized values."""
    cm_counts = confusion_matrix(y_true, y_pred)
    cm_row_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=class_labels)
    disp_counts.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    axes[0].set_title("Confusion Matrix (Raw Counts)")

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_row_norm, display_labels=class_labels)
    disp_norm.plot(ax=axes[1], cmap="Greens", values_format=".2f", colorbar=False)
    axes[1].set_title("Confusion Matrix (Row-Normalized)")

    plt.tight_layout()
    plt.show()


def plot_one_sample_per_class(
    dataset,
    model: torch.nn.Module,
    class_names: Sequence[str],
    class_labels: Sequence[str],
    mean: float,
    std: float,
    device: torch.device,
) -> None:
    """Show one validation example for each class with predicted labels."""
    chosen_indices: dict[int, int] = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_idx = int(label)
        if label_idx not in chosen_indices:
            chosen_indices[label_idx] = idx
        if len(chosen_indices) == len(class_names):
            break

    ordered_items = sorted(chosen_indices.items(), key=lambda item: item[0])
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


def plot_training_curves(history: List[EpochMetrics], output_path: Path) -> None:
    """Save training and validation curves to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [metrics.epoch for metrics in history]
    train_loss = [metrics.train_loss for metrics in history]
    val_loss = [metrics.val_loss for metrics in history]
    train_acc = [metrics.train_accuracy for metrics in history]
    val_acc = [metrics.val_accuracy for metrics in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_vs_validation(history: List[EpochMetrics]) -> None:
    """Dual plot: loss and accuracy over epochs."""
    epochs = [m.epoch for m in history]
    train_loss = [m.train_loss for m in history]
    val_loss = [m.val_loss for m in history]
    train_acc = [m.train_accuracy for m in history]
    val_acc = [m.val_accuracy for m in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "b-o", markersize=3, label="Train Accuracy")
    axes[1].plot(epochs, val_acc, "r-o", markersize=3, label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """One-vs-Rest ROC curves with AUC for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(class_names):
        binary = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """One-vs-Rest Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(class_names):
        binary = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary, probs[:, i])
        ax.plot(recall, precision, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


__all__ = [
    "collect_predictions",
    "plot_confusion_matrix",
    "plot_confusion_matrices",
    "plot_one_sample_per_class",
    "plot_per_class_accuracy",
    "plot_precision_recall_curves",
    "plot_roc_curves",
    "plot_training_curves",
]