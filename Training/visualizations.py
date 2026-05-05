from __future__ import annotations

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