from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# 0.1 radians ~= 5.73 degrees, and torchvision's shear parameter uses degrees.
SHEAR_DEGREES = 5.73


class OtsuCenterCropResize:
    """Convert to grayscale, apply Otsu binarization, tight-center crop glyph, then resize."""

    def __init__(self, size: int = 32, margin: int = 2) -> None:
        self.size = size
        self.margin = margin

    @staticmethod
    def _otsu_threshold(image: np.ndarray) -> int:
        hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0

        prob = hist / total
        omega = np.cumsum(prob)
        mu = np.cumsum(prob * np.arange(256))
        mu_total = mu[-1]

        denominator = omega * (1.0 - omega)
        denominator[denominator == 0] = 1e-12
        sigma_between = ((mu_total * omega - mu) ** 2) / denominator
        return int(np.argmax(sigma_between))

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)

        threshold = self._otsu_threshold(arr)
        binary = (arr > threshold).astype(np.uint8)

        ones_count = int(binary.sum())
        zeros_count = int(binary.size - ones_count)
        fg_mask = binary == 1 if ones_count <= zeros_count else binary == 0

        ys, xs = np.where(fg_mask)
        if ys.size == 0 or xs.size == 0:
            processed = Image.fromarray(arr, mode="L")
            return processed.resize((self.size, self.size), Image.Resampling.BILINEAR)

        y_min = max(0, int(ys.min()) - self.margin)
        y_max = min(arr.shape[0], int(ys.max()) + self.margin + 1)
        x_min = max(0, int(xs.min()) - self.margin)
        x_max = min(arr.shape[1], int(xs.max()) + self.margin + 1)

        crop_mask = fg_mask[y_min:y_max, x_min:x_max]
        glyph = np.zeros(crop_mask.shape, dtype=np.uint8)
        glyph[crop_mask] = 255

        h, w = glyph.shape
        side = max(h, w)
        square = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - h) // 2
        x_offset = (side - w) // 2
        square[y_offset : y_offset + h, x_offset : x_offset + w] = glyph

        processed = Image.fromarray(square, mode="L")
        return processed.resize((self.size, self.size), Image.Resampling.BILINEAR)


class ImageFolderSubset(Dataset[Tuple[Tensor, int]]):
    """Subset wrapper that applies a transform to selected ImageFolder indices."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices: Sequence[int],
        transform: Callable[[Image.Image], Tensor] | Callable[[Image.Image], Image.Image],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        sample_idx = self.indices[idx]
        path, target = self.base_dataset.samples[sample_idx]
        img = self.base_dataset.loader(path)
        out = self.transform(img)
        if not isinstance(out, Tensor):
            raise TypeError("Transform must return a torch.Tensor.")
        return out, target


@dataclass
class DataSetup:
    train_loader: DataLoader[Tuple[Tensor, int]]
    val_loader: DataLoader[Tuple[Tensor, int]]
    class_names: List[str]
    class_counts: Dict[str, int]
    train_size: int
    val_size: int
    total_samples: int
    mean: float
    std: float


def stratified_split_indices(
    targets: Sequence[int], val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    by_class: Dict[int, List[int]] = {}
    for i, t in enumerate(targets):
        by_class.setdefault(t, []).append(i)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, indices in by_class.items():
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_ratio))
        val_count = max(1, min(len(indices) - 1, val_count))

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_mean_std(dataset: Dataset[Tuple[Tensor, int]], batch_size: int = 256) -> Tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for images, _ in loader:
        pixel_sum += images.sum().item()
        pixel_sq_sum += (images ** 2).sum().item()
        pixel_count += images.numel()

    mean = pixel_sum / pixel_count
    variance = max((pixel_sq_sum / pixel_count) - (mean ** 2), 1e-12)
    std = math.sqrt(variance)
    return float(mean), float(std)


def _build_base_preprocess(image_size: int, use_otsu_crop_resize: bool) -> List[Callable[[Image.Image], Image.Image]]:
    if use_otsu_crop_resize:
        return [OtsuCenterCropResize(size=image_size)]

    return [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]


def build_dataloaders(
    train_dir: Path,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    image_size: int = 32,
    use_otsu_crop_resize: bool = True,
    use_train_rotation: bool = True,
    use_train_affine: bool = True,
) -> DataSetup:
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    base_dataset = datasets.ImageFolder(root=str(train_dir))
    class_names = list(base_dataset.classes)
    targets = [target for _, target in base_dataset.samples]

    train_indices, val_indices = stratified_split_indices(
        targets=targets, val_ratio=val_ratio, seed=seed
    )

    base_preprocess = _build_base_preprocess(image_size=image_size, use_otsu_crop_resize=use_otsu_crop_resize)

    preprocess_only = transforms.Compose(base_preprocess + [transforms.ToTensor()])

    stats_dataset = ImageFolderSubset(base_dataset, train_indices, preprocess_only)
    mean, std = compute_mean_std(stats_dataset)

    train_transform_steps = list(base_preprocess)
    if use_train_rotation:
        train_transform_steps.append(transforms.RandomRotation(degrees=15, fill=0))
    if use_train_affine:
        train_transform_steps.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-SHEAR_DEGREES, SHEAR_DEGREES),
                fill=0,
            )
        )
    train_transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    train_transform = transforms.Compose(train_transform_steps)

    val_transform = transforms.Compose(
        base_preprocess + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    )

    train_dataset = ImageFolderSubset(base_dataset, train_indices, train_transform)
    val_dataset = ImageFolderSubset(base_dataset, val_indices, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_counter = Counter()
    for _, target in base_dataset.samples:
        class_counter[class_names[target]] += 1

    return DataSetup(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        class_counts=dict(class_counter),
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        total_samples=len(base_dataset),
        mean=mean,
        std=std,
    )


def show_one_batch(
    loader: DataLoader[Tuple[Tensor, int]],
    class_names: Sequence[str],
    mean: float,
    std: float,
    max_images: int = 64,
) -> None:
    images, labels = next(iter(loader))
    images = images[:max_images]
    labels = labels[:max_images]

    images = images * std + mean
    images = images.clamp(0.0, 1.0)

    grid_size = int(math.ceil(math.sqrt(images.size(0))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= images.size(0):
            continue
        img = images[i].squeeze(0).cpu().numpy()
        label = class_names[int(labels[i].item())]
        ax.imshow(img, cmap="gray")
        ax.set_title(label, fontsize=8)

    fig.suptitle("One Training Batch After Preprocessing + Augmentation", fontsize=12)
    plt.tight_layout()
    plt.show()


def print_summary(setup: DataSetup) -> None:
    print("Dataset setup summary")
    print("=" * 60)
    print(f"Total samples found: {setup.total_samples}")
    print("Samples per class:")
    for class_name in sorted(setup.class_counts.keys(), key=lambda x: int(x)):
        print(f"  Class {class_name}: {setup.class_counts[class_name]}")
    print(f"Train split size: {setup.train_size}")
    print(f"Validation split size: {setup.val_size}")
    print(f"Computed train mean: {setup.mean:.6f}")
    print(f"Computed train std:  {setup.std:.6f}")
