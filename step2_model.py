from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch.nn as nn
from torch import Tensor


class ConvBNReLU(nn.Module):
    """Reusable conv -> batchnorm -> ReLU unit.

    Keeping the block in one place makes the feature extractor easier to
    read and makes it obvious that every convolution is followed by the same
    normalization and non-linearity pattern.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class DevanagariCNN(nn.Module):
    """
    Step 2 model:
    - Conv block 1: 1 -> 32 (double conv), MaxPool, Dropout(0.25)
    - Conv block 2: 32 -> 64 (double conv), MaxPool, Dropout(0.25)
    - Conv block 3: 64 -> 128 (single conv), MaxPool, Dropout(0.25)
    - Global average pooling to compress spatial information into one vector
    - Dense: 256 with BN/ReLU/Dropout(0.5)
    - Output: 10 classes

    The architecture stays compact, but global average pooling makes the
    classifier less dependent on the exact input resolution and reduces the
    number of fully connected parameters.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 32) -> None:
        super().__init__()

        # The convolutional stack extracts local stroke and texture patterns.
        self.features = nn.Sequential(
            nn.Sequential(
                ConvBNReLU(1, 32),
                ConvBNReLU(32, 32),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.25),
            ),
            nn.Sequential(
                ConvBNReLU(32, 64),
                ConvBNReLU(64, 64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.25),
            ),
            nn.Sequential(
                ConvBNReLU(64, 128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.25),
            ),
        )

        # Adaptive pooling removes the need to hard-code the flattened size.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Feature extractor: local edges -> strokes -> broader glyph patterns.
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


@dataclass(frozen=True)
class LossConfig:
    """Configuration for creating loss functions."""

    name: str = "cross_entropy"
    label_smoothing: float = 0.0


def build_loss(config: LossConfig) -> nn.Module:
    """
    Modular loss factory.

    Supported names:
    - cross_entropy (default)
    - cross_entropy_ls

    This keeps the training script decoupled from the exact loss choice and
    makes it easy to switch to label smoothing without touching the trainer.
    """

    builders: Dict[str, Callable[[LossConfig], nn.Module]] = {
        "cross_entropy": lambda c: nn.CrossEntropyLoss(),
        "cross_entropy_ls": lambda c: nn.CrossEntropyLoss(label_smoothing=c.label_smoothing),
    }

    if config.name not in builders:
        supported = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unsupported loss '{config.name}'. Supported: {supported}")

    return builders[config.name](config)


def run_dummy_forward_pass(batch_size: int = 8, image_size: int = 32, num_classes: int = 10) -> Tensor:
    """Run a quick shape check to verify the model wiring.

    This is intentionally lightweight: it does not test accuracy, only that
    tensors flow through the network and the classifier returns logits with the
    expected batch and class dimensions.
    """

    model = DevanagariCNN(num_classes=num_classes, input_size=image_size)
    model.eval()
    dummy = torch.randn(batch_size, 1, image_size, image_size)
    with torch.no_grad():
        logits = model(dummy)

    expected_shape = (batch_size, num_classes)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Dummy forward failed: got {tuple(logits.shape)}, expected {expected_shape}")

    return logits


if __name__ == "__main__":
    logits = run_dummy_forward_pass()
    loss_fn = build_loss(LossConfig(name="cross_entropy"))
    print("Dummy forward OK. Logits shape:", tuple(logits.shape))
    print("Loss function:", loss_fn.__class__.__name__)
