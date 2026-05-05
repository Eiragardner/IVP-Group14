from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataloader.dataset_utils import build_dataloaders
from step2_model import DevanagariCNN


DIGIT_UNICODE = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: Interactive single-image inference or folder scan")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_cnn",
        help="Model run folder under Training/Trained_models",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside selected model folder",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional single image path for one-shot inference. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Optional folder to scan recursively for PNG files and print predictions/confidences.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Number of decimal places to print for confidence values.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only report images with predicted confidence greater than this value.",
    )
    parser.add_argument(
        "--show-unicode",
        action="store_true",
        help="Show Devanagari digit alongside numeric label (if terminal/font supports it).",
    )
    return parser.parse_args()


def _load_model_and_transform(
    model_name: str,
    checkpoint_name: str,
) -> tuple[torch.nn.Module, object, Sequence[str], torch.device]:
    project_root = Path.cwd().resolve()
    model_run_dir = project_root / "Training" / "Trained_models" / model_name
    checkpoint_path = model_run_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    data_cfg = checkpoint["data_config"]

    # Rebuild dataloaders to recover the exact validation preprocessing pipeline.
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

    model = DevanagariCNN(
        num_classes=len(setup.class_names),
        input_size=int(data_cfg.get("image_size", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Validation transform = preprocessing + normalization, no augmentation.
    val_transform = setup.val_loader.dataset.transform
    return model, val_transform, setup.class_names, device


def _plot_probabilities(probabilities: torch.Tensor, class_names: Sequence[str]) -> None:
    probs = probabilities.detach().cpu().numpy()
    x_labels = [str(name) for name in class_names]

    plt.figure(figsize=(9, 4))
    bars = plt.bar(x_labels, probs, color="steelblue")
    plt.ylim(0, 1.0)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Class Probabilities")
    for idx, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{probs[idx]:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.show()


def _format_confidence(confidence: float, precision: int) -> str:
    return f"{confidence:.{precision}f}"


def predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    transform,
    class_names: Sequence[str],
    device: torch.device,
    show_unicode: bool = False,
    precision: int = 12,
    plot_probabilities: bool = True,
    verbose: bool = True,
) -> tuple[str, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    predicted_label = str(class_names[pred_idx])
    if verbose:
        if show_unicode:
            try:
                predicted_unicode = DIGIT_UNICODE[int(predicted_label)]
                print(f"Predicted numeral: {predicted_label} ({predicted_unicode})")
            except (ValueError, IndexError):
                print(f"Predicted numeral: {predicted_label}")
        else:
            print(f"Predicted numeral: {predicted_label}")

        print(f"Confidence: {_format_confidence(confidence, precision)}")

    if plot_probabilities:
        _plot_probabilities(probabilities=probs, class_names=class_names)

    return predicted_label, confidence


def scan_folder_for_predictions(
    folder_path: Path,
    model: torch.nn.Module,
    transform,
    class_names: Sequence[str],
    device: torch.device,
    show_unicode: bool,
    precision: int,
    min_confidence: float,
) -> None:
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    png_files = sorted(folder_path.rglob("*.png"))
    if not png_files:
        print(f"No PNG files found in: {folder_path}")
        return

    print(f"Scanning {len(png_files)} PNG files in: {folder_path}")
    matches = 0

    for image_path in png_files:
        predicted_label, confidence = predict_single_image(
            image_path=image_path,
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
            show_unicode=show_unicode,
            precision=precision,
            plot_probabilities=False,
            verbose=False,
        )

        if confidence > min_confidence:
            matches += 1
            print(f"File: {image_path} -> class {predicted_label}, confidence {_format_confidence(confidence, precision)}")

    print(f"Finished scan. Files above threshold ({min_confidence}): {matches}/{len(png_files)}")


def interactive_loop(
    model: torch.nn.Module,
    transform,
    class_names: Sequence[str],
    device: torch.device,
    show_unicode: bool,
) -> None:
    print("Type an image path to predict, or type 'quit' to exit.")

    while True:
        user_input = input("Image path> ").strip()
        if user_input.lower() == "quit":
            print("Exiting inference loop.")
            break
        if not user_input:
            continue

        try:
            predict_single_image(
                image_path=Path(user_input),
                model=model,
                transform=transform,
                class_names=class_names,
                device=device,
                show_unicode=show_unicode,
            )
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> None:
    args = parse_args()

    model, transform, class_names, device = _load_model_and_transform(
        model_name=args.model_name,
        checkpoint_name=args.checkpoint_name,
    )

    print("=" * 70)
    print(f"Model run: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_name}")
    print(f"Device: {device}")
    print("=" * 70)

    if args.folder and args.image:
        raise ValueError("Use only one of --image or --folder, not both.")

    if args.folder:
        scan_folder_for_predictions(
            folder_path=Path(args.folder),
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
            show_unicode=args.show_unicode,
            precision=args.precision,
            min_confidence=args.min_confidence,
        )
        return

    if args.image:
        predict_single_image(
            image_path=Path(args.image),
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
            show_unicode=args.show_unicode,
            precision=args.precision,
        )
        return

    interactive_loop(
        model=model,
        transform=transform,
        class_names=class_names,
        device=device,
        show_unicode=args.show_unicode,
    )


if __name__ == "__main__":
    main()
