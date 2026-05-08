from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Importing existing project modules
from step2_model import DevanagariCNN, build_loss, LossConfig, ConvBNReLU
from Training.config import DataConfig, TrainingConfig, build_default_configs
from Training.trainer import train_model
from dataloader.dataset_utils import build_dataloaders

"""
Hyperparameter tuning of the DevanagariCNN model using Optuna

GNU STP
"""


@dataclass
class OptimisationResult:
    """
    Stores results from a single hyperparameter optimisation run
    - params: Hyperparams config for the trial
    - val_accuracy: Final validation accuracy
    - val_loss: Final validation loss
    - best_epoch: Epoch with the best perf
    - training_time: Total training time
    - model_size_mb: Model size in MB
    - run_dir: Output dir path
    """
    params: Dict[str, Any]
    val_accuracy: float
    val_loss: float
    best_epoch: int
    training_time: float
    model_size_mb: float
    run_dir: str


class FlexibleCNN(DevanagariCNN):
    """
    A flexible CNN model that has configurable architecture for hyperparameter optimisation
    """

    def __init__(
            self,
            num_classes: int = 10,
            input_size: int = 32,
            conv_channels: Tuple[int, int, int] = (32, 64, 128),
            dropout_rates: Tuple[float, float] = (0.25, 0.5),
            dense_size: int = 256
    ) -> None:
        super().__init__(num_classes, input_size)

        from step2_model import ConvBNReLU
        import torch.nn as nn

        self.features = nn.Sequential(
            nn.Sequential(
                ConvBNReLU(1, conv_channels[0]),
                ConvBNReLU(conv_channels[0], conv_channels[0]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_rates[0]),
            ),
            nn.Sequential(
                ConvBNReLU(conv_channels[0], conv_channels[1]),
                ConvBNReLU(conv_channels[1], conv_channels[1]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_rates[0]),
            ),
            nn.Sequential(
                ConvBNReLU(conv_channels[1], conv_channels[2]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_rates[0]),
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[2], dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[1]),
            nn.Linear(dense_size, num_classes),
        )


class OptunaOptimiser:
    """
    Optuna-based hyperparameter optimisation for DevanagariCNN
    """

    def __init__(self, project_root: Path, output_dir: Path = None):
        self.project_root = project_root
        self.output_dir = output_dir or project_root / "optimisation_results"
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[OptimisationResult] = []
        self.optimisation_history = []

    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space and sample hyperparameters for a trial.

        Fixes vs old skopt approach:
        - scale_min/max are sampled dependently so min is always <= max
        - label_smoothing is only sampled when loss_name requires it
        - All integer types are plain Python ints (no np.int64 issues)
        """
        # Training hyperparams
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        max_epochs = trial.suggest_int('max_epochs', 20, 100)

        # Model architecture
        conv_channels_choice = trial.suggest_categorical(
            'conv_channels', ['16,32,64', '32,64,128', '64,128,256']
        )
        conv_channels = tuple(int(x) for x in conv_channels_choice.split(','))

        dropout_rates_choice = trial.suggest_categorical(
            'dropout_rates', ['0.2,0.4', '0.25,0.5', '0.3,0.6']
        )
        dropout_rates = tuple(float(x) for x in dropout_rates_choice.split(','))

        dense_size = trial.suggest_int('dense_size', 64, 512)

        # Data augmentation
        rotation_degrees = trial.suggest_int('rotation_degrees', 5, 30)
        translation_range = trial.suggest_float('translation_range', 0.05, 0.2)

        # scale_min and scale_max sampled dependently to guarantee min <= max
        scale_min = trial.suggest_float('scale_min', 0.7, 1.0)
        scale_max = trial.suggest_float('scale_max', 1.0, 1.3)
        scale_range = (scale_min, scale_max)

        # Loss config — label_smoothing only active for cross_entropy_ls
        loss_name = trial.suggest_categorical('loss_name', ['cross_entropy', 'cross_entropy_ls'])
        if loss_name == 'cross_entropy_ls':
            label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)
        else:
            # Fixed at 0 so it doesn't waste search budget when unused
            label_smoothing = 0.0

        return {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'conv_channels': conv_channels,
            'dropout_rates': dropout_rates,
            'dense_size': dense_size,
            'rotation_degrees': rotation_degrees,
            'translation_range': translation_range,
            'scale_range': scale_range,
            'loss_name': loss_name,
            'label_smoothing': label_smoothing,
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        Samples params, trains the model, reports intermediate values for pruning,
        and returns the best validation accuracy.
        """
        params = self.sample_params(trial)
        print(f"\n Testing params: {params}")

        try:
            result = self.train_single_config(
                params,
                run_name=f"optuna_trial_{trial.number}",
                trial=trial,
            )

            self.results.append(result)
            self.optimisation_history.append({
                'trial': trial.number,
                'params': {
                    **params,
                    # Serialise tuples so they survive JSON round-trips
                    'conv_channels': list(params['conv_channels']),
                    'dropout_rates': list(params['dropout_rates']),
                    'scale_range': list(params['scale_range']),
                },
                'val_accuracy': result.val_accuracy,
                'val_loss': result.val_loss,
            })

            print(f"Validation accuracy: {result.val_accuracy:.4f}")
            return result.val_accuracy

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Error training with parameters {params}: {e}")
            return 0.0

    def train_single_config(
        self,
        params: Dict[str, Any],
        run_name: str,
        trial: Optional[optuna.Trial] = None,
    ) -> OptimisationResult:
        """
        Train a single config and return the result.

        If a trial is passed, intermediate epoch accuracies are reported so
        Optuna's pruner can cut unpromising trials early.
        """
        start_time = time.time()

        data_cfg, train_cfg = build_default_configs(
            project_root=self.project_root,
            model_name=run_name,
            batch_size=params['batch_size'],
            val_ratio=0.15,
            seed=42,
            num_workers=0,
            image_size=32,
            use_preprocess_pipeline=True,
            use_train_rotation=True,
            use_train_affine=True,
            max_epochs=params['max_epochs'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            early_stopping_patience=10,
            loss_name=params['loss_name'],
            label_smoothing=params['label_smoothing'],
        )

        model = self.create_model_with_params(params)
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        # If train_model supports epoch callbacks, report intermediate values.
        # Enables Optuna's MedianPruner to cut bad trials early.
        epoch_callback = None
        if trial is not None:
            def epoch_callback(epoch: int, val_accuracy: float):
                trial.report(val_accuracy, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        training_results = train_model(data_cfg, train_cfg, epoch_callback=epoch_callback)
        training_time = time.time() - start_time

        return OptimisationResult(
            params=params,
            val_accuracy=max(m.val_accuracy for m in training_results['history']),
            val_loss=training_results['best_val_loss'],
            best_epoch=training_results['best_epoch'],
            training_time=training_time,
            model_size_mb=model_size_mb,
            run_dir=str(training_results['run_dir']),
        )

    def create_model_with_params(self, params: Dict[str, Any]) -> torch.nn.Module:
        """
        Create a model with the given hyperparams
        """
        return FlexibleCNN(
            num_classes=10,
            input_size=32,
            conv_channels=params['conv_channels'],
            dropout_rates=params['dropout_rates'],
            dense_size=params['dense_size'],
        )

    def run_optimisation(self, n_trials: int = 100, n_startup_trials: int = 10) -> List[OptimisationResult]:
        """
        Run Optuna optimisation.

        Args:
            n_trials: Total number of trials to run. 100 for reasonable for now, increase if we have enough time
            n_startup_trials: Number of random trials before TPE kicks in.
                              Equivalent to n_initial_points in skopt.
        """
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=42)

        # Starts pruning after 5 completed trials,
        # Checking at each epoch after the first 5 epochs of training.
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name='devanagari_cnn_tuning',
        )

        print(f"Running Optuna optimisation with {n_trials} trials ({n_startup_trials} random startup trials)...")

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.study = study

        print(f"\nOptimisation complete!")
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return self.results

    def save_results(self):
        """
        Save the optimisation results to files
        """
        # Detailed results as JSON
        results_dict = [asdict(result) for result in self.results]
        # Serialise any remaining tuples
        for r in results_dict:
            for k, v in r['params'].items():
                if isinstance(v, tuple):
                    r['params'][k] = list(v)
        with open(self.output_dir / "optuna_optimisation_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        # Optimisation history as JSON
        with open(self.output_dir / "optuna_optimisation_history.json", "w") as f:
            json.dump(self.optimisation_history, f, indent=2)

        # Summary as CSV
        summary_data = []
        for result in self.results:
            row = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in result.params.items()
            }
            row.update({
                'val_accuracy': result.val_accuracy,
                'val_loss': result.val_loss,
                'best_epoch': result.best_epoch,
                'training_time': result.training_time,
                'model_size_mb': result.model_size_mb,
            })
            summary_data.append(row)
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "optuna_optimisation_summary.csv", index=False)

        print(f"Results saved to: {self.output_dir}")

    def plot_optimisation_progress(self):
        """
        Plot optimisation progress and per-parameter scatter plots
        """
        if not self.optimisation_history:
            print("No optimisation history available.")
            return

        fig, ax = plt.subplots(2, 3, figsize=(18, 12))

        iterations = list(range(1, len(self.optimisation_history) + 1))
        accuracies = [h['val_accuracy'] for h in self.optimisation_history]
        best_so_far = np.maximum.accumulate(accuracies)

        # Plot 1: Optimisation progress
        ax[0, 0].plot(iterations, accuracies, 'bo-', alpha=0.6, label='Individual runs')
        ax[0, 0].plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
        ax[0, 0].set_xlabel('Trial')
        ax[0, 0].set_ylabel('Validation accuracy')
        ax[0, 0].set_title('Optuna Optimisation Progress')
        ax[0, 0].legend()
        ax[0, 0].grid(True, alpha=0.3)

        # Plot 2: Learning Rate vs Accuracy
        learning_rates = [h['params']['learning_rate'] for h in self.optimisation_history]
        ax[0, 1].scatter(learning_rates, accuracies, alpha=0.6)
        ax[0, 1].set_xscale('log')
        ax[0, 1].set_xlabel('Learning Rate')
        ax[0, 1].set_ylabel('Validation accuracy')
        ax[0, 1].set_title('Learning Rate vs Accuracy')
        ax[0, 1].grid(True, alpha=0.3)

        # Plot 3: Batch Size vs Accuracy
        batch_sizes = [h['params']['batch_size'] for h in self.optimisation_history]
        ax[0, 2].scatter(batch_sizes, accuracies, alpha=0.6)
        ax[0, 2].set_xlabel('Batch Size')
        ax[0, 2].set_ylabel('Validation accuracy')
        ax[0, 2].set_title('Batch Size vs Accuracy')
        ax[0, 2].grid(True, alpha=0.3)

        # Plot 4: Dense Size vs Accuracy
        dense_sizes = [h['params']['dense_size'] for h in self.optimisation_history]
        ax[1, 0].scatter(dense_sizes, accuracies, alpha=0.6)
        ax[1, 0].set_xlabel('Dense Layer Size')
        ax[1, 0].set_ylabel('Validation accuracy')
        ax[1, 0].set_title('Dense Size vs Accuracy')
        ax[1, 0].grid(True, alpha=0.3)

        # Plot 5: Weight Decay vs Accuracy
        weight_decays = [h['params']['weight_decay'] for h in self.optimisation_history]
        ax[1, 1].scatter(weight_decays, accuracies, alpha=0.6)
        ax[1, 1].set_xscale('log')
        ax[1, 1].set_xlabel('Weight Decay')
        ax[1, 1].set_ylabel('Validation accuracy')
        ax[1, 1].set_title('Weight Decay vs Accuracy')
        ax[1, 1].grid(True, alpha=0.3)

        # Plot 6: Convergence analysis
        window_size = min(10, len(accuracies) // 2)
        if window_size > 1:
            moving_avg = pd.Series(accuracies).rolling(window=window_size).mean()
            ax[1, 2].plot(iterations, accuracies, 'bo-', alpha=0.3, label='Individual')
            ax[1, 2].plot(iterations, moving_avg, 'r-', linewidth=2, label=f'Moving avg ({window_size})')
            ax[1, 2].set_xlabel('Trial')
            ax[1, 2].set_ylabel('Validation accuracy')
            ax[1, 2].set_title('Convergence Analysis')
            ax[1, 2].legend()
            ax[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "optuna_optimisation_plots.png", dpi=300, bbox_inches='tight')
        plt.show()

        # If plotly is installed, use Optuna's built-in visualisation
        try:
            from optuna.visualization import plot_param_importances, plot_optimization_history
            fig_importance = plot_param_importances(self.study)
            fig_importance.write_html(str(self.output_dir / "param_importances.html"))
            fig_history = plot_optimization_history(self.study)
            fig_history.write_html(str(self.output_dir / "optimization_history.html"))
            print("Optuna interactive plots saved to output directory.")
        except ImportError:
            print("Install plotly for interactive Optuna visualisations: pip install plotly")

    def get_best_config(self) -> OptimisationResult:
        """
        Return the best config found
        """
        if not self.results:
            raise ValueError("No optimisation results available.")
        return max(self.results, key=lambda r: r.val_accuracy)


def main():
    """
    Main entry point for Optuna optimisation
    """
    project_root = Path(__file__).resolve().parent.parent

    optimiser = OptunaOptimiser(project_root, output_dir=Path(__file__).resolve().parent / "optimisation_results")

    results = optimiser.run_optimisation(
        n_trials=400,
        n_startup_trials=10,
    )

    optimiser.save_results()
    optimiser.plot_optimisation_progress()

    best_config = optimiser.get_best_config()
    print(f"\nBest Configuration Found:")
    print(f"Parameters: {best_config.params}")
    print(f"Validation Accuracy: {best_config.val_accuracy:.4f}")
    print(f"Training Time: {best_config.training_time:.2f}s")
    print(f"Model Size: {best_config.model_size_mb:.2f}MB")


if __name__ == "__main__":
    main()