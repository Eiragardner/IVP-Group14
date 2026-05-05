# IVP-Group14

Handwritten Hindi (Devanagari) numeral recognition using PyTorch.

This project performs 10-class classification for digits 0-9 using grayscale images and a CNN pipeline built step-by-step.

## Project Structure

- dataset/train: class folders 0-9 used for training and validation split
- dataset/test: unlabeled test images used for submission predictions
- CSV files/train.csv: labeled metadata (reference)
- CSV files/test.csv: test IDs to predict for submission
- CSV files/sample_submission.csv: expected submission format
- Step1_Dataset_DataLoader.ipynb: interactive notebook for dataset and dataloader setup
- dataloader/dataset_utils.py: reusable preprocessing and dataloader logic
- step2_model.py: CNN architecture and loss helpers
- step3_training.py: training entrypoint
- step4_evaluation.py: validation evaluation, confusion matrices, and submission generation
- step5_inference.py: single-image interactive inference utility
- Training/Trained_models: per-model training/evaluation artifacts

## Environment Setup

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional notebook setup:

```bash
python -m ipykernel install --user --name ivp-group14 --display-name "Python (ivp-group14)"
```

## Step 1: Dataset and DataLoader

Use the notebook for Step 1:

- Open Step1_Dataset_DataLoader.ipynb
- Run dependency install/version check cells
- Run dataloader setup cells
- Verify printed summary and image grid

What Step 1 does:

1. Loads dataset from dataset/train using folder labels 0-9
2. Applies preprocessing pipeline from dataloader/dataset_utils.py
3. Creates stratified train/validation split (default val_ratio = 0.15)
4. Computes mean/std from training split only
5. Applies train-only augmentation:
- RandomRotation (+/-15)
- RandomAffine (translation, slight zoom, slight shear)
6. Builds train and validation DataLoaders

Outputs to verify:

- total samples
- per-class counts
- train/val split sizes
- one processed training batch visualization

## Step 2: Model Architecture

Run:

```bash
python step2_model.py
```

What it provides:

1. DevanagariCNN model definition
2. Reusable loss builder (cross-entropy variants)
3. Dummy forward-pass shape sanity check

## Step 3: Training

Run:

```bash
python step3_training.py
```

Training configuration is defined in step3_training.py and mapped through Training/config.py.

Main behavior:

1. Loads dataloaders using the Step 1 preprocessing setup
2. Trains with Adam optimizer
3. Uses ReduceLROnPlateau scheduler
4. Uses early stopping based on validation loss
5. Saves best checkpoint and logs

Artifacts are saved under a model-specific folder:

- Training/Trained_models/<model_name>/best_model.pt
- Training/Trained_models/<model_name>/training_log.csv
- Training/Trained_models/<model_name>/training_curves.png

Example model folders currently present:

- Training/Trained_models/baseline_cnn
- Training/Trained_models/OLD Baseline CNN

## Step 4: Evaluation and Submission

Run default model:

```bash
python step4_evaluation.py
```

Run a specific model folder:

```bash
python step4_evaluation.py --model-name "OLD Baseline CNN"
```

What Step 4 computes:

1. Loads selected checkpoint from Training/Trained_models/<model_name>
2. Rebuilds train/validation split from checkpoint data_config
3. Evaluates on the validation set
4. Prints:
- final validation accuracy
- per-class accuracy
- full classification report (precision, recall, F1)
5. Plots two confusion matrices side by side:
- raw counts
- row-normalized values
6. Displays one validation sample per class with true/pred labels (errors shown in red)
7. Creates submission CSV for test set IDs

Submission output path:

- Training/Trained_models/<model_name>/submission_step4.csv

How test submission is generated:

1. Reads IDs from CSV files/test.csv
2. Loads each image from dataset/test/<Id>.png
3. Applies the same validation transform/preprocessing
4. Runs inference with selected model
5. Writes Id,Category predictions to submission_step4.csv

## Confusion Matrices: What Data They Use

Yes, the confusion matrices in step4_evaluation.py are built from the validation set.

Exact flow:

1. build_dataloaders(...) reconstructs train/val split using parameters stored in checkpoint data_config (train_dir, val_ratio, seed, preprocessing flags)
2. _predict_loader(...) runs the model over setup.val_loader only
3. confusion_matrix(y_true, y_pred) gives raw-count matrix
4. confusion_matrix(y_true, y_pred, normalize="true") gives row-normalized matrix

Interpretation:

- Raw matrix cell [i, j]: number of class i samples predicted as class j
- Row-normalized cell [i, j]: fraction of class i predicted as class j

## End-to-End Order

1. Install dependencies
2. Run Step 1 notebook checks
3. Run Step 2 model sanity check
4. Train with Step 3
5. Evaluate and export submission with Step 4
6. Run single-image inference with Step 5

## Step 5: Inference Utility

Run interactive mode:

```bash
python step5_inference.py --model-name baseline_cnn
```

Run one image directly:

```bash
python step5_inference.py --model-name baseline_cnn --image "dataset/test/10002.png"
```

Optional Devanagari display (if your UI/font supports it):

```bash
python step5_inference.py --model-name baseline_cnn --image "dataset/test/10002.png" --show-unicode
```

What Step 5 does:

1. Loads selected checkpoint from Training/Trained_models/<model_name>
2. Rebuilds preprocessing settings from checkpoint data_config
3. Uses validation transform (preprocessing + normalization, no augmentation)
4. Predicts a single image and prints:
- predicted class label
- confidence score (top softmax probability)
5. Plots probability bars for all 10 classes
6. Supports an interactive loop until you type quit

## Notes for Team Usage

1. Keep repository root as current working directory when running scripts.
2. Use --model-name in Step 4 to evaluate the exact model folder you want.
3. Compare submission files inside each model folder to track model quality.