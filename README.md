# EfficientNet-B0 Land Use Classification

This project implements a land use classification pipeline using a pretrained EfficientNet-B0 architecture. It is designed to work with the UC Merced Land Use dataset but can be adapted for other image classification tasks.

## Overview

The script [proposed\_\_model.py](proposed__model.py) performs the following:

- Loads and preprocesses the UC Merced Land Use dataset.
- Fine-tunes a pretrained EfficientNet-B0 model.
- Automatically handles training and validation splits.
- Generates comprehensive performance visualizations and metric reports.
- Saves the best-performing model based on validation loss.

## Requirements

To run this project, you need the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pillow`

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

## Configuration

The model behavior can be customized in the `Config` class within [proposed\_\_model.py](proposed__model.py):

- `DATA_DIR`: Path to the UC Merced Land Use dataset (default: `./UC_Merced_LandUse`).
- `OUTPUT_DIR`: Directory where models and plots are saved (default: `./outputs`).
- `BATCH_SIZE`: Number of samples per training batch (default: `32`).
- `NUM_EPOCHS`: Number of training iterations (default: `10`).
- `LEARNING_RATE`: Initial learning rate for AdamW optimizer (default: `1e-4`).
- `WEIGHT_DECAY`: Weight decay for regularization (default: `1e-4`).
- `TRAIN_SPLIT`: Ratio of data used for training (default: `0.8`).

## Usage

1. **Prepare Data**: Ensure the UC Merced Land Use dataset is located in the directory specified by `Config.DATA_DIR`.
2. **Run Training**: Execute the script using Python:
   ```bash
   python proposed__model.py
   ```

## Outputs

After execution, the results are stored in the `./outputs` directory (or as configured):

### Model & History

- `efficientnet_best.pth`: The weights of the best performing model.
- `training_history.json`: Epoch-wise accuracy and loss values for both training and validation.

### Metrics & Reports

- `classification_report.txt`: Detailed per-class precision, recall, and F1-score.
- `metrics.json`: Classification metrics in a machine-readable format.

### Visualizations

- `learning_curves.png`: Plots showing accuracy and loss trends over epochs.
- `confusion_matrix.png`: A visualization of the model's predictions vs. actual labels.
- `prediction_distribution.png`: A bar chart showing the frequency of predicted classes.

## Model Details

- **Architecture**: EfficientNet-B0 (Pretrained on ImageNet).
- **Optimizer**: AdamW.
- **Loss Function**: Cross-Entropy Loss.
- **Input Resolution**: 224x224 pixels.
- **Normalization**: ImageNet standard (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
