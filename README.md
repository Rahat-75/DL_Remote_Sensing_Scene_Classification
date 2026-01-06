# DL_Remote_Sensing_Scene_Classification

This repository contains the official PyTorch implementation of an EfficientNet-B0–based deep learning model for scene classification in remote sensing imagery.
The implementation corresponds to the published research work titled:

**“Deep Learning Approaches for Scene Classification in Remote Sensing Imagery”**

The project demonstrates that EfficientNet-B0 provides an excellent balance between classification accuracy, computational efficiency, and generalization when applied to high-resolution remote sensing datasets.

---

## Research Motivation

Scene classification in remote sensing imagery plays a crucial role in urban planning, land-use analysis, environmental monitoring, and geospatial intelligence.
Traditional convolutional neural networks often struggle to maintain high accuracy while remaining computationally efficient.

This work shows that EfficientNet-B0, when fine-tuned properly, can outperform deeper CNNs and hybrid architectures while requiring fewer parameters and lower computational cost.

---

## Dataset

The experiments are conducted on the **UC Merced Land Use Dataset**, which consists of:

* 21 land-use scene categories
* 100 RGB images per class
* Total of 2100 images
* Original image size: 256 × 256 pixels

Images are resized to 224 × 224 pixels during preprocessing.

Note:
The dataset is not included in this repository due to licensing restrictions.
Users must download it separately and place it in the directory specified in the configuration section of the code.

---

## Model Architecture

* Backbone: EfficientNet-B0 (pretrained on ImageNet)
* Classifier: Fully connected layer adapted to the number of land-use classes
* Input resolution: 224 × 224 RGB images
* Normalization: ImageNet mean and standard deviation

---

## Training Configuration

The model is trained using the following setup:

* Optimizer: AdamW
* Loss function: Cross-Entropy Loss
* Batch size: 32
* Number of epochs: 10
* Learning rate: 1e-4
* Weight decay: 1e-4
* Training / validation split: 80% / 20%

All hyperparameters are managed through a centralized configuration class inside the training script for reproducibility and clarity.

---

## Installation

Install the required Python packages using:

pip install torch torchvision numpy matplotlib scikit-learn pillow

The code supports both CPU and GPU execution.
If a GPU is available, it will be used automatically.

---

## Project Structure

DL_Remote_Sensing_Scene_Classification
│
├── Remote_Sensing_Scene.py
├── README.md
├── outputs/
│   ├── efficientnet_best.pth
│   ├── training_history.json
│   ├── classification_report.txt
│   ├── metrics.json
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   └── prediction_distribution.png

---

## Usage

1. Download the UC Merced Land Use Dataset.
2. Place the dataset in the directory specified by the DATA_DIR variable in Remote_Sensing_Scene.py.
3. Run the training script using:

python Remote_Sensing_Scene.py

The script automatically handles data loading, preprocessing, model training, evaluation, and result visualization.

---

## Outputs

After execution, all results are saved in the outputs directory.

Model and logs:

* efficientnet_best.pth – Best model weights based on validation loss
* training_history.json – Epoch-wise training and validation metrics

Evaluation reports:

* classification_report.txt – Precision, recall, and F1-score for each class
* metrics.json – Machine-readable evaluation metrics

Visualizations:

* learning_curves.png – Training and validation accuracy and loss curves
* confusion_matrix.png – Confusion matrix across all classes
* prediction_distribution.png – Distribution of predicted classes

---

## Experimental Results

The EfficientNet-B0 model achieves near-perfect classification performance on the UC Merced dataset, with accuracy reaching approximately 99.5%, consistent with the published research findings.

The model demonstrates strong generalization and minimal confusion between visually similar land-use categories.

---

## Reproducibility

* Deterministic data splitting
* Centralized configuration
* Explicit saving of best model and evaluation metrics
* Clear separation of training, validation, and evaluation stages

This design ensures the experiments can be reliably reproduced and extended.

---

## Citation

If you use this code or build upon it in your research, please cite the corresponding paper:

M. H. Salman, Md. A. R. Rahat, A. Rahman, S. S. Khan, and Md. S. Rahman, “Deep Learning Approaches for Scene Classification in Remote Sensing Imagery,” Algorithms for Intelligent Systems, pp. 157–168, Oct. 2025, doi: https://doi.org/10.1007/978-981-96-7059-8_13.

---

## Author

[Md. Abdul rabbi Rahat]
Published Researcher in Deep Learning and Remote Sensing
EfficientNet | CNNs | Vision Transformers | PyTorch

---