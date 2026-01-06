import os
import json
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==============================================================================
# Configuration
# ==============================================================================


class Config:
    """Configuration class for all hyperparameters and paths."""

    DATA_DIR = "./UC_Merced_LandUse"
    OUTPUT_DIR = "./outputs"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    TRAIN_SPLIT = 0.8

    @classmethod
    def get_output_path(cls, filename):
        """Get full path for output file."""
        return os.path.join(cls.OUTPUT_DIR, filename)


# ==============================================================================
# Training and Validation Functions
# ==============================================================================


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return loss, accuracy


def validate_epoch(model, loader, criterion, device, return_predictions=False):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_predictions:
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

    loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    if return_predictions:
        return loss, accuracy, all_labels, all_predictions
    return loss, accuracy


# ==============================================================================
# Visualization Functions
# ==============================================================================


def plot_learning_curves(train_acc, val_acc, train_loss, val_loss, output_path):
    """Plot training and validation learning curves."""
    epochs = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Training vs Validation Accuracy
    axes[0].plot(epochs, train_acc, "b-", marker="o", label="Training Accuracy")
    axes[0].plot(epochs, val_acc, "orange", marker="o", label="Validation Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Training vs Validation Loss
    axes[1].plot(epochs, train_loss, "b-", marker="o", label="Training Loss")
    axes[1].plot(epochs, val_loss, "orange", marker="o", label="Validation Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to '{output_path}'")


def plot_confusion_matrix(labels, predictions, class_names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to '{output_path}'")


def plot_prediction_distribution(predictions, class_names, output_path):
    """Plot prediction distribution across classes."""
    prediction_counts = np.bincount(predictions, minlength=len(class_names))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(class_names)), prediction_counts, color="steelblue")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.xlabel("Classes")
    plt.ylabel("Number of Predictions")
    plt.title("Prediction Distribution of EfficientNet Model")

    # Add value labels on bars
    for bar, count in zip(bars, prediction_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction distribution saved to '{output_path}'")


def save_training_history(train_acc, val_acc, train_loss, val_loss, output_path):
    """Save training history to JSON file."""
    history = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to '{output_path}'")


def save_metrics_report(report, output_path):
    """Save classification metrics to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Metrics report saved to '{output_path}'")


# ==============================================================================
# Main Training Pipeline
# ==============================================================================


def main():
    """Main function to run the complete training pipeline."""

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(Config.OUTPUT_DIR)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # Data Preparation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset
    dataset = datasets.ImageFolder(root=Config.DATA_DIR, transform=transform)
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")

    # Split the dataset into training and validation sets
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # =========================================================================
    # Model Setup
    # =========================================================================
    print("\n" + "=" * 50)
    print("Setting Up Model...")
    print("=" * 50)

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(dataset.classes)
    )
    model = model.to(device)
    print("EfficientNet-B0 model loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)

    model_save_path = Config.get_output_path("efficientnet_best.pth")
    best_val_loss = float("inf")
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Store history
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved!")

    print("\nTraining completed.")

    # Save training history
    save_training_history(
        train_acc_history,
        val_acc_history,
        train_loss_history,
        val_loss_history,
        Config.get_output_path("training_history.json"),
    )

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Final Evaluation on Validation Set")
    print("=" * 50)

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    print(f"Best model loaded from {model_save_path}")

    # Final evaluation with predictions
    val_loss, val_acc, all_labels, all_predictions = validate_epoch(
        model, val_loader, criterion, device, return_predictions=True
    )
    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")

    # =========================================================================
    # Classification Report
    # =========================================================================
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)

    report = classification_report(
        all_labels, all_predictions, target_names=dataset.classes, output_dict=True
    )
    report_text = classification_report(
        all_labels, all_predictions, target_names=dataset.classes
    )
    print(report_text)

    # Display Average Metrics
    print("\nAverage Metrics:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1 Score: {report['macro avg']['f1-score']:.4f}")

    # Save metrics report
    save_metrics_report(
        report_text, Config.get_output_path("classification_report.txt")
    )

    # Save metrics as JSON
    with open(Config.get_output_path("metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics JSON saved to '{Config.get_output_path('metrics.json')}'")

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n" + "=" * 50)
    print("Generating Visualizations...")
    print("=" * 50)

    # Plot learning curves
    plot_learning_curves(
        train_acc_history,
        val_acc_history,
        train_loss_history,
        val_loss_history,
        Config.get_output_path("learning_curves.png"),
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels,
        all_predictions,
        dataset.classes,
        Config.get_output_path("confusion_matrix.png"),
    )

    # Plot prediction distribution
    plot_prediction_distribution(
        all_predictions,
        dataset.classes,
        Config.get_output_path("prediction_distribution.png"),
    )

    # =========================================================================
    # Summary
    # =========================================================================
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        print(f"\nGPU memory allocated: {gpu_memory:.2f} MB")

    print("\n" + "=" * 50)
    print("All tasks completed successfully!")
    print("=" * 50)
    print(f"\nOutputs saved to: {os.path.abspath(Config.OUTPUT_DIR)}")
    print("  - efficientnet_best.pth (trained model)")
    print("  - training_history.json (loss and accuracy per epoch)")
    print("  - classification_report.txt (detailed metrics)")
    print("  - metrics.json (metrics in JSON format)")
    print("  - learning_curves.png (accuracy and loss plots)")
    print("  - confusion_matrix.png (confusion matrix visualization)")
    print("  - prediction_distribution.png (class distribution plot)")


if __name__ == "__main__":
    main()
