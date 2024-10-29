import os
import sys
import argparse
import logging
from pathlib import Path
from fastai.vision.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from omegaconf import DictConfig
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the project paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model'
loss_curve_dir = model_dir / "loss_curves"
stats_dir = model_dir / "stats"
loss_curve_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)

# Add project directory to system path for importing modules
sys.path.append(str(project_dir / '1_HaGRID'))

# Import project-specific modules
from Transforms import Transforms
from dataset import BBoxClassificationDataset
import constants

# Argument Parser
parser = argparse.ArgumentParser(description="Train ResNet or Vision Transformer on hand gesture dataset.")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for training")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping")
parser.add_argument('--model', choices=['resnet', 'vit'], default='resnet', help="Model type to train")
parser.add_argument('--setup', choices=['frozen', 'finetuned'], default='frozen', help="Training setup")
parser.add_argument('--overwrite', action='store_true', help="Overwrite existing model files")

args = parser.parse_args()

# Load Dataset
def get_dataloaders(conf, batch_size, transforms, num_workers=2):
    train_dataset = BBoxClassificationDataset(conf, "train", transform=transforms)
    val_dataset = BBoxClassificationDataset(conf, "val", transform=transforms)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl


# Define Model
def get_model(model_type, setup, num_classes):
    if model_type == 'resnet':
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'vit':
        model = create_model("vit_base_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    if setup == 'frozen':
        for param in model.parameters():
            param.requires_grad = False
        # Enable gradients for classification layer only
        for param in (model.fc.parameters() if model_type == 'resnet' else model.head.parameters()):
            param.requires_grad = True

    return model


# Plot Loss Curve
def plot_loss_curve(train_losses, val_losses, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# Save Model Statistics
def save_model_stats(model, history, stats_filename):
    stats = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'epochs': len(history['train_loss'])
    }
    pd.DataFrame(stats).to_csv(stats_filename)


# Train Model
def train_model(conf, model_type, setup, epochs, batch_size, learning_rate, patience):
    num_classes = len(constants.targets)  # Assuming TARGETS is defined in constants
    model = get_model(model_type, setup, num_classes)

    # Apply transformations
    transforms = Transforms.CropToBBox()  # Assuming Transforms.CropToBBox() is a valid transformation
    dataloaders = get_dataloaders(conf, batch_size, transforms)
    train_dl, val_dl = dataloaders
    # TODO find out if model shuffling is necessary?
    learn = Learner(
        dls=DataLoaders(train_dl, val_dl),
        model=model,
        loss_func=CrossEntropyLossFlat(),
        metrics=[accuracy],
        cbs=[
            ProgressCallback(),
            EarlyStoppingCallback(monitor='valid_loss', patience=patience)
        ]
    )

    if setup == 'frozen':
        learn.freeze()

    learn.fit_one_cycle(epochs, learning_rate)

    # Save Model
    model_filename = model_dir / f"{model_type}_{setup}.pth"
    if args.overwrite or not model_filename.exists():
        learn.save(model_filename)

    # Save Loss Curve
    loss_curve_filename = loss_curve_dir / f"{model_type}_{setup}_loss_curve.png"
    plot_loss_curve(learn.recorder.train_loss, learn.recorder.valid_loss, loss_curve_filename)

    # Save Model Stats
    stats_filename = stats_dir / f"{model_type}_{setup}_stats.csv"
    save_model_stats(model, learn.recorder, stats_filename)

    return learn


# Evaluate Model
def evaluate_model(learn, conf):
    test_dataset = BBoxClassificationDataset(conf, "test",
                                        transform=Transforms.CropToBBox())  # Assuming Transforms.CropToBBox() is used
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    preds, targs = learn.get_preds(dl=test_dl)
    y_pred = torch.argmax(preds, dim=1).cpu().numpy()
    y_true = targs.cpu().numpy()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=constants.targets)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_filename = stats_dir / f"{args.model}_{args.setup}_confusion_matrix.png"
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=constants.targets, yticklabels=constants.targets)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(cm_filename)
    plt.close()


# Main Execution
if __name__ == "__main__":
    # Define the datasets
    config = {
        "dataset": {
            "targets": list(constants.targets.values()),
            "annotations_train": str(data_dir / "annotations" / "train"),
            "annotations_val": str(data_dir / "annotations" / "val"),
            "annotations_test": str(data_dir / "annotations" / "test"),
            "dataset_train": str(data_dir / "HaGRIDv2_dataset_512"),
            "dataset_val": str(data_dir / "HaGRIDv2_dataset_512"),
            "dataset_test": str(data_dir / "HaGRIDv2_dataset_512"),
            "subset": None,
            "one_class": True,
        }
    }
    # Convert config to DictConfig
    conf = DictConfig(config)


    learn = train_model(conf, args.model, args.setup, args.epochs, args.batch_size, args.learning_rate, args.patience)
    evaluate_model(learn, conf)
