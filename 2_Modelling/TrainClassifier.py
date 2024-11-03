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
from fastai.callback.progress import ProgressCallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # Directs logs to stdout
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

# Define the project paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / 'Data'
model_dir = project_dir / 'Model'
plot_dir = model_dir / "plots"
stats_dir = model_dir / "stats"
plot_dir.mkdir(parents=True, exist_ok=True)
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
parser.add_argument('--model', choices=['resnet34', 'vit'], default='resnet34', help="Model type to train")
parser.add_argument('--setup', choices=['frozen', 'unfrozen'], default='frozen', help="Training setup")
parser.add_argument('--subset', type=int, default=-1, help="Number of samples to use for training/val/test")

args = parser.parse_args()



# Load Dataset
def get_dataloaders(conf, batch_size, transforms, num_workers=2):
    logger.info("Loading datasets...")
    train_dataset = BBoxClassificationDataset(conf, "train", transform=transforms)
    val_dataset = BBoxClassificationDataset(conf, "val", transform=transforms)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Datasets loaded successfully.")
    return train_dl, val_dl

# Define Model
def get_model(model_type, setup, num_classes):
    logger.info(f"Initializing model: {model_type} with {setup} setup...")
    if model_type == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'vit':
        model = create_model("vit_base_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    # If model already exists, load it
    if (model_dir / f"{model_type}_{setup}_best.pth").exists():
        logger.info("Loading existing model...")
        model.load_state_dict(torch.load(model_dir / f"{model_type}_{setup}_best.pth", weights_only=True))

    if setup == 'frozen':
        for param in model.parameters():
            param.requires_grad = False
        for param in (model.fc.parameters() if model_type.startswith('resnet') else model.head.parameters()):
            param.requires_grad = True
    logger.info("Model initialized successfully.")
    return model

def save_loss_curve(learn, model_type, setup):
    logger.info("Saving loss curve plot...")
    fig, ax = plt.subplots()
    loss_curve_filename = plot_dir / f"{model_type}_{setup}_loss_curve.png"
    ax = learn.recorder.plot_loss(show_epochs=True, with_valid=True)
    plt.savefig(loss_curve_filename)
    plt.close()
    logger.info(f"Loss curve plot saved at {loss_curve_filename}")

# Train Model
def train_model(conf, model_type, setup, epochs, batch_size, learning_rate, patience):
    num_classes = len(constants.targets)
    model = get_model(model_type, setup, num_classes)
    transforms = Transforms.CropToBBox()
    dataloaders = get_dataloaders(conf, batch_size, transforms)
    train_dl, val_dl = dataloaders

    logger.info("Starting training process...")
    learn = Learner(
        dls=DataLoaders(train_dl, val_dl),
        model=model,
        loss_func=CrossEntropyLossFlat(),
        metrics=[accuracy],
        cbs=[
            ProgressCallback(),
            CSVLogger(fname=stats_dir / f"{model_type}_{setup}_training_logs.csv", append=False),
            EarlyStoppingCallback(monitor='valid_loss', patience=patience),
            SaveModelCallback(monitor='valid_loss', fname=model_dir / f"{model_type}_{setup}_best")
        ]
    )

    if setup == 'frozen':
        learn.freeze()

    model_summary_filename = stats_dir / f"{model_type}_{setup}_model_summary.txt"
    with open(model_summary_filename, 'w') as f:
        f.write(learn.summary())
    logger.info(f"Model summary saved at {model_summary_filename}")

    with learn.no_bar():
        learn.fit_one_cycle(epochs, learning_rate)

    # Save the loss curve for tracking progress
    save_loss_curve(learn, model_type, setup)
    logger.info("Training complete.")
    return learn

# Evaluate Model
def evaluate_model(learn, conf):
    logger.info("Starting evaluation...")
    test_dataset = BBoxClassificationDataset(conf, "test", transform=Transforms.CropToBBox())
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    preds, targs = learn.get_preds(dl=test_dl)
    y_pred = torch.argmax(preds, dim=1).cpu().numpy()
    y_true = targs.cpu().numpy()

    report = classification_report(y_true, y_pred, target_names=constants.targets.values(), zero_division='warn')
    report_filename = stats_dir / f"{args.model}_{args.setup}_classification_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved at {report_filename}")

    cm = confusion_matrix(y_true, y_pred)
    cm_filename = plot_dir / f"{args.model}_{args.setup}_confusion_matrix.png"
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=constants.targets, yticklabels=constants.targets)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(cm_filename)
    plt.close()
    logger.info(f"Confusion matrix saved at {cm_filename}")
    logger.info("Evaluation complete.")

# Main Execution
if __name__ == "__main__":
    subset = args.subset if args.subset > 0 else None
    config = {
        "dataset": {
            "targets": list(constants.targets.values()),
            "annotations_train": str(data_dir / "annotations" / "train"),
            "annotations_val": str(data_dir / "annotations" / "val"),
            "annotations_test": str(data_dir / "annotations" / "test"),
            "dataset_train": str(data_dir / "HaGRIDv2_dataset_512"),
            "dataset_val": str(data_dir / "HaGRIDv2_dataset_512"),
            "dataset_test": str(data_dir / "HaGRIDv2_dataset_512"),
            "subset": subset,
            "one_class": True,
        }
    }
    conf = DictConfig(config)

    learn = train_model(conf, args.model, args.setup, args.epochs, args.batch_size, args.learning_rate, args.patience)
    evaluate_model(learn, conf)
    logger.info("Process completed successfully.")
