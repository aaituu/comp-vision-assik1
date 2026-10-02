#!/usr/bin/env python3
"""Train and evaluate baseline and transfer-learning action classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASSES = ["sitting", "standing", "waving"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "manifest.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def split_records(records: list[dict[str, str]], seed: int):
    labels = [row["class"] for row in records]
    train, remainder = train_test_split(
        records, test_size=0.30, random_state=seed, stratify=labels
    )
    remainder_labels = [row["class"] for row in remainder]
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder_labels,
    )
    return train, validation, test


def save_split_csv(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    with (output_dir / "data_splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "class", "split"])
        for split_name, rows in splits.items():
            for row in rows:
                writer.writerow([row["filename"], row["class"], split_name])


def hog_features(data_dir: Path, records: list[dict[str, str]]) -> np.ndarray:
    descriptor = cv2.HOGDescriptor(
        (128, 128), (32, 32), (16, 16), (16, 16), 9
    )
    features = []
    for row in records:
        image = cv2.imread(str(data_dir / row["filename"]))
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(descriptor.compute(image).ravel())
    return np.asarray(features, dtype=np.float32)


def metric_summary(y_true: list[int], y_pred: list[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))
        ).tolist(),
    }


def train_baseline(data_dir: Path, train_rows: list[dict], test_rows: list[dict]):
    label_to_id = {name: index for index, name in enumerate(CLASSES)}
    x_train = hog_features(data_dir, train_rows)
    x_test = hog_features(data_dir, test_rows)
    y_train = np.asarray([label_to_id[row["class"]] for row in train_rows])
    y_test = np.asarray([label_to_id[row["class"]] for row in test_rows])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, C=1.0, random_state=42),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, metric_summary(y_test.tolist(), predictions.tolist()), predictions.tolist()


class ActionDataset(Dataset):
    def __init__(self, data_dir: Path, rows: list[dict], transform):
        self.data_dir = data_dir
        self.rows = rows
        self.transform = transform
        self.label_to_id = {name: index for index, name in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.data_dir / row["filename"]) as source:
            image = source.convert("RGB")
        return self.transform(image), self.label_to_id[row["class"]]


def deep_transforms():
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def evaluate_deep(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), true_labels, predictions


def train_transfer_model(
    data_dir: Path,
    train_rows: list[dict],
    validation_rows: list[dict],
    test_rows: list[dict],
    output_dir: Path,
    epochs: int,
):
    train_transform, evaluation_transform = deep_transforms()
    train_loader = DataLoader(
        ActionDataset(data_dir, train_rows, train_transform),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        ActionDataset(data_dir, validation_rows, evaluation_transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ActionDataset(data_dir, test_rows, evaluation_transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_accuracy = -1.0
    best_path = output_dir / "mobilenet_v3_small_best.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        validation_loss, validation_true, validation_pred = evaluate_deep(
            model, validation_loader, criterion, device
        )
        validation_accuracy = accuracy_score(validation_true, validation_pred)
        history["train_loss"].append(float(train_loss))
        history["validation_loss"].append(float(validation_loss))
        history["validation_accuracy"].append(float(validation_accuracy))
        print(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_true, test_predictions = evaluate_deep(
        model, test_loader, criterion, device
    )
    metrics = metric_summary(test_true, test_predictions)
    metrics["test_loss"] = float(test_loss)
    metrics["best_validation_accuracy"] = float(best_accuracy)
    metrics["device"] = str(device)
    return model, metrics, history, test_true, test_predictions


def plot_confusion(matrix: list[list[int]], title: str, path: Path) -> None:
    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.imshow(array, cmap="Greys")
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            threshold = array.max() / 2 if array.max() else 0
            axis.text(
                col,
                row,
                str(array[row, col]),
                ha="center",
                va="center",
                color="white" if array[row, col] > threshold else "black",
            )
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_figures(
    data_dir: Path,
    output_dir: Path,
    records: list[dict],
    baseline_metrics: dict,
    transfer_metrics: dict,
    history: dict,
) -> None:
    counts = [sum(row["class"] == name for row in records) for name in CLASSES]
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    axis.bar(CLASSES, counts, color=["0.25", "0.50", "0.75"], edgecolor="black")
    axis.set_ylabel("Number of images")
    axis.set_title("Dataset class distribution")
    axis.set_ylim(0, max(counts) * 1.18)
    for index, value in enumerate(counts):
        axis.text(index, value + 2, str(value), ha="center")
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(8.0, 6.0))
    for row_index, class_name in enumerate(CLASSES):
        examples = [row for row in records if row["class"] == class_name][:4]
        for col_index, row in enumerate(examples):
            with Image.open(data_dir / row["filename"]) as image:
                axes[row_index, col_index].imshow(image.convert("RGB"))
            axes[row_index, col_index].axis("off")
            if col_index == 0:
                axes[row_index, col_index].set_title(class_name, loc="left", fontsize=10)
    figure.tight_layout(pad=0.6)
    figure.savefig(output_dir / "dataset_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(5.6, 3.5))
    axis.plot(epochs, history["train_loss"], "-o", color="black", label="train loss")
    axis.plot(
        epochs,
        history["validation_loss"],
        "--s",
        color="0.45",
        label="validation loss",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Transfer learning history")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    plot_confusion(
        baseline_metrics["confusion_matrix"],
        "Baseline: HOG and Logistic Regression",
        output_dir / "confusion_baseline.png",
    )
    plot_confusion(
        transfer_metrics["confusion_matrix"],
        "Improved model: MobileNetV3-Small",
        output_dir / "confusion_transfer.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.data_dir)
    train_rows, validation_rows, test_rows = split_records(records, args.seed)
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    save_split_csv(splits, args.output_dir)

    _, baseline_metrics, baseline_predictions = train_baseline(
        args.data_dir, train_rows, test_rows
    )
    _, transfer_metrics, history, test_true, transfer_predictions = train_transfer_model(
        args.data_dir,
        train_rows,
        validation_rows,
        test_rows,
        args.output_dir,
        args.epochs,
    )

    results = {
        "seed": args.seed,
        "class_names": CLASSES,
        "dataset_size": len(records),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "baseline": baseline_metrics,
        "transfer_learning": transfer_metrics,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "true_class", "baseline_prediction", "transfer_prediction"])
        for row, baseline, transfer in zip(
            test_rows, baseline_predictions, transfer_predictions
        ):
            writer.writerow([row["filename"], row["class"], CLASSES[baseline], CLASSES[transfer]])

    create_figures(
        args.data_dir,
        args.output_dir,
        records,
        baseline_metrics,
        transfer_metrics,
        history,
    )
    print(json.dumps({"baseline": baseline_metrics, "transfer": transfer_metrics}, indent=2))


if __name__ == "__main__":
    main()
