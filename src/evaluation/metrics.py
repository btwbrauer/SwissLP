"""
Metrics computation for model evaluation.

Provides functions and classes for computing classification metrics
including accuracy, F1-score, precision, recall, and confusion matrices.

Uses scikit-learn for all metrics computation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    predictions: np.ndarray, labels: np.ndarray, average: str = "weighted"
) -> dict[str, float]:
    """
    Compute classification metrics using scikit-learn.

    Args:
        predictions: Predicted labels (1D array)
        labels: True labels (1D array)
        average: Averaging strategy for multi-class metrics

    Returns:
        Dictionary of metric names and values
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support": support.tolist(),
    }


class ClassificationMetrics:
    """Class for tracking and computing classification metrics."""

    def __init__(self, num_classes: int, class_names: list[str] | None = None):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.labels = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update metrics with new predictions and labels.

        Args:
            predictions: Predicted logits or class indices
            labels: True labels
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = predictions.argmax(dim=-1)
            predictions = predictions.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.predictions.extend(predictions)
        self.labels.extend(labels)

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics from accumulated predictions and labels.

        Returns:
            Dictionary of metric names and values
        """
        if len(self.predictions) == 0:
            # Return default values when no predictions are available
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "precision_per_class": [],
                "recall_per_class": [],
                "f1_per_class": [],
                "support": [],
            }

        predictions = np.array(self.predictions)
        labels = np.array(self.labels)

        metrics = compute_metrics(predictions, labels)

        # Add confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Add classification report
        # Only use class names for classes that are actually present
        unique_labels = np.unique(np.concatenate([labels, predictions]))
        present_class_names = [
            self.class_names[i] for i in unique_labels if i < len(self.class_names)
        ]
        report = classification_report(
            labels,
            predictions,
            target_names=present_class_names,
            labels=unique_labels,
            output_dict=True,
            zero_division=0,
        )
        metrics["classification_report"] = report

        return metrics

    def get_summary(self) -> str:
        """
        Get a formatted summary of metrics.

        Returns:
            Formatted string summary
        """
        metrics = self.compute()
        if not metrics:
            return "No metrics computed yet."

        summary = f"Accuracy: {metrics['accuracy']:.4f}\n"
        summary += f"Precision: {metrics['precision']:.4f}\n"
        summary += f"Recall: {metrics['recall']:.4f}\n"
        summary += f"F1-Score: {metrics['f1']:.4f}\n"

        return summary
