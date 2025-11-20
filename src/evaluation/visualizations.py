"""
Visualization utilities for model evaluation results.

Provides functions to create comprehensive visualizations including
confusion matrices, metric comparisons, and per-class performance.
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (2D array)
        class_names: List of class names
        model_name: Name of the model
        output_path: Path to save the figure
        normalize: Whether to normalize the matrix (show percentages)
    """
    cm = np.array(confusion_matrix)

    if normalize:
        # Normalize by row (true labels)
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        cm_normalized = np.nan_to_num(cm_normalized)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_normalized = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Normalized Count" if normalize else "Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    results_df: pd.DataFrame, output_path: Path, title: str = "Model Comparison"
) -> None:
    """
    Plot bar chart comparing metrics across models.

    Args:
        results_df: DataFrame with columns: model_name, accuracy, precision, recall, f1
        output_path: Path to save the figure
        title: Title for the plot
    """
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Check which metrics are available
    available_metrics = [m for m in metrics if m in results_df.columns]

    if not available_metrics:
        print(f"Warning: No metrics found in DataFrame. Available columns: {results_df.columns}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if metric not in available_metrics:
            axes[idx].text(
                0.5,
                0.5,
                f"{metric}\nnot available",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].set_title(metric.replace("_", " ").title(), fontweight="bold")
            continue

        ax = axes[idx]
        data = results_df[["model_name", metric]].sort_values(metric, ascending=True)

        bars = ax.barh(data["model_name"], data[metric], color=sns.color_palette("husl", len(data)))
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold", fontsize=12)
        ax.set_xlim(0, max(1.0, data[metric].max() * 1.1))

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    results: dict[str, Any], class_names: list[str], model_name: str, output_path: Path
) -> None:
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        results: Dictionary with precision_per_class, recall_per_class, f1_per_class
        class_names: List of class names
        model_name: Name of the model
        output_path: Path to save the figure
    """
    precision = results.get("precision_per_class", [])
    recall = results.get("recall_per_class", [])
    f1 = results.get("f1_per_class", [])

    if not all([precision, recall, f1]):
        print(f"Warning: Missing per-class metrics for {model_name}")
        return

    # Ensure all are the same length
    n_classes = len(class_names)
    precision = (
        precision[:n_classes]
        if len(precision) >= n_classes
        else precision + [0] * (n_classes - len(precision))
    )
    recall = (
        recall[:n_classes] if len(recall) >= n_classes else recall + [0] * (n_classes - len(recall))
    )
    f1 = f1[:n_classes] if len(f1) >= n_classes else f1 + [0] * (n_classes - len(f1))

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class Metrics: {model_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_combined_comparison(
    text_results: pd.DataFrame, speech_results: pd.DataFrame, output_path: Path
) -> None:
    """
    Create a combined comparison plot for text and speech models.

    Args:
        text_results: DataFrame with text model results
        speech_results: DataFrame with speech model results
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ["accuracy", "precision", "recall", "f1"]

    for ax_idx, (results, title) in enumerate(
        [(text_results, "Text Models"), (speech_results, "Speech Models")]
    ):
        if results.empty:
            axes[ax_idx].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axes[ax_idx].transAxes,
            )
            axes[ax_idx].set_title(title, fontweight="bold")
            continue

        ax = axes[ax_idx]
        x = np.arange(len(results))
        width = 0.2

        for idx, metric in enumerate(metrics):
            if metric in results.columns:
                offset = (idx - 1.5) * width
                values = results[metric].values
                ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(), alpha=0.8)

        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(results["model_name"], rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Model Comparison: Text vs Speech", fontsize=16, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def create_all_visualizations(
    text_results: list[dict[str, Any]],
    speech_results: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """
    Create all visualizations for model comparison.

    Args:
        text_results: List of dictionaries with text model results
        speech_results: List of dictionaries with speech model results
        class_names: List of class names
        output_dir: Directory to save all visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrames
    text_df = pd.DataFrame(text_results) if text_results else pd.DataFrame()
    speech_df = pd.DataFrame(speech_results) if speech_results else pd.DataFrame()

    # 1. Metrics comparison for text models
    if not text_df.empty and "model_name" in text_df.columns:
        plot_metrics_comparison(
            text_df,
            output_dir / "text_models_metrics_comparison.png",
            "Text Models: Metrics Comparison",
        )

    # 2. Metrics comparison for speech models
    if not speech_df.empty and "model_name" in speech_df.columns:
        plot_metrics_comparison(
            speech_df,
            output_dir / "speech_models_metrics_comparison.png",
            "Speech Models: Metrics Comparison",
        )

    # 3. Combined comparison
    if not text_df.empty or not speech_df.empty:
        plot_combined_comparison(text_df, speech_df, output_dir / "combined_comparison.png")

    # 4. Confusion matrices and per-class metrics for each model
    all_results = []
    if text_results:
        all_results.extend([("text", r) for r in text_results])
    if speech_results:
        all_results.extend([("speech", r) for r in speech_results])

    for task_type, result in all_results:
        model_name = result.get("model_name", "unknown")

        # Confusion matrix
        if "confusion_matrix" in result:
            cm = np.array(result["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                class_names,
                model_name,
                output_dir / f"{task_type}_{model_name}_confusion_matrix.png",
            )

            # Also save normalized version
            plot_confusion_matrix(
                cm,
                class_names,
                f"{model_name} (Normalized)",
                output_dir / f"{task_type}_{model_name}_confusion_matrix_normalized.png",
                normalize=True,
            )

        # Per-class metrics
        if any(
            key in result for key in ["precision_per_class", "recall_per_class", "f1_per_class"]
        ):
            plot_per_class_metrics(
                result,
                class_names,
                model_name,
                output_dir / f"{task_type}_{model_name}_per_class_metrics.png",
            )

    print(f"\n✓ All visualizations saved to: {output_dir}")
