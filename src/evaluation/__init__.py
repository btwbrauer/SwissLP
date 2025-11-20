"""
Evaluation utilities for SwissLP models.

This module provides evaluation functions and utilities for both
text and speech classification models.
"""

from .evaluator import Evaluator, SpeechEvaluator, TextEvaluator
from .metrics import ClassificationMetrics, compute_metrics
from .visualizations import (
    create_all_visualizations,
    plot_combined_comparison,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_per_class_metrics,
)

__all__ = [
    "Evaluator",
    "TextEvaluator",
    "SpeechEvaluator",
    "compute_metrics",
    "ClassificationMetrics",
    "plot_confusion_matrix",
    "plot_metrics_comparison",
    "plot_per_class_metrics",
    "plot_combined_comparison",
    "create_all_visualizations",
]
