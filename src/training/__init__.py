"""
Training module for SwissLP project.

Provides training utilities using Hugging Face Trainer for fine-tuning
text and speech classification models.
"""

from .hyperparameter_tuning import run_hyperparameter_tuning
from .model_cleanup import keep_best_model
from .optimization import optimize_single_model
from .trainer import BaseTrainer, SpeechTrainer, TextTrainer

__all__ = [
    "BaseTrainer",
    "SpeechTrainer",
    "TextTrainer",
    "run_hyperparameter_tuning",
    "keep_best_model",
    "optimize_single_model",
]
