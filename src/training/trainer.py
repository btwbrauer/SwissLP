"""
Training utilities using Hugging Face Trainer.

Provides wrappers for training text and speech classification models
with proper configuration, metrics, and checkpointing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow  # type: ignore
import numpy as np
import torch  # type: ignore
from datasets import Dataset  # type: ignore
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from ..config import Config
from ..evaluation.metrics import compute_metrics
from ..utils.mlflow_utils import ensure_mlflow_experiment


def create_compute_metrics_fn(class_names: list[str] | None = None):
    """Create a compute_metrics function for Hugging Face Trainer."""
    from transformers import EvalPrediction  # type: ignore

    def compute_metrics_fn(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        metrics = compute_metrics(predictions, labels, average="weighted")

        metrics_filtered: dict[str, float] = {}
        for k, v in metrics.items():
            if k not in ["precision_per_class", "recall_per_class", "f1_per_class", "support"]:
                if isinstance(v, (int, float, np.number)):
                    metrics_filtered[k] = float(v)

        _, _, f1_macro, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,  # type: ignore
        )
        metrics_filtered["macro_f1"] = float(f1_macro)

        if class_names:
            _, _, f1_per_class, _ = precision_recall_fscore_support(
                labels,
                predictions,
                average=None,
                zero_division=0,  # type: ignore
            )
            for i, class_name in enumerate(class_names):
                if i < len(f1_per_class):
                    metrics_filtered[f"f1_{class_name}"] = float(f1_per_class[i])

        return metrics_filtered

    return compute_metrics_fn


class MLflowMetricsCallback(TrainerCallback):
    """Callback to log metrics to MLflow during training."""

    def __init__(self):
        """Initialize MLflow metrics callback."""
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to MLflow when trainer logs metrics."""
        if logs is None:
            return

        # Filter out non-metric keys
        metrics_to_log = {}
        for key, value in logs.items():
            # Include learning_rate and epoch as they're useful metrics
            # Skip only non-numeric values and step (which is redundant with global_step)
            if key != "step" and isinstance(value, (int, float)):
                metrics_to_log[key] = float(value)

        # Log metrics to MLflow if there are any
        if metrics_to_log and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics_to_log, step=state.global_step)
            except Exception:
                # Silently fail to avoid disrupting training
                pass

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Log evaluation metrics to MLflow."""
        if logs is None:
            return

        # Filter out non-metric keys
        metrics_to_log = {}
        for key, value in logs.items():
            # Skip non-numeric values and step (epoch is useful to keep)
            if key != "step" and isinstance(value, (int, float)):
                metrics_to_log[key] = float(value)

        # Log metrics to MLflow if there are any
        if metrics_to_log and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics_to_log, step=state.global_step)
            except Exception:
                # Silently fail to avoid disrupting training
                pass


class BaseTrainer(ABC):
    """Base trainer class for text and speech models."""

    def __init__(
        self,
        config: Config,
        model: Any,
        processing_class: Any,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        data_collator: Any = None,
        mlflow_experiment_name: str | None = None,
    ):
        """Initialize base trainer."""
        if config.training is None:
            raise ValueError("Training configuration is required for training")

        # Store training config locally to help type checker
        training_config = config.training

        self.config = config
        self.model = model
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._mlflow_run_active = False
        self._mlflow_experiment_name = mlflow_experiment_name
        self._mlflow_run_created_externally = False

        # Create output directory
        output_dir = Path(training_config.output_dir) / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            max_grad_norm=training_config.max_grad_norm,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            logging_steps=training_config.logging_steps,
            eval_strategy="steps" if val_dataset is not None else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset is not None else False,
            metric_for_best_model="f1" if val_dataset is not None else None,
            fp16=training_config.fp16,
            seed=training_config.seed,
            lr_scheduler_type=training_config.lr_scheduler_type,
            report_to="none",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Disable multiprocessing to prevent segfaults in sequential training
        )

        # Create compute_metrics function
        class_names = list(config.data.dialects) if hasattr(config.data, "dialects") else None
        compute_metrics_fn = create_compute_metrics_fn(class_names)

        # Setup MLflow
        self._setup_mlflow()

        # Create callbacks
        # Note: We don't use MLflowCallback from transformers because we manually manage MLflow runs
        # This prevents duplicate runs when used with Optuna hyperparameter tuning
        # Instead, we use a custom callback to log metrics during training
        callbacks = [MLflowMetricsCallback()]
        if val_dataset is not None and training_config.early_stopping_patience is not None:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold,
            )
            callbacks.append(early_stopping)

        # Create Hugging Face Trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

    def _setup_mlflow(self):
        """Setup MLflow tracking and experiment."""
        experiment_name = self._mlflow_experiment_name or self.config.experiment_name
        ensure_mlflow_experiment(experiment_name)

        # Reuse existing run if available (e.g., from Optuna), otherwise create new one
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=self.config.experiment_name)
            self._mlflow_run_created_externally = False
        else:
            # Run was created externally (e.g., by Optuna hyperparameter tuning)
            self._mlflow_run_created_externally = True
        self._mlflow_run_active = True
        mlflow.log_params(self._get_mlflow_params())

    @abstractmethod
    def _get_mlflow_params(self) -> dict[str, str]:
        """Get parameters to log to MLflow. Must be implemented by subclasses."""
        pass

    def train(self) -> dict[str, Any]:
        """Train the model."""
        if self.config.training is None:
            raise ValueError("Training configuration is required")
        training_config = self.config.training

        # Set random seed
        torch.manual_seed(training_config.seed)
        np.random.seed(training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_config.seed)

        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model()

        # Get epoch from metrics
        epoch = (
            train_result.metrics.get("epoch")
            or getattr(train_result, "epoch", None)
            or train_result.global_step
            / max(
                len(self.train_dataset)
                // (training_config.batch_size * training_config.gradient_accumulation_steps),
                1,
            )
        )

        # Log final training metrics
        # Include all metrics from train_result (includes eval metrics if evaluation was done)
        final_metrics = {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "epoch": epoch,
        }

        # Add any evaluation metrics that were collected during training
        # Include all metrics from train_result (they're already logged during training via callback,
        # but we log them again here as final values for easy comparison)
        if hasattr(train_result, "metrics") and train_result.metrics:
            for key, value in train_result.metrics.items():
                # Include all metrics, including runtime metrics
                if isinstance(value, (int, float)):
                    final_metrics[key] = float(value)

        mlflow.log_metrics(final_metrics)

        # Log model artifacts (only if output directory exists and is accessible)
        output_dir = Path(training_config.output_dir) / self.config.experiment_name
        if output_dir.exists() and any(output_dir.iterdir()):
            try:
                mlflow.log_artifacts(str(output_dir), artifact_path="model")
            except (PermissionError, OSError):
                # Silently skip artifact logging if there are permission issues
                # This can happen in test environments or when using file-based MLflow
                pass

        # End MLflow run only if it was created by this trainer
        # If created externally (e.g., by Optuna), let the caller end it
        if not self._mlflow_run_created_externally:
            self._end_mlflow_run()

        return {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "epoch": epoch,
        }

    def _end_mlflow_run(self):
        """End MLflow run if active."""
        if self._mlflow_run_active and mlflow.active_run():
            mlflow.end_run()
        self._mlflow_run_active = False

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, float]:
        """Evaluate the model."""
        eval_dataset = dataset or self.val_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        # Ensure MLflow run is active
        if not self._mlflow_run_active:
            self._setup_mlflow()

        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

        # Log evaluation metrics to MLflow
        if mlflow.active_run():
            metrics_to_log = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_to_log[key] = float(value)
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

        return metrics

    def cleanup(self):
        """Cleanup trainer resources."""
        self._end_mlflow_run()
        if hasattr(self, "trainer"):
            del self.trainer
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup: ensure MLflow run is ended when trainer is deleted."""
        if hasattr(self, "_mlflow_run_active"):
            self._end_mlflow_run()


class TextTrainer(BaseTrainer):
    """Trainer for text classification models."""

    def __init__(
        self,
        config: Config,
        model: AutoModelForSequenceClassification,
        tokenizer: Any,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        """Initialize text trainer."""
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        super().__init__(
            config=config,
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            mlflow_experiment_name=mlflow_experiment_name,
        )

    def _get_mlflow_params(self) -> dict[str, str]:
        """Get parameters to log to MLflow."""
        training_config = self.config.training
        if training_config is None:
            raise ValueError("Training configuration is required")

        params = {
            "model_name": str(self.config.model.model_name),
            "num_labels": str(self.config.model.num_labels),
            "task_type": str(self.config.task_type),
            "learning_rate": str(training_config.learning_rate),
            "num_epochs": str(training_config.num_epochs),
            "batch_size": str(training_config.batch_size),
            "warmup_steps": str(training_config.warmup_steps),
            "weight_decay": str(training_config.weight_decay),
            "max_grad_norm": str(training_config.max_grad_norm),
            "gradient_accumulation_steps": str(training_config.gradient_accumulation_steps),
            "fp16": str(training_config.fp16),
            "lr_scheduler_type": str(training_config.lr_scheduler_type),
            "seed": str(training_config.seed),
            "train_ratio": str(self.config.data.train_ratio),
            "val_ratio": str(self.config.data.val_ratio),
            "test_ratio": str(self.config.data.test_ratio),
            "max_length": str(self.config.data.max_length),
        }
        return params


class SpeechTrainer(BaseTrainer):
    """Trainer for speech classification models."""

    def __init__(
        self,
        config: Config,
        model: Wav2Vec2ForSequenceClassification,
        processor: Any,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        """Initialize speech trainer."""

        # Create custom data collator for audio
        class AudioDataCollator:
            """Data collator for audio models with dynamic padding."""

            def __init__(self, processor: Wav2Vec2FeatureExtractor):
                self.processor = processor

            def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
                """Collate audio features with dynamic padding."""
                input_values = [f["input_values"] for f in features]
                labels = torch.tensor([f["label"] for f in features])
                batch_dict = self.processor(
                    input_values,
                    padding=True,
                    return_tensors="pt",
                )
                batch_dict["labels"] = labels
                return batch_dict

        data_collator = AudioDataCollator(processor=processor)
        super().__init__(
            config=config,
            model=model,
            processing_class=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            mlflow_experiment_name=mlflow_experiment_name,
        )

    def _get_mlflow_params(self) -> dict[str, str]:
        """Get parameters to log to MLflow."""
        training_config = self.config.training
        if training_config is None:
            raise ValueError("Training configuration is required")

        params = {
            "model_name": str(self.config.model.model_name),
            "num_labels": str(self.config.model.num_labels),
            "task_type": str(self.config.task_type),
            "learning_rate": str(training_config.learning_rate),
            "num_epochs": str(training_config.num_epochs),
            "batch_size": str(training_config.batch_size),
            "warmup_steps": str(training_config.warmup_steps),
            "weight_decay": str(training_config.weight_decay),
            "max_grad_norm": str(training_config.max_grad_norm),
            "gradient_accumulation_steps": str(training_config.gradient_accumulation_steps),
            "fp16": str(training_config.fp16),
            "lr_scheduler_type": str(training_config.lr_scheduler_type),
            "seed": str(training_config.seed),
            "train_ratio": str(self.config.data.train_ratio),
            "val_ratio": str(self.config.data.val_ratio),
            "test_ratio": str(self.config.data.test_ratio),
            "audio_sample_rate": str(self.config.data.audio_sample_rate),
            "audio_max_duration": str(self.config.data.audio_max_duration or 0.0),
        }
        return params
