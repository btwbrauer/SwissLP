"""
Hyperparameter tuning with Optuna and MLflow integration.

This module provides functions for automated hyperparameter optimization using Optuna
and automatically logs all trials to MLflow for comparison.
"""

import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow  # type: ignore
import optuna  # type: ignore
from datasets import Dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from ..config import Config
from ..models import load_german_bert, load_swissbert, load_xlm_roberta
from ..utils.constants import MODEL_DISPLAY_NAMES
from ..utils.dataset import make_text_datasets
from ..utils.device import clear_gpu_memory
from ..utils.mlflow_utils import ensure_mlflow_experiment
from .trainer import TextTrainer

logger = logging.getLogger(__name__)


def tokenize_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> Dataset:
    """
    Tokenize a dataset for training.

    Args:
        dataset: HuggingFace Dataset to tokenize
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch")
    return tokenized


def load_model_and_tokenizer(model_name: str, num_labels: int):
    """Load model and tokenizer based on model name."""
    model_name_lower = model_name.lower()
    if "swissbert" in model_name_lower:
        return load_swissbert(num_labels=num_labels)
    elif "german" in model_name_lower:
        return load_german_bert(num_labels=num_labels)
    elif "xlm" in model_name_lower:
        return load_xlm_roberta(num_labels=num_labels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_objective_function(
    config_path: str, model_name: str, mlflow_experiment_name: str
) -> Callable:
    """
    Create an objective function for Optuna optimization.

    Args:
        config_path: Path to config file
        model_name: Model name (swissbert, german_bert, xlm_roberta)
        mlflow_experiment_name: MLflow experiment name for all trials

    Returns:
        Objective function for Optuna
    """

    def objective(trial):
        """
        Objective function for Optuna optimization.

        This function is called for each trial and returns the validation F1 score
        to be maximized.
        """
        # Load base config
        config = Config.load(config_path)

        # Ensure training config exists
        if config.training is None:
            raise ValueError("Training configuration is required for hyperparameter tuning")

        training_config = config.training

        # Suggest hyperparameters using Optuna
        training_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        training_config.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        training_config.warmup_steps = trial.suggest_int("warmup_steps", 100, 500, step=50)
        training_config.weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
        training_config.lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine"]
        )

        # Optional: tune gradient accumulation steps
        if training_config.batch_size == 8:
            training_config.gradient_accumulation_steps = trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2]
            )

        # Set unique experiment name for this trial (for output directory)
        # But use main MLflow experiment name for MLflow tracking
        config.experiment_name = f"{config.experiment_name}_trial_{trial.number}"

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Trial {trial.number}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Learning rate: {training_config.learning_rate:.2e}")
        logger.info(f"Batch size: {training_config.batch_size}")
        logger.info(f"Warmup steps: {training_config.warmup_steps}")
        logger.info(f"Weight decay: {training_config.weight_decay:.4f}")
        logger.info(f"LR scheduler: {training_config.lr_scheduler_type}")

        try:
            # Create MLflow run for this trial BEFORE training starts
            # This ensures the trainer can reuse this run instead of creating a new one

            # Setup MLflow experiment
            tracking_uri = ensure_mlflow_experiment(mlflow_experiment_name)

            # Start run for this trial (nested=True allows it to be nested under parent if exists)
            mlflow.start_run(run_name=f"Trial {trial.number}", nested=True)

            # Log trial parameters to MLflow
            mlflow.log_params(trial.params)
            mlflow.log_param("trial_number", trial.number)

            # Load datasets
            train_dataset, val_dataset, test_dataset, dialect2label, _, _, _ = make_text_datasets(
                config.data.data_path,
                config.data.dialects,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio,
                test_ratio=config.data.test_ratio,
                seed=training_config.seed,
            )

            config.model.num_labels = len(dialect2label)

            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                config.model.model_name, config.model.num_labels
            )

            # Tokenize datasets
            train_dataset = tokenize_dataset(
                train_dataset, tokenizer, max_length=config.data.max_length
            )
            val_dataset = tokenize_dataset(
                val_dataset, tokenizer, max_length=config.data.max_length
            )

            # Create trainer with main MLflow experiment name
            # This ensures all trials go to the same MLflow experiment
            # The trainer will reuse the active MLflow run we just created
            trainer = TextTrainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                mlflow_experiment_name=mlflow_experiment_name,
            )

            # Train
            train_result = trainer.train()

            # Evaluate on validation set
            val_metrics = trainer.evaluate()
            val_f1 = val_metrics.get("eval_f1", val_metrics.get("f1", 0.0))

            logger.info(f"Validation F1: {val_f1:.4f}")

            # Report to Optuna
            trial.set_user_attr("train_loss", train_result["train_loss"])
            trial.set_user_attr("val_accuracy", val_metrics.get("eval_accuracy", 0.0))
            trial.set_user_attr("val_macro_f1", val_metrics.get("eval_macro_f1", 0.0))

            # Log final metrics to MLflow run
            mlflow.log_metric("eval_f1", val_f1)
            mlflow.log_metric("val_accuracy", val_metrics.get("eval_accuracy", 0.0))
            mlflow.log_metric("val_macro_f1", val_metrics.get("eval_macro_f1", 0.0))
            mlflow.log_metric("train_loss", train_result["train_loss"])

            # End MLflow run for this trial
            mlflow.end_run()

            # Cleanup
            trainer.cleanup()
            del trainer, model, tokenizer, train_dataset, val_dataset, test_dataset
            clear_gpu_memory()
            gc.collect()
            time.sleep(1)  # Give time for cleanup

            return val_f1

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("trial_state", "FAILED")
                mlflow.end_run()
            return 0.0

    return objective


def create_mlflow_callback(experiment_name: str) -> Callable:
    """
    Create MLflow callback for Optuna study.

    Args:
        experiment_name: MLflow experiment name

    Returns:
        Callback function for Optuna
    """

    def mlflow_callback(study, trial):
        """
        Log Optuna trial to MLflow.

        Note: The MLflow run is created in the objective function before training,
        so this callback just adds any additional metadata. The run is already ended
        in the objective function after training completes.
        """
        try:
            # The run should already exist from the objective function
            # Just log trial state and any additional metadata
            active_run = mlflow.active_run()
            if active_run is None:
                # Run doesn't exist (shouldn't happen, but handle gracefully)
                logger.warning(f"No active MLflow run for trial {trial.number}, skipping callback")
                return

            # Log trial state
            mlflow.log_param("trial_state", trial.state.name)

            # Log user attributes if trial is complete
            if trial.state == optuna.trial.TrialState.COMPLETE:
                for key, value in trial.user_attrs.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))
        except Exception as e:
            logger.warning(f"Failed to log trial {trial.number} to MLflow: {e}")

    return mlflow_callback




def run_hyperparameter_tuning(
    config_path: str | Path,
    model_name: str,
    n_trials: int = 20,
    timeout: int | None = None,
    study_name: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> optuna.Study:
    """
    Run hyperparameter tuning for a model.

    Args:
        config_path: Path to config file
        model_name: Model name (swissbert, german_bert, xlm_roberta)
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None = no timeout)
        study_name: Optuna study name (None = auto-generate)
        mlflow_experiment_name: MLflow experiment name (None = use model display name)

    Returns:
        Optuna study with results
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Setup MLflow
    experiment_name = mlflow_experiment_name or MODEL_DISPLAY_NAMES.get(model_name, model_name)
    tracking_uri = ensure_mlflow_experiment(experiment_name)

    # Create Optuna study
    study_name = study_name or f"{model_name}_hyperparameter_tuning"

    logger.info("=" * 70)
    logger.info("Hyperparameter Tuning with Optuna")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Study name: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info("=" * 70)

    study = optuna.create_study(
        direction="maximize",  # Maximize validation F1 score
        study_name=study_name,
    )

    # Create objective function
    objective = create_objective_function(str(config_path), model_name, experiment_name)

    # Create MLflow callback
    mlflow_callback = create_mlflow_callback(experiment_name)

    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback],
        )
    except KeyboardInterrupt:
        logger.warning("Tuning interrupted by user")

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("Tuning Results")
    logger.info("=" * 70)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    logger.info(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )

    if study.best_trial:
        logger.info("\nBest trial:")
        logger.info(f"  Value (F1): {study.best_value:.4f}")
        logger.info("  Params:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")

        # Log best trial summary to MLflow
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="Best Trial Summary"):
                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_f1", study.best_value)
                mlflow.log_metric("n_trials", len(study.trials))
                mlflow.log_metric(
                    "n_complete_trials",
                    len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                )
        except Exception as e:
            logger.warning(f"Could not log best trial to MLflow: {e}")

    logger.info("=" * 70)
    logger.info("\nView results in MLflow UI:")
    logger.info(f"  {tracking_uri}")
    logger.info(f"\nExperiment: {experiment_name}")

    return study
