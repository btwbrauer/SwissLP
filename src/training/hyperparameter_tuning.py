"""
Hyperparameter tuning with Optuna and MLflow integration.

This module provides functions for automated hyperparameter optimization using Optuna
and automatically logs all trials to MLflow for comparison.
"""

import gc
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlflow  # type: ignore
import optuna  # type: ignore
from datasets import Dataset  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
)

from ..config import Config
from ..models import load_german_bert, load_swissbert, load_xlm_roberta
from ..utils.constants import MODEL_DISPLAY_NAMES
from ..utils.dataset import make_text_datasets
from ..utils.device import clear_gpu_memory, setup_test_environment
from ..utils.mlflow_utils import ensure_mlflow_experiment
from .trainer import TextTrainer

# Setup environment to prevent segmentation faults (especially on ROCm)
setup_test_environment()

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
        # Use padding="max_length" like in notebooks to ensure consistent tensor shapes
        # Cast to dict to satisfy type checker (BatchEncoding acts like a dict)
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        return cast(dict[str, Any], tokenized)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing",
    )
    # Set format like in notebooks - explicitly specify columns
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
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


def cleanup_old_trials(
    outputs_dir: Path,
    base_experiment_name: str,
    current_trial_number: int,
    keep_best_n: int = 3,
    study: optuna.Study | None = None,
) -> None:
    """
    Clean up old trial directories, keeping only the best N trials.

    Args:
        outputs_dir: Root outputs directory
        base_experiment_name: Base experiment name (without _trial_X suffix)
        current_trial_number: Current trial number (won't be deleted)
        keep_best_n: Number of best trials to keep (default: 3)
        study: Optuna study to determine best trials (if None, keeps most recent)
    """
    if not outputs_dir.exists():
        return

    # Find all trial directories
    trial_dirs = {}
    pattern = f"{base_experiment_name}_trial_"

    for exp_dir in outputs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        if exp_dir.name.startswith(pattern):
            try:
                trial_num = int(exp_dir.name.replace(pattern, ""))
                # Don't delete current trial
                if trial_num < current_trial_number:
                    trial_dirs[trial_num] = exp_dir
            except ValueError:
                continue

    if len(trial_dirs) <= keep_best_n:
        return

    # Determine which trials to keep
    if study is not None:
        # Sort trials by value (best first)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(
            completed_trials, key=lambda t: t.value or float("-inf"), reverse=True
        )
        best_trial_numbers = {t.number for t in sorted_trials[:keep_best_n]}
    else:
        # Keep most recent N trials
        sorted_trial_nums = sorted(trial_dirs.keys(), reverse=True)
        best_trial_numbers = set(sorted_trial_nums[:keep_best_n])

    # Delete old trials that are not in the best N
    deleted_count = 0
    freed_mb = 0.0

    for trial_num, trial_dir in trial_dirs.items():
        if trial_num not in best_trial_numbers:
            try:
                # Calculate size before deletion
                size_mb = sum(f.stat().st_size for f in trial_dir.rglob("*") if f.is_file()) / (
                    1024 * 1024
                )
                shutil.rmtree(trial_dir)
                deleted_count += 1
                freed_mb += size_mb
                logger.debug(
                    f"Cleaned up old trial {trial_num}: {trial_dir.name} ({size_mb:.1f} MB)"
                )
            except Exception as e:
                logger.warning(f"Failed to delete trial {trial_num} directory: {e}")

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} old trial(s), freed {freed_mb:.1f} MB (kept best {keep_best_n} trials)"
        )


def create_objective_function(
    config_path: str,
    model_name: str,
    mlflow_experiment_name: str,
    keep_best_n_trials: int = 3,
    study_container: dict[str, optuna.Study | None] | None = None,
) -> Callable:
    """
    Create an objective function for Optuna optimization.

    Args:
        config_path: Path to config file
        model_name: Model name (swissbert, german_bert, xlm_roberta)
        mlflow_experiment_name: MLflow experiment name for all trials
        keep_best_n_trials: Number of best trials to keep (default: 3)
        study_container: Mutable container to hold study reference

    Returns:
        Objective function for Optuna
    """
    if study_container is None:
        study_container = {"study": None}

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
        
        # Reduce batch size search space to prevent OOM
        # Previous attempts with 16 crashed, so we stick to 8 and use gradient accumulation
        training_config.batch_size = trial.suggest_categorical("batch_size", [8])
        
        training_config.warmup_steps = trial.suggest_int("warmup_steps", 100, 500, step=50)
        training_config.weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
        training_config.lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine"]
        )

        # Tune gradient accumulation steps to simulate larger batch sizes
            training_config.gradient_accumulation_steps = trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]
            )

        # Set unique experiment name for this trial (for output directory)
        # But use main MLflow experiment name for MLflow tracking
        config.experiment_name = f"{config.experiment_name}_trial_{trial.number}"

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Trial {trial.number + 1:02d}")
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
            ensure_mlflow_experiment(mlflow_experiment_name)

            # Start run for this trial (nested=True allows it to be nested under parent if exists)
            # Use trial.number + 1 so first run is named _01 instead of _00
            mlflow.start_run(
                run_name=f"{mlflow_experiment_name}_{trial.number + 1:02d}", nested=True
            )

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

            # Load model and tokenizer - exactly like SwissBERT
            clear_gpu_memory()
            import torch  # Import at function level to avoid conflicts
            
            if torch.cuda.is_available():
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            
            # Load model - same approach for all models (including German BERT)
            model, tokenizer = load_model_and_tokenizer(
                config.model.model_name, config.model.num_labels
            )

            # Cast model to expected type for Trainer
            model = cast(AutoModelForSequenceClassification, model)
            # Cast tokenizer to expected type
            tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

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

            # Train with error handling
            train_result = None
            val_metrics = None
            try:
                train_result = trainer.train()
                
                # Evaluate on validation set
                val_metrics = trainer.evaluate()
                val_f1 = val_metrics.get("eval_f1", val_metrics.get("f1", 0.0))

                logger.info(f"Validation F1: {val_f1:.4f}")

                # Report to Optuna
                if train_result:
                    trial.set_user_attr("train_loss", train_result.get("train_loss", 0.0))
                if val_metrics:
                    trial.set_user_attr("val_accuracy", val_metrics.get("eval_accuracy", 0.0))
                    trial.set_user_attr("val_macro_f1", val_metrics.get("eval_macro_f1", 0.0))

                # Log final metrics to MLflow run
                if val_metrics:
                    val_f1 = val_metrics.get("eval_f1", val_metrics.get("f1", 0.0))
                    mlflow.log_metric("eval_f1", val_f1)
                    mlflow.log_metric("val_accuracy", val_metrics.get("eval_accuracy", 0.0))
                    mlflow.log_metric("val_macro_f1", val_metrics.get("eval_macro_f1", 0.0))
                if train_result:
                    mlflow.log_metric("train_loss", train_result.get("train_loss", 0.0))

            except Exception as e:
                logger.error(f"Training failed in trial {trial.number}: {e}")
                # Cleanup before re-raising
                if 'trainer' in locals():
                    trainer.cleanup()
                if 'model' in locals():
                    try:
                        import torch
                        # Use type ignore for dynamic attribute access check
                        if torch.cuda.is_available() and hasattr(model, 'cpu'):
                            model = model.cpu()  # type: ignore
                    except Exception:
                        pass
                clear_gpu_memory()
                if mlflow.active_run():
                    mlflow.end_run()
                raise
            finally:
                # Aggressive cleanup trainer resources (always run)
                if 'trainer' in locals():
                    trainer.cleanup()
                    del trainer
                # Explicitly delete model and move to CPU before deletion
                if 'model' in locals():
                    try:
                        import torch
                        # Use type ignore for dynamic attribute access check
                        if torch.cuda.is_available() and hasattr(model, 'cpu'):
                            model = model.cpu()  # type: ignore
                    except Exception:
                        pass
                    del model
                if 'tokenizer' in locals():
                    del tokenizer
                if 'train_dataset' in locals():
                    del train_dataset
                if 'val_dataset' in locals():
                    del val_dataset
                if 'test_dataset' in locals():
                    del test_dataset
                # Multiple cleanup passes for ROCm
                clear_gpu_memory()
                gc.collect()
                clear_gpu_memory()
                gc.collect()
                time.sleep(2)  # Give more time for cleanup on ROCm

            # End MLflow run for this trial
            if mlflow.active_run():
                mlflow.end_run()

            # Clean up old trial directories to free disk space
            if config.training:
                outputs_dir = Path(config.training.output_dir)
                # Extract base experiment name (remove _trial_X suffix)
                if "_trial_" in config.experiment_name:
                    base_experiment_name = config.experiment_name.rsplit("_trial_", 1)[0]
                else:
                    base_experiment_name = config.experiment_name
                cleanup_old_trials(
                    outputs_dir,
                    base_experiment_name,
                    trial.number,
                    keep_best_n=keep_best_n_trials,
                    study=study_container["study"],
                )

            return val_f1

        except OSError as e:
            if "No space left on device" in str(e) or e.errno == 28:
                logger.error(
                    f"Trial {trial.number + 1:02d} failed: Disk space exhausted. "
                    "Try cleaning up old trial directories manually or increase disk space."
                )
                # Try to clean up old trials to free space
                try:
                    if config.training:
                        outputs_dir = Path(config.training.output_dir)
                        # Extract base experiment name (remove _trial_X suffix)
                        if "_trial_" in config.experiment_name:
                            base_experiment_name = config.experiment_name.rsplit("_trial_", 1)[0]
                        else:
                            base_experiment_name = config.experiment_name
                        cleanup_old_trials(
                            outputs_dir,
                            base_experiment_name,
                            trial.number,
                            keep_best_n=1,  # Keep only 1 best trial when disk is full
                            study=study_container["study"],
                        )
                except Exception:
                    pass
            else:
                logger.error(f"Trial {trial.number + 1:02d} failed: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("trial_state", "FAILED")
                mlflow.end_run()
            return 0.0
        except Exception as e:
            logger.error(f"Trial {trial.number + 1:02d} failed: {e}", exc_info=True)
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
    keep_best_n_trials: int = 3,
) -> optuna.Study:
    """
    Run hyperparameter tuning for a model.
    
    Note: setup_test_environment() is called at module level to prevent
    segmentation faults, especially on ROCm systems.
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

    # Create mutable container to hold study reference for cleanup
    study_container: dict[str, optuna.Study | None] = {"study": None}

    # Create objective function
    objective = create_objective_function(
        str(config_path),
        model_name,
        experiment_name,
        keep_best_n_trials=keep_best_n_trials,
        study_container=study_container,
    )

    # Create MLflow callback
    mlflow_callback = create_mlflow_callback(experiment_name)

    # Update study reference in container
    study_container["study"] = study

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
