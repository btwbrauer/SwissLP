"""
Model cleanup utilities for hyperparameter optimization.

This module provides functions to keep only the best model from hyperparameter
optimization and delete others.
"""

import logging
import os
import shutil
from pathlib import Path

import mlflow  # type: ignore
import optuna  # type: ignore

logger = logging.getLogger(__name__)


def find_best_trial_from_mlflow(experiment_name: str, metric: str = "eval_f1") -> int | None:
    """
    Find the best trial number from MLflow experiment.

    Args:
        experiment_name: MLflow experiment name
        metric: Metric to optimize (default: eval_f1)

    Returns:
        Trial number of the best run, or None if not found
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    import pandas as pd

    runs: pd.DataFrame = mlflow.search_runs(  # type: ignore
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        logger.warning(f"No runs found in experiment '{experiment_name}'")
        return None

    best_run = runs.iloc[0]
    run_name = best_run.get("tags.mlflow.runName", "")

    # Extract trial number from run name (e.g., "Trial 5" -> 5)
    if "Trial" in run_name:
        try:
            trial_num = int(run_name.split()[-1])
            metric_value = best_run.get(f"metrics.{metric}", "N/A")
            logger.info(f"Best trial from MLflow: Trial {trial_num} ({metric}={metric_value:.4f})")
            return trial_num
        except (ValueError, IndexError):
            pass

    logger.warning(f"Could not extract trial number from run name: {run_name}")
    return None


def find_best_trial_from_optuna(study_name: str, storage: str | None = None) -> int | None:
    """
    Find the best trial number from Optuna study.

    Args:
        study_name: Optuna study name
        storage: Optuna storage URL (default: None, uses in-memory)

    Returns:
        Trial number of the best trial, or None if not found
    """
    if storage:
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        # Try to load from default SQLite storage
        storage_path = Path.home() / ".optuna" / f"{study_name}.db"
        if storage_path.exists():
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path}")
        else:
            logger.warning(f"Optuna study '{study_name}' not found in default location")
            return None

    if study.best_trial:
        trial_num = study.best_trial.number
        best_value = study.best_value
        logger.info(f"Best trial from Optuna: Trial {trial_num} (F1={best_value:.4f})")
        return trial_num

    return None


def find_trial_directories(outputs_dir: Path, base_experiment_name: str) -> dict[int, Path]:
    """
    Find all trial directories matching the pattern.

    Args:
        outputs_dir: Root outputs directory
        base_experiment_name: Base experiment name (without _trial_X suffix)

    Returns:
        Dictionary mapping trial numbers to directory paths
    """
    trial_dirs = {}
    pattern = f"{base_experiment_name}_trial_"

    for exp_dir in outputs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        if exp_dir.name.startswith(pattern):
            try:
                trial_num = int(exp_dir.name.replace(pattern, ""))
                trial_dirs[trial_num] = exp_dir
            except ValueError:
                continue

    return trial_dirs


def keep_best_model(
    outputs_dir: str | Path = "./outputs",
    experiment_name: str | None = None,
    study_name: str | None = None,
    base_experiment_name: str | None = None,
    metric: str = "eval_f1",
    dry_run: bool = False,
) -> tuple[Path | None, float]:
    """
    Keep only the best model and delete others.

    Args:
        outputs_dir: Root outputs directory
        experiment_name: MLflow experiment name (to find best trial)
        study_name: Optuna study name (alternative to experiment_name)
        base_experiment_name: Base experiment name (if None, inferred from directories)
        metric: Metric to optimize (default: eval_f1)
        dry_run: If True, only show what would be done without actually doing it

    Returns:
        Tuple of (best_model_path, freed_size_mb)
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        logger.error(f"Outputs directory does not exist: {outputs_path}")
        return None, 0.0

    # Find best trial
    best_trial_num = None

    if experiment_name:
        best_trial_num = find_best_trial_from_mlflow(experiment_name, metric)
    elif study_name:
        best_trial_num = find_best_trial_from_optuna(study_name)
    else:
        logger.error("Must provide either experiment_name or study_name")
        return None, 0.0

    if best_trial_num is None:
        logger.error("Could not determine best trial number")
        return None, 0.0

    # Determine base experiment name
    if base_experiment_name is None:
        # Try to infer from trial directories
        trial_dirs = {}
        for exp_dir in outputs_path.iterdir():
            if exp_dir.is_dir() and "_trial_" in exp_dir.name:
                base_name = exp_dir.name.rsplit("_trial_", 1)[0]
                trial_dirs[base_name] = True

        if len(trial_dirs) == 1:
            base_experiment_name = list(trial_dirs.keys())[0]
            logger.info(f"Inferred base experiment name: {base_experiment_name}")
        else:
            logger.error(
                "Could not infer base experiment name. Please provide base_experiment_name"
            )
            return None, 0.0

    # At this point, base_experiment_name is guaranteed to be a string
    assert base_experiment_name is not None, "base_experiment_name must be set at this point"

    # Find all trial directories
    trial_dirs = find_trial_directories(outputs_path, base_experiment_name)

    if not trial_dirs:
        logger.warning(
            f"No trial directories found matching pattern: {base_experiment_name}_trial_*"
        )
        return None, 0.0

    logger.info(f"\nFound {len(trial_dirs)} trial directories:")
    for trial_num in sorted(trial_dirs.keys()):
        size_mb = sum(f.stat().st_size for f in trial_dirs[trial_num].rglob("*") if f.is_file()) / (
            1024 * 1024
        )
        marker = " <-- BEST" if trial_num == best_trial_num else ""
        logger.info(f"  Trial {trial_num}: {trial_dirs[trial_num].name} ({size_mb:.1f} MB){marker}")

    if best_trial_num not in trial_dirs:
        logger.error(f"Best trial {best_trial_num} not found in outputs directory")
        return None, 0.0

    # Rename best model
    best_dir = trial_dirs[best_trial_num]
    best_new_name = f"{base_experiment_name}_best"
    best_new_path = outputs_path / best_new_name

    if dry_run:
        logger.info(f"\n[DRY RUN] Would rename: {best_dir.name} -> {best_new_name}")
    else:
        if best_new_path.exists():
            logger.warning(f"{best_new_name} already exists. Removing it first...")
            shutil.rmtree(best_new_path)
        best_dir.rename(best_new_path)
        logger.info(f"\n✓ Renamed best model: {best_dir.name} -> {best_new_name}")

    # Delete other trials
    other_trials = [trial_num for trial_num in trial_dirs.keys() if trial_num != best_trial_num]
    total_size_mb = 0.0

    for trial_num in other_trials:
        trial_dir = trial_dirs[trial_num]
        size_mb = sum(f.stat().st_size for f in trial_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        total_size_mb += size_mb

        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {trial_dir.name} ({size_mb:.1f} MB)")
        else:
            shutil.rmtree(trial_dir)
            logger.info(f"✓ Deleted: {trial_dir.name} ({size_mb:.1f} MB)")

    if dry_run:
        logger.info(f"\n[DRY RUN] Would free approximately {total_size_mb:.1f} MB")
    else:
        logger.info(f"\n✓ Cleanup complete! Freed approximately {total_size_mb:.1f} MB")
        logger.info(f"✓ Best model saved as: {best_new_path}")

    return best_new_path if not dry_run else None, total_size_mb
