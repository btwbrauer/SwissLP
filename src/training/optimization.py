"""
Model optimization workflow.

This module provides functions to orchestrate the complete hyperparameter
optimization process including tuning and cleanup.
"""

import logging
from pathlib import Path

from ..utils.constants import MODEL_DISPLAY_NAMES
from .hyperparameter_tuning import run_hyperparameter_tuning
from .model_cleanup import keep_best_model

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "swissbert": "swissbert.yaml",
    "german_bert": "german_bert.yaml",
    "xlm_roberta": "xlm_roberta.yaml",
}


def optimize_single_model(
    model_name: str,
    config_dir: str | Path = "configs",
    n_trials: int = 20,
    timeout: int | None = None,
    outputs_dir: str | Path = "./outputs",
    skip_cleanup: bool = False,
) -> tuple[bool, Path | None, float]:
    """
    Optimize hyperparameters for a single model.

    This function:
    1. Runs hyperparameter tuning
    2. Optionally keeps only the best model and deletes others

    Args:
        model_name: Model name (swissbert, german_bert, xlm_roberta)
        config_dir: Directory containing config files
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None = no timeout)
        outputs_dir: Outputs directory
        skip_cleanup: If True, skip cleanup (keep all trial models)

    Returns:
        Tuple of (success, best_model_path, freed_mb)
    """
    config_dir = Path(config_dir)
    outputs_dir = Path(outputs_dir)

    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        logger.error(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False, None, 0.0

    config_path = config_dir / MODEL_CONFIGS[model_name]

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False, None, 0.0

    logger.info("=" * 70)
    logger.info(
        f"Starting hyperparameter optimization for {MODEL_DISPLAY_NAMES.get(model_name, model_name)}"
    )
    logger.info("=" * 70)

    # Run hyperparameter tuning
    study = run_hyperparameter_tuning(
        config_path=config_path,
        model_name=model_name,
        n_trials=n_trials,
        timeout=timeout,
        study_name=None,  # Auto-generate
        mlflow_experiment_name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
    )

    # Save best hyperparameters to config file
    if study.best_trial:
        from ..config import Config

        config = Config.load(config_path)
        best_params = study.best_params

        if config.training is not None:
            config.training.learning_rate = best_params.get(
                "learning_rate", config.training.learning_rate
            )
            config.training.batch_size = best_params.get("batch_size", config.training.batch_size)
            config.training.warmup_steps = best_params.get(
                "warmup_steps", config.training.warmup_steps
            )
            config.training.weight_decay = best_params.get(
                "weight_decay", config.training.weight_decay
            )
            config.training.lr_scheduler_type = best_params.get(
                "lr_scheduler_type", config.training.lr_scheduler_type
            )
            if "gradient_accumulation_steps" in best_params:
                config.training.gradient_accumulation_steps = best_params[
                    "gradient_accumulation_steps"
                ]

        config.experiment_name = f"{config.experiment_name}_optimized"
        optimized_config_path = config_dir / f"{config_path.stem}_optimized.yaml"
        config.save(str(optimized_config_path))

        logger.info("\n" + "=" * 70)
        logger.info("Best Hyperparameters Saved")
        logger.info("=" * 70)
        logger.info(f"✓ Saved optimized config to: {optimized_config_path}")
        logger.info(f"✓ Best F1 Score: {study.best_value:.4f}")
        logger.info("✓ Best Parameters:")
        for key, value in best_params.items():
            logger.info(f"    {key}: {value}")

    # Cleanup: keep only best model
    best_model_path = None
    freed_mb = 0.0

    if not skip_cleanup:
        logger.info("\n" + "=" * 70)
        logger.info("Cleaning up: keeping only best model...")
        logger.info("=" * 70)

        from ..config import Config

        config = Config.load(config_path)
        best_model_path, freed_mb = keep_best_model(
            outputs_dir=outputs_dir,
            experiment_name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
            base_experiment_name=config.experiment_name,
            metric="eval_f1",
            dry_run=False,
        )

        if best_model_path:
            logger.info(f"\n✓ Optimization complete for {model_name}")
            logger.info(f"✓ Best model saved at: {best_model_path}")
            logger.info(f"✓ Freed approximately {freed_mb:.1f} MB")
        else:
            logger.warning(f"Could not determine best model for {model_name}")
    else:
        logger.info("\nSkipping cleanup (all trial models kept)")

    return True, best_model_path, freed_mb
