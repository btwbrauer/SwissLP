"""
MLflow utilities for experiment management.

Provides consolidated functions for MLflow tracking setup and experiment management.
"""

import logging
import os

import mlflow  # type: ignore

logger = logging.getLogger(__name__)


def setup_mlflow_tracking() -> str:
    """
    Setup MLflow tracking URI.

    Defaults to http://localhost:5000 (MLflow server).
    For local filesystem backend, use: export MLFLOW_TRACKING_URI="./mlruns"

    Note: Filesystem backend is deprecated. Use SQLite or remote server instead.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Handle local filesystem paths (deprecated, but still supported)
    if tracking_uri.startswith("/") and not tracking_uri.startswith("//"):
        if not os.path.exists(tracking_uri) or not os.access(tracking_uri, os.W_OK):
            logger.warning(
                f"Filesystem tracking URI '{tracking_uri}' not accessible. "
                "Falling back to http://localhost:5000. "
                "Note: Filesystem backend is deprecated. Use SQLite or remote server."
            )
            tracking_uri = "http://localhost:5000"
        else:
            logger.warning(
                f"Using deprecated filesystem backend: {tracking_uri}. "
                "Consider using SQLite (sqlite:///mlflow.db) or remote server instead."
            )

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def ensure_mlflow_experiment(experiment_name: str) -> str:
    """Ensure MLflow experiment exists and is active."""
    from mlflow.tracking import MlflowClient  # type: ignore

    tracking_uri = setup_mlflow_tracking()

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        if hasattr(experiment, "lifecycle_stage") and experiment.lifecycle_stage == "deleted":
            try:
                MlflowClient().restore_experiment(experiment.experiment_id)
            except Exception as e:
                logger.warning(f"Could not restore experiment '{experiment_name}': {e}")
        mlflow.set_experiment(experiment_name)
        return tracking_uri

    try:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "unique constraint" in error_msg:
            try:
                client = MlflowClient()
                all_experiments = client.search_experiments(view_type=1)  # 1 = ALL
                for exp in all_experiments:
                    if exp.name == experiment_name:
                        if hasattr(exp, "lifecycle_stage") and exp.lifecycle_stage == "deleted":
                            client.restore_experiment(exp.experiment_id)
                        mlflow.set_experiment(experiment_name)
                        return tracking_uri
            except Exception:
                pass
        logger.warning(f"Could not create experiment '{experiment_name}': {e}")

    return tracking_uri
