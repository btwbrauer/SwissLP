#!/usr/bin/env python3
"""
Main script for hyperparameter optimization workflow.

This script orchestrates the complete hyperparameter optimization process:
1. Runs hyperparameter tuning for specified models
2. Automatically keeps only the best model and deletes others
3. Can process multiple models sequentially

Usage:
    # Optimize a single model
    python scripts/optimize_models.py --models swissbert --n-trials 20

    # Optimize multiple models sequentially
    python scripts/optimize_models.py --models swissbert german_bert xlm_roberta --n-trials 20

    # Optimize with custom config directory
    python scripts/optimize_models.py --models swissbert --config-dir configs --n-trials 30

    # Skip cleanup (keep all trial models)
    python scripts/optimize_models.py --models swissbert --n-trials 20 --skip-cleanup
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import optimize_single_model
from src.utils.constants import MODEL_DISPLAY_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("datasets.fingerprint").setLevel(logging.ERROR)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["swissbert", "german_bert", "xlm_roberta"],
        help="Models to optimize (can specify multiple)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files (default: configs)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run per model (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per model (default: None, no timeout)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="./outputs",
        help="Outputs directory (default: ./outputs)",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup step (keep all trial models)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Hyperparameter Optimization Workflow")
    logger.info("=" * 70)
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Trials per model: {args.n_trials}")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Outputs directory: {args.outputs_dir}")
    if args.skip_cleanup:
        logger.info("Cleanup: DISABLED (all models will be kept)")
    else:
        logger.info("Cleanup: ENABLED (only best models will be kept)")
    logger.info("=" * 70)

    # Process each model sequentially
    results = {}
    for model_name in args.models:
        success, best_model_path, freed_mb = optimize_single_model(
            model_name=model_name,
            config_dir=args.config_dir,
            n_trials=args.n_trials,
            timeout=args.timeout,
            outputs_dir=args.outputs_dir,
            skip_cleanup=args.skip_cleanup,
        )
        results[model_name] = success

        if not success:
            logger.error(f"\n✗ Failed to optimize {model_name}")
        else:
            logger.info(f"\n✓ Successfully optimized {model_name}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: {status}")

    # Exit with error if any model failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
