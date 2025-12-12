#!/usr/bin/env python3
"""
Test script to verify all text models can train on GPU.

Usage:
    python scripts/test_text_models_training.py --model swissbert
    python scripts/test_text_models_training.py --model german_bert
    python scripts/test_text_models_training.py --model xlm_roberta
    python scripts/test_text_models_training.py --model all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models import load_german_bert, load_swissbert, load_xlm_roberta
from src.training import TextTrainer
from src.training.hyperparameter_tuning import tokenize_dataset
from src.utils.dataset import make_text_datasets
from src.utils.device import clear_gpu_memory, setup_test_environment

# Setup environment
setup_test_environment()

MODEL_LOADERS = {
    "swissbert": load_swissbert,
    "german_bert": load_german_bert,
    "xlm_roberta": load_xlm_roberta,
}

CONFIG_FILES = {
    "swissbert": "configs/swissbert.yaml",
    "german_bert": "configs/german_bert.yaml",
    "xlm_roberta": "configs/xlm_roberta.yaml",
}


def test_model_training(model_name: str) -> bool:
    """Test training a single model on GPU."""
    print("=" * 70)
    print(f"Testing {model_name.upper()} Training on GPU")
    print("=" * 70)

    try:
        # Load config
        config_path = CONFIG_FILES[model_name]
        config = Config.load(config_path)
        print(f"✓ Config loaded: {config.model.model_name}")

        # Reduce to minimal training for testing
        config.training.num_epochs = 1
        config.training.batch_size = 8
        config.training.save_steps = 1000
        config.training.eval_steps = 1000
        config.training.logging_steps = 50

        # Load datasets
        print("\nLoading datasets...")
        train_dataset, val_dataset, _, dialect2label, _, _, _ = make_text_datasets(
            config.data.data_path,
            config.data.dialects,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.training.seed,
        )
        print(f"✓ Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")

        # Load model
        print(f"\nLoading {model_name} model...")
        clear_gpu_memory()
        loader = MODEL_LOADERS[model_name]
        model, tokenizer = loader(num_labels=len(dialect2label))
        device = next(model.parameters()).device
        print(f"✓ Model loaded on device: {device}")

        # Tokenize
        print("\nTokenizing datasets...")
        train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=config.data.max_length)
        val_dataset = tokenize_dataset(val_dataset, tokenizer, max_length=config.data.max_length)
        print("✓ Datasets tokenized")

        # Create trainer
        print("\nCreating trainer...")
        config.experiment_name = f"test_{model_name}"
        trainer = TextTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            mlflow_experiment_name=None,  # No MLflow for testing
        )
        print("✓ Trainer created")

        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        print(f"✓ Training completed! Loss: {train_result.get('train_loss', 'N/A')}")

        # Evaluate
        print("\nEvaluating...")
        val_metrics = trainer.evaluate()
        print(f"✓ Evaluation completed! F1: {val_metrics.get('eval_f1', 'N/A')}")

        # Cleanup
        trainer.cleanup()
        del trainer, model, tokenizer
        clear_gpu_memory()

        print(f"\n{'=' * 70}")
        print(f"✓ {model_name.upper()} training test PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"✗ {model_name.upper()} training test FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test text model training on GPU")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["swissbert", "german_bert", "xlm_roberta", "all"],
        help="Model to test (or 'all' for all models)",
    )
    args = parser.parse_args()

    if args.model == "all":
        models = ["swissbert", "german_bert", "xlm_roberta"]
    else:
        models = [args.model]

    results = {}
    for model_name in models:
        results[model_name] = test_model_training(model_name)
        print("\n")

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {model_name.upper()}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()


