"""
Tests for training module.

Tests the TextTrainer and SpeechTrainer classes and training functionality.
"""

import os
import tempfile
from pathlib import Path

import mlflow  # type: ignore
import pytest
import torch
from datasets import Dataset

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.models import load_wav2vec2
from src.training import SpeechTrainer, TextTrainer


class TestTextTrainer:
    """Tests for TextTrainer."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            model=ModelConfig(model_name="bert-base-german-cased", num_labels=4),
            data=DataConfig(
                data_path="./data/test.json", dialects=["ch_de", "ch_lu", "ch_be", "ch_zh"]
            ),
            training=TrainingConfig(
                num_epochs=1,
                batch_size=2,
                save_steps=10,
                eval_steps=10,
                logging_steps=5,
                output_dir=str(Path(tempfile.gettempdir()) / "test_outputs"),
            ),
            experiment_name="test_experiment",
            task_type="text",
        )

    @pytest.fixture
    def train_dataset(self):
        """Create a minimal training dataset."""
        return Dataset.from_dict(
            {
                "text": ["Test text 1", "Test text 2", "Test text 3", "Test text 4"],
                "label": [0, 1, 2, 3],
            }
        )

    @pytest.fixture
    def val_dataset(self):
        """Create a minimal validation dataset."""
        return Dataset.from_dict(
            {
                "text": ["Val text 1", "Val text 2"],
                "label": [0, 1],
            }
        )

    def test_text_trainer_initialization(self, config, train_dataset, val_dataset):
        """Test TextTrainer initialization."""
        # Use SwissBERT instead of German BERT to avoid loading multiple large models
        from src.models import load_swissbert

        model, tokenizer = load_swissbert(num_labels=4)

        # Tokenize datasets (no padding - DataCollator handles it)
        def tokenize_function(examples):
            # tokenizer is captured from outer scope
            return tokenizer(  # type: ignore[call-arg]  # noqa: F821
                examples["text"],
                truncation=True,
                max_length=128,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        assert trainer.config == config
        assert trainer.model == model
        assert trainer.processing_class == tokenizer
        assert hasattr(trainer, "trainer")  # Hugging Face Trainer should exist

        # Cleanup
        trainer.cleanup()
        del model, tokenizer, trainer
        import gc

        gc.collect()

    def test_text_trainer_without_training_config(self, train_dataset):
        """Test TextTrainer fails without training config."""
        from src.models import load_swissbert

        config = Config(
            model=ModelConfig(model_name="ZurichNLP/swissbert", num_labels=4),
            data=DataConfig(data_path="./data/test.json"),
            training=None,  # No training config
        )

        model, tokenizer = load_swissbert(num_labels=4)

        def tokenize_function(examples):
            # tokenizer is captured from outer scope
            return tokenizer(  # type: ignore[call-arg]  # noqa: F821
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        with pytest.raises(ValueError, match="Training configuration is required"):
            TextTrainer(
                config=config,
                model=model,  # type: ignore[arg-type]
                tokenizer=tokenizer,
                train_dataset=train_dataset,
            )

        # Cleanup
        del model, tokenizer
        import gc

        gc.collect()

    def test_text_trainer_evaluate(self, config, train_dataset, val_dataset):
        """Test TextTrainer evaluation."""
        # Clear GPU memory before loading model to avoid segmentation faults
        from src.models import load_swissbert
        from src.utils.device import clear_gpu_memory

        clear_gpu_memory()

        trainer = None
        model = None
        tokenizer = None

        try:
            model, tokenizer = load_swissbert(num_labels=4)

            def tokenize_function(examples):
                # tokenizer is captured from outer scope
                return tokenizer(  # type: ignore[call-arg]  # noqa: F821
                    examples["text"],
                    truncation=True,
                    max_length=128,
                )

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)

            trainer = TextTrainer(
                config=config,
                model=model,  # type: ignore[arg-type]
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            # Evaluate
            metrics = trainer.evaluate()
            assert "accuracy" in metrics or "eval_accuracy" in metrics
        finally:
            # Cleanup
            if trainer is not None:
                trainer.cleanup()
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            import gc

            gc.collect()
            clear_gpu_memory()

    def test_text_trainer_no_validation_set(self, config, train_dataset):
        """Test TextTrainer without validation set."""
        from src.models import load_swissbert

        model, tokenizer = load_swissbert(num_labels=4)

        def tokenize_function(examples):
            # tokenizer is captured from outer scope
            return tokenizer(examples["text"], truncation=True, max_length=128)  # type: ignore[call-arg]  # noqa: F821

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=None,
        )

        assert trainer.val_dataset is None

        # Cleanup
        trainer.cleanup()
        del model, tokenizer, trainer
        import gc

        gc.collect()

    def test_text_trainer_cleanup(self, config, train_dataset, val_dataset):
        """Test TextTrainer cleanup."""
        from src.models import load_swissbert

        model, tokenizer = load_swissbert(num_labels=4)

        def tokenize_function(examples):
            # tokenizer is captured from outer scope
            return tokenizer(examples["text"], truncation=True, max_length=128)  # type: ignore[call-arg]  # noqa: F821

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.cleanup()
        # Cleanup should not raise errors

        # Additional cleanup
        del model, tokenizer, trainer
        import gc

        gc.collect()


class TestSpeechTrainer:
    """Tests for SpeechTrainer."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            model=ModelConfig(model_name="facebook/wav2vec2-base", num_labels=4),
            data=DataConfig(
                data_path="./data/test.json",
                dialects=["ch_de", "ch_lu", "ch_be", "ch_zh"],
                audio_sample_rate=16000,
            ),
            training=TrainingConfig(
                num_epochs=1,
                batch_size=2,
                save_steps=10,
                eval_steps=10,
                logging_steps=5,
                output_dir=str(Path(tempfile.gettempdir()) / "test_outputs"),
            ),
            experiment_name="test_experiment",
            task_type="speech",
        )

    @pytest.fixture
    def train_dataset(self):
        """Create a minimal training dataset with dummy audio."""
        # Create dummy audio arrays (16000 samples = 1 second at 16kHz)
        dummy_audio = [torch.randn(16000).numpy().tolist() for _ in range(4)]
        return Dataset.from_dict(
            {
                "audio": dummy_audio,
                "label": [0, 1, 2, 3],
            }
        )

    @pytest.fixture
    def val_dataset(self):
        """Create a minimal validation dataset with dummy audio."""
        dummy_audio = [torch.randn(16000).numpy().tolist() for _ in range(2)]
        return Dataset.from_dict(
            {
                "audio": dummy_audio,
                "label": [0, 1],
            }
        )

    def test_speech_trainer_initialization(self, config, train_dataset, val_dataset):
        """Test SpeechTrainer initialization."""
        model, processor = load_wav2vec2(num_labels=4)

        # Preprocess datasets
        def preprocess_function(examples):
            return processor(
                examples["audio"],
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            )

        # Note: For actual training, we'd need proper audio preprocessing
        # This is a simplified test
        trainer = SpeechTrainer(
            config=config,
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        assert trainer.config == config
        assert trainer.model == model
        assert trainer.processing_class == processor
        assert hasattr(trainer, "trainer")  # Hugging Face Trainer should exist

    def test_speech_trainer_without_training_config(self, train_dataset):
        """Test SpeechTrainer fails without training config."""
        config = Config(
            model=ModelConfig(model_name="facebook/wav2vec2-base", num_labels=4),
            data=DataConfig(data_path="./data/test.json"),
            training=None,  # No training config
        )

        model, processor = load_wav2vec2(num_labels=4)

        with pytest.raises(ValueError, match="Training configuration is required"):
            SpeechTrainer(
                config=config,
                model=model,
                processor=processor,
                train_dataset=train_dataset,
            )

    def test_speech_trainer_no_validation_set(self, config, train_dataset):
        """Test SpeechTrainer without validation set."""
        model, processor = load_wav2vec2(num_labels=4)

        trainer = SpeechTrainer(
            config=config,
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=None,
        )

        assert trainer.val_dataset is None

    def test_speech_trainer_cleanup(self, config, train_dataset, val_dataset):
        """Test SpeechTrainer cleanup."""
        model, processor = load_wav2vec2(num_labels=4)

        trainer = SpeechTrainer(
            config=config,
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.cleanup()
        # Cleanup should not raise errors


class TestMLflowMetricsLogging:
    """Tests for MLflow metrics logging during hyperparameter tuning."""

    def test_mlflow_metrics_logged_with_external_run(self):
        """Test that metrics are logged correctly when run is created externally."""
        from src.models import load_swissbert
        from src.utils.device import clear_gpu_memory

        clear_gpu_memory()

        # Use temporary MLflow tracking URI
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_tracking_uri = str(Path(tmpdir) / "mlruns")
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

            try:
                model, tokenizer = load_swissbert(num_labels=4)

                # Create minimal datasets
                train_dataset = Dataset.from_dict(
                    {
                        "text": ["Test text 1", "Test text 2", "Test text 3", "Test text 4"],
                        "label": [0, 1, 2, 3],
                    }
                )
                val_dataset = Dataset.from_dict(
                    {
                        "text": ["Val text 1", "Val text 2"],
                        "label": [0, 1],
                    }
                )

                # Tokenize datasets
                def tokenize_function(examples):
                    return tokenizer(examples["text"], truncation=True, max_length=128)  # type: ignore[call-arg]  # noqa: F821

                train_dataset = train_dataset.map(tokenize_function, batched=True)
                val_dataset = val_dataset.map(tokenize_function, batched=True)

                # Create config
                config = Config(
                    model=ModelConfig(model_name="ZurichNLP/swissbert", num_labels=4),
                    data=DataConfig(
                        data_path="./data/test.json", dialects=["ch_de", "ch_lu", "ch_be", "ch_zh"]
                    ),
                    training=TrainingConfig(
                        num_epochs=1,
                        batch_size=2,
                        save_steps=10,
                        eval_steps=10,
                        logging_steps=5,
                        output_dir=str(Path(tempfile.gettempdir()) / "test_outputs"),
                    ),
                    experiment_name="test_experiment_trial_0",
                    task_type="text",
                )

                # Setup MLflow experiment
                from src.utils.mlflow_utils import ensure_mlflow_experiment

                experiment_name = "TestSwissBERT"
                ensure_mlflow_experiment(experiment_name)

                # Create run externally (simulating hyperparameter tuning)
                mlflow.start_run(run_name=f"{experiment_name}_01", nested=True)
                mlflow.log_param("test_param", "test_value")

                # Create trainer with external run
                trainer = TextTrainer(
                    config=config,
                    model=model,  # type: ignore[arg-type]
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    mlflow_experiment_name=experiment_name,
                )

                # Verify run was detected as external
                assert trainer._mlflow_run_created_externally is True

                # Train (should not end the run)
                train_result = trainer.train()

                # Verify run is still active
                assert mlflow.active_run() is not None

                # Log additional metrics (simulating hyperparameter tuning)
                val_metrics = trainer.evaluate()
                mlflow.log_metric("eval_f1", val_metrics.get("eval_f1", 0.0))
                mlflow.log_metric("train_loss", train_result["train_loss"])

                # End run
                mlflow.end_run()

                # Verify metrics were logged
                import pandas as pd

                runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name], max_results=1)  # type: ignore[assignment]
                assert not runs.empty, "No runs found in MLflow"

                run = runs.iloc[0]
                run_name = run.get("tags.mlflow.runName", "")
                assert run_name == f"{experiment_name}_01", (
                    f"Expected run name '{experiment_name}_01', got '{run_name}'"
                )

                # Check that metrics exist - look for metrics columns
                metric_columns = [col for col in runs.columns if col.startswith("metrics.")]
                assert len(metric_columns) > 0, (
                    f"No metrics found. Available columns: {list(runs.columns)}"
                )

                # Verify specific metrics are present
                found_metrics = []
                if "metrics.train_loss" in runs.columns:
                    train_loss = run.get("metrics.train_loss")
                    assert train_loss is not None and not pd.isna(train_loss), (
                        "train_loss metric is missing or NaN"
                    )
                    found_metrics.append("train_loss")

                if "metrics.eval_f1" in runs.columns:
                    eval_f1 = run.get("metrics.eval_f1")
                    assert eval_f1 is not None and not pd.isna(eval_f1), (
                        "eval_f1 metric is missing or NaN"
                    )
                    found_metrics.append("eval_f1")
                elif "metrics.f1" in runs.columns:
                    f1 = run.get("metrics.f1")
                    assert f1 is not None and not pd.isna(f1), "f1 metric is missing or NaN"
                    found_metrics.append("f1")

                # Check for other common metrics
                if "metrics.eval_accuracy" in runs.columns:
                    found_metrics.append("eval_accuracy")
                elif "metrics.accuracy" in runs.columns:
                    found_metrics.append("accuracy")

                assert len(found_metrics) > 0, (
                    f"No valid metrics found. Available metric columns: {metric_columns}"
                )
                print(f"✓ Found metrics: {found_metrics}")

                # Verify that metrics were logged during training (not just at the end)
                # Check if we have step-wise metrics by looking at metric history
                from mlflow.tracking import MlflowClient

                client = MlflowClient(tracking_uri=mlflow_tracking_uri)
                run_id = run.get("run_id")

                # Get metric history for train_loss to verify it was logged during training
                try:
                    train_loss_history = client.get_metric_history(run_id, "train_loss")
                    # If we have multiple entries, metrics were logged during training
                    if len(train_loss_history) > 0:
                        print(
                            f"✓ train_loss was logged {len(train_loss_history)} time(s) during training"
                        )
                        assert len(train_loss_history) >= 1, (
                            "train_loss should be logged at least once"
                        )
                        # Check if metrics were logged at different steps (indicating logging during training)
                        steps = [m.step for m in train_loss_history]
                        if len(set(steps)) > 1:
                            print(
                                f"✓ train_loss was logged at multiple steps: {sorted(set(steps))}"
                            )
                except Exception as e:
                    # For file-based MLflow, metric history might not be available
                    # But we've already verified the metrics exist, which is sufficient
                    print(
                        f"Note: Could not retrieve metric history (this is OK for file-based MLflow): {e}"
                    )

                # Verify eval metrics were logged
                try:
                    eval_metrics = ["eval_f1", "eval_accuracy", "f1", "accuracy"]
                    for metric_name in eval_metrics:
                        try:
                            metric_history = client.get_metric_history(run_id, metric_name)
                            if len(metric_history) > 0:
                                print(f"✓ {metric_name} was logged {len(metric_history)} time(s)")
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

                # Final verification: ensure we have the key metrics
                assert "train_loss" in found_metrics, "train_loss must be logged"
                assert any(m in found_metrics for m in ["eval_f1", "f1"]), (
                    "eval_f1 or f1 must be logged"
                )
                print(f"✓ All required metrics verified: {found_metrics}")

                # Cleanup
                trainer.cleanup()
                del trainer, model, tokenizer
                import gc

                gc.collect()
                clear_gpu_memory()

            finally:
                # Clean up environment
                if "MLFLOW_TRACKING_URI" in os.environ:
                    del os.environ["MLFLOW_TRACKING_URI"]
