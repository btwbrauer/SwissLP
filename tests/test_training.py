"""
Tests for training module.

Tests the TextTrainer and SpeechTrainer classes and training functionality.
"""

import tempfile
from pathlib import Path

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
            return tokenizer(  # noqa: F821
                examples["text"],
                truncation=True,
                max_length=128,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,
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
            return tokenizer(  # noqa: F821
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        with pytest.raises(ValueError, match="Training configuration is required"):
            TextTrainer(
                config=config,
                model=model,
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

        try:
            model, tokenizer = load_swissbert(num_labels=4)

            def tokenize_function(examples):
                # tokenizer is captured from outer scope
                return tokenizer(  # noqa: F821
                    examples["text"],
                    truncation=True,
                    max_length=128,
                )

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)

            trainer = TextTrainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            # Evaluate
            metrics = trainer.evaluate()
            assert "accuracy" in metrics or "eval_accuracy" in metrics
        finally:
            # Cleanup
            trainer.cleanup()
            del trainer, model, tokenizer
            import gc

            gc.collect()
            clear_gpu_memory()

    def test_text_trainer_no_validation_set(self, config, train_dataset):
        """Test TextTrainer without validation set."""
        from src.models import load_swissbert

        model, tokenizer = load_swissbert(num_labels=4)

        def tokenize_function(examples):
            # tokenizer is captured from outer scope
            return tokenizer(examples["text"], truncation=True, max_length=128)  # noqa: F821

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,
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
            return tokenizer(examples["text"], truncation=True, max_length=128)  # noqa: F821

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        trainer = TextTrainer(
            config=config,
            model=model,
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
