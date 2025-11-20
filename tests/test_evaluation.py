"""
Tests for evaluation utilities.

Tests the Evaluator classes and metrics computation.
"""

import gc

import numpy as np
import pytest
import torch

from src.config import Config, DataConfig, ModelConfig
from src.evaluation import ClassificationMetrics, SpeechEvaluator, TextEvaluator, compute_metrics
from src.models import load_swissbert
from src.utils.device import clear_gpu_memory


class TestClassificationMetrics:
    """Tests for ClassificationMetrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ClassificationMetrics(num_classes=8)
        assert metrics.num_classes == 8
        assert len(metrics.class_names) == 8

    def test_metrics_update(self):
        """Test metrics update."""
        metrics = ClassificationMetrics(num_classes=3)

        predictions = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 0, 1])

        metrics.update(predictions, labels)

        assert len(metrics.predictions) == 5
        assert len(metrics.labels) == 5

    def test_metrics_compute(self):
        """Test metrics computation."""
        metrics = ClassificationMetrics(num_classes=3)

        predictions = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 0, 1])

        metrics.update(predictions, labels)
        results = metrics.compute()

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert results["accuracy"] == 1.0

    def test_metrics_compute_empty(self):
        """Test metrics computation with no predictions."""
        metrics = ClassificationMetrics(num_classes=3)

        results = metrics.compute()

        assert "accuracy" in results
        assert results["accuracy"] == 0.0

    def test_metrics_custom_class_names(self):
        """Test metrics with custom class names."""
        class_names = ["Class A", "Class B", "Class C"]
        metrics = ClassificationMetrics(num_classes=3, class_names=class_names)

        assert metrics.class_names == class_names

    def test_metrics_get_summary(self):
        """Test metrics summary generation."""
        metrics = ClassificationMetrics(num_classes=3)

        predictions = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])

        metrics.update(predictions, labels)
        summary = metrics.get_summary()

        assert isinstance(summary, str)
        assert "Accuracy" in summary
        assert "Precision" in summary

    def test_metrics_reset(self):
        """Test metrics reset."""
        metrics = ClassificationMetrics(num_classes=3)

        predictions = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])

        metrics.update(predictions, labels)
        assert len(metrics.predictions) == 3

        metrics.reset()
        assert len(metrics.predictions) == 0
        assert len(metrics.labels) == 0


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_perfect(self):
        """Test metrics computation with perfect predictions."""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compute_metrics_imperfect(self):
        """Test metrics computation with imperfect predictions."""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 1, 0, 1])

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] < 1.0
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_compute_metrics_custom_average(self):
        """Test metrics computation with custom average."""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])

        metrics = compute_metrics(predictions, labels, average="macro")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics


class TestTextEvaluator:
    """Tests for TextEvaluator."""

    def test_text_evaluator_initialization(self):
        """Test TextEvaluator initialization."""
        clear_gpu_memory()

        device = torch.device("cpu")
        model, tokenizer = load_swissbert(num_labels=3, device=device)

        try:
            # Create dummy data
            texts = ["Text 1", "Text 2", "Text 3"]
            labels = [0, 1, 0]

            from src.utils.dataset import create_text_dataloader

            data_loader = create_text_dataloader(texts, labels, batch_size=2, shuffle=False)

            config = Config(
                model=ModelConfig(model_name="test", num_labels=3, device="cpu"),
                data=DataConfig(data_path="./data/test.json"),
            )

            evaluator = TextEvaluator(config, model, tokenizer, data_loader, compute_loss=False)

            assert evaluator.model == model
            assert evaluator.tokenizer == tokenizer
            assert evaluator.model.training is False
        finally:
            if "evaluator" in locals():
                del evaluator
            if "data_loader" in locals():
                del data_loader
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            clear_gpu_memory()

    def test_text_evaluator_eval_step(self):
        """Test TextEvaluator evaluation step."""
        clear_gpu_memory()

        device = torch.device("cpu")
        model, tokenizer = load_swissbert(num_labels=3, device=device)

        try:
            texts = ["Test text"]
            labels = [0]

            from src.utils.dataset import create_text_dataloader

            data_loader = create_text_dataloader(texts, labels, batch_size=1, shuffle=False)

            config = Config(
                model=ModelConfig(model_name="test", num_labels=3, device="cpu"),
                data=DataConfig(data_path="./data/test.json"),
            )

            evaluator = TextEvaluator(config, model, tokenizer, data_loader, compute_loss=False)

            batch = next(iter(data_loader))
            result = evaluator._eval_step(batch)

            assert "predictions" in result
            assert "labels" in result
            assert "loss" not in result
        finally:
            if "evaluator" in locals():
                del evaluator
            if "data_loader" in locals():
                del data_loader
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            clear_gpu_memory()

    def test_text_evaluator_with_loss(self):
        """Test TextEvaluator with loss computation."""
        clear_gpu_memory()

        device = torch.device("cpu")
        model, tokenizer = load_swissbert(num_labels=3, device=device)

        try:
            texts = ["Test text"]
            labels = [0]

            from src.utils.dataset import create_text_dataloader

            data_loader = create_text_dataloader(texts, labels, batch_size=1, shuffle=False)

            config = Config(
                model=ModelConfig(model_name="test", num_labels=3, device="cpu"),
                data=DataConfig(data_path="./data/test.json"),
            )

            evaluator = TextEvaluator(config, model, tokenizer, data_loader, compute_loss=True)

            batch = next(iter(data_loader))
            result = evaluator._eval_step(batch)

            assert "predictions" in result
            assert "labels" in result
            assert "loss" in result
        finally:
            if "evaluator" in locals():
                del evaluator
            if "data_loader" in locals():
                del data_loader
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            clear_gpu_memory()

    def test_text_evaluator_evaluate(self):
        """Test TextEvaluator full evaluation."""
        # Clear GPU memory before loading model to avoid segmentation faults
        clear_gpu_memory()

        # Force CPU for tests to avoid CUDA threading issues
        device = torch.device("cpu")
        model, tokenizer = load_swissbert(num_labels=3, device=device)

        try:
            # Use more samples to ensure we have at least one batch
            # Include all 3 classes to match class_names
            texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
            labels = [0, 1, 2, 0, 1]

            from src.utils.dataset import create_text_dataloader

            data_loader = create_text_dataloader(texts, labels, batch_size=2, shuffle=False)

            config = Config(
                model=ModelConfig(model_name="test", num_labels=3, device="cpu"),
                data=DataConfig(data_path="./data/test.json", max_length=128),
            )

            evaluator = TextEvaluator(
                config,
                model,
                tokenizer,
                data_loader,
                compute_loss=False,
                class_names=["Class0", "Class1", "Class2"],
            )
            results = evaluator.evaluate()

            assert "accuracy" in results
            assert "precision" in results
            assert "recall" in results
            assert "f1" in results
            assert isinstance(results["accuracy"], (int, float))
        finally:
            # Cleanup
            if "evaluator" in locals():
                del evaluator
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            if tokenizer is not None:
                del tokenizer
            if "data_loader" in locals():
                del data_loader
            gc.collect()
            clear_gpu_memory()


class TestSpeechEvaluator:
    """Tests for SpeechEvaluator."""

    def test_speech_evaluator_initialization(self):
        """Test SpeechEvaluator initialization."""
        from src.models import load_wav2vec2

        model, processor = load_wav2vec2(num_labels=3)

        # Create dummy audio data
        audio_paths = []
        labels = []

        from src.utils.dataset import create_audio_dataloader

        data_loader = create_audio_dataloader(audio_paths, labels, batch_size=1, shuffle=False)

        config = Config(
            model=ModelConfig(model_name="test", num_labels=3),
            data=DataConfig(data_path="./data/test.json"),
        )

        evaluator = SpeechEvaluator(config, model, processor, data_loader, compute_loss=False)

        assert evaluator.model == model
        assert evaluator.processor == processor
        assert evaluator.model.training is False

    def test_speech_evaluator_save_results(self):
        """Test SpeechEvaluator save_results method."""
        import tempfile
        from pathlib import Path

        from src.models import load_wav2vec2

        model, processor = load_wav2vec2(num_labels=3)

        audio_paths = []
        labels = []

        from src.utils.dataset import create_audio_dataloader

        data_loader = create_audio_dataloader(audio_paths, labels, batch_size=1, shuffle=False)

        config = Config(
            model=ModelConfig(model_name="test", num_labels=3),
            data=DataConfig(data_path="./data/test.json"),
        )

        evaluator = SpeechEvaluator(config, model, processor, data_loader, compute_loss=False)

        results = {"accuracy": 0.9, "f1": 0.85}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            evaluator.save_results(results, temp_path)
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
