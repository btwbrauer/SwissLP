"""
Tests for speech model loaders.

Tests loading and basic functionality of speech classification models.
"""

import gc

import pytest
import torch

from src.models import (
    load_all_speech_models,
    load_ast,
    load_wav2vec2,
    load_whisper,
)
from src.utils.device import clear_gpu_memory


class TestWav2Vec2:
    """Test Wav2Vec2 model loading."""

    def test_load_wav2vec2_default(self):
        """Test loading default Wav2Vec2 model."""
        device = torch.device("cpu")
        model, feature_extractor = load_wav2vec2(device=device)

        assert model is not None
        assert feature_extractor is not None
        assert model.training is False  # Should be in eval mode
        assert next(model.parameters()).device.type == "cpu"

    def test_load_wav2vec2_with_labels(self):
        """Test loading Wav2Vec2 with custom number of labels."""
        device = torch.device("cpu")
        model, feature_extractor = load_wav2vec2(num_labels=4, device=device)

        assert model is not None
        assert model.config.num_labels == 4
        assert feature_extractor is not None
        assert next(model.parameters()).device.type == "cpu"

    def test_wav2vec2_device_placement(self):
        """Test that model is placed on correct device."""
        device = torch.device("cpu")
        model, _ = load_wav2vec2(device=device)

        assert next(model.parameters()).device.type == device.type

    def test_wav2vec2_custom_model_name(self):
        """Test Wav2Vec2 with custom model name."""
        device = torch.device("cpu")
        model, feature_extractor = load_wav2vec2(model_name="facebook/wav2vec2-base", device=device)

        assert model is not None
        assert feature_extractor is not None
        assert next(model.parameters()).device.type == "cpu"


class TestAST:
    """Test Audio Spectrogram Transformer model loading."""

    def test_load_ast_default(self):
        """Test loading default AST model."""
        device = torch.device("cpu")
        model, feature_extractor = load_ast(device=device)

        assert model is not None
        assert feature_extractor is not None
        assert model.training is False
        assert next(model.parameters()).device.type == "cpu"

    def test_load_ast_with_labels(self):
        """Test loading AST with custom number of labels."""
        device = torch.device("cpu")
        model, feature_extractor = load_ast(num_labels=4, device=device)

        assert model is not None
        assert model.config.num_labels == 4
        assert feature_extractor is not None
        assert next(model.parameters()).device.type == "cpu"

    def test_ast_device_placement(self):
        """Test that AST model is placed on correct device."""
        device = torch.device("cpu")
        model, _ = load_ast(device=device)

        assert next(model.parameters()).device.type == device.type


class TestWhisper:
    """Test Whisper model loading."""

    def test_load_whisper_default(self):
        """Test loading default Whisper model."""
        device = torch.device("cpu")
        model, processor = load_whisper(device=device)

        assert model is not None
        assert processor is not None
        assert model.training is False
        assert next(model.parameters()).device.type == "cpu"

    def test_whisper_device_placement(self):
        """Test that Whisper model is placed on correct device."""
        device = torch.device("cpu")
        model, _ = load_whisper(device=device)

        assert next(model.parameters()).device.type == device.type

    def test_whisper_custom_model_name(self):
        """Test Whisper with custom model name."""
        device = torch.device("cpu")
        model, processor = load_whisper(model_name="openai/whisper-base", device=device)

        assert model is not None
        assert processor is not None
        assert next(model.parameters()).device.type == "cpu"


class TestAllSpeechModels:
    """Test loading all speech models at once."""

    def test_load_all_speech_models(self):
        """Test loading all speech models."""
        clear_gpu_memory()

        device = torch.device("cpu")
        models = None

        try:
            models = load_all_speech_models(device=device)

            assert "wav2vec2" in models
            assert "ast" in models
            assert "whisper" in models

            # Check each model has model and processor
            for model_name in ["wav2vec2", "ast", "whisper"]:
                assert "model" in models[model_name]
                assert "processor" in models[model_name]
                assert models[model_name]["model"] is not None
                assert models[model_name]["processor"] is not None
                # Verify all models are on CPU
                assert next(models[model_name]["model"].parameters()).device.type == "cpu"
        finally:
            # Cleanup to prevent memory issues
            if models is not None:
                # Clean up each model
                for model_name in ["wav2vec2", "ast", "whisper"]:
                    if "model" in models.get(model_name, {}):
                        try:
                            models[model_name]["model"].cpu()
                        except Exception:
                            pass
                del models
            gc.collect()
            clear_gpu_memory()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
