"""
Unit tests for speech model loaders.
"""

import pytest
import torch
from src.classification.speech_models import (
    get_device,
    load_wav2vec2,
    load_ast,
    load_whisper,
    load_all_speech_models,
)


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_get_device(self):
        """Test that get_device returns a valid device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']


class TestWav2Vec2:
    """Test Wav2Vec2 model loading."""
    
    def test_load_wav2vec2_default(self):
        """Test loading default Wav2Vec2 model."""
        model, feature_extractor = load_wav2vec2()
        
        assert model is not None
        assert feature_extractor is not None
        assert model.training is False  # Should be in eval mode
    
    def test_load_wav2vec2_with_labels(self):
        """Test loading Wav2Vec2 with custom number of labels."""
        model, feature_extractor = load_wav2vec2(num_labels=4)
        
        assert model is not None
        assert model.config.num_labels == 4
        assert feature_extractor is not None
    
    def test_wav2vec2_device_placement(self):
        """Test that model is placed on correct device."""
        device = get_device()
        model, _ = load_wav2vec2(device=device)
        
        # Check that model is on the correct device
        assert next(model.parameters()).device.type == device.type


class TestAST:
    """Test Audio Spectrogram Transformer model loading."""
    
    def test_load_ast_default(self):
        """Test loading default AST model."""
        model, feature_extractor = load_ast()
        
        assert model is not None
        assert feature_extractor is not None
        assert model.training is False
    
    def test_load_ast_with_labels(self):
        """Test loading AST with custom number of labels."""
        model, feature_extractor = load_ast(num_labels=4)
        
        assert model is not None
        assert model.config.num_labels == 4
        assert feature_extractor is not None


class TestWhisper:
    """Test Whisper model loading."""
    
    def test_load_whisper_default(self):
        """Test loading default Whisper model."""
        model, processor = load_whisper()
        
        assert model is not None
        assert processor is not None
        assert model.training is False
    
    def test_whisper_device_placement(self):
        """Test that Whisper model is placed on correct device."""
        device = get_device()
        model, _ = load_whisper(device=device)
        
        assert next(model.parameters()).device.type == device.type


class TestAllSpeechModels:
    """Test loading all speech models at once."""
    
    def test_load_all_speech_models(self):
        """Test loading all speech models."""
        models = load_all_speech_models()
        
        assert 'wav2vec2' in models
        assert 'ast' in models
        assert 'whisper' in models
        
        # Check each model has model and processor
        for model_name in ['wav2vec2', 'ast', 'whisper']:
            assert 'model' in models[model_name]
            assert 'processor' in models[model_name]
            assert models[model_name]['model'] is not None
            assert models[model_name]['processor'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

