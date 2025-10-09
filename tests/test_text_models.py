"""
Unit tests for text model loaders.
"""

import pytest
import torch
from src.classification.text_models import (
    get_device,
    load_swissbert,
    load_german_bert,
    load_xlm_roberta,
    load_byt5,
    load_all_text_models,
)


class TestTextDeviceDetection:
    """Test device detection for text models."""
    
    def test_get_device(self):
        """Test that get_device returns a valid device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']


class TestSwissBERT:
    """Test SwissBERT model loading."""
    
    def test_load_swissbert_default(self):
        """Test loading default SwissBERT model."""
        model, tokenizer = load_swissbert()
        
        assert model is not None
        assert tokenizer is not None
        assert model.training is False
    
    def test_load_swissbert_with_labels(self):
        """Test loading SwissBERT with custom number of labels."""
        model, tokenizer = load_swissbert(num_labels=4)
        
        assert model is not None
        assert model.config.num_labels == 4
        assert tokenizer is not None
    
    def test_swissbert_device_placement(self):
        """Test that model is placed on correct device."""
        device = get_device()
        model, _ = load_swissbert(device=device)
        
        assert next(model.parameters()).device.type == device.type


class TestGermanBERT:
    """Test German BERT model loading."""
    
    def test_load_german_bert_default(self):
        """Test loading default German BERT model."""
        model, tokenizer = load_german_bert()
        
        assert model is not None
        assert tokenizer is not None
        assert model.training is False
    
    def test_load_german_bert_with_labels(self):
        """Test loading German BERT with custom number of labels."""
        model, tokenizer = load_german_bert(num_labels=4)
        
        assert model is not None
        assert model.config.num_labels == 4
        assert tokenizer is not None


class TestXLMRoberta:
    """Test XLM-RoBERTa model loading."""
    
    def test_load_xlm_roberta_default(self):
        """Test loading default XLM-RoBERTa model."""
        model, tokenizer = load_xlm_roberta()
        
        assert model is not None
        assert tokenizer is not None
        assert model.training is False
    
    def test_load_xlm_roberta_with_labels(self):
        """Test loading XLM-RoBERTa with custom number of labels."""
        model, tokenizer = load_xlm_roberta(num_labels=4)
        
        assert model is not None
        assert model.config.num_labels == 4
        assert tokenizer is not None


class TestByT5:
    """Test ByT5 model loading."""
    
    def test_load_byt5_default(self):
        """Test loading default ByT5 model."""
        model, tokenizer = load_byt5()
        
        assert model is not None
        assert tokenizer is not None
        assert model.training is False
    
    def test_byt5_device_placement(self):
        """Test that ByT5 model is placed on correct device."""
        device = get_device()
        model, _ = load_byt5(device=device)
        
        assert next(model.parameters()).device.type == device.type


class TestAllTextModels:
    """Test loading all text models at once."""
    
    def test_load_all_text_models(self):
        """Test loading all text models."""
        models = load_all_text_models()
        
        # Required models
        assert 'swissbert' in models
        assert 'german_bert' in models
        assert 'xlm_roberta' in models
        assert 'byt5' in models
        
        # Check each model has model and tokenizer
        for model_name in ['swissbert', 'german_bert', 'xlm_roberta', 'byt5']:
            assert 'model' in models[model_name]
            assert 'tokenizer' in models[model_name]
            assert models[model_name]['model'] is not None
            assert models[model_name]['tokenizer'] is not None


class TestTextModelInference:
    """Test basic inference with text models."""
    
    def test_swissbert_inference(self):
        """Test basic inference with SwissBERT."""
        model, tokenizer = load_swissbert()
        
        text = "Grüezi, wie gaht's?"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')
    
    def test_german_bert_inference(self):
        """Test basic inference with German BERT."""
        model, tokenizer = load_german_bert()
        
        text = "Guten Tag, wie geht es Ihnen?"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

