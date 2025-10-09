"""
Text model loaders for text classification and NLP tasks.

This module provides functions to load pre-trained text models:
- SwissBERT: BERT model fine-tuned on Swiss German data
- German BERT: BERT model for German language
- XLM-R: Cross-lingual model supporting 100+ languages
- ByT5: Byte-level T5 model for multilingual tasks
- fastText: Efficient text classification and word embeddings
"""

from typing import Tuple, Optional, Dict
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
)
import os
import tempfile


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_swissbert(
    model_name: str = "ZurichNLP/swissbert",
    num_labels: Optional[int] = None,
    device: Optional[torch.device] = None,
    default_language: str = "de_CH"
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load SwissBERT model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (if specified, loads classifier)
        device: Device to load model on (if None, auto-detect)
        default_language: Default language for XMOD model (default: de_CH for Swiss German)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModel.from_pretrained(model_name)
    
    # SwissBERT uses XMOD architecture which requires a default language
    model.set_default_language(default_language)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_german_bert(
    model_name: str = "bert-base-german-cased",
    num_labels: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load German BERT model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (if specified, loads classifier)
        device: Device to load model on (if None, auto-detect)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModel.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_xlm_roberta(
    model_name: str = "xlm-roberta-base",
    num_labels: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load XLM-RoBERTa (XLM-R) model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (if specified, loads classifier)
        device: Device to load model on (if None, auto-detect)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModel.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_byt5(
    model_name: str = "google/byt5-small",
    device: Optional[torch.device] = None
) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    Load ByT5 model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier (small, base, large, xl, xxl)
        device: Device to load model on (if None, auto-detect)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_fasttext(
    model_path: Optional[str] = None,
    download_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
) -> 'fasttext.FastText._FastText':
    """
    Load fastText model for text classification.
    
    Note: fastText is loaded differently as it's not a transformer model.
    This is a placeholder that shows how to integrate fastText.
    
    Args:
        model_path: Path to local fastText model file
        download_url: URL to download model if not present locally
    
    Returns:
        fastText model object
    """
    try:
        import fasttext
    except ImportError:
        raise ImportError(
            "fasttext is not available in nixpkgs python packages. "
            "You can install it separately with pip if needed: pip install fasttext"
        )
    
    if model_path is None:
        # Use a temporary directory for downloaded models
        cache_dir = os.path.expanduser("~/.cache/fasttext")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "lid.176.bin")
        
        if not os.path.exists(model_path):
            print(f"Downloading fastText model from {download_url}...")
            import urllib.request
            urllib.request.urlretrieve(download_url, model_path)
            print(f"Model saved to {model_path}")
    
    model = fasttext.load_model(model_path)
    return model


def load_all_text_models(device: Optional[torch.device] = None) -> Dict:
    """
    Load all text models for comparison/ensemble.
    
    Args:
        device: Device to load models on (if None, auto-detect)
    
    Returns:
        Dictionary containing all loaded models and their tokenizers
    """
    if device is None:
        device = get_device()
    
    print(f"Loading text models on device: {device}")
    
    models = {}
    
    print("Loading SwissBERT...")
    swissbert_model, swissbert_tokenizer = load_swissbert(device=device)
    models['swissbert'] = {
        'model': swissbert_model,
        'tokenizer': swissbert_tokenizer
    }
    
    print("Loading German BERT...")
    german_bert_model, german_bert_tokenizer = load_german_bert(device=device)
    models['german_bert'] = {
        'model': german_bert_model,
        'tokenizer': german_bert_tokenizer
    }
    
    print("Loading XLM-RoBERTa...")
    xlmr_model, xlmr_tokenizer = load_xlm_roberta(device=device)
    models['xlm_roberta'] = {
        'model': xlmr_model,
        'tokenizer': xlmr_tokenizer
    }
    
    print("Loading ByT5...")
    byt5_model, byt5_tokenizer = load_byt5(device=device)
    models['byt5'] = {
        'model': byt5_model,
        'tokenizer': byt5_tokenizer
    }
    
    # Note: fastText is optional and handled separately
    try:
        print("Loading fastText (optional)...")
        fasttext_model = load_fasttext()
        models['fasttext'] = {
            'model': fasttext_model,
            'tokenizer': None  # fastText doesn't use a tokenizer
        }
    except ImportError as e:
        print(f"Skipping fastText: {e}")
    
    print("✓ All text models loaded successfully")
    
    return models

