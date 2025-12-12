"""
Text model loaders for text classification and NLP tasks.

This module provides functions to load pre-trained text models:
- SwissBERT: BERT model fine-tuned on Swiss German data
- German BERT: BERT model for German language
- XLM-R: Cross-lingual model supporting 100+ languages
- ByT5: Byte-level T5 model for multilingual tasks
- fastText: Efficient text classification and word embeddings
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch  # type: ignore
from transformers import (  # type: ignore
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from ..utils.device import get_device


def _load_auto_model(
    model_name: str,
    num_labels: int | None = None,
    device: torch.device | None = None,
    post_load_hook: Callable[[AutoModel | AutoModelForSequenceClassification], None] | None = None,
    **kwargs,
) -> tuple[AutoModel | AutoModelForSequenceClassification, AutoTokenizer]:
    """Helper function to load AutoModel with common patterns."""
    from ..utils.logging_utils import suppress_transformers_warnings

    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with suppress_transformers_warnings():
        if num_labels is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, ignore_mismatched_sizes=True, **kwargs
            )
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)

    if post_load_hook is not None:
        post_load_hook(model)

    # Minimal approach: just move to device and set eval mode
    # Let the Trainer handle everything else
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_swissbert(
    model_name: str = "ZurichNLP/swissbert",
    num_labels: int | None = None,
    device: torch.device | None = None,
    default_language: str = "de_CH",
) -> tuple[AutoModel | AutoModelForSequenceClassification, AutoTokenizer]:
    """Load SwissBERT model and tokenizer."""

    def set_language(model: AutoModel | AutoModelForSequenceClassification) -> None:
        if hasattr(model, "set_default_language"):
            model.set_default_language(default_language)  # type: ignore

    return _load_auto_model(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
        post_load_hook=set_language,
    )


def load_german_bert(
    model_name: str = "bert-base-german-cased",
    num_labels: int | None = None,
    device: torch.device | None = None,
) -> tuple[AutoModel | AutoModelForSequenceClassification, AutoTokenizer]:
    """Load German BERT model and tokenizer.
    
    Uses AutoModel approach (exactly like SwissBERT) for ROCm compatibility.
    This is the same loading method that works for SwissBERT.
    """
    # Use the exact same approach as SwissBERT - _load_auto_model
    # This ensures identical behavior and should work on ROCm
    return _load_auto_model(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
    )


def load_xlm_roberta(
    model_name: str = "xlm-roberta-base",
    num_labels: int | None = None,
    device: torch.device | None = None,
) -> tuple[AutoModel | AutoModelForSequenceClassification, AutoTokenizer]:
    """Load XLM-RoBERTa (XLM-R) model and tokenizer."""
    return _load_auto_model(model_name=model_name, num_labels=num_labels, device=device)


def load_byt5(
    model_name: str = "google/byt5-small", device: torch.device | None = None
) -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Load ByT5 model and tokenizer."""
    from ..utils.logging_utils import suppress_transformers_warnings

    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with suppress_transformers_warnings():
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Move to device and eval mode manually since T5ForConditionalGeneration
    # might not fully support the fluent .to().eval() chain in static analysis
    model = model.to(device)
    model.eval()

    return model, tokenizer


def load_fasttext(
    model_path: str | None = None,
    download_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
) -> Any:
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
        import fasttext  # type: ignore
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


def load_all_text_models(device: torch.device | None = None) -> dict[str, dict[str, Any]]:
    """Load all text models for comparison/ensemble."""
    device = device or get_device()
    print(f"Loading text models on device: {device}")

    models = {}
    for name, loader in [
        ("swissbert", lambda: load_swissbert(device=device)),
        ("german_bert", lambda: load_german_bert(device=device)),
        ("xlm_roberta", lambda: load_xlm_roberta(device=device)),
        ("byt5", lambda: load_byt5(device=device)),
    ]:
        print(f"Loading {name}...")
        model, tokenizer = loader()
        models[name] = {"model": model, "tokenizer": tokenizer}

    try:
        print("Loading fastText (optional)...")
        models["fasttext"] = {"model": load_fasttext(), "tokenizer": None}
    except ImportError as e:
        print(f"Skipping fastText: {e}")

    print("✓ All text models loaded successfully")
    return models
