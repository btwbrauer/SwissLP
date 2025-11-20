"""
Classification module for Swiss German language processing.

This module provides pre-trained model loaders for both text and speech classification.
"""

from ..utils.device import get_device
from .speech_models import (
    load_all_speech_models,
    load_ast,
    load_wav2vec2,
    load_whisper,
)
from .text_models import (
    load_all_text_models,
    load_byt5,
    load_fasttext,
    load_german_bert,
    load_swissbert,
    load_xlm_roberta,
)

__all__ = [
    # Text models
    "load_swissbert",
    "load_german_bert",
    "load_xlm_roberta",
    "load_byt5",
    "load_fasttext",
    "load_all_text_models",
    # Speech models
    "load_wav2vec2",
    "load_ast",
    "load_whisper",
    "load_all_speech_models",
    # Utilities
    "get_device",
]
