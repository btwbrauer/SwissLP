"""
Classification module for Swiss German language processing.

This module provides pre-trained model loaders for both text and speech classification.
"""

from .text_models import (
    load_swissbert,
    load_german_bert,
    load_xlm_roberta,
    load_byt5,
    load_fasttext,
    load_all_text_models,
)

from .speech_models import (
    load_wav2vec2,
    load_ast,
    load_whisper,
    load_all_speech_models,
    get_device,
)

__all__ = [
    # Text models
    'load_swissbert',
    'load_german_bert',
    'load_xlm_roberta',
    'load_byt5',
    'load_fasttext',
    'load_all_text_models',
    # Speech models
    'load_wav2vec2',
    'load_ast',
    'load_whisper',
    'load_all_speech_models',
    # Utilities
    'get_device',
]

