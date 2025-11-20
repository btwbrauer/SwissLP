"""
Speech model loaders for audio classification and processing.

This module provides functions to load pre-trained speech models from Hugging Face:
- Wav2Vec2: Facebook's self-supervised speech representation model
- AST (Audio Spectrogram Transformer): Vision Transformer adapted for audio
- Whisper: OpenAI's robust speech recognition model
"""

from typing import Any

import torch
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ..utils.device import get_device


def load_wav2vec2(
    model_name: str = "facebook/wav2vec2-base",
    num_labels: int | None = None,
    device: torch.device | None = None,
) -> tuple[Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor]:
    """Load Wav2Vec2 model and feature extractor."""
    from ..utils.logging_utils import suppress_transformers_warnings

    device = device or get_device()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    with suppress_transformers_warnings():
        kwargs = {"num_labels": num_labels, "ignore_mismatched_sizes": True} if num_labels else {}
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, **kwargs)

    return model.to(device).eval(), feature_extractor


def load_ast(
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels: int | None = None,
    device: torch.device | None = None,
) -> tuple[ASTForAudioClassification, ASTFeatureExtractor]:
    """Load Audio Spectrogram Transformer (AST) model and feature extractor."""
    from ..utils.logging_utils import suppress_transformers_warnings

    device = device or get_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

    with suppress_transformers_warnings():
        kwargs = {"num_labels": num_labels, "ignore_mismatched_sizes": True} if num_labels else {}
        model = ASTForAudioClassification.from_pretrained(model_name, **kwargs)

    return model.to(device).eval(), feature_extractor


def load_whisper(
    model_name: str = "openai/whisper-base", device: torch.device | None = None
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load Whisper model and processor for speech recognition."""
    from ..utils.logging_utils import suppress_transformers_warnings

    device = device or get_device()
    processor = WhisperProcessor.from_pretrained(model_name)
    with suppress_transformers_warnings():
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return model.to(device).eval(), processor


def load_all_speech_models(
    device: torch.device | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all speech models for comparison/ensemble."""
    device = device or get_device()
    print(f"Loading speech models on device: {device}")

    models = {}
    for name, loader in [
        ("wav2vec2", lambda: load_wav2vec2(device=device)),
        ("ast", lambda: load_ast(device=device)),
        ("whisper", lambda: load_whisper(device=device)),
    ]:
        print(f"Loading {name}...")
        model, processor = loader()
        models[name] = {"model": model, "processor": processor}

    print("✓ All speech models loaded successfully")
    return models
