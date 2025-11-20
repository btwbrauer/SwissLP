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
    """
    Load Wav2Vec2 model and feature extractor.

    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (if None, loads pre-trained)
        device: Device to load model on (if None, auto-detect)

    Returns:
        Tuple of (model, feature_extractor)
    """
    if device is None:
        device = get_device()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    from ..utils.logging_utils import suppress_transformers_warnings

    with suppress_transformers_warnings():
        kwargs = {}
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
            kwargs["ignore_mismatched_sizes"] = True
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, **kwargs)

    model = model.to(device)
    model.eval()

    return model, feature_extractor


def load_ast(
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels: int | None = None,
    device: torch.device | None = None,
) -> tuple[ASTForAudioClassification, ASTFeatureExtractor]:
    """
    Load Audio Spectrogram Transformer (AST) model and feature extractor.

    Args:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (if None, loads pre-trained)
        device: Device to load model on (if None, auto-detect)

    Returns:
        Tuple of (model, feature_extractor)
    """
    if device is None:
        device = get_device()

    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

    from ..utils.logging_utils import suppress_transformers_warnings

    with suppress_transformers_warnings():
        kwargs = {}
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
            kwargs["ignore_mismatched_sizes"] = True
        model = ASTForAudioClassification.from_pretrained(model_name, **kwargs)

    model = model.to(device)
    model.eval()

    return model, feature_extractor


def load_whisper(
    model_name: str = "openai/whisper-base", device: torch.device | None = None
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load Whisper model and processor for speech recognition.

    Args:
        model_name: Hugging Face model identifier (tiny, base, small, medium, large)
        device: Device to load model on (if None, auto-detect)

    Returns:
        Tuple of (model, processor)
    """
    if device is None:
        device = get_device()

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    return model, processor


def load_all_speech_models(
    device: torch.device | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load all speech models for comparison/ensemble.

    Args:
        device: Device to load models on (if None, auto-detect)

    Returns:
        Dictionary containing all loaded models and their processors
    """
    if device is None:
        device = get_device()

    print(f"Loading speech models on device: {device}")

    models = {}

    print("Loading Wav2Vec2...")
    wav2vec2_model, wav2vec2_processor = load_wav2vec2(device=device)
    models["wav2vec2"] = {"model": wav2vec2_model, "processor": wav2vec2_processor}

    print("Loading AST...")
    ast_model, ast_processor = load_ast(device=device)
    models["ast"] = {"model": ast_model, "processor": ast_processor}

    print("Loading Whisper...")
    whisper_model, whisper_processor = load_whisper(device=device)
    models["whisper"] = {"model": whisper_model, "processor": whisper_processor}

    print("✓ All speech models loaded successfully")

    return models
