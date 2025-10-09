"""
Data loading utilities for text and speech datasets.

This module provides utility functions for loading and preprocessing
text and audio data for Swiss German dialect classification.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import torch
import torchaudio
import numpy as np
from pathlib import Path


def load_audio_file(
    file_path: str,
    target_sample_rate: int = 16000,
    max_duration: Optional[float] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample if necessary.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sampling rate (default: 16kHz for speech models)
        max_duration: Maximum duration in seconds (will truncate if longer)
    
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    # Truncate if needed
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    
    return waveform, sample_rate


def load_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load a text file.
    
    Args:
        file_path: Path to text file
        encoding: Text encoding (default: utf-8)
    
    Returns:
        Text content as string
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read().strip()


def load_dataset_from_directory(
    data_dir: str,
    file_extension: str = '.txt',
    label_mapping: Optional[Dict[str, int]] = None
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load a dataset from a directory structure where subdirectories are class labels.
    
    Expected structure:
        data_dir/
            class1/
                file1.txt
                file2.txt
            class2/
                file1.txt
                file2.txt
    
    Args:
        data_dir: Root directory containing class subdirectories
        file_extension: File extension to look for
        label_mapping: Optional mapping from label names to integers
    
    Returns:
        Tuple of (file_paths, labels, label_mapping)
    """
    data_dir = Path(data_dir)
    file_paths = []
    labels = []
    
    # Get all class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Create label mapping if not provided
    if label_mapping is None:
        label_mapping = {d.name: idx for idx, d in enumerate(sorted(class_dirs))}
    
    # Load files
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in label_mapping:
            continue
        
        label = label_mapping[class_name]
        
        for file_path in class_dir.glob(f"*{file_extension}"):
            file_paths.append(str(file_path))
            labels.append(label)
    
    return file_paths, labels, label_mapping


def create_audio_dataloader(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for audio files.
    
    Args:
        file_paths: List of audio file paths
        labels: List of corresponding labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
    
    Returns:
        PyTorch DataLoader
    """
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels):
            self.paths = paths
            self.labels = labels
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            audio, sr = load_audio_file(self.paths[idx])
            return {
                'audio': audio.squeeze(),
                'label': self.labels[idx],
                'path': self.paths[idx]
            }
    
    dataset = AudioDataset(file_paths, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def create_text_dataloader(
    texts: List[str],
    labels: List[int],
    batch_size: int = 16,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Returns:
        PyTorch DataLoader
    """
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            return {
                'text': self.texts[idx],
                'label': self.labels[idx]
            }
    
    dataset = TextDataset(texts, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def split_dataset(
    data: List[Any],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Tuple[List[Any], List[int]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data: List of data samples
        labels: List of labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (data, labels) tuples
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Create indices and shuffle
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    # Calculate split points
    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Split data and labels
    train_data = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_data = [data[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_data = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels)
    }


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False
) -> str:
    """
    Preprocess text with various options.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_numbers: Remove numbers
    
    Returns:
        Preprocessed text
    """
    import string
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = text.translate(str.maketrans('', '', string.digits))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

