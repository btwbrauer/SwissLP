"""
Dataset utilities for Swiss German dialect classification.

This module provides functions for loading, preprocessing, and preparing
datasets for both text and speech classification tasks using the SwissDial dataset.
"""

import json
import random
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
from datasets import Dataset  # type: ignore

# ============================================================================
# Swiss German Dataset Preparation (Project-Specific)
# ============================================================================


def load_swiss_german_data(file_path: str, dialects: list[str]) -> list[dict[str, Any]]:
    """
    Load SwissDial dataset from JSON file and filter by dialects.

    Args:
        file_path: Path to SwissDial JSON file containing sentences
        dialects: List of dialect codes to include (e.g., ["ch_de", "ch_lu"])
                 Note: "ch_de" is mapped to "de" in the data file

    Returns:
        List of examples containing the specified dialects
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Map "ch_de" to "de" for compatibility with data format
    mapped_dialects = ["de" if d == "ch_de" else d for d in dialects]

    return [ex for ex in data if all(d in ex for d in mapped_dialects)]


def split_data(
    data: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: List of data examples
        seed: Random seed for reproducibility
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train, val, test) splits
    """
    random.seed(seed)
    np.random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def normalize_quotes(text: str) -> str:
    """Normalize various quote characters to standard double quotes."""
    text = re.sub(r'[«»""]', '"', text)
    text = re.sub(r'"\s*', '"', text)
    return re.sub(r'\s*"', '"', text)


def flatten_examples(
    subset: list[dict[str, Any]], dialects: list[str], dialect2label: dict[str, int]
) -> pd.DataFrame:
    """Flatten examples to create one row per dialect per sentence."""
    rows = []
    for ex in subset:
        for d in dialects:
            data_key = "de" if d == "ch_de" else d
            if data_key not in ex:
                continue
            rows.append(
                {
                    "text": normalize_quotes(ex[data_key]),
                    "label": dialect2label[d],
                    "id": ex["id"],
                    "dialect": d,
                }
            )
    return pd.DataFrame(rows)


def get_split_ids(subset: list[dict[str, Any]]) -> list[int]:
    """Extract unique IDs from a subset."""
    return sorted({ex["id"] for ex in subset})


def make_text_datasets(
    file_path: str,
    dialects: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset, dict[str, int], list[int], list[int], list[int]]:
    """
    Create text datasets from SwissDial for Swiss German dialect classification.

    Args:
        file_path: Path to SwissDial JSON file with sentences
        dialects: List of dialect codes to include
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, dialect2label,
                 train_ids, val_ids, test_ids)
    """
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please ensure the data file exists at the specified path."
        )

    filtered = load_swiss_german_data(file_path, dialects)
    if not filtered:
        raise ValueError(
            f"No data found after filtering. File: {file_path}, Dialects: {dialects}\n"
            f"Please check that:\n"
            f"  1. The file contains valid JSON data\n"
            f"  2. The examples contain all required dialect keys: {dialects}"
        )

    train, val, test = split_data(filtered, seed, train_ratio, val_ratio, test_ratio)
    dialect2label = {d: i for i, d in enumerate(dialects)}

    train_dataset = Dataset.from_pandas(flatten_examples(train, dialects, dialect2label))
    val_dataset = Dataset.from_pandas(flatten_examples(val, dialects, dialect2label))
    test_dataset = Dataset.from_pandas(flatten_examples(test, dialects, dialect2label))

    train_ids = get_split_ids(train)
    val_ids = get_split_ids(val)
    test_ids = get_split_ids(test)

    return train_dataset, val_dataset, test_dataset, dialect2label, train_ids, val_ids, test_ids


def make_audio_dataframe(
    split_ids: list[int], dialects: list[str], dialect2label: dict[str, int], audio_root: str
) -> pd.DataFrame:
    """Create audio dataframe for a given split."""
    rows = []
    for d in dialects:
        folder = Path(audio_root) / d[3:]  # e.g., 'lu' from 'ch_lu'
        for id_val in split_ids:
            fpath = folder / f"{d}_{id_val:04}.wav"
            rows.append(
                {"audio_path": str(fpath), "label": dialect2label[d], "id": id_val, "dialect": d}
            )
    return pd.DataFrame(rows)


def make_audio_splits(
    audio_root: str,
    dialects: list[str],
    dialect2label: dict[str, int],
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create audio dataframes for train, validation, and test splits."""
    return (
        make_audio_dataframe(train_ids, dialects, dialect2label, audio_root),
        make_audio_dataframe(val_ids, dialects, dialect2label, audio_root),
        make_audio_dataframe(test_ids, dialects, dialect2label, audio_root),
    )


# ============================================================================
# Generic Data Loading Utilities
# ============================================================================


def load_audio_file(
    file_path: str, target_sample_rate: int = 16000, max_duration: float | None = None
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and resample if necessary.

    Uses soundfile for loading (works without torchcodec dependency).

    Args:
        file_path: Path to audio file
        target_sample_rate: Target sampling rate (default: 16kHz for speech models)
        max_duration: Maximum duration in seconds (will truncate if longer)

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    # Load with soundfile (returns [samples, channels] format)
    waveform_np, sample_rate = sf.read(file_path, dtype="float32")

    # Convert to torch tensor and ensure [channels, samples] format
    waveform = torch.from_numpy(waveform_np)
    if waveform.dim() == 1:
        # Mono audio: [samples] -> [1, samples]
        waveform = waveform.unsqueeze(0)
    else:
        # Multi-channel: [samples, channels] -> [channels, samples]
        waveform = waveform.T

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Truncate if needed
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    return waveform, sample_rate


def load_text_file(file_path: str | Path, encoding: str = "utf-8") -> str:
    """Load a text file."""
    return Path(file_path).read_text(encoding=encoding).strip()


def load_dataset_from_directory(
    data_dir: str | Path, file_extension: str = ".txt", label_mapping: dict[str, int] | None = None
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load a dataset from a directory structure where subdirectories are class labels."""
    data_dir_path = Path(data_dir)
    class_dirs = [d for d in data_dir_path.iterdir() if d.is_dir()]

    if label_mapping is None:
        label_mapping = {d.name: idx for idx, d in enumerate(sorted(class_dirs))}

    file_paths = []
    labels = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in label_mapping:
            continue
        label = label_mapping[class_name]
        files = list(class_dir.glob(f"*{file_extension}"))
        file_paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    return file_paths, labels, label_mapping


# ============================================================================
# PyTorch DataLoader Creation
# ============================================================================


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for audio batches with padding.

    Args:
        batch: List of samples from AudioDataset

    Returns:
        Batched dictionary with padded audio tensors
    """
    audio_list = [item["audio"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    paths = [item["path"] for item in batch]

    # Pad audio sequences to same length
    # Find max length
    max_len = max(audio.shape[-1] for audio in audio_list)

    # Pad all sequences to max_len
    padded_audio = []
    for audio in audio_list:
        if audio.dim() == 1:
            # Mono audio: [samples]
            pad_size = max_len - audio.shape[0]
            if pad_size > 0:
                padded = torch.nn.functional.pad(audio, (0, pad_size), mode="constant", value=0.0)
            else:
                padded = audio[:max_len]
            padded_audio.append(padded)
        else:
            # Multi-channel: [channels, samples]
            pad_size = max_len - audio.shape[-1]
            if pad_size > 0:
                padded = torch.nn.functional.pad(audio, (0, pad_size), mode="constant", value=0.0)
            else:
                padded = audio[..., :max_len]
            padded_audio.append(padded.squeeze())

    # Stack into batch tensor
    audio_batch = torch.stack(padded_audio)

    return {"audio": audio_batch, "label": labels, "path": paths}


def create_audio_dataloader(
    file_paths: list[str],
    labels: list[int],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
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
        def __init__(self, paths: list[str], labels: list[int]):
            self.paths = paths
            self.labels = labels

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            audio, sr = load_audio_file(self.paths[idx])
            return {"audio": audio.squeeze(), "label": self.labels[idx], "path": self.paths[idx]}

    dataset = AudioDataset(file_paths, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
    )


def create_text_dataloader(
    texts: list[str], labels: list[int], batch_size: int = 16, shuffle: bool = True
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
        def __init__(self, texts: list[str], labels: list[int]):
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {"text": self.texts[idx], "label": self.labels[idx]}

    dataset = TextDataset(texts, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# Dataset Splitting Utilities
# ============================================================================


def split_dataset(
    data: list[Any],
    labels: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, tuple[list[Any], list[int]]]:
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
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Set random seed
    np.random.seed(random_seed)

    # Create indices and shuffle
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Calculate split points
    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Split data and labels
    train_data = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_data = [data[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_data = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return {
        "train": (train_data, train_labels),
        "val": (val_data, val_labels),
        "test": (test_data, test_labels),
    }


# ============================================================================
# Text Preprocessing
# ============================================================================


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
) -> str:
    """Preprocess text with various options."""
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_numbers:
        text = text.translate(str.maketrans("", "", string.digits))
    return " ".join(text.split())
