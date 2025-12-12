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
        dialects: List of dialect codes to include (e.g., ["ch_ag", "ch_lu"])
                 Note: Only Swiss German dialects are supported (no "ch_de" mapping)

    Returns:
        List of examples containing the specified dialects
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Filter examples that contain all requested dialects
    # Note: We no longer map "ch_de" to "de" - only Swiss German dialects are used
    return [ex for ex in data if all(d in ex for d in dialects)]


def split_data(
    data: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data into train, validation, and test sets using stratified sampling.

    Args:
        data: List of data examples
        seed: Random seed for reproducibility
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        stratify_by: Optional field name to stratify by (e.g., "thema" for topic-based stratification)

    Returns:
        Tuple of (train, val, test) splits
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        # Fallback to simple random split if sklearn is not available
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

    # Use stratified split if stratification field is provided
    if stratify_by and data and stratify_by in data[0]:
        # Extract stratification labels
        stratify_labels = [ex.get(stratify_by) for ex in data]

        # First split: train vs (val+test)
        train, temp = train_test_split(
            data,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=stratify_labels,
        )

        # Second split: val vs test
        # Adjust stratification labels for temp split
        temp_labels = [ex.get(stratify_by) for ex in temp]
        val, test = train_test_split(
            temp,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=seed,
            stratify=temp_labels,
        )

        return train, val, test
    else:
        # Simple random split (maintains reproducibility)
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
            # Use dialect key directly (no mapping needed anymore)
            if d not in ex:
                continue
            rows.append(
                {
                    "text": normalize_quotes(ex[d]),
                    "label": dialect2label[d],
                    "id": ex["id"],
                    "dialect": d,
                }
            )
    return pd.DataFrame(rows)


def get_split_ids(subset: list[dict[str, Any]]) -> list[int]:
    """Extract unique IDs from a subset."""
    return sorted({ex["id"] for ex in subset})


def compute_class_weights(
    train_dataset: Dataset, label_column: str = "label"
) -> torch.Tensor | None:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting: weight = n_samples / (n_classes * count_per_class)

    Args:
        train_dataset: Training dataset with labels
        label_column: Name of the label column

    Returns:
        Tensor of class weights, or None if dataset is balanced
    """
    if len(train_dataset) == 0:
        return None

    # Extract labels
    labels = train_dataset[label_column]
    if isinstance(labels, list):
        labels = np.array(labels)
    else:
        labels = np.array(labels)

    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_labels)

    # Compute weights: n_samples / (n_classes * count_per_class)
    weights = n_samples / (n_classes * counts)

    # Normalize to sum to n_classes (optional, but helps with stability)
    weights = weights / weights.sum() * n_classes

    # Check if balanced (all classes have similar counts)
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # If imbalance is small (< 1.2), return None to indicate balanced dataset
    if imbalance_ratio < 1.2:
        return None

    # Create weight tensor ordered by label index
    weight_dict = dict(zip(unique_labels, weights))
    max_label = int(max(unique_labels))
    weight_list = [weight_dict.get(i, 1.0) for i in range(max_label + 1)]

    return torch.tensor(weight_list, dtype=torch.float32)


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

    # Use stratified split by topic if available, otherwise use random split
    stratify_by = "thema" if filtered and "thema" in filtered[0] else None
    train, val, test = split_data(
        filtered, seed, train_ratio, val_ratio, test_ratio, stratify_by=stratify_by
    )
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
