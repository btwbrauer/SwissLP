"""
Tests for dataset utilities.

Tests Swiss German dataset preparation and generic data loading utilities.
"""

import os
import tempfile
from pathlib import Path

import pytest
import soundfile as sf
import torch

from src.utils.dataset import (
    create_text_dataloader,
    load_audio_file,
    load_dataset_from_directory,
    load_text_file,
    make_text_datasets,
    preprocess_text,
    split_dataset,
)


class TestSwissGermanDataset:
    """Tests for Swiss German dataset preparation functions."""

    def test_make_text_datasets_file_not_found(self):
        """Test that make_text_datasets raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            make_text_datasets(
                file_path="./nonexistent.json",
                dialects=["ch_de", "ch_lu"],
            )

    def test_make_text_datasets_empty_result(self, tmp_path):
        """Test that make_text_datasets raises error for empty filtered data."""
        # Create empty JSON file
        test_file = tmp_path / "empty.json"
        test_file.write_text("[]")

        with pytest.raises(ValueError, match="No data found after filtering"):
            make_text_datasets(
                file_path=str(test_file),
                dialects=["ch_de", "ch_lu"],
            )


class TestAudioLoading:
    """Tests for audio file loading functionality."""

    def test_load_audio_file(self):
        """Test loading an audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sample_rate = 16000
            duration = 1.0
            waveform = torch.randn(1, int(sample_rate * duration))
            waveform = torch.clamp(waveform, -1.0, 1.0)
            waveform_np = waveform.squeeze(0).numpy()
            sf.write(temp_path, waveform_np, sample_rate)

            loaded_waveform, loaded_sr = load_audio_file(temp_path)

            assert loaded_waveform is not None
            assert loaded_sr == sample_rate
            assert loaded_waveform.shape[0] == 1  # Mono
        finally:
            os.unlink(temp_path)

    def test_load_audio_file_resampling(self):
        """Test audio resampling."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            original_sr = 44100
            target_sr = 16000
            duration = 1.0
            waveform = torch.randn(1, int(original_sr * duration))
            waveform = torch.clamp(waveform, -1.0, 1.0)
            waveform_np = waveform.squeeze(0).numpy()
            sf.write(temp_path, waveform_np, original_sr)

            loaded_waveform, loaded_sr = load_audio_file(temp_path, target_sample_rate=target_sr)

            assert loaded_sr == target_sr
            expected_samples = int(duration * target_sr)
            assert abs(loaded_waveform.shape[1] - expected_samples) < 10
        finally:
            os.unlink(temp_path)

    def test_load_audio_file_max_duration(self):
        """Test audio truncation with max_duration."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sample_rate = 16000
            duration = 2.0
            max_duration = 1.0
            waveform = torch.randn(1, int(sample_rate * duration))
            waveform = torch.clamp(waveform, -1.0, 1.0)
            waveform_np = waveform.squeeze(0).numpy()
            sf.write(temp_path, waveform_np, sample_rate)

            loaded_waveform, loaded_sr = load_audio_file(
                temp_path, target_sample_rate=sample_rate, max_duration=max_duration
            )

            assert loaded_sr == sample_rate
            expected_samples = int(max_duration * sample_rate)
            assert loaded_waveform.shape[1] <= expected_samples
        finally:
            os.unlink(temp_path)


class TestTextLoading:
    """Tests for text file loading functionality."""

    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            test_text = "Grüezi mitänand! Wie gaht's?"
            f.write(test_text)
            temp_path = f.name

        try:
            loaded_text = load_text_file(temp_path)
            assert loaded_text == test_text
        finally:
            os.unlink(temp_path)

    def test_load_text_file_encoding(self):
        """Test loading text file with custom encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="latin-1"
        ) as f:
            test_text = "Test text"
            f.write(test_text)
            temp_path = f.name

        try:
            loaded_text = load_text_file(temp_path, encoding="latin-1")
            assert loaded_text == test_text
        finally:
            os.unlink(temp_path)


class TestDatasetLoading:
    """Tests for dataset loading from directory structure."""

    def test_load_dataset_from_directory(self):
        """Test loading dataset from directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            class1_dir = Path(temp_dir) / "class1"
            class2_dir = Path(temp_dir) / "class2"
            class1_dir.mkdir()
            class2_dir.mkdir()

            (class1_dir / "file1.txt").write_text("Sample text 1")
            (class1_dir / "file2.txt").write_text("Sample text 2")
            (class2_dir / "file1.txt").write_text("Sample text 3")

            file_paths, labels, label_mapping = load_dataset_from_directory(temp_dir, ".txt")

            assert len(file_paths) == 3
            assert len(labels) == 3
            assert len(label_mapping) == 2
            assert "class1" in label_mapping
            assert "class2" in label_mapping

    def test_load_dataset_from_directory_custom_mapping(self):
        """Test loading dataset with custom label mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            class1_dir = Path(temp_dir) / "class1"
            class1_dir.mkdir()
            (class1_dir / "file1.txt").write_text("Sample text 1")

            custom_mapping = {"class1": 5}
            file_paths, labels, label_mapping = load_dataset_from_directory(
                temp_dir, ".txt", label_mapping=custom_mapping
            )

            assert len(file_paths) == 1
            assert labels[0] == 5
            assert label_mapping == custom_mapping


class TestDataLoaders:
    """Tests for PyTorch DataLoader creation."""

    def test_create_text_dataloader(self):
        """Test creating a text DataLoader."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        labels = [0, 1, 0, 1]

        dataloader = create_text_dataloader(texts, labels, batch_size=2, shuffle=False)

        assert dataloader is not None
        assert len(dataloader) == 2

        batch = next(iter(dataloader))
        assert "text" in batch
        assert "label" in batch
        assert len(batch["text"]) == 2

    def test_create_text_dataloader_shuffle(self):
        """Test text DataLoader with shuffling."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        labels = [0, 1, 0, 1]

        dataloader = create_text_dataloader(texts, labels, batch_size=2, shuffle=True)

        assert dataloader is not None
        assert len(dataloader) == 2


class TestDatasetSplit:
    """Tests for dataset splitting functionality."""

    def test_split_dataset(self):
        """Test splitting dataset into train/val/test."""
        data = list(range(100))
        labels = [i % 3 for i in range(100)]

        splits = split_dataset(
            data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        train_data, train_labels = splits["train"]
        val_data, val_labels = splits["val"]
        test_data, test_labels = splits["test"]

        assert len(train_data) == 70
        assert len(val_data) == 15
        assert len(test_data) == 15

        all_split_data = set(train_data + val_data + test_data)
        assert len(all_split_data) == 100

    def test_split_dataset_ratios_sum_to_one(self):
        """Test that split_dataset validates ratio sum."""
        data = list(range(10))
        labels = [0] * 10

        with pytest.raises(AssertionError, match="Ratios must sum to 1.0"):
            split_dataset(data, labels, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestTextPreprocessing:
    """Tests for text preprocessing utilities."""

    def test_preprocess_text_lowercase(self):
        """Test text lowercasing."""
        text = "Hello WORLD"
        result = preprocess_text(text, lowercase=True)
        assert result == "hello world"

    def test_preprocess_text_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World!"
        result = preprocess_text(text, remove_punctuation=True)
        assert result == "hello world"

    def test_preprocess_text_remove_numbers(self):
        """Test number removal."""
        text = "Hello 123 World 456"
        result = preprocess_text(text, remove_numbers=True)
        assert result == "hello world"

    def test_preprocess_text_combined(self):
        """Test combined preprocessing."""
        text = "Hello, World 123!"
        result = preprocess_text(text, lowercase=True, remove_punctuation=True, remove_numbers=True)
        assert result == "hello world"

    def test_preprocess_text_no_options(self):
        """Test preprocessing with no options enabled."""
        text = "Hello, World 123!"
        result = preprocess_text(
            text, lowercase=False, remove_punctuation=False, remove_numbers=False
        )
        assert result == "Hello, World 123!"
