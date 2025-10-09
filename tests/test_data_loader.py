"""
Unit tests for data loading utilities.
"""

import pytest
import torch
import torchaudio
import tempfile
import os
from pathlib import Path
from src.utils.data_loader import (
    load_audio_file,
    load_text_file,
    load_dataset_from_directory,
    create_audio_dataloader,
    create_text_dataloader,
    split_dataset,
    preprocess_text,
)


class TestAudioLoading:
    """Test audio file loading functionality."""
    
    def test_load_audio_file(self):
        """Test loading an audio file."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create a simple audio signal
            sample_rate = 16000
            duration = 1.0
            waveform = torch.randn(1, int(sample_rate * duration))
            torchaudio.save(temp_path, waveform, sample_rate)
            
            # Load the audio file
            loaded_waveform, loaded_sr = load_audio_file(temp_path)
            
            assert loaded_waveform is not None
            assert loaded_sr == sample_rate
            assert loaded_waveform.shape[0] == 1  # Mono
        finally:
            os.unlink(temp_path)
    
    def test_audio_resampling(self):
        """Test audio resampling."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create audio with different sample rate
            original_sr = 44100
            target_sr = 16000
            duration = 1.0
            waveform = torch.randn(1, int(original_sr * duration))
            torchaudio.save(temp_path, waveform, original_sr)
            
            # Load with resampling
            loaded_waveform, loaded_sr = load_audio_file(temp_path, target_sample_rate=target_sr)
            
            assert loaded_sr == target_sr
            expected_samples = int(duration * target_sr)
            assert abs(loaded_waveform.shape[1] - expected_samples) < 10  # Allow small tolerance
        finally:
            os.unlink(temp_path)


class TestTextLoading:
    """Test text file loading functionality."""
    
    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            test_text = "Grüezi mitänand! Wie gaht's?"
            f.write(test_text)
            temp_path = f.name
        
        try:
            loaded_text = load_text_file(temp_path)
            assert loaded_text == test_text
        finally:
            os.unlink(temp_path)


class TestDatasetLoading:
    """Test dataset loading from directory structure."""
    
    def test_load_dataset_from_directory(self):
        """Test loading dataset from directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            class1_dir = Path(temp_dir) / "class1"
            class2_dir = Path(temp_dir) / "class2"
            class1_dir.mkdir()
            class2_dir.mkdir()
            
            # Create sample files
            (class1_dir / "file1.txt").write_text("Sample text 1")
            (class1_dir / "file2.txt").write_text("Sample text 2")
            (class2_dir / "file1.txt").write_text("Sample text 3")
            
            # Load dataset
            file_paths, labels, label_mapping = load_dataset_from_directory(temp_dir, '.txt')
            
            assert len(file_paths) == 3
            assert len(labels) == 3
            assert len(label_mapping) == 2
            assert 'class1' in label_mapping
            assert 'class2' in label_mapping


class TestDataLoaders:
    """Test PyTorch DataLoader creation."""
    
    def test_create_text_dataloader(self):
        """Test creating a text DataLoader."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        labels = [0, 1, 0, 1]
        
        dataloader = create_text_dataloader(texts, labels, batch_size=2, shuffle=False)
        
        assert dataloader is not None
        assert len(dataloader) == 2  # 4 samples / batch_size 2
        
        # Check first batch
        batch = next(iter(dataloader))
        assert 'text' in batch
        assert 'label' in batch
        assert len(batch['text']) == 2


class TestDatasetSplit:
    """Test dataset splitting functionality."""
    
    def test_split_dataset(self):
        """Test splitting dataset into train/val/test."""
        data = list(range(100))
        labels = [i % 3 for i in range(100)]
        
        splits = split_dataset(
            data, labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        train_data, train_labels = splits['train']
        val_data, val_labels = splits['val']
        test_data, test_labels = splits['test']
        
        assert len(train_data) == 70
        assert len(val_data) == 15
        assert len(test_data) == 15
        
        # Check that all data is used
        all_split_data = set(train_data + val_data + test_data)
        assert len(all_split_data) == 100


class TestTextPreprocessing:
    """Test text preprocessing utilities."""
    
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
        result = preprocess_text(
            text,
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True
        )
        assert result == "hello world"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

