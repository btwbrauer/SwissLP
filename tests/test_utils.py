"""
Tests for utility functions.

Tests device detection and other utility functions.
"""

import torch

from src.utils.device import clear_gpu_memory, get_device


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_get_device_auto_detect(self):
        """Test automatic device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_explicit_cuda(self):
        """Test explicit CUDA device selection."""
        device = get_device("cuda")
        assert device.type == "cuda"

    def test_get_device_explicit_mps(self):
        """Test explicit MPS device selection."""
        device = get_device("mps")
        assert device.type == "mps"


class TestGPUMemoryManagement:
    """Tests for GPU memory management utilities."""

    def test_clear_gpu_memory(self):
        """Test clearing GPU memory cache."""
        # Should not raise any errors
        clear_gpu_memory()

    def test_clear_gpu_memory_multiple_calls(self):
        """Test that clear_gpu_memory can be called multiple times."""
        clear_gpu_memory()
        clear_gpu_memory()
        clear_gpu_memory()
        # Should not raise any errors
