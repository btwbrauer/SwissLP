"""
Device detection utilities.
"""

import gc
import os

import torch


def get_device(device: str | None = None) -> torch.device:
    """
    Automatically detect and return the best available device.

    Args:
        device: Optional device string (e.g., "cuda", "cpu", "mps")
                If None, auto-detects the best available device.

    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def clear_gpu_memory():
    """
    Clear GPU memory cache and run garbage collection.

    This is useful between training runs to prevent memory issues
    and segmentation faults when training multiple models sequentially.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset CUDA context to prevent segfaults in sequential training
        # This ensures a clean state for the next model
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass  # Ignore if not available

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def setup_test_environment():
    """
    Setup environment for testing to prevent segmentation faults.

    This function:
    - Sets PyTorch to single-threaded mode to avoid threading conflicts
    - Disables CUDA memory fragmentation
    - Sets environment variables for thread control

    This function is idempotent and can be called multiple times safely.
    """
    # Set environment variables for thread control (idempotent)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Set PyTorch to single-threaded mode to avoid threading conflicts
    # This prevents segmentation faults in tests
    # These calls may fail if already set, so we catch exceptions
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass  # Already set, ignore

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already set, ignore

    # Disable CUDA memory fragmentation for more predictable behavior
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
