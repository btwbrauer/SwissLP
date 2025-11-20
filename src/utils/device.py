"""
Device detection utilities.
"""

import gc
import os

import torch


def get_device(device: str | None = None) -> torch.device:
    """Automatically detect and return the best available device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def setup_test_environment() -> None:
    """Setup environment for testing to prevent segmentation faults."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    for setter in (torch.set_num_threads, torch.set_num_interop_threads):
        try:
            setter(1)
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
