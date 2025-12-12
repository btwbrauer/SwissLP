#!/usr/bin/env python3
"""Flush GPU memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.device import clear_gpu_memory, setup_test_environment

setup_test_environment()
clear_gpu_memory()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("✓ GPU memory flushed and synchronized")
else:
    print("⚠ No CUDA device available")







