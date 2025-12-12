"""Logging utilities for suppressing verbose output."""

import logging
import warnings
from contextlib import contextmanager

from transformers import logging as transformers_logging  # type: ignore


@contextmanager
def suppress_transformers_warnings():
    """Context manager to suppress verbose Transformers warnings during model loading."""
    transformers_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # Suppress PyTorch deprecation warnings (e.g., reduce_op)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*reduce_op.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*ReduceOp.*")
        try:
            yield
        finally:
            transformers_logging.set_verbosity_warning()
