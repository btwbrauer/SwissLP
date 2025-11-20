"""Logging utilities for suppressing verbose output."""

import logging
from contextlib import contextmanager

from transformers import logging as transformers_logging  # type: ignore


@contextmanager
def suppress_transformers_warnings():
    """Context manager to suppress verbose Transformers warnings during model loading."""
    transformers_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        yield
    finally:
        transformers_logging.set_verbosity_warning()
