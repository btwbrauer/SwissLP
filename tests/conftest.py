"""
Global pytest configuration and fixtures.

This module sets up the test environment to prevent segmentation faults
and ensures consistent test behavior across all test files.
"""

import gc

import pytest

from src.utils.device import clear_gpu_memory, setup_test_environment

# Setup test environment once at module level to prevent segfaults
# This must be done before any tests run
setup_test_environment()


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """
    Setup test session environment.

    This fixture runs once per test session to configure the environment
    for all tests, preventing segmentation faults and threading issues.
    """
    # Setup is already done at module level, but this ensures it's done
    # even if conftest is loaded after other modules
    # Function is idempotent, so safe to call multiple times
    setup_test_environment()
    yield
    # Cleanup after all tests
    gc.collect()
    clear_gpu_memory()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Cleanup fixture that runs after each test.

    This ensures proper memory cleanup between tests to prevent
    memory leaks and segmentation faults.
    """
    yield
    # Cleanup after test
    gc.collect()
    clear_gpu_memory()
