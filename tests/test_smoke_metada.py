"""
Metadata smoke tests.

These are intentionally light. They help catch packaging regressions early.
"""

from __future__ import annotations

import importlib.metadata

import pytest


def test_distribution_metadata_available() -> None:
    """
    When installed (e.g., `pip install -e .`), distribution metadata should exist.
    If running tests without installation, skip instead of failing noisily.
    """
    try:
        name = importlib.metadata.metadata("softpotato")["Name"]
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("softpotato not installed; run `pip install -e .` before pytest")

    assert name.lower() == "softpotato"


def test_distribution_version_nonempty() -> None:
    try:
        version = importlib.metadata.version("softpotato")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("softpotato not installed; run `pip install -e .` before pytest")

    assert isinstance(version, str)
    assert version.strip() != ""
