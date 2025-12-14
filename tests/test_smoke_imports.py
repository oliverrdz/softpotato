"""
Smoke tests for SoftPotato v3.0.

Goal (SP3-M0): prove the package imports cleanly and basic metadata is present.
"""

from __future__ import annotations

import importlib

import pytest


def test_import_softpotato() -> None:
    mod = importlib.import_module("softpotato")
    assert hasattr(mod, "__doc__")


def test_import_softpotato_core() -> None:
    importlib.import_module("softpotato.core")


def test_exposes_version_attribute() -> None:
    mod = importlib.import_module("softpotato")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)
    assert mod.__version__.strip() != ""


@pytest.mark.parametrize("dep", ["numpy", "scipy", "matplotlib"])
def test_runtime_dependencies_import(dep: str) -> None:
    importlib.import_module(dep)

