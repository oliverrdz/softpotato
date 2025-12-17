"""
SoftPotato v3.0

M1 public surface:
- Time grids
- Potential waveforms
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .waveforms import cv, lsv, step

# Version ---------------------------------------------------------------------

try:
    __version__ = version("softpotato")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "3.0.0-alpha2"

__all__ = [
    "cv",
    "lsv",
    "step",
    "uniform_time_grid",
]
