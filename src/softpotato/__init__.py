"""
SoftPotato v3.0

M1 public surface:
- Time grids
- Potential waveforms
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Version ---------------------------------------------------------------------

try:
    __version__ = version("softpotato")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "3.0.0-alpha1"

# Time ------------------------------------------------------------------------

from .timegrid import TimeGrid, uniform_time_grid

# Validation ------------------------------------------------------------------
from .validation import (
    validate_time,
    validate_waveform,
)

# Waveforms -------------------------------------------------------------------
from .waveforms import (
    cv,
    lsv,
    step,
    waveform_from_arrays,
)

__all__ = [
    "__version__",
    # Time
    "TimeGrid",
    "uniform_time_grid",
    # Waveforms
    "lsv",
    "cv",
    "step",
    "waveform_from_arrays",
    # Validation
    "validate_time",
    "validate_waveform",
]
