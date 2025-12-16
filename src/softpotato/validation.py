"""
Validation utilities for M1 (time grids + potential waveforms).

Canonical waveform representation:
  - NumPy array with shape (n, 2)
  - columns are [E, t]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Optional structured validation result (not required by callers)."""

    ok: bool
    message: str = ""


def _as_1d_float_array(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array; got ndim={a.ndim} and shape={a.shape}."
        )
    return a.astype(np.float64, copy=False)


def validate_time(
    t: ArrayLike, *, strictly_increasing: bool = True
) -> NDArray[np.float64]:
    """
    Validate a time vector.

    Invariants enforced:
      - 1D float array
      - finite values
      - monotonic increasing (strict by default)
    """
    tt = _as_1d_float_array(t, name="t")
    if tt.size == 0:
        raise ValueError("t must be non-empty.")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must contain only finite values.")
    dt = np.diff(tt)
    if strictly_increasing:
        if not np.all(dt > 0.0):
            raise ValueError("t must be strictly increasing.")
    else:
        if not np.all(dt >= 0.0):
            raise ValueError("t must be non-decreasing.")
    return tt


def validate_waveform(w: ArrayLike) -> NDArray[np.float64]:
    """
    Validate a waveform array.

    Invariants enforced:
      - array-like convertible to float
      - shape (n, 2), n >= 1
      - finite values
      - t strictly increasing
    """
    a = np.asarray(w, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"waveform must have shape (n, 2); got shape={a.shape}.")
    if a.shape[0] < 1:
        raise ValueError("waveform must have at least 1 row.")
    if not np.all(np.isfinite(a)):
        raise ValueError("waveform must contain only finite values.")
    # Validate time column
    validate_time(a[:, 1], strictly_increasing=True)
    return a.astype(np.float64, copy=False)
