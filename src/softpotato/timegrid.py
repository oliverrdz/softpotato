"""
Time grid utilities for M1.

A TimeGrid is an immutable container for a validated 1D time vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .validation import validate_time


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """
    Validated time grid.

    Parameters
    ----------
    t:
        1D time array (seconds). Must be strictly increasing and finite.

    Notes
    -----
    - This class is intentionally lightweight: later milestones can extend
      time-base generation without coupling to solvers.
    """

    t: NDArray[np.float64]

    def __post_init__(self) -> None:
        tt = validate_time(self.t, strictly_increasing=True)
        object.__setattr__(self, "t", tt)

    @property
    def n(self) -> int:
        """Number of time points."""
        return int(self.t.size)

    @property
    def t_start(self) -> float:
        """First time point (seconds)."""
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        """Last time point (seconds)."""
        return float(self.t[-1])

    @property
    def dt(self) -> float:
        """
        Nominal timestep (seconds).

        For uniform grids this equals the constant delta between points.
        For non-uniform grids, returns the median delta (stable, deterministic).
        """
        if self.n < 2:
            return 0.0
        return float(np.median(np.diff(self.t)))

    def as_array(self) -> NDArray[np.float64]:
        """Return the underlying 1D time array (view)."""
        return self.t


def uniform_time_grid(
    dt: float,
    *,
    n: int | None = None,
    t_end: float | None = None,
    t_start: float = 0.0,
) -> TimeGrid:
    """
    Construct a uniform time grid.

    Provide exactly one of:
      - n: number of points (t = t_start + dt * arange(n))
      - t_end: inclusive end time; requires that (t_end - t_start) / dt is an integer

    Parameters
    ----------
    dt:
        Timestep in seconds. Must be > 0.
    n:
        Number of time points (>= 1).
    t_end:
        Inclusive end time in seconds (>= t_start).
    t_start:
        Start time in seconds.

    Returns
    -------
    TimeGrid
        Validated uniform grid.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")
    if (n is None) == (t_end is None):
        raise ValueError("Provide exactly one of n or t_end.")

    if n is not None:
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be an integer >= 1.")
        t = t_start + dt * np.arange(n, dtype=np.float64)
        return TimeGrid(t=t)

    # t_end provided (n is None here)
    if t_end is None:
        # Defensive; logically unreachable due to XOR check above.
        raise ValueError("t_end must be provided when n is None.")

    t_end_f: float = float(t_end)

    if not np.isfinite(t_end_f):
        raise ValueError("t_end must be finite.")
    if t_end_f < t_start:
        raise ValueError("t_end must be >= t_start.")

    span = float(t_end_f - t_start)
    # Require exact integer number of steps for reliability.
    steps = span / float(dt) if span != 0.0 else 0.0
    k = int(np.round(steps))

    if not np.isclose(steps, k, rtol=0.0, atol=1e-12):
        raise ValueError(
            "t_end must satisfy (t_end - t_start) / dt being an integer "
            f"(got {(t_end_f - t_start) / dt:.16g})."
        )

    n_calc = k + 1
    t = t_start + dt * np.arange(n_calc, dtype=np.float64)
    return TimeGrid(t=t)
