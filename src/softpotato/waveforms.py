"""
Potential waveform generators for M1.

Canonical representation:
  - NumPy array with shape (n, 2)
  - columns are [E, t]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .timegrid import TimeGrid, uniform_time_grid
from .validation import validate_time, validate_waveform


def waveform_from_arrays(E: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
    """
    Build a canonical waveform from arrays.

    Parameters
    ----------
    E:
        Potential array (volts), length n.
    t:
        Time array (seconds), length n, strictly increasing.

    Returns
    -------
    ndarray
        Shape (n, 2) with columns [E, t].
    """
    tt = validate_time(t, strictly_increasing=True)
    EE = np.asarray(E, dtype=float).astype(np.float64, copy=False)
    if EE.ndim != 1:
        raise ValueError(
            f"E must be a 1D array; got ndim={EE.ndim} and shape={EE.shape}."
        )
    if EE.size != tt.size:
        raise ValueError(f"E and t must have same length; got {EE.size} and {tt.size}.")
    if not np.all(np.isfinite(EE)):
        raise ValueError("E must contain only finite values.")
    w = np.column_stack((EE, tt)).astype(np.float64, copy=False)
    validate_waveform(w)
    return w


def _resolve_time(
    *,
    t: ArrayLike | None = None,
    time_grid: TimeGrid | None = None,
    dt: float | None = None,
    n: int | None = None,
    t_end: float | None = None,
    t_start: float = 0.0,
) -> NDArray[np.float64]:
    if time_grid is not None:
        return time_grid.as_array()
    if t is not None:
        return validate_time(t, strictly_increasing=True)

    # Construct from dt + (n or t_end)
    if dt is None:
        raise ValueError("Provide one of: time_grid, t, or dt (with n or t_end).")
    tg = uniform_time_grid(dt, n=n, t_end=t_end, t_start=t_start)
    return tg.as_array()


def lsv(
    E_start: float,
    E_end: float,
    *,
    time_grid: TimeGrid | None = None,
    t: ArrayLike | None = None,
    dt: float | None = None,
    n: int | None = None,
    t_end: float | None = None,
    scan_rate: float | None = None,
    t_start: float = 0.0,
) -> NDArray[np.float64]:
    """
    Linear sweep voltammetry (LSV): a single linear ramp from E_start to E_end.

    Time can be provided by:
      - time_grid, or
      - t array, or
      - dt with n/t_end, or
      - scan_rate with dt (duration inferred from |E_end - E_start| / |scan_rate|)

    Returns canonical waveform (n, 2) [E, t].
    """
    if scan_rate is not None:
        if not np.isfinite(scan_rate) or scan_rate == 0.0:
            raise ValueError("scan_rate must be finite and non-zero.")
        if dt is None:
            raise ValueError("When using scan_rate, dt must be provided.")
        duration = abs(float(E_end) - float(E_start)) / abs(float(scan_rate))
        tt = _resolve_time(
            t=None, time_grid=None, dt=dt, t_end=duration, t_start=t_start
        )
    else:
        tt = _resolve_time(
            t=t, time_grid=time_grid, dt=dt, n=n, t_end=t_end, t_start=t_start
        )

    if tt.size == 1:
        EE = np.array([float(E_start)], dtype=np.float64)
    else:
        # Ensure exact endpoints in potential space for determinism.
        EE = np.linspace(float(E_start), float(E_end), tt.size, dtype=np.float64)
    return waveform_from_arrays(EE, tt)


def step(
    E_before: float,
    E_after: float,
    t_step: float,
    *,
    time_grid: TimeGrid | None = None,
    t: ArrayLike | None = None,
    dt: float | None = None,
    n: int | None = None,
    t_end: float | None = None,
    t_start: float = 0.0,
) -> NDArray[np.float64]:
    """
    Potential step: E_before for t < t_step, then E_after for t >= t_step.

    Time can be provided by:
      - time_grid, or
      - t array, or
      - dt with n/t_end
    """
    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite.")
    tt = _resolve_time(
        t=t, time_grid=time_grid, dt=dt, n=n, t_end=t_end, t_start=t_start
    )
    if float(t_step) < float(tt[0]) or float(t_step) > float(tt[-1]):
        raise ValueError("t_step must lie within [t_start, t_end] of the time grid.")

    EE = np.full(tt.shape, float(E_before), dtype=np.float64)
    EE[tt >= float(t_step)] = float(E_after)
    return waveform_from_arrays(EE, tt)


def cv(
    E_start: float,
    E_vertex: float,
    *,
    E_return: float | None = None,
    cycles: int = 1,
    time_grid: TimeGrid | None = None,
    t: ArrayLike | None = None,
    dt: float | None = None,
    n: int | None = None,
    t_end: float | None = None,
    scan_rate: float | None = None,
    t_start: float = 0.0,
) -> NDArray[np.float64]:
    """
    Cyclic voltammetry (CV): triangular waveform.

    One "cycle" is defined as:
      E_start -> E_vertex -> E_return
    where E_return defaults to E_start.

    Time can be provided by:
      - time_grid, or
      - t array (interpreted as total timeline for the entire multi-cycle waveform), or
      - dt with n/t_end, or
      - scan_rate with dt (duration inferred per segment)

    Notes
    -----
    If you provide a time array / grid explicitly, the potential is generated by mapping
    phase progress across the full timeline for the requested number of cycles.
    """
    if E_return is None:
        E_return = float(E_start)
    if not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be an integer >= 1.")

    if scan_rate is not None:
        if not np.isfinite(scan_rate) or scan_rate == 0.0:
            raise ValueError("scan_rate must be finite and non-zero.")
        if dt is None:
            raise ValueError("When using scan_rate, dt must be provided.")
        seg1 = abs(float(E_vertex) - float(E_start)) / abs(float(scan_rate))
        seg2 = abs(float(E_return) - float(E_vertex)) / abs(float(scan_rate))
        duration_one = seg1 + seg2
        total = duration_one * float(cycles)
        tt = _resolve_time(t=None, time_grid=None, dt=dt, t_end=total, t_start=t_start)
    else:
        tt = _resolve_time(
            t=t, time_grid=time_grid, dt=dt, n=n, t_end=t_end, t_start=t_start
        )

    if tt.size == 1:
        return waveform_from_arrays([float(E_start)], tt)

    # Map each timepoint into a cycle phase in [0, 1).
    total_span = float(tt[-1] - tt[0])
    if total_span <= 0.0:
        raise ValueError("Time grid must span a positive duration for CV with n>1.")

    # Normalized progress 0..1 across total timeline
    u = (tt - tt[0]) / total_span  # [0, 1]
    # Scale by cycles; get per-cycle position
    uc = u * float(cycles)
    frac = uc - np.floor(uc)  # [0,1)
    # Two-segment triangle: first half ramps start->vertex, second half ramps vertex->return
    # Use a split at seg_ratio based on potential distances to keep scan-rate-consistent phase.
    d1 = abs(float(E_vertex) - float(E_start))
    d2 = abs(float(E_return) - float(E_vertex))
    if d1 == 0.0 and d2 == 0.0:
        EE = np.full(tt.shape, float(E_start), dtype=np.float64)
        return waveform_from_arrays(EE, tt)

    seg_ratio = d1 / (d1 + d2)  # portion of cycle spent on segment 1
    EE = np.empty_like(tt, dtype=np.float64)

    m1 = frac < seg_ratio
    # Segment 1: frac in [0, seg_ratio] -> alpha in [0,1]
    if np.any(m1):
        alpha1 = frac[m1] / seg_ratio if seg_ratio > 0.0 else 0.0
        EE[m1] = float(E_start) + alpha1 * (float(E_vertex) - float(E_start))

    # Segment 2: frac in [seg_ratio, 1) -> beta in [0,1)
    m2 = ~m1
    if np.any(m2):
        beta = (frac[m2] - seg_ratio) / (1.0 - seg_ratio) if seg_ratio < 1.0 else 0.0
        EE[m2] = float(E_vertex) + beta * (float(E_return) - float(E_vertex))

    # Force last sample to land exactly at the intended final point of the last cycle.
    # For u==1.0 (the last timepoint), frac==0.0; so set explicitly:
    EE[-1] = float(E_return)

    return waveform_from_arrays(EE, tt)
