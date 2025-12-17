from __future__ import annotations

import numpy as np

from .validation import (
    build_potential_segment,
    validate_cycles,
    validate_finite_float,
    validate_positive_float,
)


def lsv(E_start: float, E_end: float, dE: float, scan_rate: float) -> np.ndarray:
    """
    Linear sweep voltammetry (LSV): single monotonic ramp E_start -> E_end.

    Sampling:
      - Potential samples use nominal increment dE (direction inferred).
      - Endpoint is enforced exactly (E_end).
      - Time is derived from scan_rate: dt_step = dE / scan_rate
      - t[i] = i * dt_step

    Returns
    -------
    np.ndarray
        Array shape (n, 2) with columns [E, t].
    """
    validate_finite_float(E_start, name="E_start")
    validate_finite_float(E_end, name="E_end")
    validate_positive_float(dE, name="dE")
    validate_positive_float(scan_rate, name="scan_rate")

    E = build_potential_segment(E_start, E_end, dE, drop_first=False)
    dt_step = dE / scan_rate
    t = dt_step * np.arange(E.size, dtype=float)

    wave = np.column_stack([E, t])
    return wave


def cv(
    E_start: float,
    E_vertex: float,
    scan_rate: float,
    dE: float,
    E_end: float | None = None,
    cycles: int = 1,
) -> np.ndarray:
    """
    Cyclic voltammetry (CV): repeat a cycle path with constant dE and derived time.

    One cycle:
      Segment A: E_start -> E_vertex
      Segment B: E_vertex -> E_end_cycle
        where E_end_cycle = E_end if provided else E_start

    Join-point de-duplication:
      - Do not repeat E_vertex at A/B boundary.
      - When repeating cycles, drop the first sample of the next cycle if it would
        duplicate the previous cycle's last sample (within tolerance).

    Time:
      - dt_step = dE / scan_rate
      - Each potential sample advances by dt_step
      - Time is strictly increasing across the full waveform.

    Returns
    -------
    np.ndarray
        Array shape (n, 2) with columns [E, t].
    """
    validate_finite_float(E_start, name="E_start")
    validate_finite_float(E_vertex, name="E_vertex")
    if E_end is not None:
        validate_finite_float(E_end, name="E_end")
    validate_positive_float(dE, name="dE")
    validate_positive_float(scan_rate, name="scan_rate")
    validate_cycles(cycles)

    E_end_cycle = E_start if E_end is None else float(E_end)

    E_all: list[np.ndarray] = []

    for k in range(cycles):
        seg_a = build_potential_segment(E_start, E_vertex, dE, drop_first=False)
        seg_b = build_potential_segment(E_vertex, E_end_cycle, dE, drop_first=True)
        E_cycle = np.concatenate([seg_a, seg_b])

        # Avoid duplicating boundary sample between cycles if it matches.
        if k > 0 and E_all:
            prev_last = E_all[-1][-1]
            if np.isclose(E_cycle[0], prev_last, atol=1e-12, rtol=0.0):
                E_cycle = E_cycle[1:]

        E_all.append(E_cycle)

    E = np.concatenate(E_all) if E_all else np.array([float(E_start)], dtype=float)
    dt_step = dE / scan_rate
    t = dt_step * np.arange(E.size, dtype=float)

    wave = np.column_stack([E, t])
    return wave


def step(E_before: float, E_after: float, dt: float, t_end: float) -> np.ndarray:
    """
    Potential step waveform.

    Time grid:
      - n = floor(t_end / dt) + 1
      - t = [0, dt, 2dt, ..., (n-1)dt]
      - No snapping: t[-1] <= t_end and t[-1] > t_end - dt

    Potential:
      - Instantaneous step at t = 0 (recommended spec):
        E[0] = E_after and E is constant E_after for all samples.

    Returns
    -------
    np.ndarray
        Array shape (n, 2) with columns [E, t].
    """
    validate_finite_float(E_before, name="E_before")
    validate_finite_float(E_after, name="E_after")
    validate_positive_float(dt, name="dt")
    validate_positive_float(t_end, name="t_end")

    n = int(np.floor(t_end / dt)) + 1
    t = dt * np.arange(n, dtype=float)

    E = np.full(n, float(E_after), dtype=float)

    wave = np.column_stack([E, t])
    return wave
