from __future__ import annotations

import numpy as np

_FLOAT_TOL = 1e-12


def _is_finite_float(x: float, *, name: str) -> None:
    """Raise ValueError if x is not a finite float."""
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite; got {x!r}")


def validate_positive_float(x: float, *, name: str) -> None:
    """Raise ValueError if x is not finite or not strictly > 0."""
    _is_finite_float(x, name=name)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0; got {x!r}")


def validate_finite_float(x: float, *, name: str) -> None:
    """Raise ValueError if x is not finite."""
    _is_finite_float(x, name=name)


def validate_cycles(cycles: int) -> None:
    """
    Validate cycles per M1 revised spec.

    - Must be int >= 1
    - TypeError for non-int, ValueError for int < 1
    """
    if not isinstance(cycles, int):
        raise TypeError(f"cycles must be an int >= 1; got {type(cycles).__name__}")
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1; got {cycles!r}")


def build_potential_segment(
    E0: float,
    E1: float,
    dE: float,
    *,
    drop_first: bool = False,
    atol: float = _FLOAT_TOL,
) -> np.ndarray:
    """
    Build a deterministic piecewise-linear potential segment E0 -> E1 with nominal step |dE|.

    Rules (per spec):
      - Samples are spaced by dE in the correct direction.
      - Ensure the final value is exactly E1 (append if needed).
      - Avoid duplicate endpoint (do not append if already equal within tolerance).
      - Optionally drop the first sample to avoid duplicating join points.

    Parameters
    ----------
    E0, E1
        Segment start and end potentials [V].
    dE
        Positive potential increment magnitude [V]. Direction is inferred from E0/E1.
    drop_first
        If True, drop the first sample (useful when concatenating segments).
    atol
        Absolute tolerance used to decide whether endpoints are already equal.

    Returns
    -------
    np.ndarray
        1D array of E samples for the segment.
    """
    validate_finite_float(E0, name="E0")
    validate_finite_float(E1, name="E1")
    validate_positive_float(dE, name="dE")

    # Degenerate segment: still return a single point.
    if np.isclose(E0, E1, atol=atol, rtol=0.0):
        E = np.array([float(E1)], dtype=float)
        return E[1:] if drop_first else E

    sign = 1.0 if E1 >= E0 else -1.0
    step = sign * dE

    distance = abs(E1 - E0)
    n_full = int(np.floor(distance / dE))
    # Always include E0. This yields [E0, E0+step, ..., E0+n_full*step]
    E = E0 + step * np.arange(n_full + 1, dtype=float)

    # Enforce exact endpoint, without duplicate.
    if not np.isclose(E[-1], E1, atol=atol, rtol=0.0):
        E = np.append(E, float(E1))

    if drop_first:
        E = E[1:]

    return E
