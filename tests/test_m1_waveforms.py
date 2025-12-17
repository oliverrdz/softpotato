from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def _assert_strictly_increasing(x: np.ndarray) -> None:
    dx = np.diff(x)
    assert np.all(dx > 0.0), f"Expected strictly increasing; min diff={dx.min()!r}"


def _assert_monotone_nonincreasing(x: np.ndarray) -> None:
    dx = np.diff(x)
    assert np.all(dx <= 0.0), f"Expected monotone nonincreasing; max diff={dx.max()!r}"


def _assert_monotone_nondecreasing(x: np.ndarray) -> None:
    dx = np.diff(x)
    assert np.all(dx >= 0.0), f"Expected monotone nondecreasing; min diff={dx.min()!r}"


def _assert_raises_one_of(
    exc_types: tuple[type[BaseException], ...],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    except Exception as e:  # pragma: no cover
        pytest.fail(f"Raised unexpected exception type: {type(e).__name__}: {e}")
    pytest.fail(f"Did not raise one of {exc_types}")


# ----------------------------
# LSV
# ----------------------------


def test_lsv_small_exact_increasing_example() -> None:
    """
    AC: lsv(0.0, 1.0, dE=0.25, scan_rate=0.5)
      - E = [0.0, 0.25, 0.5, 0.75, 1.0]
      - dt_step = 0.25 / 0.5 = 0.5
      - t = [0.0, 0.5, 1.0, 1.5, 2.0]
      - shape (5,2), time strictly increasing
    """
    import softpotato as sp

    w = sp.lsv(0.0, 1.0, 0.25, 0.5)
    assert isinstance(w, np.ndarray)
    assert w.shape == (5, 2)

    E = w[:, 0]
    t = w[:, 1]

    np.testing.assert_allclose(
        E, np.array([0.0, 0.25, 0.5, 0.75, 1.0]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        t, np.array([0.0, 0.5, 1.0, 1.5, 2.0]), rtol=0.0, atol=1e-12
    )
    _assert_strictly_increasing(t)
    _assert_monotone_nondecreasing(E)


def test_lsv_decreasing_sweep_has_decreasing_E_and_increasing_t() -> None:
    """
    AC: decreasing sweep works (E decreases, t strictly increases).
    """
    import softpotato as sp

    w = sp.lsv(1.0, 0.0, 0.25, 0.5)

    E = w[:, 0]
    t = w[:, 1]

    _assert_strictly_increasing(t)
    _assert_monotone_nonincreasing(E)
    assert math.isclose(float(E[0]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(E[-1]), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_lsv_endpoint_is_exact_even_when_not_multiple_of_dE() -> None:
    """
    Spec: always ensure final E is exactly E_end (append if needed, no duplicates).
    Pick a case where (E_end-E_start)/dE is not an integer.
    """
    import softpotato as sp

    w = sp.lsv(0.0, 1.0, 0.3, 1.0)
    E = w[:, 0]
    t = w[:, 1]

    _assert_strictly_increasing(t)
    _assert_monotone_nondecreasing(E)
    assert math.isclose(float(E[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(E[-1]), 1.0, rel_tol=0.0, abs_tol=1e-12)

    # No duplicate endpoint:
    if E.size >= 2:
        assert not math.isclose(float(E[-2]), float(E[-1]), rel_tol=0.0, abs_tol=1e-14)


# ----------------------------
# CV
# ----------------------------


def test_cv_small_exact_cycle_end_default_is_start_no_duplicate_vertex() -> None:
    """
    AC: cv(E_start=0.0, E_vertex=1.0, scan_rate=0.5, dE=0.5, E_end=None, cycles=1)
      - E_end_cycle = E_start = 0.0
      - path [0.0, 0.5, 1.0, 0.5, 0.0]
      - dt_step = 0.5/0.5 = 1.0 so t=[0,1,2,3,4]
    """
    import softpotato as sp

    w = sp.cv(0.0, 1.0, 0.5, 0.5, E_end=None, cycles=1)
    assert w.shape == (5, 2)

    E = w[:, 0]
    t = w[:, 1]

    np.testing.assert_allclose(
        E, np.array([0.0, 0.5, 1.0, 0.5, 0.0]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        t, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), rtol=0.0, atol=1e-12
    )
    _assert_strictly_increasing(t)

    # No duplicated vertex at the join between segments:
    # The vertex should appear exactly once in the whole waveform for this symmetric case.
    assert int(np.sum(np.isclose(E, 1.0, rtol=0.0, atol=1e-12))) == 1


def test_cv_cycles_two_no_duplicate_cycle_boundary_and_time_continuous() -> None:
    """
    AC: cycles=2 repeats without duplicating the boundary sample between cycles.
    For the small example, expected E length is 9:
      [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
    """
    import softpotato as sp

    w = sp.cv(0.0, 1.0, 0.5, 0.5, E_end=None, cycles=2)
    E = w[:, 0]
    t = w[:, 1]

    expected_E = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0])
    expected_t = np.arange(expected_E.size, dtype=float)  # dt_step=1

    assert w.shape == (expected_E.size, 2)
    np.testing.assert_allclose(E, expected_E, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(t, expected_t, rtol=0.0, atol=1e-12)
    _assert_strictly_increasing(t)

    # Boundary de-dup: there must not be two consecutive 0.0s at the cycle boundary.
    # The boundary occurs after the first cycle ends at 0.0; next point should be 0.5.
    boundary_idx = 4
    assert math.isclose(float(E[boundary_idx]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(E[boundary_idx + 1]), 0.5, rel_tol=0.0, abs_tol=1e-12)


# ----------------------------
# STEP
# ----------------------------


def test_step_uniform_time_grid_divisible_and_instantaneous_step_at_t0() -> None:
    """
    AC: step(E_before=0.0, E_after=1.0, dt=0.01, t_end=0.04)
      - t=[0,0.01,0.02,0.03,0.04]
      - E[0]==1.0 (instantaneous step at t=0)
    """
    import softpotato as sp

    w = sp.step(0.0, 1.0, 0.01, 0.04)
    assert w.shape == (5, 2)

    E = w[:, 0]
    t = w[:, 1]

    np.testing.assert_allclose(
        t, np.array([0.0, 0.01, 0.02, 0.03, 0.04]), rtol=0.0, atol=1e-15
    )
    _assert_strictly_increasing(t)
    np.testing.assert_allclose(np.diff(t), np.full(4, 0.01), rtol=0.0, atol=1e-15)

    assert math.isclose(float(E[0]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    np.testing.assert_allclose(E, np.full_like(t, 1.0), rtol=0.0, atol=1e-12)


def test_step_non_divisible_t_end_does_not_snap_and_last_time_is_within_one_dt() -> (
    None
):
    """
    Spec recommendation: do not snap final sample to t_end; keep strict uniform dt.
    Guarantee: t[-1] <= t_end and t[-1] > t_end - dt.
    """
    import softpotato as sp

    dt = 0.03
    t_end = 0.10
    w = sp.step(0.0, 1.0, dt, t_end)
    t = w[:, 1]

    _assert_strictly_increasing(t)
    np.testing.assert_allclose(
        np.diff(t), np.full(t.size - 1, dt), rtol=0.0, atol=1e-15
    )

    assert float(t[0]) == 0.0
    assert float(t[-1]) <= t_end + 1e-15
    assert float(t[-1]) > (t_end - dt) - 1e-15


# ----------------------------
# Validation / errors
# ----------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("lsv", (0.0, 1.0, 0.0, 1.0)),  # dE <= 0
        ("lsv", (0.0, 1.0, -0.1, 1.0)),
        ("lsv", (0.0, 1.0, 0.1, 0.0)),  # scan_rate <= 0
        ("lsv", (0.0, 1.0, 0.1, -1.0)),
        ("cv", (0.0, 1.0, 0.0, 0.1)),  # scan_rate <= 0
        ("cv", (0.0, 1.0, -1.0, 0.1)),
        ("cv", (0.0, 1.0, 1.0, 0.0)),  # dE <= 0
        ("step", (0.0, 1.0, 0.0, 0.1)),  # dt <= 0
        ("step", (0.0, 1.0, -0.1, 0.1)),
        ("step", (0.0, 1.0, 0.01, 0.0)),  # t_end <= 0
        ("step", (0.0, 1.0, 0.01, -1.0)),
    ],
)
def test_invalid_positive_constraints_raise_value_error(
    args: tuple[str, tuple[float, float, float, float]],
) -> None:
    import softpotato as sp

    fn_name, call_args = args
    fn = getattr(sp, fn_name)
    with pytest.raises(ValueError):
        fn(*call_args)


@pytest.mark.parametrize(
    "fn_name, call_args",
    [
        ("lsv", (0.0, 1.0, np.nan, 1.0)),
        ("lsv", (0.0, 1.0, 0.1, np.inf)),
        ("cv", (0.0, 1.0, 1.0, np.nan)),
        ("cv", (0.0, 1.0, np.inf, 0.1)),
        ("step", (0.0, 1.0, np.nan, 0.1)),
        ("step", (0.0, 1.0, 0.01, np.inf)),
    ],
)
def test_nonfinite_inputs_raise_value_error(
    fn_name: str,
    call_args: tuple[float, float, float, float],
) -> None:
    import softpotato as sp

    fn = getattr(sp, fn_name)
    with pytest.raises(ValueError):
        fn(*call_args)


def test_cv_cycles_must_be_int_and_at_least_one() -> None:
    import softpotato as sp

    with pytest.raises(ValueError):
        sp.cv(0.0, 1.0, 1.0, 0.1, cycles=0)

    with pytest.raises(ValueError):
        sp.cv(0.0, 1.0, 1.0, 0.1, cycles=-1)

    # Spec allows either ValueError or TypeError for non-int cycles; accept either.
    _assert_raises_one_of(
        (TypeError, ValueError), sp.cv, 0.0, 1.0, 1.0, 0.1, cycles=1.5
    )


# ----------------------------
# Property-style invariants (cheap regression checks)
# ----------------------------


def test_waveforms_return_float64_two_columns_and_time_strictly_increasing() -> None:
    """
    QA: simple invariant-style checks on a few representative parameter sets.
    """
    import softpotato as sp

    waves = [
        sp.lsv(0.0, 0.8, 0.2, 0.4),
        sp.lsv(0.8, -0.1, 0.3, 1.2),
        sp.cv(0.0, 1.0, 0.5, 0.25, E_end=None, cycles=1),
        sp.cv(0.2, -0.6, 0.8, 0.1, E_end=0.2, cycles=2),
        sp.step(0.0, 1.0, 0.02, 0.11),
    ]
    for w in waves:
        assert isinstance(w, np.ndarray)
        assert w.ndim == 2 and w.shape[1] == 2
        assert w.dtype == np.float64
        _assert_strictly_increasing(w[:, 1])
        assert np.all(np.isfinite(w)), "Waveforms must be finite"
