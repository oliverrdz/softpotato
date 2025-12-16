from __future__ import annotations

import re

import numpy as np
import pytest

import softpotato as sp


def _is_strictly_increasing(x: np.ndarray) -> bool:
    return bool(np.all(np.diff(x) > 0.0))


def test_uniform_time_grid_with_n_has_expected_invariants() -> None:
    """
    AC: Time grids
      - t[0] == 0.0 (default t_start)
      - t strictly increasing
      - constant spacing within tolerance
      - dtype float
    """
    dt = 0.01
    n = 5
    tg = sp.uniform_time_grid(dt, n=n)
    t = tg.as_array()

    assert isinstance(t, np.ndarray)
    assert t.dtype == np.float64
    assert t.shape == (n,)
    assert t[0] == 0.0
    assert _is_strictly_increasing(t)

    # Uniform spacing (tolerant to floating rounding, but should be very tight)
    diffs = np.diff(t)
    assert np.allclose(diffs, dt, rtol=0.0, atol=10 * np.finfo(np.float64).eps)

    # TimeGrid metadata (lightweight contract)
    assert tg.n == n
    assert tg.t_start == 0.0
    assert tg.t_end == pytest.approx(dt * (n - 1), rel=0.0, abs=1e-15)
    assert tg.dt == pytest.approx(dt, rel=0.0, abs=1e-15)


def test_uniform_time_grid_with_t_end_is_inclusive_and_exact_steps() -> None:
    """
    AC: Time grids
      - constructor with t_end works
      - enforces integer number of steps for (t_end - t_start)/dt
    """
    dt = 0.1
    t_end = 1.0
    tg = sp.uniform_time_grid(dt, t_end=t_end)
    t = tg.as_array()

    assert t[0] == 0.0
    assert t[-1] == pytest.approx(t_end, rel=0.0, abs=1e-15)
    assert _is_strictly_increasing(t)
    assert np.allclose(np.diff(t), dt, rtol=0.0, atol=1e-12)

    # n should be (t_end/dt) + 1 for inclusive endpoint
    expected_n = int(round(t_end / dt)) + 1
    assert tg.n == expected_n


@pytest.mark.parametrize(
    "kwargs, msg_substr",
    [
        ({"dt": 0.0, "n": 5}, "dt must be"),
        ({"dt": -1.0, "n": 5}, "dt must be"),
        ({"dt": float("nan"), "n": 5}, "dt must be"),
        ({"dt": 0.1}, "exactly one of n or t_end"),
        ({"dt": 0.1, "n": 5, "t_end": 1.0}, "exactly one of n or t_end"),
        ({"dt": 0.1, "n": 0}, "n must be"),
        ({"dt": 0.1, "n": 1.5}, "n must be"),
        ({"dt": 0.1, "t_end": float("nan")}, "t_end must be"),
        ({"dt": 0.1, "t_end": -0.1}, "t_end must be"),
    ],
)
def test_uniform_time_grid_rejects_invalid_inputs(
    kwargs: dict, msg_substr: str
) -> None:
    """
    AC: Invalid inputs raise ValueError with clear message.
    """
    with pytest.raises(ValueError, match=re.escape(msg_substr)):
        sp.uniform_time_grid(**kwargs)  # type: ignore[arg-type]


def test_uniform_time_grid_rejects_non_integer_steps() -> None:
    """
    AC: Invalid inputs raise ValueError
      - t_end must satisfy integer number of steps
    """
    dt = 0.3
    t_end = 1.0  # 1.0/0.3 is not an integer
    with pytest.raises(ValueError, match="integer"):
        sp.uniform_time_grid(dt, t_end=t_end)


def test_timegrid_rejects_non_increasing_time() -> None:
    """
    AC: Time must be strictly increasing.
    """
    t = np.array([0.0, 0.1, 0.1, 0.2], dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        sp.TimeGrid(t=t)


def test_validate_time_supports_strict_and_non_strict_modes() -> None:
    """
    AC: Validation utilities enforce monotonic time.
    """
    t_nd = np.array([0.0, 0.1, 0.1, 0.2], dtype=float)

    with pytest.raises(ValueError, match="strictly increasing"):
        sp.validate_time(t_nd, strictly_increasing=True)

    out = sp.validate_time(t_nd, strictly_increasing=False)
    assert out.dtype == np.float64
    assert out.shape == t_nd.shape
    assert np.all(np.diff(out) >= 0.0)


def test_public_api_smoke_import_and_symbols() -> None:
    """
    AC: Docs + packaging quality gates
      - import works
      - expected M1 symbols exposed
    """
    assert hasattr(sp, "__version__")
    for name in (
        "TimeGrid",
        "uniform_time_grid",
        "lsv",
        "cv",
        "step",
        "waveform_from_arrays",
        "validate_time",
        "validate_waveform",
    ):
        assert hasattr(sp, name), f"softpotato must expose {name}"
