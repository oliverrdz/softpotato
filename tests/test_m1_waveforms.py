from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import softpotato as sp


def _assert_canonical_waveform(w: np.ndarray) -> None:
    """
    AC: Waveform canonical form
      - ndarray shape (n,2) [E,t]
      - finite values
      - strictly increasing time
    """
    assert isinstance(w, np.ndarray)
    assert w.ndim == 2 and w.shape[1] == 2
    assert w.dtype == np.float64
    assert w.shape[0] >= 1
    assert np.all(np.isfinite(w))
    t = w[:, 1]
    assert np.all(np.diff(t) > 0.0), "time must be strictly increasing"


def test_waveform_from_arrays_happy_path() -> None:
    """
    AC: From-arrays constructor returns canonical (n,2) waveform.
    """
    E = np.array([0.0, 1.0, 2.0], dtype=float)
    t = np.array([0.0, 0.5, 1.0], dtype=float)

    w = sp.waveform_from_arrays(E, t)
    _assert_canonical_waveform(w)

    assert np.array_equal(w[:, 0], E.astype(np.float64))
    assert np.array_equal(w[:, 1], t.astype(np.float64))


@pytest.mark.parametrize(
    "E,t,match",
    [
        ([0.0, 1.0], [0.0], "same length"),
        ([[0.0, 1.0]], [0.0, 1.0], "E must be a 1D"),
        ([0.0, float("nan")], [0.0, 1.0], "finite"),
        ([0.0, 1.0], [0.0, 0.0], "strictly increasing"),
        ([0.0, 1.0], [0.0, float("inf")], "finite"),
    ],
)
def test_waveform_from_arrays_rejects_invalid_inputs(
    E: Any, t: Any, match: str
) -> None:
    """
    AC: From-arrays constructor rejects mismatched length, non-1D, non-finite, non-monotonic time.
    """
    with pytest.raises(ValueError, match=match):
        sp.waveform_from_arrays(E, t)


def test_validate_waveform_rejects_wrong_shape_and_nonfinite() -> None:
    """
    AC: Validation utilities enforce invariants.
    """
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        sp.validate_waveform([1.0, 2.0, 3.0])  # 1D

    with pytest.raises(ValueError, match="finite"):
        sp.validate_waveform(np.array([[0.0, 0.0], [np.nan, 0.1]]))

    with pytest.raises(ValueError, match="strictly increasing"):
        sp.validate_waveform(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_lsv_is_linear_and_matches_scan_rate_sign_and_magnitude() -> None:
    """
    AC: LSV
      - E(t) linear
      - implied scan rate matches requested magnitude
      - sign matches direction
    """
    E_start, E_end = 0.0, 1.0
    scan_rate = 0.5  # V/s
    dt = 0.1  # s

    w = sp.lsv(E_start, E_end, scan_rate=scan_rate, dt=dt)
    _assert_canonical_waveform(w)

    E = w[:, 0]
    t = w[:, 1]

    # Exact endpoints in potential space (implementation uses linspace)
    assert E[0] == pytest.approx(E_start, rel=0.0, abs=0.0)
    assert E[-1] == pytest.approx(E_end, rel=0.0, abs=0.0)

    # Linearity: E should equal affine function between endpoints
    slope = (E_end - E_start) / (t[-1] - t[0])
    E_fit = E_start + slope * (t - t[0])
    assert np.allclose(E, E_fit, rtol=0.0, atol=1e-12)

    # Implied scan rate from discrete slope should match magnitude
    implied = np.diff(E) / np.diff(t)
    assert np.allclose(implied, scan_rate, rtol=0.0, atol=1e-12)

    # Descending direction
    w2 = sp.lsv(1.0, -1.0, scan_rate=scan_rate, dt=dt)
    _assert_canonical_waveform(w2)
    implied2 = np.diff(w2[:, 0]) / np.diff(w2[:, 1])
    assert np.all(implied2 < 0.0)
    assert np.allclose(np.abs(implied2), scan_rate, rtol=0.0, atol=1e-12)


def test_cv_turning_points_and_cycle_repetition_with_scan_rate() -> None:
    """
    AC: CV
      - first segment moves E_start -> E_vertex
      - then reverses toward E_return (defaults to E_start)
      - for cycles>1, turning points repeat consistently
    """
    E_start = 0.0
    E_vertex = 1.0
    cycles = 2
    scan_rate = 1.0  # V/s
    dt = 0.25

    w = sp.cv(E_start, E_vertex, cycles=cycles, scan_rate=scan_rate, dt=dt)
    _assert_canonical_waveform(w)

    E = w[:, 0]
    t = w[:, 1]

    # For these parameters: each cycle duration = 2.0s, total = 4.0s
    # Turning points expected at t=1.0 (vertex), t=2.0 (return), t=3.0 (vertex), t=4.0 (return)
    def idx_at(time_value: float) -> int:
        hits = np.where(np.isclose(t, time_value, rtol=0.0, atol=1e-15))[0]
        assert (
            hits.size == 1
        ), f"expected exactly one sample at t={time_value}, got {hits}"
        return int(hits[0])

    i_v1 = idx_at(1.0)
    i_r1 = idx_at(2.0)
    i_v2 = idx_at(3.0)
    i_r2 = idx_at(4.0)

    assert E[0] == pytest.approx(E_start, rel=0.0, abs=0.0)
    assert E[i_v1] == pytest.approx(E_vertex, rel=0.0, abs=1e-12)
    assert E[i_r1] == pytest.approx(E_start, rel=0.0, abs=1e-12)
    assert E[i_v2] == pytest.approx(E_vertex, rel=0.0, abs=1e-12)
    assert E[i_r2] == pytest.approx(E_start, rel=0.0, abs=1e-12)

    # Monotonic segments around the first vertex (strictly increasing then decreasing)
    assert np.all(np.diff(E[: i_v1 + 1]) > 0.0)
    assert np.all(np.diff(E[i_v1 : i_r1 + 1]) < 0.0)


def test_cv_rejects_invalid_cycles() -> None:
    """
    AC: CV supports cycles>=1; invalid inputs raise ValueError.
    """
    with pytest.raises(ValueError, match="cycles"):
        sp.cv(0.0, 1.0, cycles=0, dt=0.1, t_end=1.0)


def test_step_behavior_and_range_check() -> None:
    """
    AC: Potential step
      - E_before for t < t_step
      - E_after for t >= t_step
      - t_step must be within grid range
    """
    dt = 0.5
    tg = sp.uniform_time_grid(dt, n=5)  # t = [0,0.5,1.0,1.5,2.0]
    t_step = 1.0
    w = sp.step(0.0, 1.0, t_step, time_grid=tg)
    _assert_canonical_waveform(w)

    t = w[:, 1]
    E = w[:, 0]

    before = t < t_step
    after = t >= t_step

    assert np.all(E[before] == 0.0)
    assert np.all(E[after] == 1.0)

    # Out of range must fail
    with pytest.raises(ValueError, match="within"):
        sp.step(0.0, 1.0, -0.1, time_grid=tg)
    with pytest.raises(ValueError, match="within"):
        sp.step(0.0, 1.0, 999.0, time_grid=tg)


def test_docs_waveforms_page_exists_and_mentions_public_api() -> None:
    """
    AC: Docs + packaging quality gates
      - docs mention canonical format and M1 public functions
    """
    repo_root = Path(__file__).resolve().parents[1]
    docs_page = repo_root / "docs" / "waveforms.md"
    assert docs_page.exists(), "docs/waveforms.md must exist (M1 docs gate)."

    text = docs_page.read_text(encoding="utf-8").lower()

    # Canonical representation
    assert "(n, 2)" in text or "n,2" in text
    assert "[e, t]" in text or "columns" in text

    # Public API names should be mentioned
    for name in ("uniform_time_grid", "lsv", "cv", "step", "waveform_from_arrays"):
        assert name in text, f"docs/waveforms.md must mention `{name}`"
