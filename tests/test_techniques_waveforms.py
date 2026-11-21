"""Unit tests for softpotato.techniques waveforms and core potential-control techniques."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.techniques import (
    BaseTechnique,
    CallableSegment,
    Chronoamperometry,
    ConstantSegment,
    CustomWaveform,
    CyclicVoltammetry,
    LinearRampSegment,
    LinearSweepVoltammetry,
    SinusoidalSegment,
    WaveformSegment,
)

# ---------------------------------------------------------------------------
# Public exports & BaseTechnique interface
# ---------------------------------------------------------------------------


def test_top_level_exports() -> None:
    """All core technique classes are exported at top-level and under softpotato.techniques."""
    assert sp.techniques.BaseTechnique is BaseTechnique
    assert sp.BaseTechnique is BaseTechnique
    assert sp.Chronoamperometry is Chronoamperometry
    assert sp.LinearSweepVoltammetry is LinearSweepVoltammetry
    assert sp.CyclicVoltammetry is CyclicVoltammetry
    assert sp.CustomWaveform is CustomWaveform
    assert sp.WaveformSegment is WaveformSegment
    assert sp.ConstantSegment is ConstantSegment
    assert sp.LinearRampSegment is LinearRampSegment
    assert sp.SinusoidalSegment is SinusoidalSegment
    assert sp.CallableSegment is CallableSegment


def test_base_technique_mode_aliases() -> None:
    """potential() and current() respect the technique control mode."""
    ca = Chronoamperometry(E_step=-0.5, duration=2.0)
    assert ca.mode == "potentiostatic"
    assert ca.potential(1.0) == -0.5
    assert ca(1.0) == -0.5
    assert ca.excitation(1.0) == -0.5
    with pytest.raises(ValueError, match="galvanostatic"):
        ca.current(1.0)

    galv = CustomWaveform([ConstantSegment(1e-3, duration=2.0)], mode="galvanostatic")
    assert galv.mode == "galvanostatic"
    assert galv.current(1.0) == pytest.approx(1e-3)
    assert galv(1.0) == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="potentiostatic"):
        galv.potential(1.0)


# ---------------------------------------------------------------------------
# WaveformSegment primitives
# ---------------------------------------------------------------------------


class TestWaveformSegments:
    """Tests for ConstantSegment, LinearRampSegment, SinusoidalSegment, CallableSegment."""

    def test_constant_segment_scalar_and_array(self) -> None:
        seg = ConstantSegment(value=-0.4, duration=3.0)
        assert seg.duration == 3.0
        assert seg.value == -0.4
        assert seg.start_value == -0.4
        assert seg.end_value == -0.4
        assert seg.event_times == ()

        val = seg(1.5)
        assert isinstance(val, float)
        assert val == -0.4

        t_arr = np.array([0.0, 1.5, 3.0])
        out = seg.evaluate(t_arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [-0.4, -0.4, -0.4])

    def test_linear_ramp_segment_duration_and_rate(self) -> None:
        seg = LinearRampSegment(start=0.2, end=-0.6, duration=4.0)
        assert seg.duration == 4.0
        assert seg.slope == pytest.approx(-0.2)
        assert seg.rate == pytest.approx(0.2)
        assert seg(0.0) == 0.2
        assert seg(2.0) == pytest.approx(-0.2)
        assert seg(4.0) == -0.6

        # Construct via rate
        seg_rate = LinearRampSegment(start=0.2, end=-0.6, rate=0.2)
        assert seg_rate.duration == pytest.approx(4.0)
        assert seg_rate(0.0) == 0.2
        assert seg_rate(4.0) == -0.6

        # Array evaluation including exact endpoints
        t = np.array([0.0, 1.0, 2.0, 4.0])
        expected = np.array([0.2, 0.0, -0.2, -0.6])
        np.testing.assert_allclose(seg(t), expected)
        assert seg(t)[0] == 0.2
        assert seg(t)[-1] == -0.6

    def test_sinusoidal_segment(self) -> None:
        seg = SinusoidalSegment(
            offset=0.1, amplitude=0.05, frequency=2.0, duration=1.0, phase=0.0
        )
        assert seg(0.0) == pytest.approx(0.1)
        assert seg(0.125) == pytest.approx(0.15)  # quarter period of 2 Hz
        assert seg(0.375) == pytest.approx(0.05)  # three-quarter period
        t = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
        expected = np.array([0.1, 0.15, 0.1, 0.05, 0.1])
        np.testing.assert_allclose(seg(t), expected, atol=1e-12)

    def test_callable_segment_vectorized_and_scalar_fallback(self) -> None:
        # Vectorized NumPy callable
        seg_vec = CallableSegment(lambda t: 0.2 * np.cos(2.0 * np.pi * t), duration=1.0)
        assert seg_vec(0.0) == pytest.approx(0.2)
        assert seg_vec(0.5) == pytest.approx(-0.2)
        np.testing.assert_allclose(seg_vec(np.array([0.0, 0.5, 1.0])), [0.2, -0.2, 0.2])

        # Scalar-only math callable falls back gracefully on array inputs
        seg_scalar = CallableSegment(
            lambda t: 0.1 * math.sin(math.pi * t), duration=1.0, event_times=[0.5]
        )
        assert seg_scalar.event_times == (0.5,)
        assert seg_scalar(0.5) == pytest.approx(0.1)
        np.testing.assert_allclose(
            seg_scalar(np.array([0.0, 0.5, 1.0])), [0.0, 0.1, 0.0], atol=1e-15
        )

        # Constant-returning callable broadcasts cleanly
        seg_const = CallableSegment(lambda t: 0.75, duration=2.0)
        np.testing.assert_allclose(
            seg_const(np.array([0.0, 1.0, 2.0])), [0.75, 0.75, 0.75]
        )

    def test_segment_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ConstantSegment(0.0, duration=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            ConstantSegment(0.0, duration=-1.0)
        with pytest.raises(ValueError, match="finite"):
            ConstantSegment(float("nan"), duration=1.0)
        with pytest.raises(TypeError, match="real number"):
            ConstantSegment(True, duration=1.0)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="not both"):
            LinearRampSegment(0.0, 1.0, duration=1.0, rate=1.0)
        with pytest.raises(ValueError, match="Must specify"):
            LinearRampSegment(0.0, 1.0)
        with pytest.raises(ValueError, match="start == end"):
            LinearRampSegment(0.5, 0.5, rate=0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            LinearRampSegment(0.0, 1.0, rate=-0.1)

        with pytest.raises(ValueError, match="strictly positive"):
            SinusoidalSegment(0.0, 0.1, frequency=0.0, duration=1.0)
        with pytest.raises(TypeError, match="callable"):
            CallableSegment("not_callable", duration=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CustomWaveform
# ---------------------------------------------------------------------------


class TestCustomWaveform:
    """Tests for composite CustomWaveform chaining arbitrary segments."""

    def test_chained_segments_evaluation_and_events(self) -> None:
        wf = CustomWaveform(
            [
                ConstantSegment(0.0, duration=1.0),
                LinearRampSegment(0.0, -0.5, duration=2.0),
                CallableSegment(
                    lambda t: -0.5 + 0.1 * np.sin(2.0 * np.pi * t),
                    duration=1.0,
                    event_times=[0.5],
                ),
            ]
        )
        assert isinstance(wf, BaseTechnique)
        assert wf.mode == "potentiostatic"
        assert wf.duration == pytest.approx(4.0)
        assert wf.segment_boundaries == (0.0, 1.0, 3.0, 4.0)
        assert wf.event_times == (1.0, 3.0, 3.5)

        # Scalar evaluations
        assert wf(0.0) == 0.0
        assert wf(0.5) == 0.0
        assert wf(1.0) == 0.0
        assert wf(2.0) == pytest.approx(-0.25)
        assert wf(3.0) == pytest.approx(-0.5)
        assert wf(3.25) == pytest.approx(-0.4)
        assert wf(4.0) == pytest.approx(-0.5)

        # Out-of-bounds clamping
        assert wf(-0.5) == 0.0
        assert wf(5.0) == pytest.approx(-0.5)

        # Vectorized evaluation (1D and 2D)
        t_1d = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 3.25, 4.0])
        expected = np.array([0.0, 0.0, 0.0, -0.25, -0.5, -0.4, -0.5])
        np.testing.assert_allclose(wf(t_1d), expected, atol=1e-12)

        t_2d = np.array([[0.0, 2.0], [3.25, 4.0]])
        out_2d = wf(t_2d)
        assert out_2d.shape == (2, 2)
        np.testing.assert_allclose(out_2d, [[0.0, -0.25], [-0.4, -0.5]], atol=1e-12)

        # Empty array
        empty = wf(np.array([]))
        assert empty.shape == (0,)

    def test_discontinuous_step_boundary_is_right_continuous(self) -> None:
        wf = CustomWaveform(
            [ConstantSegment(0.0, duration=1.0), ConstantSegment(1.0, duration=1.0)]
        )
        assert wf(0.999999) == 0.0
        assert wf(1.0) == 1.0
        assert wf(2.0) == 1.0
        np.testing.assert_array_equal(
            wf(np.array([0.0, 0.5, 1.0, 2.0])), [0.0, 0.0, 1.0, 1.0]
        )

    def test_time_grid_includes_all_events(self) -> None:
        wf = CustomWaveform(
            [
                ConstantSegment(0.0, duration=1.0),
                LinearRampSegment(0.0, 1.0, duration=2.0),
                ConstantSegment(1.0, duration=1.0),
            ]
        )
        grid = wf.time_grid(n_points=10)
        assert grid[0] == 0.0
        assert grid[-1] == 4.0
        assert np.all(np.diff(grid) > 0.0)
        for ev in wf.event_times:
            assert ev in grid

        grid_dt = wf.time_grid(dt=0.7)
        assert grid_dt[0] == 0.0
        assert grid_dt[-1] == 4.0
        for ev in wf.event_times:
            assert ev in grid_dt

    def test_custom_waveform_validation(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CustomWaveform([])
        with pytest.raises(TypeError, match="WaveformSegment"):
            CustomWaveform(["bad"])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="mode"):
            CustomWaveform([ConstantSegment(0.0, 1.0)], mode="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chronoamperometry
# ---------------------------------------------------------------------------


class TestChronoamperometry:
    """Tests for single-, double-, and multi-step Chronoamperometry."""

    def test_single_step(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=10.0)
        assert ca.mode == "potentiostatic"
        assert ca.duration == 10.0
        assert ca.event_times == ()
        assert ca.potentials == (-0.5,)
        assert ca.durations == (10.0,)
        assert ca.n_steps == 1
        assert ca(0.0) == -0.5
        assert ca(5.0) == -0.5
        assert ca(10.0) == -0.5
        np.testing.assert_array_equal(ca(np.linspace(0.0, 10.0, 5)), np.full(5, -0.5))

    def test_single_step_with_quiet_time(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=5.0, E_quiet=0.0, t_quiet=2.0)
        assert ca.duration == 7.0
        assert ca.event_times == (2.0,)
        assert ca.potentials == (0.0, -0.5)
        assert ca.durations == (2.0, 5.0)
        assert ca.E_quiet == 0.0
        assert ca.t_quiet == 2.0
        assert ca(0.0) == 0.0
        assert ca(1.999) == 0.0
        assert ca(2.0) == -0.5
        assert ca(7.0) == -0.5

    def test_double_step(self) -> None:
        ca = Chronoamperometry(E_step1=-0.5, E_step2=0.5, t_step1=2.0, t_step2=3.0)
        assert ca.duration == 5.0
        assert ca.event_times == (2.0,)
        assert ca.step_times == (0.0, 2.0)
        assert ca(0.0) == -0.5
        assert ca(1.5) == -0.5
        assert ca(2.0) == 0.5
        assert ca(5.0) == 0.5
        t = np.array([0.0, 1.99, 2.0, 3.5, 5.0])
        np.testing.assert_array_equal(ca(t), [-0.5, -0.5, 0.5, 0.5, 0.5])

    def test_multi_step_and_time_grid(self) -> None:
        ca = Chronoamperometry(
            potentials=[0.1, -0.4, 0.3, -0.2],
            durations=[1.0, 2.0, 1.5, 0.5],
        )
        assert ca.duration == pytest.approx(5.0)
        assert ca.event_times == (1.0, 3.0, 4.5)
        assert ca.n_steps == 4

        t = np.array([0.0, 0.5, 1.0, 2.999, 3.0, 4.499, 4.5, 5.0])
        expected = np.array([0.1, 0.1, -0.4, -0.4, 0.3, 0.3, -0.2, -0.2])
        np.testing.assert_array_equal(ca(t), expected)

        grid = ca.time_grid(n_points=11)
        assert grid[0] == 0.0
        assert grid[-1] == 5.0
        for ev in ca.event_times:
            assert ev in grid

    def test_chronoamperometry_validation(self) -> None:
        with pytest.raises(ValueError, match="Must specify"):
            Chronoamperometry()
        with pytest.raises(ValueError, match="Cannot mix"):
            Chronoamperometry(
                E_step=-0.5,
                duration=1.0,
                E_step1=0.0,
                E_step2=1.0,
                t_step1=1.0,
                t_step2=1.0,
            )
        with pytest.raises(ValueError, match="both"):
            Chronoamperometry(E_step=-0.5)
        with pytest.raises(ValueError, match="Double-step"):
            Chronoamperometry(E_step1=-0.5, E_step2=0.5, t_step1=1.0)
        with pytest.raises(ValueError, match="same length"):
            Chronoamperometry(potentials=[0.0, 1.0], durations=[1.0])
        with pytest.raises(ValueError, match="empty"):
            Chronoamperometry(potentials=[], durations=[])
        with pytest.raises(ValueError, match="strictly positive"):
            Chronoamperometry(E_step=-0.5, duration=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            Chronoamperometry(potentials=[0.0, 1.0], durations=[1.0, -0.5])
        with pytest.raises(ValueError, match="E_quiet"):
            Chronoamperometry(E_step=-0.5, duration=1.0, t_quiet=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            Chronoamperometry(E_step=-0.5, duration=1.0, E_quiet=0.0, t_quiet=-0.1)


# ---------------------------------------------------------------------------
# LinearSweepVoltammetry
# ---------------------------------------------------------------------------


class TestLinearSweepVoltammetry:
    """Tests for LinearSweepVoltammetry (LSV)."""

    def test_cathodic_and_anodic_sweeps(self) -> None:
        lsv_cath = LinearSweepVoltammetry(E_start=0.5, E_end=-0.5, scan_rate=0.1)
        assert lsv_cath.direction == -1
        assert lsv_cath.duration == pytest.approx(10.0)
        assert lsv_cath.event_times == ()
        assert lsv_cath(0.0) == 0.5
        assert lsv_cath(5.0) == pytest.approx(0.0)
        assert lsv_cath(10.0) == -0.5

        t = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        np.testing.assert_allclose(lsv_cath(t), [0.5, 0.25, 0.0, -0.25, -0.5])
        assert lsv_cath(t)[0] == 0.5
        assert lsv_cath(t)[-1] == -0.5

        lsv_anod = LinearSweepVoltammetry(E_start=-0.3, E_end=0.7, scan_rate=0.2)
        assert lsv_anod.direction == 1
        assert lsv_anod.duration == pytest.approx(5.0)
        assert lsv_anod(0.0) == -0.3
        assert lsv_anod(2.5) == pytest.approx(0.2)
        assert lsv_anod(5.0) == 0.7

    def test_time_grid_resolution_options(self) -> None:
        lsv_de = LinearSweepVoltammetry(E_start=0.0, E_end=1.0, scan_rate=0.5, dE=0.1)
        grid = lsv_de.time_grid()
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(2.0)
        assert len(grid) == 11
        np.testing.assert_allclose(lsv_de(grid), np.linspace(0.0, 1.0, 11))

        lsv_n = LinearSweepVoltammetry(
            E_start=0.0, E_end=1.0, scan_rate=0.5, n_points=5
        )
        assert len(lsv_n.time_grid()) == 5

        # Override at call time
        assert len(lsv_n.time_grid(dE=0.25)) == 5
        assert len(lsv_n.time_grid(dt=0.5)) == 5

    def test_lsv_validation(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            LinearSweepVoltammetry(E_start=0.5, E_end=0.5, scan_rate=0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            LinearSweepVoltammetry(E_start=0.0, E_end=1.0, scan_rate=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            LinearSweepVoltammetry(E_start=0.0, E_end=1.0, scan_rate=-0.1)
        with pytest.raises(ValueError, match="at most one"):
            LinearSweepVoltammetry(
                E_start=0.0, E_end=1.0, scan_rate=0.1, dE=0.01, n_points=100
            )
        lsv = LinearSweepVoltammetry(E_start=0.0, E_end=1.0, scan_rate=0.1)
        with pytest.raises(ValueError, match="at most one"):
            lsv.time_grid(n_points=10, dt=0.1)
        with pytest.raises(ValueError, match="at most one"):
            lsv.time_grid(dt=0.1, dE=0.01)


# ---------------------------------------------------------------------------
# CyclicVoltammetry
# ---------------------------------------------------------------------------


class TestCyclicVoltammetry:
    """Tests for single- and multi-cycle CyclicVoltammetry (CV)."""

    def test_two_vertex_single_and_multi_cycle(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.5, E_vertex1=-0.5, E_vertex2=0.5, scan_rate=0.5, n_cycles=3
        )
        assert cv.duration == pytest.approx(12.0)
        assert cv.event_times == (2.0, 4.0, 6.0, 8.0, 10.0)
        assert cv.cycle_time_ranges == ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0))

        # Exact vertex values (scalar)
        assert cv(0.0) == 0.5
        assert cv(2.0) == -0.5
        assert cv(4.0) == 0.5
        assert cv(6.0) == -0.5
        assert cv(8.0) == 0.5
        assert cv(10.0) == -0.5
        assert cv(12.0) == 0.5

        # Vectorized evaluation matches scalar
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 12.0])
        expected = np.array([0.5, 0.0, -0.5, 0.0, 0.5, -0.5, -0.5, 0.5])
        np.testing.assert_allclose(cv(t), expected)

        # Continuity across every switching vertex
        eps = 1e-7
        for tv in cv.event_times:
            assert abs(cv(tv - eps) - cv(tv + eps)) < 1e-6

    def test_three_vertex_multi_cycle_and_custom_e_end(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.0,
            E_vertex1=-0.5,
            E_vertex2=0.5,
            scan_rate=1.0,
            n_cycles=2,
        )
        assert cv.E_end == 0.0
        assert cv.duration == pytest.approx(4.0)
        # Direction reversals occur at -0.5 and +0.5 in each cycle
        assert cv.event_times == (0.5, 1.5, 2.5, 3.5)
        assert cv.cycle_time_ranges == ((0.0, 2.0), (2.0, 4.0))

        assert cv(0.0) == 0.0
        assert cv(0.5) == -0.5
        assert cv(1.5) == 0.5
        assert cv(2.0) == 0.0
        assert cv(2.5) == -0.5
        assert cv(3.5) == 0.5
        assert cv(4.0) == 0.0

        # Custom E_end on final cycle
        cv_end = CyclicVoltammetry(
            E_start=0.0,
            E_vertex1=-0.5,
            E_vertex2=0.5,
            E_end=0.2,
            scan_rate=1.0,
            n_cycles=2,
        )
        assert cv_end.duration == pytest.approx(3.8)
        assert cv_end.cycle_time_ranges == ((0.0, 2.0), (2.0, 3.8))
        assert cv_end(3.8) == 0.2

    def test_time_grid_includes_vertices_and_cycle_bounds(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.0,
            E_vertex1=-0.5,
            E_vertex2=0.5,
            scan_rate=1.0,
            n_cycles=2,
            dE=0.13,
        )
        grid = cv.time_grid()
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(4.0)
        assert np.all(np.diff(grid) > 0.0)
        for tv in cv.event_times:
            assert tv in grid
        for t_start, t_end in cv.cycle_time_ranges:
            assert t_start in grid
            assert t_end in grid

    def test_cv_validation(self) -> None:
        with pytest.raises(ValueError, match="E_vertex1"):
            CyclicVoltammetry(0.0, 0.0, 0.5, scan_rate=0.1)
        with pytest.raises(ValueError, match="E_vertex2"):
            CyclicVoltammetry(0.0, -0.5, -0.5, scan_rate=0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            CyclicVoltammetry(0.0, -0.5, 0.5, scan_rate=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            CyclicVoltammetry(0.0, -0.5, 0.5, scan_rate=-0.1)
        with pytest.raises(ValueError, match=">= 1"):
            CyclicVoltammetry(0.0, -0.5, 0.5, scan_rate=0.1, n_cycles=0)
        with pytest.raises(TypeError, match="integer"):
            CyclicVoltammetry(0.0, -0.5, 0.5, scan_rate=0.1, n_cycles=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite"):
            CyclicVoltammetry(0.0, -0.5, 0.5, scan_rate=0.1)(float("nan"))
