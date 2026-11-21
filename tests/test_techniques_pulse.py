"""Unit tests for pulse, sampled-current, and galvanostatic techniques (SCV, DPV, SWV, CP)."""

from __future__ import annotations

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.step import sand, sand_potential, sand_transition_time
from softpotato.constants import FARADAY
from softpotato.reactions import ButlerVolmer
from softpotato.techniques import (
    BaseTechnique,
    Chronoamperometry,
    Chronopotentiometry,
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
)

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_pulse_and_galvanostatic_exports() -> None:
    """SCV, DPV, SWV, and CP are exported from softpotato.techniques and top-level softpotato."""
    assert sp.techniques.SampledCurrentVoltammetry is SampledCurrentVoltammetry
    assert sp.SampledCurrentVoltammetry is SampledCurrentVoltammetry
    assert sp.techniques.DifferentialPulseVoltammetry is DifferentialPulseVoltammetry
    assert sp.DifferentialPulseVoltammetry is DifferentialPulseVoltammetry
    assert sp.techniques.SquareWaveVoltammetry is SquareWaveVoltammetry
    assert sp.SquareWaveVoltammetry is SquareWaveVoltammetry
    assert sp.techniques.Chronopotentiometry is Chronopotentiometry
    assert sp.Chronopotentiometry is Chronopotentiometry


# ---------------------------------------------------------------------------
# SampledCurrentVoltammetry (SCV)
# ---------------------------------------------------------------------------


class TestSampledCurrentVoltammetry:
    """Tests for SampledCurrentVoltammetry (SCV)."""

    def test_construct_from_list(self) -> None:
        scv = SampledCurrentVoltammetry(
            potentials=[-0.4, -0.2, 0.0, 0.2],
            duration=5.0,
        )
        assert isinstance(scv, BaseTechnique)
        assert scv.mode == "potentiostatic"
        assert scv.n_steps == 4
        assert len(scv) == 4
        assert scv.potentials == (-0.4, -0.2, 0.0, 0.2)
        assert scv.duration == 5.0
        assert scv.step_duration == 5.0
        assert scv.event_times == ()
        assert scv.E_start == -0.4
        assert scv.E_end == 0.2
        assert scv.dE is None

        # All constituent steps are Chronoamperometry with identical duration and time_grid
        shared_grid = scv.time_grid(n_points=51)
        assert shared_grid[0] == 0.0
        assert shared_grid[-1] == 5.0
        assert len(shared_grid) == 51
        for step, e_expected in zip(scv, scv.potentials):
            assert isinstance(step, Chronoamperometry)
            assert step.duration == scv.duration
            assert step(2.5) == e_expected
            np.testing.assert_array_equal(step.time_grid(n_points=51), shared_grid)

    def test_construct_from_numpy_array(self) -> None:
        pots = np.linspace(0.5, -0.5, 11)
        scv = SampledCurrentVoltammetry(pots, 3.0)
        assert scv.n_steps == 11
        assert scv.duration == 3.0
        np.testing.assert_allclose(scv.potentials, pots)
        assert scv[0](1.5) == pytest.approx(0.5)
        assert scv[-1](1.5) == pytest.approx(-0.5)
        assert len(scv[2:5]) == 3

    def test_construct_from_window_and_step(self) -> None:
        scv_cath = SampledCurrentVoltammetry(
            E_start=0.2, E_end=-0.2, dE=0.1, duration=4.0
        )
        assert scv_cath.n_steps == 5
        assert scv_cath.dE == 0.1
        assert scv_cath.duration == 4.0
        np.testing.assert_allclose(scv_cath.potentials, [0.2, 0.1, 0.0, -0.1, -0.2])

        # Also accepts step_height as alias for dE
        scv_anod = SampledCurrentVoltammetry(
            E_start=-0.3, E_end=0.3, step_height=0.2, duration=2.0
        )
        assert scv_anod.n_steps == 4
        assert scv_anod.dE == 0.2
        np.testing.assert_allclose(scv_anod.potentials, [-0.3, -0.1, 0.1, 0.3])

    def test_quiet_time_and_vectorized_excitation(self) -> None:
        scv = SampledCurrentVoltammetry(
            potentials=[-0.5, 0.0, 0.5],
            duration=4.0,
            E_quiet=0.1,
            t_quiet=1.0,
            sample_times=[2.0, 5.0],
        )
        assert scv.duration == 5.0
        assert scv.step_duration == 4.0
        assert scv.event_times == (1.0,)
        assert scv.sample_times == (2.0, 5.0)

        # Scalar evaluation returns 1D array of shape (n_steps,)
        np.testing.assert_array_equal(scv(0.5), [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(scv(1.0), [-0.5, 0.0, 0.5])
        np.testing.assert_array_equal(scv.potential(3.0), [-0.5, 0.0, 0.5])

        # Array evaluation returns 2D array of shape (n_steps, n_times)
        t = np.array([0.0, 0.5, 1.0, 3.0, 5.0])
        out = scv(t)
        assert out.shape == (3, 5)
        np.testing.assert_array_equal(out[:, 0], [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(out[:, 1], [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(out[:, 2], [-0.5, 0.0, 0.5])
        np.testing.assert_array_equal(out[:, 4], [-0.5, 0.0, 0.5])

        # Shared time_grid includes event_times and sample_times
        grid = scv.time_grid(n_points=11)
        assert grid[0] == 0.0
        assert grid[-1] == 5.0
        assert 1.0 in grid
        assert 2.0 in grid

        # Empty array
        assert scv(np.array([])).shape == (3, 0)

    def test_sample_currents_single_and_multiple_times(self) -> None:
        scv = SampledCurrentVoltammetry(E_start=0.2, E_end=-0.2, dE=0.1, duration=4.0)
        t = np.linspace(0.01, 4.0, 400)
        # Synthetic Cottrell-like transients: I_k(t) = -(1 / (1 + exp(10 * E_k))) / sqrt(t)
        pots_arr = np.asarray(scv.potentials)
        theta = 1.0 / (1.0 + np.exp(10.0 * pots_arr))
        currents = -theta[:, None] / np.sqrt(t[None, :])

        # Single scalar sampling time
        E_out, i_tau = scv.sample_currents(t, currents, sample_times=1.0)
        np.testing.assert_allclose(E_out, pots_arr)
        assert i_tau.shape == (scv.n_steps,)
        np.testing.assert_allclose(i_tau, -theta / np.sqrt(1.0), rtol=1e-4)

        # Multiple sampling times
        taus = [0.5, 1.0, 4.0]
        E_out2, i_multi = scv.sample_currents(t, currents, sample_times=taus)
        np.testing.assert_allclose(E_out2, pots_arr)
        assert i_multi.shape == (3, scv.n_steps)
        for idx, tau in enumerate(taus):
            np.testing.assert_allclose(i_multi[idx], -theta / np.sqrt(tau), rtol=1e-4)

        # Default sampling time is duration
        _, i_end = scv.sample_currents(t, currents)
        np.testing.assert_allclose(i_end, -theta / np.sqrt(4.0), rtol=1e-4)

    def test_scv_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            SampledCurrentVoltammetry(potentials=[0.0, 0.1])
        with pytest.raises(ValueError, match="strictly positive"):
            SampledCurrentVoltammetry(potentials=[0.0, 0.1], duration=0.0)
        with pytest.raises(ValueError, match="Must specify"):
            SampledCurrentVoltammetry(duration=1.0)
        with pytest.raises(ValueError, match="Cannot mix"):
            SampledCurrentVoltammetry(
                potentials=[0.0, 0.1],
                duration=1.0,
                E_start=0.0,
                E_end=0.1,
                dE=0.05,
            )
        with pytest.raises(ValueError, match="at most one"):
            SampledCurrentVoltammetry(
                E_start=0.0, E_end=0.1, dE=0.05, step_height=0.05, duration=1.0
            )
        with pytest.raises(ValueError, match="empty"):
            SampledCurrentVoltammetry(potentials=[], duration=1.0)
        with pytest.raises(ValueError, match="1D"):
            SampledCurrentVoltammetry(potentials=np.zeros((2, 2)), duration=1.0)
        with pytest.raises(ValueError, match="distinct"):
            SampledCurrentVoltammetry(E_start=0.2, E_end=0.2, dE=0.05, duration=1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SampledCurrentVoltammetry(E_start=0.2, E_end=-0.2, dE=-0.05, duration=1.0)
        with pytest.raises(ValueError, match="E_quiet"):
            SampledCurrentVoltammetry(potentials=[0.0], duration=1.0, t_quiet=0.5)
        with pytest.raises(ValueError, match="<= duration"):
            SampledCurrentVoltammetry(
                potentials=[0.0, 0.1], duration=1.0, sample_times=2.0
            )


# ---------------------------------------------------------------------------
# DifferentialPulseVoltammetry (DPV)
# ---------------------------------------------------------------------------


class TestDifferentialPulseVoltammetry:
    """Tests for DifferentialPulseVoltammetry (DPV)."""

    def test_cathodic_waveform_geometry_and_events(self) -> None:
        dpv = DifferentialPulseVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.1,
            pulse_amplitude=-0.05,
            pulse_width=0.1,
            pulse_period=0.5,
        )
        assert isinstance(dpv, BaseTechnique)
        assert dpv.mode == "potentiostatic"
        assert dpv.direction == -1
        assert dpv.n_steps == 5
        assert dpv.base_width == pytest.approx(0.4)
        assert dpv.duration == pytest.approx(2.5)
        np.testing.assert_allclose(dpv.base_potentials, [0.2, 0.1, 0.0, -0.1, -0.2])
        np.testing.assert_allclose(
            dpv.pulse_potentials, [0.15, 0.05, -0.05, -0.15, -0.25]
        )

        # 2 * n_steps - 1 = 9 interior transitions: rise at 0.4, 0.9, ..., 2.4; fall at 0.5, 1.0, 1.5, 2.0
        expected_events = (0.4, 0.5, 0.9, 1.0, 1.4, 1.5, 1.9, 2.0, 2.4)
        np.testing.assert_allclose(dpv.event_times, expected_events)

        # Scalar right-continuous evaluations
        assert dpv(0.0) == pytest.approx(0.2)
        assert dpv(0.3999) == pytest.approx(0.2)
        assert dpv(0.4) == pytest.approx(0.15)
        assert dpv(0.4999) == pytest.approx(0.15)
        assert dpv(0.5) == pytest.approx(0.1)
        assert dpv(2.4) == pytest.approx(-0.25)
        assert dpv(2.5) == pytest.approx(-0.25)

        # Vectorized evaluation matches scalar
        t_arr = np.array([0.0, 0.2, 0.4, 0.5, 0.9, 2.4, 2.5])
        expected_e = np.array([0.2, 0.2, 0.15, 0.1, 0.05, -0.25, -0.25])
        np.testing.assert_allclose(dpv(t_arr), expected_e)

        # time_grid includes all pulse rise/fall event_times
        grid = dpv.time_grid(n_points=51)
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(2.5)
        for ev in dpv.event_times:
            assert np.any(np.isclose(grid, ev, atol=1e-12))

    def test_sample_currents_differential(self) -> None:
        dpv = DifferentialPulseVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.1,
            pulse_amplitude=-0.05,
            pulse_width=0.1,
            pulse_period=0.5,
            sample_fraction=0.2,
        )
        t = dpv.time_grid(n_points=501)
        # Synthetic current: I(t) = 2 * E(t)
        E_t = dpv(t)
        i_t = 2.0 * E_t

        E_base, i_pre, i_pulse, i_diff = dpv.sample_currents(t, i_t)
        np.testing.assert_allclose(E_base, dpv.base_potentials)
        np.testing.assert_allclose(i_pre, 2.0 * np.asarray(dpv.base_potentials))
        np.testing.assert_allclose(i_pulse, 2.0 * np.asarray(dpv.pulse_potentials))
        np.testing.assert_allclose(i_diff, np.full(dpv.n_steps, 2.0 * (-0.05)))

    def test_dpv_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            DifferentialPulseVoltammetry(0.0, 0.0, 0.05, -0.025, 0.05, 0.5)
        with pytest.raises(ValueError, match="strictly positive"):
            DifferentialPulseVoltammetry(0.2, -0.2, 0.0, -0.025, 0.05, 0.5)
        with pytest.raises(ValueError, match="non-zero"):
            DifferentialPulseVoltammetry(0.2, -0.2, 0.05, 0.0, 0.05, 0.5)
        with pytest.raises(ValueError, match="strictly greater"):
            DifferentialPulseVoltammetry(0.2, -0.2, 0.05, -0.025, 0.5, 0.5)
        with pytest.raises(ValueError, match="strictly greater"):
            DifferentialPulseVoltammetry(0.2, -0.2, 0.05, -0.025, 0.6, 0.5)
        with pytest.raises(ValueError, match="sample_fraction"):
            DifferentialPulseVoltammetry(
                0.2, -0.2, 0.05, -0.025, 0.05, 0.5, sample_fraction=1.5
            )


# ---------------------------------------------------------------------------
# SquareWaveVoltammetry (SWV)
# ---------------------------------------------------------------------------


class TestSquareWaveVoltammetry:
    """Tests for SquareWaveVoltammetry (SWV)."""

    def test_cathodic_and_anodic_osteryoung_waveforms(self) -> None:
        # Cathodic scan (direction = -1): forward is -amplitude, reverse is +amplitude
        swv_cath = SquareWaveVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.1,
            amplitude=0.025,
            frequency=10.0,
        )
        assert swv_cath.mode == "potentiostatic"
        assert swv_cath.direction == -1
        assert swv_cath.n_steps == 5
        assert swv_cath.period == pytest.approx(0.1)
        assert swv_cath.half_period == pytest.approx(0.05)
        assert swv_cath.duration == pytest.approx(0.5)
        assert len(swv_cath.event_times) == 2 * 5 - 1
        np.testing.assert_allclose(
            swv_cath.event_times, np.arange(1, 10) * 0.05, atol=1e-12
        )

        # First cycle: E_step=0.2 -> forward=0.175, reverse=0.225
        assert swv_cath(0.0) == pytest.approx(0.175)
        assert swv_cath(0.025) == pytest.approx(0.175)
        assert swv_cath(0.05) == pytest.approx(0.225)
        assert swv_cath(0.099) == pytest.approx(0.225)
        # Second cycle: E_step=0.1 -> forward=0.075, reverse=0.125
        assert swv_cath(0.1) == pytest.approx(0.075)
        assert swv_cath(0.15) == pytest.approx(0.125)
        # Final endpoint
        assert swv_cath(0.5) == pytest.approx(-0.175)

        # Anodic scan (direction = +1): forward is +amplitude, reverse is -amplitude
        swv_anod = SquareWaveVoltammetry(
            E_start=-0.2,
            E_end=0.2,
            step_height=0.2,
            amplitude=0.05,
            frequency=5.0,
        )
        assert swv_anod.direction == 1
        assert swv_anod.n_steps == 3
        assert swv_anod(0.0) == pytest.approx(-0.15)  # -0.2 + 0.05
        assert swv_anod(0.1) == pytest.approx(-0.25)  # -0.2 - 0.05

    def test_sample_currents_forward_reverse_net(self) -> None:
        swv = SquareWaveVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.1,
            amplitude=0.05,
            frequency=10.0,
            sample_fraction=0.2,
        )
        t = swv.time_grid(n_points=501)
        # Synthetic current proportional to potential: I(t) = 10 * E(t)
        # For cathodic scan, forward E = E_step - 0.05, reverse E = E_step + 0.05
        # => i_net = i_forward - i_reverse = 10 * (-0.10) = -1.0 (negative cathodic IUPAC)
        i_t = 10.0 * swv(t)

        E_step, i_fwd, i_rev, i_net = swv.sample_currents(t, i_t)
        np.testing.assert_allclose(E_step, swv.step_potentials)
        np.testing.assert_allclose(i_fwd, 10.0 * np.asarray(swv.forward_potentials))
        np.testing.assert_allclose(i_rev, 10.0 * np.asarray(swv.reverse_potentials))
        np.testing.assert_allclose(i_net, i_fwd - i_rev)
        np.testing.assert_allclose(i_net, np.full(swv.n_steps, -1.0))

    def test_swv_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            SquareWaveVoltammetry(0.1, 0.1, 0.01, 0.025, 10.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SquareWaveVoltammetry(0.2, -0.2, 0.0, 0.025, 10.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SquareWaveVoltammetry(0.2, -0.2, 0.01, -0.025, 10.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SquareWaveVoltammetry(0.2, -0.2, 0.01, 0.025, 0.0)
        with pytest.raises(ValueError, match="sample_fraction"):
            SquareWaveVoltammetry(0.2, -0.2, 0.01, 0.025, 10.0, sample_fraction=0.0)


# ---------------------------------------------------------------------------
# Chronopotentiometry (CP)
# ---------------------------------------------------------------------------


class TestChronopotentiometry:
    """Tests for galvanostatic Chronopotentiometry (CP)."""

    def test_single_and_multi_step_current_waveforms(self) -> None:
        cp = Chronopotentiometry(current=-1e-4, duration=5.0)
        assert isinstance(cp, BaseTechnique)
        assert cp.mode == "galvanostatic"
        assert cp.duration == 5.0
        assert cp.event_times == ()
        assert cp.currents == (-1e-4,)
        assert cp.durations == (5.0,)
        assert cp.n_steps == 1
        assert cp(0.0) == -1e-4
        assert cp.current(2.5) == -1e-4
        assert cp(5.0) == -1e-4
        with pytest.raises(ValueError, match="potentiostatic"):
            cp.potential(1.0)

        # Current-reversal CP with quiet hold
        rcp = Chronopotentiometry(
            currents=[-2e-4, 2e-4],
            durations=[2.0, 3.0],
            I_quiet=0.0,
            t_quiet=0.5,
        )
        assert rcp.duration == 5.5
        assert rcp.event_times == (0.5, 2.5)
        assert rcp.step_times == (0.0, 0.5, 2.5)
        t = np.array([0.0, 0.49, 0.5, 2.49, 2.5, 5.5])
        np.testing.assert_allclose(rcp(t), [0.0, 0.0, -2e-4, -2e-4, 2e-4, 2e-4])

    def test_interfacial_flux_iupac_convention(self) -> None:
        cp = Chronopotentiometry(currents=[-1e-4, 2e-4], durations=[1.0, 1.0])
        n = 2
        area = 1e-4

        # Cathodic current (< 0) -> negative flux (< 0)
        j_cath = cp.interfacial_flux(0.5, n=n, area=area)
        assert isinstance(j_cath, float)
        assert j_cath == pytest.approx(-1e-4 / (n * FARADAY * area))
        assert j_cath < 0.0

        # Anodic current (> 0) -> positive flux (> 0)
        j_anod = cp.interfacial_flux(1.5, n=n, area=area)
        assert j_anod == pytest.approx(2e-4 / (n * FARADAY * area))
        assert j_anod > 0.0

        # Array evaluation
        t = np.array([0.2, 1.2])
        j_arr = cp.interfacial_flux(t, n=n, area=area)
        assert isinstance(j_arr, np.ndarray)
        np.testing.assert_allclose(j_arr, [j_cath, j_anod])

    def test_recover_potential_nernst_and_butler_volmer(self) -> None:
        cp = Chronopotentiometry(current=-1e-4, duration=1.0)

        # 1. Nernst inversion matches analytical sand_potential
        n, D, c_bulk, area = 1, 1e-9, 1.0, 1e-4
        tau = sand_transition_time(I=-1e-4, n=n, D=D, c_bulk=c_bulk, area=area)
        t = np.linspace(0.05 * tau, 0.95 * tau, 20)
        c_ox = sand(t, I=-1e-4, n=n, D=D, c_bulk=c_bulk, area=area)
        c_red = c_bulk - c_ox
        E_rec = cp.recover_potential(c_ox, c_red, E0=0.25, n=n)
        E_exact = sand_potential(t, tau=tau, E_half=0.25, n=n)
        np.testing.assert_allclose(E_rec, E_exact, rtol=1e-12, atol=1e-12)

        # Scalar Nernst: equal concentrations -> E == E0
        assert cp.recover_potential(0.5, 0.5, E0=0.3) == pytest.approx(0.3)

        # 2. Butler-Volmer inversion round-trip
        bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.1, n=1)
        E_bv = cp.recover_potential(c_ox, c_red, kinetics=bv, area=area)
        j_check = bv.current_density(E_bv, c_ox, c_red)
        np.testing.assert_allclose(j_check, np.full_like(t, -1e-4 / area), rtol=1e-9)

        # Scalar Butler-Volmer inversion
        e_single = cp.recover_potential(0.8, 0.2, kinetics=bv, area=area)
        assert isinstance(e_single, float)
        assert bv.current_density(e_single, 0.8, 0.2) == pytest.approx(
            -1e-4 / area, rel=1e-9
        )

    def test_detect_transition_time(self) -> None:
        n, D, c_bulk, area, I_app = 1, 1e-9, 1.0, 1e-4, -1e-4
        tau_exact = sand_transition_time(I=I_app, n=n, D=D, c_bulk=c_bulk, area=area)
        t = np.linspace(0.0, 1.5 * tau_exact, 5000)
        c_surf = sand(t, I=I_app, n=n, D=D, c_bulk=c_bulk, area=area)

        tau_det = Chronopotentiometry.detect_transition_time(
            t, c_surf, threshold=1e-3, relative=True
        )
        assert tau_det is not None
        assert tau_det == pytest.approx(tau_exact, rel=5e-3)

        # Non-depleted signal returns None
        t_short = np.linspace(0.0, 0.5 * tau_exact, 100)
        c_short = sand(t_short, I=I_app, n=n, D=D, c_bulk=c_bulk, area=area)
        assert Chronopotentiometry.detect_transition_time(t_short, c_short) is None

    def test_cp_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="Must specify"):
            Chronopotentiometry()
        with pytest.raises(ValueError, match="Cannot mix"):
            Chronopotentiometry(
                current=-1e-4, duration=1.0, currents=[-1e-4], durations=[1.0]
            )
        with pytest.raises(ValueError, match="both"):
            Chronopotentiometry(current=-1e-4)
        with pytest.raises(ValueError, match="same length"):
            Chronopotentiometry(currents=[-1e-4, 1e-4], durations=[1.0])
        with pytest.raises(ValueError, match="empty"):
            Chronopotentiometry(currents=[], durations=[])
        with pytest.raises(ValueError, match="strictly positive"):
            Chronopotentiometry(current=-1e-4, duration=0.0)
        with pytest.raises(ValueError, match="I_quiet"):
            Chronopotentiometry(current=-1e-4, duration=1.0, t_quiet=0.5)

        cp = Chronopotentiometry(current=-1e-4, duration=1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            cp.interfacial_flux(0.5, n=0, area=1e-4)
        with pytest.raises(ValueError, match="strictly positive"):
            cp.interfacial_flux(0.5, n=1, area=-1e-4)
