"""Unit tests for Rotating Disk Electrode (RDE) hydrodynamic technique."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.hydrodynamics import (
    koutecky_levich_analysis,
    levich,
    nernst_diffusion_layer,
    rad_s_to_rpm,
    rpm_to_rad_s,
)
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Mesh1D,
    Uniform1DMesh,
)
from softpotato.reactions import (
    FirstOrderReaction,
    NernstianEquilibrium,
    ReactionNetwork,
)
from softpotato.solver import get_solver
from softpotato.techniques import (
    BaseTechnique,
    Chronoamperometry,
    Chronopotentiometry,
    CyclicVoltammetry,
    LinearSweepVoltammetry,
    RDEConvectionTerm,
    RDESweep,
    RDESweepResult,
    RotatingDiskElectrode,
    simulate_rde_sweep,
)

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_rde_public_exports() -> None:
    """RDE classes and helpers are exported from softpotato.techniques and top-level."""
    assert sp.techniques.RotatingDiskElectrode is RotatingDiskElectrode
    assert sp.RotatingDiskElectrode is RotatingDiskElectrode
    assert sp.techniques.RDESweep is RDESweep
    assert sp.RDESweep is RDESweep
    assert sp.techniques.RDESweepResult is RDESweepResult
    assert sp.RDESweepResult is RDESweepResult
    assert sp.techniques.simulate_rde_sweep is simulate_rde_sweep
    assert sp.simulate_rde_sweep is simulate_rde_sweep
    assert sp.techniques.RDEConvectionTerm is RDEConvectionTerm


# ---------------------------------------------------------------------------
# Construction, rotation conversion, and validation
# ---------------------------------------------------------------------------


class TestRotatingDiskElectrodeConstruction:
    """Tests for RotatingDiskElectrode initialization, units, and validation."""

    def test_direct_lsv_with_rpm_and_omega(self) -> None:
        rde_rpm = RotatingDiskElectrode(
            E_start=0.5, E_end=-0.5, scan_rate=0.05, rpm=1600.0
        )
        assert isinstance(rde_rpm, BaseTechnique)
        assert isinstance(rde_rpm.waveform, LinearSweepVoltammetry)
        assert rde_rpm.mode == "potentiostatic"
        assert rde_rpm.model == "nernst_layer"
        assert rde_rpm.domain_factor == 1.0
        assert rde_rpm.kinematic_viscosity == pytest.approx(1e-6)
        assert rde_rpm.nu == pytest.approx(1e-6)
        assert rde_rpm.rpm == pytest.approx(1600.0)
        assert rde_rpm.omega == pytest.approx(rpm_to_rad_s(1600.0))

        rde_omega = RotatingDiskElectrode(
            E_start=0.5, E_end=-0.5, scan_rate=0.05, omega=100.0
        )
        assert rde_omega.omega == pytest.approx(100.0)
        assert rde_omega.rpm == pytest.approx(rad_s_to_rpm(100.0))
        assert "RotatingDiskElectrode" in repr(rde_omega)

    def test_wrap_cv_and_ca_waveforms(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=0.1, n_cycles=2
        )
        rde_cv = RotatingDiskElectrode(cv, rpm=900.0, model="convective_diffusion")
        assert rde_cv.waveform is cv
        assert rde_cv.model == "convective_diffusion"
        assert rde_cv.domain_factor == pytest.approx(3.0)
        assert rde_cv.duration == pytest.approx(cv.duration)
        assert rde_cv.event_times == cv.event_times
        assert rde_cv(2.0) == pytest.approx(cv(2.0))
        t_arr = np.linspace(0.0, cv.duration, 21)
        np.testing.assert_allclose(rde_cv.potential(t_arr), cv.potential(t_arr))
        np.testing.assert_allclose(rde_cv.time_grid(dE=0.05), cv.time_grid(dE=0.05))

        ca = Chronoamperometry(E_step=-0.5, duration=3.0)
        rde_ca = RotatingDiskElectrode(ca, omega=150.0)
        assert rde_ca.waveform is ca
        assert rde_ca.duration == pytest.approx(3.0)
        assert rde_ca(1.5) == pytest.approx(-0.5)

    def test_with_rotation_clones_configuration(self) -> None:
        rde1 = RotatingDiskElectrode(
            E_start=0.4,
            E_end=-0.4,
            scan_rate=0.05,
            rpm=400.0,
            kinematic_viscosity=1.2e-6,
            model="convective_diffusion",
            domain_factor=3.5,
        )
        rde2 = rde1.with_rotation(rpm=1600.0)
        assert rde2.waveform is rde1.waveform
        assert rde2.kinematic_viscosity == pytest.approx(1.2e-6)
        assert rde2.model == "convective_diffusion"
        assert rde2.domain_factor == pytest.approx(3.5)
        assert rde2.rpm == pytest.approx(1600.0)
        assert rde2.omega == pytest.approx(4.0 * rde1.omega)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # Neither omega nor rpm
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1},
            # Both omega and rpm
            {
                "E_start": 0.5,
                "E_end": -0.5,
                "scan_rate": 0.1,
                "omega": 100.0,
                "rpm": 955.0,
            },
            # Non-positive omega
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "omega": 0.0},
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "omega": -50.0},
            # Non-positive rpm
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "rpm": 0.0},
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "rpm": -400.0},
            # Non-finite omega
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "omega": math.inf},
            {"E_start": 0.5, "E_end": -0.5, "scan_rate": 0.1, "omega": math.nan},
            # Non-positive kinematic viscosity
            {
                "E_start": 0.5,
                "E_end": -0.5,
                "scan_rate": 0.1,
                "omega": 100.0,
                "kinematic_viscosity": 0.0,
            },
            {
                "E_start": 0.5,
                "E_end": -0.5,
                "scan_rate": 0.1,
                "omega": 100.0,
                "kinematic_viscosity": -1e-6,
            },
            # Invalid model
            {
                "E_start": 0.5,
                "E_end": -0.5,
                "scan_rate": 0.1,
                "omega": 100.0,
                "model": "unknown_model",
            },
            # Missing waveform and incomplete LSV args
            {"omega": 100.0},
            {"E_start": 0.5, "omega": 100.0},
            # Non-positive domain_factor
            {
                "E_start": 0.5,
                "E_end": -0.5,
                "scan_rate": 0.1,
                "omega": 100.0,
                "domain_factor": 0.0,
            },
        ],
    )
    def test_invalid_parameters_raise_value_error(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RotatingDiskElectrode(**kwargs)

    def test_waveform_validation_errors(self) -> None:
        cp = Chronopotentiometry(current=-1e-4, duration=2.0)
        with pytest.raises(ValueError, match="potentiostatic"):
            RotatingDiskElectrode(cp, omega=100.0)

        lsv = LinearSweepVoltammetry(E_start=0.5, E_end=-0.5, scan_rate=0.1)
        with pytest.raises(ValueError, match="Cannot specify both"):
            RotatingDiskElectrode(
                lsv, E_start=0.5, E_end=-0.5, scan_rate=0.1, omega=100.0
            )

        with pytest.raises(TypeError, match="BaseTechnique"):
            RotatingDiskElectrode("not_a_waveform", omega=100.0)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            RotatingDiskElectrode(
                E_start=0.5,
                E_end=-0.5,
                scan_rate=0.1,
                omega=True,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Hydrodynamics: diffusion layer thickness, axial velocity, and grid building
# ---------------------------------------------------------------------------


class TestRDEHydrodynamicsAndGrid:
    """Tests for diffusion_layer_thickness, axial_velocity, and build_grid."""

    def test_diffusion_layer_thickness_and_scaling(self) -> None:
        D = 1e-9
        nu = 1e-6
        rde_100 = RotatingDiskElectrode(
            E_start=0.5, E_end=-0.5, scan_rate=0.1, omega=100.0, kinematic_viscosity=nu
        )
        rde_400 = rde_100.with_rotation(omega=400.0)

        delta_100 = rde_100.diffusion_layer_thickness(D)
        delta_400 = rde_400.diffusion_layer_thickness(D)

        expected_100 = nernst_diffusion_layer(omega=100.0, D=D, nu=nu)
        assert delta_100 == pytest.approx(expected_100)
        # delta_N ~ omega^(-1/2) => 4x omega gives 0.5x delta_N
        assert delta_400 == pytest.approx(0.5 * delta_100)
        # delta_N ~ D^(1/3) => 8x D gives 2x delta_N
        assert rde_100.diffusion_layer_thickness(8.0 * D) == pytest.approx(
            2.0 * delta_100
        )

        with pytest.raises(ValueError):
            rde_100.diffusion_layer_thickness(0.0)

    def test_axial_velocity_profile(self) -> None:
        omega = 100.0
        nu = 1e-6
        rde = RotatingDiskElectrode(
            E_start=0.5, E_end=-0.5, scan_rate=0.1, omega=omega, kinematic_viscosity=nu
        )
        assert rde.axial_velocity(0.0) == 0.0
        x = 1e-5
        expected = -0.51023 * (omega**1.5) * (nu ** (-0.5)) * (x**2)
        assert rde.axial_velocity(x) == pytest.approx(expected)
        assert rde.axial_velocity(x) < 0.0

        x_arr = np.array([0.0, 1e-5, 2e-5])
        v_arr = rde.axial_velocity(x_arr)
        assert isinstance(v_arr, np.ndarray)
        np.testing.assert_allclose(v_arr, [0.0, expected, 4.0 * expected])

        with pytest.raises(ValueError):
            rde.axial_velocity(-1e-6)

    def test_build_grid_models_and_kinds(self) -> None:
        D = 1e-9
        rde_nernst = RotatingDiskElectrode(
            E_start=0.5, E_end=-0.5, scan_rate=0.1, rpm=1600.0, model="nernst_layer"
        )
        delta = rde_nernst.diffusion_layer_thickness(D)

        m_geo = rde_nernst.build_grid(D, n_points=60, kind="geometric", gamma=1.06)
        assert isinstance(m_geo, GeometricExpandingMesh)
        assert isinstance(m_geo, Mesh1D)
        assert m_geo.n_points == 60
        assert m_geo.x_min == pytest.approx(0.0)
        assert m_geo.x_max == pytest.approx(delta)

        m_exp = rde_nernst.build_grid(D, n_points=50, kind="exponential", beta=2.5)
        assert isinstance(m_exp, ExponentialMesh)
        assert m_exp.x_max == pytest.approx(delta)

        m_uni = rde_nernst.build_grid(D, n_points=40, kind="uniform")
        assert isinstance(m_uni, Uniform1DMesh)
        assert m_uni.x_max == pytest.approx(delta)

        rde_conv = RotatingDiskElectrode(
            E_start=0.5,
            E_end=-0.5,
            scan_rate=0.1,
            rpm=1600.0,
            model="convective_diffusion",
        )
        m_conv = rde_conv.build_grid(D, n_points=80)
        assert m_conv.x_max == pytest.approx(3.0 * delta)

        m_custom = rde_conv.build_grid(D, n_points=80, domain_factor=2.5)
        assert m_custom.x_max == pytest.approx(2.5 * delta)

        with pytest.raises(ValueError, match="Unknown mesh kind"):
            rde_nernst.build_grid(D, kind="invalid")


# ---------------------------------------------------------------------------
# Numerical verification against Levich theory
# ---------------------------------------------------------------------------


class TestRDELevichVerification:
    """Verify numerical RDE steady-state limiting currents against analytical Levich."""

    @pytest.mark.parametrize("model", ["nernst_layer", "convective_diffusion"])
    def test_steady_state_limiting_current_matches_levich(self, model: str) -> None:
        D = 1e-9  # m^2/s
        nu = 1e-6  # m^2/s
        c_bulk = 1.0  # mol/m^3 (= 1 mM)
        area = 1.9635e-5  # m^2 (5 mm disk)
        rpm = 1600.0

        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        rde = RotatingDiskElectrode(
            ca,
            rpm=rpm,
            kinematic_viscosity=nu,
            model=model,  # type: ignore[arg-type]
        )

        problem = rde.build_problem(
            D=D,
            c_bulk=c_bulk,
            n_electrons=1,
            area=area,
            n_points=80,
            grid_kind="geometric",
            gamma=1.06,
        )
        solver = get_solver("scipy_ivp")
        res = solver.solve(problem, t_span=(0.0, ca.duration), t_eval=ca.time_grid(51))
        assert res.success

        i_num = abs(float(res.currents["Ox"][-1]))
        i_lev = levich(rpm=rpm, n=1, D=D, c_bulk=c_bulk, area=area, nu=nu)
        assert rde.levich_limiting_current(
            n=1, D=D, c_bulk=c_bulk, area=area
        ) == pytest.approx(i_lev)
        # Both Nernst layer (1/1.610 = 0.6211) and full convective-diffusion (0.62045)
        # agree with Levich (0.620) to well under 0.5%
        assert i_num == pytest.approx(i_lev, rel=5e-3)

        # Concentration profile should be monotone non-decreasing from electrode to bulk
        c_final = res["Ox"][-1]
        assert np.all(np.diff(c_final) >= -1e-10)

    def test_lsv_nernstian_sigmoidal_wave_matches_levich(self) -> None:
        D = 1e-9
        nu = 1e-6
        c_bulk = 1.0
        area = 1e-4
        rpm = 900.0

        # Slow scan rate ensures quasi-steady-state sigmoidal RDE voltammogram
        rde = RotatingDiskElectrode(
            E_start=0.3,
            E_end=-0.3,
            scan_rate=0.02,
            rpm=rpm,
            kinematic_viscosity=nu,
            model="nernst_layer",
        )
        eq = NernstianEquilibrium(E0=0.0, n=1)
        problem = rde.build_problem(
            D=D,
            c_bulk=c_bulk,
            kinetics=eq,
            n_electrons={"Ox": 1, "Red": -1},
            area=area,
            n_points=60,
        )
        t_eval = rde.time_grid(n_points=121)
        res = get_solver("scipy_ivp").solve(
            problem, t_span=(0.0, rde.duration), t_eval=t_eval
        )
        assert res.success

        i_ox = res.currents["Ox"]
        i_lev = levich(rpm=rpm, n=1, D=D, c_bulk=c_bulk, area=area, nu=nu)
        # Cathodic limiting plateau at E = -0.3 V (negative under IUPAC convention)
        assert i_ox[-1] < 0.0
        assert abs(i_ox[-1]) == pytest.approx(i_lev, rel=1e-2)
        # At E = E0 = 0.0 V (midpoint), current should be ~50% of limiting current
        mid_idx = len(t_eval) // 2
        assert abs(i_ox[mid_idx]) == pytest.approx(0.5 * i_lev, rel=5e-2)

    def test_convection_term_composes_with_homogeneous_reaction(self) -> None:
        D = 1e-9
        nu = 1e-6
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        rde = RotatingDiskElectrode(
            ca, rpm=1600.0, kinematic_viscosity=nu, model="convective_diffusion"
        )
        rxn = ReactionNetwork([FirstOrderReaction(reactant="Ox", product="P", k=0.5)])
        problem = rde.build_problem(
            diffusivity={"Ox": D, "P": D},
            c_bulk=1.0,
            initial_conditions={"Ox": 1.0, "P": 0.0},
            boundary_conditions={
                "Ox": (sp.DirichletBC(0.0), sp.DirichletBC(1.0)),
                "P": (sp.NeumannBC(0.0), sp.DirichletBC(0.0)),
            },
            reactions=rxn,
            n_electrons={"Ox": 1, "P": 0},
            area=1e-4,
            n_points=50,
        )
        res = get_solver("scipy_ivp").solve(
            problem, t_span=(0.0, ca.duration), t_eval=ca.time_grid(31)
        )
        assert res.success
        assert np.all(np.isfinite(res["Ox"]))
        assert np.all(np.isfinite(res["P"]))
        assert np.max(res["P"]) > 0.0


# ---------------------------------------------------------------------------
# Multi-rotation-rate sweeps (RDESweep & simulate_rde_sweep)
# ---------------------------------------------------------------------------


class TestRDESweep:
    """Tests for RDESweep and simulate_rde_sweep."""

    def test_sweep_container_and_validation(self) -> None:
        sweep = RDESweep(
            E_start=0.4,
            E_end=-0.4,
            scan_rate=0.05,
            rpms=[400.0, 900.0, 1600.0, 2500.0],
        )
        assert len(sweep) == 4
        assert sweep.n_rates == 4
        assert sweep.model == "nernst_layer"
        assert sweep.nu == pytest.approx(1e-6)
        np.testing.assert_allclose(sweep.rpms, [400.0, 900.0, 1600.0, 2500.0])
        np.testing.assert_allclose(
            sweep.omegas, rpm_to_rad_s([400.0, 900.0, 1600.0, 2500.0])
        )
        assert isinstance(sweep[0], RotatingDiskElectrode)
        assert len(sweep[1:3]) == 2
        assert len(list(sweep)) == 4
        assert "RDESweep" in repr(sweep)

        deltas = sweep.diffusion_layer_thicknesses(D=1e-9)
        assert deltas.shape == (4,)
        assert np.all(np.diff(deltas) < 0.0)

        with pytest.raises(ValueError):
            RDESweep(E_start=0.4, E_end=-0.4, scan_rate=0.05)
        with pytest.raises(ValueError):
            RDESweep(
                E_start=0.4, E_end=-0.4, scan_rate=0.05, omegas=[100.0], rpms=[955.0]
            )
        with pytest.raises(ValueError):
            RDESweep(E_start=0.4, E_end=-0.4, scan_rate=0.05, omegas=[])
        with pytest.raises(ValueError):
            RDESweep(E_start=0.4, E_end=-0.4, scan_rate=0.05, rpms=[400.0, -100.0])

    @pytest.mark.parametrize("model", ["nernst_layer", "convective_diffusion"])
    def test_simulate_rde_sweep_levich_and_koutecky_levich(self, model: str) -> None:
        D = 1e-9
        nu = 1e-6
        c_bulk = 1.0
        area = 1.9635e-5
        rpms = [400.0, 900.0, 1600.0, 2500.0]

        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        sweep_res = simulate_rde_sweep(
            waveform=ca,
            rpms=rpms,
            kinematic_viscosity=nu,
            model=model,  # type: ignore[arg-type]
            D=D,
            c_bulk=c_bulk,
            n_electrons=1,
            area=area,
            n_points=60,
            t_eval=ca.time_grid(31),
        )

        assert isinstance(sweep_res, RDESweepResult)
        assert len(sweep_res) == 4
        assert sweep_res.species == "Ox"
        assert all(r.success for r in sweep_res.results)

        i_num = np.abs(sweep_res.limiting_currents)
        i_lev = np.asarray(
            levich(rpm=rpms, n=1, D=D, c_bulk=c_bulk, area=area, nu=nu), dtype=float
        )
        np.testing.assert_allclose(i_num, i_lev, rtol=1e-2)

        # Verify Koutecky-Levich regression recovers diffusion coefficient D
        kl = koutecky_levich_analysis(
            omega=sweep_res.omegas,
            current=i_num,
            area=area,
            c_bulk=c_bulk,
            n=1,
            nu=nu,
        )
        assert kl.r_squared > 0.9999
        assert kl.d_estimated == pytest.approx(D, rel=2e-2)

    def test_sweep_method_simulate(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.5)
        sweep = RDESweep(ca, omegas=[50.0, 200.0])
        res = sweep.simulate(
            D=1e-9, c_bulk=1.0, area=1e-4, n_points=50, t_eval=ca.time_grid(21)
        )
        assert len(res) == 2
        # 4x omega => 2x limiting current
        ratio = abs(res.limiting_currents[1]) / abs(res.limiting_currents[0])
        assert ratio == pytest.approx(2.0, rel=1e-2)
