"""Integration tests for the high-level Simulation Pipeline Orchestrator (simulate & TechniqueResult)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

import softpotato as sp
from softpotato.analytical.hydrodynamics import levich, nernst_diffusion_layer
from softpotato.analytical.step import (
    cottrell,
    cottrell_spherical,
    sand_transition_time,
)
from softpotato.analytical.voltammetry import randles_sevcik
from softpotato.constants import FARADAY
from softpotato.geometry import CylindricalGeometry, PlanarGeometry, SphericalGeometry
from softpotato.grid import GeometricExpandingMesh, Uniform1DMesh
from softpotato.reactions import (
    ButlerVolmer,
    ElectrochemicalStep,
    FirstOrderReaction,
    Mechanism,
    NernstianEquilibrium,
    Species,
    SpeciesCollection,
)
from softpotato.solver import CrankNicolson, ScipyIVPSolver
from softpotato.techniques import (
    BaseTechnique,
    Chronoamperometry,
    Chronopotentiometry,
    ConstantSegment,
    CustomWaveform,
    CyclicVoltammetry,
    DifferentialPulseVoltammetry,
    LinearRampSegment,
    LinearSweepVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
    TechniqueResult,
    simulate,
    simulate_sweep,
)

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_reversible_mechanism(
    *,
    D: float = 1e-9,
    c_ox: float = 1.0,
    c_red: float = 0.0,
    E0: float = 0.0,
    n: int = 1,
) -> Mechanism:
    """Create a standard reversible E mechanism: Ox + n e- <-> Red."""
    ox = Species("Ox", diffusivity=D, bulk_concentration=c_ox, n_electrons=n)
    red = Species("Red", diffusivity=D, bulk_concentration=c_red, n_electrons=-n)
    return Mechanism(
        species=[ox, red],
        electrochemical_steps=[
            ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=E0, n=n))
        ],
        name="ReversibleE",
    )


class _DuckTypedRDE(BaseTechnique):
    """Minimal hydrodynamic RDE technique exposing Nernst diffusion layer thickness."""

    def __init__(
        self,
        E_step: float,
        duration: float,
        omega: float,
        nu: float = 1e-6,
    ) -> None:
        self._mode = "potentiostatic"
        self._E_step = float(E_step)
        self._duration = float(duration)
        self.omega = float(omega)
        self.nu = float(nu)

    @property
    def duration(self) -> float:
        return self._duration

    def nernst_layer_thickness(self, D: float) -> float:
        return float(nernst_diffusion_layer(D=D, nu=self.nu, omega=self.omega))

    def excitation(self, t: float | object) -> float | np.ndarray:  # type: ignore[override]
        if np.isscalar(t):
            return self._E_step
        arr = np.asarray(t, dtype=float)
        return np.full(arr.shape, self._E_step, dtype=float)


# ---------------------------------------------------------------------------
# 1. Public Exports & Basic API
# ---------------------------------------------------------------------------


def test_pipeline_public_exports() -> None:
    """simulate, simulate_sweep, and TechniqueResult are exposed in techniques and top-level."""
    assert sp.techniques.simulate is simulate
    assert sp.techniques.simulate_sweep is simulate_sweep
    assert sp.techniques.TechniqueResult is TechniqueResult
    assert sp.simulate is simulate
    assert sp.simulate_sweep is simulate_sweep
    assert sp.TechniqueResult is TechniqueResult


# ---------------------------------------------------------------------------
# 2. Chronoamperometry (Single-step Cottrell & Double-step Event Splitting)
# ---------------------------------------------------------------------------


class TestChronoamperometryPipeline:
    def test_single_step_cottrell_accuracy(self) -> None:
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        ca = Chronoamperometry(E_step=-0.5, duration=2.0)

        t_eval = np.linspace(0.0, 2.0, 201)
        res = simulate(
            technique=ca,
            mechanism=mech,
            geometry=PlanarGeometry(area=area),
            t_eval=t_eval,
        )

        assert isinstance(res, TechniqueResult)
        assert res.success
        assert res.technique is ca
        assert res.species == ["Ox", "Red"]
        assert "Ox" in res
        assert "Red" in res
        assert res["Ox"].shape == (201, res.grid.size)
        np.testing.assert_allclose(res.time, t_eval)
        np.testing.assert_allclose(res.potential, -0.5)
        assert res.t is res.time
        assert res.E is res.potential
        assert res.i is res.current
        assert res.Q is res.charge

        # Check IUPAC polarity (cathodic reduction is negative) and Cottrell agreement
        mask = res.time >= 0.2
        i_cottrell = -np.abs(
            cottrell(res.time[mask], n=1, area=area, c_bulk=c_bulk, D=D)
        )
        np.testing.assert_allclose(res.current[mask], i_cottrell, rtol=3e-2)

        # Cumulative charge matches trapezoidal integral of I(t)
        expected_q = cumulative_trapezoid(res.current, res.time, initial=0.0)
        np.testing.assert_allclose(res.charge, expected_q, rtol=1e-12)
        assert res.charge[-1] < 0.0

    def test_double_step_ca_event_continuity_and_reversal(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        dca = Chronoamperometry(E_step1=-0.5, E_step2=0.5, t_step1=1.0, t_step2=1.0)
        t_eval = dca.time_grid(n_points=101)
        res = dca.simulate(mech, geometry=PlanarGeometry(area=1e-4), t_eval=t_eval)

        assert res.success
        assert res.metadata["n_segments"] == 2
        np.testing.assert_allclose(res.time, t_eval)

        # First step is cathodic (I < 0); second step oxidizes generated Red (I > 0)
        step1_mask = (res.time > 0.05) & (res.time < 0.99)
        step2_mask = res.time > 1.05
        assert np.all(res.current[step1_mask] < 0.0)
        assert np.all(res.current[step2_mask] > 0.0)

        # Interior concentration profile is continuous across the step boundary t = 1.0
        # and conserves total electroactive mass c_Ox + c_Red = 1.0 everywhere
        idx_event = int(np.argmin(np.abs(res.time - 1.0)))
        assert np.isclose(res.time[idx_event], 1.0)
        np.testing.assert_allclose(res["Ox"] + res["Red"], 1.0, atol=1e-4)

        dt_step = float(res.time[idx_event + 1] - res.time[idx_event])
        far_mask = res.grid > 5.0 * np.sqrt(1e-9 * dt_step)
        c_before = res["Ox"][idx_event - 1, far_mask]
        c_at = res["Ox"][idx_event, far_mask]
        c_after = res["Ox"][idx_event + 1, far_mask]
        assert np.all(np.isfinite(c_at))
        assert np.max(np.abs(c_after - c_at)) < 0.05
        assert np.max(np.abs(c_at - c_before)) < 0.05


# ---------------------------------------------------------------------------
# 3. Voltammetry (LSV & Multi-cycle CV)
# ---------------------------------------------------------------------------


class TestVoltammetryPipeline:
    def test_lsv_randles_sevcik_peak_current(self) -> None:
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        v = 0.1
        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        lsv = LinearSweepVoltammetry(E_start=0.3, E_end=-0.3, scan_rate=v, n_points=151)

        res = simulate(lsv, mech, geometry=PlanarGeometry(area=area))
        assert res.success
        assert res.metadata["n_segments"] == 1

        ip_num = float(np.min(res.current))
        ip_ref = -abs(float(randles_sevcik(v=v, n=1, area=area, c_bulk=c_bulk, D=D)))
        assert ip_num < 0.0
        assert ip_num == pytest.approx(ip_ref, rel=5e-2)

    def test_multi_cycle_cv_event_splitting_and_cycles_view(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        cv = CyclicVoltammetry(
            E_start=0.3,
            E_vertex1=-0.3,
            E_vertex2=0.3,
            scan_rate=0.2,
            n_cycles=2,
            n_points=201,
        )

        res = cv.simulate(mech, geometry=PlanarGeometry(area=1e-4))
        assert res.success
        # 2 cycles have 3 interior switching vertices -> 4 piecewise segments
        assert res.metadata["n_segments"] == 4
        assert len(res.cycles) == 2

        c1, c2 = res.cycles
        assert isinstance(c1, TechniqueResult)
        assert isinstance(c2, TechniqueResult)
        assert c1.time[0] == pytest.approx(0.0)
        assert c1.time[-1] == pytest.approx(cv.cycle_time_ranges[0][1])
        assert c2.time[0] == pytest.approx(cv.cycle_time_ranges[1][0])
        assert c2.time[-1] == pytest.approx(cv.duration)

        # Continuity across cycle 1 -> cycle 2 boundary
        np.testing.assert_allclose(c1.current[-1], c2.current[0], rtol=1e-12)
        np.testing.assert_allclose(c1["Ox"][-1], c2["Ox"][0], rtol=1e-12)

        # Both cathodic and anodic peaks appear in each cycle
        assert np.min(c1.current) < 0.0
        assert np.max(c1.current) > 0.0
        assert np.min(c2.current) < 0.0
        assert np.max(c2.current) > 0.0


# ---------------------------------------------------------------------------
# 4. Pulse & Sampled-Current Techniques (DPV, SWV, SCV)
# ---------------------------------------------------------------------------


class TestPulseTechniquesPipeline:
    def test_dpv_end_to_end_and_sampled_views(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        dpv = DifferentialPulseVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.05,
            pulse_amplitude=-0.025,
            pulse_width=0.05,
            pulse_period=0.25,
        )
        res = simulate(
            dpv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            t_eval=dpv.time_grid(n_points=181),
            mesh_kwargs={"n_points": 50},
        )
        assert res.success
        assert res.E_base.shape == (dpv.n_steps,)
        assert res.i_pre.shape == (dpv.n_steps,)
        assert res.i_pulse.shape == (dpv.n_steps,)
        assert res.i_diff.shape == (dpv.n_steps,)
        np.testing.assert_allclose(res.i_diff, res.i_pulse - res.i_pre)

        # Cathodic differential pulse peak occurs near E0 - dEp/2 = +0.0125 V
        idx_peak = int(np.argmin(res.i_diff))
        assert res.i_diff[idx_peak] < 0.0
        assert abs(res.E_base[idx_peak] - 0.0125) <= 0.06

    def test_swv_end_to_end_and_sampled_views(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        swv = SquareWaveVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.05,
            amplitude=0.025,
            frequency=10.0,
        )
        res = simulate(
            swv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            t_eval=swv.time_grid(n_points=181),
            mesh_kwargs={"n_points": 50},
        )
        assert res.success
        assert res.E_step.shape == (swv.n_steps,)
        assert res.i_forward.shape == (swv.n_steps,)
        assert res.i_reverse.shape == (swv.n_steps,)
        assert res.i_net.shape == (swv.n_steps,)
        np.testing.assert_allclose(res.i_net, res.i_forward - res.i_reverse)

        # Cathodic net SWV peak occurs near E0 = 0.0 V
        idx_peak = int(np.argmin(res.i_net))
        assert res.i_net[idx_peak] < 0.0
        assert abs(res.E_step[idx_peak] - 0.0) <= 0.06

    def test_scv_end_to_end_and_sampling(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        scv = SampledCurrentVoltammetry(
            potentials=[0.3, 0.0, -0.3],
            duration=1.0,
            n_points=51,
        )
        res = simulate(
            scv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            mesh_kwargs={"n_points": 50},
        )
        assert res.success
        assert res.current.shape == (3, 51)
        assert res.potential.shape == (3, 51)
        assert res.charge is not None and res.charge.shape == (3, 51)

        E_vals, i_sampled = res.sample_currents(sample_times=0.5)
        np.testing.assert_allclose(E_vals, [0.3, 0.0, -0.3])
        # Cathodic current magnitude increases from +0.3 V -> 0.0 V -> -0.3 V
        assert abs(i_sampled[0]) < abs(i_sampled[1]) < abs(i_sampled[2])
        assert i_sampled[2] < 0.0


# ---------------------------------------------------------------------------
# 5. Galvanostatic Chronopotentiometry (CP)
# ---------------------------------------------------------------------------


class TestChronopotentiometryPipeline:
    def test_cp_sand_transition_time_and_nernst_recovery(self) -> None:
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        n = 1
        # Choose applied cathodic current so theoretical Sand transition time tau ~ 1.0 s
        tau_target = 1.0
        i_mag = float(
            n
            * FARADAY
            * area
            * c_bulk
            * np.sqrt(np.pi * D)
            / (2.0 * np.sqrt(tau_target))
        )
        cp = Chronopotentiometry(current=-i_mag, duration=1.5)
        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.1, n=n)

        res = simulate(
            cp,
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=np.linspace(0.0, 1.5, 301),
        )
        assert res.success
        np.testing.assert_allclose(res.current, -i_mag)

        tau_ref = float(
            sand_transition_time(I=i_mag, n=n, area=area, c_bulk=c_bulk, D=D)
        )
        assert res.transition_time is not None
        assert res.transition_time == pytest.approx(tau_ref, rel=5e-2)

        # Recovered Nernstian potential is finite and decreases as Ox depletes
        mid_mask = (res.time > 0.1) & (res.time < 0.8 * tau_ref)
        e_mid = res.potential[mid_mask]
        assert np.all(np.isfinite(e_mid))
        assert e_mid[-1] < e_mid[0]

    def test_cp_butler_volmer_potential_recovery_and_reversal(self) -> None:
        D = 1e-9
        area = 1e-4
        ox = Species("Ox", diffusivity=D, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=D, bulk_concentration=0.5, n_electrons=-1)
        bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
        mech = Mechanism(
            species=[ox, red],
            electrochemical_steps=[ElectrochemicalStep(ox, red, bv)],
        )
        # Mild cathodic then anodic current reversal (stays below depletion)
        rcp = Chronopotentiometry(currents=[-1e-7, 1e-7], durations=[0.5, 0.5])
        res = rcp.simulate(
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=rcp.time_grid(n_points=101),
        )
        assert res.success
        assert res.metadata["n_segments"] == 2
        assert np.all(np.isfinite(res.potential))
        # Cathodic step shifts E negative; anodic reversal shifts E positive
        assert res.potential[10] < res.potential[-1]


# ---------------------------------------------------------------------------
# 6. Hydrodynamic RDE Auto-Meshing
# ---------------------------------------------------------------------------


def test_rde_auto_meshing_and_levich_limiting_current() -> None:
    D = 1e-9
    nu = 1e-6
    omega = 100.0  # rad/s
    c_bulk = 1.0
    area = 1e-4
    rde = _DuckTypedRDE(E_step=-0.5, duration=5.0, omega=omega, nu=nu)
    mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)

    res = simulate(
        rde,
        mech,
        geometry=PlanarGeometry(area=area),
        t_eval=np.linspace(0.0, 5.0, 101),
    )
    assert res.success

    # Verify auto-mesh outer boundary matches Nernst diffusion layer thickness delta_N
    delta_n = float(nernst_diffusion_layer(D=D, nu=nu, omega=omega))
    assert res.grid[0] == pytest.approx(0.0)
    assert res.grid[-1] == pytest.approx(delta_n, rel=1e-6)

    # Steady-state limiting current matches Levich equation
    i_levich = -abs(
        float(levich(omega=omega, n=1, area=area, c_bulk=c_bulk, D=D, nu=nu))
    )
    assert res.current[-1] == pytest.approx(i_levich, rel=3e-2)


# ---------------------------------------------------------------------------
# 7. Custom Grid, Geometry, Solver, Mechanism Shorthand & Sweeps
# ---------------------------------------------------------------------------


class TestOverridesAndSweeps:
    def test_spherical_and_cylindrical_auto_mesh_r0(self) -> None:
        D = 1e-9
        r0 = 5e-5
        c_bulk = 1.0
        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        t_eval = np.linspace(0.0, 2.0, 81)

        sphere = SphericalGeometry(r0=r0)
        res_sph = simulate(ca, mech, geometry=sphere, t_eval=t_eval)
        assert res_sph.success
        assert res_sph.grid[0] == pytest.approx(r0)
        # Compare against analytical spherical Cottrell current
        i_ref = -abs(
            float(cottrell_spherical(t_eval[-1], r=r0, n=1, c_bulk=c_bulk, D=D))
        )
        assert res_sph.current[-1] == pytest.approx(i_ref, rel=5e-2)

        cyl = CylindricalGeometry(r0=r0, length=1e-2)
        res_cyl = simulate(ca, mech, geometry=cyl, t_eval=t_eval)
        assert res_cyl.success
        assert res_cyl.grid[0] == pytest.approx(r0)
        assert np.all(res_cyl.current[1:] < 0.0)

    def test_custom_grid_and_solver_overrides(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        lsv = LinearSweepVoltammetry(
            E_start=0.2, E_end=-0.2, scan_rate=0.2, n_points=51
        )
        custom_mesh = Uniform1DMesh(0.0, 3e-4, n_points=61)

        res_cn = simulate(
            lsv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            grid=custom_mesh,
            solver="crank_nicolson",
            solver_kwargs={"dt": 0.02},
        )
        res_imp = simulate(
            lsv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            grid=custom_mesh.nodes,
            solver="implicit",
            solver_kwargs={"dt": 0.02},
        )
        res_bdf = simulate(
            lsv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            grid=GeometricExpandingMesh(0.0, 3e-4, n_points=51, gamma=1.05),
            solver=ScipyIVPSolver(method="BDF"),
            solver_kwargs={"rtol": 1e-5},
        )
        res_cn_inst = simulate(
            lsv,
            mech,
            geometry=PlanarGeometry(area=1e-4),
            grid=custom_mesh,
            solver=CrankNicolson(dt=0.02),
        )

        assert (
            res_cn.success
            and res_imp.success
            and res_bdf.success
            and res_cn_inst.success
        )
        assert res_cn.grid.size == 61
        np.testing.assert_allclose(res_cn.current, res_cn_inst.current, rtol=1e-12)
        assert np.min(res_cn.current) == pytest.approx(
            np.min(res_bdf.current), rel=1e-1
        )

    def test_species_collection_and_ec_mechanism_shorthand(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        col = SpeciesCollection([ox, red])
        lsv = LinearSweepVoltammetry(
            E_start=0.3, E_end=-0.3, scan_rate=0.2, n_points=61
        )

        res_col = simulate(lsv, col, geometry=PlanarGeometry(area=1e-4))
        res_list = simulate(lsv, [ox, red], geometry=PlanarGeometry(area=1e-4))
        assert res_col.success and res_list.success
        np.testing.assert_allclose(res_col.current, res_list.current, rtol=1e-10)

        # EC mechanism with homogeneous follow-up reaction
        prod = Species("P", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=0)
        ec = Mechanism(
            species=[ox, red, prod],
            reactions=[FirstOrderReaction(red, prod, k=5.0)],
            electrochemical_steps=[
                ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=0.0, n=1))
            ],
        )
        res_ec = simulate(lsv, ec, geometry=PlanarGeometry(area=1e-4))
        assert res_ec.success
        assert res_ec.species == ["Ox", "Red", "P"]
        assert np.max(res_ec["P"]) > 0.0

    def test_custom_waveform_potentiostatic_and_galvanostatic(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        wf_pot = CustomWaveform(
            [
                ConstantSegment(0.2, duration=0.5),
                LinearRampSegment(0.2, -0.3, duration=1.0),
            ],
            mode="potentiostatic",
            n_points=61,
        )
        res_pot = simulate(wf_pot, mech, geometry=PlanarGeometry(area=1e-4))
        assert res_pot.success
        assert res_pot.metadata["n_segments"] == 2
        assert np.min(res_pot.current) < 0.0

        wf_galv = CustomWaveform(
            [ConstantSegment(-1e-7, duration=0.5), ConstantSegment(0.0, duration=0.5)],
            mode="galvanostatic",
            n_points=51,
        )
        res_galv = simulate(wf_galv, mech, geometry=PlanarGeometry(area=1e-4))
        assert res_galv.success
        assert res_galv.metadata["n_segments"] == 2
        assert res_galv.current[0] == pytest.approx(-1e-7)
        assert res_galv.current[-1] == pytest.approx(0.0)

    def test_simulate_sweep_modes_and_randles_sevcik_scaling(self) -> None:
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        rates = [0.05, 0.2, 0.8]

        # 1. Factory + values mode
        res_map = simulate_sweep(
            lambda v: LinearSweepVoltammetry(0.3, -0.3, scan_rate=v, n_points=81),
            mech,
            values=rates,
            geometry=PlanarGeometry(area=1e-4),
        )
        assert list(res_map.keys()) == rates
        peaks = [abs(float(np.min(res_map[v].current))) for v in rates]
        # Ip scales as sqrt(v): 4x scan rate -> 2x peak current
        assert peaks[1] / peaks[0] == pytest.approx(2.0, rel=5e-2)
        assert peaks[2] / peaks[1] == pytest.approx(2.0, rel=5e-2)

        # 2. Sequence mode
        techs = [
            LinearSweepVoltammetry(0.3, -0.3, scan_rate=v, n_points=41)
            for v in rates[:2]
        ]
        res_list = simulate_sweep(techs, mech, geometry=PlanarGeometry(area=1e-4))
        assert isinstance(res_list, list)
        assert len(res_list) == 2
        assert all(r.success for r in res_list)

        # 3. Mapping mode
        res_dict = simulate_sweep(
            {"slow": techs[0], "fast": techs[1]},
            mech,
            geometry=PlanarGeometry(area=1e-4),
        )
        assert set(res_dict.keys()) == {"slow", "fast"}
        assert all(r.success for r in res_dict.values())

    def test_validation_and_wrong_technique_view_errors(self) -> None:
        mech = _make_reversible_mechanism()
        lsv = LinearSweepVoltammetry(0.2, -0.2, scan_rate=0.2, n_points=31)
        res = simulate(lsv, mech)

        with pytest.raises(TypeError, match="CyclicVoltammetry"):
            _ = res.cycles
        with pytest.raises(TypeError, match="SquareWaveVoltammetry"):
            _ = res.i_net
        with pytest.raises(TypeError, match="DifferentialPulseVoltammetry"):
            _ = res.i_diff
        with pytest.raises(TypeError, match="Chronopotentiometry"):
            _ = res.transition_time
        with pytest.raises(TypeError, match="SampledCurrentVoltammetry"):
            res.sample_currents()

        with pytest.raises(TypeError, match="BaseTechnique"):
            simulate("not_a_technique", mech)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="mechanism"):
            simulate(lsv, "not_a_mechanism")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="geometry"):
            simulate(lsv, mech, geometry=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="solver"):
            simulate(lsv, mech, solver=123)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="t_eval"):
            simulate(lsv, mech, t_eval=np.array([0.0, 999.0]))
        with pytest.raises(TypeError, match="single BaseTechnique"):
            simulate_sweep(lsv, mech)  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="values"):
            simulate_sweep(lambda v: lsv, mech)  # type: ignore[call-overload]
