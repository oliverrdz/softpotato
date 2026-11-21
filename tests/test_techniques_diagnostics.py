"""Unit and integration tests for electrochemical post-processing and diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.hydrodynamics import koutecky_levich, levich
from softpotato.analytical.kinetics import butler_volmer, tafel, tafel_slope
from softpotato.analytical.step import (
    cottrell,
    sand,
    sand_potential,
    sand_transition_time,
)
from softpotato.analytical.voltammetry import (
    nicholson_delta_ep,
    nicholson_psi,
    randles_sevcik,
    randles_sevcik_irreversible,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
from softpotato.geometry import PlanarGeometry
from softpotato.grid import GeometricExpandingMesh
from softpotato.reactions import (
    ButlerVolmer,
    CustomMechanism,
    ElectrochemicalStep,
    Mechanism,
    NernstianEquilibrium,
    Species,
)
from softpotato.solver import DirichletBC, NeumannBC, SolverResult
from softpotato.techniques import (
    Chronoamperometry,
    Chronopotentiometry,
    CVDiagnostics,
    CyclicVoltammetry,
    LinearSweepVoltammetry,
    NicholsonDiagnostics,
    PeakInfo,
    RDESweepResult,
    RotatingDiskElectrode,
    SandDiagnostics,
    TechniqueResult,
    analyze_cottrell,
    analyze_koutecky_levich,
    analyze_levich,
    analyze_nicholson,
    analyze_randles_sevcik,
    analyze_sand,
    analyze_tafel,
    find_peaks,
    simulate,
    simulate_rde_sweep,
    simulate_sweep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_solver_result(t: np.ndarray, **extra_conc: np.ndarray) -> SolverResult:
    """Create a minimal SolverResult for synthetic TechniqueResult unit tests."""
    x = np.array([0.0, 1.0], dtype=float)
    conc = extra_conc if extra_conc else {"Ox": np.ones((t.size, 2), dtype=float)}
    return SolverResult(
        t=t,
        x=x,
        concentrations=conc,
        fluxes={},
        right_fluxes={},
        currents={},
        left_currents={},
        right_currents={},
        success=True,
        message="ok",
    )


def _make_reversible_mechanism(
    *,
    D: float = 1e-9,
    c_ox: float = 1.0,
    c_red: float = 0.0,
    E0: float = 0.0,
    n: int = 1,
) -> Mechanism:
    ox = Species("Ox", diffusivity=D, bulk_concentration=c_ox, n_electrons=n)
    red = Species("Red", diffusivity=D, bulk_concentration=c_red, n_electrons=-n)
    return Mechanism(
        species=[ox, red],
        electrochemical_steps=[
            ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=E0, n=n))
        ],
    )


def _synthetic_cv_arrays(
    E_pc: float = -0.04,
    E_pa: float = 0.04,
    I_pc: float = -2e-4,
    I_pa: float = 1.8e-4,
    sigma: float = 0.03,
    n_leg: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth synthetic triangular CV with Gaussian peaks."""
    e_fwd = np.linspace(0.3, -0.3, n_leg)
    e_rev = np.linspace(-0.3, 0.3, n_leg)[1:]
    i_fwd = I_pc * np.exp(-0.5 * ((e_fwd - E_pc) / sigma) ** 2)
    i_rev = I_pa * np.exp(-0.5 * ((e_rev - E_pa) / sigma) ** 2)
    return np.concatenate([e_fwd, e_rev]), np.concatenate([i_fwd, i_rev])


# ---------------------------------------------------------------------------
# 1. Public Exports
# ---------------------------------------------------------------------------


def test_diagnostics_public_exports() -> None:
    """All diagnostic classes and functions are exported in techniques and top-level."""
    names = [
        "PeakInfo",
        "CVDiagnostics",
        "NicholsonDiagnostics",
        "SandDiagnostics",
        "find_peaks",
        "analyze_randles_sevcik",
        "analyze_nicholson",
        "analyze_levich",
        "analyze_koutecky_levich",
        "analyze_tafel",
        "analyze_cottrell",
        "analyze_sand",
    ]
    for name in names:
        assert hasattr(sp.techniques, name), f"Missing {name} in softpotato.techniques"
        assert hasattr(sp, name), f"Missing {name} in softpotato"
        assert getattr(sp, name) is getattr(sp.techniques, name)


# ---------------------------------------------------------------------------
# 2. Peak Detection (find_peaks)
# ---------------------------------------------------------------------------


class TestFindPeaks:
    def test_synthetic_cv_peaks_and_parabolic_refinement(self) -> None:
        """Parabolic refinement recovers off-grid E_peak, I_peak, and delta_Ep."""
        # Coarse grid where true peaks sit between sample points
        E_pc_true = -0.0432
        E_pa_true = 0.0387
        I_pc_true = -2.5e-4
        I_pa_true = 2.0e-4
        sigma = 0.04
        E, I = _synthetic_cv_arrays(
            E_pc=E_pc_true,
            E_pa=E_pa_true,
            I_pc=I_pc_true,
            I_pa=I_pa_true,
            sigma=sigma,
            n_leg=41,  # dE = 15 mV
        )

        diag_raw = find_peaks(E, I, parabolic=False)
        diag_par = find_peaks(E, I, parabolic=True)

        assert isinstance(diag_par, CVDiagnostics)
        assert diag_par.cathodic_peak is not None
        assert diag_par.anodic_peak is not None
        assert diag_par.cathodic_peak.polarity == "cathodic"
        assert diag_par.anodic_peak.polarity == "anodic"
        assert diag_par.cathodic_peak.I_peak < 0.0
        assert diag_par.anodic_peak.I_peak > 0.0

        # Parabolic refinement should be strictly more accurate than discrete grid
        err_raw_epc = abs(diag_raw.cathodic_peak.E_peak - E_pc_true)  # type: ignore[union-attr]
        err_par_epc = abs(diag_par.cathodic_peak.E_peak - E_pc_true)
        assert err_par_epc < err_raw_epc
        assert err_par_epc < 1e-3

        err_raw_epa = abs(diag_raw.anodic_peak.E_peak - E_pa_true)  # type: ignore[union-attr]
        err_par_epa = abs(diag_par.anodic_peak.E_peak - E_pa_true)
        assert err_par_epa < err_raw_epa
        assert err_par_epa < 1e-3

        # Summary metrics
        assert diag_par.delta_Ep is not None
        assert diag_par.E_half is not None
        assert diag_par.peak_current_ratio is not None
        assert math.isclose(diag_par.delta_Ep, E_pa_true - E_pc_true, abs_tol=1.5e-3)
        assert math.isclose(
            diag_par.E_half, 0.5 * (E_pa_true + E_pc_true), abs_tol=1e-3
        )
        assert math.isclose(
            diag_par.peak_current_ratio, abs(I_pa_true / I_pc_true), rel_tol=0.02
        )

        # Half-peak potential E_{p/2} for Gaussian: E_p +/- sigma * sqrt(2 ln 2)
        expected_half_width = sigma * math.sqrt(2.0 * math.log(2.0))
        assert diag_par.cathodic_peak.E_half_peak is not None
        assert math.isclose(
            diag_par.cathodic_peak.E_half_peak,
            E_pc_true + expected_half_width,
            abs_tol=2e-3,
        )
        assert diag_par.anodic_peak.E_half_peak is not None
        assert math.isclose(
            diag_par.anodic_peak.E_half_peak,
            E_pa_true - expected_half_width,
            abs_tol=2e-3,
        )

    def test_return_all_multiple_peaks(self) -> None:
        """return_all=True returns all anodic and cathodic peaks ordered by index."""
        E = np.linspace(-0.5, 0.5, 401)
        I = (
            1.0e-4 * np.exp(-0.5 * ((E + 0.25) / 0.03) ** 2)
            - 2.0e-4 * np.exp(-0.5 * ((E + 0.05) / 0.03) ** 2)
            + 3.0e-4 * np.exp(-0.5 * ((E - 0.15) / 0.03) ** 2)
            - 1.5e-4 * np.exp(-0.5 * ((E - 0.35) / 0.03) ** 2)
        )
        peaks = find_peaks(E, I, return_all=True)
        assert isinstance(peaks, list)
        assert len(peaks) == 4
        assert all(isinstance(p, PeakInfo) for p in peaks)
        assert [p.polarity for p in peaks] == [
            "anodic",
            "cathodic",
            "anodic",
            "cathodic",
        ]
        assert [p.index for p in peaks] == sorted(p.index for p in peaks)

        # Dominant summary selects largest anodic (+3e-4) and most negative cathodic (-2e-4)
        summary = find_peaks(E, I)
        assert summary.anodic_peak is not None
        assert summary.cathodic_peak is not None
        assert math.isclose(summary.anodic_peak.E_peak, 0.15, abs_tol=1e-3)
        assert math.isclose(summary.cathodic_peak.E_peak, -0.05, abs_tol=1e-3)

    def test_polarity_and_min_current_filters(self) -> None:
        E, I = _synthetic_cv_arrays(I_pc=-2e-4, I_pa=1e-4)

        an_only = find_peaks(E, I, polarity="anodic")
        assert an_only.anodic_peak is not None
        assert an_only.cathodic_peak is None
        assert an_only.delta_Ep is None

        cat_only = find_peaks(E, I, polarity="cathodic")
        assert cat_only.cathodic_peak is not None
        assert cat_only.anodic_peak is None

        high_thresh = find_peaks(E, I, min_current=1.5e-4)
        assert high_thresh.cathodic_peak is not None
        assert high_thresh.anodic_peak is None

    def test_multicycle_cv_cycle_selection(self) -> None:
        """Multi-cycle CV TechniqueResult selects requested cycle cleanly."""
        cv = CyclicVoltammetry(
            E_start=0.3,
            E_vertex1=-0.3,
            E_vertex2=0.3,
            scan_rate=0.5,
            n_cycles=2,
            dE=0.005,
        )
        mech = _make_reversible_mechanism(D=1e-9, c_ox=1.0, c_red=0.0, E0=0.0, n=1)
        res = simulate(cv, mech, geometry=PlanarGeometry(area=1e-4))

        diag_c0 = find_peaks(res, cycle=0)
        diag_last = find_peaks(res, cycle=-1)

        for diag in (diag_c0, diag_last):
            assert diag.cathodic_peak is not None
            assert diag.anodic_peak is not None
            assert diag.delta_Ep is not None
            assert diag.E_half is not None
            # Reversible 1e- CV at 298.15 K: delta_Ep ~ 57-60 mV, E_half ~ 0 V
            assert 0.052 <= diag.delta_Ep <= 0.066
            assert abs(diag.E_half) < 0.01

        # Cycle 2 cathodic peak magnitude is smaller than Cycle 1 due to depletion
        assert abs(diag_last.cathodic_peak.I_peak) < abs(diag_c0.cathodic_peak.I_peak)  # type: ignore[union-attr]

    def test_validation_errors(self) -> None:
        E, I = _synthetic_cv_arrays()
        with pytest.raises(ValueError, match="current"):
            find_peaks(E)
        with pytest.raises(ValueError, match="polarity"):
            find_peaks(E, I, polarity="invalid")  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="min_current"):
            find_peaks(E, I, min_current=-1.0)
        with pytest.raises(ValueError, match="Length mismatch"):
            find_peaks(E[:-1], I)

        t = np.linspace(0.0, 1.0, E.size)
        lsv = LinearSweepVoltammetry(E_start=0.3, E_end=-0.3, scan_rate=0.6)
        tr = TechniqueResult(
            technique=lsv,
            solver_result=_dummy_solver_result(t),
            time=t,
            potential=E,
            current=I,
        )
        with pytest.raises(ValueError, match="Do not pass 'current'"):
            find_peaks(tr, I)


# ---------------------------------------------------------------------------
# 3. Randles-Sevcik Diagnostics
# ---------------------------------------------------------------------------


class TestAnalyzeRandlesSevcik:
    def test_synthetic_raw_arrays_cathodic_and_anodic(self) -> None:
        """Recovers exact D from synthetic cathodic (IUPAC negative) & anodic arrays."""
        D_true = 7.2e-10
        n = 1.0
        area = 1e-4
        c_bulk = 2.0
        scan_rates = np.array([0.02, 0.05, 0.1, 0.2, 0.5])

        ip_cat = np.asarray(
            randles_sevcik(scan_rates, c_bulk=c_bulk, D=D_true, area=area, n=n),
            dtype=float,
        )
        fit_cat = analyze_randles_sevcik(
            scan_rates, ip_cat, n=n, area=area, c_bulk=c_bulk, fit_intercept=False
        )
        assert math.isclose(fit_cat.d, D_true, rel_tol=1e-6)
        assert fit_cat.slope < 0.0
        assert fit_cat.r_squared > 0.999999

        ip_an = np.abs(ip_cat)
        fit_an = analyze_randles_sevcik(
            scan_rates, ip_an, n=n, area=area, c_bulk=c_bulk, method="nonlinear"
        )
        assert math.isclose(fit_an.d, D_true, rel_tol=1e-4)
        assert fit_an.slope > 0.0

    def test_synthetic_irreversible_model(self) -> None:
        D_true = 5e-10
        alpha = 0.4
        scan_rates = np.array([0.05, 0.1, 0.25, 0.5])
        ip = np.asarray(
            randles_sevcik_irreversible(
                scan_rates, c_bulk=1.0, D=D_true, area=1e-4, n=1, alpha=alpha
            ),
            dtype=float,
        )
        fit = analyze_randles_sevcik(
            scan_rates,
            ip,
            n=1,
            area=1e-4,
            c_bulk=1.0,
            model="irreversible",
            alpha=alpha,
            fit_intercept=False,
        )
        assert math.isclose(fit.d, D_true, rel_tol=1e-6)

    def test_technique_result_sequence_and_mapping(self) -> None:
        """Accepts both list[TechniqueResult] and dict[v, TechniqueResult]."""
        D_true = 1e-9
        n = 1.0
        area = 1e-4
        c_bulk = 1.0
        scan_rates = [0.05, 0.1, 0.25, 0.5]

        results_list: list[TechniqueResult] = []
        results_map: dict[float, TechniqueResult] = {}
        for v in scan_rates:
            lsv = LinearSweepVoltammetry(E_start=0.3, E_end=-0.3, scan_rate=v)
            t = np.linspace(0.0, lsv.duration, 121)
            E = lsv.potential(t)
            ip_val = float(randles_sevcik(v, c_bulk=c_bulk, D=D_true, area=area, n=n))
            I = ip_val * np.exp(-0.5 * ((E + 0.03) / 0.04) ** 2)
            tr = TechniqueResult(
                technique=lsv,
                solver_result=_dummy_solver_result(t),
                time=t,
                potential=E,
                current=I,
            )
            results_list.append(tr)
            results_map[v] = tr

        fit_list = analyze_randles_sevcik(
            results_list, n=n, area=area, c_bulk=c_bulk, polarity="cathodic"
        )
        fit_map = analyze_randles_sevcik(
            results_map, n=n, area=area, c_bulk=c_bulk, polarity="cathodic"
        )
        assert math.isclose(fit_list.d, D_true, rel_tol=1e-3)
        assert math.isclose(fit_map.d, D_true, rel_tol=1e-3)

    def test_end_to_end_simulate_sweep_lsv(self) -> None:
        """End-to-end simulate_sweep -> analyze_randles_sevcik recovers D accurately."""
        D_true = 1e-9
        area = 1e-4
        c_bulk = 1.0
        mech = _make_reversible_mechanism(D=D_true, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        sweep = simulate_sweep(
            lambda v: LinearSweepVoltammetry(
                E_start=0.3, E_end=-0.3, scan_rate=v, dE=0.005
            ),
            mech,
            values=[0.05, 0.1, 0.2, 0.5],
            geometry=PlanarGeometry(area=area),
        )
        fit = analyze_randles_sevcik(
            sweep, n=1, area=area, c_bulk=c_bulk, polarity="cathodic"
        )
        assert fit.r_squared > 0.999
        assert math.isclose(fit.d, D_true, rel_tol=0.03)


# ---------------------------------------------------------------------------
# 4. Nicholson & Matsuda-Ayabe Diagnostics
# ---------------------------------------------------------------------------


class TestAnalyzeNicholson:
    @pytest.mark.parametrize("method", ["swaddle", "lavagnini", "table"])
    def test_synthetic_cv_and_scalar_recovery(
        self, method: Literal["swaddle", "lavagnini", "table"]
    ) -> None:
        """Recovers Psi, k0, Lambda, and Matsuda-Ayabe regime from synthetic CV."""
        D_true = 1e-9
        v = 0.2
        k0_true = 2.5e-5
        n = 1.0
        psi_true = k0_true / math.sqrt(
            math.pi * D_true * n * FARADAY * v / (GAS_CONSTANT * STANDARD_TEMPERATURE)
        )
        dep_true = float(
            nicholson_delta_ep(psi_true, n=n, in_volts=True, method=method)
        )

        # 1. From synthetic CV arrays
        E, I = _synthetic_cv_arrays(
            E_pc=-0.5 * dep_true, E_pa=0.5 * dep_true, n_leg=161
        )
        diag_arr = analyze_nicholson(
            E,
            I,
            D=D_true,
            scan_rate=v,
            n=n,
            method=method,
        )
        assert isinstance(diag_arr, NicholsonDiagnostics)
        assert diag_arr.peaks is not None
        assert math.isclose(diag_arr.delta_Ep, dep_true, abs_tol=5e-4)
        assert math.isclose(diag_arr.k0, k0_true, rel_tol=0.03)
        assert diag_arr.regime == "quasi-reversible"
        assert diag_arr.matsuda_ayabe.is_quasi_reversible

        # 2. From scalar delta_Ep
        diag_scalar = analyze_nicholson(
            dep_true,
            D=D_true,
            scan_rate=v,
            n=n,
            method=method,
        )
        assert diag_scalar.peaks is None
        assert math.isclose(diag_scalar.psi, psi_true, rel_tol=1e-2)
        assert math.isclose(diag_scalar.k0, k0_true, rel_tol=1e-2)

    def test_end_to_end_quasi_reversible_cv_simulation(self) -> None:
        """Simulates a quasi-reversible Butler-Volmer CV and extracts k0 via Nicholson."""
        D_true = 1e-9
        k0_true = 1e-4  # m/s (0.01 cm/s -> quasi-reversible at v=0.1 V/s)
        v = 0.1
        c_bulk = 1.0
        area = 1e-4
        cv = CyclicVoltammetry(
            E_start=0.45, E_vertex1=-0.45, E_vertex2=0.45, scan_rate=v, dE=0.005
        )
        mesh = GeometricExpandingMesh(
            0.0, 6.0 * math.sqrt(D_true * cv.duration), n_points=65, gamma=1.10
        )
        inv_half_cell = 2.0 / float(mesh.nodes[1] - mesh.nodes[0])
        bv = ButlerVolmer(k0=k0_true, alpha=0.5, E0=0.0, n=1)
        ox = Species("Ox", diffusivity=D_true, bulk_concentration=c_bulk, n_electrons=1)
        red = Species("Red", diffusivity=D_true, bulk_concentration=0.0, n_electrons=-1)

        def rates(
            c: Mapping[str, np.ndarray], x: np.ndarray, t: float
        ) -> dict[str, np.ndarray]:
            kf, kb = bv.rate_constants(float(cv.potential(t)))
            j_bv = float(kf) * c["Ox"][0] - float(kb) * c["Red"][0]
            r_ox = np.zeros_like(c["Ox"])
            r_red = np.zeros_like(c["Red"])
            r_ox[0] = -inv_half_cell * j_bv
            r_red[0] = inv_half_cell * j_bv
            return {"Ox": r_ox, "Red": r_red}

        mech = CustomMechanism(species=[ox, red], rates=rates, name="QuasiRevBV")
        res = simulate(
            cv,
            mech,
            geometry=PlanarGeometry(area=area),
            grid=mesh,
            bc_overrides={
                "Ox": (NeumannBC(0.0), DirichletBC(c_bulk)),
                "Red": (NeumannBC(0.0), DirichletBC(0.0)),
            },
            solver_kwargs={"method": "Radau", "rtol": 1e-6, "atol": 1e-9},
        )
        # Recompute exact Faradaic current from interfacial Butler-Volmer flux
        kf_arr, kb_arr = bv.rate_constants(res.potential)
        j_bv_arr = (
            np.asarray(kf_arr) * res["Ox"][:, 0] - np.asarray(kb_arr) * res["Red"][:, 0]
        )
        i_bv = -FARADAY * area * j_bv_arr
        bv_res = TechniqueResult(
            technique=cv,
            solver_result=res.solver_result,
            time=res.time,
            potential=res.potential,
            current=i_bv,
        )

        diag = analyze_nicholson(bv_res, D=D_true, n=1, alpha=0.5, method="table")
        assert diag.delta_Ep > 0.065
        assert diag.regime == "quasi-reversible"
        assert math.isclose(diag.k0, k0_true, rel_tol=0.15)
        expected_psi = nicholson_psi(
            delta_ep=diag.delta_Ep, in_volts=True, method="table"
        )
        assert isinstance(expected_psi, float)
        assert math.isclose(diag.psi, expected_psi, rel_tol=1e-12)

    def test_validation_errors(self) -> None:
        E, I = _synthetic_cv_arrays()
        with pytest.raises(ValueError, match="scan_rate"):
            analyze_nicholson(E, I, D=1e-9)
        with pytest.raises(ValueError, match="scan_rate"):
            analyze_nicholson(0.08, D=1e-9)
        # Single-polarity trace cannot compute delta_Ep
        e_lsv = np.linspace(0.3, -0.3, 81)
        i_lsv = -1e-4 * np.exp(-0.5 * ((e_lsv + 0.05) / 0.04) ** 2)
        with pytest.raises(ValueError, match="both an anodic and a cathodic peak"):
            analyze_nicholson(e_lsv, i_lsv, D=1e-9, scan_rate=0.1)


# ---------------------------------------------------------------------------
# 5. Levich & Koutecky-Levich Diagnostics (RDE)
# ---------------------------------------------------------------------------


class TestAnalyzeLevich:
    def test_synthetic_raw_arrays_cathodic_and_anodic(self) -> None:
        D_true = 6.5e-10
        nu = 1e-6
        area = 1.96e-5
        c_bulk = 1.0
        n = 1.0
        omegas = np.array([40.0, 90.0, 160.0, 250.0])

        il_an = np.asarray(
            levich(omega=omegas, n=n, D=D_true, c_bulk=c_bulk, area=area, nu=nu),
            dtype=float,
        )
        il_cat = -il_an

        fit_an = analyze_levich(
            omegas, il_an, n=n, area=area, c_bulk=c_bulk, kinematic_viscosity=nu
        )
        fit_cat = analyze_levich(
            omegas, il_cat, n=n, area=area, c_bulk=c_bulk, kinematic_viscosity=nu
        )
        assert math.isclose(fit_an.d, D_true, rel_tol=1e-6)
        assert math.isclose(fit_cat.d, D_true, rel_tol=1e-6)
        assert fit_cat.levich_constant < 0.0
        assert fit_an.r_squared > 0.999999

    def test_rde_technique_results_and_sweep_result(self) -> None:
        """Accepts both RDESweepResult and Sequence[TechniqueResult] with RotatingDiskElectrode."""
        D_true = 8e-10
        nu = 1.2e-6
        area = 1e-4
        c_bulk = 1.0
        n = 1.0
        omegas = [50.0, 100.0, 200.0, 400.0]

        tr_list: list[TechniqueResult] = []
        sr_list: list[SolverResult] = []
        tech_list: list[RotatingDiskElectrode] = []
        il_vals: list[float] = []

        for w in omegas:
            rde = RotatingDiskElectrode(
                E_start=0.4,
                E_end=-0.4,
                scan_rate=0.1,
                omega=w,
                kinematic_viscosity=nu,
            )
            t = np.linspace(0.0, rde.duration, 81)
            E = rde.potential(t)
            il = -float(levich(omega=w, n=n, D=D_true, c_bulk=c_bulk, area=area, nu=nu))
            # Sigmoidal RDE voltammogram approaching il at negative potentials
            I = il / (1.0 + np.exp((E - 0.0) / 0.0257))
            sr = _dummy_solver_result(t)
            sr.currents["Ox"] = I
            tr_list.append(
                TechniqueResult(
                    technique=rde,
                    solver_result=sr,
                    time=t,
                    potential=E,
                    current=I,
                )
            )
            sr_list.append(sr)
            tech_list.append(rde)
            il_vals.append(float(I[-1]))

        # Auto-infers kinematic_viscosity=1.2e-6 from RotatingDiskElectrode!
        fit_seq = analyze_levich(tr_list, n=n, area=area, c_bulk=c_bulk)
        assert math.isclose(fit_seq.d, D_true, rel_tol=1e-4)

        sweep_res = RDESweepResult(
            omegas=np.asarray(omegas, dtype=float),
            rpms=np.asarray(omegas, dtype=float) * 60.0 / (2.0 * math.pi),
            results=tuple(sr_list),
            limiting_currents=np.asarray(il_vals, dtype=float),
            techniques=tuple(tech_list),
            species="Ox",
        )
        fit_sweep = analyze_levich(sweep_res, n=n, area=area, c_bulk=c_bulk)
        assert math.isclose(fit_sweep.d, D_true, rel_tol=1e-4)

    def test_end_to_end_simulate_rde_sweep_levich(self) -> None:
        D_true = 1e-9
        nu = 1e-6
        area = 1e-4
        c_bulk = 1.0
        sweep_res = simulate_rde_sweep(
            E_start=0.2,
            E_end=-0.4,
            scan_rate=0.05,
            omegas=[50.0, 100.0, 200.0, 400.0],
            kinematic_viscosity=nu,
            D=D_true,
            c_bulk=c_bulk,
            area=area,
            n_electrons=1,
            n_points=40,
        )
        fit = analyze_levich(
            sweep_res, n=1, area=area, c_bulk=c_bulk, kinematic_viscosity=nu
        )
        assert fit.r_squared > 0.999
        assert math.isclose(fit.d, D_true, rel_tol=0.03)


class TestAnalyzeKouteckyLevich:
    def test_synthetic_raw_arrays_cathodic_iupac(self) -> None:
        """Handles negative IUPAC cathodic currents transparently."""
        D_true = 7e-10
        k_true = 2.5e-4
        nu = 1e-6
        area = 1e-4
        c_bulk = 1.0
        n = 1.0
        omegas = np.array([40.0, 90.0, 160.0, 250.0, 400.0])

        i_mag = np.asarray(
            koutecky_levich(
                k=k_true, omega=omegas, n=n, D=D_true, c_bulk=c_bulk, area=area, nu=nu
            ),
            dtype=float,
        )
        # Cathodic IUPAC negative currents
        fit = analyze_koutecky_levich(
            omegas,
            -i_mag,
            n=n,
            area=area,
            c_bulk=c_bulk,
            kinematic_viscosity=nu,
        )
        assert math.isclose(fit.d, D_true, rel_tol=1e-5)
        assert math.isclose(fit.k_rate, k_true, rel_tol=1e-5)
        assert fit.r_squared > 0.999999

    def test_rde_technique_results_at_chosen_potential(self) -> None:
        """Interpolates I(E, omega) at potential across RDE sweeps."""
        D_true = 8e-10
        nu = 1e-6
        area = 1e-4
        c_bulk = 1.0
        n = 1.0
        k0 = 5e-5
        alpha = 0.5
        E_target = -0.10
        omegas = [50.0, 100.0, 200.0, 400.0]

        tr_list: list[TechniqueResult] = []
        sr_list: list[SolverResult] = []
        tech_list: list[RotatingDiskElectrode] = []

        for w in omegas:
            rde = RotatingDiskElectrode(
                E_start=0.2,
                E_end=-0.4,
                scan_rate=0.05,
                omega=w,
                kinematic_viscosity=nu,
            )
            t = np.linspace(0.0, rde.duration, 121)
            E = rde.potential(t)
            k_e = k0 * np.exp(
                -alpha * n * FARADAY * E / (GAS_CONSTANT * STANDARD_TEMPERATURE)
            )
            i_mag = np.asarray(
                koutecky_levich(
                    k=k_e, omega=w, n=n, D=D_true, c_bulk=c_bulk, area=area, nu=nu
                ),
                dtype=float,
            )
            I = -i_mag
            sr = _dummy_solver_result(t)
            sr.currents["Ox"] = I
            tr_list.append(
                TechniqueResult(
                    technique=rde, solver_result=sr, time=t, potential=E, current=I
                )
            )
            sr_list.append(sr)
            tech_list.append(rde)

        k_expected = k0 * math.exp(
            -alpha * n * FARADAY * E_target / (GAS_CONSTANT * STANDARD_TEMPERATURE)
        )

        fit_seq = analyze_koutecky_levich(
            tr_list, potential=E_target, n=n, area=area, c_bulk=c_bulk
        )
        assert math.isclose(fit_seq.d, D_true, rel_tol=1e-3)
        assert math.isclose(fit_seq.k_rate, k_expected, rel_tol=1e-3)

        sweep_res = RDESweepResult(
            omegas=np.asarray(omegas, dtype=float),
            rpms=np.asarray(omegas, dtype=float) * 60.0 / (2.0 * math.pi),
            results=tuple(sr_list),
            limiting_currents=np.array([r.current[-1] for r in tr_list]),
            techniques=tuple(tech_list),
            species="Ox",
        )
        fit_sweep = analyze_koutecky_levich(
            sweep_res, potential=E_target, n=n, area=area, c_bulk=c_bulk
        )
        assert math.isclose(fit_sweep.d, D_true, rel_tol=1e-3)
        assert math.isclose(fit_sweep.k_rate, k_expected, rel_tol=1e-3)

    def test_validation_errors(self) -> None:
        omegas = np.array([50.0, 100.0, 200.0])
        with pytest.raises(ValueError, match="non-zero"):
            analyze_koutecky_levich(
                omegas, np.array([0.0, -1e-4, -2e-4]), n=1, area=1e-4, c_bulk=1.0
            )
        with pytest.raises(ValueError, match="same polarity"):
            analyze_koutecky_levich(
                omegas, np.array([1e-4, -1e-4, -2e-4]), n=1, area=1e-4, c_bulk=1.0
            )


# ---------------------------------------------------------------------------
# 6. Tafel Diagnostics
# ---------------------------------------------------------------------------


class TestAnalyzeTafel:
    def test_synthetic_cathodic_and_anodic_branches(self) -> None:
        E0 = 0.2
        i0_true = 1e-6
        alpha_true = 0.35
        n = 1.0
        area = 1e-4
        c_bulk = 1.0

        E = np.linspace(E0 - 0.25, E0 + 0.25, 251)
        eta = E - E0
        I = np.asarray(
            butler_volmer(eta, i0=i0_true, alpha=alpha_true, n=n), dtype=float
        )

        # Cathodic branch
        fit_cat = analyze_tafel(
            E,
            I,
            E0=E0,
            overpotential_window=(-0.22, -0.10),
            branch="cathodic",
            n=n,
            area=area,
            c_bulk=c_bulk,
        )
        b_cat_true = tafel_slope(alpha=alpha_true, n=n, branch="cathodic", signed=True)
        assert math.isclose(fit_cat.slope, b_cat_true, rel_tol=1e-2)
        assert math.isclose(fit_cat.alpha, alpha_true, rel_tol=1e-2)
        assert math.isclose(fit_cat.i0, i0_true, rel_tol=0.05)
        assert fit_cat.k0 is not None
        assert math.isclose(
            fit_cat.k0, i0_true / (n * FARADAY * area * c_bulk), rel_tol=0.05
        )

        # Anodic branch
        fit_an = analyze_tafel(
            E,
            I,
            E0=E0,
            overpotential_window=(0.10, 0.22),
            branch="anodic",
            n=n,
        )
        b_an_true = tafel_slope(alpha=alpha_true, n=n, branch="anodic", signed=True)
        assert math.isclose(fit_an.slope, b_an_true, rel_tol=1e-2)
        assert math.isclose(fit_an.alpha, alpha_true, rel_tol=1e-2)
        assert math.isclose(fit_an.i0, i0_true, rel_tol=0.05)

    def test_cv_technique_result_uses_forward_leg(self) -> None:
        """CV TechniqueResult isolates the forward leg so reverse scan does not bias fit."""
        E0 = 0.0
        i0_true = 5e-7
        alpha_true = 0.5
        cv = CyclicVoltammetry(
            E_start=0.0, E_vertex1=-0.25, E_vertex2=0.0, scan_rate=0.1
        )
        t = cv.time_grid(n_points=201)
        E = cv.potential(t)
        I_fwd = np.asarray(
            tafel(E - E0, i0=i0_true, alpha=alpha_true, branch="cathodic"), dtype=float
        )
        # Corrupt the reverse leg (second half) with flat depleted current
        mid = t.size // 2 + 1
        I = I_fwd.copy()
        I[mid:] = -1e-9

        tr = TechniqueResult(
            technique=cv,
            solver_result=_dummy_solver_result(t),
            time=t,
            potential=E,
            current=I,
        )
        fit = analyze_tafel(
            tr, E0=E0, overpotential_window=(-0.22, -0.08), branch="cathodic"
        )
        assert math.isclose(fit.alpha, alpha_true, rel_tol=1e-4)
        assert math.isclose(fit.i0, i0_true, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 7. Cottrell Diagnostics (Chronoamperometry)
# ---------------------------------------------------------------------------


class TestAnalyzeCottrell:
    def test_synthetic_raw_arrays_and_windowing(self) -> None:
        """Excludes t <= 0 and applies t_min / t_max windowing accurately."""
        D_true = 9e-10
        n = 1.0
        area = 1e-4
        c_bulk = 1.5
        t = np.linspace(0.0, 2.0, 201)
        I = np.zeros_like(t)
        I[1:] = -np.asarray(
            cottrell(t[1:], n=n, D=D_true, c_bulk=c_bulk, area=area), dtype=float
        )
        I[0] = -1.0  # t=0 spike must be ignored

        fit = analyze_cottrell(
            t, I, n=n, area=area, c_bulk=c_bulk, t_min=0.1, t_max=1.5
        )
        assert math.isclose(fit.d, D_true, rel_tol=1e-6)
        assert fit.slope is not None and fit.slope < 0.0
        assert fit.r_squared > 0.999999

    def test_end_to_end_chronoamperometry_simulation(self) -> None:
        D_true = 1e-9
        area = 1e-4
        c_bulk = 1.0
        mech = _make_reversible_mechanism(D=D_true, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        res = simulate(
            ca,
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=np.linspace(0.0, 2.0, 101),
        )
        fit = analyze_cottrell(res, n=1, area=area, c_bulk=c_bulk, t_min=0.1)
        assert fit.r_squared > 0.999
        assert math.isclose(fit.d, D_true, rel_tol=0.03)

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 1.0, 50)
        I = np.ones_like(t)
        with pytest.raises(ValueError, match="t_min"):
            analyze_cottrell(t, I, n=1, area=1e-4, c_bulk=1.0, t_min=0.8, t_max=0.2)
        with pytest.raises(ValueError, match="at least 2 valid points"):
            analyze_cottrell(t, I, n=1, area=1e-4, c_bulk=1.0, t_min=2.0)


# ---------------------------------------------------------------------------
# 8. Sand Diagnostics (Chronopotentiometry)
# ---------------------------------------------------------------------------


class TestAnalyzeSand:
    def test_synthetic_concentration_and_potential_arrays(self) -> None:
        """Detects tau from both surface concentration depletion and potential inflection."""
        D_true = 1e-9
        n = 1.0
        area = 1e-4
        c_bulk = 1.0
        I_app = -2e-4
        tau_true = sand_transition_time(
            I=I_app, n=n, D=D_true, c_bulk=c_bulk, area=area
        )

        t = np.linspace(0.0, 1.5 * tau_true, 401)
        c_surf = np.asarray(
            sand(t, I=I_app, n=n, D=D_true, c_bulk=c_bulk, area=area), dtype=float
        )

        diag_c = analyze_sand(
            t,
            c_surf,
            n=n,
            area=area,
            c_bulk=c_bulk,
            D=D_true,
            current=I_app,
            threshold=1e-3,
        )
        assert isinstance(diag_c, SandDiagnostics)
        assert diag_c.detection_method == "concentration"
        assert diag_c.tau is not None
        assert diag_c.tau_theoretical is not None
        assert diag_c.relative_error is not None
        assert diag_c.D_estimated is not None
        assert diag_c.sand_constant is not None
        assert math.isclose(diag_c.tau_theoretical, tau_true, rel_tol=1e-12)
        assert diag_c.relative_error < 0.01
        assert math.isclose(diag_c.D_estimated, D_true, rel_tol=0.01)
        assert math.isclose(
            diag_c.sand_constant, abs(I_app) * math.sqrt(diag_c.tau), rel_tol=1e-12
        )

        # Potential inflection detection using analytical sand_potential
        E_pot = np.asarray(
            sand_potential(t, tau=tau_true, E_half=0.0, n=n), dtype=float
        )
        diag_e = analyze_sand(
            t,
            E_pot,
            n=n,
            area=area,
            c_bulk=c_bulk,
            D=D_true,
            current=I_app,
            signal_kind="potential",
        )
        assert diag_e.detection_method == "potential"
        assert diag_e.tau is not None
        assert math.isclose(diag_e.tau, tau_true, rel_tol=0.01)

    def test_no_depletion_returns_none(self) -> None:
        t = np.linspace(0.0, 0.5, 51)
        c_surf = np.linspace(1.0, 0.5, 51)
        diag = analyze_sand(
            t, c_surf, n=1, area=1e-4, c_bulk=1.0, current=-1e-4, threshold=1e-3
        )
        assert diag.tau is None
        assert diag.detection_method == "none"
        assert diag.D_estimated is None
        assert diag.sand_constant is None

    def test_end_to_end_chronopotentiometry_simulation(self) -> None:
        D_true = 1e-9
        area = 1e-4
        c_bulk = 1.0
        I_app = -2.5e-4
        tau_true = sand_transition_time(
            I=I_app, n=1, D=D_true, c_bulk=c_bulk, area=area
        )
        mech = _make_reversible_mechanism(D=D_true, c_ox=c_bulk, c_red=0.0, E0=0.0, n=1)
        cp = Chronopotentiometry(current=I_app, duration=1.5 * tau_true)
        res = simulate(
            cp,
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=np.linspace(0.0, cp.duration, 201),
        )
        diag = analyze_sand(res, n=1, area=area, c_bulk=c_bulk, D=D_true)
        assert diag.detection_method == "concentration"
        assert diag.tau is not None
        assert diag.relative_error is not None and diag.relative_error < 0.03
        assert diag.D_estimated is not None
        assert math.isclose(diag.D_estimated, D_true, rel_tol=0.03)

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 1.0, 50)
        c = np.linspace(1.0, 0.0, 50)
        with pytest.raises(ValueError, match="current"):
            analyze_sand(t, c, n=1, area=1e-4, c_bulk=1.0)
        with pytest.raises(ValueError, match="zero"):
            analyze_sand(t, c, n=1, area=1e-4, c_bulk=1.0, current=0.0)
