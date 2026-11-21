"""End-to-end validation suite and analytical benchmarks for electrochemical techniques.

Rigorously benchmarks ``softpotato.techniques`` (``simulate``, ``simulate_sweep``,
``simulate_rde_sweep``, and diagnostic workflows) against closed-form analytical
solutions in ``softpotato.analytical`` and classical electrochemistry literature:

1. **Chronoamperometry (CA)**: Single-step planar Cottrell, double-step reversal
   (``cottrell_step``), spherical microelectrode (``cottrell_spherical``), and
   cumulative charge integration (``anson``) — all within ``< 1%`` relative error.
2. **Voltammetry (LSV & CV)**: Multi-scan-rate reversible CV vs. Randles–Ševčík
   (``< 1%`` error on :math:`I_p`, :math:`\\Delta E_p` within ``2 mV`` of theory),
   quasi-reversible Butler–Volmer CV vs. Nicholson :math:`\\Psi`, and totally
   irreversible peak shift vs. ``peak_potential_irreversible``.
3. **Pulse Voltammetry (DPV & SWV)**: Reversible peak potential localization
   (:math:`|E_p - E_{\\text{theory}}| < 5\\,\\text{mV}`), IUPAC forward/reverse
   component polarity, and peak current scaling with :math:`f^{1/2}` (SWV) and
   :math:`c^*`.
4. **Chronopotentiometry (CP)**: Sand transition time :math:`\\tau` accuracy
   (``< 1.5%`` relative error) across multiple applied currents, surface
   concentration depletion, and reversible Sand potential transient.
5. **Rotating Disk Electrode (RDE)**: Multi-:math:`\\omega` Levich limiting
   current (``< 2%`` relative error) for both Nernst-layer and convective-diffusion
   models, plus mixed kinetic/mass-transfer Koutecký–Levich current recovery.
6. **Diagnostic Round-Trip Verification & IUPAC Polarity**: Recovery of ground-truth
   :math:`D` within ``2%`` via ``analyze_cottrell``, ``analyze_randles_sevcik``,
   and ``analyze_levich``, recovery of :math:`k^0` within ``10%`` via
   ``analyze_nicholson``, and consistent IUPAC polarity across all techniques.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from softpotato.analytical.hydrodynamics import koutecky_levich, levich
from softpotato.analytical.step import (
    anson,
    cottrell,
    cottrell_spherical,
    cottrell_step,
    sand,
    sand_potential,
    sand_transition_time,
)
from softpotato.analytical.voltammetry import (
    matsuda_ayabe,
    nicholson_delta_ep,
    nicholson_psi,
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
from softpotato.geometry import PlanarGeometry, SphericalGeometry
from softpotato.grid import GeometricExpandingMesh
from softpotato.reactions import (
    ButlerVolmer,
    CustomMechanism,
    ElectrochemicalStep,
    Mechanism,
    NernstianEquilibrium,
    Species,
)
from softpotato.solver import DirichletBC, NeumannBC
from softpotato.techniques import (
    Chronoamperometry,
    Chronopotentiometry,
    CyclicVoltammetry,
    DifferentialPulseVoltammetry,
    LinearSweepVoltammetry,
    RotatingDiskElectrode,
    SquareWaveVoltammetry,
    TechniqueResult,
    analyze_cottrell,
    analyze_koutecky_levich,
    analyze_levich,
    analyze_nicholson,
    analyze_randles_sevcik,
    analyze_sand,
    find_peaks,
    simulate,
    simulate_rde_sweep,
    simulate_sweep,
)

# ==============================================================================
# Shared Helpers
# ==============================================================================


def _make_reversible_mechanism(
    *,
    D: float = 1e-9,
    c_ox: float = 1.0,
    c_red: float = 0.0,
    E0: float = 0.0,
    n: int = 1,
    T: float = STANDARD_TEMPERATURE,
) -> Mechanism:
    """Create a reversible Nernstian E mechanism: Ox + n e- <-> Red."""
    ox = Species("Ox", diffusivity=D, bulk_concentration=c_ox, n_electrons=n)
    red = Species("Red", diffusivity=D, bulk_concentration=c_red, n_electrons=-n)
    return Mechanism(
        species=[ox, red],
        electrochemical_steps=[
            ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=E0, n=n, T=T))
        ],
        name="ReversibleE",
    )


def _simulate_butler_volmer_technique(
    technique: CyclicVoltammetry | LinearSweepVoltammetry,
    *,
    k0: float,
    alpha: float = 0.5,
    D: float = 1e-9,
    c_ox: float = 1.0,
    c_red: float = 0.0,
    area: float = 1e-4,
    E0: float = 0.0,
    n: int = 1,
    T: float = STANDARD_TEMPERATURE,
    n_points: int = 75,
    gamma: float = 1.09,
    t_eval: np.ndarray | None = None,
) -> TechniqueResult:
    """Simulate a Butler–Volmer voltammetric technique via ``simulate(...)``.

    Uses a coupled ghost-node finite-volume interfacial flux with an exact
    analytical local Jacobian for stiff ``Radau`` integration, and computes the
    exact IUPAC Faradaic current from the interfacial Butler–Volmer flux.
    """
    domain_len = 6.0 * math.sqrt(D * technique.duration)
    mesh = GeometricExpandingMesh(0.0, domain_len, n_points=n_points, gamma=gamma)
    h0 = float(mesh.nodes[1] - mesh.nodes[0])
    inv_half_cell = 2.0 / h0

    bv = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=n, T=T)
    ox = Species("Ox", diffusivity=D, bulk_concentration=c_ox, n_electrons=n)
    red = Species("Red", diffusivity=D, bulk_concentration=c_red, n_electrons=-n)

    def rates(
        c: Mapping[str, np.ndarray], x: np.ndarray, t: float
    ) -> dict[str, np.ndarray]:
        kf, kb = bv.rate_constants(float(technique.potential(t)))
        j_bv = float(kf) * c["Ox"][0] - float(kb) * c["Red"][0]
        r_ox = np.zeros_like(c["Ox"])
        r_red = np.zeros_like(c["Red"])
        r_ox[0] = -inv_half_cell * j_bv
        r_red[0] = +inv_half_cell * j_bv
        return {"Ox": r_ox, "Red": r_red}

    def local_jac(
        c: Mapping[str, np.ndarray], x: np.ndarray, t: float, species: Sequence[str]
    ) -> np.ndarray:
        kf, kb = bv.rate_constants(float(technique.potential(t)))
        jac = np.zeros((2, 2, len(c["Ox"])), dtype=float)
        io = species.index("Ox")
        ir = species.index("Red")
        jac[io, io, 0] = -inv_half_cell * float(kf)
        jac[io, ir, 0] = +inv_half_cell * float(kb)
        jac[ir, io, 0] = +inv_half_cell * float(kf)
        jac[ir, ir, 0] = -inv_half_cell * float(kb)
        return jac

    mech = CustomMechanism(
        species=[ox, red],
        rates=rates,
        local_jacobian_func=local_jac,
        name=f"BV_{k0:.1e}",
    )
    res = simulate(
        technique,
        mech,
        geometry=PlanarGeometry(area=area),
        grid=mesh,
        t_eval=t_eval,
        bc_overrides={
            "Ox": (NeumannBC(0.0), DirichletBC(c_ox)),
            "Red": (NeumannBC(0.0), DirichletBC(c_red)),
        },
        solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
    )
    assert res.success, f"Butler-Volmer simulation failed: {res.message}"

    kf_arr, kb_arr = bv.rate_constants(res.potential)
    j_bv_arr = (
        np.asarray(kf_arr, dtype=float) * res["Ox"][:, 0]
        - np.asarray(kb_arr, dtype=float) * res["Red"][:, 0]
    )
    i_bv = -n * FARADAY * area * j_bv_arr
    return TechniqueResult(
        technique=technique,
        solver_result=res.solver_result,
        time=res.time,
        potential=res.potential,
        current=i_bv,
        metadata=dict(res.metadata),
    )


def _parabolic_peak(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """3-point parabolic vertex refinement around discrete index ``idx``."""
    if 0 < idx < len(x) - 1:
        x0, x1, x2 = float(x[idx - 1]), float(x[idx]), float(x[idx + 1])
        y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 0.0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (
                (x2**2) * (y0 - y1) + (x1**2) * (y2 - y0) + (x0**2) * (y1 - y2)
            ) / denom
            c = (
                x1 * x2 * (x1 - x2) * y0
                + x2 * x0 * (x2 - x0) * y1
                + x0 * x1 * (x0 - x1) * y2
            ) / denom
            if abs(a) > 1e-30:
                x_peak = -b / (2.0 * a)
                if min(x0, x2) <= x_peak <= max(x0, x2):
                    return x_peak, a * x_peak**2 + b * x_peak + c
    return float(x[idx]), float(y[idx])


# ==============================================================================
# 1. Chronoamperometry (CA) Benchmarks
# ==============================================================================


class TestChronoamperometryBenchmarks:
    """Validate single-step planar, double-step, spherical CA, and Anson charge (< 1%)."""

    def test_single_step_planar_cottrell_and_anson_charge(self) -> None:
        """Planar CA current matches Cottrell and cumulative charge matches Anson (< 1%)."""
        D = 1e-9  # m^2/s
        c_bulk = 1.0  # mol/m^3 (1 mM)
        area = 1e-4  # m^2 (1 cm^2)
        n = 1
        duration = 2.0

        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=n)
        ca = Chronoamperometry(E_step=-0.5, duration=duration)

        # Dense early timestamps capture the t^(-1/2) Cottrell singularity for accurate
        # trapezoidal charge integration via TechniqueResult.charge.
        t_eval = np.unique(
            np.concatenate(
                [
                    np.array([0.0]),
                    np.geomspace(1e-7, 0.05, 80),
                    np.linspace(0.05, duration, 120),
                ]
            )
        )
        mesh = GeometricExpandingMesh(
            0.0, 6.0 * math.sqrt(D * duration), n_points=90, gamma=1.06
        )
        res = simulate(
            ca,
            mech,
            geometry=PlanarGeometry(area=area),
            grid=mesh,
            t_eval=t_eval,
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )
        assert res.success

        # Cathodic reduction must be negative under IUPAC convention
        mask = res.time >= 0.05
        assert res.charge is not None
        assert np.all(res.current[mask] < 0.0)
        assert np.all(res.charge[mask] < 0.0)

        # Single-step Cottrell current accuracy (< 1% relative error)
        i_exact = -np.asarray(
            cottrell(res.time[mask], n=n, D=D, c_bulk=c_bulk, area=area), dtype=float
        )
        rel_err_i = np.max(np.abs(res.current[mask] - i_exact) / np.abs(i_exact))
        assert rel_err_i < 0.01, f"Planar Cottrell max rel error {rel_err_i:.4%} >= 1%"

        # Cumulative charge vs. Anson equation (< 1% relative error)
        q_exact = -np.asarray(
            anson(res.time[mask], n=n, D=D, c_bulk=c_bulk, area=area), dtype=float
        )
        rel_err_q = np.max(np.abs(res.charge[mask] - q_exact) / np.abs(q_exact))
        assert rel_err_q < 0.01, f"Anson charge max rel error {rel_err_q:.4%} >= 1%"

    def test_double_step_reversal_cottrell_step(self) -> None:
        """Double-step CA matches cottrell_step on both legs (< 1%) and reversal ratio."""
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        n = 1
        tau = 1.0
        duration = 2.0 * tau

        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=n)
        dca = Chronoamperometry(E_step1=-0.5, E_step2=0.5, t_step1=tau, t_step2=tau)

        t_eval = np.linspace(0.0, duration, 201)
        mesh = GeometricExpandingMesh(
            0.0, 6.0 * math.sqrt(D * duration), n_points=100, gamma=1.08
        )
        res = simulate(
            dca,
            mech,
            geometry=PlanarGeometry(area=area),
            grid=mesh,
            t_eval=t_eval,
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )
        assert res.success
        assert res.metadata["n_segments"] == 2

        # Forward leg (cathodic, I < 0)
        fwd_mask = (res.time >= 0.1) & (res.time <= tau - 0.01)
        i_fwd_exact = np.asarray(
            cottrell_step(
                res.time[fwd_mask], tau=tau, n=n, D=D, c_bulk=c_bulk, area=area
            ),
            dtype=float,
        )
        assert np.all(res.current[fwd_mask] < 0.0)
        rel_err_fwd = np.max(
            np.abs(res.current[fwd_mask] - i_fwd_exact) / np.abs(i_fwd_exact)
        )
        assert rel_err_fwd < 0.01, (
            f"Double-step forward max rel error {rel_err_fwd:.4%} >= 1%"
        )

        # Reversal leg (anodic, I > 0)
        rev_mask = res.time >= tau + 0.1
        i_rev_exact = np.asarray(
            cottrell_step(
                res.time[rev_mask], tau=tau, n=n, D=D, c_bulk=c_bulk, area=area
            ),
            dtype=float,
        )
        assert np.all(res.current[rev_mask] > 0.0)
        rel_err_rev = np.max(
            np.abs(res.current[rev_mask] - i_rev_exact) / np.abs(i_rev_exact)
        )
        assert rel_err_rev < 0.01, (
            f"Double-step reversal max rel error {rel_err_rev:.4%} >= 1%"
        )

        # Classic reversal ratio: -I(2*tau) / I(tau) = 1 - 1/sqrt(2)
        idx_tau = int(np.argmin(np.abs(res.time - tau)))
        ratio_sim = -float(res.current[-1]) / float(res.current[idx_tau])
        ratio_exact = 1.0 - 1.0 / math.sqrt(2.0)
        assert abs(ratio_sim - ratio_exact) / ratio_exact < 0.01

    def test_spherical_microelectrode_cottrell_spherical(self) -> None:
        """Spherical microelectrode CA matches cottrell_spherical within < 1%."""
        D = 1e-9
        c_bulk = 1.0
        r0 = 2.5e-5  # 25 um spherical microelectrode
        n = 1
        duration = 2.0

        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=0.0, n=n)
        ca = Chronoamperometry(E_step=-0.5, duration=duration)
        sphere = SphericalGeometry(r0=r0)

        t_eval = np.linspace(0.0, duration, 101)
        mesh = GeometricExpandingMesh(
            r0, r0 + 12.0 * math.sqrt(D * duration), n_points=100, gamma=1.08
        )
        res = simulate(
            ca,
            mech,
            geometry=sphere,
            grid=mesh,
            t_eval=t_eval,
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )
        assert res.success

        mask = res.time >= 0.06
        assert np.all(res.current[mask] < 0.0)
        i_exact = -np.asarray(
            cottrell_spherical(res.time[mask], r=r0, n=n, D=D, c_bulk=c_bulk),
            dtype=float,
        )
        rel_err = np.max(np.abs(res.current[mask] - i_exact) / np.abs(i_exact))
        assert rel_err < 0.01, f"Spherical Cottrell max rel error {rel_err:.4%} >= 1%"


# ==============================================================================
# 2. Voltammetry (LSV & CV) Benchmarks
# ==============================================================================


class TestVoltammetryBenchmarks:
    """Validate reversible, quasi-reversible, and irreversible LSV/CV against theory."""

    def test_reversible_cv_multi_scan_rate_randles_sevcik_and_delta_ep(self) -> None:
        """Multi-scan-rate reversible CV matches Randles–Ševčík (< 1%) and Delta_Ep (< 2 mV)."""
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        n = 1
        E0 = 0.0
        scan_rates = [0.05, 0.1, 0.25, 0.5]
        # Classical reversible CV peak separation for wide switching window (~0.45 V past E0)
        # is ~57.0-59.0 mV at 298.15 K (2.218 * RT/nF = 57.0 mV in the infinite-window limit,
        # ~58.0 mV for |E_lambda - E0| = 0.45 V).
        delta_ep_theory_mv = 58.0

        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=E0, n=n)

        for v in scan_rates:
            cv = CyclicVoltammetry(
                E_start=0.45,
                E_vertex1=-0.45,
                E_vertex2=0.45,
                scan_rate=v,
                dE=0.002,
            )
            mesh = GeometricExpandingMesh(
                0.0, 6.0 * math.sqrt(D * cv.duration), n_points=90, gamma=1.08
            )
            res = simulate(
                cv,
                mech,
                geometry=PlanarGeometry(area=area),
                grid=mesh,
                solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
            )
            assert res.success

            diag = find_peaks(res)
            assert diag.cathodic_peak is not None
            assert diag.anodic_peak is not None
            assert diag.delta_Ep is not None
            assert diag.E_half is not None

            # IUPAC polarity: cathodic peak negative, anodic peak positive
            assert diag.cathodic_peak.I_peak < 0.0
            assert diag.anodic_peak.I_peak > 0.0

            # Peak current matches Randles-Sevcik within < 1%
            ip_exact = float(
                randles_sevcik(
                    v,
                    c_bulk=c_bulk,
                    D=D,
                    area=area,
                    n=n,
                    T=STANDARD_TEMPERATURE,
                    scan_direction="cathodic",
                )
            )
            rel_err_ip = abs(diag.cathodic_peak.I_peak - ip_exact) / abs(ip_exact)
            assert rel_err_ip < 0.01, (
                f"Reversible CV I_pc error {rel_err_ip:.4%} >= 1% at v={v} V/s"
            )

            # Peak separation within 2 mV of theoretical reversible value (57-59 mV)
            delta_ep_mv = diag.delta_Ep * 1000.0
            assert abs(delta_ep_mv - delta_ep_theory_mv) < 2.0, (
                f"Reversible CV Delta_Ep={delta_ep_mv:.2f} mV deviates >= 2 mV from "
                f"{delta_ep_theory_mv:.1f} mV at v={v} V/s"
            )
            assert abs(diag.E_half - E0) < 0.002

    @pytest.mark.parametrize("k0", [1.2e-4, 7e-5, 3e-5])
    def test_quasi_reversible_butler_volmer_cv_vs_nicholson(self, k0: float) -> None:
        """Quasi-reversible Butler–Volmer CV Delta_Ep(k0, v) matches Nicholson Psi."""
        D = 1e-9  # m^2/s (1e-5 cm^2/s)
        v = 0.1  # V/s
        area = 1e-4
        c_bulk = 1.0

        cv = CyclicVoltammetry(
            E_start=0.45,
            E_vertex1=-0.45,
            E_vertex2=0.45,
            scan_rate=v,
            dE=0.002,
        )
        res = _simulate_butler_volmer_technique(
            cv, k0=k0, alpha=0.5, D=D, c_ox=c_bulk, c_red=0.0, area=area
        )

        diag = find_peaks(res)
        assert diag.delta_Ep is not None
        assert diag.cathodic_peak is not None and diag.cathodic_peak.I_peak < 0.0
        assert diag.anodic_peak is not None and diag.anodic_peak.I_peak > 0.0

        psi_true = k0 / math.sqrt(
            math.pi * D * FARADAY * v / (GAS_CONSTANT * STANDARD_TEMPERATURE)
        )
        dep_theory_mv = float(
            nicholson_delta_ep(psi_true, n=1, in_volts=False, method="table")
        )
        dep_sim_mv = diag.delta_Ep * 1000.0

        assert abs(dep_sim_mv - dep_theory_mv) < 3.5, (
            f"Quasi-reversible CV Delta_Ep={dep_sim_mv:.2f} mV vs Nicholson "
            f"{dep_theory_mv:.2f} mV for k0={k0:.1e} m/s (Psi={psi_true:.3f})"
        )

        psi_from_sim = nicholson_psi(
            delta_ep=diag.delta_Ep, n=1, in_volts=True, method="table"
        )
        assert isinstance(psi_from_sim, float)
        assert abs(psi_from_sim - psi_true) / psi_true < 0.12
        assert matsuda_ayabe(k0=k0, v=v, D=D).is_quasi_reversible

    def test_irreversible_lsv_peak_potential_shift_with_scan_rate(self) -> None:
        """Totally irreversible LSV E_p(v) matches peak_potential_irreversible and I_p."""
        D = 1e-9
        k0 = 1e-6  # m/s (1e-4 cm/s -> totally irreversible)
        alpha = 0.5
        area = 1e-4
        c_bulk = 1.0
        E0 = 0.0
        scan_rates = [0.02, 0.1, 0.5]

        ep_sim_list: list[float] = []
        ep_theory_list: list[float] = []

        for v in scan_rates:
            lsv = LinearSweepVoltammetry(
                E_start=0.2,
                E_end=-0.6,
                scan_rate=v,
                dE=0.002,
            )
            res = _simulate_butler_volmer_technique(
                lsv, k0=k0, alpha=alpha, D=D, c_ox=c_bulk, c_red=0.0, area=area, E0=E0
            )
            diag = find_peaks(res, polarity="cathodic")
            assert diag.cathodic_peak is not None
            assert diag.cathodic_peak.I_peak < 0.0

            ep_theory = float(
                peak_potential_irreversible(
                    v,
                    k0=k0,
                    E0_prime=E0,
                    D=D,
                    alpha=alpha,
                    n_alpha=1,
                    T=STANDARD_TEMPERATURE,
                    scan_direction="cathodic",
                )
            )
            ip_theory = float(
                randles_sevcik_irreversible(
                    v,
                    alpha=alpha,
                    n_alpha=1,
                    c_bulk=c_bulk,
                    D=D,
                    area=area,
                    n=1,
                    T=STANDARD_TEMPERATURE,
                    scan_direction="cathodic",
                )
            )

            assert abs(diag.cathodic_peak.E_peak - ep_theory) < 0.005, (
                f"Irreversible E_p={diag.cathodic_peak.E_peak:.4f} V vs theory "
                f"{ep_theory:.4f} V at v={v} V/s"
            )
            assert (
                abs(diag.cathodic_peak.I_peak - ip_theory) / abs(ip_theory) < 0.015
            ), (
                f"Irreversible I_p={diag.cathodic_peak.I_peak:.4e} A vs theory "
                f"{ip_theory:.4e} A at v={v} V/s"
            )

            ep_sim_list.append(diag.cathodic_peak.E_peak)
            ep_theory_list.append(ep_theory)

        # Peak shifts more negative with increasing scan rate (~ -59.2 mV/decade for alpha=0.5)
        slope_sim, _ = np.polyfit(np.log10(scan_rates), ep_sim_list, 1)
        slope_theory, _ = np.polyfit(np.log10(scan_rates), ep_theory_list, 1)
        assert slope_sim < 0.0
        assert abs(slope_sim - slope_theory) < 0.003


# ==============================================================================
# 3. Pulse Voltammetry (DPV & SWV) Benchmarks
# ==============================================================================


class TestPulseVoltammetryBenchmarks:
    """Validate DPV & SWV peak localization (< 5 mV), IUPAC polarity, and scaling."""

    def test_swv_peak_localization_polarity_and_scaling(self) -> None:
        """Reversible SWV centers at E0 (< 5 mV), obeys IUPAC polarity, scales with sqrt(f) & c*."""
        D = 1e-9
        area = 1e-4
        E0 = 0.05
        n = 1

        def _run_swv(freq: float, c_ox: float) -> TechniqueResult:
            mech = _make_reversible_mechanism(D=D, c_ox=c_ox, c_red=0.0, E0=E0, n=n)
            swv = SquareWaveVoltammetry(
                E_start=E0 + 0.18,
                E_end=E0 - 0.18,
                step_height=0.015,
                amplitude=0.025,
                frequency=freq,
            )
            mesh = GeometricExpandingMesh(
                0.0, 6.0 * math.sqrt(D * swv.duration), n_points=45, gamma=1.08
            )
            res = simulate(
                swv,
                mech,
                geometry=PlanarGeometry(area=area),
                grid=mesh,
                t_eval=swv.time_grid(n_points=4 * swv.n_steps + 1),
                solver_kwargs={"method": "Radau", "rtol": 1e-5, "atol": 1e-8},
            )
            assert res.success
            return res

        # 1. Peak localization and IUPAC component polarity at f = 10 Hz, c* = 1.0
        res_base = _run_swv(freq=10.0, c_ox=1.0)
        idx_p = int(np.argmin(res_base.i_net))
        ep_swv, ip_base = _parabolic_peak(res_base.E_step, res_base.i_net, idx_p)

        assert abs(ep_swv - E0) < 0.005, (
            f"SWV peak E_p={ep_swv:.4f} V deviates >= 5 mV from E0={E0:.4f} V"
        )
        # IUPAC polarity for cathodic scan: i_forward < 0, i_reverse > 0, i_net < 0
        assert res_base.i_forward[idx_p] < 0.0
        assert res_base.i_reverse[idx_p] > 0.0
        assert res_base.i_net[idx_p] < 0.0
        np.testing.assert_allclose(
            res_base.i_net, res_base.i_forward - res_base.i_reverse
        )

        # Symmetry check: net peak is nearly symmetric around E_p
        if 2 <= idx_p <= res_base.E_step.size - 3:
            left_val = abs(float(res_base.i_net[idx_p - 2]))
            right_val = abs(float(res_base.i_net[idx_p + 2]))
            assert abs(left_val - right_val) / max(left_val, right_val) < 0.15

        # 2. Frequency scaling: 4x frequency -> 2x peak current (I_p ~ f^(1/2))
        res_4x_f = _run_swv(freq=40.0, c_ox=1.0)
        idx_4f = int(np.argmin(res_4x_f.i_net))
        _, ip_4f = _parabolic_peak(res_4x_f.E_step, res_4x_f.i_net, idx_4f)
        assert abs(ip_4f / ip_base - 2.0) / 2.0 < 0.02, (
            f"SWV sqrt(f) scaling ratio {ip_4f / ip_base:.4f} deviates >= 2% from 2.0"
        )

        # 3. Concentration scaling: 2.5x c* -> 2.5x peak current
        res_c2 = _run_swv(freq=10.0, c_ox=2.5)
        idx_c2 = int(np.argmin(res_c2.i_net))
        _, ip_c2 = _parabolic_peak(res_c2.E_step, res_c2.i_net, idx_c2)
        assert abs(ip_c2 / ip_base - 2.5) / 2.5 < 1e-3

    def test_dpv_peak_localization_polarity_and_concentration_scaling(self) -> None:
        """Reversible cathodic DPV peaks at E0 - dEp/2 (< 5 mV) and scales linearly with c*."""
        D = 1e-9
        area = 1e-4
        E0 = 0.0
        dEp = -0.025  # Cathodic pulse amplitude
        e_peak_theory = E0 - 0.5 * dEp  # +0.0125 V

        def _run_dpv(c_ox: float) -> TechniqueResult:
            mech = _make_reversible_mechanism(D=D, c_ox=c_ox, c_red=0.0, E0=E0, n=1)
            dpv = DifferentialPulseVoltammetry(
                E_start=0.18,
                E_end=-0.18,
                step_height=0.015,
                pulse_amplitude=dEp,
                pulse_width=0.05,
                pulse_period=0.25,
            )
            mesh = GeometricExpandingMesh(
                0.0, 6.0 * math.sqrt(D * dpv.duration), n_points=45, gamma=1.08
            )
            res = simulate(
                dpv,
                mech,
                geometry=PlanarGeometry(area=area),
                grid=mesh,
                t_eval=dpv.time_grid(n_points=4 * dpv.n_steps + 1),
                solver_kwargs={"method": "Radau", "rtol": 1e-5, "atol": 1e-8},
            )
            assert res.success
            return res

        res1 = _run_dpv(c_ox=1.0)
        idx_p = int(np.argmin(res1.i_diff))
        ep_dpv, ip_1 = _parabolic_peak(res1.E_base, res1.i_diff, idx_p)

        assert res1.i_diff[idx_p] < 0.0
        assert res1.i_pulse[idx_p] < res1.i_pre[idx_p]
        np.testing.assert_allclose(res1.i_diff, res1.i_pulse - res1.i_pre)
        assert abs(ep_dpv - e_peak_theory) < 0.005, (
            f"DPV peak E_p={ep_dpv:.4f} V deviates >= 5 mV from "
            f"E0 - dEp/2 = {e_peak_theory:.4f} V"
        )

        # Linear concentration scaling: 2x c* -> 2x peak differential current
        res2 = _run_dpv(c_ox=2.0)
        idx_p2 = int(np.argmin(res2.i_diff))
        _, ip_2 = _parabolic_peak(res2.E_base, res2.i_diff, idx_p2)
        assert abs(ip_2 / ip_1 - 2.0) / 2.0 < 1e-3


# ==============================================================================
# 4. Chronopotentiometry (CP) Benchmarks
# ==============================================================================


class TestChronopotentiometryBenchmarks:
    """Validate Sand transition time (< 1.5%), surface concentration, and E(t)."""

    @pytest.mark.parametrize("tau_target", [0.4, 1.0, 2.5])
    def test_sand_transition_time_across_applied_currents(
        self, tau_target: float
    ) -> None:
        """Simulated CP transition time matches sand_transition_time within < 1.5%."""
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        n = 1
        E0 = 0.1

        # Exact cathodic current yielding theoretical transition time tau_target
        i_mag = (
            n
            * FARADAY
            * area
            * c_bulk
            * math.sqrt(math.pi * D)
            / (2.0 * math.sqrt(tau_target))
        )
        i_app = -i_mag

        tau_exact = sand_transition_time(I=i_app, n=n, D=D, c_bulk=c_bulk, area=area)
        assert math.isclose(tau_exact, tau_target, rel_tol=1e-12)

        duration = 1.3 * tau_target
        cp = Chronopotentiometry(current=i_app, duration=duration)
        mech = _make_reversible_mechanism(D=D, c_ox=c_bulk, c_red=0.0, E0=E0, n=n)
        mesh = GeometricExpandingMesh(
            0.0, 6.0 * math.sqrt(D * duration), n_points=90, gamma=1.08
        )
        t_eval = np.linspace(0.0, duration, 401)

        res = simulate(
            cp,
            mech,
            geometry=PlanarGeometry(area=area),
            grid=mesh,
            t_eval=t_eval,
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )
        assert res.success
        np.testing.assert_allclose(res.current, i_app)

        # Transition time from TechniqueResult and analyze_sand (< 1.5% rel error)
        assert res.transition_time is not None
        rel_err = abs(res.transition_time - tau_exact) / tau_exact
        assert rel_err < 0.015, (
            f"Sand tau rel error {rel_err:.4%} >= 1.5% for target tau={tau_target} s"
        )

        diag = analyze_sand(res, n=n, area=area, c_bulk=c_bulk, D=D)
        assert diag.tau is not None
        assert diag.relative_error is not None
        assert diag.relative_error < 0.015
        assert diag.D_estimated is not None
        assert abs(diag.D_estimated - D) / D < 0.02

        # Verify surface concentration transient and Nernstian Sand potential curve
        mid_mask = (res.time >= 0.1 * tau_exact) & (res.time <= 0.85 * tau_exact)
        c_exact = np.asarray(
            sand(res.time[mid_mask], I=i_app, n=n, D=D, c_bulk=c_bulk, area=area),
            dtype=float,
        )
        np.testing.assert_allclose(res["Ox"][mid_mask, 0], c_exact, rtol=1.5e-2)

        e_exact = np.asarray(
            sand_potential(res.time[mid_mask], tau=tau_exact, E_half=E0, n=n),
            dtype=float,
        )
        assert np.max(np.abs(res.potential[mid_mask] - e_exact)) < 0.003


# ==============================================================================
# 5. Rotating Disk Electrode (RDE) Benchmarks
# ==============================================================================


class TestRDEBenchmarks:
    """Validate multi-omega Levich (< 2%) and Koutecký–Levich mixed kinetic recovery."""

    @pytest.mark.parametrize("model", ["nernst_layer", "convective_diffusion"])
    def test_multi_omega_levich_limiting_current(self, model: str) -> None:
        """Multi-omega RDE limiting currents match Levich equation within < 2%."""
        D = 1e-9
        nu = 1e-6
        c_bulk = 1.0
        area = 1.9635e-5  # 5 mm diameter disk
        omegas = [40.0, 100.0, 200.0, 400.0]

        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        sweep_res = simulate_rde_sweep(
            waveform=ca,
            omegas=omegas,
            kinematic_viscosity=nu,
            model=model,  # type: ignore[arg-type]
            D=D,
            c_bulk=c_bulk,
            n_electrons=1,
            area=area,
            n_points=80,
            t_eval=ca.time_grid(41),
        )
        assert all(r.success for r in sweep_res.results)
        # IUPAC cathodic polarity: negative limiting currents
        assert np.all(sweep_res.limiting_currents < 0.0)

        i_lev_exact = -np.asarray(
            levich(omega=omegas, n=1, D=D, c_bulk=c_bulk, area=area, nu=nu),
            dtype=float,
        )
        rel_errors = np.abs(sweep_res.limiting_currents - i_lev_exact) / np.abs(
            i_lev_exact
        )
        assert np.max(rel_errors) < 0.02, (
            f"RDE ({model}) max Levich error {np.max(rel_errors):.4%} >= 2%"
        )

    def test_koutecky_levich_mixed_kinetic_mass_transfer_recovery(self) -> None:
        """Finite-kinetics RDE obeys Koutecký–Levich and recovers both D and k_f."""
        D = 1e-9
        nu = 1e-6
        c_bulk = 1.0
        area = 1e-4
        n = 1
        k0 = 5e-5  # m/s
        alpha = 0.5
        E_step = -0.12  # Mixed kinetic/mass-transfer overpotential
        omegas = [40.0, 90.0, 160.0, 250.0, 400.0]

        bv = ButlerVolmer(k0=k0, alpha=alpha, E0=0.0, n=n)
        kf_exact, kb_exact = bv.rate_constants(E_step)
        assert float(kb_exact) < 0.01 * float(kf_exact)  # Cathodic-dominated regime

        ca = Chronoamperometry(E_step=E_step, duration=2.0)
        results: list[TechniqueResult] = []

        for w in omegas:
            rde = RotatingDiskElectrode(
                ca, omega=w, kinematic_viscosity=nu, model="nernst_layer"
            )
            mesh = rde.build_grid(D=D, n_points=70, kind="geometric", gamma=1.06)
            inv_half_cell = 2.0 / float(mesh.nodes[1] - mesh.nodes[0])

            ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=n)
            red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-n)

            def rates(
                c: Mapping[str, np.ndarray],
                x: np.ndarray,
                t: float,
                _inv: float = inv_half_cell,
                _kf: float = float(kf_exact),
            ) -> dict[str, np.ndarray]:
                # Irreversible cathodic kinetic flux j = kf * c_Ox(0)
                j_k = _kf * c["Ox"][0]
                r_ox = np.zeros_like(c["Ox"])
                r_red = np.zeros_like(c["Red"])
                r_ox[0] = -_inv * j_k
                r_red[0] = +_inv * j_k
                return {"Ox": r_ox, "Red": r_red}

            mech = CustomMechanism(species=[ox, red], rates=rates, name="RDE_KL")
            res = simulate(
                rde,
                mech,
                geometry=PlanarGeometry(area=area),
                grid=mesh,
                t_eval=ca.time_grid(31),
                bc_overrides={
                    "Ox": (NeumannBC(0.0), DirichletBC(c_bulk)),
                    "Red": (NeumannBC(0.0), DirichletBC(0.0)),
                },
                solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
            )
            assert res.success
            i_k_arr = -n * FARADAY * area * float(kf_exact) * res["Ox"][:, 0]
            results.append(
                TechniqueResult(
                    technique=rde,
                    solver_result=res.solver_result,
                    time=res.time,
                    potential=res.potential,
                    current=i_k_arr,
                )
            )

        i_sim = np.array([float(r.current[-1]) for r in results])
        assert np.all(i_sim < 0.0)

        # Direct comparison against analytical Koutecky-Levich equation (< 2%)
        i_kl_exact = -np.asarray(
            koutecky_levich(
                k=float(kf_exact),
                omega=omegas,
                n=n,
                D=D,
                c_bulk=c_bulk,
                area=area,
                nu=nu,
            ),
            dtype=float,
        )
        rel_err_kl = np.max(np.abs(i_sim - i_kl_exact) / np.abs(i_kl_exact))
        assert rel_err_kl < 0.02, (
            f"Koutecky-Levich current max rel error {rel_err_kl:.4%} >= 2%"
        )

        # Round-trip recovery of both D and kinetic rate constant k_f via analyze_koutecky_levich
        kl_fit = analyze_koutecky_levich(
            results,
            potential=E_step,
            n=n,
            area=area,
            c_bulk=c_bulk,
            kinematic_viscosity=nu,
        )
        assert kl_fit.r_squared > 0.9999
        assert abs(kl_fit.d - D) / D < 0.02
        assert abs(kl_fit.k_rate - float(kf_exact)) / float(kf_exact) < 0.02


# ==============================================================================
# 6. Diagnostic Round-Trip Recovery & IUPAC Polarity Consistency
# ==============================================================================


class TestDiagnosticRoundTripBenchmarks:
    """Recover D (< 2%) via Cottrell/Randles-Sevcik/Levich and k0 (< 10%) via Nicholson."""

    def test_round_trip_analyze_cottrell_recovers_D_within_2_percent(self) -> None:
        """simulate(CA) -> analyze_cottrell recovers ground-truth D within < 2%."""
        D_true = 8.5e-10
        c_bulk = 1.5
        area = 1e-4
        n = 1

        mech = _make_reversible_mechanism(D=D_true, c_ox=c_bulk, c_red=0.0, E0=0.0, n=n)
        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        mesh = GeometricExpandingMesh(
            0.0, 6.0 * math.sqrt(D_true * ca.duration), n_points=90, gamma=1.08
        )
        res = simulate(
            ca,
            mech,
            geometry=PlanarGeometry(area=area),
            grid=mesh,
            t_eval=np.linspace(0.0, 2.0, 101),
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )
        assert res.success

        fit = analyze_cottrell(res, n=n, area=area, c_bulk=c_bulk, t_min=0.1)
        assert fit.r_squared > 0.9999
        assert fit.slope is not None and fit.slope < 0.0  # IUPAC cathodic
        assert abs(fit.d - D_true) / D_true < 0.02, (
            f"analyze_cottrell D error {abs(fit.d - D_true) / D_true:.4%} >= 2%"
        )

    def test_round_trip_analyze_randles_sevcik_recovers_D_within_2_percent(
        self,
    ) -> None:
        """simulate_sweep(CV) -> analyze_randles_sevcik recovers ground-truth D within < 2%."""
        D_true = 7.5e-10
        c_bulk = 1.2
        area = 1e-4
        n = 1
        scan_rates = [0.05, 0.1, 0.25, 0.5]

        mech = _make_reversible_mechanism(D=D_true, c_ox=c_bulk, c_red=0.0, E0=0.0, n=n)
        sweep = simulate_sweep(
            lambda v: CyclicVoltammetry(
                E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v, dE=0.003
            ),
            mech,
            values=scan_rates,
            geometry=PlanarGeometry(area=area),
            mesh_kwargs={"n_points": 85, "gamma": 1.08},
            solver_kwargs={"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
        )

        fit = analyze_randles_sevcik(
            sweep, n=n, area=area, c_bulk=c_bulk, polarity="cathodic"
        )
        assert fit.r_squared > 0.9999
        assert fit.slope < 0.0  # IUPAC cathodic
        assert abs(fit.d - D_true) / D_true < 0.02, (
            f"analyze_randles_sevcik D error {abs(fit.d - D_true) / D_true:.4%} >= 2%"
        )

    def test_round_trip_analyze_levich_recovers_D_within_2_percent(self) -> None:
        """simulate_rde_sweep -> analyze_levich recovers ground-truth D within < 2%."""
        D_true = 9.0e-10
        nu = 1.1e-6
        c_bulk = 1.0
        area = 1e-4
        n = 1
        omegas = [40.0, 90.0, 160.0, 250.0, 400.0]

        ca = Chronoamperometry(E_step=-0.5, duration=2.0)
        sweep_res = simulate_rde_sweep(
            waveform=ca,
            omegas=omegas,
            kinematic_viscosity=nu,
            model="nernst_layer",
            D=D_true,
            c_bulk=c_bulk,
            n_electrons=n,
            area=area,
            n_points=70,
            t_eval=ca.time_grid(31),
        )

        fit = analyze_levich(
            sweep_res, n=n, area=area, c_bulk=c_bulk, kinematic_viscosity=nu
        )
        assert fit.r_squared > 0.9999
        assert fit.levich_constant < 0.0  # IUPAC cathodic
        assert abs(fit.d - D_true) / D_true < 0.02, (
            f"analyze_levich D error {abs(fit.d - D_true) / D_true:.4%} >= 2%"
        )

    def test_round_trip_analyze_nicholson_recovers_k0_within_10_percent(self) -> None:
        """simulate(quasi-rev BV CV) -> analyze_nicholson recovers k0 within < 10%."""
        D_true = 1e-9
        k0_true = (
            8e-5  # m/s (Psi ~ 0.72 at v=0.1 V/s, well-conditioned Nicholson window)
        )
        v = 0.1
        area = 1e-4
        c_bulk = 1.0

        cv = CyclicVoltammetry(
            E_start=0.45,
            E_vertex1=-0.45,
            E_vertex2=0.45,
            scan_rate=v,
            dE=0.002,
        )
        res = _simulate_butler_volmer_technique(
            cv, k0=k0_true, alpha=0.5, D=D_true, c_ox=c_bulk, c_red=0.0, area=area
        )

        diag = analyze_nicholson(res, D=D_true, n=1, alpha=0.5, method="table")
        assert diag.regime == "quasi-reversible"
        assert diag.peaks is not None
        assert diag.peaks.cathodic_peak is not None
        assert diag.peaks.anodic_peak is not None
        assert diag.peaks.cathodic_peak.I_peak < 0.0
        assert diag.peaks.anodic_peak.I_peak > 0.0
        assert abs(diag.k0 - k0_true) / k0_true < 0.10, (
            f"analyze_nicholson k0 error {abs(diag.k0 - k0_true) / k0_true:.2%} >= 10% "
            f"(extracted={diag.k0:.4e} m/s vs true={k0_true:.4e} m/s)"
        )

    def test_iupac_polarity_consistency_anodic_oxidation(self) -> None:
        """Anodic oxidation of bulk Red produces positive currents across CA, LSV, SWV, DPV, CP."""
        D = 1e-9
        c_red = 1.0
        area = 1e-4
        n = 1
        mech = _make_reversible_mechanism(D=D, c_ox=0.0, c_red=c_red, E0=0.0, n=n)

        # 1. Anodic CA -> I > 0, Q > 0, analyze_cottrell slope > 0
        ca = Chronoamperometry(E_step=0.5, duration=1.0)
        res_ca = simulate(
            ca,
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=np.linspace(0.0, 1.0, 51),
        )
        assert res_ca.success
        assert res_ca.charge is not None
        assert np.all(res_ca.current[res_ca.time >= 0.1] > 0.0)
        assert res_ca.charge[-1] > 0.0
        fit_ca = analyze_cottrell(res_ca, n=n, area=area, c_bulk=c_red, t_min=0.1)
        assert fit_ca.slope is not None and fit_ca.slope > 0.0
        assert abs(fit_ca.d - D) / D < 0.03

        # 2. Anodic LSV -> anodic peak I_pa > 0 matching randles_sevcik(scan_direction="anodic")
        lsv = LinearSweepVoltammetry(E_start=-0.4, E_end=0.4, scan_rate=0.1, dE=0.005)
        res_lsv = simulate(lsv, mech, geometry=PlanarGeometry(area=area))
        assert res_lsv.success
        peaks_lsv = find_peaks(res_lsv)
        assert peaks_lsv.anodic_peak is not None
        assert peaks_lsv.cathodic_peak is None
        assert peaks_lsv.anodic_peak.I_peak > 0.0
        ip_an_exact = float(
            randles_sevcik(
                0.1, c_bulk=c_red, D=D, area=area, n=n, scan_direction="anodic"
            )
        )
        assert ip_an_exact > 0.0
        assert abs(peaks_lsv.anodic_peak.I_peak - ip_an_exact) / ip_an_exact < 0.02

        # 3. Anodic SWV -> i_net > 0 at peak near E0 = 0.0 V
        swv = SquareWaveVoltammetry(
            E_start=-0.2, E_end=0.2, step_height=0.02, amplitude=0.025, frequency=10.0
        )
        res_swv = simulate(
            swv,
            mech,
            geometry=PlanarGeometry(area=area),
            mesh_kwargs={"n_points": 55},
            t_eval=swv.time_grid(n_points=6 * swv.n_steps + 1),
        )
        assert res_swv.success
        idx_swv = int(np.argmax(res_swv.i_net))
        assert res_swv.i_net[idx_swv] > 0.0
        assert res_swv.i_forward[idx_swv] > 0.0
        assert res_swv.i_reverse[idx_swv] < 0.0

        # 4. Anodic DPV -> i_diff > 0 at peak near E0 - dEp/2 = -0.0125 V
        dpv = DifferentialPulseVoltammetry(
            E_start=-0.2,
            E_end=0.2,
            step_height=0.02,
            pulse_amplitude=0.025,
            pulse_width=0.05,
            pulse_period=0.25,
        )
        res_dpv = simulate(
            dpv,
            mech,
            geometry=PlanarGeometry(area=area),
            mesh_kwargs={"n_points": 55},
            t_eval=dpv.time_grid(n_points=6 * dpv.n_steps + 1),
        )
        assert res_dpv.success
        idx_dpv = int(np.argmax(res_dpv.i_diff))
        assert res_dpv.i_diff[idx_dpv] > 0.0
        assert res_dpv.i_pulse[idx_dpv] > res_dpv.i_pre[idx_dpv]

        # 5. Anodic CP (oxidation of Red, I > 0) -> transition time matches Sand equation
        tau_target = 1.0
        i_anodic = (
            n
            * FARADAY
            * area
            * c_red
            * math.sqrt(math.pi * D)
            / (2.0 * math.sqrt(tau_target))
        )
        cp = Chronopotentiometry(current=i_anodic, duration=1.3)
        res_cp = simulate(
            cp,
            mech,
            geometry=PlanarGeometry(area=area),
            t_eval=np.linspace(0.0, 1.3, 261),
        )
        assert res_cp.success
        assert np.all(res_cp.current > 0.0)
        assert res_cp.transition_time is not None
        assert abs(res_cp.transition_time - tau_target) / tau_target < 0.02
