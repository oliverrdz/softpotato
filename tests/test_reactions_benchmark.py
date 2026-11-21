"""Validation suite and analytical benchmarks for coupled chemical and electrochemical mechanisms.

Benchmarks ``softpotato.reactions`` integrated with ``DiffusionProblem`` and
``ScipyIVPSolver`` against exact analytical solutions, literature diagnostics,
and physical conservation laws:

1. **EC' (Catalytic) Mechanism**: Steady-state limiting current
   :math:`I_{\\text{cat}} = -n F A c^* \\sqrt{D k'}` across
   :math:`k' \\in [0.1, 100]\\,\\text{s}^{-1}`, steady-state concentration
   profile, and transient Delahay–Stiehl chronoamperometry.
2. **Chemical Equilibrium Relaxation**: Exact exponential relaxation of
   :math:`A \\rightleftharpoons B` to thermodynamic equilibrium
   :math:`c_B / c_A = K_{\\text{eq}} = k_f / k_b`.
3. **Mass Conservation**: Machine-precision conservation of total integrated
   mass :math:`\\int_0^L \\sum_i c_i(x, t)\\,dx` (< 1e-10 relative drift) for
   closed-box stoichiometric networks across uniform and non-uniform meshes.
4. **Butler–Volmer Voltammetry Diagnostics**: Cyclic voltammetry across
   reversible (:math:`k^0 \\ge 1\\,\\text{cm/s}`), quasi-reversible
   (:math:`10^{-3} < k^0 < 1\\,\\text{cm/s}`), and totally irreversible
   (:math:`k^0 \\le 10^{-4}\\,\\text{cm/s}`) regimes, verified against
   Randles–Ševčík peak currents and Nicholson's :math:`\\Psi` working curve.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.special import erf

from softpotato.analytical.voltammetry import (
    matsuda_ayabe,
    nicholson_delta_ep,
    nicholson_rate_constant,
    randles_sevcik,
    randles_sevcik_irreversible,
    randles_sevcik_quasi,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
from softpotato.geometry import PlanarGeometry
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Mesh1D,
    Uniform1DMesh,
)
from softpotato.reactions import (
    ButlerVolmer,
    CatalyticReaction,
    CustomMechanism,
    DisproportionationReaction,
    ElectrochemicalStep,
    FirstOrderReaction,
    Mechanism,
    NernstianEquilibrium,
    ReactionNetwork,
    ReversibleReaction,
    Species,
)
from softpotato.solver import (
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    ScipyIVPSolver,
)


def _integrate_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trapezoidal integral along the last axis (NumPy 1.x and 2.x compatible)."""
    dx = np.diff(x)
    return 0.5 * np.sum((y[..., :-1] + y[..., 1:]) * dx, axis=-1)


def _parabolic_peak(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """Refine peak location and value via 3-point parabolic interpolation."""
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


@lru_cache(maxsize=32)
def _simulate_butler_volmer_cv(
    k0_cm_s: float,
    *,
    v: float = 0.1,
    alpha: float = 0.5,
    D_cm2_s: float = 1e-5,
    c_bulk_mM: float = 1.0,
    area_cm2: float = 1.0,
    E0: float = 0.0,
    E_start: float = 0.45,
    E_switch: float = -0.45,
    n_points: int = 75,
    gamma: float = 1.10,
    n_eval: int = 901,
) -> dict[str, float]:
    """Simulate a 1-electron Butler-Volmer cyclic voltammogram in SI units.

    Implements the coupled interfacial Butler-Volmer Robin boundary condition at
    ``x = 0`` via the ghost-node finite-volume flux term in ``CustomMechanism``
    with an exact analytical local Jacobian for stiff ``Radau`` integration.
    """
    # Convert CGS-style electrochemical inputs to SI (m, s, mol/m^3, m/s, m^2/s)
    D = D_cm2_s * 1e-4  # cm^2/s -> m^2/s
    c_bulk = c_bulk_mM  # 1 mM == 1 mol/m^3
    area = area_cm2 * 1e-4  # cm^2 -> m^2
    k0 = k0_cm_s * 1e-2  # cm/s -> m/s

    t_switch = (E_start - E_switch) / v
    t_end = 2.0 * t_switch

    def potential(t: float) -> float:
        if t <= t_switch:
            return E_start - v * t
        return E_switch + v * (t - t_switch)

    domain_len = 6.0 * math.sqrt(D * t_end)
    mesh = GeometricExpandingMesh(0.0, domain_len, n_points=n_points, gamma=gamma)
    h0 = float(mesh.nodes[1] - mesh.nodes[0])
    inv_half_cell = 2.0 / h0

    bv = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1, T=STANDARD_TEMPERATURE)
    ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=1)
    red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-1)

    def rates(c, x, t):
        kf, kb = bv.rate_constants(potential(t))
        j_bv = float(kf) * c["Ox"][0] - float(kb) * c["Red"][0]
        r_ox = np.zeros_like(c["Ox"])
        r_red = np.zeros_like(c["Red"])
        r_ox[0] = -inv_half_cell * j_bv
        r_red[0] = +inv_half_cell * j_bv
        return {"Ox": r_ox, "Red": r_red}

    def local_jac(c, x, t, species):
        kf, kb = bv.rate_constants(potential(t))
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
        name=f"BV_CV_k0_{k0_cm_s:.1e}",
    )
    problem = mech.create_problem(
        mesh,
        boundary_conditions={
            "Ox": (NeumannBC(0.0), DirichletBC(c_bulk)),
            "Red": (NeumannBC(0.0), DirichletBC(0.0)),
        },
        geometry=PlanarGeometry(area=area),
    )

    t_eval = np.linspace(0.0, t_end, n_eval)
    result = ScipyIVPSolver(method="Radau", rtol=1e-7, atol=1e-10).solve(
        problem, t_span=(0.0, t_end), t_eval=t_eval
    )
    assert result.success, f"CV solver failed for k0={k0_cm_s} cm/s: {result.message}"

    E_arr = np.array([potential(float(ti)) for ti in result.t])
    current = result.currents["Ox"]

    fwd_mask = result.t <= t_switch
    rev_mask = result.t >= t_switch

    E_fwd, I_fwd = E_arr[fwd_mask], current[fwd_mask]
    E_rev, I_rev = E_arr[rev_mask], current[rev_mask]

    idx_c = int(np.argmin(I_fwd))
    idx_a = int(np.argmax(I_rev))
    Epc, Ipc = _parabolic_peak(E_fwd, I_fwd, idx_c)
    Epa, Ipa = _parabolic_peak(E_rev, I_rev, idx_a)

    psi = k0 / math.sqrt(
        math.pi * D * FARADAY * v / (GAS_CONSTANT * STANDARD_TEMPERATURE)
    )
    return {
        "k0_cm_s": k0_cm_s,
        "k0_m_s": k0,
        "D_m2_s": D,
        "area_m2": area,
        "c_bulk_mol_m3": c_bulk,
        "psi": psi,
        "Epc": Epc,
        "Epa": Epa,
        "delta_ep_V": Epa - Epc,
        "delta_ep_mV": (Epa - Epc) * 1000.0,
        "Ipc": Ipc,
        "Ipa": Ipa,
    }


# ---------------------------------------------------------------------------
# 1. Catalytic EC' Mechanism Benchmarks
# ---------------------------------------------------------------------------
class TestCatalyticECPrimeBenchmark:
    """Analytical benchmarks for the EC' catalytic regeneration mechanism."""

    @pytest.mark.parametrize("k_prime", [0.1, 1.0, 10.0, 100.0])
    def test_catalytic_limiting_current_across_rate_constants(
        self, k_prime: float
    ) -> None:
        """Steady-state EC' current matches I_cat = -n F A c* sqrt(D k') within < 1%."""
        D = 1e-9  # m^2/s
        c_bulk = 1.0  # mol/m^3
        area = 1e-4  # m^2
        n = 1

        # Reaction layer thickness mu = sqrt(D / k'); domain spans 10 * mu
        mu = math.sqrt(D / k_prime)
        domain_len = 10.0 * mu
        mesh = GeometricExpandingMesh(0.0, domain_len, n_points=90, gamma=1.06)

        ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=n)
        red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-n)
        network = ReactionNetwork([CatalyticReaction(red, ox, k=k_prime)])

        problem = DiffusionProblem(
            grid=mesh,
            species=[ox, red],
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(c_bulk)),
                "Red": (DirichletBC(c_bulk), DirichletBC(0.0)),
            },
            reactions=network,
            geometry=PlanarGeometry(area=area),
        )

        # Integrate to k' * t_end = 10 so transient term < 1e-5 relative to steady state
        t_end = 10.0 / k_prime
        result = ScipyIVPSolver(method="Radau", rtol=1e-7, atol=1e-9).solve(
            problem, t_span=(0.0, t_end)
        )
        assert result.success
        assert result.raw_output.njev > 0

        i_cat_exact = -n * FARADAY * area * c_bulk * math.sqrt(D * k_prime)
        i_sim = float(result.currents["Ox"][-1])
        rel_error = abs(i_sim - i_cat_exact) / abs(i_cat_exact)
        assert rel_error < 0.01, (
            f"EC' limiting current error {rel_error:.4%} >= 1% for k'={k_prime} s^-1 "
            f"(sim={i_sim:.6e} A, exact={i_cat_exact:.6e} A)"
        )

        # Verify steady-state concentration profile c_Ox(x) = c* (1 - exp(-x / mu))
        c_ox_exact = c_bulk * (1.0 - np.exp(-mesh.nodes / mu))
        np.testing.assert_allclose(
            result["Ox"][-1], c_ox_exact, rtol=1e-2, atol=1e-2 * c_bulk
        )

    def test_catalytic_transient_delahay_stiehl_curve(self) -> None:
        """Transient EC' chronoamperometry matches Delahay-Stiehl analytical solution."""
        D = 1e-9
        c_bulk = 1.0
        area = 1e-4
        k_prime = 10.0
        mu = math.sqrt(D / k_prime)
        mesh = GeometricExpandingMesh(0.0, 10.0 * mu, n_points=100, gamma=1.06)

        ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=1)
        red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-1)
        problem = DiffusionProblem(
            grid=mesh,
            species=[ox, red],
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(c_bulk)),
                "Red": (DirichletBC(c_bulk), DirichletBC(0.0)),
            },
            reactions=ReactionNetwork([CatalyticReaction(red, ox, k=k_prime)]),
            geometry=PlanarGeometry(area=area),
        )

        t_eval = np.linspace(0.0, 1.0, 41)
        result = ScipyIVPSolver(method="Radau", rtol=1e-7, atol=1e-9).solve(
            problem, t_span=(0.0, 1.0), t_eval=t_eval
        )
        assert result.success

        # Evaluate at k' * t >= 0.5 (well past initial t=0 step singularity)
        mask = result.t >= 0.05
        kt = k_prime * result.t[mask]
        i_cat = -FARADAY * area * c_bulk * math.sqrt(D * k_prime)
        i_exact = i_cat * (erf(np.sqrt(kt)) + np.exp(-kt) / np.sqrt(np.pi * kt))
        i_sim = result.currents["Ox"][mask]

        rel_errors = np.abs(i_sim - i_exact) / np.abs(i_exact)
        assert np.max(rel_errors) < 0.01

    def test_catalytic_limiting_current_via_mechanism_nernstian(self) -> None:
        """EC' Mechanism with NernstianEquilibrium (n=2) matches I_cat within < 1%."""
        D = 1e-9
        c_bulk = 2.0
        area = 1e-4
        k_prime = 25.0
        n = 2

        mu = math.sqrt(D / k_prime)
        mesh = GeometricExpandingMesh(0.0, 10.0 * mu, n_points=90, gamma=1.06)

        ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=n)
        red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-n)
        ec_prime = Mechanism(
            species=[ox, red],
            reactions=[CatalyticReaction(red, ox, k=k_prime)],
            electrochemical_steps=[
                ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=0.0, n=n))
            ],
            name="EC_prime",
        )
        problem = ec_prime.create_problem(
            mesh, potential=-0.4, geometry=PlanarGeometry(area=area)
        )
        result = ScipyIVPSolver(method="Radau", rtol=1e-7, atol=1e-9).solve(
            problem, t_span=(0.0, 10.0 / k_prime)
        )
        assert result.success

        i_cat_exact = -n * FARADAY * area * c_bulk * math.sqrt(D * k_prime)
        i_sim = float(result.currents["Ox"][-1])
        rel_error = abs(i_sim - i_cat_exact) / abs(i_cat_exact)
        assert rel_error < 0.01


# ---------------------------------------------------------------------------
# 2. Chemical Equilibrium Relaxation Benchmarks
# ---------------------------------------------------------------------------
class TestEquilibriumRelaxationBenchmark:
    """Analytical benchmarks for reversible chemical equilibrium relaxation."""

    @pytest.mark.parametrize(
        ("k_f", "k_b"),
        [
            (2.0, 1.0),
            (50.0, 10.0),
            (5.0, 20.0),
        ],
    )
    def test_reversible_exponential_relaxation_and_equilibrium_ratio(
        self, k_f: float, k_b: float
    ) -> None:
        """A <=> B exponentially relaxes to c_B / c_A = K_eq = k_f / k_b."""
        c_a0, c_b0 = 1.5, 0.5
        c_tot = c_a0 + c_b0
        k_eq = k_f / k_b
        k_relax = k_f + k_b

        a = Species("A", diffusivity=1e-5, bulk_concentration=c_a0)
        b = Species("B", diffusivity=1e-5, bulk_concentration=c_b0)
        network = ReactionNetwork([ReversibleReaction(a, b, k_f=k_f, k_b=k_b)])

        problem = DiffusionProblem(
            grid=Uniform1DMesh(0.0, 1.0, n_points=25),
            species=[a, b],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            reactions=network,
        )

        # Integrate for 20 relaxation time constants (exp(-20) ~ 2e-9)
        t_end = 20.0 / k_relax
        t_eval = np.linspace(0.0, t_end, 51)
        result = ScipyIVPSolver(method="Radau", rtol=1e-9, atol=1e-11).solve(
            problem, t_span=(0.0, t_end), t_eval=t_eval
        )
        assert result.success

        c_a_eq = c_tot / (1.0 + k_eq)
        c_b_eq = k_eq * c_tot / (1.0 + k_eq)
        exp_decay = np.exp(-k_relax * result.t)
        c_a_exact = c_a_eq + (c_a0 - c_a_eq) * exp_decay
        c_b_exact = c_b_eq + (c_b0 - c_b_eq) * exp_decay

        # Exact trajectory match across the entire relaxation curve
        np.testing.assert_allclose(result["A"][:, 0], c_a_exact, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(result["B"][:, 0], c_b_exact, rtol=1e-6, atol=1e-8)

        # Final equilibrium ratio c_B / c_A == K_eq everywhere in space
        ratio_final = result["B"][-1] / result["A"][-1]
        np.testing.assert_allclose(ratio_final, k_eq, rtol=1e-6)

    def test_coupled_diffusion_and_equilibrium_relaxation(self) -> None:
        """Spatially non-uniform A <=> B with unequal D homogenizes to K_eq."""
        k_f, k_b = 12.0, 4.0
        k_eq = k_f / k_b
        mesh = GeometricExpandingMesh(0.0, 1.0, n_points=41, gamma=1.06)

        a = Species("A", diffusivity=0.08, bulk_concentration=1.0)
        b = Species("B", diffusivity=0.05, bulk_concentration=0.0)
        network = ReactionNetwork([ReversibleReaction(a, b, k_f=k_f, k_b=k_b)])

        problem = DiffusionProblem(
            grid=mesh,
            species=[a, b],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            initial_conditions={
                "A": lambda x: 1.0 + 0.5 * np.cos(np.pi * x),
                "B": lambda x: 0.2 * np.sin(np.pi * x),
            },
            reactions=network,
        )

        result = ScipyIVPSolver(method="Radau", rtol=1e-10, atol=1e-12).solve(
            problem, t_span=(0.0, 30.0)
        )
        assert result.success

        # Equilibrium ratio c_B / c_A = K_eq everywhere at steady state
        np.testing.assert_allclose(result["B"][-1] / result["A"][-1], k_eq, rtol=1e-5)

        # Total integrated mass matches final uniform equilibrium concentration
        m0 = float(_integrate_trapz(result["A"][0] + result["B"][0], mesh.nodes))
        c_a_eq = m0 / (1.0 + k_eq)
        c_b_eq = k_eq * m0 / (1.0 + k_eq)
        np.testing.assert_allclose(result["A"][-1], c_a_eq, rtol=1e-4)
        np.testing.assert_allclose(result["B"][-1], c_b_eq, rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. Mass Conservation Across Uniform and Non-Uniform Spatial Meshes
# ---------------------------------------------------------------------------
def _all_spatial_meshes() -> list[tuple[str, Mesh1D]]:
    """Construct representative uniform and non-uniform meshes for conservation tests."""
    return [
        ("uniform", Uniform1DMesh(0.0, 1.0, n_points=35)),
        (
            "geometric_one_sided",
            GeometricExpandingMesh(0.0, 1.0, n_points=35, gamma=1.08),
        ),
        (
            "geometric_two_sided",
            GeometricExpandingMesh(0.0, 1.0, n_points=35, gamma=1.08, two_sided=True),
        ),
        (
            "exponential",
            ExponentialMesh(0.0, 1.0, n_points=35, beta=3.0, mapping="exponential"),
        ),
        (
            "exponential_tanh_two_sided",
            ExponentialMesh(
                0.0, 1.0, n_points=35, beta=3.0, mapping="tanh", two_sided=True
            ),
        ),
        (
            "exponential_sinh",
            ExponentialMesh(0.0, 1.0, n_points=35, beta=3.0, mapping="sinh"),
        ),
    ]


class TestMassConservationAcrossMeshes:
    """Verify machine-precision mass conservation (< 1e-10 relative drift) on all meshes."""

    @pytest.mark.parametrize(
        ("mesh_name", "mesh"),
        _all_spatial_meshes(),
        ids=[name for name, _ in _all_spatial_meshes()],
    )
    def test_multi_species_closed_box_mass_conservation(
        self, mesh_name: str, mesh: Mesh1D
    ) -> None:
        """Total mass int_0^L (c_A + c_B + c_C) dx is conserved to < 1e-10 across meshes."""
        a = Species("A", diffusivity=1e-2, bulk_concentration=1.0)
        b = Species("B", diffusivity=2e-2, bulk_concentration=0.5)
        c = Species("C", diffusivity=5e-3, bulk_concentration=0.2)

        # Stoichiometric network: A <=> B -> C (sum_i nu_i = 0 for every reaction)
        network = ReactionNetwork(
            [
                ReversibleReaction(a, b, k_f=15.0, k_b=5.0),
                FirstOrderReaction(b, c, k=3.0),
            ]
        )

        # Non-uniform initial conditions ensure active diffusion + reaction everywhere
        problem = DiffusionProblem(
            grid=mesh,
            species=[a, b, c],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
                "C": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            initial_conditions={
                "A": lambda x: 0.5 + np.exp(-20.0 * (x - 0.3) ** 2),
                "B": lambda x: 0.3 + 0.4 * np.cos(np.pi * x),
                "C": lambda x: 0.2 * (1.0 + x),
            },
            reactions=network,
        )

        t_eval = np.linspace(0.0, 1.0, 21)
        result = ScipyIVPSolver(method="Radau", rtol=1e-11, atol=1e-13).solve(
            problem, t_span=(0.0, 1.0), t_eval=t_eval
        )
        assert result.success, f"Solver failed on mesh '{mesh_name}': {result.message}"

        total_conc = result["A"] + result["B"] + result["C"]
        total_mass = _integrate_trapz(total_conc, mesh.nodes)
        rel_drift = np.max(np.abs(total_mass - total_mass[0])) / total_mass[0]
        assert rel_drift < 1e-10, (
            f"Mass drift {rel_drift:.3e} >= 1e-10 on mesh '{mesh_name}'"
        )

    @pytest.mark.parametrize(
        ("mesh_name", "mesh"),
        [
            ("uniform", Uniform1DMesh(0.0, 1.0, n_points=35)),
            (
                "geometric",
                GeometricExpandingMesh(0.0, 1.0, n_points=35, gamma=1.08),
            ),
            (
                "exponential",
                ExponentialMesh(0.0, 1.0, n_points=35, beta=2.5, mapping="exponential"),
            ),
        ],
        ids=["uniform", "geometric", "exponential"],
    )
    def test_nonlinear_stoichiometric_mass_conservation(
        self, mesh_name: str, mesh: Mesh1D
    ) -> None:
        """Nonlinear disproportionation (2B -> A + C, sum nu_i = 0) conserves mass to < 1e-10."""
        a = Species("A", diffusivity=1.5e-2, bulk_concentration=0.2)
        b = Species("B", diffusivity=1.0e-2, bulk_concentration=1.0)
        c = Species("C", diffusivity=2.5e-2, bulk_concentration=0.1)

        network = ReactionNetwork([DisproportionationReaction(b, a, c, k=8.0)])
        problem = DiffusionProblem(
            grid=mesh,
            species=[a, b, c],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
                "C": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            initial_conditions={
                "A": lambda x: 0.2 + 0.1 * x,
                "B": lambda x: 1.0 + 0.5 * np.exp(-15.0 * (x - 0.5) ** 2),
                "C": lambda x: 0.1 * (1.0 - x) + 0.05,
            },
            reactions=network,
        )

        result = ScipyIVPSolver(method="Radau", rtol=1e-11, atol=1e-13).solve(
            problem, t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 21)
        )
        assert result.success

        total_mass = _integrate_trapz(
            result["A"] + result["B"] + result["C"], mesh.nodes
        )
        rel_drift = np.max(np.abs(total_mass - total_mass[0])) / total_mass[0]
        assert rel_drift < 1e-10, (
            f"Nonlinear mass drift {rel_drift:.3e} >= 1e-10 on mesh '{mesh_name}'"
        )


# ---------------------------------------------------------------------------
# 4. Butler-Volmer Cyclic Voltammetry Diagnostics
# ---------------------------------------------------------------------------
class TestButlerVolmerVoltammetryDiagnostics:
    """Cyclic voltammetry diagnostics across reversible, quasi-reversible, and irreversible regimes."""

    def test_regime_transition_reversible_to_irreversible(self) -> None:
        """Delta_Ep and I_p transition smoothly from Nernstian reversible to totally irreversible."""
        v = 0.1  # V/s
        alpha = 0.5
        k0_values_cm_s = [10.0, 1.0, 0.05, 0.01, 0.003, 1e-4, 1e-5]

        results = [
            _simulate_butler_volmer_cv(k0, v=v, alpha=alpha) for k0 in k0_values_cm_s
        ]

        # Extract analytical Randles-Sevcik peak currents in SI units
        ref = results[0]
        ip_rev = float(
            randles_sevcik(
                v,
                c_bulk=ref["c_bulk_mol_m3"],
                D=ref["D_m2_s"],
                area=ref["area_m2"],
                n=1,
                T=STANDARD_TEMPERATURE,
            )
        )
        ip_irrev = float(
            randles_sevcik_irreversible(
                v,
                alpha=alpha,
                n_alpha=1,
                c_bulk=ref["c_bulk_mol_m3"],
                D=ref["D_m2_s"],
                area=ref["area_m2"],
                n=1,
                T=STANDARD_TEMPERATURE,
            )
        )

        # 1. Reversible regime (k0 >= 1 cm/s): Delta_Ep ~ 57-59 mV and I_p matches Randles-Sevcik
        for res in results[:2]:
            assert matsuda_ayabe(res["k0_m_s"], v=v, D=res["D_m2_s"]).is_reversible
            assert 55.0 <= res["delta_ep_mV"] <= 61.0, (
                f"Reversible Delta_Ep={res['delta_ep_mV']:.2f} mV outside [55, 61] mV "
                f"for k0={res['k0_cm_s']} cm/s"
            )
            assert abs(res["Ipc"] - ip_rev) / abs(ip_rev) < 0.01

        # 2. Quasi-reversible regime (1e-3 < k0 < 1 cm/s): intermediate Delta_Ep and |I_p|
        for res in results[2:5]:
            ma = matsuda_ayabe(res["k0_m_s"], v=v, D=res["D_m2_s"])
            assert ma.is_quasi_reversible
            assert res["delta_ep_mV"] > 61.0
            assert abs(ip_irrev) < abs(res["Ipc"]) < abs(ip_rev)

        # 3. Irreversible regime (k0 <= 1e-4 cm/s): large Delta_Ep, I_p matches irreversible limit
        for res in results[5:]:
            assert res["delta_ep_mV"] > 400.0
            assert abs(res["Ipc"] - ip_irrev) / abs(ip_irrev) < 0.015

        # 4. Strict monotonicity: decreasing k0 increases Delta_Ep and decreases |I_pc|
        delta_eps = [r["delta_ep_mV"] for r in results]
        peak_mags = [abs(r["Ipc"]) for r in results]
        assert all(
            delta_eps[i] < delta_eps[i + 1] for i in range(len(delta_eps) - 1)
        ), f"Delta_Ep not strictly increasing with decreasing k0: {delta_eps}"
        assert all(
            peak_mags[i] > peak_mags[i + 1] for i in range(len(peak_mags) - 1)
        ), f"|I_pc| not strictly decreasing with decreasing k0: {peak_mags}"

    def test_nicholson_psi_correlation_in_quasi_reversible_regime(self) -> None:
        """Simulated quasi-reversible Delta_Ep quantitatively tracks Nicholson's Psi curve."""
        v = 0.1
        # Values spanning Nicholson's classic tabulated working curve (0.25 < Psi < 5)
        k0_quasi_cm_s = [0.05, 0.02, 0.01, 0.005, 0.003]

        sim_delta_ep_mv: list[float] = []
        nich_delta_ep_mv: list[float] = []
        log_k0_true: list[float] = []
        log_k0_extracted: list[float] = []

        for k0_cms in k0_quasi_cm_s:
            res = _simulate_butler_volmer_cv(k0_cms, v=v, alpha=0.5)
            psi = res["psi"]
            expected_dep_mv = float(
                nicholson_delta_ep(psi, n=1, in_volts=False, method="table")
            )

            sim_delta_ep_mv.append(res["delta_ep_mV"])
            nich_delta_ep_mv.append(expected_dep_mv)

            # Each point agrees with Nicholson's tabulated Delta_Ep within 4 mV
            assert abs(res["delta_ep_mV"] - expected_dep_mv) < 4.0, (
                f"k0={k0_cms} cm/s (Psi={psi:.3f}): sim Delta_Ep={res['delta_ep_mV']:.2f} mV "
                f"vs Nicholson={expected_dep_mv:.2f} mV"
            )

            # Inverse extraction of k0 in the well-conditioned Nicholson window (Psi <= 2,
            # Delta_Ep > 70 mV), away from the flat reversible asymptote where 1 mV shifts
            # Psi by > 20%.
            if psi <= 2.0:
                k0_ext = nicholson_rate_constant(
                    res["delta_ep_V"],
                    v=v,
                    D=res["D_m2_s"],
                    n=1,
                    T=STANDARD_TEMPERATURE,
                    in_volts=True,
                    method="table",
                )
                assert abs(k0_ext - res["k0_m_s"]) / res["k0_m_s"] < 0.15, (
                    f"Extracted k0={k0_ext:.4e} m/s deviates > 15% from true "
                    f"k0={res['k0_m_s']:.4e} m/s"
                )
                log_k0_true.append(math.log10(res["k0_m_s"]))
                log_k0_extracted.append(math.log10(k0_ext))

            # Peak current also agrees closely with Matsuda-Ayabe quasi-reversible estimate
            ip_quasi = float(
                randles_sevcik_quasi(
                    v,
                    k0=res["k0_m_s"],
                    c_bulk=res["c_bulk_mol_m3"],
                    D=res["D_m2_s"],
                    area=res["area_m2"],
                    alpha=0.5,
                    n=1,
                    T=STANDARD_TEMPERATURE,
                )
            )
            assert abs(res["Ipc"] - ip_quasi) / abs(ip_quasi) < 0.05

        # Strong linear correlation between simulated and theoretical Nicholson Delta_Ep
        corr_dep = float(np.corrcoef(sim_delta_ep_mv, nich_delta_ep_mv)[0, 1])
        corr_k0 = float(np.corrcoef(log_k0_true, log_k0_extracted)[0, 1])
        assert corr_dep > 0.999
        assert corr_k0 > 0.999
