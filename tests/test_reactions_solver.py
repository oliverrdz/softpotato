"""Integration tests for DiffusionProblem, ScipyIVPSolver analytical Jacobian, and Mechanisms."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import approx_fprime

from softpotato.constants import FARADAY
from softpotato.geometry import (
    CylindricalGeometry,
    Geometry,
    PlanarGeometry,
    SphericalGeometry,
)
from softpotato.grid import GeometricExpandingMesh
from softpotato.reactions import (
    ButlerVolmer,
    CatalyticReaction,
    CustomMechanism,
    CustomReaction,
    DimerizationReaction,
    ElectrochemicalStep,
    FirstOrderReaction,
    Mechanism,
    NernstianEquilibrium,
    ReactionNetwork,
    ReversibleReaction,
    SecondOrderReaction,
    Species,
    SpeciesCollection,
    create_electrochemical_bcs,
)
from softpotato.solver import (
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    ScipyIVPSolver,
    get_solver,
)


# ---------------------------------------------------------------------------
# 1. DiffusionProblem integration with Species & SpeciesCollection
# ---------------------------------------------------------------------------
class TestDiffusionProblemSpeciesIntegration:
    def test_species_list_auto_populates_metadata(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1.5e-9, bulk_concentration=0.2, n_electrons=-1)
        prod = Species("P", diffusivity=2e-9, bulk_concentration=0.0)

        grid = np.linspace(0.0, 1e-3, 21)
        bcs = {
            "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
            "Red": (DirichletBC(1.0), DirichletBC(0.2)),
            "P": (NeumannBC(0.0), DirichletBC(0.0)),
        }
        problem = DiffusionProblem(
            grid=grid,
            species=[ox, red, prod],
            boundary_conditions=bcs,
        )

        assert problem.species == ["Ox", "Red", "P"]
        assert problem.diffusivity == {"Ox": 1e-9, "Red": 1.5e-9, "P": 2e-9}
        assert problem.initial_conditions == {"Ox": 1.0, "Red": 0.2, "P": 0.0}
        assert problem.n_electrons == {"Ox": 1, "Red": -1}
        assert problem.species_objects["Ox"] is ox

    def test_species_collection_auto_populates_metadata(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=2)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-2)
        col = SpeciesCollection([ox, red])

        problem = DiffusionProblem(
            grid=np.linspace(0.0, 1e-3, 15),
            species=col,
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
                "Red": (DirichletBC(1.0), DirichletBC(0.0)),
            },
        )
        assert problem.species == ["Ox", "Red"]
        assert problem.diffusivity == {"Ox": 1e-9, "Red": 1e-9}
        assert problem.initial_conditions == {"Ox": 1.0, "Red": 0.0}
        assert problem.n_electrons == {"Ox": 2, "Red": -2}

    def test_explicit_overrides_take_precedence_and_do_not_mutate_inputs(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        user_diff = {"Ox": 5e-9}
        user_ic = {"Red": 0.5}
        user_ne = {"Red": 1}

        problem = DiffusionProblem(
            grid=np.linspace(0.0, 1e-3, 11),
            diffusivity=user_diff,
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
                "Red": (DirichletBC(1.0), DirichletBC(0.0)),
            },
            initial_conditions=user_ic,
            species=[ox, red],
            n_electrons=user_ne,
        )

        assert problem.diffusivity == {"Ox": 5e-9, "Red": 1e-9}
        assert problem.initial_conditions == {"Ox": 1.0, "Red": 0.5}
        assert problem.n_electrons == {"Ox": 1, "Red": 1}
        # Original caller dicts must remain untouched
        assert user_diff == {"Ox": 5e-9}
        assert user_ic == {"Red": 0.5}
        assert user_ne == {"Red": 1}

    def test_species_negative_and_zero_n_electrons_solve_cleanly(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        inert = Species("P", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=0)

        grid = np.linspace(0.0, 1e-3, 31)
        network = ReactionNetwork([FirstOrderReaction(red, inert, k=2.0)])
        problem = DiffusionProblem(
            grid=grid,
            species=[ox, red, inert],
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
                "Red": (DirichletBC(1.0), DirichletBC(0.0)),
                "P": (NeumannBC(0.0), DirichletBC(0.0)),
            },
            reactions=network,
        )

        result = ScipyIVPSolver(method="Radau").solve(problem, t_span=(0.0, 0.5))
        assert result.success
        # Cathodic reduction: Ox consumed at x=0 -> negative current;
        # Red produced at x=0 with n_electrons=-1 -> also negative current;
        # Inert species P with n_electrons=0 -> zero Faradaic current.
        assert np.all(result.currents["Ox"][1:] < 0.0)
        assert np.all(result.currents["Red"][1:] < 0.0)
        np.testing.assert_allclose(result.currents["P"], 0.0)

    def test_species_validation_errors(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0)
        with pytest.raises(ValueError, match="Duplicate species"):
            DiffusionProblem(
                grid=np.linspace(0.0, 1.0, 11),
                species=[ox, ox],
                boundary_conditions={"Ox": (NeumannBC(0.0), NeumannBC(0.0))},
            )
        with pytest.raises(TypeError, match="str or Species"):
            DiffusionProblem(
                grid=np.linspace(0.0, 1.0, 11),
                species=[123],  # type: ignore[list-item]
                boundary_conditions={},
            )
        with pytest.raises(TypeError, match="Mechanism"):
            DiffusionProblem(
                grid=np.linspace(0.0, 1.0, 11),
                diffusivity={"A": 1.0},
                boundary_conditions={"A": (NeumannBC(0.0), NeumannBC(0.0))},
                initial_conditions={"A": 1.0},
                mechanism="not_a_mechanism",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 2. Exact Jacobian verification against finite differences of rhs(t, y)
# ---------------------------------------------------------------------------
class TestAnalyticalJacobianVerification:
    @pytest.mark.parametrize(
        "geometry,grid",
        [
            (PlanarGeometry(), np.linspace(0.0, 1.0, 15)),
            (
                PlanarGeometry(),
                GeometricExpandingMesh(0.0, 1.0, n_points=15, gamma=1.15),
            ),
            (SphericalGeometry(r0=0.5), np.linspace(0.5, 2.5, 15)),
            (CylindricalGeometry(r0=0.5), np.linspace(0.5, 2.5, 15)),
        ],
    )
    def test_coupled_jacobian_matches_fd_across_geometries_and_bcs(
        self, geometry, grid
    ) -> None:
        a = Species("A", diffusivity=0.02, bulk_concentration=1.0)
        b = Species("B", diffusivity=0.05, bulk_concentration=0.5)
        c = Species("C", diffusivity=0.03, bulk_concentration=0.2)

        network = ReactionNetwork(
            [
                SecondOrderReaction(a, b, c, k=4.0),
                DimerizationReaction(b, a, k=1.5),
                ReversibleReaction(c, a, k_f=2.0, k_b=0.7),
            ]
        )

        # Deliberately mix Dirichlet and Neumann BCs across species
        bcs = {
            "A": (DirichletBC(0.1), DirichletBC(1.0)),
            "B": (NeumannBC(-0.2), DirichletBC(0.5)),
            "C": (NeumannBC(0.0), NeumannBC(0.1)),
        }
        problem = DiffusionProblem(
            grid=grid,
            species=[a, b, c],
            boundary_conditions=bcs,
            reactions=network,
            geometry=geometry,
        )

        grid_arr = np.asarray(problem.grid, dtype=float)
        n_pts = problem.n_points
        species_names = list(problem.species)

        species_info: dict[str, dict[str, int | bool]] = {}
        offset = 0
        for sp in species_names:
            bc_l, bc_r = problem.boundary_conditions[sp]
            idx_start = 1 if bc_l.btype == "dirichlet" else 0
            idx_end = n_pts - 1 if bc_r.btype == "dirichlet" else n_pts
            size = idx_end - idx_start
            species_info[sp] = {
                "idx_start": idx_start,
                "idx_end": idx_end,
                "size": size,
                "offset": offset,
                "left_dirichlet": bc_l.btype == "dirichlet",
                "right_dirichlet": bc_r.btype == "dirichlet",
            }
            offset += size

        def unpack(y: np.ndarray) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for sp in species_names:
                info = species_info[sp]
                arr = np.empty(n_pts, dtype=float)
                bc_l, bc_r = problem.boundary_conditions[sp]
                s0 = int(info["idx_start"])
                s1 = int(info["idx_end"])
                off = int(info["offset"])
                sz = int(info["size"])
                arr[s0:s1] = y[off : off + sz]
                if info["left_dirichlet"]:
                    arr[0] = bc_l.evaluate(0.3)
                if info["right_dirichlet"]:
                    arr[-1] = bc_r.evaluate(0.3)
                out[sp] = arr
            return out

        def rhs(y: np.ndarray) -> np.ndarray:
            c_dict = unpack(y)
            rxn = network(c_dict, grid_arr, 0.3)
            dy = np.empty(offset, dtype=float)
            for sp in species_names:
                info = species_info[sp]
                bc_l, bc_r = problem.boundary_conditions[sp]
                lap = problem.geometry.laplacian(  # type: ignore[union-attr]
                    c_dict[sp], grid_arr, bc_left=bc_l, bc_right=bc_r, t=0.3
                )
                dcdt = problem.diffusivity[sp] * lap + rxn[sp]
                off = int(info["offset"])
                sz = int(info["size"])
                dy[off : off + sz] = dcdt[int(info["idx_start"]) : int(info["idx_end"])]
            return dy

        rng = np.random.default_rng(42)
        y_test = rng.uniform(0.2, 1.5, size=offset)
        c_dict = unpack(y_test)

        assert isinstance(problem.geometry, Geometry)
        j_diff = ScipyIVPSolver._build_diffusion_jacobian(
            problem.geometry,
            grid_arr,
            problem,
            species_info,
        )
        local_j = network.local_jacobian(c_dict, grid_arr, 0.3, species=species_names)
        j_rxn = ScipyIVPSolver._assemble_reaction_jacobian_from_local(
            local_j, species_names, species_info, n_pts, offset
        )
        j_analytic = (j_diff + j_rxn).toarray()
        j_fd = approx_fprime(y_test, rhs, 1e-7)

        np.testing.assert_allclose(j_analytic, j_fd, rtol=1e-5, atol=1e-6)

    def test_single_homogeneous_reaction_and_matrix_jacobian_protocol(self) -> None:
        a = Species("A", diffusivity=1e-4, bulk_concentration=1.0)
        b = Species("B", diffusivity=1e-4, bulk_concentration=0.0)
        grid = np.linspace(0.0, 1.0, 21)
        bcs = {
            "A": (NeumannBC(0.0), NeumannBC(0.0)),
            "B": (NeumannBC(0.0), NeumannBC(0.0)),
        }

        # Single HomogeneousReaction passed directly
        rxn = FirstOrderReaction(a, b, k=5.0)
        prob_single = DiffusionProblem(
            grid=grid, species=[a, b], boundary_conditions=bcs, reactions=rxn
        )
        res_single = ScipyIVPSolver(method="Radau").solve(
            prob_single, t_span=(0.0, 0.5)
        )
        assert res_single.success
        assert res_single.raw_output.njev > 0
        np.testing.assert_allclose(res_single["A"][-1], np.exp(-2.5), rtol=1e-4)

        # Custom object implementing only .jacobian(c, x, t, species=...) -> sparse matrix
        class MatrixJacReaction:
            def __call__(self, c, x, t):
                return {"A": -5.0 * c["A"], "B": 5.0 * c["A"]}

            def jacobian(self, c, x, t, species=None):
                n = len(x)
                d = np.full(n, 5.0)
                zeros = sparse.csc_matrix((n, n), dtype=float)
                return sparse.bmat(
                    [
                        [sparse.diags(-d), zeros],
                        [sparse.diags(d), zeros],
                    ],
                    format="csc",
                )

        prob_mat = DiffusionProblem(
            grid=grid,
            species=[a, b],
            boundary_conditions=bcs,
            reactions=MatrixJacReaction(),
        )
        res_mat = ScipyIVPSolver(method="BDF").solve(prob_mat, t_span=(0.0, 0.5))
        assert res_mat.success
        assert res_mat.raw_output.njev > 0
        np.testing.assert_allclose(res_mat["A"][-1], np.exp(-2.5), rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. Stiff reaction integration, convergence, and analytical Jacobian speedup
# ---------------------------------------------------------------------------
class TestStiffReactionSolver:
    @pytest.mark.parametrize("method", ["Radau", "BDF", "LSODA"])
    def test_stiff_catalytic_mechanism_convergence(self, method: str) -> None:
        """Stiff EC' catalytic mechanism (k' = 1e3 s^-1) converges to analytical limiting current."""
        D = 1e-9  # m^2/s
        c_bulk = 1.0  # mol/m^3
        k_cat = 1e3  # s^-1 (stiff!)
        area = 1e-4  # m^2

        # Reaction layer thickness mu = sqrt(D / k_cat) = 1e-6 m
        # Expanding mesh resolves the thin reaction layer near x=0
        mesh = GeometricExpandingMesh(0.0, 1e-4, n_points=80, gamma=1.08)
        ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=1)
        red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-1)
        network = ReactionNetwork([CatalyticReaction(red, ox, k=k_cat)])

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

        solver = ScipyIVPSolver(method=method, rtol=1e-6, atol=1e-8)
        res = solver.solve(problem, t_span=(0.0, 0.05))
        assert res.success
        assert res.raw_output.njev > 0

        # Analytical steady-state catalytic current: I_cat = -n * F * A * c* * sqrt(D * k')
        i_cat_exact = -1.0 * FARADAY * area * c_bulk * np.sqrt(D * k_cat)
        i_num = res.currents["Ox"][-1]
        np.testing.assert_allclose(i_num, i_cat_exact, rtol=0.02)

    @pytest.mark.parametrize("method", ["Radau", "BDF"])
    def test_analytical_jacobian_speedup_and_fewer_evals_vs_fd(
        self, method: str
    ) -> None:
        """Analytical Jacobian drastically reduces nfev and wall-clock time on stiff systems."""
        D = 1e-9
        k_cat = 5e3
        mesh = GeometricExpandingMesh(0.0, 1e-4, n_points=70, gamma=1.08)

        ox = Species("Ox", diffusivity=D, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-1)
        prod = Species("P", diffusivity=D, bulk_concentration=0.0, n_electrons=0)

        network = ReactionNetwork(
            [
                CatalyticReaction(red, ox, k=k_cat),
                FirstOrderReaction(red, prod, k=50.0),
            ]
        )

        problem = DiffusionProblem(
            grid=mesh,
            species=[ox, red, prod],
            boundary_conditions={
                "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
                "Red": (DirichletBC(1.0), DirichletBC(0.0)),
                "P": (NeumannBC(0.0), DirichletBC(0.0)),
            },
            reactions=network,
        )

        t0 = time.perf_counter()
        res_analytic = ScipyIVPSolver(
            method=method, rtol=1e-6, atol=1e-8, use_jacobian=True
        ).solve(problem, t_span=(0.0, 0.02))
        dt_analytic = time.perf_counter() - t0

        t1 = time.perf_counter()
        res_fd = ScipyIVPSolver(
            method=method, rtol=1e-6, atol=1e-8, use_jacobian=False
        ).solve(problem, t_span=(0.0, 0.02))
        dt_fd = time.perf_counter() - t1

        assert res_analytic.success
        assert res_fd.success

        # Solutions must match closely
        np.testing.assert_allclose(
            res_analytic["Ox"], res_fd["Ox"], rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            res_analytic["Red"], res_fd["Red"], rtol=1e-3, atol=1e-4
        )

        # Analytical Jacobian must be significantly faster than full FD Jacobian
        assert res_analytic.raw_output.njev > 0
        assert dt_analytic < dt_fd

    def test_stiff_reversible_equilibrium_and_mass_conservation(self) -> None:
        a = Species("A", diffusivity=1e-5, bulk_concentration=1.0)
        b = Species("B", diffusivity=1e-5, bulk_concentration=0.0)
        network = ReactionNetwork([ReversibleReaction(a, b, k_f=5e3, k_b=1e3)])

        problem = DiffusionProblem(
            grid=np.linspace(0.0, 1.0, 31),
            species=[a, b],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            reactions=network,
        )
        res = get_solver("scipy_ivp", method="Radau", rtol=1e-7, atol=1e-9).solve(
            problem, t_span=(0.0, 0.01)
        )
        assert res.success
        # Equilibrium ratio c_B / c_A = K = 5.0 and total concentration = 1.0
        np.testing.assert_allclose(res["A"][-1], 1.0 / 6.0, rtol=1e-5)
        np.testing.assert_allclose(res["B"][-1], 5.0 / 6.0, rtol=1e-5)
        np.testing.assert_allclose(res["A"] + res["B"], 1.0, atol=1e-8)

    def test_explicit_rk45_does_not_warn_about_unused_jac(self) -> None:
        a = Species("A", diffusivity=1e-4, bulk_concentration=1.0)
        b = Species("B", diffusivity=1e-4, bulk_concentration=0.0)
        network = ReactionNetwork([FirstOrderReaction(a, b, k=1.0)])
        problem = DiffusionProblem(
            grid=np.linspace(0.0, 1.0, 11),
            species=[a, b],
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            reactions=network,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ScipyIVPSolver(method="RK45").solve(problem, t_span=(0.0, 0.1))
        assert res.success
        jac_warnings = [w for w in caught if "jac" in str(w.message)]
        assert jac_warnings == []


# ---------------------------------------------------------------------------
# 4. Mechanism & CustomMechanism tests
# ---------------------------------------------------------------------------
class TestCustomMechanism:
    def test_mechanism_ec_create_problem_and_solve(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        prod = Species("P", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=0)

        ec = Mechanism(
            species=[ox, red, prod],
            reactions=[FirstOrderReaction(red, prod, k=100.0)],
            electrochemical_steps=[
                ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=0.0, n=1))
            ],
            name="EC",
        )
        grid = GeometricExpandingMesh(0.0, 1e-3, n_points=45, gamma=1.08)
        problem = ec.create_problem(grid, potential=-0.3)

        assert problem.species == ["Ox", "Red", "P"]
        assert problem.mechanism is ec
        res = ScipyIVPSolver(method="Radau").solve(problem, t_span=(0.0, 0.2))
        assert res.success
        assert res.raw_output.njev > 0
        # Red generated at electrode converts to P
        assert res["P"][-1, 0] > 0.1
        np.testing.assert_allclose(res.currents["P"], 0.0)

    def test_mechanism_passed_directly_to_diffusion_problem(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        mech = Mechanism(
            species=[ox, red],
            reactions=[CatalyticReaction(red, ox, k=50.0)],
            electrochemical_steps=[
                ElectrochemicalStep(ox, red, ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0))
            ],
        )
        bcs = create_electrochemical_bcs(
            NernstianEquilibrium(E0=0.0, n=1), potential=-0.3, ox=ox, red=red
        )
        problem = DiffusionProblem(
            grid=np.linspace(0.0, 1e-3, 31),
            mechanism=mech,
            boundary_conditions=bcs,
        )
        res = ScipyIVPSolver(method="BDF").solve(problem, t_span=(0.0, 0.1))
        assert res.success
        assert res.raw_output.njev > 0

    def test_custom_mechanism_with_custom_reaction_nonlinear(self) -> None:
        """CustomMechanism using CustomReaction (Michaelis-Menten kinetics)."""
        s = Species("S", diffusivity=1e-5, bulk_concentration=2.0)
        p = Species("P", diffusivity=1e-5, bulk_concentration=0.0)
        vmax, km = 10.0, 0.5

        mm_rxn = CustomReaction(
            stoichiometry={s: -1, p: 1},
            rate=lambda c, x, t: vmax * c["S"] / (km + c["S"]),
            name="Michaelis-Menten",
        )
        mech = CustomMechanism(species=[s, p], reactions=[mm_rxn], name="Enzyme")
        problem = mech.create_problem(np.linspace(0.0, 1.0, 21), bulk_bc="neumann")
        res = ScipyIVPSolver(method="Radau").solve(problem, t_span=(0.0, 0.5))
        assert res.success
        assert res.raw_output.njev > 0
        # Total mass S + P is strictly conserved at 2.0
        np.testing.assert_allclose(res["S"] + res["P"], 2.0, atol=1e-6)
        assert np.all(res["S"][-1] < res["S"][0])
        assert np.all(res["P"][-1] > 0.0)

    def test_custom_mechanism_with_raw_rates_auto_fd_jacobian(self) -> None:
        """CustomMechanism with raw rates callable gets automatic vectorized local-FD Jacobian."""
        a = Species("A", diffusivity=1e-5, bulk_concentration=1.0)
        b = Species("B", diffusivity=1e-5, bulk_concentration=0.0)
        k = 200.0

        rate_calls = 0

        def counted_rates(c, x, t):
            nonlocal rate_calls
            rate_calls += 1
            return {
                "A": -k * c["A"] ** 2,
                "B": +k * c["A"] ** 2,
            }

        mech = CustomMechanism(
            species=[a, b],
            rates=counted_rates,
            name="SecondOrderCustom",
        )
        grid = np.linspace(0.0, 1.0, 41)
        problem = mech.create_problem(grid, bulk_bc="neumann")

        rate_calls = 0
        res_jac = ScipyIVPSolver(method="Radau", use_jacobian=True).solve(
            problem, t_span=(0.0, 0.1)
        )
        calls_jac = rate_calls

        rate_calls = 0
        res_fd = ScipyIVPSolver(method="Radau", use_jacobian=False).solve(
            problem, t_span=(0.0, 0.1)
        )
        calls_fd = rate_calls

        assert res_jac.success and res_fd.success
        # Analytical closed-form for dA/dt = -k A^2, A(0) = 1: A(t) = 1 / (1 + k t)
        expected_a = 1.0 / (1.0 + k * 0.1)
        np.testing.assert_allclose(res_jac["A"][-1], expected_a, rtol=1e-4)
        np.testing.assert_allclose(res_jac["A"], res_fd["A"], rtol=1e-4, atol=1e-5)
        assert res_jac.raw_output.njev > 0
        # Vectorized local FD (2*S calls/jac) uses far fewer rate evaluations than full FD (S*N calls/jac)
        assert calls_jac < calls_fd * 0.6

    def test_custom_mechanism_with_explicit_analytical_local_jacobian(self) -> None:
        a = Species("A", diffusivity=1e-5, bulk_concentration=1.0)
        b = Species("B", diffusivity=1e-5, bulk_concentration=0.0)
        k = 500.0

        def rates(c, x, t):
            r = k * c["A"] ** 2
            return {"A": -r, "B": r}

        def local_jac(c, x, t, species):
            n = len(c["A"])
            j = np.zeros((2, 2, n), dtype=float)
            ia = species.index("A")
            ib = species.index("B")
            j[ia, ia] = -2.0 * k * c["A"]
            j[ib, ia] = +2.0 * k * c["A"]
            return j

        mech = CustomMechanism(
            species=[a, b],
            rates=rates,
            local_jacobian_func=local_jac,
            name="ExactCustom",
        )
        problem = mech.create_problem(np.linspace(0.0, 1.0, 31), bulk_bc="neumann")
        res = ScipyIVPSolver(method="BDF").solve(problem, t_span=(0.0, 0.1))
        assert res.success
        assert res.raw_output.njev > 0
        expected_a = 1.0 / (1.0 + k * 0.1)
        np.testing.assert_allclose(res["A"][-1], expected_a, rtol=1e-4)

    def test_custom_mechanism_with_custom_bc_builder(self) -> None:
        a = Species("A", diffusivity=1e-5, bulk_concentration=1.0)
        mech = CustomMechanism(
            species=[a],
            boundary_conditions_func=lambda potential, **kw: {
                "A": (DirichletBC(0.0 if potential < 0 else 1.0), DirichletBC(1.0))
            },
        )
        prob = mech.create_problem(np.linspace(0.0, 1.0, 21), potential=-0.5)
        assert prob.boundary_conditions["A"][0].evaluate(0.0) == 0.0
        res = ScipyIVPSolver(method="Radau").solve(prob, t_span=(0.0, 0.1))
        assert res.success

    def test_mechanism_subclass_auto_fd_jacobian(self) -> None:
        """Subclassing Mechanism and overriding only evaluate() gets sparse local-FD Jacobian."""

        class AutocatalyticMechanism(Mechanism):
            def __init__(self, k: float) -> None:
                a = Species("A", diffusivity=1e-5, bulk_concentration=1.0)
                b = Species("B", diffusivity=1e-5, bulk_concentration=0.1)
                super().__init__(species=[a, b], name="Autocatalytic")
                self.k = k

            def evaluate(self, c, x, t):
                r = self.k * c["A"] * c["B"]
                return {"A": -r, "B": r}

        mech = AutocatalyticMechanism(k=50.0)
        grid = np.linspace(0.0, 1.0, 25)
        problem = mech.create_problem(grid, bulk_bc="neumann")
        res = ScipyIVPSolver(method="Radau").solve(problem, t_span=(0.0, 0.2))
        assert res.success
        assert res.raw_output.njev > 0
        # Mass conservation A + B = 1.1
        np.testing.assert_allclose(res["A"] + res["B"], 1.1, atol=1e-6)
        assert np.all(res["B"][-1] > res["B"][0])

    def test_mechanism_validation_errors(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0)

        with pytest.raises(ValueError, match="at least one Species"):
            Mechanism(species=[])

        with pytest.raises(ValueError, match="must differ"):
            ElectrochemicalStep(ox, ox, NernstianEquilibrium())

        with pytest.raises(TypeError, match="HeterogeneousKinetics"):
            ElectrochemicalStep(ox, red, "invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="not present in mechanism"):
            Mechanism(
                species=[ox],
                electrochemical_steps=[
                    ElectrochemicalStep(ox, red, NernstianEquilibrium())
                ],
            )

        mech_with_step = Mechanism(
            species=[ox, red],
            electrochemical_steps=[
                ElectrochemicalStep(ox, red, NernstianEquilibrium())
            ],
        )
        with pytest.raises(ValueError, match="potential must be provided"):
            mech_with_step.create_problem(np.linspace(0.0, 1.0, 11))

        with pytest.raises(ValueError, match="without rates"):
            CustomMechanism(
                species=[ox],
                local_jacobian_func=lambda c, x, t, sp: np.zeros((1, 1, len(x))),
            )
