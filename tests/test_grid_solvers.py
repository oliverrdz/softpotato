"""Validation, order-of-convergence, and performance benchmarks for expanding grids.

Tests and benchmarks for Issue #47:
1. Analytical ground-truth benchmarks (planar Cottrell, spherical Cottrell, cylindrical,
   Fourier sine decay) on GeometricExpandingMesh and ExponentialMesh across all solvers.
2. Spatial order-of-convergence verification O(dx^2) on non-uniform and expanding grids
   for both concentration fields and interfacial fluxes.
3. Computational efficiency benchmarks comparing runtime, memory scaling, and L_2 / relative
   error norms of GeometricExpandingMesh vs Uniform1DMesh for Cottrell chronoamperometry
   and cyclic voltammetry.
4. Total mass conservation on expanding grids under zero-flux Neumann boundary conditions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

import softpotato as sp
from softpotato.analytical import (
    ConvergenceResult,
    GridComparisonEntry,
    GridComparisonResult,
    compute_error_metrics,
    cottrell,
    cottrell_cylinder,
    cottrell_spherical,
    estimate_convergence_order,
    list_benchmark_cases,
    microsphere_limiting_current,
    run_benchmark,
    run_cottrell_benchmark,
    run_fourier_decay_benchmark,
    run_grid_efficiency_benchmark,
    run_mass_conservation_benchmark,
    run_spherical_cottrell_benchmark,
    step_concentration_profile,
    verify_spatial_convergence,
)
from softpotato.geometry import CylindricalGeometry, PlanarGeometry, SphericalGeometry
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Mesh1D,
)
from softpotato.solver import (
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    get_solver,
)

# ==============================================================================
# 1. Analytical Ground-Truth Benchmarks on Expanding Grids
# ==============================================================================


class TestAnalyticalGroundTruthExpandingGrids:
    """Validate numerical solvers on expanding grids against exact analytical equations."""

    @pytest.mark.parametrize(
        "solver_name",
        ["scipy_ivp", "crank_nicolson", "implicit", "explicit"],
    )
    @pytest.mark.parametrize(
        "mesh_builder",
        [
            lambda D, t_max: GeometricExpandingMesh.from_diffusion_layer(
                D, t_max, n_points=55, gamma=1.08
            ),
            lambda D, t_max: ExponentialMesh.from_diffusion_layer(
                D, t_max, n_points=55, beta=3.8, mapping="exponential"
            ),
            lambda D, t_max: ExponentialMesh.from_diffusion_layer(
                D, t_max, n_points=55, beta=3.2, mapping="tanh"
            ),
            lambda D, t_max: ExponentialMesh.from_diffusion_layer(
                D, t_max, n_points=55, beta=3.8, mapping="sinh"
            ),
        ],
        ids=["geometric", "exp_standard", "exp_tanh", "exp_sinh"],
    )
    def test_planar_cottrell_transient_and_profile(
        self, solver_name: str, mesh_builder
    ) -> None:
        """Verify planar Cottrell current and concentration profile on expanding meshes."""
        D = 1.0e-9
        c_bulk = 1.0
        t_max = 1.0
        mesh = mesh_builder(D, t_max)

        problem = DiffusionProblem(
            grid=mesh,
            diffusivity={"Ox": D},
            boundary_conditions={"Ox": (DirichletBC(0.0), DirichletBC(c_bulk))},
            initial_conditions={"Ox": c_bulk},
            n_electrons=1,
        )

        t_eval = np.linspace(0.04, t_max, 20)
        solver_kwargs = (
            {"dt": 1e-3} if solver_name in ("crank_nicolson", "implicit") else {}
        )
        solver = get_solver(solver_name, **solver_kwargs)
        result = solver.solve(problem, t_span=(0.0, t_max), t_eval=t_eval)

        assert result.success
        assert result.mesh is mesh

        # Verify current against planar Cottrell equation
        i_num = np.abs(result.currents["Ox"])
        i_exact = cottrell(t_eval, n=1, area=1.0, D=D, c_bulk=c_bulk)
        curr_metrics = compute_error_metrics(i_num, i_exact)

        rel_tol = 0.012 if solver_name == "implicit" else 0.004
        assert curr_metrics.max_rel_error < rel_tol, (
            f"{solver_name} on {mesh} exceeded current rel_error tolerance: "
            f"{curr_metrics.max_rel_error * 100:.4f}%"
        )

        # Verify spatial concentration profile against erfc analytical solution
        c_num = result["Ox"]
        c_exact = np.empty_like(c_num)
        for k, t_k in enumerate(t_eval):
            c_exact[k, :] = step_concentration_profile(
                mesh.nodes, t=float(t_k), D=D, c_bulk=c_bulk
            )
        conc_metrics = compute_error_metrics(c_num, c_exact)
        assert conc_metrics.max_abs_error < 0.015

    @pytest.mark.parametrize(
        "solver_name",
        ["scipy_ivp", "crank_nicolson", "implicit", "explicit"],
    )
    @pytest.mark.parametrize("mesh_kind", ["geometric", "exponential"])
    def test_spherical_cottrell_transient_and_profile(
        self, solver_name: str, mesh_kind: str
    ) -> None:
        """Verify spherical Cottrell step response on GeometricExpandingMesh and ExponentialMesh."""
        D = 1.0e-9
        c_bulk = 1.0
        r0 = 50.0e-6
        t_max = 1.0

        solver_kwargs = (
            {"dt": 1e-3} if solver_name in ("crank_nicolson", "implicit") else {}
        )
        res = run_spherical_cottrell_benchmark(
            solver_name,
            n_points=60,
            mesh_kind=mesh_kind,  # type: ignore[arg-type]
            gamma=1.07,
            beta=3.8,
            r0=r0,
            D=D,
            c_bulk=c_bulk,
            t_span=(0.04, t_max),
            n_times=20,
            max_rel_error_threshold=0.015 if solver_name == "implicit" else 0.005,
            **solver_kwargs,
        )

        assert res.passed
        assert res.case_name == "spherical_cottrell"
        assert res.details["is_uniform"] is False
        assert res.flux_metrics is not None
        assert res.concentration_metrics.max_abs_error < 0.015

    def test_hemispherical_and_steady_state_spherical_on_expanding_grid(self) -> None:
        """Verify hemispherical area scaling and long-time microsphere steady-state convergence."""
        D = 1.0e-9
        c_bulk = 1.0
        r0 = 10.0e-6
        t_max = 15.0

        # 1. Hemispherical Cottrell transient
        res_hemi = run_spherical_cottrell_benchmark(
            "scipy_ivp",
            n_points=65,
            mesh_kind="geometric",
            gamma=1.06,
            r0=r0,
            D=D,
            c_bulk=c_bulk,
            hemisphere=True,
            t_span=(0.02, 1.0),
            max_rel_error_threshold=2e-3,
        )
        assert res_hemi.passed
        assert res_hemi.flux_metrics is not None
        assert res_hemi.flux_metrics.max_rel_error < 2e-3

        # 2. Long-time convergence toward steady-state microsphere_limiting_current
        mesh = GeometricExpandingMesh.from_diffusion_layer(
            D=D, t_max=t_max, x_min=r0, factor=8.0, n_points=80, gamma=1.06
        )
        geom = SphericalGeometry(r0=r0)
        problem = DiffusionProblem(
            grid=mesh,
            diffusivity={"c": D},
            boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
            initial_conditions={"c": c_bulk},
            geometry=geom,
            n_electrons=1,
        )
        res = get_solver("scipy_ivp", rtol=1e-8, atol=1e-10).solve(
            problem, t_span=(0.0, t_max), t_eval=np.array([1.0, t_max])
        )
        assert res.success
        i_num_end = float(np.abs(res.currents["c"][-1]))
        i_ana_end = float(
            cottrell_spherical(t_max, r=r0, n=1, D=D, c_bulk=c_bulk, area=geom.area)
        )
        i_ss = float(microsphere_limiting_current(r0, n=1, D=D, c_bulk=c_bulk))

        assert abs(i_num_end - i_ana_end) / i_ana_end < 2e-3
        assert abs(i_num_end - i_ss) / i_ss < 0.05

    def test_cylindrical_cottrell_on_expanding_grid(self) -> None:
        """Verify cylindrical step response on GeometricExpandingMesh against cottrell_cylinder."""
        D = 1.0e-9
        c_bulk = 1.0
        r0 = 25.0e-6
        length = 1.0e-2
        t_max = 0.5
        t_eval = np.linspace(0.02, t_max, 15)

        mesh = GeometricExpandingMesh.from_diffusion_layer(
            D=D, t_max=t_max, x_min=r0, n_points=70, gamma=1.06
        )
        geom = CylindricalGeometry(r0=r0, length=length)
        problem = DiffusionProblem(
            grid=mesh,
            diffusivity={"c": D},
            boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
            initial_conditions={"c": c_bulk},
            geometry=geom,
            n_electrons=1,
        )
        res = get_solver("scipy_ivp", rtol=1e-8, atol=1e-10).solve(
            problem, t_span=(0.0, t_max), t_eval=t_eval
        )
        assert res.success
        i_num = np.abs(res.currents["c"])
        i_ana = cottrell_cylinder(
            t_eval, r0=r0, n=1, D=D, c_bulk=c_bulk, area=geom.area
        )
        metrics = compute_error_metrics(i_num, i_ana)
        assert metrics.l2_error < 0.01
        assert metrics.max_rel_error < 0.012

    def test_benchmark_runners_with_expanding_grids(self) -> None:
        """Verify run_fourier_decay_benchmark, run_cottrell_benchmark, and run_benchmark with expanding grids."""
        assert "spherical_cottrell" in list_benchmark_cases()

        mesh_f = GeometricExpandingMesh(
            0.0, 1.0, n_points=51, gamma=1.04, two_sided=True
        )
        res_f = run_fourier_decay_benchmark("crank_nicolson", grid=mesh_f, t_end=0.1)
        assert res_f.passed
        assert res_f.details["is_uniform"] is False

        res_c = run_cottrell_benchmark(
            "scipy_ivp",
            n_points=55,
            mesh_kind="geometric",
            gamma=1.07,
            t_span=(0.02, 0.1),
        )
        assert res_c.passed
        assert res_c.details["is_uniform"] is False
        assert res_c.flux_metrics is not None
        assert res_c.flux_metrics.max_rel_error < 0.01

        res_s = run_benchmark("spherical_cottrell", "scipy_ivp", n_points=55)
        assert res_s.passed
        assert res_s.case_name == "spherical_cottrell"


# ==============================================================================
# 2. Order-of-Convergence Verification O(dx^2) on Expanding & Non-Uniform Grids
# ==============================================================================


class TestSpatialOrderOfConvergenceExpandingGrids:
    """Verify second-order O(dx^2) convergence of operators and PDE solvers on non-uniform grids."""

    @pytest.mark.parametrize(
        "mesh_factory",
        [
            lambda n: GeometricExpandingMesh(
                0.0, 1.0, n_points=n, gamma=4.0 ** (1.0 / (n - 1))
            ),
            lambda n: GeometricExpandingMesh(
                0.0,
                1.0,
                n_points=n,
                gamma=4.0 ** (1.0 / ((n - 1) // 2)),
                two_sided=True,
            ),
            lambda n: ExponentialMesh(0.0, 1.0, n_points=n, beta=1.5),
            lambda n: ExponentialMesh.from_tanh(0.0, 1.0, n_points=n, beta=1.5),
            lambda n: ExponentialMesh.from_sinh(0.0, 1.0, n_points=n, beta=1.5),
        ],
        ids=["geom_1sided", "geom_2sided", "exp_std", "exp_tanh", "exp_sinh"],
    )
    def test_planar_operator_and_flux_second_order_convergence(
        self, mesh_factory
    ) -> None:
        """Verify 2nd-order O(dx^2) convergence of non-uniform Laplacian and surface flux stencils."""
        geom = PlanarGeometry()
        n_list = (41, 81, 161, 321)
        spacings: list[float] = []
        lap_errors: list[float] = []
        flux_left_errors: list[float] = []
        flux_right_errors: list[float] = []

        D = 0.05
        for n in n_list:
            mesh = mesh_factory(n)
            x = mesh.nodes
            c = np.sin(np.pi * x)
            lap_exact = -(np.pi**2) * np.sin(np.pi * x)
            j_left_exact = -D * np.pi
            j_right_exact = D * (-np.pi)

            lap_num = geom.laplacian(
                c, mesh, bc_left=DirichletBC(0.0), bc_right=DirichletBC(0.0)
            )
            j_left_num = float(geom.surface_flux(c, mesh, diffusivity=D, side="left"))
            j_right_num = float(geom.surface_flux(c, mesh, diffusivity=D, side="right"))

            spacings.append(1.0 / (n - 1))
            lap_errors.append(
                float(np.sqrt(np.mean((lap_num[1:-1] - lap_exact[1:-1]) ** 2)))
            )
            flux_left_errors.append(abs(j_left_num - j_left_exact))
            flux_right_errors.append(abs(j_right_num - j_right_exact))

        p_lap, r2_lap = estimate_convergence_order(spacings, lap_errors)
        p_fl, r2_fl = estimate_convergence_order(spacings, flux_left_errors)
        p_fr, r2_fr = estimate_convergence_order(spacings, flux_right_errors)

        assert p_lap >= 1.95 and r2_lap >= 0.999
        assert p_fl >= 1.95 and r2_fl >= 0.999
        assert p_fr >= 1.95 and r2_fr >= 0.999

    @pytest.mark.parametrize("geometry_kind", ["spherical", "cylindrical"])
    def test_curvilinear_operator_second_order_convergence_on_expanding_grid(
        self, geometry_kind: str
    ) -> None:
        """Verify 2nd-order O(dr^2) convergence of curvilinear radial Laplacian on GeometricExpandingMesh."""
        r0 = 1.0
        r_max = 2.0
        geom = (
            SphericalGeometry(r0=r0)
            if geometry_kind == "spherical"
            else CylindricalGeometry(r0=r0)
        )
        p_sym = geom.p

        n_list = (21, 41, 81, 161)
        spacings: list[float] = []
        lap_errors: list[float] = []

        for n in n_list:
            gamma_n = 3.5 ** (1.0 / (n - 1))
            mesh = GeometricExpandingMesh(r0, r_max, n_points=n, gamma=gamma_n)
            r = mesh.nodes
            # Smooth radial test function c(r) = sin(pi * (r - r0))
            arg = np.pi * (r - r0)
            c = np.sin(arg)
            dc_dr = np.pi * np.cos(arg)
            d2c_dr2 = -(np.pi**2) * np.sin(arg)
            lap_exact = d2c_dr2 + (p_sym / r) * dc_dr

            lap_num = geom.laplacian(
                c, mesh, bc_left=DirichletBC(0.0), bc_right=DirichletBC(0.0)
            )
            spacings.append((r_max - r0) / (n - 1))
            lap_errors.append(float(np.max(np.abs(lap_num[1:-1] - lap_exact[1:-1]))))

        order, r2 = estimate_convergence_order(spacings, lap_errors)
        assert order >= 1.95
        assert r2 >= 0.999

    @pytest.mark.parametrize("mesh_kind", ["geometric", "exponential"])
    @pytest.mark.parametrize("two_sided", [False, True])
    @pytest.mark.parametrize("target", ["concentration", "flux"])
    def test_verify_spatial_convergence_expanding_meshes(
        self, mesh_kind: str, two_sided: bool, target: str
    ) -> None:
        """Verify PDE solver spatial O(dx^2) convergence on geometric and exponential meshes."""
        study = verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind=mesh_kind,  # type: ignore[arg-type]
            expansion_span=4.0,
            beta=1.5,
            two_sided=two_sided,
            target=target,  # type: ignore[arg-type]
            D=0.05,
            L=1.0,
            t_fixed=0.1,
            dt_fine=2e-5,
            expected_order=2.0,
            order_tolerance=0.2,
        )

        assert isinstance(study, ConvergenceResult)
        assert study.passed
        assert study.observed_order >= 1.85
        assert study.r_squared >= 0.995

    @pytest.mark.parametrize("mapping", ["tanh", "sinh"])
    def test_verify_spatial_convergence_conformal_mappings(self, mapping: str) -> None:
        """Verify PDE solver O(dx^2) spatial convergence on tanh and sinh conformal meshes."""
        study = verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="exponential",
            beta=1.5,
            mapping=mapping,
            dt_fine=2e-5,
            expected_order=2.0,
            order_tolerance=0.2,
        )
        assert study.passed
        assert study.observed_order >= 1.85
        assert study.r_squared >= 0.995

    def test_spherical_pde_spatial_convergence_on_expanding_grid(self) -> None:
        """Verify O(dr^2) spatial convergence of full spherical PDE integration on GeometricExpandingMesh.

        Uses the exact spherical eigenmode solution for u(r, t) = r * c(r, t):
        c(r, t) = (r0 / r) * sin(pi * (r - r0) / L) * exp(-D * (pi / L)^2 * t).
        """
        r0 = 1.0
        L = 1.0
        r_max = r0 + L
        D = 0.05
        t_fixed = 0.1
        decay = math.exp(-D * (math.pi / L) ** 2 * t_fixed)
        geom = SphericalGeometry(r0=r0)

        n_list = (21, 41, 81, 161)
        spacings: list[float] = []
        errors: list[float] = []

        for n in n_list:
            gamma_n = 3.5 ** (1.0 / (n - 1))
            mesh = GeometricExpandingMesh(r0, r_max, n_points=n, gamma=gamma_n)
            problem = DiffusionProblem(
                grid=mesh,
                diffusivity={"c": D},
                boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
                initial_conditions={
                    "c": lambda r_pts: (r0 / r_pts) * np.sin(np.pi * (r_pts - r0) / L)
                },
                geometry=geom,
            )
            res = get_solver("crank_nicolson", dt=2e-5).solve(
                problem,
                t_span=(0.0, t_fixed),
                t_eval=np.array([0.0, t_fixed]),
            )
            assert res.success
            c_exact = (r0 / mesh.nodes) * np.sin(np.pi * (mesh.nodes - r0) / L) * decay
            m = compute_error_metrics(res["c"][-1, :], c_exact)
            spacings.append(L / (n - 1))
            errors.append(m.max_abs_error)

        order, r2 = estimate_convergence_order(spacings, errors)
        assert order >= 1.90
        assert r2 >= 0.999


# ==============================================================================
# 3. Performance & Efficiency Benchmarks (GeometricExpandingMesh vs Uniform1DMesh)
# ==============================================================================


class TestPerformanceAndEfficiencyBenchmarks:
    """Verify expanding grids achieve < 0.1% relative error with 5x-10x fewer points than uniform grids."""

    def test_planar_cottrell_efficiency_benchmark(self) -> None:
        """Verify GeometricExpandingMesh (N=60) achieves <0.1% error vs Uniform1DMesh (N=60, 300, 600)."""
        comp = run_grid_efficiency_benchmark(
            case="cottrell",
            solver="scipy_ivp",
            expanding_points=60,
            uniform_points_list=(60, 300, 600),
            gamma=1.08,
            beta=4.2,
            t_span=(0.01, 1.0),
            n_times=25,
            target_rel_error=1e-3,
        )

        assert isinstance(comp, GridComparisonResult)
        assert comp.passed
        assert comp.point_reduction_factor == 10.0
        assert comp.memory_reduction_factor == 10.0
        assert comp.speedup_factor > 1.0

        geom_entry = comp.entries[0]
        exp_entry = comp.entries[1]
        uni_60 = comp.entries[2]
        uni_300 = comp.entries[3]
        uni_600 = comp.entries[4]

        assert isinstance(geom_entry, GridComparisonEntry)
        assert geom_entry.mesh_kind == "geometric"
        assert exp_entry.mesh_kind == "exponential"

        # GeometricExpandingMesh (N=60) achieves < 0.1% max relative error
        assert geom_entry.max_rel_error < 1e-3
        # And has >10x lower L_2 error than matched-size Uniform1DMesh (N=60)
        assert geom_entry.l2_error < 0.05 * uni_60.l2_error
        # And outperforms Uniform1DMesh with 5x (N=300) and 10x (N=600) more points
        assert geom_entry.max_rel_error < uni_300.max_rel_error
        assert geom_entry.max_rel_error < uni_600.max_rel_error
        assert geom_entry.l2_error < uni_600.l2_error

        summary = comp.summary()
        assert "Grid Efficiency Benchmark: cottrell" in summary
        assert "GeometricExpandingMesh" in summary
        assert "Uniform1DMesh" in summary

    def test_spherical_cottrell_efficiency_benchmark(self) -> None:
        """Verify GeometricExpandingMesh achieves <0.1% error with 5x-10x fewer points on spherical Cottrell."""
        comp = run_grid_efficiency_benchmark(
            case="spherical_cottrell",
            solver="scipy_ivp",
            expanding_points=75,
            uniform_points_list=(75, 375, 750),
            gamma=1.055,
            beta=3.5,
            r0=50e-6,
            t_span=(0.02, 1.0),
            n_times=25,
            target_rel_error=1e-3,
        )

        assert comp.passed
        assert comp.point_reduction_factor == 10.0
        assert comp.memory_reduction_factor == 10.0

        geom_entry = comp.entries[0]
        exp_entry = comp.entries[1]
        uni_75 = next(
            e for e in comp.entries if e.mesh_kind == "uniform" and e.n_points == 75
        )
        uni_375 = next(
            e for e in comp.entries if e.mesh_kind == "uniform" and e.n_points == 375
        )

        assert geom_entry.max_rel_error < 1e-3
        assert exp_entry.max_rel_error < 1e-3
        assert geom_entry.l2_error < 0.05 * uni_75.l2_error
        assert geom_entry.max_rel_error < uni_375.max_rel_error

    def test_cyclic_voltammetry_efficiency_benchmark(self) -> None:
        """Verify GeometricExpandingMesh (N=50) vs Uniform1DMesh (N=50, 300, 500) on reversible CV."""
        comp = run_grid_efficiency_benchmark(
            case="cyclic_voltammetry",
            solver="scipy_ivp",
            expanding_points=50,
            uniform_points_list=(50, 300, 500),
            gamma=1.08,
            include_exponential=False,
            scan_rate=0.1,
            target_rel_error=1e-3,
            rtol=1e-7,
            atol=1e-9,
        )

        assert comp.passed
        assert comp.point_reduction_factor == 10.0
        assert comp.memory_reduction_factor == 10.0
        assert comp.speedup_factor > 1.0

        geom_50 = comp.entries[0]
        uni_50 = comp.entries[1]
        uni_300 = comp.entries[2]

        # Peak current matches Randles-Sevcik within 0.1%
        assert geom_50.details["peak_current_rel_error"] < 1e-3
        # Waveform L_2 error on N=50 geometric mesh is >20x smaller than N=50 uniform mesh
        assert geom_50.l2_error < 0.05 * uni_50.l2_error
        # And more accurate than N=300 (6x larger) uniform mesh
        assert geom_50.l2_error < uni_300.l2_error


# ==============================================================================
# 4. Mass Conservation Under Zero-Flux Neumann Boundary Conditions
# ==============================================================================


class TestNonUniformMassConservation:
    """Verify total mass conservation on expanding and non-uniform meshes under closed boundaries."""

    @pytest.mark.parametrize(
        "solver_name",
        ["explicit", "implicit", "crank_nicolson", "scipy_ivp"],
    )
    @pytest.mark.parametrize(
        "mesh",
        [
            GeometricExpandingMesh(0.0, 1.0, n_points=61, gamma=1.04),
            GeometricExpandingMesh(0.0, 1.0, n_points=61, gamma=1.05, two_sided=True),
            ExponentialMesh(0.0, 1.0, n_points=61, beta=2.0, two_sided=True),
            ExponentialMesh.from_tanh(0.0, 1.0, n_points=61, beta=2.0, two_sided=True),
            ExponentialMesh.from_sinh(0.0, 1.0, n_points=61, beta=2.0, two_sided=True),
        ],
        ids=["geom_1s", "geom_2s", "exp_2s", "tanh_2s", "sinh_2s"],
    )
    def test_single_species_mass_conservation_all_solvers(
        self, solver_name: str, mesh: Mesh1D
    ) -> None:
        """Verify trapezoidal mass integral is conserved across all solvers and non-uniform meshes."""
        res = run_mass_conservation_benchmark(
            solver_name,
            grid=mesh,
            D=0.02,
            t_span=(0.0, 0.2),
            n_times=9,
            max_drift_threshold=2e-3,
        )
        assert res.passed, (
            f"{solver_name} failed mass conservation on {mesh}: "
            f"drift={res.details['max_relative_drift']:.4e}"
        )
        assert res.details["is_uniform"] is False

    def test_multi_species_reaction_diffusion_mass_conservation(self) -> None:
        """Verify total stoichiometric mass int (c_A + c_B) dx is conserved during A -> B conversion."""
        mesh = GeometricExpandingMesh(0.0, 1.0, n_points=71, gamma=1.05, two_sided=True)
        k_rate = 3.0

        def reactions(c_dict, x_pts, t):
            rate = k_rate * c_dict["A"]
            return {"A": -rate, "B": rate}

        problem = DiffusionProblem(
            grid=mesh,
            diffusivity={"A": 0.03, "B": 0.015},
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            initial_conditions={
                "A": lambda x: np.exp(-60.0 * (x - 0.35) ** 2),
                "B": lambda x: 0.2 * np.exp(-60.0 * (x - 0.65) ** 2),
            },
            reactions=reactions,
        )

        t_eval = np.linspace(0.0, 0.3, 10)
        res = get_solver("scipy_ivp", rtol=1e-8, atol=1e-10).solve(
            problem, t_span=(0.0, 0.3), t_eval=t_eval
        )
        assert res.success

        total_masses = np.array(
            [
                trapezoid(res["A"][k, :] + res["B"][k, :], mesh.nodes)
                for k in range(len(t_eval))
            ]
        )
        m0 = total_masses[0]
        max_rel_drift = float(np.max(np.abs(total_masses - m0)) / m0)
        assert max_rel_drift < 1e-3
        # Species A should decay while species B grows
        assert trapezoid(res["A"][-1, :], mesh.nodes) < trapezoid(
            res["A"][0, :], mesh.nodes
        )
        assert trapezoid(res["B"][-1, :], mesh.nodes) > trapezoid(
            res["B"][0, :], mesh.nodes
        )

    @pytest.mark.parametrize("geometry_kind", ["spherical", "cylindrical"])
    def test_radial_mass_conservation_on_expanding_grid(
        self, geometry_kind: str
    ) -> None:
        """Verify radial mass conservation (int r^p c(r, t) dr) under zero-flux Neumann BCs."""
        r0 = 1.0
        r_max = 2.0
        mesh = GeometricExpandingMesh(
            r0, r_max, n_points=81, gamma=1.04, two_sided=True
        )
        geom = (
            SphericalGeometry(r0=r0)
            if geometry_kind == "spherical"
            else CylindricalGeometry(r0=r0, length=1.0)
        )
        r = mesh.nodes
        weight = (
            4.0 * np.pi * (r**2) if geometry_kind == "spherical" else 2.0 * np.pi * r
        )

        problem = DiffusionProblem(
            grid=mesh,
            diffusivity={"A": 0.02},
            boundary_conditions={"A": (NeumannBC(0.0), NeumannBC(0.0))},
            initial_conditions={"A": lambda r_pts: np.exp(-60.0 * (r_pts - 1.5) ** 2)},
            geometry=geom,
        )

        t_eval = np.linspace(0.0, 0.2, 8)
        res = get_solver("scipy_ivp", rtol=1e-8, atol=1e-10).solve(
            problem, t_span=(0.0, 0.2), t_eval=t_eval
        )
        assert res.success

        masses = np.array(
            [trapezoid(res["A"][k, :] * weight, r) for k in range(len(t_eval))]
        )
        rel_drift = float(np.max(np.abs(masses - masses[0])) / masses[0])
        assert rel_drift < 2e-3


def test_grid_benchmark_exports() -> None:
    """Verify new grid benchmark symbols are exported at top-level and analytical namespaces."""
    assert sp.GridComparisonEntry is GridComparisonEntry
    assert sp.GridComparisonResult is GridComparisonResult
    assert sp.run_grid_efficiency_benchmark is run_grid_efficiency_benchmark
    assert sp.run_spherical_cottrell_benchmark is run_spherical_cottrell_benchmark
    assert sp.analytical.GridComparisonEntry is GridComparisonEntry
    assert sp.analytical.GridComparisonResult is GridComparisonResult
    assert sp.analytical.run_grid_efficiency_benchmark is run_grid_efficiency_benchmark
    assert (
        sp.analytical.run_spherical_cottrell_benchmark
        is run_spherical_cottrell_benchmark
    )


def test_grid_benchmark_validation_errors() -> None:
    """Verify invalid arguments raise clear ValueErrors."""
    with pytest.raises(ValueError, match="Unknown mesh_kind"):
        verify_spatial_convergence(mesh_kind="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expansion_span must be > 1"):
        verify_spatial_convergence(mesh_kind="geometric", expansion_span=1.0)

    with pytest.raises(ValueError, match="Unknown target"):
        verify_spatial_convergence(target="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown grid efficiency benchmark case"):
        run_grid_efficiency_benchmark(case="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="t_span"):
        run_spherical_cottrell_benchmark(t_span=(0.0, 1.0))

    with pytest.raises(ValueError, match="r0 must be positive"):
        run_spherical_cottrell_benchmark(r0=0.0)
