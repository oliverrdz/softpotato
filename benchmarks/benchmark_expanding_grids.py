"""Validation, Order-of-Convergence, and Performance Benchmarks for Expanding Grids.

Executes the comparative benchmark suite for Soft Potato's spatial discretization
and meshing module (`softpotato.grid`), evaluating:
1. Spatial order-of-convergence O(dx^2) across Uniform1DMesh, GeometricExpandingMesh,
   and ExponentialMesh (exponential, tanh, sinh).
2. Runtime speedup, memory scaling, and L_2 / relative error norms of
   GeometricExpandingMesh vs. Uniform1DMesh for planar Cottrell chronoamperometry,
   spherical Cottrell chronoamperometry, and reversible cyclic voltammetry.
3. Total mass conservation on expanding grids under zero-flux Neumann boundary conditions.

Usage
-----
    python -m benchmarks.benchmark_expanding_grids
"""

from __future__ import annotations

from typing import Any

from softpotato.analytical.benchmark import (
    ConvergenceResult,
    GridComparisonResult,
    run_grid_efficiency_benchmark,
    run_mass_conservation_benchmark,
    verify_spatial_convergence,
)


def run_all_expanding_grid_benchmarks() -> dict[str, Any]:
    """Run all expanding-grid convergence, efficiency, and conservation benchmarks."""
    convergence_studies: dict[str, ConvergenceResult] = {
        "uniform_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="uniform",
            dt_fine=2e-5,
        ),
        "geometric_1s_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="geometric",
            expansion_span=4.0,
            two_sided=False,
            dt_fine=2e-5,
        ),
        "geometric_2s_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="geometric",
            expansion_span=4.0,
            two_sided=True,
            dt_fine=2e-5,
        ),
        "geometric_1s_flux": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="geometric",
            expansion_span=4.0,
            target="flux",
            dt_fine=2e-5,
        ),
        "exponential_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="exponential",
            beta=1.5,
            mapping="exponential",
            dt_fine=2e-5,
        ),
        "tanh_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="exponential",
            beta=1.5,
            mapping="tanh",
            dt_fine=2e-5,
        ),
        "sinh_conc": verify_spatial_convergence(
            "crank_nicolson",
            n_points_list=(21, 41, 81, 161),
            mesh_kind="exponential",
            beta=1.5,
            mapping="sinh",
            dt_fine=2e-5,
        ),
    }

    efficiency_studies: dict[str, GridComparisonResult] = {
        "cottrell": run_grid_efficiency_benchmark(
            case="cottrell",
            solver="scipy_ivp",
            expanding_points=60,
            uniform_points_list=(60, 300, 600),
            gamma=1.08,
            beta=4.2,
            t_span=(0.01, 1.0),
        ),
        "spherical_cottrell": run_grid_efficiency_benchmark(
            case="spherical_cottrell",
            solver="scipy_ivp",
            expanding_points=75,
            uniform_points_list=(75, 375, 750),
            gamma=1.055,
            beta=3.5,
            r0=50e-6,
            t_span=(0.02, 1.0),
        ),
        "cyclic_voltammetry": run_grid_efficiency_benchmark(
            case="cyclic_voltammetry",
            solver="scipy_ivp",
            expanding_points=50,
            uniform_points_list=(50, 300, 500),
            gamma=1.08,
            include_exponential=False,
            scan_rate=0.1,
            rtol=1e-7,
            atol=1e-9,
        ),
    }

    mass_conservation = {
        "geometric": run_mass_conservation_benchmark(
            "crank_nicolson", mesh_kind="geometric", n_points=61, gamma=1.05
        ),
        "exponential": run_mass_conservation_benchmark(
            "crank_nicolson", mesh_kind="exponential", n_points=61, beta=2.0
        ),
    }

    return {
        "convergence": convergence_studies,
        "efficiency": efficiency_studies,
        "mass_conservation": mass_conservation,
    }


def main() -> None:
    """Execute and print formatted reports for all expanding-grid benchmarks."""
    results = run_all_expanding_grid_benchmarks()

    print("=" * 88)
    print("1. SPATIAL ORDER-OF-CONVERGENCE VERIFICATION O(dx^2)")
    print("=" * 88)
    for name, study in results["convergence"].items():
        status = "PASSED" if study.passed else "FAILED"
        print(
            f"  {name:20s} | Order p = {study.observed_order:.4f} "
            f"(Expected {study.expected_order:.1f}) | R^2 = {study.r_squared:.5f} | {status}"
        )
    print()

    print("=" * 88)
    print("2. COMPUTATIONAL EFFICIENCY & ACCURACY BENCHMARKS (UNIFORM VS EXPANDING)")
    print("=" * 88)
    for comp in results["efficiency"].values():
        print(comp.summary())
        print()

    print("=" * 88)
    print("3. NON-UNIFORM MASS CONSERVATION UNDER ZERO-FLUX NEUMANN BCs")
    print("=" * 88)
    for kind, res in results["mass_conservation"].items():
        status = "PASSED" if res.passed else "FAILED"
        drift = res.details["max_relative_drift"]
        print(f"  {kind:15s} | Max Relative Mass Drift = {drift:.4e} | {status}")


if __name__ == "__main__":
    main()
