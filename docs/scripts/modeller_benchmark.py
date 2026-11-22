"""Modeller-level automated convergence & error benchmarking script.

Demonstrates ``softpotato.analytical.benchmark`` for validating PDE solvers
against closed-form analytical solutions:
1. Multi-solver benchmark suite (``run_benchmark_suite``)
2. Spatial convergence order verification (``verify_spatial_convergence``)
3. Temporal convergence order verification (``verify_temporal_convergence``)
4. Uniform vs. expanding mesh efficiency (``run_grid_efficiency_benchmark``)
"""

from __future__ import annotations

import sys

from softpotato.analytical import benchmark as bm


def main() -> int:
    print("=" * 72)
    print("Soft Potato — Modeller Numerical Benchmark Suite")
    print("=" * 72)
    print(f"Available benchmark cases: {bm.list_benchmark_cases()}\n")

    # 1. Multi-solver benchmark suite (planar / 1D canonical benchmarks)
    print("--- 1. Multi-Solver Benchmark Suite ---")
    suite = bm.run_benchmark_suite(
        solvers=["crank_nicolson", "implicit", "scipy_ivp"],
        cases=["fourier_decay", "cottrell_step", "mass_conservation"],
    )
    print(suite.summary())
    spherical_res = bm.run_spherical_cottrell_benchmark(solver="scipy_ivp")
    sph_l2 = (
        spherical_res.concentration_metrics.l2_error
        if spherical_res.concentration_metrics
        else float("nan")
    )
    print(
        f"Spherical Cottrell (scipy_ivp): passed={spherical_res.passed}, "
        f"L2_conc={sph_l2:.3e}, t={spherical_res.execution_time:.3f}s\n"
    )

    # 2. Spatial convergence verification (2nd-order central difference)
    print("--- 2. Spatial Convergence Verification (Crank-Nicolson) ---")
    spatial_res = bm.verify_spatial_convergence(
        solver="crank_nicolson",
        n_points_list=(21, 41, 81, 161),
        mesh_kind="uniform",
        expected_order=2.0,
    )
    print(
        f"  Observed spatial order = {spatial_res.observed_order:.3f} "
        f"(expected {spatial_res.expected_order:.1f}, R^2 = {spatial_res.r_squared:.5f}) "
        f"| passed = {spatial_res.passed}"
    )
    for dx, err in zip(spatial_res.spacings, spatial_res.errors):
        print(f"    dx = {dx:.5f} -> error = {err:.4e}")
    print()

    # 3. Temporal convergence verification
    print("--- 3. Temporal Convergence Verification ---")
    cn_time = bm.verify_temporal_convergence(
        solver="crank_nicolson",
        dt_list=(0.04, 0.02, 0.01, 0.005),
        expected_order=2.0,
    )
    btcs_time = bm.verify_temporal_convergence(
        solver="implicit",
        dt_list=(0.04, 0.02, 0.01, 0.005),
        expected_order=1.0,
    )
    print(
        f"  Crank-Nicolson temporal order = {cn_time.observed_order:.3f} "
        f"(expected {cn_time.expected_order:.1f}, R^2 = {cn_time.r_squared:.5f}) "
        f"| passed = {cn_time.passed}"
    )
    print(
        f"  Implicit (BTCS) temporal order = {btcs_time.observed_order:.3f} "
        f"(expected {btcs_time.expected_order:.1f}, R^2 = {btcs_time.r_squared:.5f}) "
        f"| passed = {btcs_time.passed}\n"
    )

    # 4. Uniform vs. expanding grid efficiency
    print("--- 4. Grid Efficiency Benchmark (Cottrell Step) ---")
    grid_cmp = bm.run_grid_efficiency_benchmark(
        case="cottrell",
        solver="scipy_ivp",
        expanding_points=60,
        uniform_points_list=(60, 200, 400),
    )
    for entry in grid_cmp.entries:
        rel_l2 = (
            entry.current_metrics.l2_error
            if entry.current_metrics is not None
            else float("nan")
        )
        print(
            f"  {entry.label:<24s} | N={entry.n_points:4d} "
            f"| dx_min={entry.dx_min:.2e} | L2_current={rel_l2:.3e} "
            f"| t={entry.execution_time:.3f}s"
        )
    print(
        f"  Point reduction = {grid_cmp.point_reduction_factor:.2f}x | "
        f"Speedup = {grid_cmp.speedup_factor:.2f}x | "
        f"passed = {grid_cmp.passed}"
    )

    all_ok = (
        suite.all_passed
        and spherical_res.passed
        and spatial_res.passed
        and cn_time.passed
        and btcs_time.passed
        and grid_cmp.passed
    )
    print("\nOverall status:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
