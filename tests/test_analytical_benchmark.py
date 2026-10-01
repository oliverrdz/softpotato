"""Tests for automated solver benchmarking suite (softpotato.analytical.benchmark)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.benchmark import (
    BenchmarkCaseResult,
    BenchmarkMetrics,
    BenchmarkSuiteResult,
    ConvergenceResult,
    compute_error_metrics,
    estimate_convergence_order,
    list_benchmark_cases,
    run_benchmark,
    run_benchmark_suite,
    run_cottrell_benchmark,
    run_fourier_decay_benchmark,
    run_mass_conservation_benchmark,
    verify_spatial_convergence,
    verify_temporal_convergence,
)
from softpotato.solver import CrankNicolson, get_solver

# ==============================================================================
# 1. Error Metrics Tests
# ==============================================================================


def test_compute_error_metrics_exact_match():
    """Verify error metrics are zero when numerical matches exact reference."""
    exact = np.linspace(0.0, 1.0, 21)
    metrics = compute_error_metrics(exact, exact)

    assert metrics.rmse == 0.0
    assert metrics.max_abs_error == 0.0
    assert metrics.l2_error == 0.0
    assert metrics.mean_abs_error == 0.0
    assert metrics.mean_rel_error == 0.0
    assert metrics.max_rel_error == 0.0

    d = metrics.to_dict()
    assert isinstance(d, dict)
    assert d["rmse"] == 0.0


def test_compute_error_metrics_known_values():
    """Verify error metrics calculation against known reference values."""
    num = np.array([1.0, 2.1, 2.9])
    exact = np.array([1.0, 2.0, 3.0])

    metrics = compute_error_metrics(num, exact)

    # diffs = [0.0, 0.1, -0.1]
    # max_abs = 0.1
    # mean_abs = 0.2 / 3
    # rmse = sqrt((0 + 0.01 + 0.01) / 3) = sqrt(0.02 / 3)
    assert math.isclose(metrics.max_abs_error, 0.1, rel_tol=1e-12)
    assert math.isclose(metrics.mean_abs_error, 0.2 / 3.0, rel_tol=1e-12)
    assert math.isclose(metrics.rmse, math.sqrt(0.02 / 3.0), rel_tol=1e-12)

    norm_exact = math.sqrt(1.0 + 4.0 + 9.0)
    norm_diff = math.sqrt(0.02)
    assert math.isclose(metrics.l2_error, norm_diff / norm_exact, rel_tol=1e-12)


def test_compute_error_metrics_broadcasting():
    """Verify scalar or 1D broadcasting across 2D reference arrays."""
    exact_2d = np.ones((5, 10))
    num_scalar = 1.05

    metrics = compute_error_metrics(num_scalar, exact_2d)
    assert math.isclose(metrics.max_abs_error, 0.05, rel_tol=1e-12)
    assert math.isclose(metrics.rmse, 0.05, rel_tol=1e-12)


def test_compute_error_metrics_zero_ground_truth():
    """Verify relative errors handle exact zero reference without division by zero."""
    num = np.array([0.0, 0.01, -0.02])
    exact = np.zeros(3)

    metrics = compute_error_metrics(num, exact, eps=1e-8)
    assert math.isclose(metrics.max_abs_error, 0.02, rel_tol=1e-12)
    assert not math.isnan(metrics.mean_rel_error)
    assert not math.isinf(metrics.mean_rel_error)


def test_compute_error_metrics_shape_mismatch():
    """Verify ValueError is raised when arrays cannot be broadcast."""
    a = np.ones(5)
    b = np.ones(7)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_error_metrics(a, b)


# ==============================================================================
# 2. Convergence Order Estimation Tests
# ==============================================================================


def test_estimate_convergence_order_quadratic():
    """Verify order estimation returns slope 2.0 for pure quadratic error O(h^2)."""
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    err = 3.0 * h**2.0

    p, r2 = estimate_convergence_order(h, err)
    assert math.isclose(p, 2.0, rel_tol=1e-10)
    assert math.isclose(r2, 1.0, rel_tol=1e-6)


def test_estimate_convergence_order_linear():
    """Verify order estimation returns slope 1.0 for pure linear error O(h)."""
    h = np.array([0.2, 0.1, 0.05, 0.025])
    err = 1.5 * h

    p, r2 = estimate_convergence_order(h, err)
    assert math.isclose(p, 1.0, rel_tol=1e-10)
    assert math.isclose(r2, 1.0, rel_tol=1e-6)


def test_estimate_convergence_order_validation():
    """Verify input validation for estimate_convergence_order."""
    with pytest.raises(ValueError, match="1D arrays"):
        estimate_convergence_order(np.ones((2, 2)), np.ones((2, 2)))

    with pytest.raises(ValueError, match="Length mismatch"):
        estimate_convergence_order([0.1, 0.2], [0.01])

    with pytest.raises(ValueError, match="At least 2 points"):
        estimate_convergence_order([0.1], [0.01])

    with pytest.raises(ValueError, match="strictly positive"):
        estimate_convergence_order([0.1, -0.05], [0.01, 0.02])

    with pytest.raises(ValueError, match="strictly positive"):
        estimate_convergence_order([0.1, 0.05], [0.01, 0.0])

    with pytest.raises(ValueError, match="sufficient variation"):
        estimate_convergence_order([0.1, 0.1], [0.01, 0.02])


# ==============================================================================
# 3. Fourier Decay Benchmark Tests
# ==============================================================================


@pytest.mark.parametrize(
    "solver_name",
    ["crank_nicolson", "scipy_ivp", "implicit", "explicit"],
)
def test_fourier_decay_benchmark_all_solvers(solver_name: str):
    """Verify Fourier sine decay benchmark passes across all 4 built-in solvers."""
    res = run_fourier_decay_benchmark(
        solver_name,
        n_points=41,
        D=0.05,
        L=1.0,
        t_end=0.1,
        n_times=6,
        max_error_threshold=5e-3,
    )

    assert isinstance(res, BenchmarkCaseResult)
    assert res.passed
    assert res.case_name == "fourier_decay"
    assert res.solver_name in (
        solver_name,
        solver_name.replace("_", ""),
        solver_name.lower(),
    )
    assert res.concentration_metrics.max_abs_error < 5e-3
    assert res.concentration_metrics.rmse < 5e-3
    assert res.execution_time > 0.0
    assert res.flux_metrics is not None
    assert res.flux_metrics.max_abs_error < 5e-3


def test_fourier_decay_benchmark_solver_instance():
    """Verify run_fourier_decay_benchmark accepts BaseSolver instances directly."""
    solver = CrankNicolson(dt=0.005)
    res = run_fourier_decay_benchmark(solver, n_points=31, t_end=0.05)
    assert res.passed
    assert res.concentration_metrics.max_abs_error < 5e-3


# ==============================================================================
# 4. Cottrell Potential Step Benchmark Tests
# ==============================================================================


@pytest.mark.parametrize("solver_name", ["crank_nicolson", "scipy_ivp"])
def test_cottrell_benchmark_solvers(solver_name: str):
    """Verify Cottrell potential step benchmark passes for accurate solvers."""
    res = run_cottrell_benchmark(
        solver_name,
        n_points=81,
        D=1e-5,
        c_bulk=1e-3,
        t_span=(0.02, 0.1),
        n_times=10,
        max_error_threshold=0.05,
    )

    assert isinstance(res, BenchmarkCaseResult)
    assert res.passed
    assert res.case_name == "cottrell_step"
    assert res.concentration_metrics.max_abs_error < 0.05
    assert res.flux_metrics is not None
    # Current relative error should be small
    assert res.flux_metrics.max_rel_error < 0.05


def test_cottrell_benchmark_invalid_t_span():
    """Verify invalid t_span ranges raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        run_cottrell_benchmark("crank_nicolson", t_span=(0.0, 0.1))

    with pytest.raises(ValueError, match="strictly greater"):
        run_cottrell_benchmark("crank_nicolson", t_span=(0.2, 0.1))


# ==============================================================================
# 5. Mass Conservation Benchmark Tests
# ==============================================================================


@pytest.mark.parametrize(
    "solver_name",
    ["crank_nicolson", "implicit", "scipy_ivp"],
)
def test_mass_conservation_benchmark_solvers(solver_name: str):
    """Verify total mass is conserved under zero-flux Neumann boundary conditions."""
    res = run_mass_conservation_benchmark(
        solver_name,
        n_points=51,
        D=0.02,
        t_span=(0.0, 0.1),
        n_times=6,
        max_drift_threshold=1e-3,
    )

    assert isinstance(res, BenchmarkCaseResult)
    assert res.passed
    assert res.case_name == "mass_conservation"
    assert res.details["max_relative_drift"] < 1e-3


# ==============================================================================
# 6. Spatial Order-of-Convergence Verification
# ==============================================================================


def test_verify_spatial_convergence_crank_nicolson():
    """Verify Crank-Nicolson exhibits second-order spatial convergence O(dx^2)."""
    study = verify_spatial_convergence(
        "crank_nicolson",
        n_points_list=(21, 41, 81),
        D=0.05,
        L=1.0,
        t_fixed=0.1,
        dt_fine=1e-4,
        expected_order=2.0,
        order_tolerance=0.25,
    )

    assert isinstance(study, ConvergenceResult)
    assert study.passed
    assert study.spacing_type == "spatial"
    assert math.isclose(study.observed_order, 2.0, abs_tol=0.2)
    assert study.r_squared >= 0.99
    assert len(study.spacings) == 3
    assert len(study.errors) == 3


def test_verify_spatial_convergence_solver_instance():
    """Verify verify_spatial_convergence accepts BaseSolver instances."""
    solver = get_solver("crank_nicolson")
    study = verify_spatial_convergence(solver, n_points_list=(21, 41))
    assert study.spacing_type == "spatial"
    assert len(study.spacings) == 2


# ==============================================================================
# 7. Temporal Order-of-Convergence Verification
# ==============================================================================


def test_verify_temporal_convergence_crank_nicolson():
    """Verify Crank-Nicolson exhibits second-order temporal convergence O(dt^2)."""
    study = verify_temporal_convergence(
        "crank_nicolson",
        dt_list=(0.04, 0.02, 0.01),
        n_points=41,
        D=0.05,
        t_end=0.2,
        reference="fine_step",
        dt_ref=1e-5,
        expected_order=2.0,
        order_tolerance=0.25,
    )

    assert isinstance(study, ConvergenceResult)
    assert study.passed
    assert study.spacing_type == "temporal"
    assert math.isclose(study.observed_order, 2.0, abs_tol=0.25)
    assert study.r_squared >= 0.99


def test_verify_temporal_convergence_implicit_btcs():
    """Verify backward-Euler (implicit) exhibits first-order temporal convergence O(dt)."""
    study = verify_temporal_convergence(
        "implicit",
        dt_list=(0.04, 0.02, 0.01),
        n_points=41,
        D=0.05,
        t_end=0.2,
        reference="fine_step",
        dt_ref=1e-5,
        expected_order=1.0,
        order_tolerance=0.2,
    )

    assert isinstance(study, ConvergenceResult)
    assert study.passed
    assert study.spacing_type == "temporal"
    assert math.isclose(study.observed_order, 1.0, abs_tol=0.2)
    assert study.r_squared >= 0.99


# ==============================================================================
# 8. Benchmark Dispatcher & Suite Runner Tests
# ==============================================================================


def test_list_benchmark_cases():
    """Verify listing available benchmark cases."""
    cases = list_benchmark_cases()
    assert "fourier_decay" in cases
    assert "cottrell_step" in cases
    assert "mass_conservation" in cases


def test_run_benchmark_dispatcher():
    """Verify unified run_benchmark dispatcher routes to correct cases."""
    res_f = run_benchmark("fourier_decay", "crank_nicolson", n_points=31)
    assert res_f.case_name == "fourier_decay"
    assert res_f.passed

    res_c = run_benchmark(
        "cottrell", "crank_nicolson", n_points=41, t_span=(0.02, 0.05)
    )
    assert res_c.case_name == "cottrell_step"
    assert res_c.passed

    res_m = run_benchmark("mass_conservation", "crank_nicolson", n_points=31)
    assert res_m.case_name == "mass_conservation"
    assert res_m.passed

    with pytest.raises(ValueError, match="Unknown benchmark case"):
        run_benchmark("non_existent_benchmark", "crank_nicolson")


def test_run_benchmark_suite():
    """Verify running a multi-solver benchmark suite and generating report."""
    suite = run_benchmark_suite(
        solvers=["crank_nicolson", "scipy_ivp"],
        cases=["fourier_decay", "mass_conservation"],
    )

    assert isinstance(suite, BenchmarkSuiteResult)
    assert suite.all_passed
    assert len(suite.cases) == 4
    assert suite.total_time > 0.0

    summary = suite.summary()
    assert isinstance(summary, str)
    assert "Benchmark Case" in summary
    assert "fourier_decay" in summary
    assert "mass_conservation" in summary
    assert "PASSED" in summary


# ==============================================================================
# 9. Top-Level Package Exports Verification
# ==============================================================================


def test_top_level_exports():
    """Verify all benchmark functions and dataclasses are exported properly."""
    assert sp.BenchmarkMetrics is BenchmarkMetrics
    assert sp.analytical.BenchmarkMetrics is BenchmarkMetrics
    assert sp.ConvergenceResult is ConvergenceResult
    assert sp.analytical.ConvergenceResult is ConvergenceResult
    assert sp.BenchmarkCaseResult is BenchmarkCaseResult
    assert sp.analytical.BenchmarkCaseResult is BenchmarkCaseResult
    assert sp.BenchmarkSuiteResult is BenchmarkSuiteResult
    assert sp.analytical.BenchmarkSuiteResult is BenchmarkSuiteResult

    assert sp.compute_error_metrics is compute_error_metrics
    assert sp.estimate_convergence_order is estimate_convergence_order
    assert sp.list_benchmark_cases is list_benchmark_cases
    assert sp.run_benchmark is run_benchmark
    assert sp.run_benchmark_suite is run_benchmark_suite
    assert sp.run_fourier_decay_benchmark is run_fourier_decay_benchmark
    assert sp.run_cottrell_benchmark is run_cottrell_benchmark
    assert sp.run_mass_conservation_benchmark is run_mass_conservation_benchmark
    assert sp.verify_spatial_convergence is verify_spatial_convergence
    assert sp.verify_temporal_convergence is verify_temporal_convergence
