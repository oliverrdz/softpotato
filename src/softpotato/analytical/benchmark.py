r"""Automated numerical solver validation suite and analytical benchmarking harness.

This submodule provides a comprehensive benchmarking suite that validates numerical
PDE solvers (:mod:`softpotato.solver`) against closed-form analytical ground truths
(:mod:`softpotato.analytical`).

Features
--------
1. **Standardized Error Metrics**:

   - Pointwise absolute error (:math:`L_\infty` norm, :func:`compute_error_metrics`).
   - Root-Mean-Square Error (RMSE).
   - Discrete relative :math:`L_2` error norm.
   - Mean absolute error (MAE).
   - Mean and maximum relative errors with zero-division protection.

2. **Order-of-Convergence Verification**:

   - Spatial discretization convergence verification (:math:`\mathcal{O}(\Delta x^2)`).
   - Temporal discretization convergence verification (:math:`\mathcal{O}(\Delta t^2)` for Crank–Nicolson, :math:`\mathcal{O}(\Delta t)` for backward-Euler BTCS).
   - Empirical order estimation via log-log regression (:func:`estimate_convergence_order`).

3. **Canonical Electrochemical Benchmark Problems**:

   - Fourier sine diffusion decay (:func:`run_fourier_decay_benchmark`).
   - Planar Cottrell potential step chronoamperometry (:func:`run_cottrell_benchmark`).
   - Total mass conservation under zero-flux Neumann BCs (:func:`run_mass_conservation_benchmark`).

4. **Automated Suite Execution**:

   - Unified single-case runner (:func:`run_benchmark`).
   - Multi-solver comparative benchmark suite (:func:`run_benchmark_suite`) with formatted tabular ASCII reporting.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.integrate import trapezoid

from softpotato.analytical.step import cottrell, step_concentration_profile
from softpotato.constants import FARADAY
from softpotato.solver import (
    BaseSolver,
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    get_solver,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# ==============================================================================
# 1. Result Containers and Dataclasses
# ==============================================================================


@dataclass(frozen=True, kw_only=True)
class BenchmarkMetrics:
    r"""Standardized error metrics between numerical and analytical solutions.

    Parameters
    ----------
    rmse : float
        Root-mean-square error:

        .. math::

            \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (u_i^{\text{num}} - u_i^{\text{exact}})^2}

    max_abs_error : float
        Maximum pointwise absolute deviation (:math:`L_\infty` norm):

        .. math::

            \|u^{\text{num}} - u^{\text{exact}}\|_\infty = \max_i |u_i^{\text{num}} - u_i^{\text{exact}}|

    l2_error : float
        Discrete relative :math:`L_2` error norm:

        .. math::

            L_2 = \frac{\|u^{\text{num}} - u^{\text{exact}}\|_2}{\|u^{\text{exact}}\|_2}

        (evaluates to absolute :math:`L_2` norm if :math:`\|u^{\text{exact}}\|_2 = 0`).
    mean_abs_error : float
        Mean absolute error (MAE):

        .. math::

            \text{MAE} = \frac{1}{N} \sum_{i=1}^N |u_i^{\text{num}} - u_i^{\text{exact}}|

    mean_rel_error : float
        Mean relative error across all evaluated points:

        .. math::

            \text{MRE} = \frac{1}{N} \sum_{i=1}^N \frac{|u_i^{\text{num}} - u_i^{\text{exact}}|}{|u_i^{\text{exact}}| + \epsilon}

    max_rel_error : float
        Maximum relative error:

        .. math::

            \text{MaxRE} = \max_i \frac{|u_i^{\text{num}} - u_i^{\text{exact}}|}{|u_i^{\text{exact}}| + \epsilon}

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.benchmark import compute_error_metrics
    >>> num = np.array([1.0, 2.01, 3.0])
    >>> exact = np.array([1.0, 2.0, 3.0])
    >>> metrics = compute_error_metrics(num, exact)
    >>> round(metrics.max_abs_error, 4)
    0.01
    """

    rmse: float
    max_abs_error: float
    l2_error: float
    mean_abs_error: float
    mean_rel_error: float
    max_rel_error: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a standard Python dictionary."""
        return {
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "mean_abs_error": self.mean_abs_error,
            "mean_rel_error": self.mean_rel_error,
            "max_rel_error": self.max_rel_error,
        }


@dataclass(frozen=True, kw_only=True)
class ConvergenceResult:
    r"""Results from an order-of-convergence refinement study.

    Parameters
    ----------
    spacings : np.ndarray
        Discretization spacings tested (:math:`\Delta x` for spatial,
        :math:`\Delta t` for temporal).
    errors : np.ndarray
        Corresponding error norm evaluated for each spacing.
    observed_order : float
        Empirical convergence order :math:`p` extracted from linear regression:

        .. math::

            \ln(\text{error}) = p \cdot \ln(h) + C

    expected_order : float
        Theoretical convergence order (e.g., :math:`2.0` for second-order schemes).
    r_squared : float
        Goodness-of-fit :math:`R^2` of the log-log linear regression.
    passed : bool
        Whether :attr:`observed_order` meets :attr:`expected_order` within
        tolerance and :attr:`r_squared` is high (:math:`R^2 \ge 0.95`).
    spacing_type : {'spatial', 'temporal'}
        Discretization domain refined during the study.
    details : dict[str, Any]
        Additional context such as solver name, test case, and tolerance thresholds.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.benchmark import estimate_convergence_order
    >>> h = np.array([0.1, 0.05, 0.025])
    >>> err = 0.5 * h**2  # pure 2nd-order error
    >>> p, r2 = estimate_convergence_order(h, err)
    >>> round(p, 2)
    2.0
    >>> round(r2, 4)
    1.0
    """

    spacings: np.ndarray
    errors: np.ndarray
    observed_order: float
    expected_order: float
    r_squared: float
    passed: bool
    spacing_type: Literal["spatial", "temporal"]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BenchmarkCaseResult:
    """Execution and validation outcome for a single benchmark problem.

    Parameters
    ----------
    case_name : str
        Name of the benchmark problem (e.g. ``'fourier_decay'``, ``'cottrell'``).
    solver_name : str
        Name of the numerical solver evaluated.
    concentration_metrics : BenchmarkMetrics
        Standardized error metrics for the spatial/temporal concentration field.
    flux_metrics : BenchmarkMetrics or None
        Standardized error metrics for interfacial boundary flux or current,
        or None if not applicable for the case.
    execution_time : float
        Wall-clock simulation execution time in seconds.
    passed : bool
        Whether all error metrics satisfied the predefined tolerance criteria.
    details : dict[str, Any]
        Diagnostic metadata and parameters used for the benchmark run.
    """

    case_name: str
    solver_name: str
    concentration_metrics: BenchmarkMetrics
    flux_metrics: BenchmarkMetrics | None = None
    execution_time: float = 0.0
    passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BenchmarkSuiteResult:
    """Aggregated outcome of running a benchmark suite across solvers and test cases.

    Parameters
    ----------
    cases : list[BenchmarkCaseResult]
        Collection of all individual benchmark run results.
    total_time : float
        Total wall-clock execution time for the entire suite in seconds.
    all_passed : bool
        Flag indicating if all individual benchmark cases passed their tolerances.
    """

    cases: list[BenchmarkCaseResult]
    total_time: float
    all_passed: bool

    def summary(self) -> str:
        """Generate a formatted ASCII comparison table of all benchmark runs."""
        headers = [
            "Benchmark Case",
            "Solver",
            "Status",
            "Conc. RMSE",
            "Conc. L_inf",
            "Flux RMSE",
            "Time (s)",
        ]
        rows: list[list[str]] = []
        for res in self.cases:
            status = "PASSED" if res.passed else "FAILED"
            c_rmse = f"{res.concentration_metrics.rmse:.3e}"
            c_linf = f"{res.concentration_metrics.max_abs_error:.3e}"
            f_rmse = (
                f"{res.flux_metrics.rmse:.3e}"
                if res.flux_metrics is not None
                else "N/A"
            )
            time_str = f"{res.execution_time:.4f}"
            rows.append(
                [
                    res.case_name,
                    res.solver_name,
                    status,
                    c_rmse,
                    c_linf,
                    f_rmse,
                    time_str,
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_row = (
            "| "
            + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
            + " |"
        )
        data_rows = [
            "| "
            + " | ".join(val.ljust(w) for val, w in zip(row, col_widths, strict=True))
            + " |"
            for row in rows
        ]

        title = f"Soft Potato Solver Validation Suite (Total Time: {self.total_time:.3f}s | All Passed: {self.all_passed})"
        return "\n".join([title, sep, header_row, sep, *data_rows, sep])


# ==============================================================================
# 2. Error Computation & Regression Utilities
# ==============================================================================


def compute_error_metrics(
    numerical: ArrayLike,
    exact: ArrayLike,
    *,
    eps: float = 1e-12,
) -> BenchmarkMetrics:
    r"""Compute standardized error norms between numerical results and exact ground truths.

    Parameters
    ----------
    numerical : array-like
        Simulated numerical values (scalar, 1D, or multidimensional array).
    exact : array-like
        Exact analytical reference values corresponding to ``numerical``.
    eps : float, default 1e-12
        Small positive constant added to denominators to prevent division by zero
        when evaluating relative errors against zero-valued ground truths.

    Returns
    -------
    BenchmarkMetrics
        Dataclass containing RMSE, :math:`L_\infty` (:attr:`max_abs_error`),
        discrete :math:`L_2` relative error, MAE, mean relative error, and
        maximum relative error.

    Raises
    ------
    ValueError
        If input arrays cannot be broadcast together or have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.benchmark import compute_error_metrics
    >>> y_num = np.array([1.0, 2.05, 3.0])
    >>> y_exact = np.array([1.0, 2.0, 3.0])
    >>> m = compute_error_metrics(y_num, y_exact)
    >>> round(m.max_abs_error, 4)
    0.05
    >>> round(m.rmse, 4)
    0.0289
    """
    num_arr = np.asarray(numerical, dtype=float)
    exact_arr = np.asarray(exact, dtype=float)

    if num_arr.shape != exact_arr.shape:
        try:
            num_arr, exact_arr = np.broadcast_arrays(num_arr, exact_arr)
        except ValueError as err:
            raise ValueError(
                f"Shape mismatch: numerical shape {num_arr.shape} cannot be broadcast "
                f"with exact reference shape {exact_arr.shape}."
            ) from err

    diff = num_arr - exact_arr
    abs_diff = np.abs(diff)

    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    norm_exact = float(np.linalg.norm(exact_arr))
    if norm_exact > eps:
        l2_err = float(np.linalg.norm(diff) / norm_exact)
    else:
        l2_err = float(np.linalg.norm(diff) / math.sqrt(exact_arr.size))

    rel_diff = abs_diff / (np.abs(exact_arr) + eps)
    mean_rel = float(np.mean(rel_diff))
    max_rel = float(np.max(rel_diff))

    return BenchmarkMetrics(
        rmse=rmse,
        max_abs_error=max_abs,
        l2_error=l2_err,
        mean_abs_error=mean_abs,
        mean_rel_error=mean_rel,
        max_rel_error=max_rel,
    )


def estimate_convergence_order(
    spacings: ArrayLike,
    errors: ArrayLike,
) -> tuple[float, float]:
    r"""Estimate empirical order of convergence :math:`p` and :math:`R^2` via log-log regression.

    Fits the power law :math:`\text{error} = C \cdot h^p` in logarithmic space:

    .. math::

        \ln(\text{error}) = p \cdot \ln(h) + \ln(C)

    Parameters
    ----------
    spacings : array-like
        Array of strictly positive discretization increments :math:`h`
        (:math:`\Delta x` or :math:`\Delta t`).
    errors : array-like
        Array of strictly positive corresponding error norms.

    Returns
    -------
    order : float
        Empirical convergence order :math:`p` (slope of log-log regression).
    r_squared : float
        Coefficient of determination :math:`R^2 \in [0, 1]` indicating the linearity
        and reliability of the power-law convergence regime.

    Raises
    ------
    ValueError
        If inputs have fewer than 2 data points or contain non-positive values.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.benchmark import estimate_convergence_order
    >>> h = np.array([0.1, 0.05, 0.025, 0.0125])
    >>> err = 2.5 * h**2
    >>> p, r2 = estimate_convergence_order(h, err)
    >>> round(p, 4)
    2.0
    >>> round(r2, 4)
    1.0
    """
    h_arr = np.asarray(spacings, dtype=float)
    e_arr = np.asarray(errors, dtype=float)

    if h_arr.ndim != 1 or e_arr.ndim != 1:
        raise ValueError("spacings and errors must be 1D arrays.")
    if len(h_arr) != len(e_arr):
        raise ValueError(
            f"Length mismatch: len(spacings)={len(h_arr)} != len(errors)={len(e_arr)}."
        )
    if len(h_arr) < 2:
        raise ValueError(
            f"At least 2 points are required to compute convergence order, got {len(h_arr)}."
        )
    if np.any(h_arr <= 0):
        raise ValueError("All spacings must be strictly positive (h > 0).")
    if np.any(e_arr <= 0):
        raise ValueError("All errors must be strictly positive (error > 0).")

    log_h = np.log(h_arr)
    log_e = np.log(e_arr)

    # Ordinary least-squares regression: log_e = p * log_h + intercept
    cov = np.cov(log_h, log_e)
    var_h = cov[0, 0]
    cov_he = cov[0, 1]

    if var_h <= 1e-15:
        raise ValueError(
            "Spacings do not have sufficient variation to determine slope."
        )

    order = float(cov_he / var_h)
    intercept = float(np.mean(log_e) - order * np.mean(log_h))

    # Calculate R^2
    fitted = order * log_h + intercept
    ss_tot = float(np.sum((log_e - np.mean(log_e)) ** 2))
    ss_res = float(np.sum((log_e - fitted) ** 2))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 1.0
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    return order, r_squared


def _resolve_solver_instance(solver: BaseSolver | str) -> tuple[BaseSolver, str]:
    """Helper to resolve a BaseSolver instance and canonical name."""
    if isinstance(solver, str):
        name = solver.lower()
        instance = get_solver(name)
        return instance, name
    if isinstance(solver, BaseSolver):
        # Find registered alias or class name
        for k, cls in BaseSolver._registry.items():
            if isinstance(solver, cls):
                return solver, k
        return solver, solver.__class__.__name__
    raise TypeError(
        f"solver must be an instance of BaseSolver or a registered solver string, "
        f"got {type(solver)}."
    )


# ==============================================================================
# 3. Canonical Electrochemical Benchmark Problems
# ==============================================================================


def run_fourier_decay_benchmark(
    solver: BaseSolver | str = "crank_nicolson",
    *,
    n_points: int = 51,
    D: float = 0.05,
    L: float = 1.0,
    t_end: float = 0.2,
    n_times: int = 11,
    max_error_threshold: float = 5e-3,
    **solver_kwargs: Any,
) -> BenchmarkCaseResult:
    r"""Run 1D Fourier sine diffusion decay benchmark comparing against exact solution.

    Governing PDE:

    .. math::

        \frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}, \quad x \in [0, L]

    Initial and boundary conditions:

    .. math::

        c(x, 0) = \sin\left(\frac{\pi x}{L}\right), \quad c(0, t) = c(L, t) = 0

    Exact analytical solution:

    .. math::

        c(x, t) = \sin\left(\frac{\pi x}{L}\right) \exp\left(-D \left(\frac{\pi}{L}\right)^2 t\right)

    Exact surface flux at :math:`x = 0`:

    .. math::

        J(0, t) = -D \left.\frac{\partial c}{\partial x}\right|_{x=0} = -D \frac{\pi}{L} \exp\left(-D \left(\frac{\pi}{L}\right)^2 t\right)

    Parameters
    ----------
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    n_points : int, default 51
        Number of spatial grid points :math:`N_x`.
    D : float, default 0.05
        Diffusion coefficient :math:`D > 0`.
    L : float, default 1.0
        Domain length :math:`L > 0`.
    t_end : float, default 0.2
        Simulation end time :math:`t_{\text{end}} > 0`.
    n_times : int, default 11
        Number of time evaluation points across :math:`[0, t_{\text{end}}]`.
    max_error_threshold : float, default 5e-3
        Maximum allowable :math:`L_\infty` concentration error to consider the run passed.
    **solver_kwargs : Any
        Additional keyword arguments forwarded to the solver (e.g. ``dt``).

    Returns
    -------
    BenchmarkCaseResult
        Outcome containing concentration metrics, interfacial flux metrics,
        wall-clock runtime, and pass/fail indicator.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import run_fourier_decay_benchmark
    >>> res = run_fourier_decay_benchmark("crank_nicolson", n_points=41, t_end=0.1)
    >>> res.passed
    True
    >>> res.concentration_metrics.max_abs_error < 5e-3
    True
    """
    solver_inst, solver_name = _resolve_solver_instance(solver)

    grid = np.linspace(0.0, L, n_points)
    t_eval = np.linspace(0.0, t_end, n_times)

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"c": lambda x_pts: np.sin(np.pi * x_pts / L)},
    )

    t0 = time.perf_counter()
    result = solver_inst.solve(
        problem, t_span=(0.0, t_end), t_eval=t_eval, **solver_kwargs
    )
    wall_time = time.perf_counter() - t0

    if not result.success:
        failed_metrics = BenchmarkMetrics(
            rmse=float("inf"),
            max_abs_error=float("inf"),
            l2_error=float("inf"),
            mean_abs_error=float("inf"),
            mean_rel_error=float("inf"),
            max_rel_error=float("inf"),
        )
        return BenchmarkCaseResult(
            case_name="fourier_decay",
            solver_name=solver_name,
            concentration_metrics=failed_metrics,
            flux_metrics=None,
            execution_time=wall_time,
            passed=False,
            details={"error_message": result.message},
        )

    # Exact 2D grid evaluation
    t_grid, x_grid = np.meshgrid(t_eval, grid, indexing="ij")
    decay_rate = D * (np.pi / L) ** 2
    c_exact = np.sin(np.pi * x_grid / L) * np.exp(-decay_rate * t_grid)
    c_num = result["c"]

    conc_metrics = compute_error_metrics(c_num, c_exact)

    # Exact interfacial flux J(0, t)
    j_exact = -D * (np.pi / L) * np.exp(-decay_rate * t_eval)
    j_num = result.fluxes.get("c", None)
    flux_metrics = compute_error_metrics(j_num, j_exact) if j_num is not None else None

    passed = bool(result.success and conc_metrics.max_abs_error <= max_error_threshold)

    return BenchmarkCaseResult(
        case_name="fourier_decay",
        solver_name=solver_name,
        concentration_metrics=conc_metrics,
        flux_metrics=flux_metrics,
        execution_time=wall_time,
        passed=passed,
        details={
            "n_points": n_points,
            "D": D,
            "L": L,
            "t_end": t_end,
            "dx": float(grid[1] - grid[0]),
            "max_error_threshold": max_error_threshold,
        },
    )


def run_cottrell_benchmark(
    solver: BaseSolver | str = "crank_nicolson",
    *,
    n_points: int = 101,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    n_electrons: int = 1,
    area: float = 1e-4,
    t_span: tuple[float, float] = (0.01, 0.2),
    n_times: int = 20,
    domain_factor: float = 6.0,
    max_error_threshold: float = 5e-2,
    **solver_kwargs: Any,
) -> BenchmarkCaseResult:
    r"""Run Cottrell potential step benchmark comparing concentration profiles and flux.

    Simulates chronoamperometric semi-infinite linear diffusion following a step to
    zero surface concentration (:math:`c(0, t) = 0`). The domain length is automatically
    dimensioned as :math:`L = \text{domain\_factor} \times \sqrt{D \cdot t_{\text{end}}}`
    to ensure the right boundary satisfies semi-infinite boundary conditions
    (:math:`c(L, t) \approx c^*`).

    Exact analytical spatial concentration profile:

    .. math::

        c(x, t) = c^* \text{erf}\left(\frac{x}{2 \sqrt{D t}}\right)

    Exact Cottrell current transient:

    .. math::

        I(t) = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}}

    Parameters
    ----------
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    n_points : int, default 101
        Number of spatial grid points :math:`N_x`.
    D : float, default 1e-5
        Diffusion coefficient :math:`D > 0`.
    c_bulk : float, default 1e-3
        Bulk concentration :math:`c^* > 0`.
    n_electrons : int, default 1
        Number of electrons transferred per redox event :math:`n \ge 1`.
    area : float, default 1e-4
        Electrode area in :math:`\text{m}^2` (or :math:`\text{cm}^2`).
    t_span : tuple[float, float], default (0.01, 0.2)
        Evaluation time interval :math:`(t_{\min}, t_{\max})` with :math:`t_{\min} > 0`
        to evaluate beyond the initial Cottrell step discontinuity.
    n_times : int, default 20
        Number of evaluation time points across :attr:`t_span`.
    domain_factor : float, default 6.0
        Diffusion layer multiplier setting domain length :math:`L = \text{domain\_factor} \sqrt{D t_{\max}}`.
    max_error_threshold : float, default 5e-2
        Maximum allowable :math:`L_\infty` concentration error to consider the run passed.
    **solver_kwargs : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    BenchmarkCaseResult
        Outcome containing concentration metrics, Cottrell current metrics,
        runtime, and pass/fail indicator.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import run_cottrell_benchmark
    >>> res = run_cottrell_benchmark("scipy_ivp", n_points=81, t_span=(0.02, 0.1))
    >>> res.passed
    True
    >>> res.concentration_metrics.max_abs_error < 0.05
    True
    """
    solver_inst, solver_name = _resolve_solver_instance(solver)

    t_start, t_end = t_span
    if t_start <= 0.0:
        raise ValueError(
            f"t_span start must be strictly positive (t_start > 0), got {t_start}."
        )
    if t_start >= t_end:
        raise ValueError(
            f"t_span end must be strictly greater than start, got {t_span}."
        )

    l_domain = domain_factor * math.sqrt(D * t_end)
    grid = np.linspace(0.0, l_domain, n_points)

    # Solve from 0.0 to t_end so transient starts at t=0
    # Include 0.0 in integration grid, then evaluate metrics for t >= t_start
    dt_eval = (t_end - t_start) / max(1, n_times - 1)
    n_pre_steps = max(1, math.ceil(t_start / dt_eval))
    pre_t = np.linspace(0.0, t_start, n_pre_steps + 1)[:-1]
    t_eval_window = np.linspace(t_start, t_end, n_times)
    full_t_eval = np.concatenate([pre_t, t_eval_window])

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
        initial_conditions={"c": c_bulk},
    )

    t0 = time.perf_counter()
    result = solver_inst.solve(
        problem, t_span=(0.0, t_end), t_eval=full_t_eval, **solver_kwargs
    )
    wall_time = time.perf_counter() - t0

    if not result.success:
        failed_metrics = BenchmarkMetrics(
            rmse=float("inf"),
            max_abs_error=float("inf"),
            l2_error=float("inf"),
            mean_abs_error=float("inf"),
            mean_rel_error=float("inf"),
            max_rel_error=float("inf"),
        )
        return BenchmarkCaseResult(
            case_name="cottrell_step",
            solver_name=solver_name,
            concentration_metrics=failed_metrics,
            flux_metrics=None,
            execution_time=wall_time,
            passed=False,
            details={"error_message": result.message},
        )

    # Filter to evaluation time window [t_start, t_end]
    mask = result.t >= (t_start - 1e-12)
    t_eval = result.t[mask]
    c_num = result["c"][mask, :]

    # Exact concentration field across all evaluated times and grid points
    c_exact = np.zeros_like(c_num)
    for k, t_k in enumerate(t_eval):
        c_exact[k, :] = step_concentration_profile(
            grid, t=float(t_k), D=D, c_bulk=c_bulk
        )

    conc_metrics = compute_error_metrics(c_num, c_exact)

    # Compare Cottrell currents: for an oxidation, I_num = -n * F * A * J(0) > 0
    # J(0) = -D * dc/dx < 0 when reactant is consumed at x=0, so I_num > 0 matches IUPAC anodic sign.
    flux_metrics = None
    if "c" in result.fluxes:
        j_num = result.fluxes["c"][mask]
        i_num = -n_electrons * FARADAY * area * j_num
        i_exact = cottrell(t_eval, n=n_electrons, D=D, c_bulk=c_bulk, area=area)
        flux_metrics = compute_error_metrics(i_num, i_exact)

    passed = bool(result.success and conc_metrics.max_abs_error <= max_error_threshold)

    return BenchmarkCaseResult(
        case_name="cottrell_step",
        solver_name=solver_name,
        concentration_metrics=conc_metrics,
        flux_metrics=flux_metrics,
        execution_time=wall_time,
        passed=passed,
        details={
            "n_points": n_points,
            "D": D,
            "c_bulk": c_bulk,
            "t_span": t_span,
            "domain_length": l_domain,
            "max_error_threshold": max_error_threshold,
        },
    )


def run_mass_conservation_benchmark(
    solver: BaseSolver | str = "crank_nicolson",
    *,
    n_points: int = 61,
    D: float = 0.02,
    L: float = 1.0,
    t_span: tuple[float, float] = (0.0, 0.2),
    n_times: int = 11,
    max_drift_threshold: float = 1e-3,
    **solver_kwargs: Any,
) -> BenchmarkCaseResult:
    r"""Run mass conservation benchmark under zero-flux Neumann boundary conditions.

    Solves 1D diffusion on a closed interval with zero-flux boundaries:

    .. math::

        \left.\frac{\partial c}{\partial x}\right|_{x=0} = 0, \quad \left.\frac{\partial c}{\partial x}\right|_{x=L} = 0

    initialized with an internal Gaussian concentration pulse:

    .. math::

        c(x, 0) = \exp\left(-50 \left(x - \frac{L}{2}\right)^2\right)

    Analytical conservation invariant:

    .. math::

        M(t) = \int_0^L c(x, t) \, dx = M(0) = \text{constant}, \quad \forall t \ge 0

    Parameters
    ----------
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    n_points : int, default 61
        Number of spatial grid points :math:`N_x`.
    D : float, default 0.02
        Diffusion coefficient :math:`D > 0`.
    L : float, default 1.0
        Domain length :math:`L > 0`.
    t_span : tuple[float, float], default (0.0, 0.2)
        Simulation time span :math:`(t_0, t_{\text{end}})`.
    n_times : int, default 11
        Number of evaluation time points.
    max_drift_threshold : float, default 1e-3
        Maximum relative mass drift :math:`|M(t) - M(0)| / M(0)` allowed.
    **solver_kwargs : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    BenchmarkCaseResult
        Outcome containing mass error metrics, execution time, and pass/fail indicator.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import run_mass_conservation_benchmark
    >>> res = run_mass_conservation_benchmark("crank_nicolson", n_points=51)
    >>> res.passed
    True
    >>> res.details["max_relative_drift"] < 1e-3
    True
    """
    solver_inst, solver_name = _resolve_solver_instance(solver)

    grid = np.linspace(0.0, L, n_points)
    t_eval = np.linspace(t_span[0], t_span[1], n_times)

    pulse_center = 0.5 * L
    init_func = lambda x_pts: np.exp(-50.0 * (x_pts - pulse_center) ** 2)

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (NeumannBC(0.0), NeumannBC(0.0))},
        initial_conditions={"c": init_func},
    )

    t0 = time.perf_counter()
    result = solver_inst.solve(problem, t_span=t_span, t_eval=t_eval, **solver_kwargs)
    wall_time = time.perf_counter() - t0

    if not result.success:
        failed_metrics = BenchmarkMetrics(
            rmse=float("inf"),
            max_abs_error=float("inf"),
            l2_error=float("inf"),
            mean_abs_error=float("inf"),
            mean_rel_error=float("inf"),
            max_rel_error=float("inf"),
        )
        return BenchmarkCaseResult(
            case_name="mass_conservation",
            solver_name=solver_name,
            concentration_metrics=failed_metrics,
            flux_metrics=None,
            execution_time=wall_time,
            passed=False,
            details={"error_message": result.message},
        )

    # Compute numerical total mass M(t) = integral c(x, t) dx
    c = result["c"]
    masses = np.array([trapezoid(c[k, :], grid) for k in range(len(t_eval))])
    m0 = masses[0]

    exact_masses = np.full_like(masses, m0)
    mass_metrics = compute_error_metrics(masses, exact_masses)

    rel_drifts = np.abs(masses - m0) / (abs(m0) + 1e-15)
    max_drift = float(np.max(rel_drifts))

    passed = bool(result.success and max_drift <= max_drift_threshold)

    return BenchmarkCaseResult(
        case_name="mass_conservation",
        solver_name=solver_name,
        concentration_metrics=mass_metrics,
        flux_metrics=None,
        execution_time=wall_time,
        passed=passed,
        details={
            "initial_mass": float(m0),
            "final_mass": float(masses[-1]),
            "max_relative_drift": max_drift,
            "max_drift_threshold": max_drift_threshold,
        },
    )


# ==============================================================================
# 4. Order-of-Convergence Verification
# ==============================================================================


def verify_spatial_convergence(
    solver: BaseSolver | str = "crank_nicolson",
    *,
    n_points_list: Sequence[int] = (21, 41, 81, 161),
    D: float = 0.05,
    L: float = 1.0,
    t_fixed: float = 0.1,
    dt_fine: float = 1e-4,
    expected_order: float = 2.0,
    order_tolerance: float = 0.25,
    metric: Literal["max_abs_error", "rmse", "l2_error"] = "max_abs_error",
    **solver_kwargs: Any,
) -> ConvergenceResult:
    r"""Verify spatial discretization order across grid refinement :math:`\Delta x`.

    Executes a sequence of spatial grid refinements on the Fourier sine decay problem,
    using a fine time step :attr:`dt_fine` so spatial error dominates. Computes the
    empirical convergence order :math:`p` via log-log regression:

    .. math::

        \ln(\text{error}) = p \cdot \ln(\Delta x) + C

    Parameters
    ----------
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    n_points_list : sequence of int, default (21, 41, 81, 161)
        List of spatial grid point counts to evaluate.
    D : float, default 0.05
        Diffusion coefficient :math:`D > 0`.
    L : float, default 1.0
        Domain length :math:`L > 0`.
    t_fixed : float, default 0.1
        Fixed time at which error is evaluated.
    dt_fine : float, default 1e-4
        Small time step size ensuring temporal error is negligible.
    expected_order : float, default 2.0
        Theoretical spatial order of convergence (default 2.0 for central differences).
    order_tolerance : float, default 0.25
        Permissible deficit below :attr:`expected_order` to consider the test passed.
    metric : {'max_abs_error', 'rmse', 'l2_error'}, default 'max_abs_error'
        Error norm to regress against :math:`\Delta x`.
    **solver_kwargs : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    ConvergenceResult
        Dataclass containing grid spacings :math:`\Delta x`, error array,
        empirical order :math:`p`, :math:`R^2`, and pass/fail indicator.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import verify_spatial_convergence
    >>> res = verify_spatial_convergence("crank_nicolson", n_points_list=(21, 41, 81))
    >>> res.passed
    True
    >>> round(res.observed_order, 1)
    2.0
    """
    solver_inst, solver_name = _resolve_solver_instance(solver)

    dx_list: list[float] = []
    error_list: list[float] = []

    decay_factor = math.exp(-D * (math.pi / L) ** 2 * t_fixed)

    for n_pts in n_points_list:
        grid = np.linspace(0.0, L, n_pts)
        dx = float(grid[1] - grid[0])

        problem = DiffusionProblem(
            grid=grid,
            diffusivity={"c": D},
            boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
            initial_conditions={"c": lambda x_pts: np.sin(np.pi * x_pts / L)},
        )

        kwargs = dict(solver_kwargs)
        if "dt" not in kwargs:
            kwargs["dt"] = dt_fine

        res = solver_inst.solve(
            problem,
            t_span=(0.0, t_fixed),
            t_eval=np.array([0.0, t_fixed]),
            **kwargs,
        )
        if not res.success:
            raise RuntimeError(
                f"Solver {solver_name} failed during spatial convergence run: {res.message}"
            )

        c_num = res["c"][-1, :]
        c_exact = np.sin(np.pi * grid / L) * decay_factor

        m = compute_error_metrics(c_num, c_exact)
        err_val = getattr(m, metric)

        dx_list.append(dx)
        error_list.append(err_val)

    dx_arr = np.array(dx_list, dtype=float)
    err_arr = np.array(error_list, dtype=float)

    order, r2 = estimate_convergence_order(dx_arr, err_arr)
    passed = bool(order >= (expected_order - order_tolerance) and r2 >= 0.95)

    return ConvergenceResult(
        spacings=dx_arr,
        errors=err_arr,
        observed_order=order,
        expected_order=expected_order,
        r_squared=r2,
        passed=passed,
        spacing_type="spatial",
        details={
            "solver_name": solver_name,
            "metric": metric,
            "n_points_list": list(n_points_list),
            "order_tolerance": order_tolerance,
        },
    )


def verify_temporal_convergence(
    solver: BaseSolver | str = "crank_nicolson",
    *,
    dt_list: Sequence[float] = (0.04, 0.02, 0.01, 0.005),
    n_points: int = 51,
    D: float = 0.05,
    L: float = 1.0,
    t_end: float = 0.2,
    reference: Literal["fine_step", "analytical"] = "fine_step",
    dt_ref: float = 1e-5,
    expected_order: float = 2.0,
    order_tolerance: float = 0.3,
    metric: Literal["max_abs_error", "rmse", "l2_error"] = "max_abs_error",
    **solver_kwargs: Any,
) -> ConvergenceResult:
    r"""Verify temporal discretization order across time-step refinement :math:`\Delta t`.

    Executes a sequence of temporal step refinements on the Fourier sine decay problem.
    To prevent spatial discretization errors from masking temporal convergence, the
    reference solution can be computed either analytically or via a high-resolution
    self-consistent numerical baseline (:attr:`reference='fine_step'`).

    Linear regression in logarithmic coordinates determines the temporal order :math:`p`:

    .. math::

        \ln(\text{error}) = p \cdot \ln(\Delta t) + C

    Parameters
    ----------
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    dt_list : sequence of float, default (0.04, 0.02, 0.01, 0.005)
        Sequence of time-step sizes :math:`\Delta t` to evaluate.
    n_points : int, default 51
        Number of spatial grid points :math:`N_x`.
    D : float, default 0.05
        Diffusion coefficient :math:`D > 0`.
    L : float, default 1.0
        Domain length :math:`L > 0`.
    t_end : float, default 0.2
        Simulation end time :math:`t_{\text{end}}`.
    reference : {'fine_step', 'analytical'}, default 'fine_step'
        Reference standard for error computation. ``'fine_step'`` (recommended)
        runs the solver on the exact same spatial grid with :attr:`dt_ref`,
        cleanly canceling spatial discretization error and isolating the temporal order.
    dt_ref : float, default 1e-5
        Reference time-step used when ``reference='fine_step'``.
    expected_order : float, default 2.0
        Expected theoretical temporal order (e.g. 2.0 for Crank–Nicolson, 1.0 for BTCS).
    order_tolerance : float, default 0.3
        Permissible deficit below :attr:`expected_order` to consider the test passed.
    metric : {'max_abs_error', 'rmse', 'l2_error'}, default 'max_abs_error'
        Error norm to regress against :math:`\Delta t`.
    **solver_kwargs : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    ConvergenceResult
        Dataclass containing time steps :math:`\Delta t`, error array,
        empirical temporal order :math:`p`, :math:`R^2`, and pass/fail indicator.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import verify_temporal_convergence
    >>> # Crank-Nicolson should exhibit 2nd-order temporal convergence (order ~ 2)
    >>> res = verify_temporal_convergence("crank_nicolson", dt_list=(0.04, 0.02, 0.01), expected_order=2.0)
    >>> res.passed
    True
    >>> round(res.observed_order, 1)
    2.0
    """
    solver_inst, solver_name = _resolve_solver_instance(solver)

    grid = np.linspace(0.0, L, n_points)
    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"c": lambda x_pts: np.sin(np.pi * x_pts / L)},
    )

    if reference == "fine_step":
        ref_kwargs = dict(solver_kwargs)
        ref_kwargs["dt"] = dt_ref
        ref_res = solver_inst.solve(
            problem,
            t_span=(0.0, t_end),
            t_eval=np.array([0.0, t_end]),
            **ref_kwargs,
        )
        if not ref_res.success:
            raise RuntimeError(
                f"Reference solver failed during temporal convergence run: {ref_res.message}"
            )
        c_ref = ref_res["c"][-1, :]
    else:
        decay_factor = math.exp(-D * (math.pi / L) ** 2 * t_end)
        c_ref = np.sin(np.pi * grid / L) * decay_factor

    dt_arr = np.array(dt_list, dtype=float)
    error_list: list[float] = []

    for dt_val in dt_arr:
        kwargs = dict(solver_kwargs)
        kwargs["dt"] = float(dt_val)

        res = solver_inst.solve(
            problem,
            t_span=(0.0, t_end),
            t_eval=np.array([0.0, t_end]),
            **kwargs,
        )
        if not res.success:
            raise RuntimeError(
                f"Solver {solver_name} failed at dt={dt_val}: {res.message}"
            )

        c_num = res["c"][-1, :]
        m = compute_error_metrics(c_num, c_ref)
        error_list.append(getattr(m, metric))

    err_arr = np.array(error_list, dtype=float)
    order, r2 = estimate_convergence_order(dt_arr, err_arr)
    passed = bool(order >= (expected_order - order_tolerance) and r2 >= 0.95)

    return ConvergenceResult(
        spacings=dt_arr,
        errors=err_arr,
        observed_order=order,
        expected_order=expected_order,
        r_squared=r2,
        passed=passed,
        spacing_type="temporal",
        details={
            "solver_name": solver_name,
            "metric": metric,
            "reference": reference,
            "dt_list": list(dt_list),
            "order_tolerance": order_tolerance,
        },
    )


# ==============================================================================
# 5. Benchmark Suite Orchestration
# ==============================================================================


def list_benchmark_cases() -> list[str]:
    """Return list of standard benchmark case names."""
    return ["fourier_decay", "cottrell_step", "mass_conservation"]


def run_benchmark(
    case: Literal["fourier_decay", "cottrell_step", "cottrell", "mass_conservation"],
    solver: BaseSolver | str = "crank_nicolson",
    **kwargs: Any,
) -> BenchmarkCaseResult:
    """Unified entry point to run a specific benchmark case with a solver.

    Parameters
    ----------
    case : {'fourier_decay', 'cottrell_step', 'cottrell', 'mass_conservation'}
        Benchmark problem to execute.
    solver : BaseSolver or str, default "crank_nicolson"
        Numerical solver instance or registered string name.
    **kwargs : Any
        Parameters passed to the specific benchmark runner function.

    Returns
    -------
    BenchmarkCaseResult
        Outcome of the benchmark run.

    Raises
    ------
    ValueError
        If an unrecognized benchmark case name is provided.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import run_benchmark
    >>> res = run_benchmark("fourier_decay", "crank_nicolson")
    >>> res.passed
    True
    """
    case_norm = case.lower().strip()
    if case_norm == "fourier_decay":
        return run_fourier_decay_benchmark(solver, **kwargs)
    if case_norm in ("cottrell", "cottrell_step"):
        return run_cottrell_benchmark(solver, **kwargs)
    if case_norm == "mass_conservation":
        return run_mass_conservation_benchmark(solver, **kwargs)

    raise ValueError(
        f"Unknown benchmark case '{case}'. Available cases: {list_benchmark_cases()}."
    )


def run_benchmark_suite(
    solvers: Sequence[BaseSolver | str] | None = None,
    cases: (
        Sequence[
            Literal["fourier_decay", "cottrell_step", "cottrell", "mass_conservation"]
        ]
        | None
    ) = None,
    **kwargs: Any,
) -> BenchmarkSuiteResult:
    """Execute a comprehensive benchmark suite across multiple solvers and cases.

    Parameters
    ----------
    solvers : sequence of (BaseSolver or str), optional
        List of solvers to evaluate. Defaults to unconditionally stable solvers:
        ``('crank_nicolson', 'scipy_ivp', 'implicit')``.
    cases : sequence of str, optional
        List of benchmark case names to run. Defaults to all standard cases:
        ``('fourier_decay', 'cottrell_step', 'mass_conservation')``.
    **kwargs : Any
        Additional keyword arguments forwarded to benchmark runs.

    Returns
    -------
    BenchmarkSuiteResult
        Aggregated outcomes of all benchmark runs with summary reporting.

    Examples
    --------
    >>> from softpotato.analytical.benchmark import run_benchmark_suite
    >>> suite = run_benchmark_suite(solvers=["crank_nicolson", "scipy_ivp"], cases=["fourier_decay"])
    >>> suite.all_passed
    True
    >>> "crank_nicolson" in suite.summary()
    True
    """
    eval_solvers: Sequence[BaseSolver | str] = (
        ("crank_nicolson", "scipy_ivp", "implicit") if solvers is None else solvers
    )

    eval_cases: Sequence[
        Literal["fourier_decay", "cottrell_step", "cottrell", "mass_conservation"]
    ] = (
        ("fourier_decay", "cottrell_step", "mass_conservation")
        if cases is None
        else cases
    )

    all_results: list[BenchmarkCaseResult] = []
    t_start = time.perf_counter()

    for s in eval_solvers:
        for c in eval_cases:
            res = run_benchmark(c, s, **kwargs)
            all_results.append(res)

    total_time = time.perf_counter() - t_start
    all_passed = bool(all(res.passed for res in all_results))

    return BenchmarkSuiteResult(
        cases=all_results,
        total_time=total_time,
        all_passed=all_passed,
    )
