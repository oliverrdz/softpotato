"""
Numerical solver wrapper module (sp.solver.ivp).
Wraps stiff initial-value problem integrators from SciPy (e.g., BDF, Radau).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


def solve_ivp(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: np.ndarray | None = None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    **kwargs: Any,
) -> OdeResult:
    """
    Factory wrapping stiff initial-value problem integrators for time integration
    of semi-discrete PDE systems.

    Parameters:
    -----------
    fun : callable
        Right-hand side of the system: ``fun(t, y)``.
    t_span : tuple of floats
        Interval of integration ``(t_start, t_end)``.
    y0 : ndarray
        Initial state vector (must be 1D).
    t_eval : ndarray, optional
        Times at which to store the computed solution.
    method : str, default="BDF"
        Integration method. Typically stiff solvers like "BDF" or "Radau".
    rtol : float, default=1e-6
        Relative error tolerance.
    atol : float, default=1e-8
        Absolute error tolerance.
    **kwargs : dict
        Additional keyword arguments passed directly to ``scipy.integrate.solve_ivp``.

    Returns:
    --------
    OdeResult
        A SciPy OdeResult object containing the solution arrays (t, y, etc.).
    """
    if y0.ndim != 1:
        raise ValueError("Initial state y0 must be a 1D NumPy array.")

    solution = scipy_solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )

    if not solution.success:
        raise RuntimeError(f"Numerical integration failed: {solution.message}")

    return solution
