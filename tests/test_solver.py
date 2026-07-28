"""
Unit test suite for the numerical solver wrapper (sp.solver).
"""

import numpy as np
import pytest

from softpotato.solver.ivp import solve_ivp


def test_solve_ivp_exponential_decay() -> None:
    """Test solver on a simple stiff exponential decay ODE: dy/dt = -1000y."""

    def decay_fun(t: float, y: np.ndarray) -> np.ndarray:
        return -1000.0 * y

    t_span = (0.0, 0.01)
    y0 = np.array([1.0])
    t_eval = np.linspace(0.0, 0.01, 11)

    sol = solve_ivp(
        fun=decay_fun,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
    )

    assert sol.success
    assert sol.t.shape == (11,)
    assert sol.y.shape == (1, 11)

    # Analytical solution: y(t) = exp(-1000 * t)
    expected = np.exp(-1000.0 * sol.t)
    np.testing.assert_allclose(sol.y[0], expected, rtol=1e-4, atol=1e-4)


def test_solve_ivp_invalid_y0() -> None:
    """Test that non-1D y0 raises ValueError."""

    def dummy_fun(t: float, y: np.ndarray) -> np.ndarray:
        return y

    with pytest.raises(ValueError, match="Initial state y0 must be a 1D NumPy array"):
        solve_ivp(
            fun=dummy_fun,
            t_span=(0.0, 1.0),
            y0=np.array([[1.0]]),
        )


def test_solve_ivp_failure_handling() -> None:
    """Test that solver handles exceptions during function evaluation."""

    def bad_fun(t: float, y: np.ndarray) -> np.ndarray:
        raise ZeroDivisionError("Simulated integration error")

    with pytest.raises(ZeroDivisionError):
        solve_ivp(
            fun=bad_fun,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
        )
