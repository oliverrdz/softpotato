"""Tests for diffusion solvers and extensible BaseSolver architecture."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from softpotato.solver import (
    BaseSolver,
    CrankNicolson,
    DiffusionProblem,
    DirichletBC,
    ExplicitFiniteDifference,
    ImplicitFiniteDifference,
    NeumannBC,
    ScipyIVPSolver,
    SolverResult,
    get_solver,
    list_solvers,
    register_solver,
)


def test_solver_registry():
    """Verify built-in solvers are registered and discoverable."""
    solvers = list_solvers()
    assert "explicit" in solvers
    assert "implicit" in solvers
    assert "crank_nicolson" in solvers
    assert "scipy_ivp" in solvers
    assert "ftcs" in solvers
    assert "btcs" in solvers
    assert "solve_ivp" in solvers

    assert isinstance(get_solver("explicit"), ExplicitFiniteDifference)
    assert isinstance(get_solver("implicit"), ImplicitFiniteDifference)
    assert isinstance(get_solver("crank_nicolson"), CrankNicolson)
    assert isinstance(get_solver("scipy_ivp"), ScipyIVPSolver)

    with pytest.raises(KeyError, match="Unknown solver"):
        get_solver("non_existent_solver")


def test_user_defined_solver_subclass():
    """Verify a user can create a custom solver via subclassing and auto-register it."""

    class MockUserSolver(BaseSolver, name="mock_custom_subclass"):
        def _run_solver(self, problem, t_span, t_eval, **kwargs):
            t = np.array([t_span[0], t_span[1]]) if t_eval is None else t_eval
            records = {
                sp: np.ones((len(t), problem.n_points)) for sp in problem.species
            }
            return SolverResult(
                t=t,
                x=problem.grid,
                concentrations=records,
                success=True,
                message="Mock success",
            )

    assert "mock_custom_subclass" in list_solvers()
    solver = get_solver("mock_custom_subclass")
    assert isinstance(solver, MockUserSolver)

    problem = DiffusionProblem(
        grid=np.linspace(0, 1, 11),
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"A": 1.0},
    )
    result = solver.solve(problem, t_span=(0.0, 1.0))
    assert result.success
    assert "A" in result.concentrations
    assert result["A"].shape == (2, 11)


def test_user_defined_solver_decorator():
    """Verify a user can register a custom solver using @register_solver."""

    @register_solver("mock_decorated_solver")
    class DecoratedSolver(BaseSolver):
        def _run_solver(self, problem, t_span, t_eval, **kwargs):
            t = np.array([t_span[0], t_span[1]]) if t_eval is None else t_eval
            return SolverResult(
                t=t,
                x=problem.grid,
                concentrations={
                    sp: np.zeros((len(t), problem.n_points)) for sp in problem.species
                },
            )

    assert "mock_decorated_solver" in list_solvers()
    solver = get_solver("mock_decorated_solver")
    assert isinstance(solver, DecoratedSolver)


@pytest.mark.parametrize(
    "solver_name",
    ["explicit", "implicit", "crank_nicolson", "scipy_ivp"],
)
def test_analytical_dirichlet_benchmark(solver_name):
    """Test all solvers against exact analytical solution for 1D diffusion.

    Equation: dc/dt = D * d²c/dx² on x in [0, 1]
    Initial: c(x, 0) = sin(pi * x)
    Boundary: c(0, t) = c(1, t) = 0
    Exact: c(x, t) = sin(pi * x) * exp(-D * pi² * t)
    """
    L = 1.0
    D = 0.05
    x = np.linspace(0, L, 41)
    t_eval = np.linspace(0.0, 0.5, 11)

    problem = DiffusionProblem(
        grid=x,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"c": lambda x_pts: np.sin(np.pi * x_pts / L)},
    )

    solver = get_solver(solver_name)
    result = solver.solve(problem, t_span=(0.0, 0.5), t_eval=t_eval)

    assert result.success
    c_num = result["c"]

    # Compute analytical solution at all (t, x)
    T, X = np.meshgrid(t_eval, x, indexing="ij")
    c_exact = np.sin(np.pi * X / L) * np.exp(-D * (np.pi / L) ** 2 * T)

    # Maximum absolute error across all space and time points
    max_error = np.max(np.abs(c_num - c_exact))
    assert max_error < 5e-3, (
        f"{solver_name} exceeded tolerance: max error = {max_error:.4e}"
    )


@pytest.mark.parametrize(
    "solver_name",
    ["explicit", "implicit", "crank_nicolson", "scipy_ivp"],
)
def test_neumann_mass_conservation(solver_name):
    """Verify total mass conservation under zero-flux Neumann boundary conditions."""
    x = np.linspace(0, 1.0, 51)
    # Gaussian pulse centered at x=0.5
    init_func = lambda x_pts: np.exp(-100.0 * (x_pts - 0.5) ** 2)

    problem = DiffusionProblem(
        grid=x,
        diffusivity={"A": 0.02},
        boundary_conditions={"A": (NeumannBC(0.0), NeumannBC(0.0))},
        initial_conditions={"A": init_func},
    )

    solver = get_solver(solver_name)
    t_eval = np.linspace(0.0, 0.2, 5)
    result = solver.solve(problem, t_span=(0.0, 0.2), t_eval=t_eval)

    assert result.success
    c = result["A"]

    # Integrate total mass at each recorded time step
    initial_mass = trapezoid(c[0, :], x)
    for k in range(len(t_eval)):
        current_mass = trapezoid(c[k, :], x)
        rel_diff = abs(current_mass - initial_mass) / initial_mass
        assert rel_diff < 1e-3, (
            f"{solver_name} failed mass conservation at step {k}: {rel_diff:.4e}"
        )


def test_explicit_cfl_validation():
    """Verify ExplicitFiniteDifference raises ValueError when dt violates CFL condition."""
    x = np.linspace(0, 1.0, 21)  # dx = 0.05
    D = 1.0
    # CFL limit: dx² / (2*D) = 0.0025 / 2 = 1.25e-3
    problem = DiffusionProblem(
        grid=x,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"c": 1.0},
    )

    solver = ExplicitFiniteDifference(dt=0.01)  # 0.01 >> 1.25e-3 (violates CFL)
    with pytest.raises(ValueError, match="exceeds the CFL stability limit"):
        solver.solve(problem, t_span=(0.0, 0.1))


def test_multi_species_diffusion():
    """Verify multi-species simulation with different diffusion coefficients."""
    x = np.linspace(0, 1.0, 31)
    problem = DiffusionProblem(
        grid=x,
        diffusivity={"fast": 0.1, "slow": 0.01},
        boundary_conditions={
            "fast": (DirichletBC(0.0), DirichletBC(1.0)),
            "slow": (DirichletBC(0.0), DirichletBC(1.0)),
        },
        initial_conditions={"fast": 0.0, "slow": 0.0},
    )

    for s_name in ["scipy_ivp", "crank_nicolson", "explicit", "implicit"]:
        solver = get_solver(s_name)
        result = solver.solve(problem, t_span=(0.0, 0.1))
        assert "fast" in result.concentrations
        assert "slow" in result.concentrations
        # The faster species must have diffused further into the domain
        assert np.mean(result["fast"][-1, :]) > np.mean(result["slow"][-1, :])


def test_reaction_diffusion():
    """Verify homogeneous reaction kinetics R_A = -k * c_A."""
    k_rate = 2.0
    x = np.linspace(0, 1.0, 21)

    def reactions(c_dict, x_pts, t):
        return {"A": -k_rate * c_dict["A"]}

    problem = DiffusionProblem(
        grid=x,
        diffusivity={"A": 0.01},
        boundary_conditions={"A": (NeumannBC(0.0), NeumannBC(0.0))},
        initial_conditions={"A": 1.0},
        reactions=reactions,
    )

    solver = get_solver("scipy_ivp")
    t_eval = np.linspace(0.0, 0.5, 6)
    result = solver.solve(problem, t_span=(0.0, 0.5), t_eval=t_eval)

    assert result.success
    # With uniform initial condition and zero-flux BCs, c_A(t) = exp(-k * t) uniformly
    for i, t_val in enumerate(t_eval):
        c_expected = np.exp(-k_rate * t_val)
        c_sim = result["A"][i, :]
        np.testing.assert_allclose(c_sim, c_expected, rtol=1e-3, atol=1e-3)


def test_input_validation():
    """Verify invalid parameters trigger helpful exceptions."""
    # Non-monotonic grid
    with pytest.raises(ValueError, match="monotonically increasing"):
        DiffusionProblem(
            grid=np.array([0.0, 0.5, 0.2]),
            diffusivity={"A": 1.0},
            boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(0.0))},
            initial_conditions={"A": 1.0},
        )

    # Inverted t_span
    prob = DiffusionProblem(
        grid=np.linspace(0, 1, 10),
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"A": 1.0},
    )
    solver = get_solver("scipy_ivp")
    with pytest.raises(ValueError, match="strictly greater"):
        solver.solve(prob, t_span=(1.0, 0.0))
