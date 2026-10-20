"""Tests for diffusion solvers and extensible BaseSolver architecture."""

from typing import ClassVar

import numpy as np
import pytest
from scipy.integrate import trapezoid

from softpotato.analytical.microelectrodes import microsphere_limiting_current
from softpotato.analytical.step import cottrell_spherical
from softpotato.geometry import PlanarGeometry, SphericalGeometry
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


def test_user_defined_solver_subclass_with_solver_name():
    """Verify solver registration using solver_name keyword argument."""

    class MockSolverName(BaseSolver, solver_name="mock_solver_name"):
        def _run_solver(self, problem, t_span, t_eval, **kwargs):
            t = np.array([t_span[0], t_span[1]]) if t_eval is None else t_eval
            return SolverResult(t=t, x=problem.grid, concentrations={})

    assert "mock_solver_name" in list_solvers()
    assert isinstance(get_solver("mock_solver_name"), MockSolverName)


def test_python_310_abcmeta_name_collision_simulation():
    """Simulate Python 3.10 ABCMeta.__new__ parameter collision with 'name'."""
    from softpotato.solver.base import _BaseSolverMeta

    # In Python <= 3.10, ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)
    # had 'name' as positional-or-keyword, not positional-only.
    class SimulatedPy310ABCMeta(type):
        def __new__(mcls, name, bases, namespace, **kwargs):
            return super().__new__(mcls, name, bases, namespace, **kwargs)

    class SimulatedSolverMeta(_BaseSolverMeta, SimulatedPy310ABCMeta):
        pass

    class SimulatedBaseSolver(metaclass=SimulatedSolverMeta):
        _registry: ClassVar[dict[str, type]] = {}

    # This raises TypeError in Python <= 3.10 if 'name' is not popped by _BaseSolverMeta.__new__.
    class SubSolver(SimulatedBaseSolver, name="simulated_sub_solver"):
        pass

    assert "simulated_sub_solver" in SimulatedBaseSolver._registry


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


@pytest.mark.parametrize(
    "solver_name",
    ["explicit", "implicit", "crank_nicolson", "scipy_ivp"],
)
def test_left_and_right_boundary_fluxes(solver_name):
    """Verify left_fluxes and right_fluxes calculation and electrode-normal sign convention."""
    L = 2.0
    D = 0.05
    x = np.linspace(0.0, L, 41)
    t_eval = np.linspace(0.0, 0.2, 6)

    # Steady-state generation-collection profile: c(x) = 1 - x / L
    # Left electrode (x=0) generates species (c=1 -> flux into solution > 0)
    # Right electrode (x=L) consumes species (c=0 -> flux into solution < 0)
    problem = DiffusionProblem(
        grid=x,
        diffusivity={"A": D},
        boundary_conditions={"A": (DirichletBC(1.0), DirichletBC(0.0))},
        initial_conditions={"A": lambda x_pts: 1.0 - x_pts / L},
    )

    solver = get_solver(solver_name)
    result = solver.solve(problem, t_span=(0.0, 0.2), t_eval=t_eval)

    assert result.left_fluxes is result.fluxes
    assert "A" in result.left_fluxes
    assert "A" in result.right_fluxes

    expected_left = np.full_like(t_eval, D / L)
    expected_right = np.full_like(t_eval, -D / L)

    np.testing.assert_allclose(
        result.left_fluxes["A"], expected_left, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        result.right_fluxes["A"], expected_right, rtol=1e-5, atol=1e-7
    )


def test_diffusion_problem_geometry_resolution():
    """Verify DiffusionProblem geometry resolution, defaults, and boundary validation."""
    grid = np.linspace(0.0, 1.0, 21)

    # 1. Default geometry is PlanarGeometry with area=1.0
    prob_default = DiffusionProblem(
        grid=grid,
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
        initial_conditions={"A": 0.0},
    )
    assert isinstance(prob_default.geometry, PlanarGeometry)
    assert prob_default.geometry.area == 1.0

    # 2. String alias resolution
    prob_planar = DiffusionProblem(
        grid=grid,
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
        initial_conditions={"A": 0.0},
        geometry="planar",
    )
    assert isinstance(prob_planar.geometry, PlanarGeometry)

    # 3. Spherical geometry string alias and r0 inference
    r_grid = np.linspace(25e-6, 100e-6, 31)
    prob_sph = DiffusionProblem(
        grid=r_grid,
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
        initial_conditions={"A": 0.0},
        geometry="spherical",
    )
    assert isinstance(prob_sph.geometry, SphericalGeometry)
    assert np.isclose(prob_sph.geometry.r0, 25e-6)

    # 4. Explicit Geometry instance
    geom_inst = SphericalGeometry(r0=10e-6, hemisphere=True)
    prob_inst = DiffusionProblem(
        grid=np.linspace(10e-6, 50e-6, 21),
        diffusivity={"A": 1.0},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
        initial_conditions={"A": 0.0},
        geometry=geom_inst,
    )
    assert prob_inst.geometry is geom_inst
    assert prob_inst.geometry.hemisphere is True

    # 5. Invalid geometry type
    with pytest.raises(TypeError, match="geometry must be a str or Geometry instance"):
        DiffusionProblem(
            grid=grid,
            diffusivity={"A": 1.0},
            boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
            initial_conditions={"A": 0.0},
            geometry=12345,  # type: ignore[arg-type]
        )

    # 6. Radial coordinate constraint (r0 > 0)
    with pytest.raises(ValueError, match="strictly positive radial coordinates"):
        DiffusionProblem(
            grid=np.linspace(0.0, 1.0, 21),  # grid[0] == 0.0 violates r0 > 0
            diffusivity={"A": 1.0},
            boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
            initial_conditions={"A": 0.0},
            geometry="spherical",
        )


@pytest.mark.parametrize(
    "solver_name",
    ["explicit", "implicit", "crank_nicolson", "scipy_ivp"],
)
def test_spherical_cottrell_benchmark(solver_name):
    """Verify numerical spherical step response against analytical cottrell_spherical.

    Acceptance criterion: relative error < 0.5% across the transient (t in [0.05, 2.0] s).
    """
    n = 1
    D = 1.0e-9  # m^2/s
    c_bulk = 1.0  # mol/m^3
    r0 = 25.0e-6  # 25 um
    t_max = 2.0  # seconds
    delta = 6.0 * np.sqrt(D * t_max)

    grid = np.linspace(r0, r0 + delta, 250)
    geom = SphericalGeometry(r0=r0)

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
        initial_conditions={"c": c_bulk},
        geometry=geom,
        n_electrons=n,
    )

    t_eval = np.linspace(0.05, t_max, 50)
    solver_kwargs = {}
    if solver_name in ("crank_nicolson", "implicit"):
        solver_kwargs["dt"] = 5e-4

    solver = get_solver(solver_name, **solver_kwargs)
    result = solver.solve(problem, t_span=(0.0, t_max), t_eval=t_eval)

    assert result.success
    assert "c" in result.currents
    assert result.currents["c"].shape == (len(t_eval),)

    # Cathodic reduction currents are negative under IUPAC convention
    assert np.all(result.currents["c"] < 0.0)

    i_num = np.abs(result.currents["c"])
    i_ana = cottrell_spherical(t_eval, r=r0, n=n, D=D, c_bulk=c_bulk, area=geom.area)

    rel_error = np.max(np.abs(i_num - i_ana) / i_ana)
    assert rel_error < 0.005, (
        f"{solver_name} exceeded 0.5% tolerance on spherical Cottrell: {rel_error * 100:.3f}%"
    )


def test_spherical_steady_state_limiting_current():
    """Verify long-time numerical current converges toward microsphere_limiting_current."""
    n = 1
    D = 1.0e-9  # m^2/s
    c_bulk = 1.0  # mol/m^3
    r0 = 5.0e-6  # 5 um microelectrode
    t_max = 20.0  # seconds
    delta = 8.0 * np.sqrt(D * t_max)

    # Use non-uniform grid to resolve the boundary layer near r0 accurately
    grid = np.geomspace(r0, r0 + delta, 180)
    geom = SphericalGeometry(r0=r0)

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
        initial_conditions={"c": c_bulk},
        geometry=geom,
        n_electrons=n,
    )

    solver = get_solver("scipy_ivp")
    result = solver.solve(problem, t_span=(0.0, t_max), t_eval=[1.0, t_max])

    assert result.success
    i_num_last = np.abs(result.currents["c"][-1])
    i_ss = microsphere_limiting_current(r0, n=n, D=D, c_bulk=c_bulk)

    # At t = 20 s (theta = D*t/r0^2 = 800), transient is within 2.5% of steady-state
    rel_diff_ss = abs(i_num_last - i_ss) / i_ss
    assert rel_diff_ss < 0.025, (
        f"Long-time spherical current did not match steady-state: {rel_diff_ss * 100:.2f}%"
    )

    # And matches analytical Cottrell spherical transient to < 0.1%
    i_ana_last = cottrell_spherical(
        t_max, r=r0, n=n, D=D, c_bulk=c_bulk, area=geom.area
    )
    rel_diff_ana = abs(i_num_last - i_ana_last) / i_ana_last
    assert rel_diff_ana < 0.001, (
        f"Long-time spherical current did not match analytical Cottrell: {rel_diff_ana * 100:.3f}%"
    )


def test_diffusion_problem_electrodes_option():
    """Verify DiffusionProblem options for electrodes ('left', 'right', 'both')."""
    # 1. Parameter validation
    with pytest.raises(
        ValueError, match="electrodes must be 'left', 'right', or 'both'"
    ):
        DiffusionProblem(
            grid=np.linspace(0, 1, 10),
            diffusivity={"A": 1e-5},
            boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
            initial_conditions={"A": 1.0},
            electrodes="center",  # type: ignore[arg-type]
        )

    # 2. Symmetric two-electrode cell (electrodes='both')
    D = 1e-9
    c0 = 1.0
    L = 1e-4
    grid = np.linspace(0, L, 51)
    t_eval = np.linspace(0.01, 0.2, 10)

    problem_both = DiffusionProblem(
        grid=grid,
        diffusivity={"A": D},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(0.0))},
        initial_conditions={"A": c0},
        electrodes="both",
        n_electrons=1,
    )
    assert problem_both.electrodes == "both"
    assert problem_both.electrode_side == "both"

    solver = get_solver("scipy_ivp")
    res_both = solver.solve(problem_both, t_span=(0.0, 0.2), t_eval=t_eval)

    assert "A" in res_both.left_currents
    assert "A" in res_both.right_currents
    assert "A" in res_both.currents

    # In symmetric cell with identical Dirichlet BCs, left and right currents must be identical
    np.testing.assert_allclose(
        res_both.left_currents["A"],
        res_both.right_currents["A"],
        rtol=1e-5,
    )
    # Total current must equal sum of left and right currents
    np.testing.assert_allclose(
        res_both.currents["A"],
        res_both.left_currents["A"] + res_both.right_currents["A"],
        rtol=1e-12,
    )

    # 3. Right electrode only (electrodes='right')
    problem_right = DiffusionProblem(
        grid=grid,
        diffusivity={"A": D},
        boundary_conditions={"A": (DirichletBC(c0), DirichletBC(0.0))},
        initial_conditions={"A": c0},
        electrode_side="right",  # test alias
        n_electrons=1,
    )
    res_right = solver.solve(problem_right, t_span=(0.0, 0.2), t_eval=t_eval)

    assert "A" in res_right.right_currents
    assert "A" not in res_right.left_currents
    np.testing.assert_allclose(
        res_right.currents["A"],
        res_right.right_currents["A"],
        rtol=1e-12,
    )
    # Cathodic reduction at right electrode must be negative
    assert np.all(res_right.currents["A"] < 0.0)

    # 4. Left electrode only (default)
    problem_left = DiffusionProblem(
        grid=grid,
        diffusivity={"A": D},
        boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(c0))},
        initial_conditions={"A": c0},
        n_electrons=1,
    )
    res_left = solver.solve(problem_left, t_span=(0.0, 0.2), t_eval=t_eval)

    assert "A" in res_left.left_currents
    assert "A" not in res_left.right_currents
    np.testing.assert_allclose(
        res_left.currents["A"],
        res_left.left_currents["A"],
        rtol=1e-12,
    )
