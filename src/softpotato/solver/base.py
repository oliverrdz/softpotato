"""Base classes, problem definitions, and result structures for diffusion solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Literal
import numpy as np


@dataclass
class BoundaryCondition:
    r"""Specification of a 1D boundary condition.

    Parameters
    ----------
    btype : {'dirichlet', 'neumann'}
        Type of boundary condition.
        - ``'dirichlet'``: fixed concentration, :math:`c(x_{\text{bound}}, t) = \text{value}(t)`.
        - ``'neumann'``: fixed concentration gradient, :math:`\left.\frac{\partial c}{\partial x}\right|_{x_{\text{bound}}} = \text{value}(t)`.
    value : float or Callable[[float], float]
        Fixed value or a time-dependent callable :math:`f(t)` returning a float.

    Examples
    --------
    Constant concentration boundary condition:

    >>> from softpotato.solver import BoundaryCondition
    >>> bc = BoundaryCondition("dirichlet", 1.0)
    >>> bc.evaluate(t=5.0)
    1.0

    Time-dependent concentration gradient (Neumann flux):

    >>> bc_dynamic = BoundaryCondition("neumann", lambda t: -0.1 * t)
    >>> bc_dynamic.evaluate(t=2.0)
    -0.2
    """

    btype: Literal["dirichlet", "neumann"]
    value: float | Callable[[float], float]

    def evaluate(self, t: float) -> float:
        """Evaluate boundary value at time t."""
        if callable(self.value):
            return float(self.value(t))
        return float(self.value)


def DirichletBC(value: float | Callable[[float], float]) -> BoundaryCondition:
    r"""Convenience constructor for a Dirichlet boundary condition :math:`c = \text{value}`.

    Parameters
    ----------
    value : float or Callable[[float], float]
        Fixed concentration or time-dependent function :math:`f(t)`.

    Returns
    -------
    BoundaryCondition
        A boundary condition instance configured with ``btype='dirichlet'``.

    Examples
    --------
    >>> from softpotato.solver import DirichletBC
    >>> bc = DirichletBC(0.0)
    >>> bc.btype
    'dirichlet'
    >>> bc.evaluate(10.0)
    0.0
    """
    return BoundaryCondition(btype="dirichlet", value=value)


def NeumannBC(value: float | Callable[[float], float]) -> BoundaryCondition:
    r"""Convenience constructor for a Neumann boundary condition :math:`\frac{\partial c}{\partial x} = \text{value}`.

    Parameters
    ----------
    value : float or Callable[[float], float]
        Fixed gradient :math:`\partial c / \partial x` or time-dependent function :math:`g(t)`.

    Returns
    -------
    BoundaryCondition
        A boundary condition instance configured with ``btype='neumann'``.

    Examples
    --------
    >>> from softpotato.solver import NeumannBC
    >>> bc = NeumannBC(0.0)
    >>> bc.btype
    'neumann'
    >>> bc.evaluate(0.0)
    0.0
    """
    return BoundaryCondition(btype="neumann", value=value)


@dataclass
class DiffusionProblem:
    r"""Formulation of a 1D multi-species chemical diffusion problem.

    Governing equation for each species :math:`i`:

    .. math::

        \frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(\mathbf{c}, x, t)

    Parameters
    ----------
    grid : np.ndarray
        1D spatial grid points :math:`x \in [x_{\min}, x_{\max}]`.
    diffusivity : dict[str, float]
        Mapping from chemical species name to its diffusion coefficient :math:`D_i > 0`.
    boundary_conditions : dict[str, tuple[BoundaryCondition, BoundaryCondition]]
        Mapping from species name to a tuple of ``(left_bc, right_bc)``.
    initial_conditions : dict[str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]]
        Mapping from species name to initial concentration profile (scalar, array, or callable).
    reactions : Callable[[dict[str, np.ndarray], np.ndarray, float], dict[str, np.ndarray]], optional
        Reaction kinetics term returning ``{species: reaction_rate_array}``. Defaults to None.
    species : list[str], optional
        List of chemical species names. Inferred from diffusivity if not explicitly provided.

    Examples
    --------
    Setting up a 1D problem for two species ('A' and 'B'):

    >>> import numpy as np
    >>> from softpotato.solver import DiffusionProblem, DirichletBC, NeumannBC
    >>>
    >>> grid = np.linspace(0.0, 1.0, 50)
    >>> problem = DiffusionProblem(
    ...     grid=grid,
    ...     diffusivity={"A": 1e-4, "B": 2e-4},
    ...     boundary_conditions={
    ...         "A": (DirichletBC(0.0), DirichletBC(1.0)),
    ...         "B": (NeumannBC(0.0), DirichletBC(0.0)),
    ...     },
    ...     initial_conditions={"A": 1.0, "B": 0.0},
    ... )
    >>> problem.species
    ['A', 'B']
    >>> problem.n_points
    50
    >>> bool(np.isclose(problem.dx, 1.0 / 49.0))
    True
    """

    grid: np.ndarray
    diffusivity: dict[str, float]
    boundary_conditions: dict[str, tuple[BoundaryCondition, BoundaryCondition]]
    initial_conditions: dict[str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]]
    reactions: Callable[[dict[str, np.ndarray], np.ndarray, float], dict[str, np.ndarray]] | None = None
    species: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float)
        if self.grid.ndim != 1 or len(self.grid) < 3:
            raise ValueError("grid must be a 1D array with at least 3 points.")
        if np.any(np.diff(self.grid) <= 0):
            raise ValueError("grid points must be strictly monotonically increasing.")

        if not self.species:
            self.species = list(self.diffusivity.keys())

        for sp in self.species:
            if sp not in self.diffusivity:
                raise KeyError(f"Species '{sp}' missing in diffusivity dict.")
            if self.diffusivity[sp] <= 0:
                raise ValueError(f"Diffusivity for species '{sp}' must be positive.")
            if sp not in self.boundary_conditions:
                raise KeyError(f"Species '{sp}' missing in boundary_conditions dict.")
            if sp not in self.initial_conditions:
                raise KeyError(f"Species '{sp}' missing in initial_conditions dict.")

    @property
    def n_points(self) -> int:
        """Number of spatial grid points."""
        return len(self.grid)

    @property
    def dx(self) -> float:
        """Grid step size for uniform grids."""
        diffs = np.diff(self.grid)
        if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
            raise ValueError("Grid is non-uniform; uniform dx is not defined.")
        return float(diffs[0])

    @property
    def is_uniform(self) -> bool:
        """Check whether the spatial grid is uniformly spaced."""
        diffs = np.diff(self.grid)
        return bool(np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8))

    def get_initial_profile(self, sp: str) -> np.ndarray:
        """Evaluate initial concentration profile for a given species."""
        ic = self.initial_conditions[sp]
        if callable(ic):
            arr = np.asarray(ic(self.grid), dtype=float)
        elif np.isscalar(ic):
            arr = np.full_like(self.grid, float(ic), dtype=float)
        else:
            arr = np.asarray(ic, dtype=float)

        if arr.shape != self.grid.shape:
            raise ValueError(
                f"Initial condition for '{sp}' has shape {arr.shape}, expected {self.grid.shape}"
            )
        return arr


@dataclass
class SolverResult:
    """Standardized output container for diffusion simulations.

    Parameters
    ----------
    t : np.ndarray
        1D array of time steps of shape (N_t,).
    x : np.ndarray
        1D array of spatial coordinates of shape (N_x,).
    concentrations : dict[str, np.ndarray]
        Dictionary mapping species name to 2D concentration array of shape (N_t, N_x).
    fluxes : dict[str, np.ndarray], optional
        Dictionary mapping species name to surface flux at left boundary x=0 of shape (N_t,).
    success : bool, default True
        Flag indicating if the numerical solution completed successfully.
    message : str, default "Success"
        Descriptive status message.
    raw_output : Any, optional
        Underlying solver output (e.g., OdeResult for SciPy).

    Examples
    --------
    Accessing simulation results:

    >>> import numpy as np
    >>> from softpotato.solver import SolverResult
    >>>
    >>> res = SolverResult(
    ...     t=np.linspace(0, 1, 5),
    ...     x=np.linspace(0, 1, 10),
    ...     concentrations={"A": np.ones((5, 10))},
    ...     fluxes={"A": np.zeros(5)},
    ... )
    >>> res.species
    ['A']
    >>> res["A"].shape
    (5, 10)
    >>> res.fluxes["A"].shape
    (5,)
    """

    t: np.ndarray
    x: np.ndarray
    concentrations: dict[str, np.ndarray]
    fluxes: dict[str, np.ndarray] = field(default_factory=dict)
    success: bool = True
    message: str = "Success"
    raw_output: Any = None

    def __getitem__(self, species_name: str) -> np.ndarray:
        """Shorthand to access concentrations for a species: result['A']."""
        return self.concentrations[species_name]

    @property
    def species(self) -> list[str]:
        """List of species present in the solution."""
        return list(self.concentrations.keys())


class BaseSolver(ABC):
    """Abstract Base Class for all diffusion solvers.

    Supports automatic registration of subclasses using `name="..."`:

    Examples
    --------
    Creating and registering a user-made solver:

    >>> import numpy as np
    >>> from softpotato.solver import BaseSolver, SolverResult, get_solver, list_solvers
    >>>
    >>> class CustomSolver(BaseSolver, name="custom_demo"):
    ...     def _run_solver(self, problem, t_span, t_eval, **kwargs):
    ...         t = np.array([t_span[0], t_span[1]]) if t_eval is None else t_eval
    ...         c = {sp: np.zeros((len(t), problem.n_points)) for sp in problem.species}
    ...         return SolverResult(t=t, x=problem.grid, concentrations=c)
    >>>
    >>> "custom_demo" in list_solvers()
    True
    >>> my_solver = get_solver("custom_demo")
    >>> isinstance(my_solver, CustomSolver)
    True
    """

    _registry: ClassVar[dict[str, type[BaseSolver]]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name:
            cls._registry[name.lower()] = cls

    def __init__(self, **options: Any) -> None:
        """Initialize solver with optional solver-specific configuration."""
        self.options = options

    def solve(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """Solve the diffusion problem over the specified time span.

        Parameters
        ----------
        problem : DiffusionProblem
            Problem definition containing grid, species, diffusivity, and boundary conditions.
        t_span : tuple[float, float]
            Time integration interval (t_start, t_end).
        t_eval : np.ndarray, optional
            Specific time points to evaluate and store the solution. If None, default
            evaluation time points will be generated by the solver.
        **kwargs : Any
            Additional run-time options passed to the underlying solver engine.

        Returns
        -------
        SolverResult
            Standardized container holding time, coordinates, and concentration arrays.
        """
        self.validate(problem, t_span, t_eval)
        result = self._run_solver(problem, t_span, t_eval, **kwargs)
        return self._post_process(problem, result)

    @abstractmethod
    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        """Core numerical calculation loop or integrator dispatch."""
        raise NotImplementedError

    def validate(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
    ) -> None:
        """Sanity and physical validity checks before integration."""
        if not isinstance(problem, DiffusionProblem):
            raise TypeError(f"problem must be an instance of DiffusionProblem, got {type(problem)}")
        if t_span[0] >= t_span[1]:
            raise ValueError(f"t_span[1] must be strictly greater than t_span[0], got {t_span}")
        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float)
            if t_eval.ndim != 1:
                raise ValueError("t_eval must be a 1D array.")
            if len(t_eval) < 2:
                raise ValueError("t_eval must have at least 2 points.")
            if t_eval[0] < t_span[0] or t_eval[-1] > t_span[1]:
                raise ValueError(f"t_eval [{t_eval[0]}, {t_eval[-1]}] must lie within t_span {t_span}.")
            if np.any(np.diff(t_eval) <= 0):
                raise ValueError("t_eval points must be strictly monotonically increasing.")

    def _post_process(self, problem: DiffusionProblem, result: SolverResult) -> SolverResult:
        """Compute surface fluxes and perform post-run integrity checks."""
        # Calculate surface flux J = -D * dc/dx at left boundary x=0 using 2nd order difference
        if problem.is_uniform:
            dx = problem.dx
            for sp in problem.species:
                if sp not in result.fluxes:
                    c = result.concentrations[sp]
                    D = problem.diffusivity[sp]
                    # 2nd-order forward difference at x=0: (-3*c0 + 4*c1 - c2) / (2*dx)
                    dcdx_left = (-3.0 * c[:, 0] + 4.0 * c[:, 1] - c[:, 2]) / (2.0 * dx)
                    result.fluxes[sp] = -D * dcdx_left
        return result


def register_solver(name: str) -> Callable[[type[BaseSolver]], type[BaseSolver]]:
    """Decorator to register a custom solver class under a given name.

    Parameters
    ----------
    name : str
        Name identifier to associate with the decorated solver class.

    Returns
    -------
    Callable[[type[BaseSolver]], type[BaseSolver]]
        Decorator registering the class into BaseSolver._registry.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.solver import BaseSolver, SolverResult, register_solver, get_solver
    >>>
    >>> @register_solver("spectral_decorator_demo")
    ... class SpectralSolver(BaseSolver):
    ...     def _run_solver(self, problem, t_span, t_eval, **kwargs):
    ...         return SolverResult(t=np.array([0, 1]), x=problem.grid, concentrations={})
    >>>
    >>> solver = get_solver("spectral_decorator_demo")
    >>> isinstance(solver, SpectralSolver)
    True
    """

    def decorator(cls: type[BaseSolver]) -> type[BaseSolver]:
        if not issubclass(cls, BaseSolver):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseSolver.")
        BaseSolver._registry[name.lower()] = cls
        return cls

    return decorator


def get_solver(name: str, **kwargs: Any) -> BaseSolver:
    """Instantiate a solver by its registered name.

    Parameters
    ----------
    name : str
        Registered solver identifier (case-insensitive), e.g. 'scipy_ivp', 'crank_nicolson',
        'implicit', 'explicit'.
    **kwargs : Any
        Options passed to the solver's `__init__`.

    Returns
    -------
    BaseSolver
        An instance of the requested solver.

    Examples
    --------
    >>> from softpotato.solver import get_solver
    >>> solver = get_solver("scipy_ivp", method="Radau")
    >>> solver.options["method"]
    'Radau'
    >>> solver_cn = get_solver("crank_nicolson", dt=0.01)
    >>> solver_cn.options["dt"]
    0.01
    """
    normalized = name.lower()
    if normalized not in BaseSolver._registry:
        available = ", ".join(sorted(BaseSolver._registry.keys()))
        raise KeyError(f"Unknown solver '{name}'. Available solvers: {available}")
    return BaseSolver._registry[normalized](**kwargs)


def list_solvers() -> list[str]:
    """Return a list of all currently registered solver names.

    Returns
    -------
    list[str]
        Alphabetically sorted list of registered solver names.

    Examples
    --------
    >>> from softpotato.solver import list_solvers
    >>> solvers = list_solvers()
    >>> "scipy_ivp" in solvers
    True
    >>> "crank_nicolson" in solvers
    True
    """
    return sorted(BaseSolver._registry.keys())
