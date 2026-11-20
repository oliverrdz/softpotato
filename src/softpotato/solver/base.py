"""Base classes, problem definitions, and result structures for diffusion solvers."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np

from softpotato.geometry import Geometry, get_geometry
from softpotato.grid.base import Mesh1D

if TYPE_CHECKING:
    from softpotato.reactions.homogeneous import HomogeneousReaction, ReactionNetwork
    from softpotato.reactions.mechanisms import Mechanism
    from softpotato.reactions.species import Species, SpeciesCollection


@dataclass
class BoundaryCondition:
    r"""Specification of a 1D boundary condition.

    Parameters
    ----------
    btype : {'dirichlet', 'neumann'}
        Type of boundary condition:

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


@dataclass(init=False)
class DiffusionProblem:
    r"""Formulation of a 1D multi-species chemical diffusion problem.

    Governing equation for each species :math:`i`:

    .. math::

        \frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(\mathbf{c}, x, t)

    Parameters
    ----------
    grid : np.ndarray or Mesh1D
        1D spatial grid points :math:`x \in [x_{\min}, x_{\max}]` or a
        :class:`~softpotato.grid.Mesh1D` mesh object (e.g.,
        :class:`~softpotato.grid.Uniform1DMesh`,
        :class:`~softpotato.grid.GeometricExpandingMesh`,
        :class:`~softpotato.grid.ExponentialMesh`).
    diffusivity : dict[str, float], optional
        Mapping from chemical species name to its diffusion coefficient :math:`D_i > 0`.
        Automatically populated from ``Species`` objects when ``species`` contains
        ``Species`` instances or a ``SpeciesCollection`` (or when ``mechanism`` is
        supplied). Explicit entries override species defaults.
    boundary_conditions : dict[str, tuple[BoundaryCondition, BoundaryCondition]]
        Mapping from species name to a tuple of ``(left_bc, right_bc)``.
    initial_conditions : dict[str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]], optional
        Mapping from species name to initial concentration profile (scalar, array, or callable).
        Automatically populated from ``Species.bulk_concentration`` when
        ``Species`` instances are provided. Explicit entries override species defaults.
    reactions : ReactionNetwork, HomogeneousReaction, Mechanism, or Callable, optional
        Reaction kinetics term returning ``{species: reaction_rate_array}``.
        When a ``ReactionNetwork``, ``HomogeneousReaction``, ``Mechanism``, or
        any object implementing ``local_jacobian`` / ``jacobian`` is provided,
        stiff solvers such as :class:`ScipyIVPSolver` (``Radau``, ``BDF``)
        automatically assemble the coupled analytical diffusion-reaction
        Jacobian. Defaults to None.
    species : Sequence[str | Species] or SpeciesCollection, optional
        List of chemical species names, ``Species`` instances, or a
        ``SpeciesCollection``. Normalized to ``list[str]`` after extracting
        species metadata. Inferred from ``mechanism`` or ``diffusivity`` if not
        explicitly provided.
    geometry : Geometry | str, default 'planar'
        Coordinate geometry specifying differential operators (:math:`\nabla^2`),
        normal surface flux evaluations, and active electrode area. Can be a
        registered alias (e.g., ``'planar'``, ``'spherical'``, ``'cylindrical'``)
        or a :class:`~softpotato.geometry.Geometry` instance. Defaults to ``'planar'``.
    n_electrons : int | dict[str, int], optional
        Number of electrons transferred per reaction for Faradaic current
        calculations. Can be an integer or a species-to-integer mapping.
        Automatically populated from ``Species.n_electrons`` when ``Species``
        instances define it, otherwise defaults to ``1``.
    electrodes : {'left', 'right', 'both'}, default 'left'
        Domain boundary or boundaries configured as active electrodes for Faradaic
        current evaluation. Defaults to ``'left'``. If ``'both'``, currents from both
        sides are evaluated on ``SolverResult.left_currents`` and ``SolverResult.right_currents``,
        and summed on ``SolverResult.currents``.
    electrode_side : {'left', 'right', 'both'}, optional
        Alias for ``electrodes``.
    mechanism : Mechanism, optional
        High-level chemical or electrochemical ``Mechanism`` (including
        ``CustomMechanism``). Automatically populates ``species``,
        ``diffusivity``, ``initial_conditions``, ``n_electrons``, and
        ``reactions`` when not explicitly provided.

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

    grid: np.ndarray | Mesh1D
    diffusivity: dict[str, float]
    boundary_conditions: dict[str, tuple[BoundaryCondition, BoundaryCondition]]
    initial_conditions: dict[
        str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]
    ]
    reactions: (
        ReactionNetwork
        | HomogeneousReaction
        | Mechanism
        | Callable[[dict[str, np.ndarray], np.ndarray, float], dict[str, np.ndarray]]
        | None
    )
    species: list[str]
    geometry: Geometry | str
    n_electrons: int | dict[str, int]
    electrodes: Literal["left", "right", "both"]
    electrode_side: Literal["left", "right", "both"] | None
    mechanism: Mechanism | None
    mesh: Mesh1D | None = field(default=None, repr=False)
    species_objects: dict[str, Species] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        grid: np.ndarray | Mesh1D,
        diffusivity: Mapping[str, float] | None = None,
        boundary_conditions: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        initial_conditions: Mapping[
            str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]
        ]
        | None = None,
        reactions: (
            ReactionNetwork
            | HomogeneousReaction
            | Mechanism
            | Callable[
                [dict[str, np.ndarray], np.ndarray, float], dict[str, np.ndarray]
            ]
            | None
        ) = None,
        species: Sequence[str | Species] | SpeciesCollection | None = None,
        geometry: Geometry | str = "planar",
        n_electrons: int | Mapping[str, int] | None = None,
        electrodes: Literal["left", "right", "both"] = "left",
        electrode_side: Literal["left", "right", "both"] | None = None,
        mechanism: Mechanism | None = None,
    ) -> None:
        from softpotato.reactions.mechanisms import Mechanism
        from softpotato.reactions.species import Species, SpeciesCollection

        if electrode_side is not None:
            electrodes = electrode_side
        if electrodes not in ("left", "right", "both"):
            raise ValueError(
                f"electrodes must be 'left', 'right', or 'both', got {electrodes!r}."
            )
        self.electrodes = electrodes
        self.electrode_side = electrodes

        if isinstance(geometry, str):
            resolved_geom: Geometry = get_geometry(geometry)
        elif isinstance(geometry, Geometry):
            resolved_geom = geometry
        else:
            raise TypeError(
                f"geometry must be a str or Geometry instance, got {type(geometry)}."
            )
        self.geometry = resolved_geom

        if isinstance(grid, Mesh1D):
            self.mesh = grid
            self.grid = resolved_geom.validate_grid(grid.nodes)
        else:
            self.mesh = None
            self.grid = resolved_geom.validate_grid(grid)

        # Copy caller dicts so auto-population never mutates external mappings
        self.diffusivity = dict(diffusivity) if diffusivity is not None else {}
        self.boundary_conditions = (
            dict(boundary_conditions) if boundary_conditions is not None else {}
        )
        self.initial_conditions = (
            dict(initial_conditions) if initial_conditions is not None else {}
        )
        self.reactions = reactions
        self.mechanism = mechanism

        # Populate from Mechanism when provided
        raw_species: Sequence[str | Species] | SpeciesCollection | None = species
        if mechanism is not None:
            if not isinstance(mechanism, Mechanism):
                raise TypeError(
                    "mechanism must be an instance of Mechanism, "
                    f"got {type(mechanism).__name__}."
                )
            if not raw_species:
                raw_species = mechanism.species
            if self.reactions is None:
                self.reactions = mechanism

        # Normalize species input (list[str], Sequence[str | Species], SpeciesCollection)
        species_names: list[str] = []
        species_objs: dict[str, Species] = {}

        if isinstance(raw_species, SpeciesCollection):
            for sp_obj in raw_species.values():
                species_names.append(sp_obj.name)
                species_objs[sp_obj.name] = sp_obj
        elif raw_species:
            for item in raw_species:
                if isinstance(item, Species):
                    if item.name in species_names:
                        raise ValueError(
                            f"Duplicate species '{item.name}' in species list."
                        )
                    species_names.append(item.name)
                    species_objs[item.name] = item
                elif isinstance(item, str):
                    if item in species_names:
                        raise ValueError(f"Duplicate species '{item}' in species list.")
                    species_names.append(item)
                else:
                    raise TypeError(
                        "species entries must be str or Species instances, "
                        f"got {type(item).__name__}."
                    )
        else:
            species_names = list(self.diffusivity.keys())

        self.species = species_names
        self.species_objects = species_objs

        # Auto-populate diffusivity, initial_conditions, and n_electrons from Species
        if species_objs:
            for name, sp_obj in species_objs.items():
                self.diffusivity.setdefault(name, sp_obj.diffusivity)
                self.initial_conditions.setdefault(name, sp_obj.bulk_concentration)

            sp_n_e = {
                name: sp_obj.n_electrons
                for name, sp_obj in species_objs.items()
                if sp_obj.n_electrons is not None
            }
            if n_electrons is None:
                self.n_electrons = sp_n_e if sp_n_e else 1
            elif isinstance(n_electrons, Mapping):
                merged_n_e = dict(sp_n_e)
                merged_n_e.update(n_electrons)
                self.n_electrons = merged_n_e
            else:
                self.n_electrons = n_electrons
        else:
            if n_electrons is None:
                self.n_electrons = 1
            elif isinstance(n_electrons, Mapping):
                self.n_electrons = dict(n_electrons)
            else:
                self.n_electrons = n_electrons

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
    def bounds(self) -> tuple[float, float]:
        """Spatial domain boundaries (x_min, x_max)."""
        return float(self.grid[0]), float(self.grid[-1])

    @property
    def dx_array(self) -> np.ndarray:
        """1D array of consecutive spatial interval spacings."""
        if self.mesh is not None:
            return self.mesh.dx
        return np.diff(self.grid)

    @property
    def dx_min(self) -> float:
        """Minimum spatial interval spacing across the grid."""
        return float(np.min(self.dx_array))

    @property
    def dx_max(self) -> float:
        """Maximum spatial interval spacing across the grid."""
        return float(np.max(self.dx_array))

    @property
    def dx(self) -> float:
        """Grid step size for uniform grids."""
        if not self.is_uniform:
            raise ValueError("Grid is non-uniform; uniform dx is not defined.")
        return float(self.dx_array[0])

    @property
    def is_uniform(self) -> bool:
        """Check whether the spatial grid is uniformly spaced."""
        if self.mesh is not None:
            return self.mesh.is_uniform
        diffs = np.diff(self.grid)
        return bool(np.allclose(diffs, diffs[0], rtol=1e-5, atol=0.0))

    def get_initial_profile(self, sp: str) -> np.ndarray:
        """Evaluate initial concentration profile for a given species."""
        grid_arr = np.asarray(self.grid, dtype=float)
        ic = self.initial_conditions[sp]
        if callable(ic):
            arr = np.asarray(ic(grid_arr), dtype=float)
        elif np.isscalar(ic):
            arr = np.full_like(grid_arr, float(cast(Any, ic)), dtype=float)
        else:
            arr = np.asarray(ic, dtype=float)

        if arr.shape != grid_arr.shape:
            raise ValueError(
                f"Initial condition for '{sp}' has shape {arr.shape}, expected {grid_arr.shape}"
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
    right_fluxes : dict[str, np.ndarray], optional
        Dictionary mapping species name to surface flux at right boundary x=L of shape (N_t,)
        under the electrode-normal convention (pointing into the solution domain).
    currents : dict[str, np.ndarray], optional
        Dictionary mapping species name to Faradaic current at the active electrode(s)
        of shape (N_t,) in Amperes. If ``electrodes='both'``, contains the net cell current
        (:math:`I_{\text{net}} = I_{\text{left}} + I_{\text{right}}`).
    left_currents : dict[str, np.ndarray], optional
        Dictionary mapping species name to Faradaic current at the left boundary of shape (N_t,)
        in Amperes.
    right_currents : dict[str, np.ndarray], optional
        Dictionary mapping species name to Faradaic current at the right boundary of shape (N_t,)
        in Amperes.
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
    ...     right_fluxes={"A": np.zeros(5)},
    ... )
    >>> res.species
    ['A']
    >>> res["A"].shape
    (5, 10)
    >>> res.fluxes["A"].shape
    (5,)
    >>> res.left_fluxes["A"].shape
    (5,)
    >>> res.right_fluxes["A"].shape
    (5,)
    >>> res.left_currents
    {}
    >>> res.right_currents
    {}
    """

    t: np.ndarray
    x: np.ndarray
    concentrations: dict[str, np.ndarray]
    fluxes: dict[str, np.ndarray] = field(default_factory=dict)
    right_fluxes: dict[str, np.ndarray] = field(default_factory=dict)
    currents: dict[str, np.ndarray] = field(default_factory=dict)
    left_currents: dict[str, np.ndarray] = field(default_factory=dict)
    right_currents: dict[str, np.ndarray] = field(default_factory=dict)
    success: bool = True
    message: str = "Success"
    raw_output: Any = None
    mesh: Mesh1D | None = None

    def __getitem__(self, species_name: str) -> np.ndarray:
        """Shorthand to access concentrations for a species: result['A']."""
        return self.concentrations[species_name]

    @property
    def left_fluxes(self) -> dict[str, np.ndarray]:
        """Surface fluxes at the left boundary (alias for ``fluxes``)."""
        return self.fluxes

    @property
    def species(self) -> list[str]:
        """List of species present in the solution."""
        return list(self.concentrations.keys())


class _BaseSolverMeta(ABCMeta):
    """Metaclass for BaseSolver ensuring compatibility across Python versions.

    In Python <= 3.10, ABCMeta.__new__ has `name` as a positional-or-keyword
    parameter rather than positional-only (/), so passing `name="..."` during
    class definition results in `TypeError: ABCMeta.__new__() got multiple values
    for argument 'name'`. We intercept and pop `name` (and optional `solver_name`)
    before delegating to ABCMeta.__new__.
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        solver_name = kwargs.pop("name", None)
        if solver_name is None:
            solver_name = kwargs.pop("solver_name", None)
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if solver_name is not None and hasattr(cls, "_registry"):
            cls._registry[solver_name.lower()] = cls
        return cls


class BaseSolver(ABC, metaclass=_BaseSolverMeta):
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

    def __init_subclass__(
        cls,
        name: str | None = None,
        solver_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        resolved_name = name or solver_name
        if resolved_name:
            cls._registry[resolved_name.lower()] = cls

    def __init__(self, **options: Any) -> None:
        """Initialize solver with optional solver-specific configuration.

        Parameters
        ----------
        **options : Any
            Arbitrary solver configuration options stored in ``self.options``.
            Concrete solver subclasses extract their parameters (such as ``dt``
            or SciPy integration options ``method``, ``rtol``, ``atol``) from
            this dictionary.
        """
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
            Additional run-time options passed to the underlying solver engine,
            temporarily overriding any matching configuration options set in ``__init__``.

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
            raise TypeError(
                f"problem must be an instance of DiffusionProblem, got {type(problem)}"
            )
        if t_span[0] >= t_span[1]:
            raise ValueError(
                f"t_span[1] must be strictly greater than t_span[0], got {t_span}"
            )
        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float)
            if t_eval.ndim != 1:
                raise ValueError("t_eval must be a 1D array.")
            if len(t_eval) < 2:
                raise ValueError("t_eval must have at least 2 points.")
            if t_eval[0] < t_span[0] or t_eval[-1] > t_span[1]:
                raise ValueError(
                    f"t_eval [{t_eval[0]}, {t_eval[-1]}] must lie within t_span {t_span}."
                )
            if np.any(np.diff(t_eval) <= 0):
                raise ValueError(
                    "t_eval points must be strictly monotonically increasing."
                )

    def _post_process(
        self, problem: DiffusionProblem, result: SolverResult
    ) -> SolverResult:
        """Compute surface fluxes, Faradaic currents, and perform post-run integrity checks."""
        # Calculate surface fluxes and Faradaic currents using problem.geometry.
        # Sign convention (electrode-normal into the solution domain):
        # - Left boundary (r = r_0 / x = 0, inward normal n_left = +r / +x):
        #     J_left(t) = -D * (dc/dn)|_{left}
        # - Right boundary (r = r_max / x = L, inward normal n_right = -r / -x):
        #     J_right(t) = -D * (dc/dn)|_{right} = +D * (dc/dr)|_{right}
        # Faradaic current:
        #     I = n * F * A * J
        geom = (
            get_geometry(problem.geometry)
            if isinstance(problem.geometry, str)
            else problem.geometry
        )
        if result.mesh is None and problem.mesh is not None:
            result.mesh = problem.mesh
        for sp in problem.species:
            if sp not in result.concentrations:
                continue
            c = result.concentrations[sp]
            D = problem.diffusivity[sp]
            if sp not in result.fluxes:
                result.fluxes[sp] = np.asarray(
                    geom.surface_flux(c, problem.grid, diffusivity=D, side="left")
                )
            if sp not in result.right_fluxes:
                result.right_fluxes[sp] = np.asarray(
                    geom.surface_flux(c, problem.grid, diffusivity=D, side="right")
                )

            n_electrons_val = getattr(problem, "n_electrons", 1)
            n_e = (
                n_electrons_val.get(sp, 1)
                if isinstance(n_electrons_val, dict)
                else n_electrons_val
            )

            # Compute Faradaic currents based on problem.electrodes ('left', 'right', or 'both')
            if (
                problem.electrodes in ("left", "both")
                and sp not in result.left_currents
            ):
                if n_e == 0:
                    result.left_currents[sp] = np.zeros_like(
                        result.fluxes[sp], dtype=float
                    )
                elif n_e < 0:
                    result.left_currents[sp] = -np.asarray(
                        geom.current(
                            result.fluxes[sp],
                            n_electrons=-n_e,
                            side="left",
                            grid=problem.grid,
                        ),
                        dtype=float,
                    )
                else:
                    result.left_currents[sp] = np.asarray(
                        geom.current(
                            result.fluxes[sp],
                            n_electrons=n_e,
                            side="left",
                            grid=problem.grid,
                        ),
                        dtype=float,
                    )
            if (
                problem.electrodes in ("right", "both")
                and sp not in result.right_currents
            ):
                if n_e == 0:
                    result.right_currents[sp] = np.zeros_like(
                        result.right_fluxes[sp], dtype=float
                    )
                elif n_e < 0:
                    result.right_currents[sp] = -np.asarray(
                        geom.current(
                            result.right_fluxes[sp],
                            n_electrons=-n_e,
                            side="right",
                            grid=problem.grid,
                        ),
                        dtype=float,
                    )
                else:
                    result.right_currents[sp] = np.asarray(
                        geom.current(
                            result.right_fluxes[sp],
                            n_electrons=n_e,
                            side="right",
                            grid=problem.grid,
                        ),
                        dtype=float,
                    )

            if sp not in result.currents:
                if problem.electrodes == "left":
                    result.currents[sp] = result.left_currents[sp]
                elif problem.electrodes == "right":
                    result.currents[sp] = result.right_currents[sp]
                elif problem.electrodes == "both":
                    result.currents[sp] = (
                        result.left_currents[sp] + result.right_currents[sp]
                    )
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
        Configuration options forwarded to the solver's constructor:

        - For ``'explicit'`` / ``'ftcs'``:
          - ``dt`` (float, optional): Fixed time step size satisfying the CFL stability limit.
        - For ``'implicit'`` / ``'btcs'``:
          - ``dt`` (float, optional): Time step size (defaults to 100 intervals).
        - For ``'crank_nicolson'``:
          - ``dt`` (float, optional): Time step size (defaults to 100 intervals).
        - For ``'scipy_ivp'`` / ``'solve_ivp'``:
          - ``method`` (str, default 'Radau'): Integration algorithm ('Radau', 'BDF', 'RK45', etc.).
          - ``rtol`` (float, default 1e-6): Relative error tolerance.
          - ``atol`` (float, default 1e-8): Absolute error tolerance.

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
