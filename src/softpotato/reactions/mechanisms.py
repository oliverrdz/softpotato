r"""Electrochemical and chemical reaction mechanism abstractions.

A :class:`Mechanism` packages the complete chemical and electrochemical model
independently of the spatial grid and numerical solver:

- **Species** (:class:`~softpotato.reactions.SpeciesCollection`): diffusivities,
  bulk concentrations, ionic charges, standard potentials, and electron counts.
- **Homogeneous chemical steps** (:class:`~softpotato.reactions.ReactionNetwork`):
  coupled solution-phase reactions with automated analytical Jacobians.
- **Heterogeneous charge-transfer steps** (:class:`ElectrochemicalStep`):
  interfacial electrode kinetics (Butler–Volmer, Tafel, or Nernstian
  equilibrium) paired with oxidized and reduced species.

Custom mechanisms can be defined in two ways:

1. **Declaratively with** :class:`CustomMechanism`: combine built-in reactions,
   :class:`~softpotato.reactions.CustomReaction`, arbitrary rate callables
   ``rates(c, x, t)`` (with optional analytical Jacobian or automatic vectorized
   finite-difference local Jacobian), and optional electrochemical steps or
   custom boundary condition builders.
2. **By subclassing** :class:`Mechanism`: override :meth:`Mechanism.evaluate`
   (and optionally :meth:`Mechanism.local_jacobian` and/or
   :meth:`Mechanism.boundary_conditions`) for reusable, parameterized models.

Every :class:`Mechanism` can be passed directly to
:class:`~softpotato.solver.DiffusionProblem` via ``mechanism=...`` (or
``reactions=...``) or converted into a ready-to-solve problem with
:meth:`Mechanism.create_problem`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import sparse as _sparse

from .heterogeneous import (
    HeterogeneousKinetics,
    NernstianEquilibrium,
    create_electrochemical_bcs,
)
from .homogeneous import (
    ConcentrationDict,
    HomogeneousReaction,
    ReactionNetwork,
    SpeciesLike,
    _species_name,
)
from .species import Species, SpeciesCollection

if TYPE_CHECKING:
    from softpotato.geometry import Geometry
    from softpotato.grid.base import Mesh1D
    from softpotato.solver.base import BoundaryCondition, DiffusionProblem

__all__ = [
    "CustomMechanism",
    "ElectrochemicalStep",
    "Mechanism",
]

#: Signature for a custom homogeneous rate callable ``rates(c, x, t) -> dict``.
MechanismRatesFunction = Callable[
    [ConcentrationDict, Any, float], dict[str, np.ndarray]
]
#: Signature for a custom local Jacobian callable ``jac(c, x, t, species) -> ndarray``.
MechanismJacobianFunction = Callable[
    [ConcentrationDict, Any, float, Sequence[str]], np.ndarray
]


@dataclass(frozen=True)
class ElectrochemicalStep:
    r"""Specification of a single interfacial electron-transfer step.

    Represents the heterogeneous redox reaction at an electrode boundary:

    .. math::

        \text{Ox} + n e^- \rightleftharpoons \text{Red}

    Parameters
    ----------
    ox : Species or str
        Oxidized species involved in the charge-transfer step.
    red : Species or str
        Reduced species involved in the charge-transfer step.
    kinetics : HeterogeneousKinetics or NernstianEquilibrium
        Interfacial charge-transfer model (:class:`ButlerVolmer`,
        :class:`TafelKinetics`, :class:`NernstianEquilibrium`, or a custom
        :class:`HeterogeneousKinetics` subclass).
    electrode_side : {'left', 'right'}, default 'left'
        Domain boundary where this electrochemical step takes place.
    bulk_bc : {'dirichlet', 'neumann'}, default 'dirichlet'
        Default boundary condition type applied at the opposite (bulk) boundary.
    """

    ox: SpeciesLike
    red: SpeciesLike
    kinetics: HeterogeneousKinetics | NernstianEquilibrium
    electrode_side: Literal["left", "right"] = "left"
    bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet"

    def __post_init__(self) -> None:
        ox_name = _species_name(self.ox, "ox")
        red_name = _species_name(self.red, "red")
        if ox_name == red_name:
            raise ValueError(
                f"Oxidized and reduced species must differ, both got '{ox_name}'."
            )
        if not isinstance(self.kinetics, (HeterogeneousKinetics, NernstianEquilibrium)):
            raise TypeError(
                "kinetics must be an instance of HeterogeneousKinetics or "
                f"NernstianEquilibrium, got {type(self.kinetics).__name__}."
            )
        if self.electrode_side not in ("left", "right"):
            raise ValueError(
                f"electrode_side must be 'left' or 'right', got {self.electrode_side!r}."
            )
        if self.bulk_bc not in ("dirichlet", "neumann"):
            raise ValueError(
                f"bulk_bc must be 'dirichlet' or 'neumann', got {self.bulk_bc!r}."
            )

    @property
    def ox_name(self) -> str:
        """Canonical name of the oxidized species."""
        return _species_name(self.ox, "ox")

    @property
    def red_name(self) -> str:
        """Canonical name of the reduced species."""
        return _species_name(self.red, "red")


def _normalize_species_collection(
    species: SpeciesCollection | Iterable[Species],
) -> SpeciesCollection:
    """Convert an iterable of Species or a SpeciesCollection into a non-empty SpeciesCollection."""
    if isinstance(species, SpeciesCollection):
        collection = SpeciesCollection(species.values())
    else:
        collection = SpeciesCollection(species)
    if len(collection) == 0:
        raise ValueError("A Mechanism requires at least one Species.")
    return collection


def _normalize_reaction_network(
    reactions: ReactionNetwork
    | HomogeneousReaction
    | Iterable[HomogeneousReaction]
    | None,
    species_names: Sequence[str],
) -> ReactionNetwork:
    """Normalize reactions input into a ReactionNetwork ordered by species_names."""
    if reactions is None:
        return ReactionNetwork(species=species_names)
    if isinstance(reactions, ReactionNetwork):
        return ReactionNetwork(reactions.reactions, species=species_names)
    if isinstance(reactions, HomogeneousReaction):
        return ReactionNetwork([reactions], species=species_names)
    return ReactionNetwork(reactions, species=species_names)


class _MechanismWithInterfacialKinetics:
    """Wrap a Mechanism to inject state-dependent interfacial flux at electrode nodes."""

    def __init__(
        self,
        mechanism: Mechanism,
        eval_potential: Callable[[float], float],
        step_specs: Sequence[tuple[str, str, HeterogeneousKinetics, int, float]],
    ) -> None:
        self._mechanism = mechanism
        self._eval_potential = eval_potential
        self._step_specs = tuple(step_specs)

    def __call__(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[str, np.ndarray]:
        base = self._mechanism.evaluate(c, x, t)
        rates: dict[str, np.ndarray] = {
            name: np.array(arr, dtype=float, copy=True) for name, arr in base.items()
        }
        e_val = float(self._eval_potential(t))
        for ox_name, red_name, kinetics, node_idx, inv_half_cell in self._step_specs:
            if ox_name not in rates:
                rates[ox_name] = np.zeros_like(c[ox_name], dtype=float)
            if red_name not in rates:
                rates[red_name] = np.zeros_like(c[red_name], dtype=float)
            kf, kb = kinetics.rate_constants(e_val)
            j_int = float(kf) * float(c[ox_name][node_idx]) - float(kb) * float(
                c[red_name][node_idx]
            )
            rates[ox_name][node_idx] -= inv_half_cell * j_int
            rates[red_name][node_idx] += inv_half_cell * j_int
        return rates

    def local_jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return homogeneous local Jacobian augmented with interfacial flux derivatives."""
        names = list(species) if species is not None else self._mechanism.species_names
        jac = np.array(
            self._mechanism.local_jacobian(c, x, t, species=names),
            dtype=float,
            copy=True,
        )
        e_val = float(self._eval_potential(t))
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        for ox_name, red_name, kinetics, node_idx, inv_half_cell in self._step_specs:
            kf, kb = kinetics.rate_constants(e_val)
            kf_f = float(kf)
            kb_f = float(kb)
            io = name_to_idx.get(ox_name)
            ir = name_to_idx.get(red_name)
            if io is not None:
                jac[io, io, node_idx] -= inv_half_cell * kf_f
                if ir is not None:
                    jac[io, ir, node_idx] += inv_half_cell * kb_f
            if ir is not None:
                if io is not None:
                    jac[ir, io, node_idx] += inv_half_cell * kf_f
                jac[ir, ir, node_idx] -= inv_half_cell * kb_f
        return jac

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mechanism, name)


class Mechanism:
    r"""Modular chemical and electrochemical reaction mechanism.

    Encapsulates species metadata, homogeneous solution-phase reactions, and
    optional heterogeneous electrode charge-transfer steps into a single
    reusable object that integrates directly with
    :class:`~softpotato.solver.DiffusionProblem` and
    :class:`~softpotato.solver.ScipyIVPSolver`.

    Subclasses can override :meth:`evaluate` (and optionally
    :meth:`local_jacobian` and :meth:`boundary_conditions`) to implement custom
    mechanisms. When a subclass overrides :meth:`evaluate` without overriding
    :meth:`local_jacobian`, the node-wise local Jacobian is computed
    automatically via vectorized central finite differences, preserving the
    block-diagonal sparsity structure required by stiff solvers.

    Parameters
    ----------
    species : SpeciesCollection or Iterable[Species]
        Chemical species participating in the mechanism.
    reactions : ReactionNetwork, HomogeneousReaction, or Iterable[HomogeneousReaction], optional
        Homogeneous solution-phase reaction(s).
    electrochemical_steps : Iterable[ElectrochemicalStep], optional
        Heterogeneous electron-transfer steps occurring at electrode boundaries.
    name : str, optional
        Human-readable label for the mechanism.

    Examples
    --------
    EC mechanism with Nernstian electron transfer and first-order decay:

    >>> import numpy as np
    >>> from softpotato.reactions import (
    ...     ElectrochemicalStep,
    ...     FirstOrderReaction,
    ...     Mechanism,
    ...     NernstianEquilibrium,
    ...     Species,
    ... )
    >>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
    >>> red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
    >>> prod = Species("P", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=0)
    >>> ec = Mechanism(
    ...     species=[ox, red, prod],
    ...     reactions=[FirstOrderReaction(red, prod, k=10.0)],
    ...     electrochemical_steps=[
    ...         ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=0.0, n=1))
    ...     ],
    ...     name="EC",
    ... )
    >>> problem = ec.create_problem(np.linspace(0.0, 1e-3, 31), potential=-0.2)
    >>> problem.species
    ['Ox', 'Red', 'P']
    """

    #: Relative step used for central finite-difference local Jacobians.
    fd_rel_step: float = float(np.finfo(float).eps ** (1.0 / 3.0))

    def __init__(
        self,
        species: SpeciesCollection | Iterable[Species],
        reactions: ReactionNetwork
        | HomogeneousReaction
        | Iterable[HomogeneousReaction]
        | None = None,
        electrochemical_steps: Iterable[ElectrochemicalStep] = (),
        *,
        name: str | None = None,
    ) -> None:
        self._species = _normalize_species_collection(species)
        self._reactions = _normalize_reaction_network(reactions, self._species.names)
        self._electrochemical_steps: list[ElectrochemicalStep] = []
        for step in electrochemical_steps:
            self.add_electrochemical_step(step)
        self.name = name or type(self).__name__

    # -- properties & builders ----------------------------------------------
    @property
    def species(self) -> SpeciesCollection:
        """Ordered :class:`SpeciesCollection` of species in the mechanism."""
        return self._species

    @property
    def species_names(self) -> list[str]:
        """Ordered list of species names."""
        return self._species.names

    @property
    def reactions(self) -> ReactionNetwork:
        """Homogeneous :class:`ReactionNetwork` of the mechanism."""
        return self._reactions

    @property
    def electrochemical_steps(self) -> tuple[ElectrochemicalStep, ...]:
        """Tuple of registered :class:`ElectrochemicalStep` objects."""
        return tuple(self._electrochemical_steps)

    def add_reaction(self, reaction: HomogeneousReaction) -> Mechanism:
        """Add a homogeneous reaction to the mechanism (returns ``self``)."""
        self._reactions.add(reaction)
        return self

    def add_electrochemical_step(self, step: ElectrochemicalStep) -> Mechanism:
        """Register an interfacial :class:`ElectrochemicalStep` (returns ``self``)."""
        if not isinstance(step, ElectrochemicalStep):
            raise TypeError(
                f"Expected an ElectrochemicalStep instance, got {type(step).__name__}."
            )
        for sp_name in (step.ox_name, step.red_name):
            if sp_name not in self._species:
                raise ValueError(
                    f"ElectrochemicalStep references species '{sp_name}' not present "
                    f"in mechanism species {self._species.names}."
                )
        self._electrochemical_steps.append(step)
        return self

    # -- homogeneous kinetics & Jacobian interface --------------------------
    def _profile_shape(
        self, c: ConcentrationDict, names: Sequence[str]
    ) -> tuple[int, ...]:
        shapes = [np.shape(c[name]) for name in names if name in c]
        return np.broadcast_shapes(*shapes) if shapes else ()

    def evaluate(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        """Evaluate combined homogeneous species source terms :math:`R_i(c, x, t)`."""
        return self._reactions.evaluate(c, x, t)

    def __call__(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        """Alias for :meth:`evaluate` (``DiffusionProblem.reactions`` signature)."""
        return self.evaluate(c, x, t)

    def _finite_difference_local_jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        names: Sequence[str],
    ) -> np.ndarray:
        """Compute node-wise local Jacobian via vectorized central finite differences."""
        shape = self._profile_shape(c, names)
        n_sp = len(names)
        jac = np.zeros((n_sp, n_sp, *shape), dtype=float)

        for k, name_k in enumerate(names):
            if name_k not in c:
                continue
            ck = np.asarray(c[name_k], dtype=float)
            scale = np.maximum(np.abs(ck), np.max(np.abs(ck), initial=0.0))
            scale = np.where(scale > 0.0, scale, 1.0)
            h = self.fd_rel_step * scale

            c_plus = dict(c)
            c_minus = dict(c)
            c_plus[name_k] = ck + h
            c_minus[name_k] = ck - h

            r_plus = self.evaluate(c_plus, x, t)
            r_minus = self.evaluate(c_minus, x, t)

            for i, name_i in enumerate(names):
                if name_i in r_plus or name_i in r_minus:
                    rp = np.asarray(r_plus.get(name_i, 0.0), dtype=float)
                    rm = np.asarray(r_minus.get(name_i, 0.0), dtype=float)
                    jac[i, k] = (rp - rm) / (2.0 * h)

        return jac

    def local_jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
    ) -> np.ndarray:
        r"""Node-wise kinetic Jacobian :math:`J_{ik}(x) = \partial R_i / \partial c_k`.

        Returns an array of shape ``(n_species, n_species, *profile_shape)``.
        """
        names = list(species) if species is not None else self._species.names
        if type(self).evaluate is Mechanism.evaluate:
            return self._reactions.local_jacobian(c, x, t, species=names)
        return self._finite_difference_local_jacobian(c, x, t, names)

    def jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
        sparse: bool = True,
    ) -> Any:
        """Full ``(S*N, S*N)`` sparse or dense kinetic Jacobian in species-major order."""
        local = self.local_jacobian(c, x, t, species=species)
        n_species = local.shape[0]
        if local.ndim == 2:
            local = local[:, :, np.newaxis]
        elif local.ndim != 3:
            raise ValueError(
                "jacobian() requires 1D concentration profiles; "
                f"got profile shape {local.shape[2:]}."
            )
        n_points = local.shape[2]
        size = n_species * n_points
        nodes = np.arange(n_points)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for i in range(n_species):
            for k in range(n_species):
                block = local[i, k]
                if np.any(block != 0.0):
                    rows.append(i * n_points + nodes)
                    cols.append(k * n_points + nodes)
                    data.append(block)
        if data:
            coo = _sparse.coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(size, size),
            )
        else:
            coo = _sparse.coo_matrix((size, size), dtype=float)
        mat = coo.tocsr()
        return mat if sparse else mat.toarray()

    # -- boundary conditions & problem construction -------------------------
    def boundary_conditions(
        self,
        potential: float | Callable[[float], float] | None = None,
        *,
        electrode_side: Literal["left", "right"] | None = None,
        bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet",
        overrides: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        **bc_kwargs: Any,
    ) -> dict[str, tuple[BoundaryCondition, BoundaryCondition]]:
        """Build boundary conditions for all species in the mechanism.

        Electroactive species from :attr:`electrochemical_steps` receive
        matched interfacial boundary conditions via
        :func:`~softpotato.reactions.create_electrochemical_bcs`. Non-electroactive
        species receive zero-flux (:func:`~softpotato.solver.NeumannBC`) at the
        electrode boundary and bulk concentration (or zero-flux) at the bulk
        boundary.
        """
        from softpotato.solver.base import DirichletBC, NeumannBC

        if bulk_bc not in ("dirichlet", "neumann"):
            raise ValueError(
                f"bulk_bc must be 'dirichlet' or 'neumann', got {bulk_bc!r}."
            )
        if electrode_side is not None and electrode_side not in ("left", "right"):
            raise ValueError(
                f"electrode_side must be 'left' or 'right', got {electrode_side!r}."
            )

        bcs: dict[str, tuple[BoundaryCondition, BoundaryCondition]] = {}

        if self._electrochemical_steps:
            if potential is None:
                raise ValueError(
                    "potential must be provided to generate boundary conditions "
                    "for a mechanism with electrochemical steps."
                )
            for step in self._electrochemical_steps:
                side = (
                    electrode_side
                    if electrode_side is not None
                    else step.electrode_side
                )
                step_bcs = create_electrochemical_bcs(
                    step.kinetics,
                    potential=potential,
                    ox=self._species[step.ox_name],
                    red=self._species[step.red_name],
                    electrode_side=side,
                    bulk_bc=bulk_bc,
                    **bc_kwargs,
                )
                bcs.update(step_bcs)

        default_side: Literal["left", "right"] = (
            electrode_side
            if electrode_side is not None
            else (
                self._electrochemical_steps[0].electrode_side
                if self._electrochemical_steps
                else "left"
            )
        )

        for sp_name, sp_obj in self._species.items():
            if sp_name in bcs:
                continue
            elec_bc = NeumannBC(0.0)
            far_bc = (
                DirichletBC(sp_obj.bulk_concentration)
                if bulk_bc == "dirichlet"
                else NeumannBC(0.0)
            )
            bcs[sp_name] = (
                (elec_bc, far_bc) if default_side == "left" else (far_bc, elec_bc)
            )

        if overrides:
            bcs.update(overrides)

        return bcs

    def create_problem(
        self,
        grid: np.ndarray | Mesh1D,
        *,
        potential: float | Callable[[float], float] | None = None,
        boundary_conditions: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        geometry: Geometry | str = "planar",
        diffusivity: Mapping[str, float] | None = None,
        initial_conditions: Mapping[
            str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]
        ]
        | None = None,
        n_electrons: int | dict[str, int] | None = None,
        electrodes: Literal["left", "right", "both"] = "left",
        electrode_side: Literal["left", "right", "both"] | None = None,
        bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet",
        bc_overrides: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        **bc_kwargs: Any,
    ) -> DiffusionProblem:
        """Create a :class:`~softpotato.solver.DiffusionProblem` from this mechanism.

        Parameters
        ----------
        grid : np.ndarray or Mesh1D
            1D spatial grid or mesh.
        potential : float or Callable[[float], float], optional
            Electrode potential :math:`E` or waveform :math:`E(t)` used to build
            boundary conditions when ``boundary_conditions`` is not explicitly given.
        boundary_conditions : Mapping[str, tuple[BoundaryCondition, BoundaryCondition]], optional
            Explicit boundary conditions per species (overrides automatic generation).
        geometry : Geometry or str, default 'planar'
            Coordinate geometry passed to :class:`~softpotato.solver.DiffusionProblem`.
        diffusivity : Mapping[str, float], optional
            Optional diffusivity overrides.
        initial_conditions : Mapping, optional
            Optional initial condition overrides.
        n_electrons : int or dict[str, int], optional
            Optional electron transfer count overrides.
        electrodes : {'left', 'right', 'both'}, default 'left'
            Active electrode boundary for current evaluation.
        electrode_side : {'left', 'right', 'both'}, optional
            Alias for ``electrodes``.
        bulk_bc : {'dirichlet', 'neumann'}, default 'dirichlet'
            Bulk boundary condition type when auto-generating BCs.
        bc_overrides : Mapping, optional
            Per-species boundary condition overrides applied on top of generated BCs.
        **bc_kwargs : Any
            Additional keyword arguments forwarded to :meth:`boundary_conditions`.

        Returns
        -------
        DiffusionProblem
            Configured reaction-diffusion problem ready for solving.
        """
        from softpotato.solver.base import DiffusionProblem, NeumannBC

        if boundary_conditions is not None:
            resolved_bcs = dict(boundary_conditions)
            if bc_overrides:
                resolved_bcs.update(bc_overrides)
            reactions_obj: Any = self
        else:
            bc_side: Literal["left", "right"] | None = None
            target_side = electrode_side if electrode_side is not None else electrodes
            if target_side in ("left", "right"):
                bc_side = target_side
            resolved_bcs = self.boundary_conditions(
                potential=potential,
                electrode_side=bc_side,
                bulk_bc=bulk_bc,
                overrides=bc_overrides,
                **bc_kwargs,
            )
            reactions_obj = self

            override_names = set(bc_overrides) if bc_overrides else set()
            het_steps = [
                step
                for step in self._electrochemical_steps
                if isinstance(step.kinetics, HeterogeneousKinetics)
                and step.ox_name not in override_names
                and step.red_name not in override_names
            ]
            if het_steps and potential is not None:
                x_nodes = np.asarray(
                    grid.nodes if hasattr(grid, "nodes") else grid, dtype=float
                )
                if x_nodes.ndim != 1 or x_nodes.size < 2:
                    raise ValueError(
                        "grid must be 1D with at least 2 points for heterogeneous kinetics."
                    )
                if callable(potential):
                    eval_potential: Callable[[float], float] = potential
                else:
                    const_e = float(potential)

                    def eval_potential(_t: float) -> float:
                        return const_e

                step_specs: list[tuple[str, str, HeterogeneousKinetics, int, float]] = []
                for step in het_steps:
                    side = bc_side if bc_side is not None else step.electrode_side
                    if side == "left":
                        node_idx = 0
                        inv_half_cell = 2.0 / float(x_nodes[1] - x_nodes[0])
                        resolved_bcs[step.ox_name] = (
                            NeumannBC(0.0),
                            resolved_bcs[step.ox_name][1],
                        )
                        resolved_bcs[step.red_name] = (
                            NeumannBC(0.0),
                            resolved_bcs[step.red_name][1],
                        )
                    else:
                        node_idx = -1
                        inv_half_cell = 2.0 / float(x_nodes[-1] - x_nodes[-2])
                        resolved_bcs[step.ox_name] = (
                            resolved_bcs[step.ox_name][0],
                            NeumannBC(0.0),
                        )
                        resolved_bcs[step.red_name] = (
                            resolved_bcs[step.red_name][0],
                            NeumannBC(0.0),
                        )
                    step_specs.append(
                        (step.ox_name, step.red_name, step.kinetics, node_idx, inv_half_cell)
                    )

                reactions_obj = _MechanismWithInterfacialKinetics(
                    self, eval_potential, step_specs
                )

        return DiffusionProblem(
            grid=grid,
            diffusivity=dict(diffusivity) if diffusivity is not None else {},
            boundary_conditions=resolved_bcs,
            initial_conditions=(
                dict(initial_conditions) if initial_conditions is not None else {}
            ),
            reactions=reactions_obj,
            species=self._species,
            geometry=geometry,
            n_electrons=n_electrons,
            electrodes=electrodes,
            electrode_side=electrode_side,
            mechanism=self,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"species={self._species.names!r}, "
            f"n_reactions={len(self._reactions)}, "
            f"n_electrochemical_steps={len(self._electrochemical_steps)})"
        )


class CustomMechanism(Mechanism):
    r"""User-defined chemical or electrochemical reaction mechanism.

    Extends :class:`Mechanism` with optional functional hooks so users can
    define arbitrary mechanisms without subclassing:

    - ``reactions``: built-in or :class:`~softpotato.reactions.CustomReaction`
      objects (analytical Jacobian assembled automatically).
    - ``rates``: arbitrary user callable ``rates(c, x, t) -> dict[str, ndarray]``
      added on top of (or used instead of) ``reactions``.
    - ``local_jacobian_func``: optional analytical node-wise Jacobian
      ``jac(c, x, t, species) -> ndarray`` of shape ``(S, S, N)`` for ``rates``.
      If omitted, vectorized central finite differences are computed
      automatically (only ``2 * S`` evaluations of ``rates`` per step).
    - ``electrochemical_steps``: list of :class:`ElectrochemicalStep` objects.
    - ``boundary_conditions_func``: optional custom callable returning the
      boundary condition dictionary for :meth:`boundary_conditions`.

    Parameters
    ----------
    species : SpeciesCollection or Iterable[Species]
        Chemical species in the mechanism.
    reactions : ReactionNetwork, HomogeneousReaction, or Iterable[HomogeneousReaction], optional
        Structured homogeneous reactions.
    rates : Callable[[ConcentrationDict, Any, float], dict[str, np.ndarray]], optional
        Custom homogeneous rate law returning ``{species_name: rate_array}``.
    local_jacobian_func : Callable[[ConcentrationDict, Any, float, Sequence[str]], np.ndarray], optional
        Analytical node-wise Jacobian for ``rates`` of shape ``(S, S, *profile_shape)``.
    electrochemical_steps : Iterable[ElectrochemicalStep], optional
        Heterogeneous electron-transfer steps at electrode boundaries.
    boundary_conditions_func : Callable, optional
        Custom boundary condition builder invoked by :meth:`boundary_conditions`.
    name : str, optional
        Human-readable label for this mechanism.

    Examples
    --------
    Custom nonlinear mechanism with an arbitrary rate law and automatic local
    Jacobian:

    >>> import numpy as np
    >>> from softpotato.reactions import CustomMechanism, Species
    >>> a = Species("A", diffusivity=1e-9, bulk_concentration=1.0)
    >>> b = Species("B", diffusivity=1e-9, bulk_concentration=0.0)
    >>> mech = CustomMechanism(
    ...     species=[a, b],
    ...     rates=lambda c, x, t: {
    ...         "A": -5.0 * c["A"] / (1.0 + c["A"]),
    ...         "B": +5.0 * c["A"] / (1.0 + c["A"]),
    ...     },
    ...     name="MichaelisMenten",
    ... )
    >>> c = {"A": np.array([1.0, 1.0]), "B": np.array([0.0, 0.0])}
    >>> mech(c, None, 0.0)["B"]
    array([2.5, 2.5])
    """

    def __init__(
        self,
        species: SpeciesCollection | Iterable[Species],
        reactions: ReactionNetwork
        | HomogeneousReaction
        | Iterable[HomogeneousReaction]
        | None = None,
        *,
        rates: MechanismRatesFunction | None = None,
        local_jacobian_func: MechanismJacobianFunction | None = None,
        electrochemical_steps: Iterable[ElectrochemicalStep] = (),
        boundary_conditions_func: Callable[
            ..., dict[str, tuple[BoundaryCondition, BoundaryCondition]]
        ]
        | None = None,
        name: str | None = None,
    ) -> None:
        if rates is not None and not callable(rates):
            raise TypeError("rates must be a callable rates(c, x, t).")
        if local_jacobian_func is not None and not callable(local_jacobian_func):
            raise TypeError(
                "local_jacobian_func must be a callable jac(c, x, t, species)."
            )
        if local_jacobian_func is not None and rates is None:
            raise ValueError("local_jacobian_func cannot be provided without rates.")
        if boundary_conditions_func is not None and not callable(
            boundary_conditions_func
        ):
            raise TypeError("boundary_conditions_func must be callable.")

        super().__init__(
            species=species,
            reactions=reactions,
            electrochemical_steps=electrochemical_steps,
            name=name or "CustomMechanism",
        )
        self._custom_rates = rates
        self._custom_local_jacobian = local_jacobian_func
        self._custom_bc_func = boundary_conditions_func

    def _eval_custom_rates(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[str, np.ndarray]:
        assert self._custom_rates is not None
        raw = self._custom_rates(c, x, t)
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Custom rates callable must return a mapping, got {type(raw).__name__}."
            )
        shape = self._profile_shape(c, self._species.names)
        result: dict[str, np.ndarray] = {}
        for name, val in raw.items():
            if name not in self._species:
                raise KeyError(
                    f"Custom rates returned unknown species '{name}'. "
                    f"Known species: {self._species.names}."
                )
            arr = np.asarray(val, dtype=float)
            bshape = np.broadcast_shapes(arr.shape, shape)
            result[name] = np.array(np.broadcast_to(arr, bshape), dtype=float)
        return result

    def evaluate(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        """Evaluate combined rates from structured reactions and custom rate callable."""
        total = self._reactions.evaluate(c, x, t)
        if self._custom_rates is not None:
            custom = self._eval_custom_rates(c, x, t)
            for name, arr in custom.items():
                total[name] = total[name] + arr
        return total

    def local_jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Compute node-wise kinetic Jacobian combining structured and custom terms."""
        names = list(species) if species is not None else self._species.names
        jac = self._reactions.local_jacobian(c, x, t, species=names)
        if self._custom_rates is None:
            return jac

        if self._custom_local_jacobian is not None:
            custom_jac = np.asarray(
                self._custom_local_jacobian(c, x, t, names), dtype=float
            )
            if custom_jac.shape != jac.shape:
                raise ValueError(
                    f"local_jacobian_func returned shape {custom_jac.shape}, "
                    f"expected {jac.shape}."
                )
            return jac + custom_jac

        # Vectorized central finite differences on the custom rate contribution only
        shape = self._profile_shape(c, names)
        custom_fd = np.zeros((len(names), len(names), *shape), dtype=float)
        for k, name_k in enumerate(names):
            if name_k not in c:
                continue
            ck = np.asarray(c[name_k], dtype=float)
            scale = np.maximum(np.abs(ck), np.max(np.abs(ck), initial=0.0))
            scale = np.where(scale > 0.0, scale, 1.0)
            h = self.fd_rel_step * scale

            c_plus = dict(c)
            c_minus = dict(c)
            c_plus[name_k] = ck + h
            c_minus[name_k] = ck - h

            r_plus = self._eval_custom_rates(c_plus, x, t)
            r_minus = self._eval_custom_rates(c_minus, x, t)

            for i, name_i in enumerate(names):
                if name_i in r_plus or name_i in r_minus:
                    rp = r_plus.get(name_i, 0.0)
                    rm = r_minus.get(name_i, 0.0)
                    custom_fd[i, k] = (rp - rm) / (2.0 * h)

        return jac + custom_fd

    def boundary_conditions(
        self,
        potential: float | Callable[[float], float] | None = None,
        *,
        electrode_side: Literal["left", "right"] | None = None,
        bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet",
        overrides: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        **bc_kwargs: Any,
    ) -> dict[str, tuple[BoundaryCondition, BoundaryCondition]]:
        """Build boundary conditions using ``boundary_conditions_func`` if provided."""
        if self._custom_bc_func is not None:
            bcs = dict(
                self._custom_bc_func(
                    potential,
                    electrode_side=electrode_side,
                    bulk_bc=bulk_bc,
                    **bc_kwargs,
                )
            )
            if overrides:
                bcs.update(overrides)
            return bcs
        return super().boundary_conditions(
            potential=potential,
            electrode_side=electrode_side,
            bulk_bc=bulk_bc,
            overrides=overrides,
            **bc_kwargs,
        )
