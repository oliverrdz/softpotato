r"""Homogeneous chemical reaction kinetics and automated kinetic Jacobians.

Coupled solution-phase chemical reactions (EC, EC', catalytic regeneration,
dimerization, disproportionation, ...) add source/sink terms to the transport
equation of every species :math:`i`:

.. math::

    \frac{\partial c_i}{\partial t} = D_i \nabla^2 c_i + R_i(\mathbf{c}),
    \qquad R_i(\mathbf{c}) = \sum_j \nu_{i,j}\, r_j(\mathbf{c})

where :math:`\nu_{i,j}` is the net stoichiometric coefficient of species
:math:`i` in reaction :math:`j` (negative for consumed species, positive for
produced species) and :math:`r_j` is the reaction rate.

Every reaction also provides its analytical kinetic Jacobian

.. math::

    J_{ik} = \frac{\partial R_i}{\partial c_k}
           = \sum_j \nu_{i,j} \frac{\partial r_j}{\partial c_k},

which :class:`ReactionNetwork` assembles into the (sparse) block matrix
required by stiff implicit integrators (``Radau``, ``BDF``).

Building your own reaction
--------------------------
Two extension points are provided:

* :class:`CustomReaction` wraps an arbitrary user-supplied rate law
  ``rate(c, x, t)`` together with its stoichiometry. Analytical rate
  derivatives may be supplied optionally; any missing derivative is computed
  automatically by central finite differences.
* :class:`HomogeneousReaction` can be subclassed directly by implementing the
  :attr:`~HomogeneousReaction.stoichiometry` property and the
  :meth:`~HomogeneousReaction.rate` method (and optionally
  :meth:`~HomogeneousReaction.rate_derivatives`).

Examples
--------
EC mechanism with a follow-up first-order decay of the product ``Red``:

>>> import numpy as np
>>> from softpotato.reactions import FirstOrderReaction, ReactionNetwork
>>> network = ReactionNetwork([FirstOrderReaction("Red", "P", k=2.0)])
>>> c = {"Ox": np.array([1.0, 1.0]), "Red": np.array([0.5, 0.25]), "P": np.zeros(2)}
>>> rates = network(c, np.array([0.0, 1.0]), 0.0)
>>> rates["Red"]
array([-1. , -0.5])
>>> rates["P"]
array([1. , 0.5])
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse as _sparse

from .species import Species

__all__ = [
    "CatalyticReaction",
    "CustomReaction",
    "DimerizationReaction",
    "DisproportionationReaction",
    "FirstOrderReaction",
    "HomogeneousReaction",
    "MassActionReaction",
    "ReactionNetwork",
    "ReversibleReaction",
    "SecondOrderReaction",
]

#: Concentration mapping passed to rate laws: ``{species_name: array}``.
ConcentrationDict = Mapping[str, Any]
#: A species may be referred to by its name or by a :class:`Species` object.
SpeciesLike = str | Species
#: Signature of a user-defined rate law ``r(c, x, t)``.
RateFunction = Callable[[ConcentrationDict, Any, float], Any]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _species_name(species: SpeciesLike, field_name: str = "species") -> str:
    """Return the species name for a ``str`` or :class:`Species` argument."""
    if isinstance(species, Species):
        return species.name
    if not isinstance(species, str) or not species.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string or a Species instance, "
            f"got {species!r}."
        )
    return species


def _validate_rate_constant(value: object, field_name: str) -> float:
    """Validate a finite, non-negative real rate constant and return a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(
            f"{field_name} must be a real number, got {type(value).__name__}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value}.")
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got {value}.")
    return result


def _validate_order_mapping(
    mapping: Mapping[SpeciesLike, int], field_name: str
) -> dict[str, int]:
    """Validate a mapping of species to positive integer stoichiometric orders."""
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping of species to stoichiometric "
            f"coefficients, got {type(mapping).__name__}."
        )
    result: dict[str, int] = {}
    for species, coeff in mapping.items():
        name = _species_name(species, f"{field_name} species")
        if isinstance(coeff, bool) or not isinstance(coeff, (int, np.integer)):
            raise TypeError(
                f"Stoichiometric coefficient of '{name}' in {field_name} must be "
                f"a positive int, got {type(coeff).__name__}."
            )
        if coeff <= 0:
            raise ValueError(
                f"Stoichiometric coefficient of '{name}' in {field_name} must be "
                f"positive, got {coeff}."
            )
        if name in result:
            raise ValueError(f"Species '{name}' appears twice in {field_name}.")
        result[name] = int(coeff)
    return result


def _get_concentration(c: ConcentrationDict, name: str) -> np.ndarray:
    """Fetch a species concentration as a float array with a clear error."""
    try:
        value = c[name]
    except KeyError:
        raise KeyError(
            f"Concentration of species '{name}' is required by the reaction "
            f"but was not provided. Available species: {sorted(c)}."
        ) from None
    return np.asarray(value, dtype=float)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class HomogeneousReaction(ABC):
    r"""Abstract base class for a single homogeneous (solution-phase) reaction.

    A reaction is fully described by its net stoichiometry :math:`\nu_i` and
    its rate law :math:`r(\mathbf{c}, x, t)`. The species source terms and the
    kinetic Jacobian follow automatically:

    .. math::

        R_i = \nu_i\, r, \qquad
        \frac{\partial R_i}{\partial c_k} = \nu_i \frac{\partial r}{\partial c_k}

    Subclass this to implement a new reaction type. Required members:

    - :attr:`stoichiometry`: mapping ``{species: net coefficient}``.
    - :meth:`rate`: the reaction rate :math:`r(\mathbf{c}, x, t)`.

    Optional members:

    - :attr:`dependencies`: species whose concentrations the rate depends on
      (defaults to the species in :attr:`stoichiometry`).
    - :meth:`rate_derivatives`: analytical :math:`\partial r/\partial c_k`.
      The default implementation uses central finite differences, so it only
      needs overriding for speed or accuracy.

    Rate laws must be *local*: the rate at grid node :math:`k` may only depend
    on the concentrations at that same node (true for all homogeneous
    kinetics). This makes the kinetic Jacobian block-diagonal in space.

    Examples
    --------
    A Michaelis–Menten enzymatic conversion :math:`S \rightarrow P` written as
    a subclass:

    >>> import numpy as np
    >>> from softpotato.reactions import HomogeneousReaction
    >>> class MichaelisMenten(HomogeneousReaction):
    ...     def __init__(self, vmax, km):
    ...         self.vmax, self.km = vmax, km
    ...     @property
    ...     def stoichiometry(self):
    ...         return {"S": -1, "P": 1}
    ...     def rate(self, c, x, t):
    ...         s = np.asarray(c["S"], dtype=float)
    ...         return self.vmax * s / (self.km + s)
    >>> rxn = MichaelisMenten(vmax=2.0, km=1.0)
    >>> rxn.rates({"S": np.array([1.0]), "P": np.array([0.0])}, None, 0.0)
    {'S': array([-1.]), 'P': array([1.])}
    >>> jac = rxn.jacobian({"S": np.array([1.0]), "P": np.array([0.0])}, None, 0.0)
    >>> bool(np.isclose(jac[("S", "S")][0], -0.5))
    True
    """

    #: Relative step used for central finite-difference rate derivatives.
    fd_rel_step: float = float(np.finfo(float).eps ** (1.0 / 3.0))

    # -- required interface -------------------------------------------------
    @property
    @abstractmethod
    def stoichiometry(self) -> Mapping[str, float]:
        """Net stoichiometric coefficients ``{species: nu_i}`` (products > 0)."""

    @abstractmethod
    def rate(self, c: ConcentrationDict, x: Any, t: float) -> Any:
        r"""Reaction rate :math:`r(\mathbf{c}, x, t)` (scalar or array)."""

    # -- optional interface -------------------------------------------------
    @property
    def dependencies(self) -> tuple[str, ...]:
        """Species whose concentrations the rate law depends on."""
        return tuple(self.stoichiometry)

    def rate_derivatives(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[str, np.ndarray]:
        r"""Partial derivatives :math:`\partial r/\partial c_k` for each dependency.

        The default implementation uses vectorized central finite differences:
        because rate laws are local, perturbing an entire concentration profile
        at once yields the node-wise derivative at every grid point.

        Parameters
        ----------
        c : Mapping[str, array_like]
            Concentrations keyed by species name.
        x : array_like
            Spatial grid (passed through to :meth:`rate`).
        t : float
            Time.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping ``{species: dr/dc_species}``.
        """
        return self._finite_difference_derivatives(c, x, t, self.dependencies)

    # -- derived quantities -------------------------------------------------
    @property
    def species(self) -> tuple[str, ...]:
        """All species involved (stoichiometry first, then extra dependencies)."""
        names = list(self.stoichiometry)
        for name in self.dependencies:
            if name not in names:
                names.append(name)
        return tuple(names)

    def rates(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        r"""Species source terms :math:`R_i = \nu_i r` for this reaction.

        Parameters
        ----------
        c : Mapping[str, array_like]
            Concentrations keyed by species name.
        x : array_like
            Spatial grid.
        t : float
            Time.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping ``{species: R_i}`` for every species with non-zero
            stoichiometry.
        """
        r = self._broadcast(self.rate(c, x, t), c)
        return {name: nu * r for name, nu in self.stoichiometry.items() if nu != 0}

    def jacobian(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[tuple[str, str], np.ndarray]:
        r"""Node-wise kinetic Jacobian entries :math:`\partial R_i/\partial c_k`.

        Returns
        -------
        dict[tuple[str, str], np.ndarray]
            Mapping ``{(i, k): dR_i/dc_k}``; only structurally non-zero entries
            are included.
        """
        derivs = self.rate_derivatives(c, x, t)
        jac: dict[tuple[str, str], np.ndarray] = {}
        for name_i, nu in self.stoichiometry.items():
            if nu == 0:
                continue
            for name_k, dr in derivs.items():
                jac[(name_i, name_k)] = nu * self._broadcast(dr, c)
        return jac

    def __call__(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        """Alias for :meth:`rates`, usable directly as ``DiffusionProblem.reactions``."""
        return self.rates(c, x, t)

    # -- helpers ------------------------------------------------------------
    def _shape(self, c: ConcentrationDict) -> tuple[int, ...]:
        shapes = [np.shape(c[name]) for name in self.species if name in c]
        return np.broadcast_shapes(*shapes) if shapes else ()

    def _broadcast(self, value: Any, c: ConcentrationDict) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        shape = np.broadcast_shapes(arr.shape, self._shape(c))
        return np.array(np.broadcast_to(arr, shape), dtype=float)

    def _finite_difference_derivatives(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        names: Iterable[str],
    ) -> dict[str, np.ndarray]:
        """Central finite-difference derivatives of :meth:`rate` w.r.t. ``names``."""
        derivs: dict[str, np.ndarray] = {}
        for name in names:
            ck = _get_concentration(c, name)
            scale = np.maximum(np.abs(ck), np.max(np.abs(ck), initial=0.0))
            scale = np.where(scale > 0.0, scale, 1.0)
            h = self.fd_rel_step * scale
            c_plus = dict(c)
            c_minus = dict(c)
            c_plus[name] = ck + h
            c_minus[name] = ck - h
            r_plus = np.asarray(self.rate(c_plus, x, t), dtype=float)
            r_minus = np.asarray(self.rate(c_minus, x, t), dtype=float)
            derivs[name] = (r_plus - r_minus) / (2.0 * h)
        return derivs


# ---------------------------------------------------------------------------
# User-defined reactions
# ---------------------------------------------------------------------------
class CustomReaction(HomogeneousReaction):
    r"""User-defined homogeneous reaction with an arbitrary rate law.

    Build your own reaction by supplying its net stoichiometry and a rate law
    ``rate(c, x, t)``. Source terms are :math:`R_i = \nu_i\, r(\mathbf{c}, x, t)`.
    Analytical derivatives :math:`\partial r / \partial c_k` can be supplied
    through ``rate_derivatives``; any dependency without one is differentiated
    automatically with central finite differences, so the kinetic Jacobian is
    always available.

    Parameters
    ----------
    stoichiometry : Mapping[str | Species, float]
        Net stoichiometric coefficients ``{species: nu_i}``: negative for
        consumed species and positive for produced species. Non-integer
        coefficients are allowed for non-elementary rate laws.
    rate : Callable[[dict[str, np.ndarray], np.ndarray, float], array_like]
        Rate law :math:`r(\mathbf{c}, x, t)`. Receives the concentration
        dictionary, the spatial grid and the time, and returns a scalar or an
        array broadcastable to the concentration profiles. The rate law must be
        local (node-wise).
    rate_derivatives : Mapping[str | Species, Callable], optional
        Analytical partial derivatives ``{species: dr_dc(c, x, t)}``.
    dependencies : Sequence[str | Species], optional
        Species the rate law depends on. Defaults to the species in
        ``stoichiometry`` plus those in ``rate_derivatives``. List here any
        species that affect the rate without being consumed or produced
        (e.g., an enzyme or a homogeneous catalyst).
    name : str, optional
        Human-readable label used in ``repr``.

    Examples
    --------
    Michaelis–Menten kinetics :math:`S \rightarrow P` with
    :math:`r = V_{max} c_S / (K_M + c_S)`:

    >>> import numpy as np
    >>> from softpotato.reactions import CustomReaction
    >>> mm = CustomReaction(
    ...     stoichiometry={"S": -1, "P": +1},
    ...     rate=lambda c, x, t: 2.0 * c["S"] / (1.0 + c["S"]),
    ...     name="Michaelis-Menten",
    ... )
    >>> c = {"S": np.array([1.0, 3.0]), "P": np.zeros(2)}
    >>> mm.rates(c, None, 0.0)["P"]
    array([1. , 1.5])
    >>> jac = mm.jacobian(c, None, 0.0)  # finite-difference derivatives
    >>> bool(np.allclose(jac[("P", "S")], 2.0 / (1.0 + c["S"]) ** 2))
    True

    Supplying the analytical derivative instead:

    >>> mm_exact = CustomReaction(
    ...     stoichiometry={"S": -1, "P": +1},
    ...     rate=lambda c, x, t: 2.0 * c["S"] / (1.0 + c["S"]),
    ...     rate_derivatives={"S": lambda c, x, t: 2.0 / (1.0 + c["S"]) ** 2},
    ... )
    >>> mm_exact.jacobian(c, None, 0.0)[("S", "S")]
    array([-0.5  , -0.125])
    """

    def __init__(
        self,
        stoichiometry: Mapping[SpeciesLike, float],
        rate: RateFunction,
        rate_derivatives: Mapping[SpeciesLike, RateFunction] | None = None,
        dependencies: Sequence[SpeciesLike] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(stoichiometry, Mapping) or not stoichiometry:
            raise ValueError("stoichiometry must be a non-empty mapping.")
        stoich: dict[str, float] = {}
        for species, nu in stoichiometry.items():
            sp_name = _species_name(species, "stoichiometry species")
            if isinstance(nu, bool) or not isinstance(nu, (int, float, np.number)):
                raise TypeError(
                    f"Stoichiometric coefficient of '{sp_name}' must be a real "
                    f"number, got {type(nu).__name__}."
                )
            if not math.isfinite(float(nu)):
                raise ValueError(
                    f"Stoichiometric coefficient of '{sp_name}' must be finite."
                )
            if sp_name in stoich:
                raise ValueError(f"Species '{sp_name}' appears twice in stoichiometry.")
            stoich[sp_name] = nu
        if all(nu == 0 for nu in stoich.values()):
            raise ValueError(
                "At least one stoichiometric coefficient must be non-zero."
            )
        if not callable(rate):
            raise TypeError("rate must be a callable rate(c, x, t).")

        derivs: dict[str, RateFunction] = {}
        for species, func in (rate_derivatives or {}).items():
            sp_name = _species_name(species, "rate_derivatives species")
            if not callable(func):
                raise TypeError(
                    f"rate derivative for '{sp_name}' must be a callable "
                    f"dr_dc(c, x, t)."
                )
            derivs[sp_name] = func

        if dependencies is None:
            deps = list(stoich)
            deps += [s for s in derivs if s not in deps]
        else:
            deps = []
            for species in dependencies:
                sp_name = _species_name(species, "dependencies species")
                if sp_name not in deps:
                    deps.append(sp_name)
            missing = [s for s in derivs if s not in deps]
            if missing:
                raise ValueError(
                    f"rate_derivatives given for species {missing} that are not "
                    f"listed in dependencies."
                )

        self._stoichiometry = stoich
        self._rate = rate
        self._rate_derivatives = derivs
        self._dependencies = tuple(deps)
        self.name = name

    @property
    def stoichiometry(self) -> dict[str, float]:
        """Net stoichiometric coefficients ``{species: nu_i}``."""
        return dict(self._stoichiometry)

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Species the rate law depends on."""
        return self._dependencies

    def rate(self, c: ConcentrationDict, x: Any, t: float) -> Any:
        """Evaluate the user-supplied rate law."""
        return self._rate(c, x, t)

    def rate_derivatives(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[str, np.ndarray]:
        """Analytical derivatives where supplied, finite differences otherwise."""
        derivs = {
            name: np.asarray(func(c, x, t), dtype=float)
            for name, func in self._rate_derivatives.items()
        }
        missing = [name for name in self._dependencies if name not in derivs]
        if missing:
            derivs.update(self._finite_difference_derivatives(c, x, t, missing))
        return {name: derivs[name] for name in self._dependencies}

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return f"CustomReaction({label}stoichiometry={self._stoichiometry!r})"


# ---------------------------------------------------------------------------
# Mass-action kinetics
# ---------------------------------------------------------------------------
class MassActionReaction(HomogeneousReaction):
    r"""General elementary reaction obeying the law of mass action.

    .. math::

        \sum_i a_i\, X_i \;\underset{k_b}{\overset{k_f}{\rightleftharpoons}}\;
        \sum_j b_j\, Y_j,
        \qquad r = k_f \prod_i c_{X_i}^{a_i} - k_b \prod_j c_{Y_j}^{b_j}

    The net stoichiometry is :math:`\nu = b - a`, so a species appearing on
    both sides (e.g. a catalyst) has its coefficients cancelled. Rate
    derivatives are computed analytically.

    This class is also a convenient way to build custom *elementary*
    mechanisms with arbitrary stoichiometry.

    Parameters
    ----------
    reactants : Mapping[str | Species, int]
        Reactant species and their (positive integer) stoichiometric
        coefficients :math:`a_i`, which are also the reaction orders.
    products : Mapping[str | Species, int]
        Product species and their coefficients :math:`b_j`. May be empty
        (e.g. irreversible decay into untracked species).
    k_f : float
        Forward rate constant :math:`k_f \ge 0`.
    k_b : float, default 0.0
        Backward rate constant :math:`k_b \ge 0`.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import MassActionReaction
    >>> rxn = MassActionReaction({"A": 1, "B": 2}, {"C": 1}, k_f=3.0)
    >>> c = {"A": np.array([2.0]), "B": np.array([1.0]), "C": np.array([0.0])}
    >>> rxn.rate(c, None, 0.0)
    array([6.])
    >>> rxn.rates(c, None, 0.0)
    {'A': array([-6.]), 'B': array([-12.]), 'C': array([6.])}
    """

    def __init__(
        self,
        reactants: Mapping[SpeciesLike, int],
        products: Mapping[SpeciesLike, int],
        k_f: float,
        k_b: float = 0.0,
    ) -> None:
        self._reactants = _validate_order_mapping(reactants, "reactants")
        self._products = _validate_order_mapping(products, "products")
        if not self._reactants:
            raise ValueError("A mass-action reaction requires at least one reactant.")
        self._k_f = _validate_rate_constant(k_f, "k_f")
        self._k_b = _validate_rate_constant(k_b, "k_b")
        if self._k_b > 0.0 and not self._products:
            raise ValueError("A reversible reaction (k_b > 0) requires products.")

        stoich: dict[str, int] = {}
        for name, a in self._reactants.items():
            stoich[name] = stoich.get(name, 0) - a
        for name, b in self._products.items():
            stoich[name] = stoich.get(name, 0) + b
        self._stoichiometry = stoich

    # -- properties ---------------------------------------------------------
    @property
    def reactants(self) -> dict[str, int]:
        """Reactant stoichiometric coefficients (forward reaction orders)."""
        return dict(self._reactants)

    @property
    def products(self) -> dict[str, int]:
        """Product stoichiometric coefficients (backward reaction orders)."""
        return dict(self._products)

    @property
    def k_f(self) -> float:
        """Forward rate constant."""
        return self._k_f

    @property
    def k_b(self) -> float:
        """Backward rate constant."""
        return self._k_b

    @property
    def is_reversible(self) -> bool:
        """``True`` if the backward rate constant is non-zero."""
        return self._k_b > 0.0

    @property
    def stoichiometry(self) -> dict[str, int]:
        """Net stoichiometric coefficients ``{species: b_i - a_i}``."""
        return dict(self._stoichiometry)

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Species appearing in a non-zero rate term."""
        names = list(self._reactants)
        if self.is_reversible:
            names += [s for s in self._products if s not in names]
        return tuple(names)

    @property
    def species(self) -> tuple[str, ...]:
        """All species appearing in the reaction equation."""
        names = list(self._reactants)
        names += [s for s in self._products if s not in names]
        return tuple(names)

    # -- kinetics -----------------------------------------------------------
    @staticmethod
    def _monomial(c: ConcentrationDict, orders: Mapping[str, int]) -> np.ndarray:
        result = np.asarray(1.0)
        for name, order in orders.items():
            result = result * _get_concentration(c, name) ** order
        return np.asarray(result, dtype=float)

    @staticmethod
    def _monomial_derivative(
        c: ConcentrationDict, orders: Mapping[str, int], wrt: str
    ) -> np.ndarray:
        result = np.asarray(float(orders[wrt]))
        for name, order in orders.items():
            power = order - 1 if name == wrt else order
            if power > 0:
                result = result * _get_concentration(c, name) ** power
        return np.asarray(result, dtype=float)

    def rate(self, c: ConcentrationDict, x: Any, t: float) -> np.ndarray:
        r"""Net mass-action rate :math:`k_f \prod c^{a} - k_b \prod c^{b}`."""
        r = self._k_f * self._monomial(c, self._reactants)
        if self.is_reversible:
            r = r - self._k_b * self._monomial(c, self._products)
        return np.asarray(r, dtype=float)

    def rate_derivatives(
        self, c: ConcentrationDict, x: Any, t: float
    ) -> dict[str, np.ndarray]:
        r"""Analytical derivatives :math:`\partial r / \partial c_k`."""
        derivs: dict[str, np.ndarray] = {}
        for name in self.dependencies:
            d = np.asarray(0.0)
            if name in self._reactants:
                d = d + self._k_f * self._monomial_derivative(c, self._reactants, name)
            if self.is_reversible and name in self._products:
                d = d - self._k_b * self._monomial_derivative(c, self._products, name)
            derivs[name] = np.asarray(d, dtype=float)
        return derivs

    def __repr__(self) -> str:
        def side(orders: Mapping[str, int]) -> str:
            if not orders:
                return "∅"
            return " + ".join(f"{n if n > 1 else ''}{s}" for s, n in orders.items())

        arrow = "<=>" if self.is_reversible else "->"
        rates = f"k_f={self._k_f!r}" + (
            f", k_b={self._k_b!r}" if self.is_reversible else ""
        )
        return (
            f"{type(self).__name__}("
            f"{side(self._reactants)} {arrow} {side(self._products)}, {rates})"
        )


class FirstOrderReaction(MassActionReaction):
    r"""Irreversible first-order (unimolecular) reaction :math:`A \xrightarrow{k} B`.

    .. math::

        R_A = -k c_A, \qquad R_B = +k c_A

    Parameters
    ----------
    reactant : str or Species
        Decaying species :math:`A`.
    product : str or Species or None
        Product species :math:`B`. ``None`` models decay into an electro-inactive,
        untracked product.
    k : float
        First-order rate constant :math:`k \ge 0` (s\ :sup:`-1`).

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import FirstOrderReaction
    >>> rxn = FirstOrderReaction("A", "B", k=0.5)
    >>> rxn.rates({"A": np.array([2.0]), "B": np.array([0.0])}, None, 0.0)
    {'A': array([-1.]), 'B': array([1.])}
    """

    def __init__(
        self, reactant: SpeciesLike, product: SpeciesLike | None, k: float
    ) -> None:
        products = {} if product is None else {product: 1}
        super().__init__({reactant: 1}, products, k_f=k)

    @property
    def k(self) -> float:
        """First-order rate constant."""
        return self.k_f


class ReversibleReaction(MassActionReaction):
    r"""Reversible first-order equilibrium :math:`A \underset{k_b}{\overset{k_f}{\rightleftharpoons}} B`.

    .. math::

        R_A = -k_f c_A + k_b c_B, \qquad R_B = +k_f c_A - k_b c_B

    Parameters
    ----------
    reactant : str or Species
        Species :math:`A`.
    product : str or Species
        Species :math:`B`.
    k_f : float
        Forward rate constant :math:`k_f \ge 0`.
    k_b : float
        Backward rate constant :math:`k_b \ge 0`.

    Examples
    --------
    >>> from softpotato.reactions import ReversibleReaction
    >>> rxn = ReversibleReaction("A", "B", k_f=10.0, k_b=2.0)
    >>> rxn.equilibrium_constant
    5.0
    >>> rxn2 = ReversibleReaction.from_equilibrium("A", "B", K=5.0, k_f=10.0)
    >>> rxn2.k_b
    2.0
    """

    def __init__(
        self, reactant: SpeciesLike, product: SpeciesLike, k_f: float, k_b: float
    ) -> None:
        super().__init__({reactant: 1}, {product: 1}, k_f=k_f, k_b=k_b)

    @classmethod
    def from_equilibrium(
        cls, reactant: SpeciesLike, product: SpeciesLike, K: float, k_f: float
    ) -> ReversibleReaction:
        r"""Create from the equilibrium constant :math:`K = k_f / k_b` and :math:`k_f`."""
        K = _validate_rate_constant(K, "K")
        if K == 0.0:
            raise ValueError("Equilibrium constant K must be positive.")
        k_f = _validate_rate_constant(k_f, "k_f")
        return cls(reactant, product, k_f=k_f, k_b=k_f / K)

    @property
    def equilibrium_constant(self) -> float:
        r"""Equilibrium constant :math:`K = k_f / k_b` (``inf`` if :math:`k_b = 0`)."""
        return self.k_f / self.k_b if self.k_b > 0.0 else math.inf


class SecondOrderReaction(MassActionReaction):
    r"""Irreversible bimolecular reaction :math:`A + B \xrightarrow{k} C`.

    .. math::

        R_A = R_B = -k c_A c_B, \qquad R_C = +k c_A c_B

    Parameters
    ----------
    reactant_1, reactant_2 : str or Species
        Reacting species :math:`A` and :math:`B` (must differ; use
        :class:`DimerizationReaction` for :math:`2A \rightarrow P`).
    product : str or Species or None
        Product :math:`C`; ``None`` if untracked.
    k : float
        Second-order rate constant :math:`k \ge 0`
        (concentration\ :sup:`-1` s\ :sup:`-1`).

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import SecondOrderReaction
    >>> rxn = SecondOrderReaction("A", "B", "C", k=2.0)
    >>> c = {"A": np.array([1.0]), "B": np.array([3.0]), "C": np.array([0.0])}
    >>> rxn.rates(c, None, 0.0)
    {'A': array([-6.]), 'B': array([-6.]), 'C': array([6.])}
    """

    def __init__(
        self,
        reactant_1: SpeciesLike,
        reactant_2: SpeciesLike,
        product: SpeciesLike | None,
        k: float,
    ) -> None:
        if _species_name(reactant_1) == _species_name(reactant_2):
            raise ValueError(
                "SecondOrderReaction requires two distinct reactants; "
                "use DimerizationReaction for 2A -> P."
            )
        products = {} if product is None else {product: 1}
        super().__init__({reactant_1: 1, reactant_2: 1}, products, k_f=k)

    @property
    def k(self) -> float:
        """Second-order rate constant."""
        return self.k_f


class CatalyticReaction(MassActionReaction):
    r"""Catalytic regeneration step of an EC' mechanism.

    .. math::

        B + Z \xrightarrow{k} A + Y

    The electrogenerated species :math:`B` reacts with a substrate :math:`Z`,
    regenerating the electroactive species :math:`A`:

    .. math::

        R_B = R_Z = -k c_B c_Z, \qquad R_A = R_Y = +k c_B c_Z

    If ``substrate`` is ``None`` the substrate is assumed to be in large excess
    and the step reduces to pseudo-first-order regeneration
    :math:`B \xrightarrow{k'} A` with :math:`k' = k c_Z^*`.

    Parameters
    ----------
    reactant : str or Species
        Electrogenerated catalyst form :math:`B`.
    regenerated : str or Species
        Regenerated catalyst form :math:`A`.
    k : float
        Rate constant (second-order, or pseudo-first-order if
        ``substrate is None``).
    substrate : str or Species or None, default None
        Substrate :math:`Z` (explicitly tracked if given).
    product : str or Species or None, default None
        Substrate-derived product :math:`Y` (explicitly tracked if given).

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import CatalyticReaction
    >>> rxn = CatalyticReaction("Red", "Ox", k=5.0, substrate="Z", product="Y")
    >>> c = {"Red": np.array([0.2]), "Ox": np.array([0.8]),
    ...      "Z": np.array([1.0]), "Y": np.array([0.0])}
    >>> rxn.rates(c, None, 0.0)
    {'Red': array([-1.]), 'Z': array([-1.]), 'Ox': array([1.]), 'Y': array([1.])}
    """

    def __init__(
        self,
        reactant: SpeciesLike,
        regenerated: SpeciesLike,
        k: float,
        substrate: SpeciesLike | None = None,
        product: SpeciesLike | None = None,
    ) -> None:
        reactants: dict[SpeciesLike, int] = {reactant: 1}
        if substrate is not None:
            reactants[substrate] = 1
        products: dict[SpeciesLike, int] = {regenerated: 1}
        if product is not None:
            products[product] = 1
        super().__init__(reactants, products, k_f=k)

    @property
    def k(self) -> float:
        """Catalytic rate constant."""
        return self.k_f


class DimerizationReaction(MassActionReaction):
    r"""Irreversible dimerization :math:`2A \xrightarrow{k} A_2`.

    With rate :math:`r = k c_A^2`:

    .. math::

        R_A = -2 k c_A^2, \qquad R_{A_2} = +k c_A^2

    Parameters
    ----------
    monomer : str or Species
        Monomer species :math:`A`.
    dimer : str or Species or None
        Dimer species :math:`A_2`; ``None`` if untracked.
    k : float
        Rate constant :math:`k \ge 0`.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import DimerizationReaction
    >>> rxn = DimerizationReaction("B", "B2", k=1.0)
    >>> rxn.rates({"B": np.array([2.0]), "B2": np.array([0.0])}, None, 0.0)
    {'B': array([-8.]), 'B2': array([4.])}
    """

    def __init__(
        self, monomer: SpeciesLike, dimer: SpeciesLike | None, k: float
    ) -> None:
        products = {} if dimer is None else {dimer: 1}
        super().__init__({monomer: 2}, products, k_f=k)

    @property
    def k(self) -> float:
        """Dimerization rate constant."""
        return self.k_f


class DisproportionationReaction(MassActionReaction):
    r"""Irreversible disproportionation :math:`2B \xrightarrow{k} A + C`.

    With rate :math:`r = k c_B^2`:

    .. math::

        R_B = -2 k c_B^2, \qquad R_A = R_C = +k c_B^2

    Parameters
    ----------
    reactant : str or Species
        Disproportionating species :math:`B`.
    product_1, product_2 : str or Species or None
        Products :math:`A` and :math:`C` (``None`` if untracked). If both refer
        to the same species it receives a coefficient of 2.
    k : float
        Rate constant :math:`k \ge 0`.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.reactions import DisproportionationReaction
    >>> rxn = DisproportionationReaction("B", "A", "C", k=0.5)
    >>> rxn.rates({"B": np.array([2.0]), "A": np.zeros(1), "C": np.zeros(1)}, None, 0.0)
    {'B': array([-4.]), 'A': array([2.]), 'C': array([2.])}
    """

    def __init__(
        self,
        reactant: SpeciesLike,
        product_1: SpeciesLike | None,
        product_2: SpeciesLike | None,
        k: float,
    ) -> None:
        products: dict[SpeciesLike, int] = {}
        for prod in (product_1, product_2):
            if prod is not None:
                name = _species_name(prod, "product")
                products[name] = products.get(name, 0) + 1
        super().__init__({reactant: 2}, products, k_f=k)

    @property
    def k(self) -> float:
        """Disproportionation rate constant."""
        return self.k_f


# ---------------------------------------------------------------------------
# Reaction network
# ---------------------------------------------------------------------------
class ReactionNetwork:
    r"""Collection of homogeneous reactions with summed rates and Jacobian.

    Computes the combined species source terms
    :math:`R_i = \sum_j \nu_{i,j} r_j` and the kinetic Jacobian
    :math:`\partial R_i / \partial c_k`. Instances are callable with the
    signature expected by :class:`~softpotato.solver.DiffusionProblem`
    (``reactions(c_dict, grid, t) -> dict``) and can be passed directly as
    ``DiffusionProblem(..., reactions=network)``.

    Parameters
    ----------
    reactions : Iterable[HomogeneousReaction], optional
        Initial reactions.
    species : Sequence[str | Species], optional
        Explicit species ordering used for the Jacobian and stoichiometry
        matrix (e.g. ``problem.species``). Defaults to the order of first
        appearance across reactions. If given, it must contain every species
        involved in the reactions.

    Examples
    --------
    EC' catalytic mechanism with pseudo-first-order regeneration, solved with
    the SciPy Method of Lines solver:

    >>> import numpy as np
    >>> from softpotato.reactions import CatalyticReaction, ReactionNetwork
    >>> from softpotato.solver import DiffusionProblem, DirichletBC, get_solver
    >>> network = ReactionNetwork([CatalyticReaction("Red", "Ox", k=10.0)])
    >>> problem = DiffusionProblem(
    ...     grid=np.linspace(0.0, 1e-3, 41),
    ...     diffusivity={"Ox": 1e-9, "Red": 1e-9},
    ...     boundary_conditions={
    ...         "Ox": (DirichletBC(0.0), DirichletBC(1.0)),
    ...         "Red": (DirichletBC(1.0), DirichletBC(0.0)),
    ...     },
    ...     initial_conditions={"Ox": 1.0, "Red": 0.0},
    ...     reactions=network,
    ... )
    >>> result = get_solver("scipy_ivp").solve(problem, t_span=(0.0, 1.0))
    >>> result.success
    True

    Kinetic Jacobian (species-major block ordering, sparse by default):

    >>> c = {"Ox": np.ones(41), "Red": np.zeros(41)}
    >>> J = network.jacobian(c, problem.grid, 0.0, species=problem.species)
    >>> J.shape
    (82, 82)
    >>> network.local_jacobian(c, problem.grid, 0.0, species=problem.species)[:, :, 0]
    array([[  0.,  10.],
           [  0., -10.]])
    """

    def __init__(
        self,
        reactions: Iterable[HomogeneousReaction] = (),
        species: Sequence[SpeciesLike] | None = None,
    ) -> None:
        self._reactions: list[HomogeneousReaction] = []
        self._species_order: list[str] | None = None
        if species is not None:
            order: list[str] = []
            for item in species:
                name = _species_name(item)
                if name in order:
                    raise ValueError(f"Species '{name}' is listed twice.")
                order.append(name)
            self._species_order = order
        for reaction in reactions:
            self.add(reaction)

    # -- container interface ------------------------------------------------
    def add(self, reaction: HomogeneousReaction) -> None:
        """Append a reaction to the network.

        Raises
        ------
        TypeError
            If ``reaction`` is not a :class:`HomogeneousReaction`.
        ValueError
            If an explicit species order was given and the reaction involves a
            species not contained in it.
        """
        if not isinstance(reaction, HomogeneousReaction):
            raise TypeError(
                f"Expected a HomogeneousReaction instance, got "
                f"{type(reaction).__name__}."
            )
        if self._species_order is not None:
            unknown = [s for s in reaction.species if s not in self._species_order]
            if unknown:
                raise ValueError(
                    f"Reaction {reaction!r} involves species {unknown} not present "
                    f"in the network species list {self._species_order}."
                )
        self._reactions.append(reaction)

    @property
    def reactions(self) -> list[HomogeneousReaction]:
        """List of reactions in the network."""
        return list(self._reactions)

    def __iter__(self) -> Iterator[HomogeneousReaction]:
        return iter(self._reactions)

    def __len__(self) -> int:
        return len(self._reactions)

    def __getitem__(self, index: int) -> HomogeneousReaction:
        return self._reactions[index]

    def __repr__(self) -> str:
        return f"ReactionNetwork({self._reactions!r})"

    @property
    def species(self) -> list[str]:
        """Ordered species names used for Jacobian and stoichiometry indexing."""
        if self._species_order is not None:
            return list(self._species_order)
        names: list[str] = []
        for reaction in self._reactions:
            for name in reaction.species:
                if name not in names:
                    names.append(name)
        return names

    @property
    def stoichiometry_matrix(self) -> np.ndarray:
        r"""Net stoichiometric matrix :math:`\nu_{i,j}` of shape ``(n_species, n_reactions)``."""
        names = self.species
        index = {name: i for i, name in enumerate(names)}
        nu = np.zeros((len(names), len(self._reactions)), dtype=float)
        for j, reaction in enumerate(self._reactions):
            for name, coeff in reaction.stoichiometry.items():
                nu[index[name], j] += coeff
        return nu

    # -- kinetics -----------------------------------------------------------
    def _profile_shape(self, c: ConcentrationDict) -> tuple[int, ...]:
        shapes = [np.shape(c[name]) for name in self.species if name in c]
        return np.broadcast_shapes(*shapes) if shapes else ()

    def evaluate(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        r"""Combined species source terms :math:`R_i = \sum_j \nu_{i,j} r_j`.

        Parameters
        ----------
        c : Mapping[str, array_like]
            Concentration profiles keyed by species name.
        x : array_like
            Spatial grid.
        t : float
            Time.

        Returns
        -------
        dict[str, np.ndarray]
            Source term for every species in :attr:`species` (zeros for
            species that are not consumed or produced).
        """
        shape = self._profile_shape(c)
        total = {name: np.zeros(shape, dtype=float) for name in self.species}
        for reaction in self._reactions:
            for name, rate in reaction.rates(c, x, t).items():
                total[name] = total[name] + rate
        return total

    def __call__(self, c: ConcentrationDict, x: Any, t: float) -> dict[str, np.ndarray]:
        """Alias for :meth:`evaluate` (``DiffusionProblem.reactions`` signature)."""
        return self.evaluate(c, x, t)

    def local_jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
    ) -> np.ndarray:
        r"""Node-wise kinetic Jacobian :math:`J_{ik}(x) = \partial R_i / \partial c_k`.

        Parameters
        ----------
        c : Mapping[str, array_like]
            Concentration profiles keyed by species name.
        x : array_like
            Spatial grid.
        t : float
            Time.
        species : Sequence[str], optional
            Species ordering for the rows/columns; defaults to :attr:`species`.
            Extra species not involved in any reaction give zero rows/columns;
            omitting a species involved in a reaction raises ``KeyError``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_species, n_species, *profile_shape)``.
        """
        names = list(species) if species is not None else self.species
        index = {name: i for i, name in enumerate(names)}
        shape = self._profile_shape(c)
        jac = np.zeros((len(names), len(names), *shape), dtype=float)
        for reaction in self._reactions:
            for (name_i, name_k), value in reaction.jacobian(c, x, t).items():
                if name_i not in index or name_k not in index:
                    raise KeyError(
                        f"Jacobian entry ({name_i!r}, {name_k!r}) refers to a species "
                        f"not contained in the species ordering {names}."
                    )
                jac[index[name_i], index[name_k]] += value
        return jac

    def jacobian(
        self,
        c: ConcentrationDict,
        x: Any,
        t: float,
        species: Sequence[str] | None = None,
        sparse: bool = True,
    ) -> Any:
        r"""Full kinetic Jacobian of the semi-discretized reaction term.

        The state vector is ordered species-major,
        :math:`\mathbf{y} = [c_{1,0}, \ldots, c_{1,N-1}, c_{2,0}, \ldots]`, so the
        Jacobian is an ``(S*N, S*N)`` block matrix whose ``(i, k)`` block is the
        diagonal matrix :math:`\mathrm{diag}(\partial R_i / \partial c_k)`.

        Parameters
        ----------
        c : Mapping[str, array_like]
            1D concentration profiles keyed by species name.
        x : array_like
            Spatial grid.
        t : float
            Time.
        species : Sequence[str], optional
            Species ordering; defaults to :attr:`species`.
        sparse : bool, default True
            Return a ``scipy.sparse.csr_matrix`` if ``True``, otherwise a dense
            ``numpy.ndarray``.

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
            Kinetic Jacobian of shape ``(S*N, S*N)``.
        """
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
