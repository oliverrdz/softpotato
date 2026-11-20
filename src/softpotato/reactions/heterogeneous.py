r"""Heterogeneous electrochemical charge-transfer kinetics and boundary helpers.

This module provides object-oriented interfacial electron-transfer kinetic models
and boundary condition generators for :class:`~softpotato.solver.DiffusionProblem`:

- :class:`HeterogeneousKinetics`: Abstract base class for finite-rate interfacial
  charge-transfer models (:math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`).
- :class:`ButlerVolmer`: General reversible/quasi-reversible Butler–Volmer
  kinetics parameterized by :math:`k^0`, :math:`\alpha`, and :math:`E^{0\prime}`.
- :class:`TafelKinetics` (alias ``Tafel``): High-overpotential / irreversible
  Tafel charge-transfer kinetics for cathodic, anodic, or automatic branches.
- :class:`NernstianEquilibrium`: Reversible thermodynamic equilibrium helper
  coupling surface concentrations to electrode potential :math:`E(t)`.
- :func:`create_electrochemical_bcs`: Helper that constructs matched
  ``(left_bc, right_bc)`` tuples for a redox pair in
  :class:`~softpotato.solver.DiffusionProblem`.

All kinetic and thermodynamic equations delegate directly to the validated
implementations in :mod:`softpotato.analytical.kinetics`.

Conventions
-----------
For the reduction reaction :math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`:

- Forward rate constant :math:`k_f(E)` governs reduction (:math:`\text{Ox} \to \text{Red}`).
- Backward rate constant :math:`k_b(E)` governs oxidation (:math:`\text{Red} \to \text{Ox}`).
- Net interfacial reduction flux is:

  .. math::

      J = k_f(E) c_{\text{Ox}}(0, t) - k_b(E) c_{\text{Red}}(0, t)

- Under the default IUPAC sign convention (``anodic_positive=True``), Faradaic
  current density is :math:`j = -n F J = n F [k_b(E) c_{\text{Red}} - k_f(E) c_{\text{Ox}}]`,
  so oxidation currents are positive (:math:`j > 0` for :math:`E > E^{0\prime}`) and
  reduction currents are negative (:math:`j < 0` for :math:`E < E^{0\prime}`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from softpotato.analytical import kinetics as _ak
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
from softpotato.reactions.species import Species
from softpotato.solver.base import BoundaryCondition, DirichletBC, NeumannBC

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _validate_finite_real(value: object, name: str) -> float:
    """Validate that a value is a finite real number (excluding bool) and return float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value}.")
    return result


def _validate_positive_finite(value: object, name: str) -> float:
    """Validate that a value is a strictly positive finite real number."""
    result = _validate_finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive ({name} > 0), got {value}.")
    return result


def _validate_alpha(alpha: object) -> float:
    """Validate that transfer coefficient alpha is finite and in (0, 1)."""
    result = _validate_finite_real(alpha, "Transfer coefficient alpha")
    if not (0.0 < result < 1.0):
        raise ValueError(
            "Transfer coefficient alpha must be strictly between 0 and 1 "
            f"(0 < alpha < 1), got {alpha}."
        )
    return result


def _validate_non_negative_conc(
    value: float | ArrayLike, name: str
) -> tuple[float | np.ndarray, bool]:
    """Validate non-negative scalar or array concentration."""
    if np.isscalar(value):
        val = _validate_finite_real(value, name)
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative (>= 0), got {value!r}.")
        return val, True
    arr = np.asarray(value, dtype=float)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (>= 0).")
    return arr, False


class HeterogeneousKinetics(ABC):
    r"""Abstract base class for heterogeneous interfacial charge-transfer kinetics.

    Models single-step electron transfer at an electrode interface:

    .. math::

        \text{Ox} + n e^- \underset{k_b(E)}{\overset{k_f(E)}{\rightleftharpoons}} \text{Red}

    Subclasses must implement :meth:`rate_constants` and provide attributes
    ``E0``, ``n``, ``T``, ``R``, and ``F``.
    """

    E0: float
    n: float
    T: float
    R: float
    F: float

    def overpotential(self, E: float | ArrayLike) -> float | np.ndarray:
        r"""Compute formal overpotential :math:`\eta = E - E^{0\prime}`.

        Parameters
        ----------
        E : float or array-like
            Electrode potential :math:`E` in Volts (V).

        Returns
        -------
        float or np.ndarray
            Overpotential :math:`E - E^{0\prime}` in Volts (V).
        """
        if np.isscalar(E):
            return float(np.asarray(E, dtype=float).item() - self.E0)
        return np.asarray(E, dtype=float) - self.E0

    @abstractmethod
    def rate_constants(
        self, E: float | ArrayLike
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""Evaluate forward and backward heterogeneous rate constants :math:`(k_f, k_b)`.

        Parameters
        ----------
        E : float or array-like
            Electrode potential :math:`E` in Volts (V).

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray)
            ``(k_f, k_b)`` where ``k_f`` is the forward (cathodic reduction) rate
            constant and ``k_b`` is the backward (anodic oxidation) rate constant.
        """
        raise NotImplementedError

    def flux(
        self,
        E: float | ArrayLike,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike = 0.0,
    ) -> float | np.ndarray:
        r"""Evaluate net interfacial reduction flux :math:`J`.

        .. math::

            J = k_f(E) c_{\text{Ox}} - k_b(E) c_{\text{Red}}

        Parameters
        ----------
        E : float or array-like
            Electrode potential :math:`E` in Volts (V).
        c_ox : float or array-like
            Surface concentration of oxidized species :math:`c_{\text{Ox}}(0, t) \ge 0`.
        c_red : float or array-like, default 0.0
            Surface concentration of reduced species :math:`c_{\text{Red}}(0, t) \ge 0`.

        Returns
        -------
        float or np.ndarray
            Net interfacial flux :math:`J` (positive for net reduction of Ox to Red).
        """
        ox_val, ox_scalar = _validate_non_negative_conc(c_ox, "c_ox")
        red_val, red_scalar = _validate_non_negative_conc(c_red, "c_red")
        kf, kb = self.rate_constants(E)
        result = kf * ox_val - kb * red_val
        if np.isscalar(E) and ox_scalar and red_scalar:
            return float(np.asarray(result, dtype=float).item())
        return np.asarray(result, dtype=float)

    def current_density(
        self,
        E: float | ArrayLike,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike = 0.0,
        *,
        anodic_positive: bool = True,
    ) -> float | np.ndarray:
        r"""Evaluate net interfacial Faradaic current density :math:`j`.

        Parameters
        ----------
        E : float or array-like
            Electrode potential :math:`E` in Volts (V).
        c_ox : float or array-like
            Surface concentration of oxidized species :math:`c_{\text{Ox}}(0, t) \ge 0`.
        c_red : float or array-like, default 0.0
            Surface concentration of reduced species :math:`c_{\text{Red}}(0, t) \ge 0`.
        anodic_positive : bool, default True
            If ``True`` (IUPAC convention), anodic oxidation current is positive
            (:math:`j = -n F J`). If ``False`` (polarographic convention),
            cathodic reduction current is positive (:math:`j = +n F J`).

        Returns
        -------
        float or np.ndarray
            Interfacial Faradaic current density :math:`j` in :math:`\text{A}/\text{m}^2`
            or :math:`\text{A}/\text{cm}^2`.
        """
        j_flux = self.flux(E, c_ox, c_red)
        sign = -1.0 if anodic_positive else 1.0
        result = sign * self.n * self.F * j_flux
        if np.isscalar(j_flux):
            return float(result)
        return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class ButlerVolmer(HeterogeneousKinetics):
    r"""Butler–Volmer heterogeneous charge-transfer kinetics.

    Models forward (cathodic) and backward (anodic) electron-transfer rate constants
    for :math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`:

    .. math::

        k_f(E) = k^0 \exp\left(-\frac{\alpha n F}{R T}(E - E^{0\prime})\right)

    .. math::

        k_b(E) = k^0 \exp\left(\frac{(1 - \alpha) n F}{R T}(E - E^{0\prime})\right)

    Rate constants, exchange current density, Faradaic current density, and Tafel
    slopes are evaluated by delegating to :mod:`softpotato.analytical.kinetics`.

    Parameters
    ----------
    k0 : float
        Standard heterogeneous rate constant :math:`k^0 > 0` in
        :math:`\text{m}/\text{s}` or :math:`\text{cm}/\text{s}`.
    alpha : float, default 0.5
        Cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    E0 : float, default 0.0
        Formal reduction potential :math:`E^{0\prime}` in Volts (V).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Examples
    --------
    >>> from softpotato.reactions import ButlerVolmer
    >>> bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.2, n=1)
    >>> kf, kb = bv.rate_constants(0.2)
    >>> kf == kb == 1e-4
    True
    >>> bv.current_density(0.2, c_ox=1.0, c_red=1.0)
    0.0
    """

    k0: float
    alpha: float = 0.5
    E0: float = 0.0
    n: float = 1
    T: float = STANDARD_TEMPERATURE
    R: float = GAS_CONSTANT
    F: float = FARADAY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "k0", _validate_positive_finite(self.k0, "Standard rate constant k0")
        )
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(
            self, "E0", _validate_finite_real(self.E0, "Formal potential E0")
        )
        object.__setattr__(
            self, "n", _validate_positive_finite(self.n, "Number of electrons n")
        )
        object.__setattr__(
            self, "T", _validate_positive_finite(self.T, "Temperature T")
        )
        object.__setattr__(
            self, "R", _validate_positive_finite(self.R, "Gas constant R")
        )
        object.__setattr__(
            self, "F", _validate_positive_finite(self.F, "Faraday constant F")
        )

    def k_f(self, E: float | ArrayLike) -> float | np.ndarray:
        r"""Evaluate forward (cathodic reduction) rate constant :math:`k_f(E)`."""
        eta = self.overpotential(E)
        return _ak.tafel(
            eta,
            i0=self.k0,
            alpha=self.alpha,
            branch="cathodic",
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
            anodic_positive=False,
        )

    def k_b(self, E: float | ArrayLike) -> float | np.ndarray:
        r"""Evaluate backward (anodic oxidation) rate constant :math:`k_b(E)`."""
        eta = self.overpotential(E)
        return _ak.tafel(
            eta,
            i0=self.k0,
            alpha=self.alpha,
            branch="anodic",
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
            anodic_positive=True,
        )

    def rate_constants(
        self, E: float | ArrayLike
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""Evaluate :math:`(k_f(E), k_b(E))` at electrode potential :math:`E`."""
        return self.k_f(E), self.k_b(E)

    def exchange_current_density(self, c_ox: float = 1.0, c_red: float = 1.0) -> float:
        r"""Compute exchange current density :math:`j_0` via :func:`softpotato.analytical.kinetics.exchange_current_density`."""
        return _ak.exchange_current_density(
            self.k0,
            c_ox=c_ox,
            c_red=c_red,
            alpha=self.alpha,
            n=self.n,
            F=self.F,
        )

    def current_density(
        self,
        E: float | ArrayLike,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike = 0.0,
        *,
        anodic_positive: bool = True,
    ) -> float | np.ndarray:
        r"""Evaluate net Butler–Volmer current density via :func:`softpotato.analytical.kinetics.butler_volmer_current_density`."""
        all_scalar = np.isscalar(E) and np.isscalar(c_ox) and np.isscalar(c_red)
        eta = self.overpotential(E)
        if not all_scalar:
            eta = np.asarray(eta, dtype=float)

        j0_unit = _ak.exchange_current_density(
            self.k0,
            c_ox=1.0,
            c_red=1.0,
            alpha=self.alpha,
            n=self.n,
            F=self.F,
        )
        return _ak.butler_volmer_current_density(
            eta,
            j0=j0_unit,
            alpha=self.alpha,
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
            c_ox_surf=c_ox,
            c_red_surf=c_red,
            c_ox_bulk=1.0,
            c_red_bulk=1.0,
            anodic_positive=anodic_positive,
        )

    def tafel_slope(
        self,
        branch: Literal["anodic", "cathodic"] = "anodic",
        *,
        signed: bool = True,
    ) -> float:
        r"""Compute theoretical Tafel slope :math:`b` in :math:`\text{V}/\text{dec}` via :func:`softpotato.analytical.kinetics.tafel_slope`."""
        return _ak.tafel_slope(
            alpha=self.alpha,
            n=self.n,
            branch=branch,
            T=self.T,
            R=self.R,
            F=self.F,
            signed=signed,
        )


@dataclass(frozen=True)
class TafelKinetics(HeterogeneousKinetics):
    r"""High-overpotential / irreversible Tafel heterogeneous charge-transfer kinetics.

    Retains only the dominant exponential branch of Butler–Volmer kinetics:

    - ``branch='cathodic'`` (irreversible reduction):
      :math:`k_f(E) = k^0 \exp(-\alpha n F (E - E^{0\prime}) / (R T))`, :math:`k_b(E) = 0`.
    - ``branch='anodic'`` (irreversible oxidation):
      :math:`k_f(E) = 0`, :math:`k_b(E) = k^0 \exp((1 - \alpha) n F (E - E^{0\prime}) / (R T))`.
    - ``branch='auto'``: selects the anodic branch for :math:`E \ge E^{0\prime}` and
      the cathodic branch for :math:`E < E^{0\prime}`.

    All rate constants, current densities, Tafel slopes, and inverse overpotentials
    delegate directly to :mod:`softpotato.analytical.kinetics`.

    Parameters
    ----------
    k0 : float
        Standard heterogeneous rate constant :math:`k^0 > 0` in
        :math:`\text{m}/\text{s}` or :math:`\text{cm}/\text{s}`.
    alpha : float, default 0.5
        Cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    E0 : float, default 0.0
        Formal reduction potential :math:`E^{0\prime}` in Volts (V).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    branch : {'auto', 'anodic', 'cathodic'}, default 'auto'
        Polarization branch to model.
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Examples
    --------
    >>> from softpotato.reactions import TafelKinetics
    >>> tk = TafelKinetics(k0=1e-4, alpha=0.5, E0=0.0, n=1, branch="cathodic")
    >>> kf, kb = tk.rate_constants(-0.1)
    >>> kf > 1e-4 and kb == 0.0
    True
    >>> round(tk.tafel_slope(signed=True) * 1000, 1)
    -118.3
    """

    k0: float
    alpha: float = 0.5
    E0: float = 0.0
    n: float = 1
    branch: Literal["auto", "anodic", "cathodic"] = "auto"
    T: float = STANDARD_TEMPERATURE
    R: float = GAS_CONSTANT
    F: float = FARADAY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "k0", _validate_positive_finite(self.k0, "Standard rate constant k0")
        )
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(
            self, "E0", _validate_finite_real(self.E0, "Formal potential E0")
        )
        object.__setattr__(
            self, "n", _validate_positive_finite(self.n, "Number of electrons n")
        )
        if not isinstance(self.branch, str):
            raise TypeError(
                f"branch must be a string ('auto', 'anodic', or 'cathodic'), "
                f"got {type(self.branch).__name__}."
            )
        branch_norm = self.branch.lower().strip()
        if branch_norm not in ("auto", "anodic", "cathodic"):
            raise ValueError(
                f"Invalid branch '{self.branch}'. Must be 'auto', 'anodic', or 'cathodic'."
            )
        object.__setattr__(self, "branch", branch_norm)
        object.__setattr__(
            self, "T", _validate_positive_finite(self.T, "Temperature T")
        )
        object.__setattr__(
            self, "R", _validate_positive_finite(self.R, "Gas constant R")
        )
        object.__setattr__(
            self, "F", _validate_positive_finite(self.F, "Faraday constant F")
        )

    def rate_constants(
        self, E: float | ArrayLike
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""Evaluate Tafel rate constants :math:`(k_f(E), k_b(E))` via :func:`softpotato.analytical.kinetics.tafel`."""
        is_scalar = np.isscalar(E)
        eta = self.overpotential(E)

        if self.branch == "cathodic":
            kf = _ak.tafel(
                eta,
                i0=self.k0,
                alpha=self.alpha,
                branch="cathodic",
                n=self.n,
                T=self.T,
                R=self.R,
                F=self.F,
                anodic_positive=False,
            )
            kb: float | np.ndarray = (
                0.0 if is_scalar else np.zeros_like(kf, dtype=float)
            )
            return kf, kb

        if self.branch == "anodic":
            kb = _ak.tafel(
                eta,
                i0=self.k0,
                alpha=self.alpha,
                branch="anodic",
                n=self.n,
                T=self.T,
                R=self.R,
                F=self.F,
                anodic_positive=True,
            )
            kf = 0.0 if is_scalar else np.zeros_like(kb, dtype=float)
            return kf, kb

        # branch == "auto"
        kf_raw = _ak.tafel(
            eta,
            i0=self.k0,
            alpha=self.alpha,
            branch="cathodic",
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
            anodic_positive=False,
        )
        kb_raw = _ak.tafel(
            eta,
            i0=self.k0,
            alpha=self.alpha,
            branch="anodic",
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
            anodic_positive=True,
        )
        if is_scalar:
            if float(eta) >= 0.0:
                return 0.0, float(kb_raw)
            return float(kf_raw), 0.0
        eta_arr = np.asarray(eta, dtype=float)
        return (
            np.where(eta_arr < 0.0, kf_raw, 0.0),
            np.where(eta_arr >= 0.0, kb_raw, 0.0),
        )

    def current_density(
        self,
        E: float | ArrayLike,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike = 0.0,
        *,
        anodic_positive: bool = True,
    ) -> float | np.ndarray:
        r"""Evaluate Tafel Faradaic current density via :func:`softpotato.analytical.kinetics.tafel`."""
        ox_val, ox_scalar = _validate_non_negative_conc(c_ox, "c_ox")
        red_val, red_scalar = _validate_non_negative_conc(c_red, "c_red")
        all_scalar = np.isscalar(E) and ox_scalar and red_scalar
        eta = self.overpotential(E)
        j0_unit = _ak.exchange_current_density(
            self.k0,
            c_ox=1.0,
            c_red=1.0,
            alpha=self.alpha,
            n=self.n,
            F=self.F,
        )

        if self.branch == "anodic":
            j_unit = _ak.tafel(
                eta,
                i0=j0_unit,
                alpha=self.alpha,
                branch="anodic",
                n=self.n,
                T=self.T,
                R=self.R,
                F=self.F,
                anodic_positive=anodic_positive,
            )
            res = j_unit * red_val
        elif self.branch == "cathodic":
            j_unit = _ak.tafel(
                eta,
                i0=j0_unit,
                alpha=self.alpha,
                branch="cathodic",
                n=self.n,
                T=self.T,
                R=self.R,
                F=self.F,
                anodic_positive=anodic_positive,
            )
            res = j_unit * ox_val
        else:
            j_anodic = (
                _ak.tafel(
                    eta,
                    i0=j0_unit,
                    alpha=self.alpha,
                    branch="anodic",
                    n=self.n,
                    T=self.T,
                    R=self.R,
                    F=self.F,
                    anodic_positive=anodic_positive,
                )
                * red_val
            )
            j_cathodic = (
                _ak.tafel(
                    eta,
                    i0=j0_unit,
                    alpha=self.alpha,
                    branch="cathodic",
                    n=self.n,
                    T=self.T,
                    R=self.R,
                    F=self.F,
                    anodic_positive=anodic_positive,
                )
                * ox_val
            )
            eta_arr = np.asarray(eta, dtype=float)
            res = np.where(eta_arr >= 0.0, j_anodic, j_cathodic)

        if all_scalar:
            return float(np.asarray(res, dtype=float).item())
        return np.asarray(res, dtype=float)

    def tafel_slope(
        self,
        branch: Literal["anodic", "cathodic"] | None = None,
        *,
        signed: bool = True,
    ) -> float:
        r"""Compute theoretical Tafel slope :math:`b` in :math:`\text{V}/\text{dec}` via :func:`softpotato.analytical.kinetics.tafel_slope`."""
        resolved: Literal["anodic", "cathodic"]
        if branch is None:
            if self.branch == "auto":
                raise ValueError(
                    "Tafel slope requires an explicit branch ('anodic' or 'cathodic') "
                    "when TafelKinetics.branch='auto'."
                )
            resolved = self.branch
        else:
            resolved = branch
        return _ak.tafel_slope(
            alpha=self.alpha,
            n=self.n,
            branch=resolved,
            T=self.T,
            R=self.R,
            F=self.F,
            signed=signed,
        )

    def overpotential_for_current(
        self,
        current_density: float | ArrayLike,
        *,
        c_ox: float = 1.0,
        c_red: float = 1.0,
        branch: Literal["anodic", "cathodic"] | None = None,
    ) -> float | np.ndarray:
        r"""Compute activation overpotential :math:`\eta` for a target current density via :func:`softpotato.analytical.kinetics.tafel_overpotential`."""
        resolved: Literal["anodic", "cathodic"]
        if branch is None:
            if self.branch == "auto":
                raise ValueError(
                    "overpotential_for_current requires an explicit branch "
                    "('anodic' or 'cathodic') when TafelKinetics.branch='auto'."
                )
            resolved = self.branch
        else:
            resolved = branch

        c_ref = c_red if resolved == "anodic" else c_ox
        _validate_positive_finite(c_ref, "c_red" if resolved == "anodic" else "c_ox")
        j0_branch = _ak.exchange_current_density(
            self.k0,
            c_ox=c_ref,
            c_red=c_ref,
            alpha=self.alpha,
            n=self.n,
            F=self.F,
        )
        return _ak.tafel_overpotential(
            current_density,
            i0=j0_branch,
            alpha=self.alpha,
            branch=resolved,
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
        )


Tafel = TafelKinetics


@dataclass(frozen=True)
class NernstianEquilibrium:
    r"""Reversible Nernstian charge-transfer equilibrium helper.

    Enforces thermodynamic surface equilibrium at an electrode interface:

    .. math::

        \frac{c_{\text{Ox}}(0, t)}{c_{\text{Red}}(0, t)} =
        \exp\left(\frac{n F}{R T}(E(t) - E^{0\prime})\right)

    All thermodynamic calculations delegate directly to
    :func:`~softpotato.analytical.kinetics.nernst`,
    :func:`~softpotato.analytical.kinetics.nernst_ratio`, and
    :func:`~softpotato.analytical.kinetics.nernst_equilibrium_concentrations`.

    Parameters
    ----------
    E0 : float, default 0.0
        Formal reduction potential :math:`E^{0\prime}` in Volts (V).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Examples
    --------
    >>> from softpotato.reactions import NernstianEquilibrium
    >>> eq = NernstianEquilibrium(E0=0.2, n=1)
    >>> eq.concentration_ratio(0.2)
    1.0
    >>> eq.surface_concentrations(0.2, c_total=2.0)
    (1.0, 1.0)
    """

    E0: float = 0.0
    n: float = 1
    T: float = STANDARD_TEMPERATURE
    R: float = GAS_CONSTANT
    F: float = FARADAY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "E0", _validate_finite_real(self.E0, "Formal potential E0")
        )
        object.__setattr__(
            self, "n", _validate_positive_finite(self.n, "Number of electrons n")
        )
        object.__setattr__(
            self, "T", _validate_positive_finite(self.T, "Temperature T")
        )
        object.__setattr__(
            self, "R", _validate_positive_finite(self.R, "Gas constant R")
        )
        object.__setattr__(
            self, "F", _validate_positive_finite(self.F, "Faraday constant F")
        )

    def potential(
        self,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike | None = None,
    ) -> float | np.ndarray:
        r"""Compute equilibrium electrode potential :math:`E_{\text{eq}}` via :func:`softpotato.analytical.kinetics.nernst`."""
        return _ak.nernst(
            c_ox,
            c_red=c_red,
            E0=self.E0,
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
        )

    def concentration_ratio(self, E: float | ArrayLike) -> float | np.ndarray:
        r"""Compute equilibrium ratio :math:`c_{\text{Ox}} / c_{\text{Red}}` via :func:`softpotato.analytical.kinetics.nernst_ratio`."""
        return _ak.nernst_ratio(
            E,
            E0=self.E0,
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
        )

    def surface_concentrations(
        self,
        E: float | ArrayLike,
        c_total: float | ArrayLike = 1.0,
        *,
        D_ox: float | None = None,
        D_red: float | None = None,
        c_ox_bulk: float | None = None,
        c_red_bulk: float | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""Compute equilibrium surface concentrations :math:`(c_{\text{Ox}}(0), c_{\text{Red}}(0))`.

        When ``D_ox`` and ``D_red`` are omitted or equal, delegates directly to
        :func:`softpotato.analytical.kinetics.nernst_equilibrium_concentrations`
        with :math:`c_{\text{Ox}}(0) + c_{\text{Red}}(0) = c_{\text{total}}`.

        When unequal diffusivities ``D_ox != D_red`` are provided, applies the
        semi-infinite planar flux-balanced Nernstian relation using the reversible
        half-wave potential
        :math:`E_{1/2} = E^{0\prime} - \frac{R T}{n F} \ln\sqrt{D_{\text{Ox}} / D_{\text{Red}}}`.

        Parameters
        ----------
        E : float or array-like
            Electrode potential :math:`E` in Volts (V).
        c_total : float or array-like, default 1.0
            Total electroactive concentration :math:`c_{\text{Ox}}^* + c_{\text{Red}}^*`.
        D_ox : float, optional
            Diffusion coefficient of oxidized species (:math:`D_{\text{Ox}} > 0`).
        D_red : float, optional
            Diffusion coefficient of reduced species (:math:`D_{\text{Red}} > 0`).
        c_ox_bulk : float, optional
            Bulk concentration of oxidized species (used when ``D_ox != D_red``).
        c_red_bulk : float, optional
            Bulk concentration of reduced species (used when ``D_ox != D_red``).

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray)
            Equilibrium surface concentrations ``(c_ox_surf, c_red_surf)``.
        """
        if D_ox is not None:
            D_ox = _validate_positive_finite(D_ox, "D_ox")
        if D_red is not None:
            D_red = _validate_positive_finite(D_red, "D_red")

        if D_ox is None or D_red is None or math.isclose(D_ox, D_red, rel_tol=1e-12):
            return _ak.nernst_equilibrium_concentrations(
                E,
                c_total=c_total,
                E0=self.E0,
                n=self.n,
                T=self.T,
                R=self.R,
                F=self.F,
            )

        xi = math.sqrt(D_ox / D_red)
        e_half = self.E0 - (self.R * self.T / (self.n * self.F)) * math.log(xi)

        if c_ox_bulk is None and c_red_bulk is None:
            c_ox_b: float | np.ndarray = _validate_non_negative_conc(
                c_total, "c_total"
            )[0]
            c_red_b: float | np.ndarray = 0.0
        else:
            c_ox_b = (
                0.0
                if c_ox_bulk is None
                else _validate_non_negative_conc(c_ox_bulk, "c_ox_bulk")[0]
            )
            c_red_b = (
                0.0
                if c_red_bulk is None
                else _validate_non_negative_conc(c_red_bulk, "c_red_bulk")[0]
            )

        s_ox = c_ox_b + c_red_b / xi
        s_red = xi * c_ox_b + c_red_b

        f_ox, f_red = _ak.nernst_equilibrium_concentrations(
            E,
            c_total=1.0,
            E0=e_half,
            n=self.n,
            T=self.T,
            R=self.R,
            F=self.F,
        )
        c_ox_out = s_ox * f_ox
        c_red_out = s_red * f_red

        if np.isscalar(E) and np.isscalar(s_ox) and np.isscalar(s_red):
            return (
                float(np.asarray(c_ox_out, dtype=float).item()),
                float(np.asarray(c_red_out, dtype=float).item()),
            )
        return np.asarray(c_ox_out, dtype=float), np.asarray(c_red_out, dtype=float)


def _resolve_species_params(
    spec: Species | str,
    role: str,
    diffusivity: float | None,
    bulk_conc: float | None,
    default_bulk: float,
) -> tuple[str, float | None, float]:
    """Resolve species name, diffusivity, and bulk concentration from Species or str."""
    d_val: float | None
    if isinstance(spec, Species):
        name = spec.name
        d_val = (
            spec.diffusivity
            if diffusivity is None
            else _validate_positive_finite(diffusivity, f"D_{role}")
        )
        c_val = (
            spec.bulk_concentration
            if bulk_conc is None
            else _validate_finite_real(bulk_conc, f"c_{role}_bulk")
        )
    elif isinstance(spec, str):
        if not spec.strip():
            raise ValueError(f"{role} species name must be a non-empty string.")
        name = spec
        d_val = (
            None
            if diffusivity is None
            else _validate_positive_finite(diffusivity, f"D_{role}")
        )
        c_val = (
            default_bulk
            if bulk_conc is None
            else _validate_finite_real(bulk_conc, f"c_{role}_bulk")
        )
    else:
        raise TypeError(
            f"{role} must be a Species instance or str, got {type(spec).__name__}."
        )

    if c_val < 0.0:
        raise ValueError(f"c_{role}_bulk must be non-negative (>= 0), got {c_val}.")
    return name, d_val, c_val


def create_electrochemical_bcs(
    kinetics: HeterogeneousKinetics | NernstianEquilibrium,
    potential: float | Callable[[float], float],
    ox: Species | str = "Ox",
    red: Species | str = "Red",
    *,
    D_ox: float | None = None,
    D_red: float | None = None,
    c_ox_bulk: float | None = None,
    c_red_bulk: float | None = None,
    c_ox_surface: float | Callable[[float], float] | None = None,
    c_red_surface: float | Callable[[float], float] | None = None,
    electrode_side: Literal["left", "right"] = "left",
    bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet",
) -> dict[str, tuple[BoundaryCondition, BoundaryCondition]]:
    r"""Construct matched ``(left_bc, right_bc)`` boundary conditions for a redox pair.

    Generates a dictionary mapping ``{ox_name: (left_bc, right_bc), red_name: (left_bc, right_bc)}``
    compatible with :class:`~softpotato.solver.DiffusionProblem`:

    - For :class:`NernstianEquilibrium`: generates time-dependent
      :func:`~softpotato.solver.DirichletBC` conditions enforcing the Nernst
      equilibrium surface concentrations at the electrode boundary.
    - For :class:`HeterogeneousKinetics` (e.g., :class:`ButlerVolmer`,
      :class:`TafelKinetics`): generates matched :func:`~softpotato.solver.NeumannBC`
      flux boundary conditions satisfying flux continuity
      :math:`J_{\text{Ox}} = -J_{\text{Red}}`:

      .. math::

          \left.\frac{\partial c_{\text{Ox}}}{\partial x}\right|_{\text{left}} = +\frac{J(t)}{D_{\text{Ox}}}, \qquad
          \left.\frac{\partial c_{\text{Red}}}{\partial x}\right|_{\text{left}} = -\frac{J(t)}{D_{\text{Red}}}

    Parameters
    ----------
    kinetics : HeterogeneousKinetics or NernstianEquilibrium
        Interfacial charge-transfer model or Nernstian equilibrium helper.
    potential : float or Callable[[float], float]
        Constant electrode potential :math:`E` or time-dependent waveform :math:`E(t)` in Volts.
    ox : Species or str, default 'Ox'
        Oxidized species object or name.
    red : Species or str, default 'Red'
        Reduced species object or name.
    D_ox : float, optional
        Diffusion coefficient of oxidized species. Inferred from ``ox`` if a
        :class:`~softpotato.reactions.Species` is passed; required for
        :class:`HeterogeneousKinetics` when ``ox`` is a string.
    D_red : float, optional
        Diffusion coefficient of reduced species. Inferred from ``red`` if a
        :class:`~softpotato.reactions.Species` is passed; required for
        :class:`HeterogeneousKinetics` when ``red`` is a string.
    c_ox_bulk : float, optional
        Bulk concentration of oxidized species (inferred from ``ox`` if
        :class:`~softpotato.reactions.Species`, else defaults to ``1.0``).
    c_red_bulk : float, optional
        Bulk concentration of reduced species (inferred from ``red`` if
        :class:`~softpotato.reactions.Species`, else defaults to ``0.0``).
    c_ox_surface : float or Callable[[float], float], optional
        Prescribed surface concentration :math:`c_{\text{Ox}}(0, t)` used when
        evaluating Neumann kinetic fluxes (defaults to ``c_ox_bulk``).
    c_red_surface : float or Callable[[float], float], optional
        Prescribed surface concentration :math:`c_{\text{Red}}(0, t)` used when
        evaluating Neumann kinetic fluxes (defaults to ``c_red_bulk``).
    electrode_side : {'left', 'right'}, default 'left'
        Domain boundary acting as the working electrode.
    bulk_bc : {'dirichlet', 'neumann'}, default 'dirichlet'
        Boundary condition type at the opposite (bulk) boundary: ``'dirichlet'``
        fixes bulk concentrations :math:`c_i^*`, while ``'neumann'`` applies a
        zero-flux condition (:math:`\partial c_i / \partial x = 0`).

    Returns
    -------
    dict[str, tuple[BoundaryCondition, BoundaryCondition]]
        Dictionary ready to pass to ``DiffusionProblem(..., boundary_conditions=...)``.

    Examples
    --------
    >>> from softpotato.reactions import (
    ...     NernstianEquilibrium,
    ...     Species,
    ...     create_electrochemical_bcs,
    ... )
    >>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0)
    >>> red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0)
    >>> eq = NernstianEquilibrium(E0=0.0, n=1)
    >>> bcs = create_electrochemical_bcs(eq, potential=0.0, ox=ox, red=red)
    >>> bcs["Ox"][0].evaluate(0.0), bcs["Red"][0].evaluate(0.0)
    (0.5, 0.5)
    """
    if electrode_side not in ("left", "right"):
        raise ValueError(
            f"electrode_side must be 'left' or 'right', got {electrode_side!r}."
        )
    if bulk_bc not in ("dirichlet", "neumann"):
        raise ValueError(f"bulk_bc must be 'dirichlet' or 'neumann', got {bulk_bc!r}.")

    ox_name, d_ox_val, c_ox_b = _resolve_species_params(
        ox, "ox", D_ox, c_ox_bulk, default_bulk=1.0
    )
    red_name, d_red_val, c_red_b = _resolve_species_params(
        red, "red", D_red, c_red_bulk, default_bulk=0.0
    )
    if ox_name == red_name:
        raise ValueError(
            f"Oxidized and reduced species must have distinct names, both got '{ox_name}'."
        )

    if callable(potential):
        eval_potential = potential
    else:
        const_e = _validate_finite_real(potential, "potential")

        def eval_potential(_t: float) -> float:
            return const_e

    electrode_bc_ox: BoundaryCondition
    electrode_bc_red: BoundaryCondition

    if isinstance(kinetics, NernstianEquilibrium):
        c_total = c_ox_b + c_red_b

        def _ox_dirichlet(t: float) -> float:
            c_ox_s, _ = kinetics.surface_concentrations(
                float(eval_potential(t)),
                c_total=c_total,
                D_ox=d_ox_val,
                D_red=d_red_val,
                c_ox_bulk=c_ox_b,
                c_red_bulk=c_red_b,
            )
            return float(c_ox_s)

        def _red_dirichlet(t: float) -> float:
            _, c_red_s = kinetics.surface_concentrations(
                float(eval_potential(t)),
                c_total=c_total,
                D_ox=d_ox_val,
                D_red=d_red_val,
                c_ox_bulk=c_ox_b,
                c_red_bulk=c_red_b,
            )
            return float(c_red_s)

        electrode_bc_ox = DirichletBC(_ox_dirichlet)
        electrode_bc_red = DirichletBC(_red_dirichlet)

    elif isinstance(kinetics, HeterogeneousKinetics):
        if d_ox_val is None or d_red_val is None:
            raise ValueError(
                "D_ox and D_red must be provided (directly or via Species instances) "
                "when creating Neumann boundary conditions for HeterogeneousKinetics."
            )

        if callable(c_ox_surface):
            eval_c_ox = c_ox_surface
        else:
            c_ox_const = (
                c_ox_b
                if c_ox_surface is None
                else float(_validate_non_negative_conc(c_ox_surface, "c_ox_surface")[0])
            )

            def eval_c_ox(_t: float) -> float:
                return c_ox_const

        if callable(c_red_surface):
            eval_c_red = c_red_surface
        else:
            c_red_const = (
                c_red_b
                if c_red_surface is None
                else float(
                    _validate_non_negative_conc(c_red_surface, "c_red_surface")[0]
                )
            )

            def eval_c_red(_t: float) -> float:
                return c_red_const

        side_sign = 1.0 if electrode_side == "left" else -1.0
        d_ox_f = d_ox_val
        d_red_f = d_red_val

        def _ox_neumann(t: float) -> float:
            flux_val = float(
                kinetics.flux(
                    float(eval_potential(t)),
                    float(eval_c_ox(t)),
                    float(eval_c_red(t)),
                )
            )
            return side_sign * flux_val / d_ox_f

        def _red_neumann(t: float) -> float:
            flux_val = float(
                kinetics.flux(
                    float(eval_potential(t)),
                    float(eval_c_ox(t)),
                    float(eval_c_red(t)),
                )
            )
            return -side_sign * flux_val / d_red_f

        electrode_bc_ox = NeumannBC(_ox_neumann)
        electrode_bc_red = NeumannBC(_red_neumann)

    else:
        raise TypeError(
            "kinetics must be an instance of HeterogeneousKinetics or "
            f"NernstianEquilibrium, got {type(kinetics).__name__}."
        )

    if bulk_bc == "dirichlet":
        bulk_bc_ox = DirichletBC(c_ox_b)
        bulk_bc_red = DirichletBC(c_red_b)
    else:
        bulk_bc_ox = NeumannBC(0.0)
        bulk_bc_red = NeumannBC(0.0)

    if electrode_side == "left":
        return {
            ox_name: (electrode_bc_ox, bulk_bc_ox),
            red_name: (electrode_bc_red, bulk_bc_red),
        }
    return {
        ox_name: (bulk_bc_ox, electrode_bc_ox),
        red_name: (bulk_bc_red, electrode_bc_red),
    }


__all__ = [
    "ButlerVolmer",
    "HeterogeneousKinetics",
    "NernstianEquilibrium",
    "Tafel",
    "TafelKinetics",
    "create_electrochemical_bcs",
]
