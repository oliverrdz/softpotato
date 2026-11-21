"""Abstract base class and shared validation utilities for electrochemical techniques."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from softpotato.geometry import Geometry
    from softpotato.grid.base import Mesh1D
    from softpotato.reactions.mechanisms import Mechanism
    from softpotato.reactions.species import Species, SpeciesCollection
    from softpotato.solver.base import BaseSolver

    from .result import TechniqueResult

ControlMode = Literal["potentiostatic", "galvanostatic"]

_VALID_MODES: tuple[str, ...] = ("potentiostatic", "galvanostatic")
_DEFAULT_TIME_GRID_POINTS = 1001


@dataclass(frozen=True)
class PlotSpec:
    """Declarative specification for a plot supported by an electrochemical technique.

    Attributes
    ----------
    kind : str
        Canonical concise plot key (e.g., ``'voltammogram'``, ``'i_vs_t'``,
        ``'E_vs_t'``, ``'c_vs_x'``).
    title : str
        Default plot title.
    x_label : str
        Default horizontal axis label including units.
    y_label : str
        Default vertical axis label including units.
    description : str, default ''
        Short human-readable description of the plot.
    """

    kind: str
    title: str
    x_label: str
    y_label: str
    description: str = ""


PLOT_ALIASES: dict[str, str] = {
    "current_vs_potential": "voltammogram",
    "i_vs_e": "voltammogram",
    "current_vs_time": "i_vs_t",
    "chronoamperogram": "i_vs_t",
    "potential_vs_time": "E_vs_t",
    "e_vs_t": "E_vs_t",
    "chronopotentiogram": "E_vs_t",
    "concentration_profile": "c_vs_x",
    "concentration_vs_distance": "c_vs_x",
    "charge_vs_time": "Q_vs_t",
    "q_vs_t": "Q_vs_t",
    "chronocoulogram": "Q_vs_t",
    "differential_pulse": "dpv",
    "differential_pulse_voltammogram": "dpv",
    "square_wave": "swv",
    "square_wave_voltammogram": "swv",
    "sampled_current": "scv",
    "sampled_current_voltammogram": "scv",
}

_CANONICAL_PLOT_KINDS: dict[str, str] = {
    "voltammogram": "voltammogram",
    "i_vs_t": "i_vs_t",
    "e_vs_t": "E_vs_t",
    "c_vs_x": "c_vs_x",
    "q_vs_t": "Q_vs_t",
    "dpv": "dpv",
    "swv": "swv",
    "scv": "scv",
}

_UNIVERSAL_PLOT_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec(
        kind="i_vs_t",
        title="Current vs. Time",
        x_label="$t$ / s",
        y_label="$I$ / A",
        description="Net Faradaic current transient I(t).",
    ),
    PlotSpec(
        kind="E_vs_t",
        title="Potential vs. Time",
        x_label="$t$ / s",
        y_label="$E$ / V",
        description="Electrode potential E(t) vs. time.",
    ),
    PlotSpec(
        kind="c_vs_x",
        title="Concentration Profile",
        x_label="$x$ / m",
        y_label=r"$c$ / $\mathrm{mol\,m^{-3}}$",
        description="Spatial species concentration profile c(x, t) at a specified time.",
    ),
)


def resolve_plot_kind(kind: str) -> str:
    """Resolve a concise plot key or verbose alias to its canonical plot kind.

    Parameters
    ----------
    kind : str
        Concise plot key (e.g., ``'i_vs_t'``, ``'E_vs_t'``, ``'c_vs_x'``,
        ``'voltammogram'``) or verbose alias (e.g., ``'current_vs_time'``,
        ``'potential_vs_time'``, ``'concentration_profile'``).

    Returns
    -------
    str
        Canonical plot key if recognized, otherwise the stripped input string.
    """
    if not isinstance(kind, str):
        raise TypeError(f"Plot kind must be a string, got {type(kind).__name__}.")
    cleaned = kind.strip()
    lowered = cleaned.lower()
    if lowered in _CANONICAL_PLOT_KINDS:
        return _CANONICAL_PLOT_KINDS[lowered]
    if lowered in PLOT_ALIASES:
        return PLOT_ALIASES[lowered]
    return cleaned


def _validate_finite_real(value: object, name: str) -> float:
    """Validate that a value is a finite real number (excluding bool) and return float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _validate_positive_finite(value: object, name: str) -> float:
    """Validate that a value is a strictly positive finite real number."""
    result = _validate_finite_real(value, name)
    if result <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive ({name} > 0), got {value!r}."
        )
    return result


def _validate_non_negative_finite(value: object, name: str) -> float:
    """Validate that a value is a non-negative finite real number."""
    result = _validate_finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative ({name} >= 0), got {value!r}.")
    return result


def _validate_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate that a value is an integer >= minimum (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _validate_mode(mode: object) -> ControlMode:
    """Validate control mode is either 'potentiostatic' or 'galvanostatic'."""
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}.")
    return mode  # type: ignore[return-value]


def _normalize_time_input(t: float | ArrayLike) -> tuple[float | np.ndarray, bool]:
    """Validate and normalize scalar or array time input.

    Returns
    -------
    tuple[float | np.ndarray, bool]
        ``(normalized_value, is_scalar)``.
    """
    if np.isscalar(t):
        return _validate_finite_real(t, "t"), True
    arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("All time values in t must be finite.")
    return arr, False


def _deduplicate_interior_events(
    times: Iterable[float],
    duration: float,
    *,
    tol: float | None = None,
) -> tuple[float, ...]:
    """Sort and deduplicate strictly interior event times in ``(0, duration)``."""
    if tol is None:
        tol = 1e-12 * max(1.0, duration)
    cleaned: list[float] = []
    for raw in sorted(float(t) for t in times):
        if raw <= tol or raw >= duration - tol:
            continue
        if not cleaned or (raw - cleaned[-1]) > tol:
            cleaned.append(raw)
    return tuple(cleaned)


def _merge_time_grid(
    base_times: np.ndarray,
    duration: float,
    required_times: Iterable[float],
    *,
    tol: float | None = None,
) -> np.ndarray:
    """Merge ``base_times`` with exact ``0.0``, ``duration``, and ``required_times``.

    Near-duplicate points within ``tol`` of any required timestamp are dropped in
    favor of the exact required timestamp so that endpoints and switching/step
    events are represented exactly.
    """
    if tol is None:
        tol = 1e-12 * max(1.0, duration)

    exact_points = sorted(
        {
            0.0,
            float(duration),
            *(float(t) for t in required_times if 0.0 <= float(t) <= duration),
        }
    )
    # Deduplicate exact_points within tol while keeping exact 0.0 and duration
    pinned: list[float] = []
    for pt in exact_points:
        if not pinned or (pt - pinned[-1]) > tol:
            pinned.append(pt)
    if pinned[-1] != duration:
        pinned[-1] = float(duration)
    pinned_arr = np.asarray(pinned, dtype=float)

    base = np.asarray(base_times, dtype=float).ravel()
    in_bounds = base[(base > tol) & (base < duration - tol)]

    if in_bounds.size > 0:
        # Remove base points within tol of any pinned timestamp
        idx = np.searchsorted(pinned_arr, in_bounds)
        left_dist = np.abs(
            in_bounds - pinned_arr[np.clip(idx - 1, 0, len(pinned_arr) - 1)]
        )
        right_dist = np.abs(
            in_bounds - pinned_arr[np.clip(idx, 0, len(pinned_arr) - 1)]
        )
        keep_mask = (left_dist > tol) & (right_dist > tol)
        filtered = in_bounds[keep_mask]
        merged = np.sort(np.concatenate([pinned_arr, filtered]))
    else:
        merged = pinned_arr

    return merged


class BaseTechnique(ABC):
    r"""Abstract base class for electrochemical experimental techniques.

    Encapsulates a time-dependent excitation protocol :math:`E(t)` (potentiostatic)
    or :math:`I(t)` (galvanostatic) over :math:`t \in [0, t_{\max}]`, together with
    exact interior discontinuity/vertex timestamps (:attr:`event_times`) so adaptive
    ODE integrators can accurately resolve step edges and switching vertices.

    Every ``BaseTechnique`` instance is directly callable (``technique(t)``),
    supporting both scalar floats (for solver boundary callables) and NumPy arrays
    (for vectorized post-processing and plotting).

    Attributes
    ----------
    mode : {'potentiostatic', 'galvanostatic'}
        Control mode of the excitation waveform (default ``'potentiostatic'``).
    duration : float
        Total experiment duration :math:`t_{\max} > 0` in seconds.
    event_times : tuple[float, ...]
        Sorted strictly interior discontinuity or switching timestamps in
        :math:`(0, t_{\max})`.
    """

    _mode: ControlMode = "potentiostatic"

    @property
    def mode(self) -> ControlMode:
        """Control mode: ``'potentiostatic'`` or ``'galvanostatic'``."""
        return self._mode

    @property
    @abstractmethod
    def duration(self) -> float:
        """Total experiment duration in seconds (:math:`t_{\\max} > 0`)."""
        raise NotImplementedError

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior discontinuity/vertex timestamps in ``(0, duration)``."""
        return ()

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        """Return technique-specific :class:`PlotSpec` entries (default plot first)."""
        return ()

    @property
    def plot_specs(self) -> dict[str, PlotSpec]:
        """Ordered mapping of canonical plot kind to :class:`PlotSpec`.

        Includes technique-specific plots first (with the primary/default plot
        listed first), followed by universal plots (``'i_vs_t'``, ``'E_vs_t'``,
        ``'c_vs_x'``) without duplicates.
        """
        specs: dict[str, PlotSpec] = {}
        for spec in self._technique_plots():
            specs.setdefault(spec.kind, spec)

        universal = _UNIVERSAL_PLOT_SPECS
        if self.mode == "galvanostatic":
            universal = (
                _UNIVERSAL_PLOT_SPECS[1],
                _UNIVERSAL_PLOT_SPECS[0],
                _UNIVERSAL_PLOT_SPECS[2],
            )
        for spec in universal:
            specs.setdefault(spec.kind, spec)
        return specs

    @property
    def available_plots(self) -> tuple[str, ...]:
        """Ordered tuple of canonical plot kinds supported by this technique."""
        return tuple(self.plot_specs.keys())

    @property
    def default_plot(self) -> str:
        """Canonical plot kind used by default when plotting results of this technique."""
        return self.available_plots[0]

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    @abstractmethod
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        """Evaluate the excitation signal at time ``t``.

        Parameters
        ----------
        t : float or array-like
            Time(s) in seconds at which to evaluate the waveform.

        Returns
        -------
        float or np.ndarray
            Excitation value(s) (potential in V for potentiostatic mode, or
            current in A for galvanostatic mode). Returns a Python ``float``
            for scalar inputs and a ``np.ndarray`` matching the input shape
            for array-like inputs.
        """
        raise NotImplementedError

    @overload
    def __call__(self, t: float) -> float: ...

    @overload
    def __call__(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def __call__(self, t: float | ArrayLike) -> float | np.ndarray:
        """Callable alias for :meth:`excitation`."""
        return self.excitation(t)

    @overload
    def potential(self, t: float) -> float: ...

    @overload
    def potential(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def potential(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def potential(self, t: float | ArrayLike) -> float | np.ndarray:
        """Evaluate applied electrode potential :math:`E(t)` in Volts.

        Only valid for ``mode='potentiostatic'`` techniques.

        Parameters
        ----------
        t : float or array-like
            Time(s) in seconds.

        Returns
        -------
        float or np.ndarray
            Applied potential :math:`E(t)` in Volts.

        Raises
        ------
        ValueError
            If the technique is not potentiostatic.
        """
        if self.mode != "potentiostatic":
            raise ValueError(
                "potential() is only available for potentiostatic techniques, "
                f"got mode={self.mode!r}."
            )
        return self.excitation(t)

    @overload
    def current(self, t: float) -> float: ...

    @overload
    def current(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def current(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def current(self, t: float | ArrayLike) -> float | np.ndarray:
        """Evaluate applied current :math:`I(t)` in Amperes.

        Only valid for ``mode='galvanostatic'`` techniques.

        Parameters
        ----------
        t : float or array-like
            Time(s) in seconds.

        Returns
        -------
        float or np.ndarray
            Applied current :math:`I(t)` in Amperes.

        Raises
        ------
        ValueError
            If the technique is not galvanostatic.
        """
        if self.mode != "galvanostatic":
            raise ValueError(
                "current() is only available for galvanostatic techniques, "
                f"got mode={self.mode!r}."
            )
        return self.excitation(t)

    def _default_time_grid_points(self) -> int:
        """Default number of grid points when ``n_points`` and ``dt`` are omitted."""
        return _DEFAULT_TIME_GRID_POINTS

    def _extra_grid_times(self) -> tuple[float, ...]:
        """Additional critical timestamps to guarantee in :meth:`time_grid`."""
        return self.event_times

    def time_grid(
        self,
        n_points: int | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Generate evaluation timestamps covering ``[0, duration]`` and all events.

        Parameters
        ----------
        n_points : int, optional
            Number of base uniformly spaced timestamps (must be ``>= 2``).
            Mutually exclusive with ``dt``.
        dt : float, optional
            Base uniform time step in seconds (must be ``> 0``).
            Mutually exclusive with ``n_points``.

        Returns
        -------
        np.ndarray
            Strictly increasing 1D array of timestamps starting at ``0.0``,
            ending at :attr:`duration`, and guaranteed to contain every
            timestamp in :attr:`event_times`.

        Raises
        ------
        ValueError
            If both ``n_points`` and ``dt`` are provided, or if values are invalid.
        """
        if n_points is not None and dt is not None:
            raise ValueError("Specify at most one of 'n_points' or 'dt'.")

        if n_points is not None:
            n = _validate_positive_int(n_points, "n_points", minimum=2)
            base = np.linspace(0.0, self.duration, n, dtype=float)
        elif dt is not None:
            dt_val = _validate_positive_finite(dt, "dt")
            n_steps = math.floor(self.duration / dt_val)
            base = np.arange(n_steps + 1, dtype=float) * dt_val
        else:
            n = self._default_time_grid_points()
            base = np.linspace(0.0, self.duration, n, dtype=float)

        return _merge_time_grid(base, self.duration, self._extra_grid_times())

    def simulate(
        self,
        mechanism: Mechanism | SpeciesCollection | Sequence[Species] | Species,
        geometry: Geometry | str = "planar",
        grid: np.ndarray | Mesh1D | None = None,
        solver: BaseSolver | str = "scipy_ivp",
        t_eval: np.ndarray | Sequence[float] | None = None,
        solver_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> TechniqueResult:
        """Simulate this technique with ``mechanism`` via :func:`~softpotato.techniques.simulate`.

        Convenience shorthand equivalent to
        ``simulate(self, mechanism, geometry=geometry, grid=grid, solver=solver, t_eval=t_eval, solver_kwargs=solver_kwargs, **kwargs)``.
        """
        from .pipeline import simulate as _simulate

        return _simulate(
            self,
            mechanism,
            geometry=geometry,
            grid=grid,
            solver=solver,
            t_eval=t_eval,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
