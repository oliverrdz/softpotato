"""Galvanostatic Chronopotentiometry (CP) current-step waveforms and helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.optimize import brentq

from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

from .base import (
    BaseTechnique,
    PlotSpec,
    _deduplicate_interior_events,
    _normalize_time_input,
    _validate_finite_real,
    _validate_non_negative_finite,
    _validate_positive_finite,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from softpotato.reactions.heterogeneous import HeterogeneousKinetics


class Chronopotentiometry(BaseTechnique):
    r"""Galvanostatic step-current waveform generator (Chronopotentiometry).

    Supports two construction modes, each optionally preceded by an initial
    resting/quiet current hold ``(I_quiet, t_quiet)``:

    1. **Single constant current step**:
       ``Chronopotentiometry(current=..., duration=...)``
    2. **Multi-step current sequence** (e.g., current-reversal chronopotentiometry):
       ``Chronopotentiometry(currents=[...], durations=[...])``

    Uses IUPAC polarity conventions: negative current (:math:`I < 0`) is cathodic
    (reduction) and positive current (:math:`I > 0`) is anodic (oxidation).

    At step transitions :math:`t = t_k`, the waveform is right-continuous:
    :math:`I(t) = I_k` for :math:`t \in [t_k, t_{k+1})`, and
    :math:`I(t_{\max}) = I_{\text{last}}`.

    Parameters
    ----------
    current : float, optional
        Constant applied current :math:`I` in Amperes. Must be paired with
        ``duration``.
    duration : float, optional
        Single-step duration in seconds (:math:`> 0`). Must be paired with
        ``current``.
    currents : Sequence[float], optional
        Ordered sequence of applied currents in Amperes for multi-step
        experiments (such as current-reversal chronopotentiometry).
    durations : Sequence[float], optional
        Ordered sequence of step durations in seconds (:math:`> 0`), matching
        the length of ``currents``.
    I_quiet : float, optional
        Optional initial quiet/resting current in Amperes applied before the
        first step when ``t_quiet > 0``.
    t_quiet : float, default 0.0
        Optional initial quiet/resting duration in seconds (:math:`\ge 0`).
        If ``t_quiet > 0``, ``I_quiet`` must be provided.

    Examples
    --------
    Constant-current cathodic chronopotentiometry:

    >>> from softpotato.techniques import Chronopotentiometry
    >>> cp = Chronopotentiometry(current=-1e-4, duration=5.0)
    >>> cp.mode
    'galvanostatic'
    >>> cp(2.5)
    -0.0001

    Current-reversal chronopotentiometry:

    >>> rcp = Chronopotentiometry(currents=[-1e-4, 1e-4], durations=[2.0, 3.0])
    >>> rcp.event_times
    (2.0,)
    >>> rcp(1.5), rcp(2.0)
    (-0.0001, 0.0001)
    """

    def __init__(
        self,
        current: float | None = None,
        duration: float | None = None,
        *,
        currents: Sequence[float] | None = None,
        durations: Sequence[float] | None = None,
        I_quiet: float | None = None,
        t_quiet: float = 0.0,
    ) -> None:
        self._mode = "galvanostatic"

        single_any = current is not None or duration is not None
        multi_any = currents is not None or durations is not None

        n_modes = int(single_any) + int(multi_any)
        if n_modes == 0:
            raise ValueError(
                "Must specify either single-step (current, duration) or "
                "multi-step (currents, durations)."
            )
        if n_modes > 1:
            raise ValueError("Cannot mix single-step and multi-step parameters.")

        step_currents: list[float] = []
        step_durations: list[float] = []

        t_quiet_val = _validate_non_negative_finite(t_quiet, "t_quiet")
        i_quiet_val: float | None = (
            _validate_finite_real(I_quiet, "I_quiet") if I_quiet is not None else None
        )
        if t_quiet_val > 0.0:
            if i_quiet_val is None:
                raise ValueError("I_quiet must be provided when t_quiet > 0.")
            step_currents.append(i_quiet_val)
            step_durations.append(t_quiet_val)

        if single_any:
            if current is None or duration is None:
                raise ValueError(
                    "Single-step mode requires both 'current' and 'duration'."
                )
            step_currents.append(_validate_finite_real(current, "current"))
            step_durations.append(_validate_positive_finite(duration, "duration"))
        else:
            if currents is None or durations is None:
                raise ValueError(
                    "Multi-step mode requires both 'currents' and 'durations'."
                )
            if isinstance(currents, (str, bytes)) or isinstance(
                durations, (str, bytes)
            ):
                raise TypeError("'currents' and 'durations' must be numeric sequences.")
            cur_seq = list(currents)
            dur_seq = list(durations)
            if len(cur_seq) == 0 or len(dur_seq) == 0:
                raise ValueError("'currents' and 'durations' must not be empty.")
            if len(cur_seq) != len(dur_seq):
                raise ValueError(
                    f"'currents' (len={len(cur_seq)}) and 'durations' "
                    f"(len={len(dur_seq)}) must have the same length."
                )
            for idx, (i_val, d_val) in enumerate(zip(cur_seq, dur_seq)):
                step_currents.append(_validate_finite_real(i_val, f"currents[{idx}]"))
                step_durations.append(
                    _validate_positive_finite(d_val, f"durations[{idx}]")
                )

        self._I_quiet = i_quiet_val
        self._t_quiet = t_quiet_val
        self._currents = tuple(step_currents)
        self._durations = tuple(step_durations)

        starts: list[float] = []
        cursor = 0.0
        for d in self._durations:
            starts.append(cursor)
            cursor += d

        self._starts = np.asarray(starts, dtype=float)
        self._values = np.asarray(self._currents, dtype=float)
        self._duration = float(cursor)
        self._event_times = _deduplicate_interior_events(starts[1:], self._duration)

    @property
    def currents(self) -> tuple[float, ...]:
        """Ordered tuple of step currents in Amperes (including quiet hold if active)."""
        return self._currents

    @property
    def durations(self) -> tuple[float, ...]:
        """Ordered tuple of step durations in seconds (including quiet hold if active)."""
        return self._durations

    @property
    def I_quiet(self) -> float | None:
        """Initial quiet/resting current in Amperes, or ``None``."""
        return self._I_quiet

    @property
    def t_quiet(self) -> float:
        """Initial quiet/resting duration in seconds."""
        return self._t_quiet

    @property
    def n_steps(self) -> int:
        """Total number of constant-current segments."""
        return len(self._currents)

    @property
    def step_times(self) -> tuple[float, ...]:
        """Start timestamps of each current step."""
        return tuple(float(x) for x in self._starts)

    @property
    def duration(self) -> float:
        """Total experiment duration in seconds."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior current step transition timestamps in ``(0, duration)``."""
        return self._event_times

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        n_steps = len(self._values)

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return float(self._values[0])
            if normalized >= self._duration:
                return float(self._values[-1])
            idx = int(np.searchsorted(self._starts, normalized, side="right")) - 1
            idx = min(max(idx, 0), n_steps - 1)
            return float(self._values[idx])

        assert isinstance(normalized, np.ndarray)
        orig_shape = normalized.shape
        flat = normalized.ravel()
        if flat.size == 0:
            return np.empty(orig_shape, dtype=float)

        clamped = np.clip(flat, 0.0, self._duration)
        indices = np.searchsorted(self._starts, clamped, side="right") - 1
        indices = np.clip(indices, 0, n_steps - 1)
        indices = np.where(clamped >= self._duration, n_steps - 1, indices)
        return self._values[indices].reshape(orig_shape)

    @overload
    def interfacial_flux(
        self,
        t: float,
        n: float,
        area: float,
        *,
        F: float = FARADAY,
    ) -> float: ...

    @overload
    def interfacial_flux(
        self,
        t: np.ndarray,
        n: float,
        area: float,
        *,
        F: float = FARADAY,
    ) -> np.ndarray: ...

    @overload
    def interfacial_flux(
        self,
        t: float | ArrayLike,
        n: float,
        area: float,
        *,
        F: float = FARADAY,
    ) -> float | np.ndarray: ...

    def interfacial_flux(
        self,
        t: float | ArrayLike,
        n: float,
        area: float,
        *,
        F: float = FARADAY,
    ) -> float | np.ndarray:
        r"""Compute interfacial molar flux :math:`J(t) = \frac{I(t)}{n F A}`.

        Follows IUPAC polarity conventions: cathodic reduction currents
        (:math:`I < 0`) yield negative flux (:math:`J < 0`, species consumed at
        the left electrode boundary), and anodic oxidation currents (:math:`I > 0`)
        yield positive flux (:math:`J > 0`).

        For direct Neumann boundary condition assignment at the left boundary
        (:math:`x = x_0`) where :math:`J_{\text{left}} = -D \frac{\partial c}{\partial x}`:

        - Oxidized species (consumed on reduction):
          :math:`\left.\frac{\partial c_{\text{Ox}}}{\partial x}\right|_{x_0} = -\frac{J(t)}{D_{\text{Ox}}}`
        - Reduced species (generated on reduction):
          :math:`\left.\frac{\partial c_{\text{Red}}}{\partial x}\right|_{x_0} = +\frac{J(t)}{D_{\text{Red}}}`

        Parameters
        ----------
        t : float or array-like
            Time(s) in seconds.
        n : float
            Number of electrons transferred per reaction (:math:`n > 0`).
        area : float
            Active electrode surface area :math:`A > 0` in :math:`\text{m}^2`
            or :math:`\text{cm}^2`.
        F : float, default FARADAY
            Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

        Returns
        -------
        float or np.ndarray
            Interfacial molar flux :math:`J(t)` in :math:`\text{mol}/(\text{m}^2\cdot\text{s})`
            or :math:`\text{mol}/(\text{cm}^2\cdot\text{s})`.
        """
        n_val = _validate_positive_finite(n, "n")
        area_val = _validate_positive_finite(area, "area")
        f_val = _validate_positive_finite(F, "F")
        denom = n_val * f_val * area_val
        i_val = self.excitation(t)
        if isinstance(i_val, float):
            return i_val / denom
        return i_val / denom

    def recover_potential(
        self,
        c_ox: float | ArrayLike,
        c_red: float | ArrayLike,
        *,
        E0: float = 0.0,
        n: float = 1.0,
        T: float = STANDARD_TEMPERATURE,
        R: float = GAS_CONSTANT,
        F: float = FARADAY,
        kinetics: HeterogeneousKinetics | None = None,
        t: float | ArrayLike | None = None,
        current: float | ArrayLike | None = None,
        area: float | None = None,
    ) -> float | np.ndarray:
        r"""Recover electrode surface potential :math:`E(t)` from surface concentrations.

        - **Reversible (Nernstian) recovery** (``kinetics=None``):

          .. math::

              E(t) = E^{0\prime} + \frac{R T}{n F} \ln\left(\frac{c_{\text{Ox}}(0, t)}{c_{\text{Red}}(0, t)}\right)

        - **Finite-rate (Butler–Volmer / heterogeneous kinetics) recovery**
          (``kinetics`` provided): solves
          :math:`j_{\text{kinetics}}(E, c_{\text{Ox}}, c_{\text{Red}}) = I(t) / A`
          for :math:`E(t)` via safeguarded root-finding (:func:`scipy.optimize.brentq`).

        Parameters
        ----------
        c_ox : float or array-like
            Surface concentration of oxidized species :math:`c_{\text{Ox}}(0, t)`.
        c_red : float or array-like
            Surface concentration of reduced species :math:`c_{\text{Red}}(0, t)`.
        E0 : float, default 0.0
            Formal potential :math:`E^{0\prime}` in Volts (used when ``kinetics=None``).
        n : float, default 1.0
            Number of electrons transferred (:math:`n > 0`, used when ``kinetics=None``).
        T : float, default STANDARD_TEMPERATURE
            Temperature in Kelvin (:math:`T > 0`, used when ``kinetics=None``).
        R : float, default GAS_CONSTANT
            Molar gas constant (:math:`R > 0`, used when ``kinetics=None``).
        F : float, default FARADAY
            Faraday constant (:math:`F > 0`, used when ``kinetics=None``).
        kinetics : HeterogeneousKinetics, optional
            Optional heterogeneous kinetics model (e.g., :class:`ButlerVolmer`)
            for Butler–Volmer potential inversion. Requires ``area`` and either
            ``t`` or ``current``.
        t : float or array-like, optional
            Time(s) in seconds used to evaluate ``self.current(t)`` when
            ``kinetics`` is provided and ``current`` is omitted.
        current : float or array-like, optional
            Explicit applied current(s) in Amperes when ``kinetics`` is provided.
            Mutually exclusive with ``t``.
        area : float, optional
            Electrode surface area :math:`A > 0` (required when ``kinetics`` is
            provided).

        Returns
        -------
        float or np.ndarray
            Recovered electrode potential :math:`E(t)` in Volts. Returns ``np.nan``
            at points where inversion is undefined (e.g. depleted concentrations
            with non-zero opposing flux).
        """
        is_scalar = np.isscalar(c_ox) and np.isscalar(c_red)
        ox_arr, red_arr = np.broadcast_arrays(
            np.asarray(c_ox, dtype=float),
            np.asarray(c_red, dtype=float),
        )

        if kinetics is None:
            e0_val = _validate_finite_real(E0, "E0")
            n_val = _validate_positive_finite(n, "n")
            t_val = _validate_positive_finite(T, "T")
            r_val = _validate_positive_finite(R, "R")
            f_val = _validate_positive_finite(F, "F")
            vt = (r_val * t_val) / (n_val * f_val)

            with np.errstate(divide="ignore", invalid="ignore"):
                pot = np.where(
                    (ox_arr > 0.0) & (red_arr > 0.0),
                    e0_val + vt * np.log(ox_arr / red_arr),
                    np.where(
                        (ox_arr <= 0.0) & (red_arr > 0.0),
                        -np.inf,
                        np.where((ox_arr > 0.0) & (red_arr <= 0.0), np.inf, np.nan),
                    ),
                )
            if is_scalar:
                return float(pot.item())
            return np.asarray(pot, dtype=float)

        # Butler-Volmer / HeterogeneousKinetics inversion
        if area is None:
            raise ValueError("'area' must be provided when 'kinetics' is specified.")
        area_val = _validate_positive_finite(area, "area")

        if t is not None and current is not None:
            raise ValueError("Specify at most one of 't' or 'current'.")
        if current is not None:
            i_raw = np.asarray(current, dtype=float)
            is_scalar = is_scalar and np.isscalar(current)
        elif t is not None:
            i_raw = np.asarray(self.excitation(t), dtype=float)
            is_scalar = is_scalar and np.isscalar(t)
        elif self.n_steps == 1:
            i_raw = np.asarray(self._currents[0], dtype=float)
        else:
            raise ValueError(
                "Multi-step Chronopotentiometry requires 't' or 'current' when "
                "'kinetics' is provided."
            )

        ox_b, red_b, i_b = np.broadcast_arrays(ox_arr, red_arr, i_raw)
        j_target = i_b / area_val

        out = np.full(ox_b.shape, np.nan, dtype=float)
        flat_ox = ox_b.ravel()
        flat_red = red_b.ravel()
        flat_j = j_target.ravel()
        flat_out = out.ravel()

        e0_ref = float(kinetics.E0)
        for idx in range(flat_ox.size):
            co = float(flat_ox[idx])
            cr = float(flat_red[idx])
            jt = float(flat_j[idx])
            if not (math.isfinite(co) and math.isfinite(cr) and math.isfinite(jt)):
                continue
            if co < 0.0 or cr < 0.0:
                continue
            if co == 0.0 and cr == 0.0:
                continue
            if jt < 0.0 and co == 0.0:
                continue
            if jt > 0.0 and cr == 0.0:
                continue

            def _residual(
                E_val: float,
                _co: float = co,
                _cr: float = cr,
                _jt: float = jt,
            ) -> float:
                return float(kinetics.current_density(E_val, _co, _cr)) - _jt

            # Bracket around E0 (current_density is monotonically increasing in E)
            lo = e0_ref - 0.5
            hi = e0_ref + 0.5
            f_lo = _residual(lo)
            f_hi = _residual(hi)
            width = 0.5
            for _ in range(12):
                if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                    break
                width *= 2.0
                if not math.isfinite(f_lo) or f_lo > 0.0:
                    lo = e0_ref - width
                    f_lo = _residual(lo)
                if not math.isfinite(f_hi) or f_hi < 0.0:
                    hi = e0_ref + width
                    f_hi = _residual(hi)

            if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                flat_out[idx] = float(brentq(_residual, lo, hi, xtol=1e-12, rtol=1e-12))

        if is_scalar:
            return float(out.item())
        return out

    @staticmethod
    def detect_transition_time(
        t: ArrayLike,
        c_surface: ArrayLike,
        *,
        threshold: float = 1e-3,
        relative: bool = True,
    ) -> float | None:
        r"""Detect the chronopotentiometric transition time :math:`\tau`.

        Finds the earliest timestamp :math:`\tau` at which the electroactive
        reactant surface concentration :math:`c(0, t)` depletes to or below the
        specified threshold, using linear interpolation between adjacent samples.

        Parameters
        ----------
        t : array-like
            Strictly increasing 1D time array in seconds.
        c_surface : array-like
            1D surface concentration array :math:`c(0, t)` matching ``t``.
        threshold : float, default 1e-3
            Depletion threshold (:math:`\ge 0`). When ``relative=True``, interpreted
            as a fraction of the initial surface concentration ``c_surface[0]``
            (i.e. ``c_thresh = threshold * c_surface[0]``); otherwise an absolute
            concentration threshold.
        relative : bool, default True
            Whether ``threshold`` is relative to ``c_surface[0]``.

        Returns
        -------
        float or None
            Interpolated transition time :math:`\tau` in seconds, or ``None`` if
            surface concentration never drops to or below the threshold.
        """
        t_arr = np.asarray(t, dtype=float)
        c_arr = np.asarray(c_surface, dtype=float)
        if t_arr.ndim != 1 or c_arr.ndim != 1:
            raise ValueError("'t' and 'c_surface' must be 1D arrays.")
        if t_arr.size < 2 or t_arr.shape != c_arr.shape:
            raise ValueError("'t' and 'c_surface' must have the same length (>= 2).")
        if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(c_arr)):
            raise ValueError("All values in 't' and 'c_surface' must be finite.")
        if np.any(np.diff(t_arr) <= 0.0):
            raise ValueError("'t' must be strictly increasing.")

        thresh_val = _validate_non_negative_finite(threshold, "threshold")
        if relative:
            if c_arr[0] <= 0.0:
                raise ValueError(
                    "Initial surface concentration c_surface[0] must be positive "
                    "when relative=True."
                )
            cutoff = thresh_val * float(c_arr[0])
        else:
            cutoff = thresh_val

        below = np.flatnonzero(c_arr <= cutoff)
        if below.size == 0:
            return None

        idx = int(below[0])
        if idx == 0:
            return float(t_arr[0])

        c0 = float(c_arr[idx - 1])
        c1 = float(c_arr[idx])
        t0 = float(t_arr[idx - 1])
        t1 = float(t_arr[idx])
        if c0 == c1:
            return t1
        frac = (c0 - cutoff) / (c0 - c1)
        return float(t0 + frac * (t1 - t0))

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="E_vs_t",
                title="Chronopotentiogram",
                x_label="Time, $t$ / s",
                y_label="Potential, $E$ / V",
                description="Electrode potential response E(t) under controlled current.",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Chronopotentiometry(currents={self._currents!r}, "
            f"durations={self._durations!r})"
        )
