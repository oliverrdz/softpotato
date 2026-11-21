"""Pulse and sampled-current voltammetry waveforms (SCV, DPV, SWV)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, overload

import numpy as np

from .base import (
    BaseTechnique,
    PlotSpec,
    _deduplicate_interior_events,
    _normalize_time_input,
    _validate_finite_real,
    _validate_non_negative_finite,
    _validate_positive_finite,
    _validate_positive_int,
)
from .chronoamperometry import Chronoamperometry

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _validate_sample_fraction(value: object) -> float:
    """Validate that sample_fraction is a finite real number in (0, 1]."""
    frac = _validate_positive_finite(value, "sample_fraction")
    if frac > 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {value!r}.")
    return frac


def _build_step_potentials(
    E_start: float,
    E_end: float,
    step_height: float,
) -> tuple[tuple[float, ...], int]:
    """Generate staircase potentials from ``E_start`` toward ``E_end`` with ``step_height > 0``.

    Returns
    -------
    tuple[tuple[float, ...], int]
        ``(potentials, direction)`` where ``direction`` is ``+1`` or ``-1``.
    """
    direction = 1 if E_end > E_start else -1
    span = abs(E_end - E_start)
    n_steps = math.floor(span / step_height + 1e-12) + 1
    potentials = tuple(
        float(E_start + direction * k * step_height) for k in range(n_steps)
    )
    return potentials, direction


def _sample_window_mean(
    t: np.ndarray,
    i: np.ndarray,
    t_win_start: float,
    t_win_end: float,
    *,
    include_end: bool = False,
) -> float:
    """Average current ``i`` over ``[t_win_start, t_win_end)`` (or ``<=`` if ``include_end``).

    Falls back to linear interpolation at the window midpoint when no discrete
    timestamps lie inside the sampling window.
    """
    if include_end:
        mask = (t >= t_win_start) & (t <= t_win_end)
    else:
        mask = (t >= t_win_start) & (t < t_win_end)
    if np.any(mask):
        return float(np.mean(i[mask]))
    t_mid = 0.5 * (t_win_start + t_win_end)
    return float(np.interp(t_mid, t, i))


def _validate_time_current_pair(
    t: ArrayLike,
    i: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 1D matching time and current arrays for post-processing sampling."""
    t_arr = np.asarray(t, dtype=float)
    i_arr = np.asarray(i, dtype=float)
    if t_arr.ndim != 1 or i_arr.ndim != 1:
        raise ValueError(
            f"'t' and 'i' must be 1D arrays, got shapes {t_arr.shape} and {i_arr.shape}."
        )
    if t_arr.size < 2:
        raise ValueError("'t' must contain at least 2 timestamps.")
    if t_arr.shape != i_arr.shape:
        raise ValueError(
            f"'t' (len={t_arr.size}) and 'i' (len={i_arr.size}) must have the same length."
        )
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(i_arr)):
        raise ValueError("All values in 't' and 'i' must be finite.")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("'t' must be strictly increasing.")
    return t_arr, i_arr


class SampledCurrentVoltammetry(BaseTechnique):
    r"""Sampled-Current Voltammetry (SCV) potential-step family generator.

    Applies a series of independent constant-potential steps across a potential
    window, where every step shares the exact same total duration :math:`t_{\max}`
    and evaluation time grid. Sampling the resulting current transients
    :math:`I_k(t)` at one or more fixed sampling times :math:`\tau \in (0, t_{\max}]`
    yields the sampled-current voltammogram :math:`I(\tau)` vs. :math:`E_k`.

    Supports two mutually exclusive ways to specify the step potentials:

    1. **Explicit potentials**: ``SampledCurrentVoltammetry(potentials=[...], duration=...)``
       accepting a Python list, tuple, or 1D NumPy array of potentials.
    2. **Potential window + step**:
       ``SampledCurrentVoltammetry(E_start=..., E_end=..., dE=..., duration=...)``
       (or ``step_height=...`` as an alias for ``dE``).

    Parameters
    ----------
    potentials : Sequence[float] or np.ndarray, optional
        Explicit 1D sequence or NumPy array of step potentials in Volts.
        Mutually exclusive with ``(E_start, E_end, dE / step_height)``.
    duration : float, optional
        Constant-potential step duration in seconds (:math:`> 0`) applied
        uniformly to every potential step. Required.
    E_start : float, optional
        Initial potential of the potential window in Volts.
    E_end : float, optional
        Final potential of the potential window in Volts (must differ from
        ``E_start``).
    dE : float, optional
        Potential step increment :math:`\Delta E > 0` in Volts used with
        ``E_start`` and ``E_end``. Mutually exclusive with ``step_height``.
    step_height : float, optional
        Alias for ``dE`` (:math:`\Delta E > 0` in Volts). Mutually exclusive
        with ``dE``.
    E_quiet : float, optional
        Optional initial quiet/resting potential in Volts applied before every
        step when ``t_quiet > 0``.
    t_quiet : float, default 0.0
        Optional initial quiet/resting duration in seconds (:math:`\ge 0`)
        applied identically to every step. If ``t_quiet > 0``, ``E_quiet`` must
        be provided.
    sample_times : float or Sequence[float] or np.ndarray, optional
        Optional default sampling time(s) :math:`\tau \in (0, t_{\max}]` in
        seconds for :meth:`sample_currents`. Any configured ``sample_times`` are
        guaranteed to appear in :meth:`time_grid`.
    n_points : int, optional
        Optional default number of points (:math:`\ge 2`) for :meth:`time_grid`.

    Examples
    --------
    From an explicit list or NumPy array:

    >>> import numpy as np
    >>> from softpotato.techniques import SampledCurrentVoltammetry
    >>> scv = SampledCurrentVoltammetry(potentials=[-0.4, -0.2, 0.0, 0.2], duration=5.0)
    >>> scv.n_steps
    4
    >>> scv.duration
    5.0

    From a potential window and step increment:

    >>> scv_win = SampledCurrentVoltammetry(
    ...     E_start=0.2, E_end=-0.2, dE=0.1, duration=2.0
    ... )
    >>> len(scv_win)
    5
    >>> tuple(round(e, 4) for e in scv_win.potentials)
    (0.2, 0.1, 0.0, -0.1, -0.2)
    """

    def __init__(
        self,
        potentials: Sequence[float] | np.ndarray | None = None,
        duration: float | None = None,
        *,
        E_start: float | None = None,
        E_end: float | None = None,
        dE: float | None = None,
        step_height: float | None = None,
        E_quiet: float | None = None,
        t_quiet: float = 0.0,
        sample_times: float | Sequence[float] | np.ndarray | None = None,
        n_points: int | None = None,
    ) -> None:
        self._mode = "potentiostatic"

        if duration is None:
            raise ValueError("'duration' must be provided and strictly positive.")
        step_dur = _validate_positive_finite(duration, "duration")

        if dE is not None and step_height is not None:
            raise ValueError("Specify at most one of 'dE' or 'step_height'.")
        resolved_de = dE if dE is not None else step_height

        explicit_mode = potentials is not None
        window_mode = any(x is not None for x in (E_start, E_end, resolved_de))

        if not explicit_mode and not window_mode:
            raise ValueError(
                "Must specify either 'potentials' or ('E_start', 'E_end', 'dE'/'step_height')."
            )
        if explicit_mode and window_mode:
            raise ValueError(
                "Cannot mix 'potentials' with ('E_start', 'E_end', 'dE'/'step_height')."
            )

        if explicit_mode:
            if isinstance(potentials, (str, bytes)):
                raise TypeError("'potentials' must be a numeric sequence or 1D array.")
            if isinstance(potentials, np.ndarray):
                if potentials.ndim != 1:
                    raise ValueError(
                        f"'potentials' array must be 1D, got shape {potentials.shape}."
                    )
                raw_pots = potentials.tolist()
            elif isinstance(potentials, Sequence):
                raw_pots = list(potentials)
            else:
                raise TypeError(
                    "'potentials' must be a numeric sequence or 1D numpy.ndarray, "
                    f"got {type(potentials).__name__}."
                )
            if len(raw_pots) == 0:
                raise ValueError("'potentials' must not be empty.")
            resolved_potentials = tuple(
                _validate_finite_real(v, f"potentials[{idx}]")
                for idx, v in enumerate(raw_pots)
            )
            self._E_start = resolved_potentials[0]
            self._E_end = resolved_potentials[-1]
            self._dE: float | None = None
        else:
            if E_start is None or E_end is None or resolved_de is None:
                raise ValueError(
                    "Window mode requires all of 'E_start', 'E_end', and 'dE' (or 'step_height')."
                )
            e_start_val = _validate_finite_real(E_start, "E_start")
            e_end_val = _validate_finite_real(E_end, "E_end")
            if e_start_val == e_end_val:
                raise ValueError("E_start and E_end must be distinct.")
            de_val = _validate_positive_finite(
                resolved_de, "dE" if dE is not None else "step_height"
            )
            resolved_potentials, _ = _build_step_potentials(
                e_start_val, e_end_val, de_val
            )
            self._E_start = e_start_val
            self._E_end = e_end_val
            self._dE = de_val

        t_quiet_val = _validate_non_negative_finite(t_quiet, "t_quiet")
        e_quiet_val: float | None = (
            _validate_finite_real(E_quiet, "E_quiet") if E_quiet is not None else None
        )
        if t_quiet_val > 0.0 and e_quiet_val is None:
            raise ValueError("E_quiet must be provided when t_quiet > 0.")

        self._default_n_points = (
            _validate_positive_int(n_points, "n_points", minimum=2)
            if n_points is not None
            else 1001
        )

        self._potentials: tuple[float, ...] = resolved_potentials
        self._potentials_arr = np.asarray(resolved_potentials, dtype=float)
        self._step_duration = step_dur
        self._E_quiet = e_quiet_val
        self._t_quiet = t_quiet_val
        self._duration = float(t_quiet_val + step_dur)
        self._event_times = (
            _deduplicate_interior_events([t_quiet_val], self._duration)
            if t_quiet_val > 0.0
            else ()
        )

        # Validate optional default sample_times against [0, duration]
        if sample_times is not None:
            self._sample_times: tuple[float, ...] | None = self._normalize_sample_times(
                sample_times
            )[0]
        else:
            self._sample_times = None

        # Construct uniform Chronoamperometry objects for every potential step
        self._steps: tuple[Chronoamperometry, ...] = tuple(
            Chronoamperometry(
                E_step=e_val,
                duration=step_dur,
                E_quiet=e_quiet_val,
                t_quiet=t_quiet_val,
            )
            for e_val in self._potentials
        )

    def _normalize_sample_times(
        self,
        sample_times: float | Sequence[float] | np.ndarray,
    ) -> tuple[tuple[float, ...], bool]:
        """Validate scalar or 1D sequence of sampling times in ``(0, duration]``."""
        if np.isscalar(sample_times):
            tau = _validate_positive_finite(sample_times, "sample_times")
            if tau > self._duration:
                raise ValueError(
                    f"sample_times ({tau}) must be <= duration ({self._duration})."
                )
            return (tau,), True

        if isinstance(sample_times, (str, bytes)):
            raise TypeError("'sample_times' must be a real number or numeric sequence.")
        if isinstance(sample_times, np.ndarray):
            if sample_times.ndim != 1:
                raise ValueError(
                    f"'sample_times' array must be 1D, got shape {sample_times.shape}."
                )
            raw = sample_times.tolist()
        elif isinstance(sample_times, Sequence):
            raw = list(sample_times)
        else:
            raise TypeError(
                "'sample_times' must be a real number or numeric sequence, "
                f"got {type(sample_times).__name__}."
            )
        if len(raw) == 0:
            raise ValueError("'sample_times' must not be empty.")
        cleaned: list[float] = []
        for idx, val in enumerate(raw):
            tau = _validate_positive_finite(val, f"sample_times[{idx}]")
            if tau > self._duration:
                raise ValueError(
                    f"sample_times[{idx}] ({tau}) must be <= duration ({self._duration})."
                )
            cleaned.append(tau)
        return tuple(cleaned), False

    @property
    def potentials(self) -> tuple[float, ...]:
        """Ordered tuple of constant step potentials in Volts."""
        return self._potentials

    @property
    def n_steps(self) -> int:
        """Number of constant-potential steps in the potential window."""
        return len(self._potentials)

    @property
    def step_duration(self) -> float:
        """Duration of the constant-potential hold in seconds (same for all steps)."""
        return self._step_duration

    @property
    def duration(self) -> float:
        """Total duration ``t_quiet + step_duration`` in seconds (same for all steps)."""
        return self._duration

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="scv",
                title="Sampled-Current Voltammogram",
                x_label="$E$ / V",
                y_label=r"$I(\tau)$ / A",
                description="Current sampled at fixed time(s) tau vs. step potential.",
            ),
        )

    @property
    def E_start(self) -> float:
        """Initial potential of the window in Volts."""
        return self._E_start

    @property
    def E_end(self) -> float:
        """Final potential of the window in Volts."""
        return self._E_end

    @property
    def dE(self) -> float | None:
        """Potential step increment in Volts if constructed in window mode, else ``None``."""
        return self._dE

    @property
    def E_quiet(self) -> float | None:
        """Optional quiet/resting potential in Volts, or ``None``."""
        return self._E_quiet

    @property
    def t_quiet(self) -> float:
        """Optional quiet/resting duration in seconds."""
        return self._t_quiet

    @property
    def sample_times(self) -> tuple[float, ...] | None:
        """Default sampling times in seconds, or ``None``."""
        return self._sample_times

    @property
    def steps(self) -> tuple[Chronoamperometry, ...]:
        """Tuple of :class:`Chronoamperometry` objects, one per potential step."""
        return self._steps

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior transition timestamps shared by all steps."""
        return self._event_times

    def __len__(self) -> int:
        return len(self._steps)

    @overload
    def __getitem__(self, index: int) -> Chronoamperometry: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Chronoamperometry, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Chronoamperometry | tuple[Chronoamperometry, ...]:
        return self._steps[index]

    def __iter__(self) -> Iterator[Chronoamperometry]:
        return iter(self._steps)

    def _default_time_grid_points(self) -> int:
        return self._default_n_points

    def _extra_grid_times(self) -> tuple[float, ...]:
        if self._sample_times is None:
            return self._event_times
        return self._event_times + self._sample_times

    def excitation(self, t: float | ArrayLike) -> np.ndarray:  # type: ignore[override]
        """Evaluate applied potentials across all steps at time(s) ``t``.

        Parameters
        ----------
        t : float or array-like
            Time(s) in seconds.

        Returns
        -------
        np.ndarray
            For scalar ``t``, a 1D array of shape ``(n_steps,)``.
            For array-like ``t`` of shape ``S``, an array of shape
            ``(n_steps, *S)`` where row ``k`` is step ``k``'s potential waveform.
        """
        normalized, is_scalar = _normalize_time_input(t)
        if is_scalar:
            assert isinstance(normalized, float)
            if self._t_quiet > 0.0 and normalized < self._t_quiet:
                assert self._E_quiet is not None
                return np.full(self.n_steps, self._E_quiet, dtype=float)
            return self._potentials_arr.copy()

        assert isinstance(normalized, np.ndarray)
        out_shape = (self.n_steps, *normalized.shape)
        if normalized.size == 0:
            return np.empty(out_shape, dtype=float)

        # Broadcast step potentials across the time shape
        pot_bc = self._potentials_arr.reshape((self.n_steps,) + (1,) * normalized.ndim)
        out = np.broadcast_to(pot_bc, out_shape).astype(float, copy=True)
        if self._t_quiet > 0.0:
            assert self._E_quiet is not None
            quiet_mask = normalized < self._t_quiet
            if np.any(quiet_mask):
                out[:, quiet_mask] = self._E_quiet
        return out

    def sample_currents(
        self,
        t: ArrayLike,
        currents: ArrayLike,
        sample_times: float | Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract sampled-current voltammogram(s) from step current transients.

        Parameters
        ----------
        t : array-like
            Shared strictly increasing 1D time array of shape ``(n_times,)`` in seconds.
        currents : array-like
            2D array (or sequence of 1D arrays) of shape ``(n_steps, n_times)``
            containing the current transient :math:`I_k(t)` for each potential step.
        sample_times : float or Sequence[float] or np.ndarray, optional
            Sampling time(s) :math:`\\tau \\in (0, t_{\\max}]` in seconds.
            Defaults to :attr:`sample_times` if configured, otherwise ``self.duration``.

        Returns
        -------
        potentials : np.ndarray
            1D array of step potentials of shape ``(n_steps,)``.
        sampled_currents : np.ndarray
            1D array of shape ``(n_steps,)`` when a single scalar sampling time
            is requested (or defaulted to ``duration``), or 2D array of shape
            ``(n_sample_times, n_steps)`` when a sequence of sampling times is given.
        """
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim != 1 or t_arr.size < 2:
            raise ValueError("'t' must be a 1D array with at least 2 timestamps.")
        if not np.all(np.isfinite(t_arr)) or np.any(np.diff(t_arr) <= 0.0):
            raise ValueError("'t' must be finite and strictly increasing.")

        i_arr = np.asarray(currents, dtype=float)
        if i_arr.ndim != 2 or i_arr.shape != (self.n_steps, t_arr.size):
            raise ValueError(
                f"'currents' must have shape ({self.n_steps}, {t_arr.size}), "
                f"got {i_arr.shape}."
            )
        if not np.all(np.isfinite(i_arr)):
            raise ValueError("All values in 'currents' must be finite.")

        if sample_times is None:
            if self._sample_times is not None:
                taus = self._sample_times
                scalar_tau = False
            else:
                taus = (self._duration,)
                scalar_tau = True
        else:
            taus, scalar_tau = self._normalize_sample_times(sample_times)

        tau_arr = np.asarray(taus, dtype=float)
        sampled = np.empty((tau_arr.size, self.n_steps), dtype=float)
        for k in range(self.n_steps):
            sampled[:, k] = np.interp(tau_arr, t_arr, i_arr[k])

        if scalar_tau:
            return self._potentials_arr.copy(), sampled[0]
        return self._potentials_arr.copy(), sampled

    def __repr__(self) -> str:
        return (
            f"SampledCurrentVoltammetry(n_steps={self.n_steps}, "
            f"E_start={self._E_start!r}, E_end={self._E_end!r}, "
            f"duration={self._duration!r})"
        )


class DifferentialPulseVoltammetry(BaseTechnique):
    r"""Differential Pulse Voltammetry (DPV) waveform generator.

    Superimposes constant-amplitude potential pulses :math:`\Delta E_p` of width
    :math:`t_p` onto a linear staircase ramp from :math:`E_{\text{start}}` to
    :math:`E_{\text{end}}` with step increment :math:`\Delta E_s > 0` and cycle
    period :math:`\tau > t_p`.

    Within each cycle :math:`k \in \{0, \dots, N-1\}` over
    :math:`[k\tau, (k+1)\tau)`:

    - **Base hold** (:math:`t \in [k\tau, k\tau + \tau - t_p)`):
      :math:`E(t) = E_{\text{base}, k}`
    - **Pulse hold** (:math:`t \in [k\tau + \tau - t_p, (k+1)\tau)`):
      :math:`E(t) = E_{\text{base}, k} + \Delta E_p`

    Currents are sampled near the end of each base hold (:math:`I_{\text{pre}}`)
    and near the end of each pulse hold (:math:`I_{\text{pulse}}`) via
    :meth:`sample_currents`, yielding :math:`\Delta I = I_{\text{pulse}} - I_{\text{pre}}`.

    Parameters
    ----------
    E_start : float
        Initial staircase base potential in Volts.
    E_end : float
        Final staircase base potential in Volts (must differ from ``E_start``).
    step_height : float
        Staircase potential step magnitude :math:`\Delta E_s > 0` in Volts.
    pulse_amplitude : float
        Pulse potential amplitude :math:`\Delta E_p \neq 0` in Volts (positive
        for anodic pulses, negative for cathodic pulses).
    pulse_width : float
        Pulse duration :math:`t_p > 0` in seconds.
    pulse_period : float
        Total cycle period :math:`\tau > t_p` in seconds.
    sample_fraction : float, default 0.1
        Fraction :math:`\in (0, 1]` of the base/pulse segment duration near the
        end of each segment over which current is averaged in :meth:`sample_currents`.

    Examples
    --------
    >>> from softpotato.techniques import DifferentialPulseVoltammetry
    >>> dpv = DifferentialPulseVoltammetry(
    ...     E_start=0.2,
    ...     E_end=-0.2,
    ...     step_height=0.05,
    ...     pulse_amplitude=-0.025,
    ...     pulse_width=0.05,
    ...     pulse_period=0.5,
    ... )
    >>> dpv.n_steps
    9
    >>> dpv.duration
    4.5
    """

    def __init__(
        self,
        E_start: float,
        E_end: float,
        step_height: float,
        pulse_amplitude: float,
        pulse_width: float,
        pulse_period: float,
        *,
        sample_fraction: float = 0.1,
    ) -> None:
        self._mode = "potentiostatic"
        self._E_start = _validate_finite_real(E_start, "E_start")
        self._E_end = _validate_finite_real(E_end, "E_end")
        if self._E_start == self._E_end:
            raise ValueError("E_start and E_end must be distinct.")

        self._step_height = _validate_positive_finite(step_height, "step_height")
        self._pulse_amplitude = _validate_finite_real(
            pulse_amplitude, "pulse_amplitude"
        )
        if self._pulse_amplitude == 0.0:
            raise ValueError("pulse_amplitude must be non-zero.")

        self._pulse_width = _validate_positive_finite(pulse_width, "pulse_width")
        self._pulse_period = _validate_positive_finite(pulse_period, "pulse_period")
        if self._pulse_width >= self._pulse_period:
            raise ValueError(
                "pulse_period must be strictly greater than pulse_width "
                f"(got pulse_width={self._pulse_width!r}, pulse_period={self._pulse_period!r})."
            )
        self._sample_fraction = _validate_sample_fraction(sample_fraction)

        self._base_width = float(self._pulse_period - self._pulse_width)
        self._base_potentials, self._direction = _build_step_potentials(
            self._E_start, self._E_end, self._step_height
        )
        self._pulse_potentials = tuple(
            float(e + self._pulse_amplitude) for e in self._base_potentials
        )
        self._n_steps = len(self._base_potentials)
        self._duration = float(self._n_steps * self._pulse_period)

        # Build flat arrays for O(log N) right-continuous evaluation
        seg_starts: list[float] = []
        seg_values: list[float] = []
        for k in range(self._n_steps):
            t0 = k * self._pulse_period
            t_rise = (k + 1) * self._pulse_period - self._pulse_width
            seg_starts.append(t0)
            seg_values.append(self._base_potentials[k])
            seg_starts.append(t_rise)
            seg_values.append(self._pulse_potentials[k])

        self._starts = np.asarray(seg_starts, dtype=float)
        self._values = np.asarray(seg_values, dtype=float)
        self._event_times = _deduplicate_interior_events(seg_starts[1:], self._duration)

    @property
    def E_start(self) -> float:
        """Initial staircase base potential in Volts."""
        return self._E_start

    @property
    def E_end(self) -> float:
        """Final staircase base potential in Volts."""
        return self._E_end

    @property
    def step_height(self) -> float:
        """Staircase potential step magnitude :math:`\\Delta E_s` in Volts."""
        return self._step_height

    @property
    def pulse_amplitude(self) -> float:
        """Pulse potential amplitude :math:`\\Delta E_p` in Volts."""
        return self._pulse_amplitude

    @property
    def pulse_width(self) -> float:
        """Pulse duration :math:`t_p` in seconds."""
        return self._pulse_width

    @property
    def pulse_period(self) -> float:
        """Total cycle period :math:`\\tau` in seconds."""
        return self._pulse_period

    @property
    def base_width(self) -> float:
        """Pre-pulse base hold duration :math:`\\tau - t_p` in seconds."""
        return self._base_width

    @property
    def sample_fraction(self) -> float:
        """Sampling window fraction near the end of each base/pulse segment."""
        return self._sample_fraction

    @property
    def direction(self) -> int:
        """Scan direction: ``+1`` (anodic) or ``-1`` (cathodic)."""
        return self._direction

    @property
    def n_steps(self) -> int:
        """Number of staircase steps / pulses."""
        return self._n_steps

    @property
    def base_potentials(self) -> tuple[float, ...]:
        """Ordered tuple of staircase base potentials :math:`E_{\\text{base}, k}` in Volts."""
        return self._base_potentials

    @property
    def pulse_potentials(self) -> tuple[float, ...]:
        """Ordered tuple of pulse potentials :math:`E_{\\text{base}, k} + \\Delta E_p` in Volts."""
        return self._pulse_potentials

    @property
    def duration(self) -> float:
        """Total experiment duration :math:`N \\tau` in seconds."""
        return self._duration

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="dpv",
                title="Differential Pulse Voltammogram",
                x_label=r"$E_{\mathrm{base}}$ / V",
                y_label=r"$\Delta I$ / A",
                description="Differential pulse current (i_pulse - i_pre) vs. base potential.",
            ),
        )

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior pulse rise and fall timestamps in ``(0, duration)``."""
        return self._event_times

    def _default_time_grid_points(self) -> int:
        return max(1001, 20 * self._n_steps + 1)

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        n_seg = len(self._values)

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return float(self._values[0])
            if normalized >= self._duration:
                return float(self._values[-1])
            idx = int(np.searchsorted(self._starts, normalized, side="right")) - 1
            idx = min(max(idx, 0), n_seg - 1)
            return float(self._values[idx])

        assert isinstance(normalized, np.ndarray)
        orig_shape = normalized.shape
        flat = normalized.ravel()
        if flat.size == 0:
            return np.empty(orig_shape, dtype=float)

        clamped = np.clip(flat, 0.0, self._duration)
        indices = np.searchsorted(self._starts, clamped, side="right") - 1
        indices = np.clip(indices, 0, n_seg - 1)
        indices = np.where(clamped >= self._duration, n_seg - 1, indices)
        return self._values[indices].reshape(orig_shape)

    def sample_currents(
        self,
        t: ArrayLike,
        i: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample pre-pulse, pulse, and differential currents across all steps.

        Parameters
        ----------
        t : array-like
            Strictly increasing 1D time array in seconds.
        i : array-like
            1D Faradaic current array in Amperes matching ``t``.

        Returns
        -------
        E_base : np.ndarray
            Staircase base potentials of shape ``(n_steps,)``.
        i_pre : np.ndarray
            Currents sampled near the end of each base hold, shape ``(n_steps,)``.
        i_pulse : np.ndarray
            Currents sampled near the end of each pulse hold, shape ``(n_steps,)``.
        i_diff : np.ndarray
            Differential current ``i_pulse - i_pre``, shape ``(n_steps,)``.
        """
        t_arr, i_arr = _validate_time_current_pair(t, i)

        w_base = self._sample_fraction * self._base_width
        w_pulse = self._sample_fraction * self._pulse_width

        i_pre = np.empty(self._n_steps, dtype=float)
        i_pulse = np.empty(self._n_steps, dtype=float)

        for k in range(self._n_steps):
            t_rise = (k + 1) * self._pulse_period - self._pulse_width
            t_end = (k + 1) * self._pulse_period

            i_pre[k] = _sample_window_mean(
                t_arr, i_arr, t_rise - w_base, t_rise, include_end=False
            )
            i_pulse[k] = _sample_window_mean(
                t_arr,
                i_arr,
                t_end - w_pulse,
                t_end,
                include_end=(k == self._n_steps - 1),
            )

        E_base = np.asarray(self._base_potentials, dtype=float)
        i_diff = i_pulse - i_pre
        return E_base, i_pre, i_pulse, i_diff

    def __repr__(self) -> str:
        return (
            f"DifferentialPulseVoltammetry(E_start={self._E_start!r}, "
            f"E_end={self._E_end!r}, step_height={self._step_height!r}, "
            f"pulse_amplitude={self._pulse_amplitude!r}, "
            f"pulse_width={self._pulse_width!r}, pulse_period={self._pulse_period!r})"
        )


class SquareWaveVoltammetry(BaseTechnique):
    r"""Osteryoung Square-Wave Voltammetry (SWV) waveform generator.

    Superimposes a symmetric square wave of amplitude :math:`E_{\text{sw}} > 0`
    and frequency :math:`f > 0` (period :math:`\tau = 1/f`, half-period
    :math:`t_p = 1/(2f)`) onto a linear staircase ramp from :math:`E_{\text{start}}`
    to :math:`E_{\text{end}}` with step height :math:`\Delta E_s > 0`.

    Within each cycle :math:`k \in \{0, \dots, N-1\}` over
    :math:`[k\tau, (k+1)\tau)` with scan direction
    :math:`s = \operatorname{sgn}(E_{\text{end}} - E_{\text{start}}) \in \{+1, -1\}`:

    - **Forward half-cycle** (:math:`t \in [k\tau, k\tau + t_p)`):
      :math:`E(t) = E_{\text{step}, k} + s \, E_{\text{sw}}`
    - **Reverse half-cycle** (:math:`t \in [k\tau + t_p, (k+1)\tau)`):
      :math:`E(t) = E_{\text{step}, k} - s \, E_{\text{sw}}`

    Currents are sampled near the end of each forward (:math:`I_f`) and reverse
    (:math:`I_r`) half-period via :meth:`sample_currents`, yielding the net
    square-wave current :math:`\Delta I = I_f - I_r` under IUPAC polarity.

    Parameters
    ----------
    E_start : float
        Initial staircase potential in Volts.
    E_end : float
        Final staircase potential in Volts (must differ from ``E_start``).
    step_height : float
        Staircase potential step magnitude :math:`\Delta E_s > 0` in Volts.
    amplitude : float
        Square-wave pulse amplitude :math:`E_{\text{sw}} > 0` in Volts.
    frequency : float
        Square-wave frequency :math:`f > 0` in Hz.
    sample_fraction : float, default 0.1
        Fraction :math:`\in (0, 1]` of each half-period near the end of the
        half-cycle over which current is averaged in :meth:`sample_currents`.

    Examples
    --------
    >>> from softpotato.techniques import SquareWaveVoltammetry
    >>> swv = SquareWaveVoltammetry(
    ...     E_start=0.2,
    ...     E_end=-0.2,
    ...     step_height=0.05,
    ...     amplitude=0.025,
    ...     frequency=10.0,
    ... )
    >>> swv.n_steps
    9
    >>> swv.period
    0.1
    >>> swv.half_period
    0.05
    """

    def __init__(
        self,
        E_start: float,
        E_end: float,
        step_height: float,
        amplitude: float,
        frequency: float,
        *,
        sample_fraction: float = 0.1,
    ) -> None:
        self._mode = "potentiostatic"
        self._E_start = _validate_finite_real(E_start, "E_start")
        self._E_end = _validate_finite_real(E_end, "E_end")
        if self._E_start == self._E_end:
            raise ValueError("E_start and E_end must be distinct.")

        self._step_height = _validate_positive_finite(step_height, "step_height")
        self._amplitude = _validate_positive_finite(amplitude, "amplitude")
        self._frequency = _validate_positive_finite(frequency, "frequency")
        self._sample_fraction = _validate_sample_fraction(sample_fraction)

        self._period = 1.0 / self._frequency
        self._half_period = 0.5 * self._period

        self._step_potentials, self._direction = _build_step_potentials(
            self._E_start, self._E_end, self._step_height
        )
        signed_amp = self._direction * self._amplitude
        self._forward_potentials = tuple(
            float(e + signed_amp) for e in self._step_potentials
        )
        self._reverse_potentials = tuple(
            float(e - signed_amp) for e in self._step_potentials
        )
        self._n_steps = len(self._step_potentials)
        self._duration = float(self._n_steps * self._period)

        seg_starts: list[float] = []
        seg_values: list[float] = []
        for k in range(self._n_steps):
            t_fwd = k / self._frequency
            t_rev = (2 * k + 1) / (2.0 * self._frequency)
            seg_starts.append(t_fwd)
            seg_values.append(self._forward_potentials[k])
            seg_starts.append(t_rev)
            seg_values.append(self._reverse_potentials[k])

        self._starts = np.asarray(seg_starts, dtype=float)
        self._values = np.asarray(seg_values, dtype=float)
        self._event_times = _deduplicate_interior_events(seg_starts[1:], self._duration)

    @property
    def E_start(self) -> float:
        """Initial staircase potential in Volts."""
        return self._E_start

    @property
    def E_end(self) -> float:
        """Final staircase potential in Volts."""
        return self._E_end

    @property
    def step_height(self) -> float:
        """Staircase potential step magnitude :math:`\\Delta E_s` in Volts."""
        return self._step_height

    @property
    def amplitude(self) -> float:
        """Square-wave pulse amplitude :math:`E_{\\text{sw}}` in Volts."""
        return self._amplitude

    @property
    def frequency(self) -> float:
        """Square-wave frequency :math:`f` in Hz."""
        return self._frequency

    @property
    def period(self) -> float:
        """Square-wave cycle period :math:`\\tau = 1 / f` in seconds."""
        return self._period

    @property
    def half_period(self) -> float:
        """Square-wave half-period :math:`t_p = 1 / (2 f)` in seconds."""
        return self._half_period

    @property
    def sample_fraction(self) -> float:
        """Sampling window fraction near the end of each half-cycle."""
        return self._sample_fraction

    @property
    def direction(self) -> int:
        """Scan direction: ``+1`` (anodic) or ``-1`` (cathodic)."""
        return self._direction

    @property
    def n_steps(self) -> int:
        """Number of staircase steps / square-wave cycles."""
        return self._n_steps

    @property
    def step_potentials(self) -> tuple[float, ...]:
        """Ordered tuple of staircase step potentials :math:`E_{\\text{step}, k}` in Volts."""
        return self._step_potentials

    @property
    def forward_potentials(self) -> tuple[float, ...]:
        """Ordered tuple of forward half-cycle potentials in Volts."""
        return self._forward_potentials

    @property
    def reverse_potentials(self) -> tuple[float, ...]:
        """Ordered tuple of reverse half-cycle potentials in Volts."""
        return self._reverse_potentials

    @property
    def duration(self) -> float:
        """Total experiment duration :math:`N / f` in seconds."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior half-period transition timestamps in ``(0, duration)``."""
        return self._event_times

    def _default_time_grid_points(self) -> int:
        return max(1001, 20 * self._n_steps + 1)

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        n_seg = len(self._values)

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return float(self._values[0])
            if normalized >= self._duration:
                return float(self._values[-1])
            idx = int(np.searchsorted(self._starts, normalized, side="right")) - 1
            idx = min(max(idx, 0), n_seg - 1)
            return float(self._values[idx])

        assert isinstance(normalized, np.ndarray)
        orig_shape = normalized.shape
        flat = normalized.ravel()
        if flat.size == 0:
            return np.empty(orig_shape, dtype=float)

        clamped = np.clip(flat, 0.0, self._duration)
        indices = np.searchsorted(self._starts, clamped, side="right") - 1
        indices = np.clip(indices, 0, n_seg - 1)
        indices = np.where(clamped >= self._duration, n_seg - 1, indices)
        return self._values[indices].reshape(orig_shape)

    def sample_currents(
        self,
        t: ArrayLike,
        i: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample forward, reverse, and net currents across all square-wave cycles.

        Parameters
        ----------
        t : array-like
            Strictly increasing 1D time array in seconds.
        i : array-like
            1D Faradaic current array in Amperes matching ``t``.

        Returns
        -------
        E_step : np.ndarray
            Staircase potentials of shape ``(n_steps,)``.
        i_forward : np.ndarray
            Forward half-cycle currents of shape ``(n_steps,)``.
        i_reverse : np.ndarray
            Reverse half-cycle currents of shape ``(n_steps,)``.
        i_net : np.ndarray
            Net square-wave current ``i_forward - i_reverse``, shape ``(n_steps,)``.
        """
        t_arr, i_arr = _validate_time_current_pair(t, i)

        w_half = self._sample_fraction * self._half_period

        i_forward = np.empty(self._n_steps, dtype=float)
        i_reverse = np.empty(self._n_steps, dtype=float)

        for k in range(self._n_steps):
            t_fwd_end = (2 * k + 1) / (2.0 * self._frequency)
            t_rev_end = (k + 1) / self._frequency

            i_forward[k] = _sample_window_mean(
                t_arr, i_arr, t_fwd_end - w_half, t_fwd_end, include_end=False
            )
            i_reverse[k] = _sample_window_mean(
                t_arr,
                i_arr,
                t_rev_end - w_half,
                t_rev_end,
                include_end=(k == self._n_steps - 1),
            )

        E_step = np.asarray(self._step_potentials, dtype=float)
        i_net = i_forward - i_reverse
        return E_step, i_forward, i_reverse, i_net

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="swv",
                title="Square-Wave Voltammogram",
                x_label=r"$E_{\mathrm{step}}$ / V",
                y_label="$I$ / A",
                description="Net square-wave current (i_forward - i_reverse) vs. step potential.",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"SquareWaveVoltammetry(E_start={self._E_start!r}, "
            f"E_end={self._E_end!r}, step_height={self._step_height!r}, "
            f"amplitude={self._amplitude!r}, frequency={self._frequency!r})"
        )
