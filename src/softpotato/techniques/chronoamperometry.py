"""Chronoamperometry (single-, double-, and multi-step) potential waveforms."""

from __future__ import annotations

from collections.abc import Sequence
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
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Chronoamperometry(BaseTechnique):
    r"""Potentiostatic step-potential waveform generator (Chronoamperometry).

    Supports three construction modes, each optionally preceded by a resting
    (quiet) potential hold ``(E_quiet, t_quiet)``:

    1. **Single potential step**: ``Chronoamperometry(E_step=..., duration=...)``
    2. **Double potential step**:
       ``Chronoamperometry(E_step1=..., E_step2=..., t_step1=..., t_step2=...)``
    3. **Arbitrary multi-step sequence**:
       ``Chronoamperometry(potentials=[...], durations=[...])``

    At step transitions :math:`t = t_k`, the waveform is right-continuous:
    :math:`E(t) = E_k` for :math:`t \in [t_k, t_{k+1})`, and
    :math:`E(t_{\max}) = E_{\text{last}}`.

    Parameters
    ----------
    E_step : float, optional
        Single-step applied potential in Volts. Must be paired with ``duration``.
    duration : float, optional
        Single-step duration in seconds (:math:`> 0`). Must be paired with ``E_step``.
    E_step1 : float, optional
        First step potential in Volts for double-step chronoamperometry.
    E_step2 : float, optional
        Second (reversal) step potential in Volts for double-step chronoamperometry.
    t_step1 : float, optional
        First step duration in seconds (:math:`> 0`).
    t_step2 : float, optional
        Second step duration in seconds (:math:`> 0`).
    potentials : Sequence[float], optional
        Ordered sequence of step potentials in Volts for multi-step experiments.
    durations : Sequence[float], optional
        Ordered sequence of step durations in seconds (:math:`> 0`), matching
        the length of ``potentials``.
    E_quiet : float, optional
        Optional initial quiet/resting potential in Volts applied before the
        first step when ``t_quiet > 0``.
    t_quiet : float, default 0.0
        Optional initial quiet/resting duration in seconds (:math:`\ge 0`).
        If ``t_quiet > 0``, ``E_quiet`` must be provided.

    Examples
    --------
    Single-step Cottrell experiment:

    >>> from softpotato.techniques import Chronoamperometry
    >>> ca = Chronoamperometry(E_step=-0.5, duration=5.0)
    >>> ca.duration
    5.0
    >>> ca(2.5)
    -0.5

    Double-step chronoamperometry:

    >>> dca = Chronoamperometry(E_step1=-0.5, E_step2=0.5, t_step1=2.0, t_step2=3.0)
    >>> dca.event_times
    (2.0,)
    >>> dca(1.9), dca(2.0), dca(5.0)
    (-0.5, 0.5, 0.5)
    """

    def __init__(
        self,
        E_step: float | None = None,
        duration: float | None = None,
        *,
        E_step1: float | None = None,
        E_step2: float | None = None,
        t_step1: float | None = None,
        t_step2: float | None = None,
        potentials: Sequence[float] | None = None,
        durations: Sequence[float] | None = None,
        E_quiet: float | None = None,
        t_quiet: float = 0.0,
    ) -> None:
        self._mode = "potentiostatic"

        single_any = E_step is not None or duration is not None
        double_any = any(x is not None for x in (E_step1, E_step2, t_step1, t_step2))
        multi_any = potentials is not None or durations is not None

        n_modes = int(single_any) + int(double_any) + int(multi_any)
        if n_modes == 0:
            raise ValueError(
                "Must specify one of: single-step (E_step, duration), "
                "double-step (E_step1, E_step2, t_step1, t_step2), or "
                "multi-step (potentials, durations)."
            )
        if n_modes > 1:
            raise ValueError(
                "Cannot mix single-step, double-step, and multi-step parameters."
            )

        step_potentials: list[float] = []
        step_durations: list[float] = []

        t_quiet_val = _validate_non_negative_finite(t_quiet, "t_quiet")
        e_quiet_val: float | None = (
            _validate_finite_real(E_quiet, "E_quiet") if E_quiet is not None else None
        )
        if t_quiet_val > 0.0:
            if e_quiet_val is None:
                raise ValueError("E_quiet must be provided when t_quiet > 0.")
            step_potentials.append(e_quiet_val)
            step_durations.append(t_quiet_val)

        if single_any:
            if E_step is None or duration is None:
                raise ValueError(
                    "Single-step mode requires both 'E_step' and 'duration'."
                )
            step_potentials.append(_validate_finite_real(E_step, "E_step"))
            step_durations.append(_validate_positive_finite(duration, "duration"))
        elif double_any:
            if any(x is None for x in (E_step1, E_step2, t_step1, t_step2)):
                raise ValueError(
                    "Double-step mode requires all of 'E_step1', 'E_step2', "
                    "'t_step1', and 't_step2'."
                )
            step_potentials.append(_validate_finite_real(E_step1, "E_step1"))
            step_potentials.append(_validate_finite_real(E_step2, "E_step2"))
            step_durations.append(_validate_positive_finite(t_step1, "t_step1"))
            step_durations.append(_validate_positive_finite(t_step2, "t_step2"))
        else:
            if potentials is None or durations is None:
                raise ValueError(
                    "Multi-step mode requires both 'potentials' and 'durations'."
                )
            if isinstance(potentials, (str, bytes)) or isinstance(
                durations, (str, bytes)
            ):
                raise TypeError(
                    "'potentials' and 'durations' must be numeric sequences."
                )
            pot_seq = list(potentials)
            dur_seq = list(durations)
            if len(pot_seq) == 0 or len(dur_seq) == 0:
                raise ValueError("'potentials' and 'durations' must not be empty.")
            if len(pot_seq) != len(dur_seq):
                raise ValueError(
                    f"'potentials' (len={len(pot_seq)}) and 'durations' "
                    f"(len={len(dur_seq)}) must have the same length."
                )
            for idx, (e_val, d_val) in enumerate(zip(pot_seq, dur_seq)):
                step_potentials.append(
                    _validate_finite_real(e_val, f"potentials[{idx}]")
                )
                step_durations.append(
                    _validate_positive_finite(d_val, f"durations[{idx}]")
                )

        self._E_quiet = e_quiet_val
        self._t_quiet = t_quiet_val
        self._potentials = tuple(step_potentials)
        self._durations = tuple(step_durations)

        starts: list[float] = []
        cursor = 0.0
        for d in self._durations:
            starts.append(cursor)
            cursor += d

        self._starts = np.asarray(starts, dtype=float)
        self._values = np.asarray(self._potentials, dtype=float)
        self._duration = float(cursor)
        self._event_times = _deduplicate_interior_events(starts[1:], self._duration)

    @property
    def potentials(self) -> tuple[float, ...]:
        """Ordered tuple of step potentials in Volts (including quiet hold if active)."""
        return self._potentials

    @property
    def durations(self) -> tuple[float, ...]:
        """Ordered tuple of step durations in seconds (including quiet hold if active)."""
        return self._durations

    @property
    def E_quiet(self) -> float | None:
        """Initial quiet/resting potential in Volts, or ``None``."""
        return self._E_quiet

    @property
    def t_quiet(self) -> float:
        """Initial quiet/resting duration in seconds."""
        return self._t_quiet

    @property
    def n_steps(self) -> int:
        """Total number of constant-potential segments."""
        return len(self._potentials)

    @property
    def step_times(self) -> tuple[float, ...]:
        """Start timestamps of each potential step."""
        return tuple(float(x) for x in self._starts)

    @property
    def duration(self) -> float:
        """Total experiment duration in seconds."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior step transition timestamps in ``(0, duration)``."""
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

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="i_vs_t",
                title="Chronoamperogram",
                x_label="Time, $t$ / s",
                y_label="Current, $I$ / A",
                description="Faradaic current transient following potential step(s).",
            ),
            PlotSpec(
                kind="Q_vs_t",
                title="Chronocoulogram",
                x_label="Time, $t$ / s",
                y_label="Charge, $Q$ / C",
                description="Cumulative Faradaic charge Q(t) vs. time.",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Chronoamperometry(potentials={self._potentials!r}, "
            f"durations={self._durations!r})"
        )
