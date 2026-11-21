"""Voltammetry potential waveforms: Linear Sweep (LSV) and Cyclic Voltammetry (CV)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, overload

import numpy as np

from .base import (
    BaseTechnique,
    PlotSpec,
    _deduplicate_interior_events,
    _normalize_time_input,
    _validate_finite_real,
    _validate_positive_finite,
    _validate_positive_int,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class LinearSweepVoltammetry(BaseTechnique):
    r"""Unidirectional linear potential sweep waveform (LSV).

    Sweeps electrode potential linearly from :math:`E_{\text{start}}` to
    :math:`E_{\text{end}}` at scan rate :math:`v > 0` (V/s):

    .. math::

        E(t) = E_{\text{start}} + \operatorname{sgn}(E_{\text{end}} - E_{\text{start}}) \cdot v \cdot t

    for :math:`t \in [0, |E_{\text{end}} - E_{\text{start}}| / v]`.

    Parameters
    ----------
    E_start : float
        Initial potential :math:`E_{\text{start}}` in Volts.
    E_end : float
        Final potential :math:`E_{\text{end}}` in Volts (must differ from ``E_start``).
    scan_rate : float
        Scan rate magnitude :math:`v > 0` in V/s.
    dE : float, optional
        Default potential step resolution :math:`\Delta E > 0` in Volts used by
        :meth:`time_grid` when no resolution arguments are passed. Mutually
        exclusive with ``n_points``.
    n_points : int, optional
        Default number of points (:math:`\ge 2`) used by :meth:`time_grid` when
        no resolution arguments are passed. Mutually exclusive with ``dE``.

    Examples
    --------
    >>> from softpotato.techniques import LinearSweepVoltammetry
    >>> lsv = LinearSweepVoltammetry(E_start=0.5, E_end=-0.5, scan_rate=0.1)
    >>> lsv.duration
    10.0
    >>> lsv(5.0)
    0.0
    """

    def __init__(
        self,
        E_start: float,
        E_end: float,
        scan_rate: float,
        *,
        dE: float | None = None,
        n_points: int | None = None,
    ) -> None:
        self._mode = "potentiostatic"
        self._E_start = _validate_finite_real(E_start, "E_start")
        self._E_end = _validate_finite_real(E_end, "E_end")
        if self._E_start == self._E_end:
            raise ValueError("E_start and E_end must be distinct.")

        self._scan_rate = _validate_positive_finite(scan_rate, "scan_rate")

        if dE is not None and n_points is not None:
            raise ValueError("Specify at most one of 'dE' or 'n_points'.")
        self._dE = _validate_positive_finite(dE, "dE") if dE is not None else None
        self._n_points = (
            _validate_positive_int(n_points, "n_points", minimum=2)
            if n_points is not None
            else None
        )

        self._direction = 1 if self._E_end > self._E_start else -1
        self._duration = abs(self._E_end - self._E_start) / self._scan_rate

    @property
    def E_start(self) -> float:
        """Initial electrode potential in Volts."""
        return self._E_start

    @property
    def E_end(self) -> float:
        """Final electrode potential in Volts."""
        return self._E_end

    @property
    def scan_rate(self) -> float:
        """Scan rate magnitude :math:`v` in V/s."""
        return self._scan_rate

    @property
    def direction(self) -> int:
        """Sweep direction: ``+1`` (anodic/positive) or ``-1`` (cathodic/negative)."""
        return self._direction

    @property
    def dE(self) -> float | None:
        """Default potential step resolution in Volts, or ``None``."""
        return self._dE

    @property
    def n_points(self) -> int | None:
        """Default number of time grid points, or ``None``."""
        return self._n_points

    @property
    def duration(self) -> float:
        """Total sweep duration :math:`|E_{\\text{end}} - E_{\\text{start}}| / v` in seconds."""
        return self._duration

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        slope = self._direction * self._scan_rate

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return self._E_start
            if normalized >= self._duration:
                return self._E_end
            return self._E_start + slope * normalized

        assert isinstance(normalized, np.ndarray)
        clamped = np.clip(normalized, 0.0, self._duration)
        values = self._E_start + slope * clamped
        values = np.where(clamped <= 0.0, self._E_start, values)
        values = np.where(clamped >= self._duration, self._E_end, values)
        return values

    def time_grid(
        self,
        n_points: int | None = None,
        dt: float | None = None,
        dE: float | None = None,
    ) -> np.ndarray:
        """Generate evaluation timestamps for the linear sweep.

        Parameters
        ----------
        n_points : int, optional
            Number of uniformly spaced timestamps (``>= 2``).
        dt : float, optional
            Uniform time step in seconds (``> 0``).
        dE : float, optional
            Uniform potential step in Volts (``> 0``), converted to
            ``dt = dE / scan_rate``.

        Returns
        -------
        np.ndarray
            Strictly increasing 1D array of timestamps in ``[0, duration]``.
        """
        n_specified = sum(x is not None for x in (n_points, dt, dE))
        if n_specified > 1:
            raise ValueError("Specify at most one of 'n_points', 'dt', or 'dE'.")

        if dE is not None:
            dE_val = _validate_positive_finite(dE, "dE")
            return super().time_grid(dt=dE_val / self._scan_rate)
        if n_points is not None or dt is not None:
            return super().time_grid(n_points=n_points, dt=dt)

        if self._n_points is not None:
            return super().time_grid(n_points=self._n_points)
        if self._dE is not None:
            return super().time_grid(dt=self._dE / self._scan_rate)
        return super().time_grid()

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="voltammogram",
                title="Linear Sweep Voltammogram",
                x_label="Potential, $E$ / V",
                y_label="Current, $I$ / A",
                description="Faradaic current vs. applied potential during linear sweep.",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"LinearSweepVoltammetry(E_start={self._E_start!r}, "
            f"E_end={self._E_end!r}, scan_rate={self._scan_rate!r})"
        )


class CyclicVoltammetry(BaseTechnique):
    r"""Multi-cycle triangular potential waveform generator (Cyclic Voltammetry).

    Each cycle sweeps linearly along the vertex sequence:

    - :math:`E_{\text{start}} \to E_{\text{vertex1}} \to E_{\text{vertex2}} \to E_{\text{start}}`
      for intermediate cycles (with the final leg omitted when
      :math:`E_{\text{vertex2}} = E_{\text{start}}`), and
    - :math:`E_{\text{start}} \to E_{\text{vertex1}} \to E_{\text{vertex2}} \to E_{\text{end}}`
      for the final cycle (where ``E_end`` defaults to ``E_start``, and the final
      leg is omitted when :math:`E_{\text{end}} = E_{\text{vertex2}}`).

    Exact switching vertex timestamps (where :math:`dE/dt` reverses sign) are
    recorded in :attr:`event_times`, and per-cycle time intervals are exposed via
    :attr:`cycle_time_ranges`.

    Parameters
    ----------
    E_start : float
        Initial potential :math:`E_{\text{start}}` in Volts.
    E_vertex1 : float
        First switching vertex potential :math:`E_{\lambda 1}` in Volts
        (must differ from ``E_start``).
    E_vertex2 : float
        Second switching vertex potential :math:`E_{\lambda 2}` in Volts
        (must differ from ``E_vertex1``). Set equal to ``E_start`` for a
        standard two-vertex triangular wave.
    scan_rate : float
        Scan rate magnitude :math:`v > 0` in V/s.
    E_end : float, optional
        Final potential at the end of the last cycle in Volts. Defaults to
        ``E_start``.
    n_cycles : int, default 1
        Number of complete voltammetric cycles (:math:`\ge 1`).
    dE : float, optional
        Default potential step resolution :math:`\Delta E > 0` in Volts for
        :meth:`time_grid`. Mutually exclusive with ``n_points``.
    n_points : int, optional
        Default number of points (:math:`\ge 2`) for :meth:`time_grid`.
        Mutually exclusive with ``dE``.

    Examples
    --------
    >>> from softpotato.techniques import CyclicVoltammetry
    >>> cv = CyclicVoltammetry(
    ...     E_start=0.5, E_vertex1=-0.5, E_vertex2=0.5, scan_rate=0.5, n_cycles=2
    ... )
    >>> cv.duration
    8.0
    >>> cv.event_times
    (2.0, 4.0, 6.0)
    >>> cv.cycle_time_ranges
    ((0.0, 4.0), (4.0, 8.0))
    """

    def __init__(
        self,
        E_start: float,
        E_vertex1: float,
        E_vertex2: float,
        scan_rate: float,
        *,
        E_end: float | None = None,
        n_cycles: int = 1,
        dE: float | None = None,
        n_points: int | None = None,
    ) -> None:
        self._mode = "potentiostatic"
        self._E_start = _validate_finite_real(E_start, "E_start")
        self._E_vertex1 = _validate_finite_real(E_vertex1, "E_vertex1")
        self._E_vertex2 = _validate_finite_real(E_vertex2, "E_vertex2")
        self._E_end = (
            self._E_start if E_end is None else _validate_finite_real(E_end, "E_end")
        )

        if self._E_vertex1 == self._E_start:
            raise ValueError("E_vertex1 must differ from E_start.")
        if self._E_vertex2 == self._E_vertex1:
            raise ValueError("E_vertex2 must differ from E_vertex1.")

        self._scan_rate = _validate_positive_finite(scan_rate, "scan_rate")
        self._n_cycles = _validate_positive_int(n_cycles, "n_cycles", minimum=1)

        if dE is not None and n_points is not None:
            raise ValueError("Specify at most one of 'dE' or 'n_points'.")
        self._dE = _validate_positive_finite(dE, "dE") if dE is not None else None
        self._n_points = (
            _validate_positive_int(n_points, "n_points", minimum=2)
            if n_points is not None
            else None
        )

        # Build piecewise-linear legs across all cycles
        leg_starts: list[float] = []
        leg_e0: list[float] = []
        leg_e1: list[float] = []
        leg_durations: list[float] = []
        leg_slopes: list[float] = []
        cycle_ranges: list[tuple[float, float]] = []

        cursor = 0.0
        for cycle_idx in range(self._n_cycles):
            cycle_start = cursor
            cycle_end_target = (
                self._E_end if cycle_idx == self._n_cycles - 1 else self._E_start
            )
            waypoints = [self._E_start, self._E_vertex1, self._E_vertex2]
            if cycle_end_target != self._E_vertex2:
                waypoints.append(cycle_end_target)

            for e_from, e_to in itertools.pairwise(waypoints):
                dur = abs(e_to - e_from) / self._scan_rate
                slope = self._scan_rate if e_to > e_from else -self._scan_rate
                leg_starts.append(cursor)
                leg_e0.append(e_from)
                leg_e1.append(e_to)
                leg_durations.append(dur)
                leg_slopes.append(slope)
                cursor += dur

            cycle_ranges.append((cycle_start, cursor))

        self._leg_starts = np.asarray(leg_starts, dtype=float)
        self._leg_e0 = np.asarray(leg_e0, dtype=float)
        self._leg_e1 = np.asarray(leg_e1, dtype=float)
        self._leg_durations = np.asarray(leg_durations, dtype=float)
        self._leg_slopes = np.asarray(leg_slopes, dtype=float)
        self._duration = float(cursor)
        self._cycle_time_ranges: tuple[tuple[float, float], ...] = tuple(cycle_ranges)

        # Switching vertices are interior transitions where dE/dt reverses sign
        switching_times: list[float] = []
        for idx in range(1, len(leg_starts)):
            if leg_slopes[idx - 1] * leg_slopes[idx] < 0.0:
                switching_times.append(leg_starts[idx])
        self._event_times = _deduplicate_interior_events(
            switching_times, self._duration
        )

    @property
    def E_start(self) -> float:
        """Initial potential in Volts."""
        return self._E_start

    @property
    def E_vertex1(self) -> float:
        """First switching vertex potential in Volts."""
        return self._E_vertex1

    @property
    def E_vertex2(self) -> float:
        """Second switching vertex potential in Volts."""
        return self._E_vertex2

    @property
    def E_end(self) -> float:
        """Final potential at the end of the last cycle in Volts."""
        return self._E_end

    @property
    def scan_rate(self) -> float:
        """Scan rate magnitude :math:`v` in V/s."""
        return self._scan_rate

    @property
    def n_cycles(self) -> int:
        """Number of voltammetric cycles."""
        return self._n_cycles

    @property
    def dE(self) -> float | None:
        """Default potential step resolution in Volts, or ``None``."""
        return self._dE

    @property
    def n_points(self) -> int | None:
        """Default number of time grid points, or ``None``."""
        return self._n_points

    @property
    def duration(self) -> float:
        """Total multi-cycle experiment duration in seconds."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted switching vertex timestamps where scan direction reverses."""
        return self._event_times

    @property
    def cycle_time_ranges(self) -> tuple[tuple[float, float], ...]:
        """Per-cycle ``(t_start, t_end)`` intervals for slicing multi-cycle results."""
        return self._cycle_time_ranges

    def _extra_grid_times(self) -> tuple[float, ...]:
        cycle_bounds = tuple(
            t for start, end in self._cycle_time_ranges for t in (start, end)
        )
        return self._event_times + cycle_bounds

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        n_legs = len(self._leg_starts)

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return float(self._leg_e0[0])
            if normalized >= self._duration:
                return float(self._leg_e1[-1])
            idx = int(np.searchsorted(self._leg_starts, normalized, side="right")) - 1
            idx = min(max(idx, 0), n_legs - 1)
            dt_local = normalized - float(self._leg_starts[idx])
            if dt_local <= 0.0:
                return float(self._leg_e0[idx])
            if dt_local >= float(self._leg_durations[idx]):
                return float(self._leg_e1[idx])
            return float(self._leg_e0[idx] + self._leg_slopes[idx] * dt_local)

        assert isinstance(normalized, np.ndarray)
        orig_shape = normalized.shape
        flat = normalized.ravel()
        if flat.size == 0:
            return np.empty(orig_shape, dtype=float)

        clamped = np.clip(flat, 0.0, self._duration)
        indices = np.searchsorted(self._leg_starts, clamped, side="right") - 1
        indices = np.clip(indices, 0, n_legs - 1)
        at_end = clamped >= self._duration
        indices = np.where(at_end, n_legs - 1, indices)

        dt_local = clamped - self._leg_starts[indices]
        e0 = self._leg_e0[indices]
        e1 = self._leg_e1[indices]
        durs = self._leg_durations[indices]
        slopes = self._leg_slopes[indices]

        values = e0 + slopes * dt_local
        values = np.where(dt_local <= 0.0, e0, values)
        values = np.where((dt_local >= durs) | at_end, e1, values)
        return values.reshape(orig_shape)

    def time_grid(
        self,
        n_points: int | None = None,
        dt: float | None = None,
        dE: float | None = None,
    ) -> np.ndarray:
        """Generate evaluation timestamps including all switching vertices and cycle bounds.

        Parameters
        ----------
        n_points : int, optional
            Number of uniformly spaced base timestamps (``>= 2``).
        dt : float, optional
            Uniform time step in seconds (``> 0``).
        dE : float, optional
            Uniform potential step in Volts (``> 0``), converted to
            ``dt = dE / scan_rate``.

        Returns
        -------
        np.ndarray
            Strictly increasing 1D array of timestamps in ``[0, duration]``.
        """
        n_specified = sum(x is not None for x in (n_points, dt, dE))
        if n_specified > 1:
            raise ValueError("Specify at most one of 'n_points', 'dt', or 'dE'.")

        if dE is not None:
            dE_val = _validate_positive_finite(dE, "dE")
            return super().time_grid(dt=dE_val / self._scan_rate)
        if n_points is not None or dt is not None:
            return super().time_grid(n_points=n_points, dt=dt)

        if self._n_points is not None:
            return super().time_grid(n_points=self._n_points)
        if self._dE is not None:
            return super().time_grid(dt=self._dE / self._scan_rate)
        return super().time_grid()

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        return (
            PlotSpec(
                kind="voltammogram",
                title="Cyclic Voltammogram",
                x_label="Potential, $E$ / V",
                y_label="Current, $I$ / A",
                description="Faradaic current vs. applied potential across voltammetric cycle(s).",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"CyclicVoltammetry(E_start={self._E_start!r}, "
            f"E_vertex1={self._E_vertex1!r}, E_vertex2={self._E_vertex2!r}, "
            f"E_end={self._E_end!r}, scan_rate={self._scan_rate!r}, "
            f"n_cycles={self._n_cycles!r})"
        )
