"""Composable waveform segment primitives and CustomWaveform generator."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, overload

import numpy as np

from .base import (
    BaseTechnique,
    ControlMode,
    _deduplicate_interior_events,
    _normalize_time_input,
    _validate_finite_real,
    _validate_mode,
    _validate_positive_finite,
    _validate_positive_int,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class WaveformSegment(ABC):
    r"""Abstract base class for a single time-localized waveform segment.

    A segment is defined over local time :math:`\tau \in [0, \Delta t]` with
    strictly positive duration :math:`\Delta t > 0`. Segments can be chained
    sequentially inside :class:`CustomWaveform`.
    """

    def __init__(self, duration: float) -> None:
        self._duration = _validate_positive_finite(duration, "duration")

    @property
    def duration(self) -> float:
        """Segment duration in seconds (:math:`\\Delta t > 0`)."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Optional interior event timestamps in local time ``(0, duration)``."""
        return ()

    @property
    def start_value(self) -> float:
        """Excitation value at the start of the segment (:math:`\\tau = 0`)."""
        return float(self.evaluate(0.0))

    @property
    def end_value(self) -> float:
        """Excitation value at the end of the segment (:math:`\\tau = \\Delta t`)."""
        return float(self.evaluate(self._duration))

    @overload
    def evaluate(self, t_local: float) -> float: ...

    @overload
    def evaluate(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    @abstractmethod
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray:
        """Evaluate the segment at local time(s) ``t_local`` in ``[0, duration]``."""
        raise NotImplementedError

    @overload
    def __call__(self, t_local: float) -> float: ...

    @overload
    def __call__(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    def __call__(self, t_local: float | ArrayLike) -> float | np.ndarray:
        """Callable alias for :meth:`evaluate`."""
        return self.evaluate(t_local)


class ConstantSegment(WaveformSegment):
    r"""Constant-value hold segment :math:`y(\tau) = y_0` for :math:`\tau \in [0, \Delta t]`.

    Parameters
    ----------
    value : float
        Constant excitation level (potential in V or current in A).
    duration : float
        Hold duration in seconds (:math:`\Delta t > 0`).
    """

    def __init__(self, value: float, duration: float) -> None:
        super().__init__(duration)
        self._value = _validate_finite_real(value, "value")

    @property
    def value(self) -> float:
        """Constant excitation level."""
        return self._value

    @overload
    def evaluate(self, t_local: float) -> float: ...

    @overload
    def evaluate(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t_local)
        if is_scalar:
            return self._value
        assert isinstance(normalized, np.ndarray)
        return np.full(normalized.shape, self._value, dtype=float)

    def __repr__(self) -> str:
        return f"ConstantSegment(value={self._value!r}, duration={self._duration!r})"


class LinearRampSegment(WaveformSegment):
    r"""Linear ramp segment from ``start`` to ``end`` over ``duration``.

    .. math::

        y(\tau) = y_{\text{start}} + \frac{y_{\text{end}} - y_{\text{start}}}{\Delta t} \, \tau

    Parameters
    ----------
    start : float
        Starting excitation value at :math:`\tau = 0`.
    end : float
        Ending excitation value at :math:`\tau = \Delta t`.
    duration : float, optional
        Ramp duration in seconds (:math:`\Delta t > 0`). Mutually exclusive
        with ``rate``.
    rate : float, optional
        Positive ramp rate magnitude :math:`|dy/dt| > 0`. When given, duration
        is computed as :math:`|y_{\text{end}} - y_{\text{start}}| / \text{rate}`.
        Requires ``start != end``.
    """

    def __init__(
        self,
        start: float,
        end: float,
        duration: float | None = None,
        *,
        rate: float | None = None,
    ) -> None:
        self._start = _validate_finite_real(start, "start")
        self._end = _validate_finite_real(end, "end")

        if duration is not None and rate is not None:
            raise ValueError("Specify either 'duration' or 'rate', not both.")
        if duration is None and rate is None:
            raise ValueError("Must specify either 'duration' or 'rate'.")

        if rate is not None:
            rate_val = _validate_positive_finite(rate, "rate")
            span = abs(self._end - self._start)
            if span == 0.0:
                raise ValueError("Cannot infer duration from 'rate' when start == end.")
            resolved_duration = span / rate_val
        else:
            resolved_duration = _validate_positive_finite(duration, "duration")

        super().__init__(resolved_duration)
        self._slope = (self._end - self._start) / self._duration

    @property
    def start(self) -> float:
        """Starting value at ``t_local = 0``."""
        return self._start

    @property
    def end(self) -> float:
        """Ending value at ``t_local = duration``."""
        return self._end

    @property
    def slope(self) -> float:
        """Signed ramp rate ``(end - start) / duration``."""
        return self._slope

    @property
    def rate(self) -> float:
        """Unsigned ramp rate magnitude ``abs(slope)``."""
        return abs(self._slope)

    @overload
    def evaluate(self, t_local: float) -> float: ...

    @overload
    def evaluate(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t_local)
        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return self._start
            if normalized >= self._duration:
                return self._end
            return self._start + self._slope * normalized

        assert isinstance(normalized, np.ndarray)
        clamped = np.clip(normalized, 0.0, self._duration)
        values = self._start + self._slope * clamped
        values = np.where(clamped <= 0.0, self._start, values)
        values = np.where(clamped >= self._duration, self._end, values)
        return values

    def __repr__(self) -> str:
        return (
            f"LinearRampSegment(start={self._start!r}, end={self._end!r}, "
            f"duration={self._duration!r})"
        )


class SinusoidalSegment(WaveformSegment):
    r"""Sinusoidal excitation segment over :math:`\tau \in [0, \Delta t]`.

    .. math::

        y(\tau) = y_0 + A \sin(2 \pi f \tau + \phi)

    Parameters
    ----------
    offset : float
        DC offset level :math:`y_0`.
    amplitude : float
        Peak oscillation amplitude :math:`A` (must be finite; typically ``>= 0``).
    frequency : float
        Oscillation frequency :math:`f > 0` in Hz.
    duration : float
        Segment duration in seconds (:math:`\Delta t > 0`).
    phase : float, default 0.0
        Initial phase angle :math:`\phi` in radians.
    """

    def __init__(
        self,
        offset: float,
        amplitude: float,
        frequency: float,
        duration: float,
        phase: float = 0.0,
    ) -> None:
        super().__init__(duration)
        self._offset = _validate_finite_real(offset, "offset")
        self._amplitude = _validate_finite_real(amplitude, "amplitude")
        self._frequency = _validate_positive_finite(frequency, "frequency")
        self._phase = _validate_finite_real(phase, "phase")

    @property
    def offset(self) -> float:
        """DC offset level."""
        return self._offset

    @property
    def amplitude(self) -> float:
        """Oscillation amplitude."""
        return self._amplitude

    @property
    def frequency(self) -> float:
        """Oscillation frequency in Hz."""
        return self._frequency

    @property
    def phase(self) -> float:
        """Initial phase in radians."""
        return self._phase

    @overload
    def evaluate(self, t_local: float) -> float: ...

    @overload
    def evaluate(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t_local)
        omega = 2.0 * math.pi * self._frequency
        if is_scalar:
            assert isinstance(normalized, float)
            tau = min(max(normalized, 0.0), self._duration)
            return self._offset + self._amplitude * math.sin(omega * tau + self._phase)

        assert isinstance(normalized, np.ndarray)
        tau_arr = np.clip(normalized, 0.0, self._duration)
        return self._offset + self._amplitude * np.sin(omega * tau_arr + self._phase)

    def __repr__(self) -> str:
        return (
            f"SinusoidalSegment(offset={self._offset!r}, amplitude={self._amplitude!r}, "
            f"frequency={self._frequency!r}, duration={self._duration!r}, "
            f"phase={self._phase!r})"
        )


class CallableSegment(WaveformSegment):
    r"""Arbitrary user-defined callable segment :math:`y(\tau) = f(\tau)`.

    Supports both vectorized NumPy callables and scalar-only Python callables
    (automatically falling back to element-wise evaluation for arrays if a
    scalar-only callable is provided).

    Parameters
    ----------
    func : Callable[[float | np.ndarray], float | np.ndarray]
        Function evaluated on local segment time :math:`\tau \in [0, \Delta t]`.
    duration : float
        Segment duration in seconds (:math:`\Delta t > 0`).
    event_times : Sequence[float], optional
        Optional interior discontinuity timestamps in local time ``(0, duration)``.
    """

    def __init__(
        self,
        func: Callable[[float | np.ndarray], float | np.ndarray],
        duration: float,
        *,
        event_times: Sequence[float] = (),
    ) -> None:
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}.")
        super().__init__(duration)
        self._func = func
        validated_events = [
            _validate_finite_real(t_ev, "event_times entry") for t_ev in event_times
        ]
        self._event_times = _deduplicate_interior_events(
            validated_events, self._duration
        )

    @property
    def func(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Underlying segment callable."""
        return self._func

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior local event timestamps in ``(0, duration)``."""
        return self._event_times

    @overload
    def evaluate(self, t_local: float) -> float: ...

    @overload
    def evaluate(self, t_local: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray: ...

    def evaluate(self, t_local: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t_local)
        if is_scalar:
            assert isinstance(normalized, float)
            tau = min(max(normalized, 0.0), self._duration)
            val = float(self._func(tau))
            if not math.isfinite(val):
                raise ValueError(
                    f"CallableSegment returned non-finite value {val!r} at t={tau}."
                )
            return val

        assert isinstance(normalized, np.ndarray)
        tau_arr = np.clip(normalized, 0.0, self._duration)
        if tau_arr.size == 0:
            return np.empty(tau_arr.shape, dtype=float)

        try:
            raw = self._func(tau_arr)
            if np.isscalar(raw):
                result = np.full(tau_arr.shape, float(raw), dtype=float)
            else:
                result = np.asarray(raw, dtype=float)
                if result.shape != tau_arr.shape:
                    result = np.broadcast_to(result, tau_arr.shape).astype(
                        float, copy=True
                    )
        except (TypeError, ValueError):
            flat = [float(self._func(float(ti))) for ti in tau_arr.ravel()]
            result = np.asarray(flat, dtype=float).reshape(tau_arr.shape)

        if not np.all(np.isfinite(result)):
            raise ValueError("CallableSegment returned non-finite values.")
        return result

    def __repr__(self) -> str:
        func_name = getattr(self._func, "__name__", repr(self._func))
        return f"CallableSegment(func={func_name}, duration={self._duration!r})"


class CustomWaveform(BaseTechnique):
    """Composite electrochemical waveform chaining arbitrary :class:`WaveformSegment` objects.

    Aggregates total :attr:`duration` and interior :attr:`event_times` across all
    chained segments automatically. At interior segment boundaries, evaluation is
    right-continuous (``t in [t_k, t_{k+1})`` evaluates segment ``k``), while the
    final endpoint ``t == duration`` evaluates the last segment at its end time.

    Parameters
    ----------
    segments : Sequence[WaveformSegment]
        Non-empty ordered sequence of :class:`WaveformSegment` instances.
    mode : {'potentiostatic', 'galvanostatic'}, default 'potentiostatic'
        Control mode of the composite excitation signal.
    n_points : int, optional
        Default number of points for :meth:`time_grid` when neither ``n_points``
        nor ``dt`` is passed (default 1001).

    Examples
    --------
    >>> from softpotato.techniques import (
    ...     ConstantSegment,
    ...     CustomWaveform,
    ...     LinearRampSegment,
    ... )
    >>> wf = CustomWaveform([
    ...     ConstantSegment(0.0, duration=1.0),
    ...     LinearRampSegment(0.0, -0.5, duration=2.0),
    ...     ConstantSegment(-0.5, duration=1.0),
    ... ])
    >>> wf.duration
    4.0
    >>> wf.event_times
    (1.0, 3.0)
    >>> wf(2.0)
    -0.25
    """

    def __init__(
        self,
        segments: Sequence[WaveformSegment],
        *,
        mode: ControlMode = "potentiostatic",
        n_points: int | None = None,
    ) -> None:
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            raise TypeError(
                f"segments must be a sequence of WaveformSegment, got {type(segments).__name__}."
            )
        if len(segments) == 0:
            raise ValueError("segments must not be empty.")

        seg_list: list[WaveformSegment] = []
        for idx, seg in enumerate(segments):
            if not isinstance(seg, WaveformSegment):
                raise TypeError(
                    f"segments[{idx}] must be a WaveformSegment instance, "
                    f"got {type(seg).__name__}."
                )
            seg_list.append(seg)

        self._mode = _validate_mode(mode)
        self._default_n_points = (
            _validate_positive_int(n_points, "n_points", minimum=2)
            if n_points is not None
            else 1001
        )
        self._segments: tuple[WaveformSegment, ...] = tuple(seg_list)

        starts: list[float] = []
        events: list[float] = []
        cursor = 0.0
        for idx, seg in enumerate(self._segments):
            starts.append(cursor)
            if idx > 0:
                events.append(cursor)
            for local_ev in seg.event_times:
                events.append(cursor + local_ev)
            cursor += seg.duration

        self._starts = np.asarray(starts, dtype=float)
        self._duration = float(cursor)
        self._event_times = _deduplicate_interior_events(events, self._duration)

    @property
    def segments(self) -> tuple[WaveformSegment, ...]:
        """Ordered tuple of constituent :class:`WaveformSegment` objects."""
        return self._segments

    @property
    def segment_boundaries(self) -> tuple[float, ...]:
        """Timestamps of all segment boundaries including ``0.0`` and :attr:`duration`."""
        return tuple(float(x) for x in self._starts) + (self._duration,)

    @property
    def duration(self) -> float:
        """Total composite waveform duration in seconds."""
        return self._duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior segment boundaries and segment-internal event times."""
        return self._event_times

    def _default_time_grid_points(self) -> int:
        return self._default_n_points

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        normalized, is_scalar = _normalize_time_input(t)
        n_seg = len(self._segments)

        if is_scalar:
            assert isinstance(normalized, float)
            if normalized <= 0.0:
                return float(self._segments[0].evaluate(0.0))
            if normalized >= self._duration:
                last = self._segments[-1]
                return float(last.evaluate(last.duration))
            idx = int(np.searchsorted(self._starts, normalized, side="right")) - 1
            idx = min(max(idx, 0), n_seg - 1)
            seg = self._segments[idx]
            t_local = min(max(normalized - float(self._starts[idx]), 0.0), seg.duration)
            return float(seg.evaluate(t_local))

        assert isinstance(normalized, np.ndarray)
        orig_shape = normalized.shape
        flat = normalized.ravel()
        if flat.size == 0:
            return np.empty(orig_shape, dtype=float)

        clamped = np.clip(flat, 0.0, self._duration)
        indices = np.searchsorted(self._starts, clamped, side="right") - 1
        indices = np.clip(indices, 0, n_seg - 1)
        # Ensure exact final endpoint evaluates at the end of the last segment
        at_end = clamped >= self._duration
        indices = np.where(at_end, n_seg - 1, indices)

        out = np.empty(flat.shape, dtype=float)
        for seg_idx, seg in enumerate(self._segments):
            mask = indices == seg_idx
            if not np.any(mask):
                continue
            t_local_arr = np.clip(
                clamped[mask] - self._starts[seg_idx], 0.0, seg.duration
            )
            if seg_idx == n_seg - 1:
                t_local_arr = np.where(at_end[mask], seg.duration, t_local_arr)
            out[mask] = seg.evaluate(t_local_arr)

        return out.reshape(orig_shape)

    def __repr__(self) -> str:
        return (
            f"CustomWaveform(n_segments={len(self._segments)}, "
            f"duration={self._duration!r}, mode={self._mode!r})"
        )
