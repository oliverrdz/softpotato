"""
Voltammetric techniques and waveform generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Technique(ABC):
    """
    Abstract base class for electrochemical excitation techniques.
    """

    @property
    @abstractmethod
    def time(self) -> np.ndarray:
        """1D array of time points in seconds."""

    @property
    @abstractmethod
    def potential(self) -> np.ndarray:
        """1D array of electrode potentials in Volts."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Time step between potential points in seconds."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """Total duration of the experiment in seconds."""


class CyclicVoltammetry(Technique):
    """
    Cyclic Voltammetry (CV) excitation waveform generator.

    Enforces strict CGS units (scan rate in V/s, potential in V, time in s).
    """

    def __init__(
        self,
        E_initial: float,
        E_vertex1: float,
        scan_rate: float,
        n_sweeps: int = 2,
        dE: float = 0.001,
        E_vertex2: float | None = None,
    ) -> None:
        """
        Initialize a Cyclic Voltammetry waveform.

        Parameters
        ----------
        E_initial : float
            Initial potential in Volts.
        E_vertex1 : float
            First vertex (switching) potential in Volts.
        scan_rate : float
            Potential scan rate in V/s (:math:`\\nu > 0`).
        n_sweeps : int, optional
            Number of linear sweeps (half-cycles), by default 2 (one full cycle).
        dE : float, optional
            Potential step increment in Volts (:math:`dE > 0`), by default 0.001 (1 mV).
        E_vertex2 : float or None, optional
            Second vertex potential in Volts. If None, defaults to `E_initial`.

        Raises
        ------
        TypeError
            If parameter types are invalid.
        ValueError
            If `scan_rate <= 0`, `n_sweeps < 1`, `dE <= 0`, or if vertex potentials are identical.
        """
        if isinstance(E_initial, bool) or not isinstance(E_initial, (int, float)):
            raise TypeError(
                f"E_initial must be a real number, got {type(E_initial).__name__}."
            )
        if isinstance(E_vertex1, bool) or not isinstance(E_vertex1, (int, float)):
            raise TypeError(
                f"E_vertex1 must be a real number, got {type(E_vertex1).__name__}."
            )

        if isinstance(scan_rate, bool) or not isinstance(scan_rate, (int, float)):
            raise TypeError(
                f"scan_rate must be a real number, got {type(scan_rate).__name__}."
            )
        if scan_rate <= 0:
            raise ValueError(f"scan_rate must be positive, got {scan_rate}.")

        if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, int):
            raise TypeError(
                f"n_sweeps must be an integer, got {type(n_sweeps).__name__}."
            )
        if n_sweeps < 1:
            raise ValueError(f"n_sweeps must be at least 1, got {n_sweeps}.")

        if isinstance(dE, bool) or not isinstance(dE, (int, float)):
            raise TypeError(f"dE must be a real number, got {type(dE).__name__}.")
        if dE <= 0:
            raise ValueError(f"dE must be positive, got {dE}.")

        self._E_initial = float(E_initial)
        self._E_vertex1 = float(E_vertex1)
        self._scan_rate = float(scan_rate)
        self._n_sweeps = int(n_sweeps)
        self._dE = float(dE)

        if E_vertex2 is not None:
            if isinstance(E_vertex2, bool) or not isinstance(E_vertex2, (int, float)):
                raise TypeError(
                    f"E_vertex2 must be a real number, got {type(E_vertex2).__name__}."
                )
            self._E_vertex2 = float(E_vertex2)
        else:
            self._E_vertex2 = self._E_initial

        if np.isclose(self._E_initial, self._E_vertex1):
            raise ValueError("E_initial and E_vertex1 must be distinct potentials.")

        # Construct segments
        # Sweep 1: E_initial -> E_vertex1
        # Sweep 2: E_vertex1 -> E_vertex2
        # Sweep 3: E_vertex2 -> E_vertex1
        # Sweep 4: E_vertex1 -> E_vertex2, etc.
        segments: list[np.ndarray] = []
        for s in range(self._n_sweeps):
            if s == 0:
                e_start, e_end = self._E_initial, self._E_vertex1
            elif s % 2 == 1:
                e_start, e_end = self._E_vertex1, self._E_vertex2
            else:
                e_start, e_end = self._E_vertex2, self._E_vertex1

            span = abs(e_end - e_start)
            n_pts = max(int(np.round(span / self._dE)), 1)
            seg = np.linspace(e_start, e_end, n_pts + 1, dtype=np.float64)
            if s == 0:
                segments.append(seg)
            else:
                segments.append(seg[1:])  # Avoid repeating junction point

        self._potential = np.concatenate(segments)
        self._dt = self._dE / self._scan_rate
        self._time = np.arange(len(self._potential), dtype=np.float64) * self._dt

    @property
    def E_initial(self) -> float:
        """Initial potential in Volts."""
        return self._E_initial

    @property
    def E_vertex1(self) -> float:
        """First vertex potential in Volts."""
        return self._E_vertex1

    @property
    def E_vertex2(self) -> float:
        """Second vertex potential in Volts."""
        return self._E_vertex2

    @property
    def scan_rate(self) -> float:
        """Scan rate in V/s."""
        return self._scan_rate

    @property
    def n_sweeps(self) -> int:
        """Number of linear sweeps."""
        return self._n_sweeps

    @property
    def dE(self) -> float:
        """Potential step increment in Volts."""
        return self._dE

    @property
    def dt(self) -> float:
        """Time step between samples in seconds."""
        return self._dt

    @property
    def time(self) -> np.ndarray:
        """1D array of time points in seconds."""
        return self._time.copy()

    @property
    def potential(self) -> np.ndarray:
        """1D array of electrode potentials in Volts."""
        return self._potential.copy()

    @property
    def duration(self) -> float:
        """Total duration of the experiment in seconds."""
        return float(self._time[-1]) if len(self._time) > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CyclicVoltammetry(E_initial={self._E_initial}, E_vertex1={self._E_vertex1}, "
            f"scan_rate={self._scan_rate}, n_sweeps={self._n_sweeps}, dE={self._dE})"
        )


def linear_sweep(*args: Any, **kwargs: Any) -> Any:
    """
    Simulate a linear sweep voltammetry experiment.

    .. warning::
        Planned for v3.1.0.
    """
    raise NotImplementedError("Linear sweep voltammetry is planned for v3.1.0.")
