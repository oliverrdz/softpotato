"""Base data structures for experimental technique waveform generators."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Waveform:
    """Container storing continuous potential signal time-series arrays.

    Attributes
    ----------
    t : np.ndarray
        1D array of time coordinates in seconds (s).
    E : np.ndarray
        1D array of applied potential values in Volts (V).
    """

    t: np.ndarray
    E: np.ndarray

    def __post_init__(self) -> None:
        """Validate array dimensions and alignment."""
        if not isinstance(self.t, np.ndarray) or not isinstance(self.E, np.ndarray):
            raise TypeError("t and E must be NumPy arrays.")
        if self.t.ndim != 1 or self.E.ndim != 1:
            raise ValueError("t and E must be 1D arrays.")
        if len(self.t) != len(self.E):
            raise ValueError("t and E must have the same length.")