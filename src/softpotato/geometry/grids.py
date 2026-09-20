"""
Spatial grid representations and generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Grid(ABC):
    """
    Abstract base class for spatial discretization grids.
    """

    @property
    @abstractmethod
    def nodes(self) -> int:
        """Total number of spatial grid nodes."""

    @property
    @abstractmethod
    def x_max(self) -> float:
        """Maximum spatial domain boundary in cm."""

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """1D array of spatial node coordinates in cm."""

    @property
    @abstractmethod
    def dx(self) -> float | np.ndarray:
        """Spatial node spacing in cm."""


class UniformGrid(Grid):
    """
    Uniformly spaced 1D spatial grid for finite difference simulations.

    All spatial dimensions enforce strict CGS units (cm).
    """

    def __init__(self, x_max: float, nodes: int) -> None:
        """
        Initialize a uniform 1D spatial grid.

        Parameters
        ----------
        x_max : float
            Maximum distance from the electrode surface in cm (:math:`x_{\\max} > 0`).
        nodes : int
            Number of spatial nodes (:math:`N \\ge 2`).

        Raises
        ------
        TypeError
            If parameters have invalid types.
        ValueError
            If `x_max <= 0` or `nodes < 2`.
        """
        if isinstance(x_max, bool) or not isinstance(x_max, (int, float)):
            raise TypeError(f"x_max must be a real number, got {type(x_max).__name__}.")
        if x_max <= 0:
            raise ValueError(f"x_max must be positive, got {x_max}.")

        if isinstance(nodes, bool) or not isinstance(nodes, int):
            raise TypeError(f"nodes must be an integer, got {type(nodes).__name__}.")
        if nodes < 2:
            raise ValueError(f"nodes must be at least 2, got {nodes}.")

        self._x_max = float(x_max)
        self._nodes = int(nodes)
        self._dx = self._x_max / (self._nodes - 1)
        self._x = np.linspace(0.0, self._x_max, self._nodes, dtype=np.float64)

    @property
    def nodes(self) -> int:
        """Total number of spatial grid nodes."""
        return self._nodes

    @property
    def x_max(self) -> float:
        """Maximum spatial domain boundary in cm."""
        return self._x_max

    @property
    def dx(self) -> float:
        """Uniform spatial node spacing in cm."""
        return self._dx

    @property
    def x(self) -> np.ndarray:
        """1D array of spatial node coordinates in cm."""
        return self._x.copy()

    def __repr__(self) -> str:
        return (
            f"UniformGrid(x_max={self._x_max}, nodes={self._nodes}, dx={self._dx:.4e})"
        )


def generate_exponential_grid(*args: Any, **kwargs: Any) -> Any:
    """
    Generate an exponentially expanding spatial grid.

    .. warning::
        This function is not implemented yet and raises :exc:`NotImplementedError`.

    .. todo::
        Implementation planned for v3.3.0.
    """
    raise NotImplementedError("Exponential expanding grids are planned for v3.3.0.")
