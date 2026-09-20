"""
Electrode geometry representations and spatial differential operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .grids import Grid


class Electrode(ABC):
    """
    Abstract base class for electrode geometries.
    """

    @property
    @abstractmethod
    def area(self) -> float:
        """Electrode geometric surface area in cm²."""

    @property
    @abstractmethod
    def grid(self) -> Grid:
        """Spatial discretization grid associated with this electrode."""

    @abstractmethod
    def laplacian(self, profile: np.ndarray) -> np.ndarray:
        """
        Compute the spatial Laplacian (second derivative) for diffusion.

        Parameters
        ----------
        profile : np.ndarray
            1D array of concentration values across grid nodes in mol/cm³.

        Returns
        -------
        np.ndarray
            1D array representing the spatial Laplacian at interior nodes.
        """


class PlanarElectrode(Electrode):
    """
    Planar electrode geometry for 1D semi-infinite diffusion.

    Enforces strict CGS units (area in cm²).
    """

    def __init__(self, area: float, grid: Grid) -> None:
        """
        Initialize a planar electrode.

        Parameters
        ----------
        area : float
            Electrode geometric area in cm² (:math:`A > 0`).
        grid : Grid
            Spatial discretization grid.

        Raises
        ------
        TypeError
            If area is not a real number or grid is not an instance of Grid.
        ValueError
            If `area <= 0`.
        """
        if isinstance(area, bool) or not isinstance(area, (int, float)):
            raise TypeError(f"area must be a real number, got {type(area).__name__}.")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}.")

        if not isinstance(grid, Grid):
            raise TypeError(f"grid must be a Grid instance, got {type(grid).__name__}.")

        self._area = float(area)
        self._grid = grid

    @property
    def area(self) -> float:
        """Electrode geometric surface area in cm²."""
        return self._area

    @property
    def grid(self) -> Grid:
        """Spatial discretization grid associated with this electrode."""
        return self._grid

    def laplacian(self, profile: np.ndarray) -> np.ndarray:
        """
        Compute the 1D planar spatial Laplacian: :math:`\\frac{\\partial^2 c}{\\partial x^2}`.

        Uses a vectorized 3-point central difference stencil:
        :math:`\\frac{c_{i+1} - 2c_i + c_{i-1}}{\\Delta x^2}` for interior nodes.

        Parameters
        ----------
        profile : np.ndarray
            1D array of concentration values across grid nodes.

        Returns
        -------
        np.ndarray
            Laplacian values for interior nodes :math:`1 \\le i < N - 1`.
        """
        dx = self._grid.dx
        if isinstance(dx, np.ndarray):
            raise NotImplementedError(
                "Non-uniform grid Laplacian is planned for v3.3.0."
            )
        return (profile[2:] - 2.0 * profile[1:-1] + profile[:-2]) / (dx**2)

    def __repr__(self) -> str:
        return f"PlanarElectrode(area={self._area}, grid={self._grid!r})"


class SphericalElectrode(Electrode):
    """
    Spherical electrode geometry representation.

    .. warning::
        Full implementation planned for v3.3.0.
    """

    def __init__(self, radius: float, grid: Grid) -> None:
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise TypeError(
                f"radius must be a real number, got {type(radius).__name__}."
            )
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}.")
        self._radius = float(radius)
        self._grid = grid
        self._area = 4.0 * np.pi * (self._radius**2)

    @property
    def area(self) -> float:
        return self._area

    @property
    def grid(self) -> Grid:
        return self._grid

    def laplacian(self, profile: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Spherical diffusion Laplacian is planned for v3.3.0."
        )


class CylindricalElectrode(Electrode):
    """
    Cylindrical electrode geometry representation.

    .. warning::
        Full implementation planned for v3.3.0.
    """

    def __init__(self, radius: float, length: float, grid: Grid) -> None:
        self._radius = float(radius)
        self._length = float(length)
        self._grid = grid
        self._area = 2.0 * np.pi * self._radius * self._length

    @property
    def area(self) -> float:
        return self._area

    @property
    def grid(self) -> Grid:
        return self._grid

    def laplacian(self, profile: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Cylindrical diffusion Laplacian is planned for v3.3.0."
        )
