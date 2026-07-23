from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from softpotato.core.abcs import BaseMesh


class Uniform1DMesh(BaseMesh, BaseModel):
    """Uniform 1D spatial grid generator for finite difference discretizations.

    Inherits from :class:`BaseMesh` and utilizes Pydantic for input validation.

    Parameters
    ----------
    x_min : float
        Lower boundary of the spatial domain.
    x_max : float
        Upper boundary of the spatial domain. Must be strictly greater than `x_min`.
    n_points : int
        Total number of grid nodes. Must be at least 2.
    """

    x_min: float = Field(..., description="Lower boundary of the spatial domain")
    x_max: float = Field(..., description="Upper boundary of the spatial domain")
    n_points: int = Field(..., ge=2, description="Total number of spatial grid nodes")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, x_min: float, x_max: float, n_points: int, **data: Any) -> None:
        super().__init__(x_min=x_min, x_max=x_max, n_points=n_points, **data)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "Uniform1DMesh":
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be strictly greater than x_min ({self.x_min})."
            )
        return self

    def get_nodes(self) -> np.ndarray:
        """Return 1D NumPy array of spatial node coordinates.

        Returns
        -------
        np.ndarray
            Uniformly spaced 1D array from `x_min` to `x_max`.
        """
        return np.linspace(self.x_min, self.x_max, self.n_points)

    def num_nodes(self) -> int:
        """Return total number of spatial nodes.

        Returns
        -------
        int
            Number of spatial nodes `n_points`.
        """
        return self.n_points

    @property
    def L(self) -> float:
        """Total domain length $L = x_{\\text{max}} - x_{\\text{min}}$."""
        return float(self.x_max - self.x_min)

    @property
    def dx(self) -> float:
        """Uniform grid spacing $\Delta x = \\frac{L}{n_{\\text{points}} - 1}$."""
        return float(self.L / (self.n_points - 1))
