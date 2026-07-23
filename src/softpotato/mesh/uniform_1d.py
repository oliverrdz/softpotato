from typing import Any
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field, model_validator

from softpotato.core.abcs import BaseMesh


class _CallableInt(int):
    """Integer subclass supporting both property access (`.num_nodes`) and method calls (`.num_nodes()`)."""

    def __call__(self) -> int:
        return int(self)


class Uniform1DMesh(PydanticBaseModel, BaseMesh):
    """
    Uniform 1D Spatial Mesh implementation.

    Decouples spatial grid generation from numerical solvers while fulfilling
    the `BaseMesh` interface with strict Pydantic validation.
    """

    x_min: float = Field(..., description="Start coordinate of 1D domain (m)")
    x_max: float = Field(..., description="End coordinate of 1D domain (m)")
    n_nodes: int = Field(
        ..., alias="n_points", ge=2, description="Number of spatial grid nodes"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }

    def __init__(
        self,
        x_min: float = ...,
        x_max: float = ...,
        num_nodes: int | None = None,
        **data: Any,
    ) -> None:
        """Allow positional args (x_min, x_max, num_nodes/n_points) and keyword aliases."""
        if num_nodes is not None:
            data["n_nodes"] = num_nodes
        elif "num_nodes" in data:
            data["n_nodes"] = data.pop("num_nodes")

        super().__init__(x_min=x_min, x_max=x_max, **data)

    @model_validator(mode="after")
    def _validate_domain_bounds(self) -> "Uniform1DMesh":
        """Verify upper domain bound strictly exceeds lower domain bound."""
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be strictly greater than x_min ({self.x_min})."
            )
        return self

    @property
    def L(self) -> float:
        """Total domain length (m)."""
        return float(self.x_max - self.x_min)

    @property
    def num_nodes(self) -> _CallableInt:
        """Total number of spatial grid nodes."""
        return _CallableInt(self.n_nodes)

    @property
    def n_points(self) -> int:
        """Alias for number of spatial grid nodes."""
        return self.n_nodes

    @property
    def x(self) -> np.ndarray:
        """1D array of spatial node coordinates (m)."""
        return np.linspace(self.x_min, self.x_max, self.n_nodes)

    @property
    def dx(self) -> float:
        """Uniform grid spacing (m)."""
        return float((self.x_max - self.x_min) / (self.n_nodes - 1))

    def get_nodes(self) -> np.ndarray:
        """Return array of spatial node coordinates (m)."""
        return self.x
