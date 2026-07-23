from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseMesh(ABC):
    """Abstract interface for 1D/2D/3D spatial grids."""

    @abstractmethod
    def get_nodes(self) -> np.ndarray:
        """Return coordinate array of spatial grid nodes."""

    @abstractmethod
    def num_nodes(self) -> int:
        """Return total number of spatial nodes."""


class BaseBoundaryCondition(ABC):
    """Abstract interface for boundary fluxes or concentrations."""

    @abstractmethod
    def apply(
        self, matrix: Any, rhs: np.ndarray, mesh: BaseMesh, t: float
    ) -> tuple[Any, np.ndarray]:
        """Modify system matrices or RHS vector according to boundary physics."""


class BaseModel(ABC):
    """Abstract interface for symbolic PDE physics and species definitions."""

    @abstractmethod
    def get_species(self) -> dict[str, Any]:
        """Return defined chemical species and transport coefficients."""


class BaseDiscretizer(ABC):
    """Abstract interface for converting symbolic equations into system matrices."""

    @abstractmethod
    def assemble(self, model: BaseModel, mesh: BaseMesh) -> tuple[Any, np.ndarray]:
        """Build spatial differential operator matrices."""


class BaseSolver(ABC):
    """Abstract interface for time integration engines."""

    @abstractmethod
    def solve(self, rhs_fn: Any, y0: np.ndarray, t_span: tuple[float, float]) -> Any:
        """Step the discrete ODE/DAE system forward in time."""
