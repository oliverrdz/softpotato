from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
import numpy as np


class BaseMesh(ABC):
    """Abstract interface for 1D/2D/3D spatial grids."""

    @abstractmethod
    def get_nodes(self) -> np.ndarray:
        """Return coordinate array of spatial grid nodes."""
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        """Return total number of spatial nodes."""
        pass


class BaseBoundaryCondition(ABC):
    """Abstract interface for boundary fluxes or concentrations."""

    @abstractmethod
    def apply(self, matrix: Any, rhs: np.ndarray, mesh: BaseMesh, t: float) -> Tuple[Any, np.ndarray]:
        """Modify system matrices or RHS vector according to boundary physics."""
        pass


class BaseModel(ABC):
    """Abstract interface for symbolic PDE physics and species definitions."""

    @abstractmethod
    def get_species(self) -> Dict[str, Any]:
        """Return defined chemical species and transport coefficients."""
        pass


class BaseDiscretizer(ABC):
    """Abstract interface for converting symbolic equations into system matrices."""

    @abstractmethod
    def assemble(self, model: BaseModel, mesh: BaseMesh) -> Tuple[Any, np.ndarray]:
        """Build spatial differential operator matrices."""
        pass


class BaseSolver(ABC):
    """Abstract interface for time integration engines."""

    @abstractmethod
    def solve(self, rhs_fn: Any, y0: np.ndarray, t_span: Tuple[float, float]) -> Any:
        """Step the discrete ODE/DAE system forward in time."""
        pass
