from abc import ABC, abstractmethod
from typing import Any, Dict, List

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
    """Abstract Base Class for physical and transport models in Soft Potato."""

    @property
    @abstractmethod
    def species_names(self) -> List[str]:
        """Return the list of species identifiers managed by this model."""
        pass

    @property
    @abstractmethod
    def num_species(self) -> int:
        """Return the total number of species in the transport system."""
        pass

    @abstractmethod
    def get_diffusion_coefficients(self) -> Dict[str, float]:
        """Return a mapping of species names to their diffusion coefficients ($m^2/s$)."""
        pass

    @abstractmethod
    def get_initial_conditions(self, x_grid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate initial concentration distributions $C(x, t=0)$ across spatial grid coordinates.
        
        Parameters
        ----------
        x_grid : np.ndarray
            1D array of spatial node coordinates ($m$).
            
        Returns
        -------
        Dict[str, np.ndarray]
            Mapping of species names to concentration arrays ($mol/m^3$).
        """
        pass


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
