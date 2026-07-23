from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp


class BaseMesh(ABC):
    """Abstract Base Class for spatial grids."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Total number of spatial nodes."""

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """1D array of node coordinates."""

    @property
    @abstractmethod
    def dx(self) -> float | np.ndarray:
        """Grid spacing (step size)."""


class BaseDiscretizer(ABC):
    """Abstract Base Class for spatial discretization operators."""

    @abstractmethod
    def build_laplacian_matrix(self, mesh: BaseMesh) -> sp.csc_matrix:
        """
        Construct second-order spatial derivative operator matrix for 1D diffusion.

        Parameters
        ----------
        mesh : BaseMesh
            Spatial mesh definition.

        Returns
        -------
        sp.csc_matrix
            Sparse spatial derivative operator matrix (N x N).
        """

    @abstractmethod
    def build_system_matrix(self, mesh: BaseMesh, model: BaseModel) -> sp.csc_matrix:
        """
        Construct block-diagonal system operator matrix combining all species.

        Parameters
        ----------
        mesh : BaseMesh
            Spatial mesh definition.
        model : BaseModel
            Species and transport model containing diffusion coefficients.

        Returns
        -------
        sp.csc_matrix
            Sparse system matrix of size (num_species * N, num_species * N).
        """


class BaseModel(ABC):
    """Abstract Base Class for physical and transport models in Soft Potato."""

    @property
    @abstractmethod
    def species_names(self) -> list[str]:
        """Return the list of species identifiers managed by this model."""

    @property
    @abstractmethod
    def num_species(self) -> int:
        """Return the total number of species in the transport system."""

    @abstractmethod
    def get_diffusion_coefficients(self) -> dict[str, float]:
        """Return a mapping of species names to their diffusion coefficients ($m^2/s$)."""

    @abstractmethod
    def get_initial_conditions(self, x_grid: np.ndarray) -> dict[str, np.ndarray]:
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


class BaseBoundaryCondition(ABC):
    """Abstract Base Class for interface flux and concentration boundary conditions."""

    @abstractmethod
    def apply(
        self, state: np.ndarray, t: float, mesh: BaseMesh, model: BaseModel
    ) -> Any:
        """Apply boundary conditions to system state or system operators."""


class BaseSolver(ABC):
    """Abstract interface for time integration engines."""

    @abstractmethod
    def solve(self, rhs_fn: Any, y0: np.ndarray, t_span: tuple[float, float]) -> Any:
        """Step the discrete ODE/DAE system forward in time."""
