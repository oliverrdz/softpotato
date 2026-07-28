"""
Module: src.softpotato.core.pde_builder
Responsibility: Formulates second-order spatial derivative matrices and reaction terms
               for the coupled PDE system:
               ∂c_i/∂t = D_i * (∂²c_i/∂x²) + R_i(c, t)
"""

import numpy as np

from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism


class PDEAssembler:
    """
    Assembler for system partial differential equations (PDEs) in 1D simulation domains.
    Computes spatial diffusion operators using finite differences and integrates
    homogeneous reaction kinetics.
    """

    def __init__(self, grid: Grid, mechanism: Mechanism) -> None:
        """
        Initializes the PDEAssembler with a spatial grid and a composite mechanism.

        Parameters:
        -----------
        grid : Grid
            The 1D spatial grid containing node coordinates (x) and intervals (dx).
        mechanism : Mechanism
            The composite reaction mechanism defining species and pathways.
        """
        self.grid = grid
        self.mechanism = mechanism
        self.n_nodes = len(grid.x)

    def build_diffusion_matrix(self, diffusion_coefficient: float) -> np.ndarray:
        """
        Constructs a vectorized second-order spatial finite-difference matrix
        for 1D Fickian diffusion on the given grid.

        Parameters:
        -----------
        diffusion_coefficient : float
            The diffusion coefficient (D) for a specific chemical species (m²/s).

        Returns:
        --------
        np.ndarray
            A 2D square matrix of shape (n_nodes, n_nodes) representing the
            discrete diffusion operator.
        """
        dx = self.grid.dx
        n = self.n_nodes
        D = diffusion_coefficient

        # Initialize sparse-like matrix for 1D second derivative (central difference approximation)
        # For non-uniform expanding grids, dx can be an array or scalar step vector.
        D_matrix = np.zeros((n, n))

        if isinstance(dx, np.ndarray) and len(dx) > 1:
            # Non-uniform grid finite difference stencil for second derivatives
            # Using conservative central weighting across adjacent intervals
            dx_forward = dx[:-1]
            dx_backward = dx[1:]

            # Constructing sub-diagonals and main diagonal via vector slicing
            lower = 2.0 / (dx_forward * (dx_forward + dx_backward))
            upper = 2.0 / (dx_backward * (dx_forward + dx_backward))
            main = -(lower + upper)

            np.fill_diagonal(D_matrix[1:-1, :-2], lower)
            np.fill_diagonal(D_matrix[1:-1, 1:-1], main)
            np.fill_diagonal(D_matrix[1:-1, 2:], upper)
        else:
            # Uniform grid spacing fallback
            h = dx if isinstance(dx, (int, float)) else dx[0]
            scalar = D / (h**2)
            main = -2.0 * scalar
            off_diag = 1.0 * scalar

            np.fill_diagonal(D_matrix[1:-1, :-2], off_diag)
            np.fill_diagonal(D_matrix[1:-1, 1:-1], main)
            np.fill_diagonal(D_matrix[1:-1, 2:], off_diag)

        # Scale by diffusion coefficient D
        return D * D_matrix

    def evaluate_reaction_terms(
        self, concentrations: dict[str, np.ndarray], rate_constants: dict[str, float]
    ) -> dict[str, np.ndarray]:
        """
        Evaluates the homogeneous reaction rate terms R_i(c, t) across all spatial nodes
        for each participating chemical species without internal Python loops.

        Parameters:
        -----------
        concentrations : Dict[str, np.ndarray]
            Current concentration profiles c(x) for each species.
        rate_constants : Dict[str, float]
            Forward and backward rate constants for chemical steps.

        Returns:
        --------
        Dict[str, np.ndarray]
            Net reaction rate contributions (R_i) for each species at every spatial node.
        """
        # Placeholder for vectorized homogeneous kinetic rate evaluations
        reaction_rates = {
            species: np.zeros(self.n_nodes) for species in self.mechanism.get_species()
        }

        # Vectorized reaction rate calculations for chemical steps (C steps) can be
        # injected here dynamically based on mechanism definitions.
        return reaction_rates
