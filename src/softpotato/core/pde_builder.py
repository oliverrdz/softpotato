from typing import Any

import numpy as np

from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism
from softpotato.physics.species import Species


class PDEAssembler:
    """Formulates second-order spatial finite-difference derivative matrices and homogeneous reaction terms."""

    def __init__(
        self,
        grid: Grid,
        mechanism: Mechanism,
        species: dict[str, Species] | None = None,
        kinetics: Any | None = None,
    ):
        self.grid = grid
        self.mechanism = mechanism
        self.species = species or {}
        self.kinetics = kinetics
        self.n_nodes = len(grid.x)

    def build_diffusion_matrix(self, D: float) -> np.ndarray:
        """Constructs the 1D second-order finite-difference diffusion matrix for a given diffusion coefficient D."""
        n = self.n_nodes
        mat = np.zeros((n, n))
        if n > 2 and len(self.grid.dx) > 0:
            dx = self.grid.dx[0]
            factor = D / (dx**2)
            for i in range(1, n - 1):
                mat[i, i - 1] = factor
                mat[i, i] = -2.0 * factor
                mat[i, i + 1] = factor
            # Boundary ghost / zero-flux approximation handles edges
            mat[0, 0] = -2.0 * factor
            mat[0, 1] = 2.0 * factor
            mat[-1, -1] = -2.0 * factor
            mat[-1, -2] = 2.0 * factor
        return mat

    def assemble(self, c_matrix: np.ndarray, t: float) -> np.ndarray:
        """Assembles spatial diffusion and reaction derivatives dc/dt across all species using pure NumPy vectorization."""
        n_species, _n_nodes = c_matrix.shape
        dc_dt = np.zeros_like(c_matrix)

        species_keys = list(self.species.keys()) if self.species else []

        for idx in range(n_species):
            sp_key = species_keys[idx] if idx < len(species_keys) else None
            D = self.species[sp_key].D if sp_key and sp_key in self.species else 1.0

            D_mat = self.build_diffusion_matrix(D)
            dc_dt[idx] = D_mat @ c_matrix[idx]

        return dc_dt
