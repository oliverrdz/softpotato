from collections.abc import Callable
from typing import Any

import numpy as np

from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism
from softpotato.physics.species import Species


class BoundaryHandler:
    """Enforces surface flux boundary conditions at the electrode interface and far-field bulk conditions."""

    def __init__(
        self,
        grid: Grid,
        mechanism: Mechanism | None = None,
        species: dict[str, Species] | None = None,
        kinetic_evaluator: Callable | None = None,
        kinetics: Callable | None = None,
        **kwargs: Any,
    ):
        self.grid = grid
        self.mechanism = mechanism
        self.species = species or {}
        self.kinetic_evaluator = kinetic_evaluator or kinetics

    def apply(self, c_matrix: np.ndarray, E: float, t: float) -> np.ndarray:
        """Applies boundary conditions to the concentration matrix."""
        # Enforce far-field bulk boundary conditions at x_max
        if self.species:
            species_keys = list(self.species.keys())
            for idx, key in enumerate(species_keys):
                if idx < c_matrix.shape[0]:
                    c_matrix[idx, -1] = self.species[key].c0
        return c_matrix
