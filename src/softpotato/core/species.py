import numpy as np
from typing import Optional

class Species:
    """
    Defines an electroactive or purely chemical species for simulation.
    """
    def __init__(
        self,
        name: str,
        D: float,
        c_bulk: float,
        charge: int = 0
    ) -> None:
        """
        Args:
            name: String identifier for the species (e.g., 'O', 'R', 'Cat').
            D: Diffusion coefficient in cm²/s. 
            c_bulk: Initial bulk concentration in mol/cm³.
            charge: Integer ionic charge (z). Defaults to 0. Required if migration 
                    (Nernst-Planck) or double-layer effects are added later.
        """
        self.name: str = name
        self.D: float = D
        self.c_bulk: float = c_bulk
        self.charge: int = charge
        
        # Spatial concentration array (mol/cm³). 
        # Left uninitialized until the grid geometry is defined by the solver.
        self.c_profile: Optional[np.ndarray] = None

    def initialize_profile(self, n_nodes: int) -> None:
        """
        Allocates the contiguous 1D numpy array for the finite difference solver.
        
        Args:
            n_nodes: Number of spatial grid nodes (from softpotato.geometry.grids).
        """
        # Using np.float64 for numerical stability in FDM matrix operations
        self.c_profile = np.full(n_nodes, self.c_bulk, dtype=np.float64)

    def reset(self) -> None:
        """Resets the concentration profile to bulk conditions between simulation runs."""
        if self.c_profile is not None:
            self.c_profile.fill(self.c_bulk)

    def __repr__(self) -> str:
        return f"Species({self.name}, D={self.D:.2e} cm²/s, C*={self.c_bulk:.2e} mol/cm³)"
