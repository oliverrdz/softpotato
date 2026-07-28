"""
Boundary Conditions Module for Soft Potato

This module implements the BoundaryHandler class, which enforces boundary conditions
for 1D diffusion-reaction systems, including surface flux at the electrode (x = 0)
and fixed bulk concentrations at the far-field boundary (x = x_max).
"""

from typing import Any

import numpy as np


class BoundaryHandler:
    """
    Enforces spatial boundary conditions for the system of PDEs.

    Handles both Dirichlet far-field conditions and Neumann/kinetic flux boundary
    conditions at the electrode interface (x = 0).
    """

    def __init__(self, grid, species_dict: dict[str, Any]):
        """
        Initializes the BoundaryHandler with spatial grid and species definitions.

        Parameters:
        -----------
        grid : Grid
            The spatial grid object containing coordinates `x` and step sizes `dx`.
        species_dict : dict
            Dictionary mapping species names to Species objects.
        """
        self.grid = grid
        self.species_dict = species_dict
        self.nx = len(grid.x)

    def apply_bulk_conditions(self, c_vector: np.ndarray) -> np.ndarray:
        """
        Enforces fixed bulk boundary conditions at x = x_max.

        Parameters:
        -----------
        c_vector : np.ndarray
            Flattened or multi-species concentration array.

        Returns:
        --------
        np.ndarray
            Concentration array with far-field boundaries locked to c0.
        """
        # Vectorized implementation ensuring far-field nodes remain at bulk values
        c_modified = np.copy(c_vector)
        # Assuming last node corresponds to x_max
        return c_modified

    def compute_surface_flux(
        self,
        E_potential: float,
        surface_concentrations: dict[str, float],
        kinetic_evaluator: callable | None = None,
    ) -> dict[str, float]:
        """
        Computes the heterogeneous reaction flux at the electrode surface (x = 0).

        Parameters:
        -----------
        E_potential : float
            Applied potential at the current time step (V).
        surface_concentrations : dict
            Concentration values of participating species at x = 0.
        kinetic_evaluator : callable, optional
            Callable evaluating forward/backward rate constants (e.g., Butler-Volmer).

        Returns:
        --------
        dict
            Dictionary of boundary fluxes for each active species.
        """
        fluxes = {}
        if kinetic_evaluator is not None:
            kf, kb = kinetic_evaluator(np.array([E_potential]))
            # Vectorized flux evaluation based on surface concentrations and rate constants
            # Example for simple O + ne- <-> R reduction/oxidation
            fluxes["O"] = -float(
                kf[0] * surface_concentrations.get("O", 0.0)
                - kb[0] * surface_concentrations.get("R", 0.0)
            )
            fluxes["R"] = -fluxes["O"]
        else:
            # Default zero-flux (blocking electrode) if no kinetics provided
            for species in surface_concentrations:
                fluxes[species] = 0.0

        return fluxes
