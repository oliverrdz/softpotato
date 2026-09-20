"""
Chemical and electroactive species definition with strict CGS unit enforcement.
"""

import numpy as np


class Species:
    """
    Defines an electroactive or purely chemical species for simulation.

    Enforces CGS unit storage across all attributes:
    - Diffusion coefficient (D) in cm²/s.
    - Bulk concentration (c_bulk) in mol/cm³ (1 mM = 1e-6 mol/cm³).
    - Spatial concentration profile (c_profile) in mol/cm³.
    """

    def __init__(
        self,
        name: str,
        D: float,
        c_bulk: float = 0.0,
        charge: int = 0,
    ) -> None:
        """
        Initialize a chemical species with CGS units.

        Args:
            name: String identifier for the species (e.g., 'O', 'R', 'Cat').
            D: Diffusion coefficient in cm²/s (must be non-negative).
            c_bulk: Initial bulk concentration in mol/cm³ (must be non-negative).
                Defaults to 0.0. Note: 1 mM = 1e-6 mol/cm³.
            charge: Integer ionic charge (z). Defaults to 0. Required if migration
                (Nernst-Planck) or double-layer effects are included.

        Raises:
            ValueError: If name is empty, or D < 0, or c_bulk < 0.
            TypeError: If types of arguments are invalid.
        """
        self.name = name
        self.D = D
        self.c_bulk = c_bulk
        self.charge = charge

        # Spatial concentration array (mol/cm³).
        # Left uninitialized until the grid geometry is defined by the solver.
        self._c_profile: np.ndarray | None = None

    @property
    def name(self) -> str:
        """String identifier of the species."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Species name must be a non-empty string.")
        self._name = value.strip()

    @property
    def D(self) -> float:
        """Diffusion coefficient in cm²/s."""
        return self._D

    @D.setter
    def D(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Diffusion coefficient D must be a real number, got {type(value).__name__}."
            )
        if value < 0:
            raise ValueError(
                f"Diffusion coefficient D must be non-negative (cm²/s), got {value}."
            )
        self._D = float(value)

    @property
    def c_bulk(self) -> float:
        """Initial bulk concentration in mol/cm³."""
        return self._c_bulk

    @c_bulk.setter
    def c_bulk(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Bulk concentration c_bulk must be a real number, got {type(value).__name__}."
            )
        if value < 0:
            raise ValueError(
                f"Bulk concentration c_bulk must be non-negative (mol/cm³), got {value}."
            )
        self._c_bulk = float(value)

    @property
    def charge(self) -> int:
        """Ionic charge / valence number (z)."""
        return self._charge

    @charge.setter
    def charge(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Charge must be an integer, got {type(value).__name__}.")
        self._charge = int(value)

    @property
    def c_profile(self) -> np.ndarray | None:
        """Spatial concentration profile array across grid nodes in mol/cm³."""
        return self._c_profile

    @c_profile.setter
    def c_profile(self, value: np.ndarray | None) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError("c_profile must be a numpy ndarray or None.")
            if value.ndim != 1:
                raise ValueError(f"c_profile must be a 1D array, got {value.ndim}D.")
        self._c_profile = value

    def initialize_profile(self, n_nodes: int) -> None:
        """
        Allocates the contiguous 1D numpy array for the finite difference solver.

        Args:
            n_nodes: Number of spatial grid nodes (must be > 0).

        Raises:
            ValueError: If n_nodes <= 0.
            TypeError: If n_nodes is not an integer.
        """
        if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
            raise TypeError(
                f"n_nodes must be an integer, got {type(n_nodes).__name__}."
            )
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be greater than 0, got {n_nodes}.")
        self._c_profile = np.full(n_nodes, self._c_bulk, dtype=np.float64)

    def reset(self) -> None:
        """Resets the concentration profile to bulk conditions between simulation runs."""
        if self._c_profile is not None:
            self._c_profile.fill(self._c_bulk)

    def __repr__(self) -> str:
        return (
            f"Species(name='{self.name}', D={self.D:.2e} cm²/s, "
            f"c_bulk={self.c_bulk:.2e} mol/cm³, charge={self.charge})"
        )

    def __str__(self) -> str:
        return (
            f"Species {self.name} (D={self.D:.2e} cm²/s, "
            f"c_bulk={self.c_bulk:.2e} mol/cm³, z={self.charge})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Species):
            return False
        return (
            self.name == other.name
            and self.D == other.D
            and self.c_bulk == other.c_bulk
            and self.charge == other.charge
        )
