from dataclasses import dataclass


@dataclass
class Species:
    """
    Dataclass representing the mass-transport properties of a chemical entity.

    Parameters
    ----------
    D : float
        Diffusion coefficient in m^2/s. Must be strictly positive (D > 0).
    c0 : float
        Initial bulk concentration in mol/m^3. Must be non-negative (c0 >= 0).

    Raises
    ------
    ValueError
        If D <= 0 or c0 < 0.
    """

    D: float
    c0: float

    def __post_init__(self):
        """Validates the physical mass-transport parameters after instantiation."""
        if self.D <= 0:
            raise ValueError(
                f"Diffusion coefficient D must be strictly positive (D > 0). Got: {self.D}"
            )
        if self.c0 < 0:
            raise ValueError(
                f"Initial bulk concentration c0 must be non-negative (c0 >= 0). Got: {self.c0}"
            )
