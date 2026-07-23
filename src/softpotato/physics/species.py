import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, NonNegativeFloat, PositiveFloat

from softpotato.core.abcs import BaseModel


class Species(PydanticBaseModel):
    """
    Data model representing an individual chemical species in solution.

    Attributes
    ----------
    name : str
        Unique identifier for the species (e.g., 'R', 'O').
    diffusion_coefficient : PositiveFloat
        Diffusion coefficient $D$ in $m^2/s$.
    bulk_concentration : NonNegativeFloat
        Initial bulk concentration $C^*$ in $mol/m^3$.
    """

    name: str = Field(..., description="Chemical species identifier")
    diffusion_coefficient: PositiveFloat = Field(
        ..., description="Diffusion coefficient D in m^2/s"
    )
    bulk_concentration: NonNegativeFloat = Field(
        default=0.0, description="Bulk/Initial concentration C* in mol/m^3"
    )


class TwoSpeciesModel(BaseModel):
    """
    Two-Species Transport Model for electroactive species $R$ (Reductant) and $O$ (Oxidant).

    Decouples physical parameters and initial conditions from numerical discretization.
    """

    def __init__(
        self,
        species_r: Species | None = None,
        species_o: Species | None = None,
        D_R: float = 1e-9,
        D_O: float = 1e-9,
        C_R_bulk: float = 1.0,
        C_O_bulk: float = 0.0,
    ) -> None:
        """
        Initialize the two-species transport model.

        Can be initialized either by passing explicit `Species` objects or by passing
        individual scalar transport coefficients.
        """
        if species_r is not None:
            self._species_r = species_r
        else:
            self._species_r = Species(
                name="R",
                diffusion_coefficient=D_R,
                bulk_concentration=C_R_bulk,
            )

        if species_o is not None:
            self._species_o = species_o
        else:
            self._species_o = Species(
                name="O",
                diffusion_coefficient=D_O,
                bulk_concentration=C_O_bulk,
            )

    @property
    def species_r(self) -> Species:
        """Return species object for Reductant R."""
        return self._species_r

    @property
    def species_o(self) -> Species:
        """Return species object for Oxidant O."""
        return self._species_o

    @property
    def species_names(self) -> list[str]:
        """Return list of managed species identifiers."""
        return [self._species_r.name, self._species_o.name]

    @property
    def num_species(self) -> int:
        """Return number of active species in the system."""
        return 2

    def get_diffusion_coefficients(self) -> dict[str, float]:
        """Return species diffusion coefficients ($m^2/s$)."""
        return {
            self._species_r.name: self._species_r.diffusion_coefficient,
            self._species_o.name: self._species_o.diffusion_coefficient,
        }

    def get_initial_conditions(self, x_grid: np.ndarray) -> dict[str, np.ndarray]:
        """
        Generate uniform bulk initial concentrations $C_R(x, 0) = C_R^*$ and $C_O(x, 0) = C_O^*$
        across spatial coordinates.
        """
        grid_shape = x_grid.shape
        return {
            self._species_r.name: np.full(
                grid_shape, self._species_r.bulk_concentration, dtype=np.float64
            ),
            self._species_o.name: np.full(
                grid_shape, self._species_o.bulk_concentration, dtype=np.float64
            ),
        }

    def get_initial_state_vector(self, x_grid: np.ndarray) -> np.ndarray:
        """
        Construct a flattened 1D state vector concatenating $[C_R, C_O]$ for numerical solvers.

        Parameters
        ----------
        x_grid : np.ndarray
            Spatial node coordinates array of size $N$.

        Returns
        -------
        np.ndarray
            Flattened array of size $2N$ structured as $[C_{R,0}, ..., C_{R,N-1}, C_{O,0}, ..., C_{O,N-1}]$.
        """
        ics = self.get_initial_conditions(x_grid)
        return np.concatenate([ics[self._species_r.name], ics[self._species_o.name]])

    def __repr__(self) -> str:
        return (
            f"TwoSpeciesModel("
            f"R(D={self._species_r.diffusion_coefficient:.2e} m^2/s, C*={self._species_r.bulk_concentration:.2f} mol/m^3), "
            f"O(D={self._species_o.diffusion_coefficient:.2e} m^2/s, C*={self._species_o.bulk_concentration:.2f} mol/m^3))"
        )
