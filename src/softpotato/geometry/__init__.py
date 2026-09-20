"""
Electrode geometries and spatial FDM grid generation.
"""

from .electrodes import (
    CylindricalElectrode,
    Electrode,
    PlanarElectrode,
    SphericalElectrode,
)
from .grids import (
    Grid,
    UniformGrid,
    generate_exponential_grid,
)

__all__ = [
    "CylindricalElectrode",
    "Electrode",
    "Grid",
    "PlanarElectrode",
    "SphericalElectrode",
    "UniformGrid",
    "generate_exponential_grid",
]
