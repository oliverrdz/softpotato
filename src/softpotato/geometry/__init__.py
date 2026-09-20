"""
Electrode geometries and spatial FDM grid generation.
"""

from .grids import generate_exponential_grid, generate_linear_grid
from .electrodes import PlanarElectrode, SphericalElectrode, CylindricalElectrode

__all__ = [
    "generate_exponential_grid",
    "generate_linear_grid",
    "PlanarElectrode",
    "SphericalElectrode",
    "CylindricalElectrode"
]
