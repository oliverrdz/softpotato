"""
Electrode geometries and spatial FDM grid generation.
"""

from .electrodes import CylindricalElectrode, PlanarElectrode, SphericalElectrode
from .grids import generate_exponential_grid, generate_linear_grid

__all__ = [
    "CylindricalElectrode",
    "PlanarElectrode",
    "SphericalElectrode",
    "generate_exponential_grid",
    "generate_linear_grid",
]
