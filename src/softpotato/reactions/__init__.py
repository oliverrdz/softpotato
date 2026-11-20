"""Chemical species and electrochemical reaction kinetics for Soft Potato.

This subpackage provides modular abstractions for chemical species metadata and
heterogeneous charge-transfer kinetics.

Examples
--------
>>> from softpotato.reactions import ButlerVolmer, Species, SpeciesCollection
>>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
>>> red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
>>> species = SpeciesCollection([ox, red])
>>> species.diffusivities
{'Ox': 1e-09, 'Red': 1e-09}
>>> bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
>>> bv.rate_constants(0.0)
(0.0001, 0.0001)
"""

from .heterogeneous import (
    ButlerVolmer,
    HeterogeneousKinetics,
    NernstianEquilibrium,
    Tafel,
    TafelKinetics,
    create_electrochemical_bcs,
)
from .species import Species, SpeciesCollection

__all__ = [
    "ButlerVolmer",
    "HeterogeneousKinetics",
    "NernstianEquilibrium",
    "Species",
    "SpeciesCollection",
    "Tafel",
    "TafelKinetics",
    "create_electrochemical_bcs",
]
