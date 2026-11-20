"""Chemical species and electrochemical reaction kinetics for Soft Potato.

This subpackage provides modular abstractions for chemical species metadata and
(in upcoming Milestone 3 additions) homogeneous and heterogeneous reaction kinetics.

Examples
--------
>>> from softpotato.reactions import Species, SpeciesCollection
>>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
>>> red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
>>> species = SpeciesCollection([ox, red])
>>> species.diffusivities
{'Ox': 1e-09, 'Red': 1e-09}
"""

from .species import Species, SpeciesCollection

__all__ = [
    "Species",
    "SpeciesCollection",
]
