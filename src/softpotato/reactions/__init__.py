"""Chemical species and electrochemical reaction kinetics for Soft Potato.

This subpackage provides modular abstractions for chemical species metadata,
homogeneous (solution-phase) reaction kinetics with automated analytical
Jacobians, and (in upcoming Milestone 3 additions) heterogeneous kinetics.

Examples
--------
>>> from softpotato.reactions import Species, SpeciesCollection
>>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
>>> red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
>>> species = SpeciesCollection([ox, red])
>>> species.diffusivities
{'Ox': 1e-09, 'Red': 1e-09}

Homogeneous reactions are combined in a :class:`ReactionNetwork`, which can be
passed directly to ``DiffusionProblem(..., reactions=network)``:

>>> from softpotato.reactions import FirstOrderReaction, ReactionNetwork
>>> network = ReactionNetwork([FirstOrderReaction(red, "P", k=1.0)])
>>> network.species
['Red', 'P']
"""

from .homogeneous import (
    CatalyticReaction,
    CustomReaction,
    DimerizationReaction,
    DisproportionationReaction,
    FirstOrderReaction,
    HomogeneousReaction,
    MassActionReaction,
    ReactionNetwork,
    ReversibleReaction,
    SecondOrderReaction,
)
from .species import Species, SpeciesCollection

__all__ = [
    "CatalyticReaction",
    "CustomReaction",
    "DimerizationReaction",
    "DisproportionationReaction",
    "FirstOrderReaction",
    "HomogeneousReaction",
    "MassActionReaction",
    "ReactionNetwork",
    "ReversibleReaction",
    "SecondOrderReaction",
    "Species",
    "SpeciesCollection",
]
