"""
Mechanism builder for the softpotato framework.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Mechanism:
    """
    Composite representation of an electrochemical reaction mechanism.

    Attributes
    ----------
    steps : List[Any]
        A list of elementary reaction steps (e.g., ElectronTransferStep, ChemicalStep)
        that make up the composite mechanism.
    """

    steps: list[Any] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        """Returns the total number of elementary steps in the mechanism."""
        return len(self.steps)

    def get_species(self) -> set[str]:
        """
        Extracts all unique chemical species involved in the mechanism.

        Returns
        -------
        Set[str]
            A set of string identifiers for all participating species.
        """
        species = set()
        for step in self.steps:
            # Check for Electron Transfer (E) step attributes
            if hasattr(step, "ox_species"):
                species.add(step.ox_species)
            if hasattr(step, "red_species"):
                species.add(step.red_species)

            # Future-proofing for Chemical (C) step attributes
            if hasattr(step, "reactants"):
                species.update(step.reactants)
            if hasattr(step, "products"):
                species.update(step.products)

        return species


def build(steps: list[Any]) -> Mechanism:
    """
    Merges individual elementary steps into a composite reaction scheme.

    Parameters
    ----------
    steps : List[Any]
        A list of elementary reaction steps.

    Returns
    -------
    Mechanism
        The composite mechanism object ready for the simulator.

    Raises
    ------
    ValueError
        If the steps list is empty.
    """
    if not steps:
        raise ValueError("A mechanism must contain at least one elementary step.")

    return Mechanism(steps=steps)
