"""
Unit tests for the softpotato mechanism builder module.
"""

import pytest

from softpotato.mechanism.builder import Mechanism, build
from softpotato.mechanism.steps import ElectronTransferStep


def test_build_single_e_mechanism():
    """Test building a simple single-step E mechanism."""
    e_step = ElectronTransferStep(ox_species="O", red_species="R")
    mech = build([e_step])

    assert isinstance(mech, Mechanism)
    assert mech.n_steps == 1
    assert mech.steps[0].ox_species == "O"

    # Check species extraction
    species = mech.get_species()
    assert species == {"O", "R"}


def test_build_ee_mechanism():
    """Test building a multi-step EE mechanism using the composite pattern."""
    step_1 = ElectronTransferStep(
        ox_species="O", red_species="I", e_formal=0.1, k0=0.01
    )
    step_2 = ElectronTransferStep(
        ox_species="I", red_species="R", e_formal=-0.2, k0=0.005
    )

    mech = build([step_1, step_2])

    assert mech.n_steps == 2
    assert mech.steps[0].red_species == "I"
    assert mech.steps[1].ox_species == "I"

    # Check that the intermediate 'I' is properly captured
    species = mech.get_species()
    assert species == {"O", "I", "R"}


def test_build_empty_mechanism_raises_error():
    """Test that building a mechanism without steps raises a ValueError."""
    with pytest.raises(ValueError, match="at least one elementary step"):
        build([])
