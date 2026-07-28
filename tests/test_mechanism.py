"""
Unit tests for the softpotato mechanism module.
"""

import pytest

from softpotato.mechanism.steps import ElectronTransferStep


def test_electron_transfer_step_defaults():
    """Test the E step initialization with default thermodynamic/kinetic values in SI units."""
    e_step = ElectronTransferStep(ox_species="Ox", red_species="Red")

    assert e_step.ox_species == "Ox"
    assert e_step.red_species == "Red"
    assert e_step.n_electrons == 1
    assert e_step.e_formal == 0.0
    assert e_step.k0 == 0.01  # m/s
    assert e_step.alpha == 0.5


def test_electron_transfer_step_custom_parameters():
    """Test the E step initialization with user-defined parameters."""
    e_step = ElectronTransferStep(
        ox_species="A",
        red_species="B",
        n_electrons=2,
        e_formal=0.25,
        k0=0.0005,  # m/s
        alpha=0.3,
    )

    assert e_step.n_electrons == 2
    assert e_step.e_formal == 0.25
    assert e_step.k0 == 0.0005
    assert e_step.alpha == 0.3


def test_electron_transfer_step_invalid_electrons():
    """Test that zero or negative electrons raise a ValueError."""
    with pytest.raises(ValueError, match="must be at least 1"):
        ElectronTransferStep(ox_species="O", red_species="R", n_electrons=0)

    with pytest.raises(ValueError, match="must be at least 1"):
        ElectronTransferStep(ox_species="O", red_species="R", n_electrons=-2)


def test_electron_transfer_step_invalid_rate_constant():
    """Test that a negative standard rate constant raises a ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        ElectronTransferStep(ox_species="O", red_species="R", k0=-0.1)


def test_electron_transfer_step_invalid_alpha():
    """Test that out-of-bounds charge transfer coefficients raise a ValueError."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ElectronTransferStep(ox_species="O", red_species="R", alpha=-0.1)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ElectronTransferStep(ox_species="O", red_species="R", alpha=1.5)
