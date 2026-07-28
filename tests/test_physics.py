import pytest

from softpotato.physics.species import Species


def test_species_valid_instantiation():
    """Test that a valid Species object is created with correct SI unit attributes."""
    # Example: D = 1e-9 m^2/s, c0 = 1.0 mol/m^3
    species = Species(D=1e-9, c0=1.0)
    assert species.D == 1e-9
    assert species.c0 == 1.0


def test_species_zero_concentration_is_valid():
    """Test that an initial concentration of exactly 0 is permitted."""
    species = Species(D=1e-9, c0=0.0)
    assert species.c0 == 0.0


def test_species_invalid_diffusion_coefficient():
    """Test that an invalid (zero or negative) diffusion coefficient raises a ValueError."""
    with pytest.raises(
        ValueError, match="Diffusion coefficient D must be strictly positive"
    ):
        Species(D=0.0, c0=1.0)

    with pytest.raises(
        ValueError, match="Diffusion coefficient D must be strictly positive"
    ):
        Species(D=-1e-9, c0=1.0)


def test_species_invalid_concentration():
    """Test that a negative initial concentration raises a ValueError."""
    with pytest.raises(
        ValueError, match="Initial bulk concentration c0 must be non-negative"
    ):
        Species(D=1e-9, c0=-1.0)
