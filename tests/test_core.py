"""
Tests for core species module.
"""

import numpy as np
import pytest

from softpotato.core.species import Species


def test_species_initialization_defaults() -> None:
    """Verifies default parameters when initializing Species."""
    spec = Species(name="O", D=1e-5)
    assert spec.name == "O"
    assert spec.D == 1e-5
    assert spec.c_bulk == 0.0
    assert spec.charge == 0
    assert spec.c_profile is None


def test_species_initialization_custom() -> None:
    """Verifies custom parameters for Species."""
    spec = Species(name="O", D=1.0e-5, c_bulk=1.0e-6, charge=1)
    assert spec.name == "O"
    assert spec.D == 1.0e-5
    assert spec.c_bulk == 1.0e-6
    assert spec.charge == 1
    assert spec.c_profile is None


def test_species_name_validation() -> None:
    """Verifies validation of species name."""
    with pytest.raises(ValueError, match="non-empty string"):
        Species(name="", D=1e-5)

    with pytest.raises(ValueError, match="non-empty string"):
        Species(name="   ", D=1e-5)

    with pytest.raises(ValueError, match="non-empty string"):
        # Type error or value error for non-string
        Species(name=123, D=1e-5)  # type: ignore[arg-type]


def test_species_diffusion_validation() -> None:
    """Verifies validation of diffusion coefficient D."""
    with pytest.raises(ValueError, match="non-negative"):
        Species(name="O", D=-1e-5)

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D="fast")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=True)  # type: ignore[arg-type]

    # D = 0 is valid (e.g. immobilized / adsorbed species)
    spec = Species(name="Immobilized", D=0.0)
    assert spec.D == 0.0


def test_species_c_bulk_validation() -> None:
    """Verifies validation of bulk concentration c_bulk."""
    with pytest.raises(ValueError, match="non-negative"):
        Species(name="O", D=1e-5, c_bulk=-1e-6)

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=1e-5, c_bulk="1mM")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=1e-5, c_bulk=False)  # type: ignore[arg-type]


def test_species_charge_validation() -> None:
    """Verifies validation of ionic charge."""
    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge="1")  # type: ignore[arg-type]

    # Negative charges are valid (e.g., anions)
    spec = Species(name="Anion", D=1e-5, charge=-2)
    assert spec.charge == -2


def test_species_profile_initialization_and_reset() -> None:
    """Verifies concentration profile allocation and reset."""
    spec = Species(name="O", D=1e-5, c_bulk=1e-6)
    assert spec.c_profile is None

    spec.initialize_profile(n_nodes=100)
    assert spec.c_profile is not None
    assert spec.c_profile.shape == (100,)
    assert spec.c_profile.dtype == np.float64
    assert np.all(spec.c_profile == 1e-6)

    # Modify the profile
    spec.c_profile[0] = 0.0
    spec.c_profile[1:10] = 0.5e-6
    assert spec.c_profile[0] == 0.0

    # Reset back to bulk concentration
    spec.reset()
    assert np.all(spec.c_profile == 1e-6)

    # Invalid n_nodes
    with pytest.raises(ValueError, match="greater than 0"):
        spec.initialize_profile(n_nodes=0)

    with pytest.raises(ValueError, match="greater than 0"):
        spec.initialize_profile(n_nodes=-10)

    with pytest.raises(TypeError, match="integer"):
        spec.initialize_profile(n_nodes=50.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        spec.initialize_profile(n_nodes=True)  # type: ignore[arg-type]


def test_species_setters() -> None:
    """Verifies property setters and their validations."""
    spec = Species(name="O", D=1e-5, c_bulk=1e-6, charge=0)

    spec.name = "Oxidant"
    assert spec.name == "Oxidant"
    with pytest.raises(ValueError):
        spec.name = ""

    spec.D = 2e-5
    assert spec.D == 2e-5
    with pytest.raises(ValueError):
        spec.D = -1.0

    spec.c_bulk = 5e-6
    assert spec.c_bulk == 5e-6
    with pytest.raises(ValueError):
        spec.c_bulk = -2.0

    spec.charge = -1
    assert spec.charge == -1
    with pytest.raises(TypeError):
        spec.charge = 1.2  # type: ignore[assignment]

    # c_profile setter validation
    spec.c_profile = np.array([1.0, 2.0, 3.0])
    assert len(spec.c_profile) == 3

    with pytest.raises(TypeError, match="numpy ndarray"):
        spec.c_profile = [1.0, 2.0]  # type: ignore[assignment]

    with pytest.raises(ValueError, match="1D array"):
        spec.c_profile = np.zeros((3, 3))

    spec.c_profile = None
    assert spec.c_profile is None
    # Reset when profile is None should not raise error
    spec.reset()


def test_species_equality_and_representations() -> None:
    """Verifies __repr__, __str__, and __eq__."""
    spec1 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=1)
    spec2 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=1)
    spec3 = Species(name="R", D=1e-5, c_bulk=1e-6, charge=1)
    spec4 = Species(name="O", D=2e-5, c_bulk=1e-6, charge=1)
    spec5 = Species(name="O", D=1e-5, c_bulk=0.0, charge=1)
    spec6 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=0)

    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1 != spec4
    assert spec1 != spec5
    assert spec1 != spec6
    assert spec1 != "O"

    assert "Species(name='O'" in repr(spec1)
    assert "cm²/s" in repr(spec1)
    assert "mol/cm³" in repr(spec1)

    assert "Species O" in str(spec1)
    assert "cm²/s" in str(spec1)
    assert "mol/cm³" in str(spec1)
