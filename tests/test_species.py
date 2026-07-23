import numpy as np
import pytest
from pydantic import ValidationError

from softpotato.physics.species import Species, TwoSpeciesModel


def test_species_validation():
    """Verify species parameter validation rules."""
    species_r = Species(name="R", diffusion_coefficient=1e-9, bulk_concentration=1.0)
    assert species_r.name == "R"
    assert species_r.diffusion_coefficient == 1e-9
    assert species_r.bulk_concentration == 1.0

    # Test invalid diffusion coefficient (non-positive)
    with pytest.raises(ValidationError):
        Species(name="R", diffusion_coefficient=0.0, bulk_concentration=1.0)

    # Test invalid concentration (negative)
    with pytest.raises(ValidationError):
        Species(name="R", diffusion_coefficient=1e-9, bulk_concentration=-0.5)


def test_two_species_model_default_initialization():
    """Verify default initialization of species R and O."""
    model = TwoSpeciesModel()

    assert model.num_species == 2
    assert model.species_names == ["R", "O"]

    diff_coeffs = model.get_diffusion_coefficients()
    assert diff_coeffs["R"] == 1e-9
    assert diff_coeffs["O"] == 1e-9


def test_two_species_model_custom_initialization():
    """Verify model creation with custom species objects."""
    r = Species(name="Red", diffusion_coefficient=2e-9, bulk_concentration=5.0)
    o = Species(name="Ox", diffusion_coefficient=1.5e-9, bulk_concentration=0.0)

    model = TwoSpeciesModel(species_r=r, species_o=o)

    assert model.species_names == ["Red", "Ox"]
    assert model.get_diffusion_coefficients() == {"Red": 2e-9, "Ox": 1.5e-9}


def test_initial_conditions_and_state_vector():
    """Test evaluation of initial spatial distributions and state vector ordering."""
    model = TwoSpeciesModel(D_R=1e-9, D_O=1e-9, C_R_bulk=1.0, C_O_bulk=0.0)
    x_grid = np.linspace(0.0, 1e-4, 5)

    ics = model.get_initial_conditions(x_grid)
    assert np.allclose(ics["R"], 1.0)
    assert np.allclose(ics["O"], 0.0)
    assert len(ics["R"]) == 5

    state_vec = model.get_initial_state_vector(x_grid)
    assert state_vec.shape == (10,)
    assert np.allclose(state_vec[:5], 1.0)
    assert np.allclose(state_vec[5:], 0.0)
