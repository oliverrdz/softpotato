import numpy as np
import pytest

from softpotato.physics.kinetics import butler_volmer, kinetics, nernst, tafel
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


def test_butler_volmer_standard_potential():
    """
    At E = E0, the overpotential is zero. Therefore, both kf and kb
    must perfectly equal the standard rate constant k0.
    """
    params = {"k0": 1e-3, "alpha": 0.5, "E0": 0.0, "n": 1.0, "T": 298.15}
    e_array = np.array([0.0])

    kf, kb = butler_volmer(e_array, params)

    np.testing.assert_allclose(kf, [1e-3], err_msg="kf does not equal k0 at E=E0")
    np.testing.assert_allclose(kb, [1e-3], err_msg="kb does not equal k0 at E=E0")


def test_butler_volmer_vectorization():
    """
    Ensures the calculation handles 1D arrays natively and that physical
    trends hold true (kf increases at negative potentials, kb at positive).
    """
    params = {"k0": 1e-3, "alpha": 0.5, "E0": 0.0}
    # Array: Negative potential (reduction), Standard potential, Positive potential (oxidation)
    e_array = np.array([-0.1, 0.0, 0.1])

    kf, kb = butler_volmer(e_array, params)

    assert kf.shape == (3,), "Output kf array shape mismatch"
    assert kb.shape == (3,), "Output kb array shape mismatch"

    # Reduction is exponentially faster at more negative potentials
    assert kf[0] > kf[1]
    # Oxidation is exponentially faster at more positive potentials
    assert kb[2] > kb[1]


def test_kinetics_factory_instantiation():
    """
    Validates that the factory pattern correctly returns a callable function
    when given a valid string identifier.
    """
    params = {"k0": 1e-3, "alpha": 0.5, "E0": 0.0}
    bv_func = kinetics("bv", params)

    assert callable(bv_func), "Factory did not return a callable function"

    e_array = np.array([0.0])
    kf, kb = bv_func(e_array)
    # Assert both forward and backward rate constants equal k0 at E = E0
    np.testing.assert_allclose(kf, [1e-3], err_msg="kf factory output mismatch")
    np.testing.assert_allclose(kb, [1e-3], err_msg="kb factory output mismatch")


def test_kinetics_factory_invalid_model():
    """
    Ensures the factory raises a clean ValueError for unsupported models.
    """
    params = {"k0": 1e-3}
    with pytest.raises(ValueError, match="not supported"):
        kinetics("INVALID_MODEL", params)


def test_nernst_standard_potential():
    """
    At the formal potential (E = E0), the Nernst equation mandates that the
    surface concentrations of the oxidized and reduced species are perfectly equal.
    Therefore, the ratio theta must be exactly 1.0.
    """
    params = {"E0": 0.0, "n": 1.0, "T": 298.15}
    e_array = np.array([0.0])

    theta = nernst(e_array, params)

    np.testing.assert_allclose(
        theta, [1.0], err_msg="Nernst ratio theta does not equal 1.0 at E=E0"
    )


def test_nernst_vectorization():
    """
    Ensures the calculation handles 1D arrays natively and that physical
    trends hold true (theta < 1 at reducing potentials, theta > 1 at oxidizing).
    """
    params = {"E0": 0.0, "n": 1.0, "T": 298.15}

    # Array: Negative (reducing), Standard potential, Positive (oxidizing)
    e_array = np.array([-0.05916, 0.0, 0.05916])

    theta = nernst(e_array, params)

    assert theta.shape == (3,), "Output theta array shape mismatch"

    # At approx 59.16 mV negative of E0, the ratio c_O/c_R should be ~0.1
    np.testing.assert_allclose(theta[0], 0.1, rtol=1e-3)

    # At approx 59.16 mV positive of E0, the ratio c_O/c_R should be ~10.0
    np.testing.assert_allclose(theta[2], 10.0, rtol=1e-3)


def test_tafel_standard_potential():
    """
    At the formal potential (E = E0), the overpotential is zero.
    Therefore, both kf and kb must perfectly equal the standard rate constant k0.
    """
    params = {"k0": 1e-3, "E0": 0.0, "bc": 0.120, "ba": 0.120}
    e_array = np.array([0.0])

    kf, kb = tafel(e_array, params)

    np.testing.assert_allclose(kf, [1e-3], err_msg="Tafel kf does not equal k0 at E=E0")
    np.testing.assert_allclose(kb, [1e-3], err_msg="Tafel kb does not equal k0 at E=E0")


def test_tafel_slope_scaling():
    """
    Verifies that shifting the potential by exactly one Tafel slope (e.g., 120 mV)
    results in exactly a 10-fold (one decade) increase in the respective rate constant.
    """
    params = {"k0": 1e-3, "E0": 0.0, "bc": 0.120, "ba": 0.120}

    # Array: One cathodic decade (-0.120V), Standard (0.0V), One anodic decade (+0.120V)
    e_array = np.array([-0.120, 0.0, 0.120])

    kf, kb = tafel(e_array, params)

    # At -120 mV, kf should be 10x larger than k0 (1e-2 m/s)
    np.testing.assert_allclose(kf[0], 1e-2, err_msg="kf did not scale by one decade")

    # At +120 mV, kb should be 10x larger than k0 (1e-2 m/s)
    np.testing.assert_allclose(kb[2], 1e-2, err_msg="kb did not scale by one decade")


def test_kinetics_factory_tafel():
    """
    Validates that the factory pattern correctly returns the Tafel callable
    when given the 'TAFEL' string identifier.
    """
    params = {"k0": 5e-4, "E0": 0.1, "bc": 0.060, "ba": 0.060}
    tafel_func = kinetics("tafel", params)

    assert callable(tafel_func), "Factory did not return a callable function for TAFEL"

    # Test exactly at E0
    e_array = np.array([0.1])
    kf, kb = tafel_func(e_array)

    np.testing.assert_allclose(kf, [5e-4])
    np.testing.assert_allclose(kb, [5e-4])
