"""Tests for centralized physical and electrochemical constants."""

import math

import softpotato as sp
from softpotato.constants import (
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    FARADAY,
    GAS_CONSTANT,
    STANDARD_TEMPERATURE,
    T_STD,
    VACUUM_PERMITTIVITY,
    F,
    R,
)


def test_electrochemical_triad_values():
    """Verify primary electrochemical constants match expected CODATA/SI values."""
    assert isinstance(FARADAY, float)
    assert isinstance(GAS_CONSTANT, float)
    assert isinstance(STANDARD_TEMPERATURE, float)

    # Faraday constant: ~96485.33212 C/mol
    assert math.isclose(FARADAY, 96485.3321233, rel_tol=1e-6)

    # Molar gas constant: ~8.3144626 J/(mol*K)
    assert math.isclose(GAS_CONSTANT, 8.3144626, rel_tol=1e-6)

    # Standard temperature: 298.15 K (25 °C)
    assert STANDARD_TEMPERATURE == 298.15


def test_shorthand_aliases():
    """Verify shorthand aliases match canonical constant definitions."""
    assert F == FARADAY
    assert R == GAS_CONSTANT
    assert T_STD == STANDARD_TEMPERATURE


def test_foundational_constants_values():
    """Verify foundational physical constants match CODATA/SI definitions."""
    assert isinstance(AVOGADRO, float)
    assert isinstance(BOLTZMANN, float)
    assert isinstance(ELEMENTARY_CHARGE, float)
    assert isinstance(VACUUM_PERMITTIVITY, float)

    assert math.isclose(AVOGADRO, 6.02214076e23, rel_tol=1e-8)
    assert math.isclose(BOLTZMANN, 1.380649e-23, rel_tol=1e-8)
    assert math.isclose(ELEMENTARY_CHARGE, 1.602176634e-19, rel_tol=1e-8)
    assert math.isclose(VACUUM_PERMITTIVITY, 8.8541878e-12, rel_tol=1e-6)


def test_physical_consistency_relations():
    """Verify fundamental physical relationships among constants."""
    # F = e * N_A
    assert math.isclose(FARADAY, ELEMENTARY_CHARGE * AVOGADRO, rel_tol=1e-12)

    # R = k_B * N_A
    assert math.isclose(GAS_CONSTANT, BOLTZMANN * AVOGADRO, rel_tol=1e-12)


def test_top_level_reexports():
    """Verify constants and aliases are properly re-exported at the package level."""
    assert sp.FARADAY == FARADAY
    assert sp.GAS_CONSTANT == GAS_CONSTANT
    assert sp.STANDARD_TEMPERATURE == STANDARD_TEMPERATURE
    assert sp.F == F
    assert sp.R == R
    assert sp.T_STD == T_STD
    assert sp.AVOGADRO == AVOGADRO
    assert sp.BOLTZMANN == BOLTZMANN
    assert sp.ELEMENTARY_CHARGE == ELEMENTARY_CHARGE
    assert sp.VACUUM_PERMITTIVITY == VACUUM_PERMITTIVITY
    assert sp.constants is not None


def test_module_all_exports():
    """Verify all items declared in __all__ exist on softpotato.constants."""
    from softpotato import constants

    for name in constants.__all__:
        assert hasattr(constants, name)


def test_package_all_exports():
    """Verify all items declared in __all__ exist on softpotato."""
    for name in sp.__all__:
        assert hasattr(sp, name)
