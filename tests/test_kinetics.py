"""
Unit tests for electrochemical kinetics models.
"""

from __future__ import annotations

import numpy as np
import pytest

from softpotato.kinetics import ButlerVolmer, KineticsModel


def test_butler_volmer_initialization() -> None:
    """Tests ButlerVolmer initialization and properties."""
    bv = ButlerVolmer(k0=1e-3, alpha=0.5)
    assert isinstance(bv, KineticsModel)
    assert bv.k0 == 1e-3
    assert bv.alpha == 0.5
    assert "ButlerVolmer" in repr(bv)


def test_butler_volmer_validation() -> None:
    """Tests validation of ButlerVolmer parameters."""
    # Invalid k0
    with pytest.raises(TypeError, match="real number"):
        ButlerVolmer(k0="1e-3")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        ButlerVolmer(k0=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        ButlerVolmer(k0=-1e-3)

    # Invalid alpha
    with pytest.raises(TypeError, match="real number"):
        ButlerVolmer(k0=1e-3, alpha="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 0 and 1"):
        ButlerVolmer(k0=1e-3, alpha=-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ButlerVolmer(k0=1e-3, alpha=1.1)


def test_butler_volmer_rates_at_equilibrium() -> None:
    """Tests that k_f == k_b == k_0 when E == E0."""
    bv = ButlerVolmer(k0=0.01, alpha=0.5)
    kf, kb = bv.calculate_rates(E=0.2, E0=0.2, n_electrons=1)

    assert isinstance(kf, float)
    assert isinstance(kb, float)
    assert np.isclose(kf, 0.01)
    assert np.isclose(kb, 0.01)


def test_butler_volmer_rates_overpotential() -> None:
    """Tests rate constants under positive and negative overpotentials."""
    bv = ButlerVolmer(k0=0.01, alpha=0.5)

    # Reduction overpotential (E < E0): kf > kb
    kf_red, kb_red = bv.calculate_rates(E=-0.1, E0=0.0, n_electrons=1)
    assert kf_red > kb_red

    # Oxidation overpotential (E > E0): kb > kf
    kf_ox, kb_ox = bv.calculate_rates(E=0.1, E0=0.0, n_electrons=1)
    assert kb_ox > kf_ox


def test_butler_volmer_rates_vectorized() -> None:
    """Tests rate calculation with a NumPy array of potentials."""
    bv = ButlerVolmer(k0=0.01, alpha=0.5)
    potentials = np.linspace(-0.2, 0.2, 50)
    kf_arr, kb_arr = bv.calculate_rates(E=potentials, E0=0.0, n_electrons=1)

    assert isinstance(kf_arr, np.ndarray)
    assert isinstance(kb_arr, np.ndarray)
    assert len(kf_arr) == 50
    assert len(kb_arr) == 50
    # kf should be monotonically decreasing with increasing E
    assert np.all(np.diff(kf_arr) < 0)
    # kb should be monotonically increasing with increasing E
    assert np.all(np.diff(kb_arr) > 0)


def test_butler_volmer_flux() -> None:
    """Tests net flux calculation J = kf * c_O - kb * c_R."""
    bv = ButlerVolmer(k0=0.01, alpha=0.5)
    # At E = E0 and c_O = c_R, J = 0
    flux_eq = bv.calculate_flux(E=0.0, E0=0.0, n_electrons=1, c_O=1e-6, c_R=1e-6)
    assert np.isclose(flux_eq, 0.0)

    # Under reduction potential with only O present: J > 0
    flux_red = bv.calculate_flux(E=-0.1, E0=0.0, n_electrons=1, c_O=1e-6, c_R=0.0)
    assert flux_red > 0.0
