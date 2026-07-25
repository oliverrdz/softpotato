from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from softpotato.physics import ButlerVolmerBC

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_technique():
    """Mock technique returning a constant potential of 0.0 V."""
    technique = MagicMock()
    technique.side_effect = lambda t: 0.0
    return technique


@pytest.fixture
def mock_mesh():
    """Mock 1D mesh with 5 nodes and dx = 1e-4 m."""
    mesh = MagicMock()
    mesh.num_nodes = 5
    mesh.dx = 1e-4
    mesh.x = np.linspace(0, 4e-4, 5)
    return mesh


@pytest.fixture
def mock_model():
    """Mock transport model with species R and O."""
    model = MagicMock()
    model.species_names = ["R", "O"]
    model.get_diffusion_coefficients.return_value = {"R": 1e-9, "O": 1e-9}
    model.get_initial_conditions.return_value = {
        "R": np.ones(5),
        "O": np.zeros(5),
    }
    return model


# =============================================================================
# Unit Tests
# =============================================================================


def test_butler_volmer_init(mock_technique):
    """Test proper initialization and precomputed scaling factor f."""
    bc = ButlerVolmerBC(
        technique=mock_technique,
        E0=0.1,
        k0=1e-3,
        alpha=0.4,
        n=1,
        T=298.15,
        A=2.0,
    )

    assert bc.E0 == 0.1
    assert bc.k0 == 1e-3
    assert bc.alpha == 0.4
    assert bc.n == 1
    assert bc.T == 298.15
    assert bc.A == 2.0

    # Check pre-calculated constant f = n*F / (R*T)
    expected_f = (1 * 96485.3321) / (8.3144626 * 298.15)
    assert pytest.approx(bc.f, rel=1e-5) == expected_f


def test_rate_constants_at_zero_overpotential(mock_technique):
    """At E(t) = E0 (eta = 0), both anodic and cathodic rate constants must equal k0."""
    bc = ButlerVolmerBC(technique=mock_technique, E0=0.0, k0=1e-4, alpha=0.5)

    k_a, k_c = bc.get_rate_constants(t=0.0)

    assert pytest.approx(k_a, rel=1e-6) == 1e-4
    assert pytest.approx(k_c, rel=1e-6) == 1e-4


def test_rate_constants_anodic_bias():
    """At E(t) > E0 (eta > 0), k_a > k0 and k_c < k0."""
    technique = lambda t: 0.1  # +100 mV overpotential
    bc = ButlerVolmerBC(technique=technique, E0=0.0, k0=1e-4, alpha=0.5)

    k_a, k_c = bc.get_rate_constants(t=0.0)

    assert k_a > 1e-4
    assert k_c < 1e-4


def test_rate_constants_cathodic_bias():
    """At E(t) < E0 (eta < 0), k_a < k0 and k_c > k0."""
    technique = lambda t: -0.1  # -100 mV overpotential
    bc = ButlerVolmerBC(technique=technique, E0=0.0, k0=1e-4, alpha=0.5)

    k_a, k_c = bc.get_rate_constants(t=0.0)

    assert k_a < 1e-4
    assert k_c > 1e-4


def test_apply_boundary_condition(mock_technique, mock_mesh, mock_model):
    """Test 2nd-order surface flux solve at x=0 and bulk concentration enforcement at x=L."""
    bc = ButlerVolmerBC(technique=mock_technique, E0=0.0, k0=1e-4, alpha=0.5)

    # State vector: [C_R0, C_R1, C_R2, C_R3, C_R4, C_O0, C_O1, C_O2, C_O3, C_O4]
    state = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,  # Species R
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Species O
        ],
        dtype=float,
    )

    updated_state = bc.apply(state=state, t=0.0, mesh=mock_mesh, model=mock_model)

    assert updated_state.shape == state.shape

    C_R_updated = updated_state[:5]
    C_O_updated = updated_state[5:]

    # Bulk conditions (node N-1 = node 4) must match initial conditions
    assert C_R_updated[-1] == 1.0
    assert C_O_updated[-1] == 0.0

    # Surface concentrations (node 0) must be updated and non-NaN
    assert not np.isnan(C_R_updated[0])
    assert not np.isnan(C_O_updated[0])


def test_calculate_current(mock_mesh, mock_model):
    """Test 3-point 2nd-order finite difference current calculation."""
    bc = ButlerVolmerBC(technique=MagicMock(), n=1, A=1.0)

    # State with non-zero O gradient at surface: C_O0 = 0.2, C_O1 = 0.1, C_O2 = 0.0
    state = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,  # C_R
            0.2,
            0.1,
            0.0,
            0.0,
            0.0,  # C_O
        ],
        dtype=float,
    )

    # Derivative dC_O/dx = (-3*0.2 + 4*0.1 - 0.0) / (2 * 1e-4) = -0.2 / 2e-4 = -1000.0 mol/m^4
    # i = -n * F * A * D_O * (dC_O/dx)
    # i = -1 * 96485.3321 * 1.0 * 1e-9 * (-1000.0) = 0.0964853321 A
    current = bc.calculate_current(state=state, mesh=mock_mesh, model=mock_model)

    expected_current = 1 * 96485.3321 * 1.0 * 1e-9 * 1000.0
    assert pytest.approx(current, rel=1e-5) == expected_current


def test_calculate_current_custom_area_override(mock_mesh, mock_model):
    """Test overriding area A in calculate_current."""
    bc = ButlerVolmerBC(technique=MagicMock(), n=1, A=1.0)

    state = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.1, 0.0, 0.0, 0.0], dtype=float)

    # Passing custom area A = 2.0 should double the output current
    i_default = bc.calculate_current(state=state, mesh=mock_mesh, model=mock_model)
    i_custom = bc.calculate_current(
        state=state, mesh=mock_mesh, model=mock_model, A=2.0
    )

    assert pytest.approx(i_custom, rel=1e-6) == 2.0 * i_default
