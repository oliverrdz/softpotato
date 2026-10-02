"""Tests for Scanning Electrochemical Microscopy (SECM) analytical approach curves."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.secm import (
    secm_approach_curve,
    secm_approach_negative_feedback,
    secm_approach_positive_feedback,
    secm_limiting_current_infinite,
    secm_tip_current,
)
from softpotato.constants import FARADAY

# ==============================================================================
# 1. Positive Feedback Tests
# ==============================================================================


def test_secm_positive_feedback_scalar():
    """Verify positive feedback produces correct float scalars and known values."""
    # At L = 1.0, RG = 10.0
    val_1 = secm_approach_positive_feedback(1.0, RG=10.0)
    assert isinstance(val_1, float)
    assert math.isclose(val_1, 1.562778, rel_tol=1e-5)

    # At L = 0.1, RG = 10.0
    val_01 = secm_approach_positive_feedback(0.1, RG=10.0)
    assert isinstance(val_01, float)
    assert val_01 > val_1
    assert math.isclose(val_01, 8.3851, rel_tol=1e-4)


def test_secm_positive_feedback_asymptotics():
    """Verify asymptotic behavior of positive feedback as L -> inf and L -> 0."""
    rg = 10.0

    # L -> infinity: normalized current must approach 1.0 exactly
    val_inf = secm_approach_positive_feedback(1e4, RG=rg)
    assert math.isclose(val_inf, 1.0, abs_tol=1e-4)

    # L -> 0: normalized current diverges ~ pi / (4 * beta * L)
    val_small = secm_approach_positive_feedback(1e-4, RG=rg)
    assert val_small > 5000.0


def test_secm_positive_feedback_rg_10_literature_benchmark():
    """Verify agreement with Cornut & Lefrou (2008) Eq. (20b) simplified formula for RG=10."""
    distances = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])

    # Eq. (20b): N_T^cond = 0.652 + 0.772 / arctan(L) - 0.0911 * arctan(L)
    atan_d = np.arctan(distances)
    eq_20b = 0.652 + 0.772 / atan_d - 0.0911 * atan_d

    actual = secm_approach_positive_feedback(distances, RG=10.0)
    # The paper notes Eq. 20b is a rounded fit to Eq. 18; relative error should be < 0.3%
    np.testing.assert_allclose(actual, eq_20b, rtol=3e-3)


def test_secm_positive_feedback_vector_monotonicity():
    """Verify array inputs and strict monotonic decrease with increasing distance."""
    distances = np.geomspace(0.01, 10.0, 50)
    i_t = secm_approach_positive_feedback(distances, RG=5.0)

    assert isinstance(i_t, np.ndarray)
    assert i_t.shape == distances.shape
    # Strictly decreasing as tip retreats from conductive substrate
    assert np.all(np.diff(i_t) < 0)
    # All values must be > 1.0 (positive feedback)
    assert np.all(i_t > 1.0)


def test_secm_positive_feedback_broadcasting():
    """Verify broadcasting across distance L and insulator ratio RG arrays."""
    L = np.array([[0.5], [1.0], [2.0]])  # Shape (3, 1)
    RG = np.array([2.0, 5.0, 10.0, 20.0])  # Shape (4,)

    result = secm_approach_positive_feedback(L, RG=RG)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)


def test_secm_positive_feedback_validation():
    """Verify input validation triggers appropriate ValueErrors."""
    # L <= 0
    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_positive_feedback(0.0, RG=10.0)

    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_positive_feedback(-1.0, RG=10.0)

    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_positive_feedback(np.array([1.0, -0.5]), RG=10.0)

    # RG <= 1
    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_positive_feedback(1.0, RG=1.0)

    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_positive_feedback(1.0, RG=0.5)

    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_positive_feedback(1.0, RG=np.array([10.0, 0.9]))


# ==============================================================================
# 2. Negative Feedback Tests
# ==============================================================================


def test_secm_negative_feedback_scalar():
    """Verify negative feedback produces correct float scalars and known values."""
    # At L = 1.0, RG = 10.0
    val_1 = secm_approach_negative_feedback(1.0, RG=10.0)
    assert isinstance(val_1, float)
    assert math.isclose(val_1, 0.4983, rel_tol=1e-3)

    # At L = 0.05, RG = 10.0 (strong hindered diffusion)
    val_005 = secm_approach_negative_feedback(0.05, RG=10.0)
    assert isinstance(val_005, float)
    assert val_005 < val_1
    assert math.isclose(val_005, 0.0337, rel_tol=2e-2)


def test_secm_negative_feedback_asymptotics():
    """Verify asymptotic behavior of negative feedback as L -> inf and L -> 0."""
    rg = 10.0

    # L -> infinity: normalized current approaches 1.0
    val_inf = secm_approach_negative_feedback(1e4, RG=rg)
    assert math.isclose(val_inf, 1.0, abs_tol=1e-3)

    # L -> 0: hindered diffusion drives current toward 0.0
    val_small = secm_approach_negative_feedback(1e-4, RG=rg)
    assert math.isclose(val_small, 0.0, abs_tol=1e-3)
    assert val_small > 0.0


def test_secm_negative_feedback_rg_10_literature_benchmark():
    """Verify agreement with Cornut & Lefrou (2008) Eq. (20c) simplified formula for RG=10."""
    distances = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])

    # Eq. (20c):
    # num = 0.912 * L + 1.57
    # den = 0.912 * L + 1.59 + 2.3 / L + 0.0636 * ln(1 + 15.7 / L)
    num = 0.912 * distances + 1.57
    den = (
        0.912 * distances
        + 1.59
        + 2.3 / distances
        + 0.0636 * np.log(1.0 + 15.7 / distances)
    )
    eq_20c = num / den

    actual = secm_approach_negative_feedback(distances, RG=10.0)
    np.testing.assert_allclose(actual, eq_20c, rtol=4e-3)


def test_secm_negative_feedback_vector_monotonicity():
    """Verify array inputs and strict monotonic increase with distance from insulator."""
    distances = np.geomspace(0.01, 10.0, 50)
    i_t = secm_approach_negative_feedback(distances, RG=5.0)

    assert isinstance(i_t, np.ndarray)
    assert i_t.shape == distances.shape
    # Strictly increasing as tip retreats from insulating substrate
    assert np.all(np.diff(i_t) > 0)
    # Values bounded between 0 and 1
    assert np.all(i_t > 0.0)
    assert np.all(i_t < 1.0)


def test_secm_negative_feedback_rg_effect():
    """Verify that larger RG leads to stronger diffusion blocking (lower current) at small L."""
    L = 0.2
    rgs = np.array([1.5, 2.0, 5.0, 10.0, 20.0])
    i_vals = secm_approach_negative_feedback(L, RG=rgs)

    # Larger insulator shroud blocks diffusion more effectively => current decreases
    assert np.all(np.diff(i_vals) < 0)


def test_secm_negative_feedback_validation():
    """Verify input validation triggers appropriate ValueErrors."""
    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_negative_feedback(0.0)

    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_negative_feedback(-0.1)

    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_negative_feedback(1.0, RG=1.0)

    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_negative_feedback(1.0, RG=0.8)


# ==============================================================================
# 3. Unified Substrate Kinetics Approach Curve Tests
# ==============================================================================


def test_secm_approach_curve_limiting_cases():
    """Verify that approach_curve perfectly reproduces positive and negative feedback limits."""
    L = np.linspace(0.1, 2.0, 10)
    rg = 10.0

    # kappa = infinity must equal pure positive feedback
    curve_pos = secm_approach_curve(L, RG=rg, kappa=np.inf)
    pos_exact = secm_approach_positive_feedback(L, RG=rg)
    np.testing.assert_allclose(curve_pos, pos_exact, rtol=1e-14)

    # kappa = 0 must equal pure negative feedback
    curve_neg = secm_approach_curve(L, RG=rg, kappa=0.0)
    neg_exact = secm_approach_negative_feedback(L, RG=rg)
    np.testing.assert_allclose(curve_neg, neg_exact, rtol=1e-14)


def test_secm_approach_curve_intermediate_kinetics():
    """Verify intermediate substrate kinetics lie strictly between negative and positive limits."""
    L = 0.5
    rg = 10.0

    i_neg = secm_approach_negative_feedback(L, RG=rg)
    i_pos = secm_approach_positive_feedback(L, RG=rg)

    kappas = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
    i_curves = [secm_approach_curve(L, RG=rg, kappa=k) for k in kappas]

    # Monotonically increasing with substrate rate constant kappa
    assert np.all(np.diff(i_curves) > 0)

    for i_c in i_curves:
        assert i_neg < i_c < i_pos


def test_secm_approach_curve_broadcasting():
    """Verify 3D broadcasting across L, RG, and kappa."""
    L = np.array([0.2, 0.5, 1.0])[:, None, None]  # (3, 1, 1)
    RG = np.array([2.0, 10.0])[None, :, None]  # (1, 2, 1)
    kappa = np.array([0.0, 1.0, 10.0, np.inf])[None, None, :]  # (1, 1, 4)

    out = secm_approach_curve(L, RG=RG, kappa=kappa)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2, 4)


def test_secm_approach_curve_validation():
    """Verify validation on kappa, L, and RG."""
    with pytest.raises(ValueError, match="Substrate kinetics parameter kappa"):
        secm_approach_curve(1.0, RG=10.0, kappa=-0.5)

    with pytest.raises(ValueError, match="Normalized distance L"):
        secm_approach_curve(-1.0, RG=10.0, kappa=1.0)

    with pytest.raises(ValueError, match="Insulator ratio RG"):
        secm_approach_curve(1.0, RG=0.9, kappa=1.0)


# ==============================================================================
# 4. Dimensional Tip Current & Limiting Current Tests
# ==============================================================================


def test_secm_limiting_current_infinite_scalar():
    """Verify Saito limiting current at infinite distance."""
    radius = 1e-5  # 10 um
    n = 1
    D = 1e-5  # cm^2/s = 1e-9 m^2/s
    c_bulk = 1e-3  # mol/cm^3 = 1 mol/m^3
    F = FARADAY

    expected = 4.0 * n * F * D * c_bulk * radius
    actual = secm_limiting_current_infinite(radius, n=n, D=D, c_bulk=c_bulk, F=F)

    assert isinstance(actual, float)
    assert math.isclose(actual, expected, rel_tol=1e-14)


def test_secm_limiting_current_infinite_vector():
    """Verify Saito limiting current vectorization."""
    radii = np.array([5e-6, 10e-6, 25e-6])
    expected = 4.0 * 1 * FARADAY * 1e-5 * 1e-3 * radii
    actual = secm_limiting_current_infinite(radii)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == radii.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_secm_tip_current_scaling_and_sign():
    """Verify dimensional current scaling and IUPAC sign conventions."""
    L = 1.0
    rg = 10.0
    radius = 1.25e-5  # 12.5 um
    n = 1
    D = 1e-5
    c_bulk = 1e-3

    i_inf = 4.0 * n * FARADAY * D * c_bulk * radius
    i_norm_pos = secm_approach_positive_feedback(L, RG=rg)

    # Cathodic reduction (default): negative
    i_cat = secm_tip_current(
        L,
        RG=rg,
        radius=radius,
        n=n,
        D=D,
        c_bulk=c_bulk,
        reduction=True,
    )
    assert isinstance(i_cat, float)
    assert i_cat < 0.0
    assert math.isclose(i_cat, -i_norm_pos * i_inf, rel_tol=1e-14)

    # Anodic oxidation: positive
    i_an = secm_tip_current(
        L,
        RG=rg,
        radius=radius,
        n=n,
        D=D,
        c_bulk=c_bulk,
        reduction=False,
    )
    assert isinstance(i_an, float)
    assert i_an > 0.0
    assert math.isclose(i_an, i_norm_pos * i_inf, rel_tol=1e-14)


def test_secm_tip_current_array():
    """Verify dimensional current with array distances."""
    L = np.array([0.5, 1.0, 2.0])
    i_arr = secm_tip_current(L, RG=10.0, radius=1e-5)
    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == L.shape
    assert np.all(i_arr < 0)


# ==============================================================================
# 5. Top-Level Namespace and Subpackage Re-exports
# ==============================================================================


def test_secm_reexports():
    """Verify that all SECM analytical functions are accessible via package exports."""
    # Via softpotato.analytical
    assert hasattr(sp.analytical, "secm")
    assert hasattr(sp.analytical, "secm_approach_positive_feedback")
    assert hasattr(sp.analytical, "secm_approach_negative_feedback")
    assert hasattr(sp.analytical, "secm_approach_curve")
    assert hasattr(sp.analytical, "secm_limiting_current_infinite")
    assert hasattr(sp.analytical, "secm_tip_current")

    # Via top-level softpotato
    assert hasattr(sp, "secm_approach_positive_feedback")
    assert hasattr(sp, "secm_approach_negative_feedback")
    assert hasattr(sp, "secm_approach_curve")
    assert hasattr(sp, "secm_limiting_current_infinite")
    assert hasattr(sp, "secm_tip_current")
