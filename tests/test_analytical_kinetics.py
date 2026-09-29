"""Tests for thermodynamics and interfacial kinetics (softpotato.analytical.kinetics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softpotato.analytical.kinetics import (
    TafelResult,
    butler_volmer,
    butler_volmer_current_density,
    butler_volmer_linear,
    charge_transfer_resistance,
    exchange_current,
    exchange_current_density,
    nernst,
    nernst_equilibrium_concentrations,
    nernst_potential,
    nernst_ratio,
    tafel,
    tafel_analysis,
    tafel_overpotential,
    tafel_slope,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

# ==============================================================================
# 1. Thermodynamics: Nernst Equation Tests
# ==============================================================================


def test_nernst_equal_concentrations():
    """Verify Nernst equation returns formal potential E0 when c_ox == c_red."""
    # E0 = 0.0 V
    assert math.isclose(float(nernst(1.0, 1.0, E0=0.0)), 0.0, abs_tol=1e-12)
    # E0 = 0.45 V
    assert math.isclose(float(nernst(5.0e-3, 5.0e-3, E0=0.45)), 0.45, rel_tol=1e-12)
    # Alias nernst_potential
    assert math.isclose(float(nernst_potential(2.0, 2.0, E0=0.25)), 0.25, rel_tol=1e-12)


def test_nernst_slope_and_temperature():
    """Verify standard Nernstian slope (2.303 RT/nF) at standard and elevated temperatures."""
    # At 298.15 K, n=1: 10-fold concentration ratio produces ~59.16 mV shift
    expected_shift_n1 = (GAS_CONSTANT * STANDARD_TEMPERATURE / FARADAY) * math.log(10.0)
    e_val_n1 = nernst(10.0, 1.0, E0=0.0, n=1, T=STANDARD_TEMPERATURE)
    assert math.isclose(float(e_val_n1), expected_shift_n1, rel_tol=1e-10)
    assert math.isclose(float(e_val_n1), 0.0591596, rel_tol=1e-4)

    # For n=2: 10-fold ratio produces half the shift (~29.58 mV)
    e_val_n2 = nernst(10.0, 1.0, E0=0.0, n=2, T=STANDARD_TEMPERATURE)
    assert math.isclose(float(e_val_n2), expected_shift_n1 / 2.0, rel_tol=1e-10)

    # At 350.0 K
    expected_shift_350k = (GAS_CONSTANT * 350.0 / FARADAY) * math.log(10.0)
    e_val_350k = nernst(10.0, 1.0, E0=0.0, n=1, T=350.0)
    assert math.isclose(float(e_val_350k), expected_shift_350k, rel_tol=1e-10)


def test_nernst_ratio_input():
    """Verify nernst when called with a single ratio argument (c_red is None)."""
    # Ratio = 1.0 => E0
    assert math.isclose(float(nernst(1.0, E0=0.5)), 0.5, rel_tol=1e-12)

    # Ratio = 100.0 => E0 + 2 * (ln(10) RT/F)
    two_decades = 2.0 * (GAS_CONSTANT * STANDARD_TEMPERATURE / FARADAY) * math.log(10.0)
    assert math.isclose(float(nernst(100.0, E0=0.1)), 0.1 + two_decades, rel_tol=1e-10)


def test_nernst_vectorization():
    """Verify nernst handles NumPy arrays, broadcasting, and returns correct shapes."""
    c_ox = np.array([1.0, 10.0, 100.0])
    c_red = 1.0
    e_vec = nernst(c_ox, c_red, E0=0.0)
    assert isinstance(e_vec, np.ndarray)
    assert e_vec.shape == (3,)
    np.testing.assert_allclose(
        e_vec,
        [0.0, 0.0591596, 0.1183192],
        rtol=1e-4,
    )

    # Both arrays
    c_ox_2d = np.array([[1.0, 10.0], [100.0, 1000.0]])
    c_red_2d = np.array([[1.0, 1.0], [1.0, 1.0]])
    e_2d = nernst(c_ox_2d, c_red_2d, E0=0.0)
    assert isinstance(e_2d, np.ndarray)
    assert e_2d.shape == (2, 2)


def test_nernst_zero_concentrations():
    """Verify asymptotic +/- inf behavior when one concentration is zero."""
    # c_ox = 0, c_red > 0 => -inf
    val_neg = float(nernst(0.0, 1.0, E0=0.0))
    assert np.isneginf(val_neg)

    # c_red = 0, c_ox > 0 => +inf
    val_pos = float(nernst(1.0, 0.0, E0=0.0))
    assert np.isposinf(val_pos)

    # Vector with zeros
    c_ox = np.array([0.0, 1.0, 1.0])
    c_red = np.array([1.0, 1.0, 0.0])
    res = nernst(c_ox, c_red, E0=0.0)
    assert np.isneginf(res[0])
    assert math.isclose(res[1], 0.0, abs_tol=1e-12)
    assert np.isposinf(res[2])


def test_nernst_invalid_inputs():
    """Verify nernst raises appropriate ValueErrors on invalid physical inputs."""
    # Negative concentration
    with pytest.raises(ValueError, match="must be non-negative"):
        nernst(-1.0, 1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        nernst(1.0, -1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        nernst(-0.5)

    # Both zero
    with pytest.raises(ValueError, match="cannot both be zero"):
        nernst(0.0, 0.0)

    # Invalid n, T, R, F
    with pytest.raises(ValueError, match="Number of electrons n must be positive"):
        nernst(1.0, 1.0, n=0)
    with pytest.raises(ValueError, match="Temperature T must be positive"):
        nernst(1.0, 1.0, T=-10.0)
    with pytest.raises(ValueError, match="Faraday constant F must be positive"):
        nernst(1.0, 1.0, F=0)
    with pytest.raises(ValueError, match="Molar gas constant R must be positive"):
        nernst(1.0, 1.0, R=-1)


def test_nernst_ratio_and_roundtrip():
    """Verify nernst_ratio and mathematical roundtrip with nernst."""
    # At E = E0, ratio is 1.0
    assert math.isclose(float(nernst_ratio(0.25, E0=0.25)), 1.0, rel_tol=1e-12)

    # Potential shift of 59.16 mV => ratio of 10.0
    shift = (GAS_CONSTANT * STANDARD_TEMPERATURE / FARADAY) * math.log(10.0)
    ratio_recovered = nernst_ratio(shift, E0=0.0)
    assert math.isclose(float(ratio_recovered), 10.0, rel_tol=1e-10)

    # Roundtrip: nernst(nernst_ratio(E)) == E
    e_targets = np.linspace(-0.2, 0.2, 9)
    ratios = nernst_ratio(e_targets, E0=0.1)
    e_reconstructed = nernst(ratios, E0=0.1)
    np.testing.assert_allclose(e_reconstructed, e_targets, rtol=1e-12)


def test_nernst_equilibrium_concentrations():
    """Verify nernst_equilibrium_concentrations fractions, sum conservation, and stability."""
    # At E = E0, both concentrations are exactly c_total / 2
    c_ox, c_red = nernst_equilibrium_concentrations(0.2, c_total=4.0, E0=0.2)
    assert isinstance(c_ox, float) and isinstance(c_red, float)
    assert math.isclose(c_ox, 2.0, rel_tol=1e-12)
    assert math.isclose(c_red, 2.0, rel_tol=1e-12)

    # Vectorized check: sum must equal c_total across all potentials
    e_grid = np.linspace(-0.5, 0.5, 51)
    c_ox_vec, c_red_vec = nernst_equilibrium_concentrations(e_grid, c_total=1.5, E0=0.0)
    np.testing.assert_allclose(c_ox_vec + c_red_vec, 1.5, rtol=1e-12)

    # Extreme potentials: must not raise OverflowError
    c_ox_hi, c_red_hi = nernst_equilibrium_concentrations(10.0, c_total=1.0, E0=0.0)
    assert math.isclose(c_ox_hi, 1.0, rel_tol=1e-12)
    assert math.isclose(c_red_hi, 0.0, abs_tol=1e-12)

    c_ox_lo, c_red_lo = nernst_equilibrium_concentrations(-10.0, c_total=1.0, E0=0.0)
    assert math.isclose(c_ox_lo, 0.0, abs_tol=1e-12)
    assert math.isclose(c_red_lo, 1.0, rel_tol=1e-12)

    # Negative total concentration raises ValueError
    with pytest.raises(ValueError, match="c_total must be non-negative"):
        nernst_equilibrium_concentrations(0.0, c_total=-1.0)


# ==============================================================================
# 2. Exchange Current & Charge Transfer Resistance Tests
# ==============================================================================


def test_exchange_current_density_and_current():
    """Verify exchange current density and current scaling."""
    k0 = 1e-2  # cm/s
    c_ox = 1e-3  # mol/cm^3
    c_red = 1e-3  # mol/cm^3
    alpha = 0.5
    n = 1
    area = 2.0  # cm^2

    expected_j0 = n * FARADAY * k0 * c_ox
    j0 = exchange_current_density(k0, c_ox=c_ox, c_red=c_red, alpha=alpha, n=n)
    assert math.isclose(j0, expected_j0, rel_tol=1e-12)

    # Area scaling
    i0 = exchange_current(k0, c_ox=c_ox, c_red=c_red, alpha=alpha, area=area, n=n)
    assert math.isclose(i0, j0 * area, rel_tol=1e-12)

    # Asymmetric concentrations
    j0_asym = exchange_current_density(k0, c_ox=4e-3, c_red=1e-3, alpha=0.5, n=1)
    # (4e-3)^0.5 * (1e-3)^0.5 = 2e-3
    assert math.isclose(j0_asym, n * FARADAY * k0 * 2e-3, rel_tol=1e-12)


def test_charge_transfer_resistance():
    """Verify charge transfer resistance R_ct = RT / (n F I0)."""
    i0 = 1.0e-3  # A
    n = 1
    t_val = 298.15
    expected_rct = (GAS_CONSTANT * t_val) / (n * FARADAY * i0)

    r_ct = charge_transfer_resistance(i0, n=n, T=t_val)
    assert math.isclose(r_ct, expected_rct, rel_tol=1e-12)
    assert math.isclose(r_ct, 25.6929, rel_tol=1e-4)

    # Input validation
    with pytest.raises(ValueError, match="Exchange current i0 must be positive"):
        charge_transfer_resistance(0.0)
    with pytest.raises(ValueError, match="Exchange current i0 must be positive"):
        charge_transfer_resistance(-1.0e-4)


# ==============================================================================
# 3. Butler–Volmer Equation Tests
# ==============================================================================


def test_butler_volmer_zero_overpotential():
    """Verify current is strictly zero at equilibrium (eta = 0)."""
    assert math.isclose(float(butler_volmer(0.0, i0=2.5e-3)), 0.0, abs_tol=1e-15)
    assert math.isclose(
        float(butler_volmer_current_density(0.0, j0=10.0)), 0.0, abs_tol=1e-15
    )


def test_butler_volmer_symmetry():
    """Verify Butler–Volmer current is anti-symmetric for alpha = 0.5."""
    i0 = 1.5e-3
    eta_vals = np.array([0.01, 0.025, 0.05, 0.1, 0.2])

    i_anodic = butler_volmer(eta_vals, i0=i0, alpha=0.5)
    i_cathodic = butler_volmer(-eta_vals, i0=i0, alpha=0.5)

    assert isinstance(i_anodic, np.ndarray)
    np.testing.assert_allclose(i_anodic, -i_cathodic, rtol=1e-12)


def test_butler_volmer_asymmetry():
    """Verify asymmetric Butler–Volmer response for alpha != 0.5."""
    i0 = 1.0
    eta = 0.05
    # Anodic: (1 - alpha) = 0.3; Cathodic: alpha = 0.7
    i_pos = butler_volmer(eta, i0=i0, alpha=0.7)
    i_neg = butler_volmer(-eta, i0=i0, alpha=0.7)

    # For alpha = 0.7, cathodic branch has higher sensitivity than anodic
    assert abs(i_neg) > abs(i_pos)


def test_butler_volmer_linear_asymptotic():
    """Verify Butler–Volmer approaches linear approximation as eta -> 0."""
    i0 = 2.0e-3
    r_ct = charge_transfer_resistance(i0, n=1)

    # Small overpotentials: eta = 1 mV, 2 mV, 5 mV
    etas = np.array([0.001, 0.002, 0.005])
    i_bv = butler_volmer(etas, i0=i0, n=1)
    i_lin = butler_volmer_linear(etas, i0=i0, n=1)
    i_ohm = etas / r_ct

    # Compare linear formula with Ohm's law via R_ct
    np.testing.assert_allclose(i_lin, i_ohm, rtol=1e-12)

    # For eta <= 5 mV, deviation between full BV and linear approximation is < 0.2%
    rel_error = np.abs(i_bv - i_lin) / i_bv
    assert np.all(rel_error < 0.002)


def test_butler_volmer_sign_convention():
    """Verify toggle between IUPAC (anodic positive) and US (cathodic positive) conventions."""
    eta = 0.05
    i0 = 1.0
    i_iupac = butler_volmer(eta, i0=i0, anodic_positive=True)
    i_us = butler_volmer(eta, i0=i0, anodic_positive=False)

    assert i_iupac > 0.0
    assert i_us < 0.0
    assert math.isclose(i_iupac, -i_us, rel_tol=1e-12)


def test_butler_volmer_mass_transport_correction():
    """Verify generalized Butler–Volmer with surface concentration correction."""
    i0 = 1.0e-3
    eta = 0.1

    # Activation control baseline
    i_act = butler_volmer(eta, i0=i0)

    # Depleted reduced species at the surface suppresses anodic oxidation current
    i_depleted_red = butler_volmer(
        eta,
        i0=i0,
        c_red_surf=0.5,
        c_red_bulk=1.0,
    )
    assert i_depleted_red < i_act

    # Zero reduced species at surface eliminates anodic oxidation term completely
    i_zero_red = butler_volmer(
        eta,
        i0=i0,
        c_red_surf=0.0,
        c_red_bulk=1.0,
    )
    assert i_zero_red < 0.0  # Only residual cathodic counter-current remains


def test_butler_volmer_invalid_inputs():
    """Verify parameter validations in Butler–Volmer."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        butler_volmer(0.05, alpha=1.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        butler_volmer(0.05, alpha=0.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        butler_volmer(0.05, i0=-1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        butler_volmer(0.05, c_ox_surf=-0.1)
    with pytest.raises(ValueError, match="must be positive"):
        butler_volmer(0.05, c_ox_surf=0.5, c_ox_bulk=-1.0)


# ==============================================================================
# 4. Tafel Approximation & Slope Tests
# ==============================================================================


def test_tafel_slope():
    """Verify theoretical Tafel slope b for anodic and cathodic branches."""
    # At 298.15 K, n=1, alpha=0.5:
    # b_a = ln(10) RT / (0.5 * F) ~ 0.11822 V/dec (118.2 mV/dec)
    b_a = tafel_slope(alpha=0.5, n=1, branch="anodic")
    assert math.isclose(b_a * 1000.0, 118.22, rel_tol=1e-3)

    # Signed cathodic slope
    b_c = tafel_slope(alpha=0.5, n=1, branch="cathodic", signed=True)
    assert math.isclose(b_c * 1000.0, -118.22, rel_tol=1e-3)

    # Unsigned cathodic slope
    b_c_mag = tafel_slope(alpha=0.5, n=1, branch="cathodic", signed=False)
    assert math.isclose(b_c_mag * 1000.0, 118.22, rel_tol=1e-3)

    # Alpha = 0.25 => b_a = ln(10)RT / (0.75 F) ~ 78.8 mV/dec
    b_a_075 = tafel_slope(alpha=0.25, n=1, branch="anodic")
    assert math.isclose(b_a_075 * 1000.0, 78.81, rel_tol=1e-3)

    with pytest.raises(ValueError, match="Invalid branch"):
        tafel_slope(branch="unknown")


def test_tafel_branches_and_convergence():
    """Verify Tafel approximations and convergence to Butler–Volmer at high overpotential."""
    i0 = 1.0e-4
    alpha = 0.5
    n = 1

    # At high overpotentials (|eta| >= 250 mV), counter-reaction contributes < 0.01%
    eta_high_anodic = 0.250
    i_bv_anodic = butler_volmer(eta_high_anodic, i0=i0, alpha=alpha, n=n)
    i_tafel_anodic = tafel(eta_high_anodic, i0=i0, alpha=alpha, n=n, branch="anodic")
    assert math.isclose(i_bv_anodic, i_tafel_anodic, rel_tol=1e-4)

    eta_high_cathodic = -0.250
    i_bv_cathodic = butler_volmer(eta_high_cathodic, i0=i0, alpha=alpha, n=n)
    i_tafel_cathodic = tafel(
        eta_high_cathodic, i0=i0, alpha=alpha, n=n, branch="cathodic"
    )
    assert math.isclose(i_bv_cathodic, i_tafel_cathodic, rel_tol=1e-4)

    # Branch = 'auto' automatically selects appropriate branch
    etas = np.array([0.25, -0.25])
    i_auto = tafel(etas, i0=i0, alpha=alpha, branch="auto")
    assert math.isclose(i_auto[0], i_tafel_anodic, rel_tol=1e-12)
    assert math.isclose(i_auto[1], i_tafel_cathodic, rel_tol=1e-12)


def test_tafel_overpotential_roundtrip():
    """Verify roundtrip between tafel and tafel_overpotential."""
    i0 = 2.0e-4
    alpha = 0.55
    eta_orig = 0.150

    i_calc = tafel(eta_orig, i0=i0, alpha=alpha, branch="anodic")
    eta_reconstructed = tafel_overpotential(i_calc, i0=i0, alpha=alpha, branch="anodic")
    assert math.isclose(float(eta_reconstructed), eta_orig, rel_tol=1e-10)

    # Cathodic
    eta_cath = -0.150
    i_calc_c = tafel(eta_cath, i0=i0, alpha=alpha, branch="cathodic")
    eta_rec_c = tafel_overpotential(i_calc_c, i0=i0, alpha=alpha, branch="cathodic")
    assert math.isclose(float(eta_rec_c), eta_cath, rel_tol=1e-10)

    with pytest.raises(ValueError, match="Current must be non-zero"):
        tafel_overpotential(0.0)


# ==============================================================================
# 5. Tafel Regression Diagnostics (tafel_analysis) Tests
# ==============================================================================


def test_tafel_analysis_exact_recovery():
    """Verify exact parameter recovery from pure Tafel data and realistic Butler–Volmer data."""
    true_i0 = 2.5e-4  # A
    true_alpha = 0.45
    n = 1
    T = STANDARD_TEMPERATURE

    # 1. Exact mathematical recovery from pure Tafel data (machine precision)
    eta_pts = np.linspace(0.15, 0.30, 40)
    current_tafel = tafel(eta_pts, i0=true_i0, alpha=true_alpha, n=n, T=T, branch="anodic")
    res_exact = tafel_analysis(eta_pts, current_tafel, branch="anodic", n=n, T=T)
    assert isinstance(res_exact, TafelResult)
    assert res_exact.branch == "anodic"
    assert res_exact.r_squared > 0.99999999
    assert math.isclose(res_exact.i0, true_i0, rel_tol=1e-10)
    assert math.isclose(res_exact.alpha, true_alpha, rel_tol=1e-10)

    expected_slope = tafel_slope(alpha=true_alpha, n=n, branch="anodic", T=T)
    assert math.isclose(res_exact.slope, expected_slope, rel_tol=1e-10)
    assert math.isclose(res_exact.slope_mv, expected_slope * 1000.0, rel_tol=1e-10)

    # 2. Realistic recovery from Butler–Volmer data (within ~0.5% due to minor back-reaction)
    current_bv = butler_volmer(eta_pts, i0=true_i0, alpha=true_alpha, n=n, T=T)
    res_bv = tafel_analysis(eta_pts, current_bv, branch="anodic", n=n, T=T)
    assert res_bv.r_squared > 0.9999
    assert math.isclose(res_bv.i0, true_i0, rel_tol=0.005)
    assert math.isclose(res_bv.alpha, true_alpha, rel_tol=0.005)
    assert math.isclose(res_bv.slope, expected_slope, rel_tol=0.005)


def test_tafel_analysis_cathodic():
    """Verify parameter recovery on cathodic polarization data."""
    true_i0 = 5.0e-5  # A
    true_alpha = 0.60
    n = 1

    eta_pts = np.linspace(-0.30, -0.15, 50)

    # 1. Exact recovery from cathodic Tafel branch
    current_tafel_c = tafel(
        eta_pts, i0=true_i0, alpha=true_alpha, n=n, branch="cathodic"
    )
    res_exact = tafel_analysis(eta_pts, current_tafel_c, branch="cathodic", n=n)
    assert res_exact.branch == "cathodic"
    assert res_exact.r_squared > 0.99999999
    assert math.isclose(res_exact.i0, true_i0, rel_tol=1e-10)
    assert math.isclose(res_exact.alpha, true_alpha, rel_tol=1e-10)
    expected_slope_c = tafel_slope(alpha=true_alpha, n=n, branch="cathodic", signed=True)
    assert math.isclose(res_exact.slope, expected_slope_c, rel_tol=1e-10)

    # 2. Realistic recovery from cathodic Butler-Volmer
    current_bv_c = butler_volmer(eta_pts, i0=true_i0, alpha=true_alpha, n=n)
    res_bv = tafel_analysis(eta_pts, current_bv_c, branch="cathodic", n=n)
    assert res_bv.branch == "cathodic"
    assert res_bv.r_squared > 0.9999
    assert math.isclose(res_bv.i0, true_i0, rel_tol=0.005)
    assert math.isclose(res_bv.alpha, true_alpha, rel_tol=0.005)
    assert math.isclose(res_bv.slope, expected_slope_c, rel_tol=0.005)


def test_tafel_analysis_with_k0_extraction():
    """Verify k0 extraction when electrode area and bulk concentration are provided."""
    true_k0 = 1.5e-3  # cm/s
    area = 0.196  # cm^2 (e.g. 5 mm disk)
    c_bulk = 1e-3  # mol/cm^3
    n = 1
    true_i0 = exchange_current(
        true_k0, c_ox=c_bulk, c_red=c_bulk, alpha=0.5, area=area, n=n
    )

    eta_pts = np.linspace(0.12, 0.28, 40)
    i_pts = butler_volmer(eta_pts, i0=true_i0, alpha=0.5, n=n)

    res = tafel_analysis(
        eta_pts,
        i_pts,
        branch="anodic",
        area=area,
        c_bulk=c_bulk,
        n=n,
    )
    assert res.k0 is not None
    assert math.isclose(res.k0, true_k0, rel_tol=0.01)


def test_tafel_analysis_fit_range_filtering():
    """Verify user fit_range filters full potential curve correctly."""
    true_i0 = 1e-4
    # Full voltammogram from -0.3 V to +0.3 V
    eta_full = np.linspace(-0.3, 0.3, 201)
    i_full = butler_volmer(eta_full, i0=true_i0, alpha=0.5)

    # Restrict fit strictly to 0.15 V <= eta <= 0.25 V
    res = tafel_analysis(
        eta_full,
        i_full,
        branch="anodic",
        fit_range=(0.15, 0.25),
    )
    assert math.isclose(res.i0, true_i0, rel_tol=0.01)
    assert math.isclose(res.alpha, 0.5, rel_tol=0.01)


def test_tafel_analysis_error_handling():
    """Verify error handling on length mismatches, invalid ranges, and sparse data."""
    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        tafel_analysis([0.1, 0.2], [1.0])

    # Invalid fit_range
    with pytest.raises(ValueError, match="fit_range must satisfy eta_min < eta_max"):
        tafel_analysis([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], fit_range=(0.5, 0.2))

    # Fewer than 3 points
    with pytest.raises(ValueError, match="At least 3 valid data points are required"):
        tafel_analysis([0.1, 0.2], [1e-3, 2e-3], branch="anodic")

    # Invalid branch
    with pytest.raises(ValueError, match="Invalid branch"):
        tafel_analysis([0.1, 0.2, 0.3], [1e-3, 2e-3, 3e-3], branch="invalid")


# ==============================================================================
# 6. Subpackage Integration Tests
# ==============================================================================


def test_import_from_analytical_subpackage():
    """Verify all public kinetics symbols are importable from softpotato.analytical."""
    import softpotato.analytical as an

    assert hasattr(an, "kinetics")
    assert hasattr(an, "nernst")
    assert hasattr(an, "nernst_potential")
    assert hasattr(an, "nernst_ratio")
    assert hasattr(an, "nernst_equilibrium_concentrations")
    assert hasattr(an, "exchange_current")
    assert hasattr(an, "exchange_current_density")
    assert hasattr(an, "charge_transfer_resistance")
    assert hasattr(an, "butler_volmer")
    assert hasattr(an, "butler_volmer_current_density")
    assert hasattr(an, "butler_volmer_linear")
    assert hasattr(an, "tafel")
    assert hasattr(an, "tafel_slope")
    assert hasattr(an, "tafel_overpotential")
    assert hasattr(an, "tafel_analysis")
    assert hasattr(an, "TafelResult")


def test_import_from_top_level_package():
    """Verify all public kinetics symbols are importable directly from softpotato."""
    import softpotato as sp

    assert hasattr(sp, "analytical")
    assert hasattr(sp, "nernst")
    assert hasattr(sp, "nernst_potential")
    assert hasattr(sp, "nernst_ratio")
    assert hasattr(sp, "nernst_equilibrium_concentrations")
    assert hasattr(sp, "exchange_current")
    assert hasattr(sp, "exchange_current_density")
    assert hasattr(sp, "charge_transfer_resistance")
    assert hasattr(sp, "butler_volmer")
    assert hasattr(sp, "butler_volmer_current_density")
    assert hasattr(sp, "butler_volmer_linear")
    assert hasattr(sp, "tafel")
    assert hasattr(sp, "tafel_slope")
    assert hasattr(sp, "tafel_overpotential")
    assert hasattr(sp, "tafel_analysis")
    assert hasattr(sp, "TafelResult")
