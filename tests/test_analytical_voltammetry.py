"""Tests for voltammetry and kinetic diagnostics (softpotato.analytical.voltammetry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softpotato.analytical.voltammetry import (
    MatsudaAyabeResult,
    NicholsonResult,
    matsuda_ayabe,
    matsuda_ayabe_lambda,
    nicholson_delta_ep,
    nicholson_psi,
    nicholson_rate_constant,
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
    randles_sevcik_quasi,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

# ==============================================================================
# 1. Randles–Ševčík Reversible Peak Current Tests
# ==============================================================================


def test_randles_sevcik_classical_constant():
    """Verify standard Randles–Ševčík constant 2.686e5 C mol^-1/2 J^-1/2 at 25 °C."""
    # 0.4463 * sqrt(F^3 / (R * T))
    theoretical_const = 0.4463 * math.sqrt(
        (FARADAY**3) / (GAS_CONSTANT * STANDARD_TEMPERATURE)
    )
    assert math.isclose(theoretical_const, 2.686e5, rel_tol=1e-3)

    # Ip for n=1, A=1.0 cm^2, D=1e-5 cm^2/s, c*=1e-3 mol/cm^3, v=1.0 V/s
    ip = randles_sevcik(1.0, c_bulk=1e-3, D=1e-5, area=1.0, n=1, T=STANDARD_TEMPERATURE)
    expected_ip = (
        theoretical_const * (1.0**1.5) * 1.0 * math.sqrt(1e-5) * 1e-3 * math.sqrt(1.0)
    )
    assert math.isclose(ip, expected_ip, rel_tol=1e-10)


def test_randles_sevcik_proportionalities():
    """Verify physical scaling of Randles–Ševčík peak current with parameters."""
    v_base = 0.1
    c_base = 1e-6  # mol/cm^3
    D_base = 1e-5  # cm^2/s
    a_base = 0.07  # cm^2
    n_base = 1

    ip_base = randles_sevcik(v_base, c_bulk=c_base, D=D_base, area=a_base, n=n_base)

    # 1. Square root scan rate: 4x scan rate => 2x current
    ip_4v = randles_sevcik(4.0 * v_base, c_bulk=c_base, D=D_base, area=a_base, n=n_base)
    assert math.isclose(ip_4v, 2.0 * ip_base, rel_tol=1e-10)

    # 2. Linear with concentration
    ip_3c = randles_sevcik(v_base, c_bulk=3.0 * c_base, D=D_base, area=a_base, n=n_base)
    assert math.isclose(ip_3c, 3.0 * ip_base, rel_tol=1e-10)

    # 3. Linear with area
    ip_2a = randles_sevcik(v_base, c_bulk=c_base, D=D_base, area=2.0 * a_base, n=n_base)
    assert math.isclose(ip_2a, 2.0 * ip_base, rel_tol=1e-10)

    # 4. Square root with diffusivity: 4x D => 2x current
    ip_4D = randles_sevcik(v_base, c_bulk=c_base, D=4.0 * D_base, area=a_base, n=n_base)
    assert math.isclose(ip_4D, 2.0 * ip_base, rel_tol=1e-10)

    # 5. n^(3/2) dependence: n=2 => 2^(3/2) = 2.8284x current
    ip_n2 = randles_sevcik(v_base, c_bulk=c_base, D=D_base, area=a_base, n=2)
    assert math.isclose(ip_n2, (2.0**1.5) * ip_base, rel_tol=1e-10)


def test_randles_sevcik_scan_direction():
    """Verify IUPAC sign convention for cathodic and anodic sweeps."""
    ip_cathodic = randles_sevcik(0.05, scan_direction="cathodic")
    ip_anodic = randles_sevcik(0.05, scan_direction="anodic")

    assert ip_cathodic > 0.0
    assert ip_anodic < 0.0
    assert math.isclose(ip_cathodic, -ip_anodic, rel_tol=1e-12)


def test_randles_sevcik_vectorization():
    """Verify vectorization with 1D and 2D arrays, and scalar return types."""
    v_scalars = 0.2
    assert isinstance(randles_sevcik(v_scalars), float)

    v_1d = np.array([0.01, 0.04, 0.09, 0.16])
    res_1d = randles_sevcik(v_1d)
    assert isinstance(res_1d, np.ndarray)
    assert res_1d.shape == (4,)
    # Scaling check: sqrt(v) ratios 1:2:3:4
    np.testing.assert_allclose(res_1d / res_1d[0], [1.0, 2.0, 3.0, 4.0], rtol=1e-10)

    v_2d = np.array([[0.01, 0.04], [0.09, 0.16]])
    res_2d = randles_sevcik(v_2d)
    assert isinstance(res_2d, np.ndarray)
    assert res_2d.shape == (2, 2)


def test_randles_sevcik_validation():
    """Verify parameter validations and error handling."""
    with pytest.raises(ValueError, match="Scan rate v must be strictly positive"):
        randles_sevcik(0.0)
    with pytest.raises(ValueError, match="Scan rate v must be strictly positive"):
        randles_sevcik(-0.1)
    with pytest.raises(ValueError, match="Diffusion coefficient D must be positive"):
        randles_sevcik(0.1, D=-1e-5)
    with pytest.raises(ValueError, match="Electrode area must be positive"):
        randles_sevcik(0.1, area=0.0)
    with pytest.raises(ValueError, match="Number of electrons n must be positive"):
        randles_sevcik(0.1, n=0)
    with pytest.raises(ValueError, match="Temperature T must be positive"):
        randles_sevcik(0.1, T=-298.15)
    with pytest.raises(ValueError, match="Invalid scan_direction"):
        randles_sevcik(0.1, scan_direction="forward")  # type: ignore[arg-type]


# ==============================================================================
# 2. Randles–Ševčík Irreversible Peak Current Tests
# ==============================================================================


def test_randles_sevcik_irreversible_ratio():
    """Verify theoretical ratio of irreversible to reversible peak current."""
    # Ratio = (0.4958 / 0.4463) * sqrt(alpha * n_alpha / n)
    v = 0.1
    c = 1e-3
    D = 1e-5
    area = 1.0
    alpha = 0.5
    n = 1
    n_alpha = 1

    ip_rev = randles_sevcik(v, c_bulk=c, D=D, area=area, n=n)
    ip_irrev = randles_sevcik_irreversible(
        v, alpha=alpha, n_alpha=n_alpha, c_bulk=c, D=D, area=area, n=n
    )

    expected_ratio = (0.4958 / 0.4463) * math.sqrt(alpha * n_alpha / n)
    assert math.isclose(ip_irrev / ip_rev, expected_ratio, rel_tol=1e-10)
    # For alpha=0.5, ratio is approximately 0.7855
    assert math.isclose(ip_irrev / ip_rev, 0.78553, rel_tol=1e-3)


def test_randles_sevcik_irreversible_scaling():
    """Verify scaling with alpha, n_alpha, and scan rate."""
    v = 0.05
    ip_base = randles_sevcik_irreversible(v, alpha=0.25)
    ip_4alpha = randles_sevcik_irreversible(
        v, alpha=0.50
    )  # 2x alpha => sqrt(2)x current
    assert math.isclose(ip_4alpha, math.sqrt(2.0) * ip_base, rel_tol=1e-10)

    # Anodic sign
    ip_anodic = randles_sevcik_irreversible(v, scan_direction="anodic")
    assert ip_anodic < 0.0
    assert math.isclose(ip_anodic, -randles_sevcik_irreversible(v), rel_tol=1e-12)


def test_randles_sevcik_irreversible_validation():
    """Verify validation of alpha and other parameters."""
    with pytest.raises(
        ValueError, match="Transfer coefficient alpha must be strictly between 0 and 1"
    ):
        randles_sevcik_irreversible(0.1, alpha=0.0)
    with pytest.raises(
        ValueError, match="Transfer coefficient alpha must be strictly between 0 and 1"
    ):
        randles_sevcik_irreversible(0.1, alpha=1.0)
    with pytest.raises(ValueError, match="n_alpha must be positive"):
        randles_sevcik_irreversible(0.1, n_alpha=0)


# ==============================================================================
# 3. Randles–Ševčík Quasi-Reversible Peak Current Tests
# ==============================================================================


def test_randles_sevcik_quasi_limits():
    """Verify quasi-reversible peak current converges to reversible and irreversible limits."""
    v = 0.1
    c = 1e-3
    D = 1e-5
    area = 1.0
    alpha = 0.5
    n = 1

    ip_rev = randles_sevcik(v, c_bulk=c, D=D, area=area, n=n)
    ip_irrev = randles_sevcik_irreversible(
        v, alpha=alpha, c_bulk=c, D=D, area=area, n=n
    )

    # 1. Large Lambda (e.g. 500) approaches reversible limit within 0.5%
    ip_large_lam = randles_sevcik_quasi(
        v, lambda_param=500.0, c_bulk=c, D=D, area=area, n=n
    )
    assert math.isclose(ip_large_lam, ip_rev, rel_tol=0.005)

    # 2. Small Lambda (e.g. 1e-4) approaches irreversible limit within 0.5%
    ip_small_lam = randles_sevcik_quasi(
        v, lambda_param=1e-4, alpha=alpha, c_bulk=c, D=D, area=area, n=n
    )
    assert math.isclose(ip_small_lam, ip_irrev, rel_tol=0.005)

    # 3. Intermediate Lambda (e.g. Lambda = 1.0): strictly between irrev and rev
    ip_intermed = randles_sevcik_quasi(
        v, lambda_param=1.0, alpha=alpha, c_bulk=c, D=D, area=area, n=n
    )
    assert ip_irrev < ip_intermed < ip_rev


def test_randles_sevcik_quasi_with_k0():
    """Verify quasi-reversible evaluation from standard rate constant k0."""
    v = 0.1
    k0 = 1.0  # Very large k0 => high reversibility
    ip_k0_fast = randles_sevcik_quasi(v, k0=k0)
    ip_rev = randles_sevcik(v)
    assert math.isclose(ip_k0_fast, ip_rev, rel_tol=0.01)

    # Very small k0 => irreversible
    ip_k0_slow = randles_sevcik_quasi(v, k0=1e-6, alpha=0.5)
    ip_irrev = randles_sevcik_irreversible(v, alpha=0.5)
    assert math.isclose(ip_k0_slow, ip_irrev, rel_tol=0.01)


def test_randles_sevcik_quasi_validation():
    """Verify error when neither k0 nor lambda_param is provided."""
    with pytest.raises(
        ValueError, match="Either 'k0' or 'lambda_param' must be provided"
    ):
        randles_sevcik_quasi(0.1)


# ==============================================================================
# 4. Nicholson Method Tests (nicholson_psi)
# ==============================================================================


def test_nicholson_table_points():
    """Verify Nicholson method approximates original 1965 Table 1 benchmarks."""
    # Nicholson Table 1:
    # Delta_Ep (mV) vs. Psi:
    # 72 mV -> 2.0
    # 84 mV -> 1.0
    # 105 mV -> 0.50
    # 141 mV -> 0.25
    # 212 mV -> 0.10

    benchmarks = [
        (0.072, 2.0),
        (0.084, 1.0),
        (0.105, 0.50),
        (0.141, 0.25),
        (0.212, 0.10),
    ]

    for dep_v, expected_psi in benchmarks:
        # Table method should match very closely
        psi_table = nicholson_psi(dep_v, in_volts=True, method="table")
        assert math.isclose(float(psi_table), expected_psi, rel_tol=0.05)

        # Swaddle method matches within ~10-15% across the entire range
        psi_swaddle = nicholson_psi(dep_v, in_volts=True, method="swaddle")
        assert math.isclose(float(psi_swaddle), expected_psi, rel_tol=0.20)

        # Lavagnini method (valid up to ~140 mV)
        if dep_v <= 0.141:
            psi_lav = nicholson_psi(dep_v, in_volts=True, method="lavagnini")
            assert math.isclose(float(psi_lav), expected_psi, rel_tol=0.15)


def test_nicholson_forward_inverse_roundtrip():
    """Verify forward (Psi -> Delta_Ep) and inverse (Delta_Ep -> Psi) consistency."""
    psi_inputs = [0.25, 0.5, 1.0, 2.0, 5.0]

    for method in ("swaddle", "lavagnini"):
        for psi_orig in psi_inputs:
            # Forward: psi -> delta_ep (V)
            dep_calc = nicholson_delta_ep(psi_orig, n=1, in_volts=True, method=method)
            # Inverse: delta_ep -> psi
            psi_rec = nicholson_psi(dep_calc, n=1, in_volts=True, method=method)
            assert math.isclose(float(psi_rec), psi_orig, rel_tol=1e-7)


def test_nicholson_rate_constant_extraction():
    """Verify extraction of standard rate constant k0 from CV parameters."""
    true_k0 = 2.5e-3  # m/s
    v = 0.1  # V/s
    D = 1e-9  # m^2/s
    n = 1
    T = STANDARD_TEMPERATURE

    # Calculate theoretical Psi
    # Psi = k0 / sqrt(pi * D * n * F * v / (R * T))
    psi_true = true_k0 / math.sqrt(math.pi * D * n * FARADAY * v / (GAS_CONSTANT * T))

    # Calculate corresponding Delta_Ep using forward Swaddle
    dep = nicholson_delta_ep(psi_true, n=n, in_volts=True, method="swaddle")

    # Recover k0
    k0_extracted = nicholson_rate_constant(
        dep, v=v, D=D, n=n, T=T, in_volts=True, method="swaddle"
    )
    assert math.isclose(k0_extracted, true_k0, rel_tol=1e-6)

    # Test via rich NicholsonResult
    res = nicholson_psi(
        delta_ep=dep, v=v, D=D, n=n, T=T, in_volts=True, method="swaddle"
    )
    assert isinstance(res, NicholsonResult)
    assert math.isclose(res.psi, psi_true, rel_tol=1e-6)
    assert res.k0 is not None
    assert math.isclose(res.k0, true_k0, rel_tol=1e-6)
    assert math.isclose(res.delta_ep, dep, rel_tol=1e-10)
    assert math.isclose(res.delta_ep_mv, dep * 1000.0, rel_tol=1e-10)


def test_nicholson_vectorization():
    """Verify vectorization of forward and inverse Nicholson calculations."""
    deps = np.array([0.070, 0.085, 0.110, 0.150])
    psis = nicholson_psi(deps, in_volts=True, method="swaddle")
    assert isinstance(psis, np.ndarray)
    assert psis.shape == (4,)
    # Peak separation increases => Psi must monotonically decrease
    assert np.all(np.diff(psis) < 0)

    # Forward vectorization
    deps_reconstructed = nicholson_delta_ep(psis, in_volts=True, method="swaddle")
    assert isinstance(deps_reconstructed, np.ndarray)
    np.testing.assert_allclose(deps_reconstructed, deps, rtol=1e-7)


def test_nicholson_validation():
    """Verify error handling on invalid peak separations and conflicting arguments."""
    with pytest.raises(ValueError, match="Either 'delta_ep' or 'psi' must be provided"):
        nicholson_psi()
    with pytest.raises(ValueError, match="Cannot provide both 'delta_ep' and 'psi'"):
        nicholson_psi(delta_ep=0.07, psi=2.0)
    with pytest.raises(
        ValueError, match="Peak potential separation delta_ep must be positive"
    ):
        nicholson_psi(delta_ep=-0.05)
    with pytest.raises(ValueError, match="greater than 59.0 mV"):
        # For n=1, 55 mV is below the Nernstian limit for Swaddle
        nicholson_psi(delta_ep=0.055, method="swaddle")
    with pytest.raises(ValueError, match="Invalid method"):
        nicholson_psi(delta_ep=0.072, method="unknown")  # type: ignore[arg-type]


# ==============================================================================
# 5. Matsuda–Ayabe Reversibility Diagnostics Tests
# ==============================================================================


def test_matsuda_ayabe_lambda_calculation():
    """Verify exact formula calculation for Lambda = k0 / sqrt(D n F v / RT)."""
    k0 = 1e-3  # cm/s
    v = 0.2  # V/s
    D = 1e-5  # cm^2/s
    n = 1
    T = STANDARD_TEMPERATURE

    lam = matsuda_ayabe_lambda(k0=k0, v=v, D=D, n=n, T=T)
    expected_lam = k0 / math.sqrt((D * n * FARADAY * v) / (GAS_CONSTANT * T))
    assert math.isclose(float(lam), expected_lam, rel_tol=1e-12)


def test_matsuda_ayabe_classification_zones():
    """Verify reversibility classification boundaries (Lambda >= 15, 1e-3 to 15, <= 1e-3)."""
    v = 0.1
    D = 1e-5

    # 1. Reversible: Lambda >= 15
    res_rev = matsuda_ayabe(k0=0.5, v=v, D=D)
    assert isinstance(res_rev, MatsudaAyabeResult)
    assert res_rev.lambda_param >= 15.0
    assert res_rev.classification == "reversible"
    assert res_rev.is_reversible is True
    assert res_rev.is_quasi_reversible is False
    assert res_rev.is_irreversible is False

    # 2. Quasi-reversible: 10^-3 < Lambda < 15
    res_quasi = matsuda_ayabe(k0=5e-4, v=v, D=D)
    assert 1e-3 < res_quasi.lambda_param < 15.0
    assert res_quasi.classification == "quasi-reversible"
    assert res_quasi.is_reversible is False
    assert res_quasi.is_quasi_reversible is True
    assert res_quasi.is_irreversible is False

    # 3. Totally irreversible: Lambda <= 10^-3
    res_irrev = matsuda_ayabe(k0=1e-6, v=v, D=D)
    assert res_irrev.lambda_param <= 1e-3
    assert res_irrev.classification == "irreversible"
    assert res_irrev.is_reversible is False
    assert res_irrev.is_quasi_reversible is False
    assert res_irrev.is_irreversible is True


def test_matsuda_ayabe_lambda_vectorization():
    """Verify vectorization of matsuda_ayabe_lambda across scan rates and rate constants."""
    v_arr = np.array([0.01, 0.1, 1.0])
    k0 = 1e-3
    lams = matsuda_ayabe_lambda(k0, v_arr)
    assert isinstance(lams, np.ndarray)
    assert lams.shape == (3,)
    # Inverse square-root with v: 100x v => 10x smaller Lambda
    assert math.isclose(lams[0] / lams[2], 10.0, rel_tol=1e-10)


# ==============================================================================
# 6. Totally Irreversible Peak Potential Shift Tests
# ==============================================================================


def test_peak_potential_irreversible_slope():
    """Verify theoretical peak potential shift of 29.6 / (alpha * n_alpha) mV/decade."""
    alpha = 0.5
    n_alpha = 1
    k0 = 1e-4
    E0_prime = 0.2

    # Cathodic shift: Ep shifts negatively with increasing v
    v1 = 0.01
    v2 = 0.10  # 1 decade increase
    ep1 = peak_potential_irreversible(
        v1, k0=k0, E0_prime=E0_prime, alpha=alpha, n_alpha=n_alpha
    )
    ep2 = peak_potential_irreversible(
        v2, k0=k0, E0_prime=E0_prime, alpha=alpha, n_alpha=n_alpha
    )

    expected_shift = (math.log(10.0) * GAS_CONSTANT * STANDARD_TEMPERATURE) / (
        2.0 * alpha * n_alpha * FARADAY
    )
    assert math.isclose(ep1 - ep2, expected_shift, rel_tol=1e-10)
    # For alpha=0.5, 2 * alpha = 1.0, so shift is 59.16 mV
    assert math.isclose(ep1 - ep2, 0.05916, rel_tol=1e-3)

    # Anodic shift: Ep shifts positively with increasing v
    ep1_an = peak_potential_irreversible(
        v1,
        k0=k0,
        E0_prime=E0_prime,
        alpha=alpha,
        n_alpha=n_alpha,
        scan_direction="anodic",
    )
    ep2_an = peak_potential_irreversible(
        v2,
        k0=k0,
        E0_prime=E0_prime,
        alpha=alpha,
        n_alpha=n_alpha,
        scan_direction="anodic",
    )
    assert math.isclose(ep2_an - ep1_an, expected_shift, rel_tol=1e-10)


def test_peak_potential_irreversible_vectorization():
    """Verify vectorization across multiple scan rates."""
    v_pts = np.logspace(-2, 0, 5)
    eps = peak_potential_irreversible(v_pts, k0=1e-4)
    assert isinstance(eps, np.ndarray)
    assert eps.shape == (5,)
    # Cathodic Ep decreases monotonically with scan rate
    assert np.all(np.diff(eps) < 0)


# ==============================================================================
# 7. Subpackage and Top-Level Package Exports Tests
# ==============================================================================


def test_import_from_analytical_subpackage():
    """Verify all public voltammetry symbols are exposed in softpotato.analytical."""
    import softpotato.analytical as an

    assert hasattr(an, "voltammetry")
    assert hasattr(an, "randles_sevcik")
    assert hasattr(an, "randles_sevcik_irreversible")
    assert hasattr(an, "randles_sevcik_quasi")
    assert hasattr(an, "nicholson_psi")
    assert hasattr(an, "nicholson_delta_ep")
    assert hasattr(an, "nicholson_rate_constant")
    assert hasattr(an, "matsuda_ayabe")
    assert hasattr(an, "matsuda_ayabe_lambda")
    assert hasattr(an, "peak_potential_irreversible")
    assert hasattr(an, "NicholsonResult")
    assert hasattr(an, "MatsudaAyabeResult")


def test_import_from_top_level_package():
    """Verify all public voltammetry symbols are exposed at top-level in softpotato."""
    import softpotato as sp

    assert hasattr(sp, "voltammetry") or hasattr(sp.analytical, "voltammetry")
    assert hasattr(sp, "randles_sevcik")
    assert hasattr(sp, "randles_sevcik_irreversible")
    assert hasattr(sp, "randles_sevcik_quasi")
    assert hasattr(sp, "nicholson_psi")
    assert hasattr(sp, "nicholson_delta_ep")
    assert hasattr(sp, "nicholson_rate_constant")
    assert hasattr(sp, "matsuda_ayabe")
    assert hasattr(sp, "matsuda_ayabe_lambda")
    assert hasattr(sp, "peak_potential_irreversible")
    assert hasattr(sp, "NicholsonResult")
    assert hasattr(sp, "MatsudaAyabeResult")
