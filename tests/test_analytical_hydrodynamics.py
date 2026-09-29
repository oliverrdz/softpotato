"""Tests for hydrodynamic and convection equations (softpotato.analytical.hydrodynamics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.hydrodynamics import (
    KouteckyLevichResult,
    RRDEResult,
    collection_efficiency,
    koutecky_levich,
    koutecky_levich_analysis,
    levich,
    levich_constant,
    nernst_diffusion_layer,
    rad_s_to_rpm,
    ring_collection_current,
    ring_limiting_current,
    rotating_ring_disk,
    rpm_to_rad_s,
    shielding_factor,
)
from softpotato.constants import FARADAY

# ==============================================================================
# 1. Rotational Unit Conversions
# ==============================================================================


def test_rpm_to_rad_s():
    """Verify rpm_to_rad_s conversions and scalar/array handling."""
    # 60 rpm = 2 * pi rad/s, 900 rpm = 30 * pi rad/s
    assert math.isclose(float(rpm_to_rad_s(60.0)), 2.0 * math.pi, rel_tol=1e-12)
    assert math.isclose(float(rpm_to_rad_s(900.0)), 30.0 * math.pi, rel_tol=1e-12)

    # Vectorized
    rpms = np.array([0.0, 60.0, 120.0, 600.0])
    w_vec = rpm_to_rad_s(rpms)
    assert isinstance(w_vec, np.ndarray)
    np.testing.assert_allclose(w_vec, (2.0 * np.pi / 60.0) * rpms, rtol=1e-12)

    # Input validation
    with pytest.raises(ValueError, match="rpm must be non-negative"):
        rpm_to_rad_s(-10.0)


def test_rad_s_to_rpm():
    """Verify rad_s_to_rpm conversions and round-trip consistency."""
    assert math.isclose(float(rad_s_to_rpm(2.0 * math.pi)), 60.0, rel_tol=1e-12)

    # Round trip
    orig_rpm = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    w = rpm_to_rad_s(orig_rpm)
    recovered_rpm = rad_s_to_rpm(w)
    np.testing.assert_allclose(recovered_rpm, orig_rpm, rtol=1e-12)

    # Input validation
    with pytest.raises(ValueError, match="omega must be non-negative"):
        rad_s_to_rpm(-1.0)


# ==============================================================================
# 2. Levich Equation Tests
# ==============================================================================


def test_levich_scalar_and_vector():
    """Verify Levich equation against exact analytical values."""
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 1.0
    nu = 0.01
    w_val = 100.0  # rad/s

    expected = (
        0.620
        * n
        * FARADAY
        * area
        * (D ** (2.0 / 3.0))
        * (nu ** (-1.0 / 6.0))
        * c_bulk
        * math.sqrt(w_val)
    )
    i_scalar = levich(w_val, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    assert isinstance(i_scalar, float)
    assert math.isclose(i_scalar, expected, rel_tol=1e-12)

    # RPM input
    rpm_val = float(rad_s_to_rpm(w_val))
    i_rpm = levich(rpm=rpm_val, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    assert isinstance(i_rpm, float)
    assert math.isclose(i_rpm, expected, rel_tol=1e-12)

    # Vectorized
    w_arr = np.array([25.0, 100.0, 400.0])
    i_arr = levich(w_arr, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    assert isinstance(i_arr, np.ndarray)
    np.testing.assert_allclose(i_arr, expected * np.sqrt(w_arr / w_val), rtol=1e-12)


def test_levich_zero_rotation():
    """Verify Levich equation returns 0 at omega = 0."""
    assert levich(0.0) == 0.0
    arr = levich(np.array([0.0, 100.0]))
    assert arr[0] == 0.0
    assert arr[1] > 0.0


def test_levich_scaling_laws():
    """Verify physical scaling laws of the Levich equation."""
    base = levich(100.0, D=1e-5, nu=0.01, c_bulk=1e-3)

    # 4x rotation -> 2x current (sqrt(omega))
    quad_w = levich(400.0, D=1e-5, nu=0.01, c_bulk=1e-3)
    assert math.isclose(quad_w, 2.0 * base, rel_tol=1e-12)

    # 8x diffusivity -> 4x current (D^(2/3))
    oct_d = levich(100.0, D=8e-5, nu=0.01, c_bulk=1e-3)
    assert math.isclose(oct_d, 4.0 * base, rel_tol=1e-12)

    # 64x viscosity -> 0.5x current (nu^(-1/6))
    visc_nu = levich(100.0, D=1e-5, nu=0.64, c_bulk=1e-3)
    assert math.isclose(visc_nu, 0.5 * base, rel_tol=1e-12)


def test_levich_constant_and_nernst_layer():
    """Verify Levich constant B and Nernst diffusion layer thickness delta."""
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 0.5
    nu = 0.01
    w = 100.0

    b = levich_constant(n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    i_lim = levich(w, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    assert math.isclose(i_lim, b * math.sqrt(w), rel_tol=1e-12)

    # Nernst stagnant diffusion layer: delta = 1.610 * D^(1/3) * nu^(1/6) * w^(-1/2)
    delta = nernst_diffusion_layer(w, D=D, nu=nu)
    expected_delta = 1.610 * (D ** (1.0 / 3.0)) * (nu ** (1.0 / 6.0)) / math.sqrt(w)
    assert math.isclose(float(delta), expected_delta, rel_tol=1e-12)

    # Vectorized delta
    delta_arr = nernst_diffusion_layer(np.array([25.0, 100.0]), D=D, nu=nu)
    assert isinstance(delta_arr, np.ndarray)
    assert delta_arr[0] == 2.0 * delta_arr[1]

    # Approximate link: I_L = n * F * A * D * c* / delta
    # Since 0.620 * 1.610 = 0.9982 ~= 1.0
    i_from_delta = n * FARADAY * area * D * c_bulk / float(delta)
    assert math.isclose(i_lim, i_from_delta, rel_tol=2e-3)


def test_levich_input_validation():
    """Verify invalid parameters in levich raise ValueErrors."""
    with pytest.raises(ValueError, match="Either angular velocity 'omega'"):
        levich()

    with pytest.raises(ValueError, match="Specify either 'omega' or 'rpm'"):
        levich(omega=100.0, rpm=1000.0)

    with pytest.raises(ValueError, match="omega must be non-negative"):
        levich(-5.0)

    with pytest.raises(ValueError, match="Number of electrons n must be positive"):
        levich(100.0, n=0)

    with pytest.raises(ValueError, match="Diffusion coefficient D must be positive"):
        levich(100.0, D=-1e-5)

    with pytest.raises(
        ValueError, match="Bulk concentration c_bulk must be non-negative"
    ):
        levich(100.0, c_bulk=-1.0)

    with pytest.raises(ValueError, match="Electrode area must be positive"):
        levich(100.0, area=0.0)

    with pytest.raises(ValueError, match="Kinematic viscosity nu must be positive"):
        levich(100.0, nu=0.0)


# ==============================================================================
# 3. Koutecký–Levich Equation Tests
# ==============================================================================


def test_koutecky_levich_limits():
    """Verify Koutecký–Levich converges to kinetic or diffusion limits."""
    # Fast kinetics limit: I_K -> inf ==> I -> I_L
    i_l = 1e-3
    i_fast_kinetics = koutecky_levich(i_k=float("inf"), i_lim=i_l)
    assert math.isclose(i_fast_kinetics, i_l, rel_tol=1e-12)

    # Fast mass-transport limit: I_L -> inf ==> I -> I_K
    i_k = 2e-3
    i_fast_mass_transport = koutecky_levich(i_k=i_k, i_lim=float("inf"))
    assert math.isclose(i_fast_mass_transport, i_k, rel_tol=1e-12)

    # General mixed control: 1/I = 1/I_K + 1/I_L
    expected_mixed = (i_k * i_l) / (i_k + i_l)
    assert math.isclose(
        koutecky_levich(i_k=i_k, i_lim=i_l), expected_mixed, rel_tol=1e-12
    )


def test_koutecky_levich_from_rate_constant_and_rotation():
    """Verify koutecky_levich evaluated from rate constant k and rotation speed."""
    k_rate = 1e-2  # cm/s
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 0.1963
    nu = 0.01
    w = 100.0

    i_k_expected = n * FARADAY * area * k_rate * c_bulk
    i_l_expected = levich(w, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
    expected_tot = (i_k_expected * i_l_expected) / (i_k_expected + i_l_expected)

    i_calc = koutecky_levich(
        omega=w,
        k=k_rate,
        n=n,
        D=D,
        c_bulk=c_bulk,
        area=area,
        nu=nu,
    )
    assert math.isclose(i_calc, expected_tot, rel_tol=1e-12)

    # Vectorized omega
    w_vec = np.array([25.0, 100.0, 400.0])
    i_vec = koutecky_levich(
        omega=w_vec,
        k=k_rate,
        n=n,
        D=D,
        c_bulk=c_bulk,
        area=area,
        nu=nu,
    )
    assert isinstance(i_vec, np.ndarray)
    assert len(i_vec) == 3


def test_koutecky_levich_analysis():
    """Verify Koutecký–Levich linear regression extracts true kinetic and transport parameters."""
    n = 1
    D_true = 1e-5
    c_bulk = 1e-3
    area = 0.1963
    nu = 0.01
    k_true = 0.02  # cm/s

    rpms = np.array([400.0, 900.0, 1600.0, 2500.0])
    w_vals = rpm_to_rad_s(rpms)
    i_vals = koutecky_levich(
        omega=w_vals,
        k=k_true,
        n=n,
        D=D_true,
        c_bulk=c_bulk,
        area=area,
        nu=nu,
    )

    res = koutecky_levich_analysis(
        omega=w_vals,
        current=i_vals,
        area=area,
        c_bulk=c_bulk,
        n=n,
        nu=nu,
    )

    assert isinstance(res, KouteckyLevichResult)
    # R^2 should be 1.0 for perfect synthetic data
    assert res.r_squared > 0.99999
    # Extracted rate constant matches within 1e-6 relative error
    assert math.isclose(res.k_rate, k_true, rel_tol=1e-6)
    # Extracted D matches within 1e-6 relative error
    assert math.isclose(res.d_estimated, D_true, rel_tol=1e-6)
    # Levich constant B
    b_exact = levich_constant(n=n, D=D_true, c_bulk=c_bulk, area=area, nu=nu)
    assert math.isclose(res.levich_constant, b_exact, rel_tol=1e-6)


def test_koutecky_levich_input_validation():
    """Verify input validation for Koutecký–Levich functions."""
    with pytest.raises(
        ValueError, match="Either kinetic current 'i_k' or rate constant 'k'"
    ):
        koutecky_levich(omega=100.0)

    with pytest.raises(ValueError, match="Provide either 'i_k' or 'k', but not both"):
        koutecky_levich(omega=100.0, i_k=1e-3, k=0.01)

    with pytest.raises(ValueError, match="(?i)rate constant k must be non-negative"):
        koutecky_levich(omega=100.0, k=-0.01)

    # Regression validation
    with pytest.raises(ValueError, match="requires at least 2 data points"):
        koutecky_levich_analysis(omega=100.0, current=1e-3)

    with pytest.raises(ValueError, match="Length mismatch"):
        koutecky_levich_analysis(
            omega=np.array([100.0, 200.0, 300.0]), current=np.array([1e-3, 2e-3])
        )

    with pytest.raises(
        ValueError,
        match="All rotation speeds in Koutecký–Levich analysis must be positive",
    ):
        koutecky_levich_analysis(
            omega=np.array([0.0, 100.0]), current=np.array([1e-3, 2e-3])
        )

    with pytest.raises(
        ValueError,
        match="All measured currents in Koutecký–Levich analysis must be positive",
    ):
        koutecky_levich_analysis(
            omega=np.array([100.0, 200.0]), current=np.array([-1e-3, 2e-3])
        )


# ==============================================================================
# 4. RRDE & Collection Efficiency Tests
# ==============================================================================


def test_collection_efficiency_pine_geometry():
    """Verify collection efficiency for standard commercial Pine RRDE tip."""
    # Standard Pine commercial tip dimensions:
    # Disk radius r1 = 0.25 cm (5.0 mm OD)
    # Ring inner radius r2 = 0.325 cm (6.5 mm ID)
    # Ring outer radius r3 = 0.375 cm (7.5 mm OD)
    # Literature/manufacturer theoretical N = 0.2555 (25.55%)
    r1 = 0.25
    r2 = 0.325
    r3 = 0.375

    n_eff = collection_efficiency(r1, r2, r3)
    assert isinstance(n_eff, float)
    assert math.isclose(n_eff, 0.2555, abs_tol=1e-4)


def test_collection_efficiency_benchmarks():
    """Verify collection efficiency across geometric parameter ratios from Bard & Faulkner."""
    # Table 9.4 benchmarks in Bard & Faulkner:
    # r2/r1 = 1.05, r3/r2 = 1.10 -> N approx 0.2385
    # r2/r1 = 1.05, r3/r2 = 1.20 -> N approx 0.3402
    # r2/r1 = 1.20, r3/r2 = 1.10 -> N approx 0.2079
    r1 = 1.0
    n1 = collection_efficiency(r1=r1, r2=1.05 * r1, r3=1.10 * 1.05 * r1)
    assert math.isclose(n1, 0.2385, abs_tol=1e-3)

    n2 = collection_efficiency(r1=r1, r2=1.05 * r1, r3=1.20 * 1.05 * r1)
    assert math.isclose(n2, 0.3402, abs_tol=1e-3)

    n3 = collection_efficiency(r1=r1, r2=1.20 * r1, r3=1.10 * 1.20 * r1)
    assert math.isclose(n3, 0.2079, abs_tol=1e-3)


def test_collection_efficiency_monotonicity():
    """Verify physical properties of N: 0 < N < 1 and monotonic increase with ring width."""
    r1 = 0.2
    r2 = 0.25

    # Expanding outer ring increases collection efficiency
    r3_vals = [0.28, 0.32, 0.38, 0.50]
    n_vals = [collection_efficiency(r1, r2, r3) for r3 in r3_vals]

    for val in n_vals:
        assert 0.0 < val < 1.0

    # Strictly monotonically increasing with outer radius
    for i in range(len(n_vals) - 1):
        assert n_vals[i] < n_vals[i + 1]


def test_collection_efficiency_radii_validation():
    """Verify invalid radii ordering triggers helpful ValueErrors."""
    with pytest.raises(ValueError, match="Disk radius r1 must be strictly positive"):
        collection_efficiency(0.0, 1.0, 2.0)

    with pytest.raises(
        ValueError,
        match="Ring inner radius r2 must be strictly greater than disk radius",
    ):
        collection_efficiency(1.0, 1.0, 2.0)

    with pytest.raises(
        ValueError,
        match="Ring inner radius r2 must be strictly greater than disk radius",
    ):
        collection_efficiency(1.0, 0.8, 2.0)

    with pytest.raises(
        ValueError,
        match="Ring outer radius r3 must be strictly greater than ring inner radius",
    ):
        collection_efficiency(1.0, 1.5, 1.5)

    with pytest.raises(
        ValueError,
        match="Ring outer radius r3 must be strictly greater than ring inner radius",
    ):
        collection_efficiency(1.0, 1.5, 1.2)


# ==============================================================================
# 5. RRDE Ring Limiting Currents & Shielding
# ==============================================================================


def test_ring_collection_current():
    """Verify ring collection current from disk Faradaic generation."""
    i_disk = 100e-6  # 100 uA
    n_eff = 0.2555

    i_ring = ring_collection_current(i_disk, N=n_eff)
    # Cathodic disk (positive) produces anodic ring collection (negative)
    assert math.isclose(i_ring, -n_eff * i_disk, rel_tol=1e-12)

    # Different ring electron stoichiometry (e.g. n_ring = 2, n_disk = 1)
    i_ring_multi = ring_collection_current(i_disk, N=n_eff, n_ring=2, n_disk=1)
    assert math.isclose(i_ring_multi, -n_eff * 2.0 * i_disk, rel_tol=1e-12)

    # Vectorized
    i_d_vec = np.array([50e-6, 100e-6, 200e-6])
    i_r_vec = ring_collection_current(i_d_vec, N=n_eff)
    assert isinstance(i_r_vec, np.ndarray)
    np.testing.assert_allclose(i_r_vec, -n_eff * i_d_vec, rtol=1e-12)

    # Radii computation
    i_r_from_radii = ring_collection_current(i_disk, r1=0.25, r2=0.325, r3=0.375)
    assert math.isclose(i_r_from_radii, -n_eff * i_disk, rel_tol=1e-3)


def test_ring_limiting_current_unshielded_and_shielded():
    """Verify unshielded and shielded ring limiting currents."""
    r1 = 0.25
    r2 = 0.325
    r3 = 0.375
    w = 100.0

    # Unshielded limiting current
    i_r_unshielded = ring_limiting_current(w, r2=r2, r3=r3)
    assert isinstance(i_r_unshielded, float)
    assert i_r_unshielded > 0.0

    # Shielded limiting current: disk reduces reactant flux
    i_r_shielded = ring_limiting_current(w, r1=r1, r2=r2, r3=r3, shielded=True)
    assert isinstance(i_r_shielded, float)
    assert 0.0 < i_r_shielded < i_r_unshielded

    # Exact shielding relation: I_R,sh = I_R,L - N * I_D,L
    n_eff = collection_efficiency(r1, r2, r3)
    disk_area = math.pi * (r1**2)
    i_disk_l = levich(w, area=disk_area)
    expected_shielded = i_r_unshielded - (n_eff * i_disk_l)
    assert math.isclose(i_r_shielded, expected_shielded, rel_tol=1e-12)


def test_shielding_factor():
    """Verify geometric shielding factor S calculation."""
    r1 = 0.25
    r2 = 0.325
    r3 = 0.375

    sf = shielding_factor(r1, r2, r3)
    assert isinstance(sf, float)
    assert 0.0 < sf < 1.0

    # S = 1 - I_R,sh / I_R,L
    w = 100.0
    i_unshielded = ring_limiting_current(w, r2=r2, r3=r3)
    i_shielded = ring_limiting_current(w, r1=r1, r2=r2, r3=r3, shielded=True)
    expected_sf = 1.0 - (i_shielded / i_unshielded)
    assert math.isclose(sf, expected_sf, rel_tol=1e-12)


def test_rotating_ring_disk_unified_solver():
    """Verify rotating_ring_disk unified interface returns complete RRDEResult."""
    r1 = 0.25
    r2 = 0.325
    r3 = 0.375
    rpm_val = 1600.0

    res = rotating_ring_disk(r1, r2, r3, rpm=rpm_val)
    assert isinstance(res, RRDEResult)
    assert math.isclose(res.N, 0.2555, abs_tol=1e-4)
    assert math.isclose(res.area_disk, math.pi * (r1**2), rel_tol=1e-12)
    assert math.isclose(res.area_ring, math.pi * (r3**2 - r2**2), rel_tol=1e-12)
    assert res.i_disk_lim is not None
    assert res.i_ring_lim_unshielded is not None
    assert res.i_ring_lim_shielded is not None
    assert res.i_ring_lim_shielded < res.i_ring_lim_unshielded


# ==============================================================================
# 6. Numerical PDE Solver Cross-Validation Benchmark
# ==============================================================================


def test_rde_numerical_benchmark_agreement():
    """Validate Levich analytical equation against simulated RDE voltammetry parameters.

    Matches parameters from examples/rde_cyclic_voltammetry_solve_ivp_tutorial.ipynb:
    n = 1, D = 1e-5 cm^2/s, c_bulk = 1e-6 mol/cm^3, A = 0.07 cm^2, nu = 0.01 cm^2/s.
    """
    n = 1
    D = 1e-5
    c_bulk = 1e-6
    area = 0.07
    nu = 0.01

    # Rotation series from tutorial: 400, 900, 1600, 2500 rpm
    rpms = [400, 900, 1600, 2500]
    expected_currents_uA = [27.10, 40.65, 54.20, 67.75]

    for rpm_val, i_expected_uA in zip(rpms, expected_currents_uA):
        i_calc_A = levich(rpm=rpm_val, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
        i_calc_uA = float(i_calc_A) * 1e6
        # Compare with tutorial values (< 0.2% tolerance)
        assert math.isclose(i_calc_uA, i_expected_uA, rel_tol=2e-3)


# ==============================================================================
# 7. Package Exports Verification
# ==============================================================================


def test_hydrodynamics_package_exports():
    """Verify all hydrodynamics symbols are exported at package and subpackage levels."""
    # Top-level exports
    assert sp.levich is levich
    assert sp.levich_constant is levich_constant
    assert sp.nernst_diffusion_layer is nernst_diffusion_layer
    assert sp.koutecky_levich is koutecky_levich
    assert sp.koutecky_levich_analysis is koutecky_levich_analysis
    assert sp.collection_efficiency is collection_efficiency
    assert sp.ring_limiting_current is ring_limiting_current
    assert sp.ring_collection_current is ring_collection_current
    assert sp.shielding_factor is shielding_factor
    assert sp.rotating_ring_disk is rotating_ring_disk
    assert sp.rpm_to_rad_s is rpm_to_rad_s
    assert sp.rad_s_to_rpm is rad_s_to_rpm
    assert sp.KouteckyLevichResult is KouteckyLevichResult
    assert sp.RRDEResult is RRDEResult

    # Subpackage exports
    assert sp.analytical.levich is levich
    assert sp.analytical.koutecky_levich is koutecky_levich
    assert sp.analytical.collection_efficiency is collection_efficiency
    assert sp.analytical.rotating_ring_disk is rotating_ring_disk
    assert sp.analytical.hydrodynamics.levich is levich


# ==============================================================================
# 8. Edge Case & Input Validation Coverage
# ==============================================================================


def test_hydrodynamics_additional_edge_cases():
    """Verify remaining validation branches across all hydrodynamic functions."""
    # levich rpm negative and F <= 0
    with pytest.raises(ValueError, match="rpm must be non-negative"):
        levich(rpm=-100.0)

    with pytest.raises(ValueError, match="Faraday constant F must be positive"):
        levich(100.0, F=0.0)

    # nernst_diffusion_layer D <= 0 and nu <= 0
    with pytest.raises(ValueError, match="Diffusion coefficient D must be positive"):
        nernst_diffusion_layer(100.0, D=0.0)

    with pytest.raises(ValueError, match="Kinematic viscosity nu must be positive"):
        nernst_diffusion_layer(100.0, nu=0.0)

    # koutecky_levich_analysis c_bulk <= 0
    with pytest.raises(
        ValueError, match="Bulk concentration c_bulk must be strictly positive"
    ):
        koutecky_levich_analysis(
            omega=np.array([100.0, 200.0]), current=np.array([1e-3, 2e-3]), c_bulk=0.0
        )

    # ring_limiting_current r2 <= 0, r3 <= r2, shielded without r1
    with pytest.raises(ValueError, match="Ring inner radius r2 must be positive"):
        ring_limiting_current(100.0, r2=0.0, r3=1.0)

    with pytest.raises(
        ValueError, match="Ring outer radius r3 must be greater than r2"
    ):
        ring_limiting_current(100.0, r2=1.0, r3=0.8)

    with pytest.raises(
        ValueError, match="Disk radius 'r1' must be provided to compute shielded"
    ):
        ring_limiting_current(100.0, r2=1.0, r3=1.5, shielded=True)

    # ring_collection_current n_ring <= 0, missing r1/r2/r3, invalid N
    with pytest.raises(
        ValueError, match="Electron numbers n_ring and n_disk must be positive"
    ):
        ring_collection_current(1e-3, N=0.2, n_ring=0)

    with pytest.raises(
        ValueError, match="Either collection efficiency 'N' or electrode radii"
    ):
        ring_collection_current(1e-3, r1=0.2)

    with pytest.raises(ValueError, match="Collection efficiency N must be in"):
        ring_collection_current(1e-3, N=1.5)

    # shielding_factor invalid N
    with pytest.raises(ValueError, match="Collection efficiency N must be in"):
        shielding_factor(0.2, 0.3, 0.4, N=0.0)

    # rotating_ring_disk invalid n_ring
    with pytest.raises(ValueError, match="Number of ring electrons must be positive"):
        rotating_ring_disk(0.2, 0.3, 0.4, rpm=1000.0, n_ring=0)

    # rotating_ring_disk without rotation rate
    res_no_rot = rotating_ring_disk(0.2, 0.3, 0.4)
    assert res_no_rot.i_disk_lim is None
    assert res_no_rot.i_ring_lim_unshielded is None
    assert res_no_rot.i_ring_lim_shielded is None

    # koutecky_levich_analysis with negative slope fallback to NaN
    res_neg_slope = koutecky_levich_analysis(
        omega=np.array([100.0, 400.0]),
        current=np.array([2e-3, 1e-3]),
    )
    assert math.isnan(res_neg_slope.d_estimated)
    assert math.isnan(res_neg_slope.levich_constant)
