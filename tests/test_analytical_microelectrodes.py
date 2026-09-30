"""Tests for microelectrode analytical expressions (softpotato.analytical.microelectrodes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softpotato.analytical.microelectrodes import (
    mahon_oldham_transient,
    microband_limiting_current,
    microdisc_limiting_current,
    microdisc_transient,
    microhemisphere_limiting_current,
    microsphere_limiting_current,
)
from softpotato.analytical.step import cottrell, cottrell_spherical
from softpotato.constants import FARADAY

# ==============================================================================
# 1. Microdisc Limiting Current (Saito) Tests
# ==============================================================================


def test_microdisc_limiting_current_scalar():
    """Verify Saito limiting current produces correct scalar float value."""
    radius = 1e-5  # 10 um
    n = 1
    D = 1e-9  # m^2/s
    c_bulk = 1.0  # mol/m^3
    F = FARADAY

    expected = 4.0 * n * F * D * c_bulk * radius
    i_val = microdisc_limiting_current(radius, n=n, D=D, c_bulk=c_bulk, F=F)

    assert isinstance(i_val, float)
    assert math.isclose(i_val, expected, rel_tol=1e-14)


def test_microdisc_limiting_current_vector():
    """Verify Saito limiting current handles array inputs and broadcasting."""
    radii = np.array([5e-6, 10e-6, 25e-6, 50e-6])
    n = 2
    D = 5e-6  # cm^2/s
    c_bulk = 1e-3  # mol/cm^3

    expected = 4.0 * n * FARADAY * D * c_bulk * radii
    i_arr = microdisc_limiting_current(radii, n=n, D=D, c_bulk=c_bulk)

    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == radii.shape
    np.testing.assert_allclose(i_arr, expected, rtol=1e-14)


def test_microdisc_limiting_current_validation():
    """Verify input validation triggers appropriate ValueErrors."""
    with pytest.raises(ValueError, match="Electrode radius must be positive"):
        microdisc_limiting_current(0.0)

    with pytest.raises(ValueError, match="Electrode radius must be positive"):
        microdisc_limiting_current(-1e-5)

    with pytest.raises(ValueError, match="Electrode radius must be positive"):
        microdisc_limiting_current(np.array([1e-5, -1e-5]))

    with pytest.raises(ValueError, match="Number of electrons n must be positive"):
        microdisc_limiting_current(1e-5, n=0)

    with pytest.raises(ValueError, match="Diffusion coefficient D must be positive"):
        microdisc_limiting_current(1e-5, D=-1e-9)

    with pytest.raises(
        ValueError, match="Bulk concentration c_bulk must be non-negative"
    ):
        microdisc_limiting_current(1e-5, c_bulk=-1.0)

    with pytest.raises(ValueError, match="Faraday constant F must be positive"):
        microdisc_limiting_current(1e-5, F=0.0)


# ==============================================================================
# 2. Hemispherical and Spherical Limiting Currents Tests
# ==============================================================================


def test_hemisphere_and_sphere_limiting_current():
    """Verify spherical and hemispherical limiting currents and geometric ratios."""
    radius = 1e-5
    n = 1
    D = 1e-9
    c_bulk = 1.0

    i_hemi = microhemisphere_limiting_current(radius, n=n, D=D, c_bulk=c_bulk)
    i_sphere = microsphere_limiting_current(radius, n=n, D=D, c_bulk=c_bulk)

    expected_hemi = 2.0 * math.pi * n * FARADAY * D * c_bulk * radius
    expected_sphere = 4.0 * math.pi * n * FARADAY * D * c_bulk * radius

    assert isinstance(i_hemi, float)
    assert isinstance(i_sphere, float)
    assert math.isclose(i_hemi, expected_hemi, rel_tol=1e-14)
    assert math.isclose(i_sphere, expected_sphere, rel_tol=1e-14)

    # Ratio of hemisphere to full sphere is exactly 0.5
    assert math.isclose(i_hemi / i_sphere, 0.5, rel_tol=1e-14)


def test_sphere_limiting_current_cross_validation():
    """Cross-validate microsphere limiting current against cottrell_spherical at long times."""
    radius = 2.5e-5
    n = 1
    D = 1e-9
    c_bulk = 1.0

    i_sphere_analytical = microsphere_limiting_current(radius, n=n, D=D, c_bulk=c_bulk)
    # Long time (t = 1e9 s) spherical Cottrell transient
    i_sphere_transient = cottrell_spherical(1e9, r=radius, n=n, D=D, c_bulk=c_bulk)

    assert math.isclose(i_sphere_analytical, i_sphere_transient, rel_tol=1e-4)


def test_hemisphere_and_sphere_validation():
    """Verify invalid inputs trigger ValueErrors for hemisphere and sphere models."""
    with pytest.raises(ValueError, match="Electrode radius must be positive"):
        microhemisphere_limiting_current(0.0)

    with pytest.raises(ValueError, match="Electrode radius must be positive"):
        microsphere_limiting_current(-1e-5)


# ==============================================================================
# 3. Shoup & Szabo Microdisc Transient Tests
# ==============================================================================


def test_microdisc_transient_scalar_and_vector():
    """Verify Shoup & Szabo transient returns float scalar or NumPy array correctly."""
    radius = 1e-5
    t_val = 1.0
    i_scalar = microdisc_transient(t_val, radius=radius)
    assert isinstance(i_scalar, float)

    t_arr = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    i_arr = microdisc_transient(t_arr, radius=radius)
    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == t_arr.shape
    # Current decays monotonically with time
    assert np.all(np.diff(i_arr) < 0)


def test_microdisc_transient_zero_time():
    """Verify transient returns np.inf at t = 0 without error."""
    radius = 1e-5
    assert math.isinf(microdisc_transient(0.0, radius=radius))

    i_vec = microdisc_transient(np.array([0.0, 1.0]), radius=radius)
    assert np.isinf(i_vec[0])
    assert not np.isinf(i_vec[1])


def test_microdisc_transient_asymptotics():
    """Verify Shoup & Szabo converges to Cottrell at short times and Saito at long times."""
    radius = 1e-4  # 100 um
    n = 1
    D = 1e-9
    c_bulk = 1.0

    # 1. Short time limit (t = 1e-6 s): should match planar Cottrell within 1%
    t_short = 1e-6
    i_ss_short = microdisc_transient(t_short, radius=radius, n=n, D=D, c_bulk=c_bulk)
    area = math.pi * (radius**2)
    i_cottrell = cottrell(t_short, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(i_ss_short, i_cottrell, rel_tol=0.01)

    # 2. Long time limit (radius = 1 um, t = 1e4 s): should match Saito steady state within 0.05%
    radius_small = 1e-6
    t_long = 1e4
    i_ss_long = microdisc_transient(
        t_long, radius=radius_small, n=n, D=D, c_bulk=c_bulk
    )
    i_saito = microdisc_limiting_current(radius=radius_small, n=n, D=D, c_bulk=c_bulk)
    assert math.isclose(i_ss_long, i_saito, rel_tol=0.0005)


def test_microdisc_transient_validation():
    """Verify input validations for microdisc_transient."""
    with pytest.raises(ValueError, match="Microdisc radius must be positive"):
        microdisc_transient(1.0, radius=0.0)

    with pytest.raises(ValueError, match="Time t must be non-negative"):
        microdisc_transient(-0.5, radius=1e-5)

    with pytest.raises(ValueError, match="Time t must be non-negative"):
        microdisc_transient(np.array([0.1, -0.2]), radius=1e-5)


# ==============================================================================
# 4. Mahon & Oldham Microdisc Transient Tests
# ==============================================================================


def test_mahon_oldham_transient_scalar_and_vector():
    """Verify Mahon & Oldham transient returns float scalar or NumPy array correctly."""
    radius = 1e-5
    t_val = 1.0
    i_scalar = mahon_oldham_transient(t_val, radius=radius)
    assert isinstance(i_scalar, float)

    t_arr = np.array([0.01, 0.1, 1.0, 10.0])
    i_arr = mahon_oldham_transient(t_arr, radius=radius)
    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == t_arr.shape
    # Current decays monotonically with time
    assert np.all(np.diff(i_arr) < 0)


def test_mahon_oldham_transient_zero_time():
    """Verify Mahon & Oldham transient returns np.inf at t = 0 without error."""
    radius = 1e-5
    assert math.isinf(mahon_oldham_transient(0.0, radius=radius))

    i_vec = mahon_oldham_transient(np.array([0.0, 1.0]), radius=radius)
    assert np.isinf(i_vec[0])
    assert not np.isinf(i_vec[1])


def test_mahon_oldham_branches_and_continuity():
    """Verify continuity between short-time and long-time branches at sigma = 1.281."""
    radius = 1e-5
    D = 1e-9
    # sigma = D * t / a^2 = 1.281 => t = 1.281 * a^2 / D
    t_boundary = 1.281 * (radius**2) / D

    i_short = mahon_oldham_transient(
        t_boundary, radius=radius, D=D, branch="short_time"
    )
    i_long = mahon_oldham_transient(t_boundary, radius=radius, D=D, branch="long_time")
    i_auto = mahon_oldham_transient(t_boundary, radius=radius, D=D, branch="auto")

    # Relative difference across the boundary is less than 0.005%
    rel_diff = abs(i_short - i_long) / i_long
    assert rel_diff < 5e-5
    assert math.isclose(i_auto, i_short, rel_tol=1e-12)


def test_mahon_oldham_asymptotics():
    """Verify Mahon & Oldham converges to Cottrell at short times and Saito at long times."""
    radius = 1e-4
    n = 1
    D = 1e-9
    c_bulk = 1.0

    # 1. Short time limit (t = 1e-6 s)
    t_short = 1e-6
    i_mo_short = mahon_oldham_transient(t_short, radius=radius, n=n, D=D, c_bulk=c_bulk)
    area = math.pi * (radius**2)
    i_cottrell = cottrell(t_short, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(i_mo_short, i_cottrell, rel_tol=0.005)

    # 2. Long time limit (radius = 1 um, t = 1e4 s): should match Saito steady state within 0.05%
    radius_small = 1e-6
    t_long = 1e4
    i_mo_long = mahon_oldham_transient(
        t_long, radius=radius_small, n=n, D=D, c_bulk=c_bulk
    )
    i_saito = microdisc_limiting_current(radius=radius_small, n=n, D=D, c_bulk=c_bulk)
    assert math.isclose(i_mo_long, i_saito, rel_tol=0.0005)


def test_mahon_oldham_vs_shoup_szabo_agreement():
    """Verify Mahon & Oldham and Shoup & Szabo agree within 0.7% across 8 orders of magnitude in time."""
    radius = 1e-5
    D = 1e-9
    c_bulk = 1.0
    # Spanning sigma from 1e-4 to 1e4
    sigma_vals = np.logspace(-4, 4, 30)
    t_vals = sigma_vals * (radius**2) / D

    i_mo = mahon_oldham_transient(t_vals, radius=radius, D=D, c_bulk=c_bulk)
    i_ss = microdisc_transient(t_vals, radius=radius, D=D, c_bulk=c_bulk)

    assert isinstance(i_mo, np.ndarray)
    assert isinstance(i_ss, np.ndarray)
    rel_dev = np.abs(i_mo - i_ss) / i_ss
    assert np.max(rel_dev) < 0.007  # Maximum deviation < 0.7%


def test_mahon_oldham_validation():
    """Verify input validation for mahon_oldham_transient."""
    with pytest.raises(ValueError, match="Microdisc radius must be positive"):
        mahon_oldham_transient(1.0, radius=0.0)

    with pytest.raises(ValueError, match="Time t must be non-negative"):
        mahon_oldham_transient(-0.1, radius=1e-5)

    with pytest.raises(ValueError, match="Unknown branch"):
        mahon_oldham_transient(1.0, radius=1e-5, branch="invalid_branch")  # type: ignore[arg-type]


# ==============================================================================
# 5. Microband Limiting Current Tests
# ==============================================================================


def test_microband_limiting_current_scalar_and_vector():
    """Verify microband limiting current evaluates accurately for scalar and array."""
    width = 1e-5  # 10 um
    length = 1e-3  # 1 mm
    n = 1
    D = 1e-9  # m^2/s
    c_bulk = 1.0  # mol/m^3
    F = FARADAY

    # t = 1.0 s => 0.64 * D * t / w^2 = 0.64 * 1e-9 * 1.0 / 1e-10 = 6.4 > 1
    t_val = 1.0
    expected = (2.0 * math.pi * n * F * D * c_bulk * length) / math.log(
        (0.64 * D * t_val) / (width**2)
    )

    i_scalar = microband_limiting_current(
        t_val, width=width, length=length, n=n, D=D, c_bulk=c_bulk, F=F
    )
    assert isinstance(i_scalar, float)
    assert math.isclose(i_scalar, expected, rel_tol=1e-14)

    # Vector of times
    t_arr = np.array([0.5, 1.0, 2.0, 5.0])
    i_arr = microband_limiting_current(
        t_arr, width=width, length=length, n=n, D=D, c_bulk=c_bulk, F=F
    )
    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == t_arr.shape
    # Current decays monotonically with time
    assert np.all(np.diff(i_arr) < 0)


def test_microband_limiting_current_scaling():
    """Verify linear scaling with length and independence of orientation."""
    width = 1e-5
    t_val = 2.0

    i_1mm = microband_limiting_current(t_val, width=width, length=1e-3)
    i_2mm = microband_limiting_current(t_val, width=width, length=2e-3)

    assert math.isclose(i_2mm, 2.0 * i_1mm, rel_tol=1e-14)


def test_microband_limiting_current_domain_check():
    """Verify ValueError is raised when 0.64 * D * t / width**2 <= 1.0."""
    width = 1e-5
    D = 1e-9
    # t_crit = w^2 / (0.64 * D) = 1e-10 / 0.64e-9 = 0.15625 s
    t_invalid = 0.1  # 0.64 * D * t / w^2 = 0.64 < 1.0

    with pytest.raises(
        ValueError, match="Quasi-steady-state microband equation requires"
    ):
        microband_limiting_current(t_invalid, width=width, D=D)

    with pytest.raises(
        ValueError, match="Quasi-steady-state microband equation requires"
    ):
        microband_limiting_current(np.array([1.0, 0.05]), width=width, D=D)


def test_microband_limiting_current_validation():
    """Verify geometric and physical parameter validations."""
    with pytest.raises(ValueError, match="Microband width must be positive"):
        microband_limiting_current(1.0, width=0.0)

    with pytest.raises(ValueError, match="Microband length must be positive"):
        microband_limiting_current(1.0, width=1e-5, length=-1.0)

    with pytest.raises(ValueError, match="Time t must be strictly positive"):
        microband_limiting_current(0.0, width=1e-5)


# ==============================================================================
# 6. Package Re-export Tests
# ==============================================================================


def test_import_from_analytical_subpackage():
    """Verify all public microelectrode symbols are exposed in softpotato.analytical."""
    import softpotato.analytical as an

    assert hasattr(an, "microelectrodes")
    assert hasattr(an, "microdisc_limiting_current")
    assert hasattr(an, "microdisc_transient")
    assert hasattr(an, "mahon_oldham_transient")
    assert hasattr(an, "microhemisphere_limiting_current")
    assert hasattr(an, "microsphere_limiting_current")
    assert hasattr(an, "microband_limiting_current")


def test_import_from_top_level_package():
    """Verify all public microelectrode symbols are exposed at top-level in softpotato."""
    import softpotato as sp

    assert hasattr(sp, "microdisc_limiting_current")
    assert hasattr(sp, "microdisc_transient")
    assert hasattr(sp, "mahon_oldham_transient")
    assert hasattr(sp, "microhemisphere_limiting_current")
    assert hasattr(sp, "microsphere_limiting_current")
    assert hasattr(sp, "microband_limiting_current")
