"""
Unit tests for the analytical electrochemical solutions module.
"""

from __future__ import annotations

import numpy as np
import pytest

from softpotato.analytical import (
    anson,
    catalytic_current,
    cottrell,
    koutecky_levich,
    levich,
    nicholson_psi,
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
    spherical_cottrell,
    steady_state_microdisc,
    steady_state_microhemisphere,
)
from softpotato.core.constants import FARADAY, GAS_CONSTANT

# ==============================================================================
# 1. Potential Step & Chronocoulometry (step.py)
# ==============================================================================


def test_cottrell_scalar_and_benchmark() -> None:
    """Tests Cottrell equation with scalar inputs against analytical benchmark."""
    n = 1
    area = 1.0  # cm^2
    D = 1e-5  # cm^2/s
    c_bulk = 1e-6  # mol/cm^3 (1 mM)
    t = 1.0  # s

    expected = (n * FARADAY * area * c_bulk * np.sqrt(D)) / np.sqrt(np.pi * t)
    result = cottrell(t=t, n=n, area=area, D=D, c_bulk=c_bulk)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)
    assert np.isclose(result, 1.721455507e-4, rtol=1e-6)


def test_cottrell_vectorized() -> None:
    """Tests Cottrell equation with 1D and 2D arrays."""
    t_1d = np.array([0.1, 0.5, 1.0, 2.0])
    res_1d = cottrell(t=t_1d, n=1, area=1.0, D=1e-5, c_bulk=1e-6)

    assert isinstance(res_1d, np.ndarray)
    assert res_1d.shape == (4,)
    # Verify t^(-1/2) scaling: i(0.1) / i(1.0) == sqrt(1.0 / 0.1)
    assert np.isclose(res_1d[0] / res_1d[2], np.sqrt(10.0), rtol=1e-12)

    # 2D array
    t_2d = t_1d.reshape(2, 2)
    res_2d = cottrell(t=t_2d, n=1, area=1.0, D=1e-5, c_bulk=1e-6)
    assert res_2d.shape == (2, 2)


def test_cottrell_validation() -> None:
    """Tests input validation for Cottrell equation."""
    # Non-positive time
    with pytest.raises(ValueError, match="strictly positive"):
        cottrell(t=0.0, n=1, area=1.0, D=1e-5, c_bulk=1e-6)
    with pytest.raises(ValueError, match="strictly positive"):
        cottrell(t=-1.0, n=1, area=1.0, D=1e-5, c_bulk=1e-6)
    with pytest.raises(ValueError, match="strictly positive"):
        cottrell(t=np.array([1.0, 0.0]), n=1, area=1.0, D=1e-5, c_bulk=1e-6)

    # Invalid n
    with pytest.raises(ValueError, match="n must be >= 1"):
        cottrell(t=1.0, n=0, area=1.0, D=1e-5, c_bulk=1e-6)
    with pytest.raises(TypeError, match="integer"):
        cottrell(t=1.0, n=1.5, area=1.0, D=1e-5, c_bulk=1e-6)  # type: ignore[arg-type]

    # Invalid area, D, c_bulk
    with pytest.raises(ValueError, match="area must be positive"):
        cottrell(t=1.0, n=1, area=0.0, D=1e-5, c_bulk=1e-6)
    with pytest.raises(ValueError, match="D must be positive"):
        cottrell(t=1.0, n=1, area=1.0, D=-1e-5, c_bulk=1e-6)
    with pytest.raises(ValueError, match="c_bulk must be non-negative"):
        cottrell(t=1.0, n=1, area=1.0, D=1e-5, c_bulk=-1e-6)


def test_anson_scalar_and_benchmark() -> None:
    """Tests Anson equation for chronocoulometry with scalar inputs."""
    n = 1
    area = 0.0707
    D = 1e-5
    c_bulk = 1e-6
    t = 4.0
    q_dl = 1.5e-6  # Coulombs
    q_ads = 3.0e-6  # Coulombs

    faradaic_diff = (2.0 * n * FARADAY * area * c_bulk * np.sqrt(D * t)) / np.sqrt(
        np.pi
    )
    expected = faradaic_diff + q_dl + q_ads

    result = anson(t=t, n=n, area=area, D=D, c_bulk=c_bulk, q_dl=q_dl, q_ads=q_ads)
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # At t = 0, charge equals q_dl + q_ads
    q_zero = anson(t=0.0, n=n, area=area, D=D, c_bulk=c_bulk, q_dl=q_dl, q_ads=q_ads)
    assert np.isclose(q_zero, q_dl + q_ads, rtol=1e-12)


def test_anson_vectorized_and_validation() -> None:
    """Tests Anson equation vectorization and validation."""
    t = np.array([0.0, 1.0, 4.0])
    res = anson(t=t, n=1, area=1.0, D=1e-5, c_bulk=1e-6, q_dl=1e-6, q_ads=2e-6)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert np.isclose(res[0], 3e-6, rtol=1e-12)

    with pytest.raises(ValueError, match="non-negative"):
        anson(t=-0.1, n=1, area=1.0, D=1e-5, c_bulk=1e-6)


def test_spherical_cottrell_scalar_and_benchmark() -> None:
    """Tests spherical Cottrell equation with scalar and array inputs."""
    n = 1
    r0 = 0.05  # cm
    D = 1e-5
    c_bulk = 1e-6
    t = 1.0

    area = 4.0 * np.pi * (r0**2)
    expected = (
        n * FARADAY * area * c_bulk * D * (1.0 / np.sqrt(np.pi * D * t) + 1.0 / r0)
    )

    result = spherical_cottrell(t=t, n=n, r0=r0, D=D, c_bulk=c_bulk)
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Long-time limit approaches steady-state 4 * pi * n * F * D * C * r0
    t_long = 1e8
    i_ss = 4.0 * np.pi * n * FARADAY * D * c_bulk * r0
    assert np.isclose(
        spherical_cottrell(t=t_long, n=n, r0=r0, D=D, c_bulk=c_bulk),
        i_ss,
        rtol=1e-4,
    )


# ==============================================================================
# 2. Voltammetry (voltammetry.py)
# ==============================================================================


def test_randles_sevcik_scalar_and_benchmark() -> None:
    """Tests reversible Randles-Sevcik equation against benchmark."""
    n = 1
    area = 0.0707
    D = 1e-5
    c_bulk = 1e-6
    v = 0.1  # V/s
    T = 298.15

    expected = (
        0.4463
        * n
        * FARADAY
        * area
        * c_bulk
        * np.sqrt((n * FARADAY * v * D) / (GAS_CONSTANT * T))
    )
    result = randles_sevcik(n=n, area=area, D=D, c_bulk=c_bulk, scan_rate=v, T=T)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Vectorized scan rates
    v_arr = np.array([0.01, 0.04, 0.09, 0.16])
    res_arr = randles_sevcik(n=n, area=area, D=D, c_bulk=c_bulk, scan_rate=v_arr)
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape == (4,)
    # Verify sqrt(v) scaling
    assert np.isclose(res_arr[3] / res_arr[0], np.sqrt(0.16 / 0.01), rtol=1e-12)


def test_randles_sevcik_irreversible_and_validation() -> None:
    """Tests irreversible Randles-Sevcik equation."""
    n = 1
    area = 0.0707
    D = 1e-5
    c_bulk = 1e-6
    v = 0.1
    alpha = 0.5
    T = 298.15

    expected = (
        0.4958
        * n
        * FARADAY
        * area
        * c_bulk
        * np.sqrt(D)
        * np.sqrt((alpha * FARADAY * v) / (GAS_CONSTANT * T))
    )
    result = randles_sevcik_irreversible(
        n=n, area=area, D=D, c_bulk=c_bulk, scan_rate=v, alpha=alpha, T=T
    )
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Validation
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        randles_sevcik_irreversible(
            n=n, area=area, D=D, c_bulk=c_bulk, scan_rate=v, alpha=0.0
        )
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        randles_sevcik_irreversible(
            n=n, area=area, D=D, c_bulk=c_bulk, scan_rate=v, alpha=1.0
        )


def test_peak_potential_irreversible() -> None:
    """Tests peak potential calculation for totally irreversible system."""
    e0 = 0.0
    D = 1e-5
    k0 = 1e-3
    v = 0.1
    alpha = 0.5
    n_alpha = 1
    T = 298.15

    rt_over_anf = (GAS_CONSTANT * T) / (alpha * n_alpha * FARADAY)
    term1 = 0.780
    term2 = np.log(np.sqrt(D) / k0)
    term3 = np.log(np.sqrt((alpha * n_alpha * FARADAY * v) / (GAS_CONSTANT * T)))
    expected = e0 - rt_over_anf * (term1 + term2 + term3)

    result = peak_potential_irreversible(
        e0=e0, D=D, k0=k0, scan_rate=v, alpha=alpha, n_alpha=n_alpha, T=T
    )
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Vectorized scan rate
    v_arr = np.array([0.05, 0.1, 0.2])
    res_arr = peak_potential_irreversible(e0=e0, D=D, k0=k0, scan_rate=v_arr)
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape == (3,)
    # Ep shifts negatively as scan rate increases for reduction
    assert res_arr[0] > res_arr[1] > res_arr[2]


# ==============================================================================
# 3. Microelectrodes (microelectrodes.py)
# ==============================================================================


def test_steady_state_microdisc() -> None:
    """Tests Saito steady-state microdisc equation with parameter 'a'."""
    n = 1
    a = 1e-3  # 10 um radius
    D = 1e-5
    c_bulk = 1e-6

    expected = 4.0 * n * FARADAY * D * c_bulk * a
    result = steady_state_microdisc(n=n, a=a, D=D, c_bulk=c_bulk)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)
    assert np.isclose(result, 3.8594132849e-9, rtol=1e-6)

    # Backward compatibility with keyword 'radius'
    res_radius = steady_state_microdisc(n=n, radius=a, D=D, c_bulk=c_bulk)
    assert np.isclose(res_radius, expected, rtol=1e-12)

    # Vectorized
    a_arr = np.linspace(1e-4, 25e-4, 10)
    res_arr = steady_state_microdisc(n=n, a=a_arr, D=D, c_bulk=c_bulk)
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape == (10,)


def test_steady_state_microhemisphere() -> None:
    """Tests steady-state microhemisphere equation."""
    n = 1
    r0 = 1e-3
    D = 1e-5
    c_bulk = 1e-6

    expected = 2.0 * np.pi * n * FARADAY * D * c_bulk * r0
    result = steady_state_microhemisphere(n=n, r0=r0, D=D, c_bulk=c_bulk)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Ratio between microhemisphere (2*pi) and microdisc (4): pi / 2
    disc_val = steady_state_microdisc(n=n, a=r0, D=D, c_bulk=c_bulk)
    assert np.isclose(result / disc_val, np.pi / 2.0, rtol=1e-12)


# ==============================================================================
# 4. Hydrodynamics (hydrodynamics.py)
# ==============================================================================


def test_levich_scalar_and_benchmark() -> None:
    """Tests Levich RDE limiting current."""
    n = 1
    area = 0.1963  # 5 mm diameter RDE
    D = 1e-5
    c_bulk = 1e-6
    omega = 100.0  # rad/s (~955 rpm)
    nu = 0.01  # cm^2/s

    expected = (
        0.62
        * n
        * FARADAY
        * area
        * (D ** (2.0 / 3.0))
        * np.sqrt(omega)
        * (nu ** (-1.0 / 6.0))
        * c_bulk
    )
    result = levich(n=n, area=area, D=D, c_bulk=c_bulk, omega=omega, nu=nu)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Vectorized
    w_arr = np.array([25.0, 100.0, 400.0])
    res_arr = levich(n=n, area=area, D=D, c_bulk=c_bulk, omega=w_arr, nu=nu)
    assert res_arr.shape == (3,)
    assert np.isclose(res_arr[1] / res_arr[0], np.sqrt(100.0 / 25.0), rtol=1e-12)


def test_koutecky_levich_net_current() -> None:
    """Tests Koutecký-Levich equation returning net current."""
    n = 1
    area = 0.1963
    D = 1e-5
    c_bulk = 1e-6
    omega = 100.0
    k_f = 0.01  # cm/s
    nu = 0.01

    ik = n * FARADAY * area * k_f * c_bulk
    il = levich(n=n, area=area, D=D, c_bulk=c_bulk, omega=omega, nu=nu)
    expected = (ik * il) / (ik + il)

    result = koutecky_levich(
        n=n, area=area, D=D, c_bulk=c_bulk, omega=omega, k_f=k_f, nu=nu
    )
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Net current must be strictly less than both Ik and IL
    assert result < ik
    assert result < il

    # Fast kinetics limit: k_f -> very large => I -> IL
    fast_result = koutecky_levich(
        n=n, area=area, D=D, c_bulk=c_bulk, omega=omega, k_f=1e6, nu=nu
    )
    assert np.isclose(fast_result, il, rtol=1e-5)


# ==============================================================================
# 5. Reversibility (reversibility.py)
# ==============================================================================


def test_nicholson_psi_benchmark() -> None:
    """Tests Nicholson's reversibility parameter Psi."""
    k0 = 0.01  # cm/s
    D_O = 1e-5
    scan_rate = 0.1
    n = 1
    T = 298.15

    expected = k0 / np.sqrt(
        np.pi * D_O * (n * FARADAY * scan_rate) / (GAS_CONSTANT * T)
    )
    result = nicholson_psi(k0=k0, D_O=D_O, scan_rate=scan_rate, n=n, T=T)

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # When D_O != D_R
    D_R = 0.5e-5
    alpha = 0.5
    expected_dr = (k0 * ((D_O / D_R) ** (alpha / 2.0))) / np.sqrt(
        np.pi * D_O * (n * FARADAY * scan_rate) / (GAS_CONSTANT * T)
    )
    result_dr = nicholson_psi(
        k0=k0, D_O=D_O, scan_rate=scan_rate, n=n, D_R=D_R, alpha=alpha, T=T
    )
    assert np.isclose(result_dr, expected_dr, rtol=1e-12)


# ==============================================================================
# 6. Mechanisms (mechanisms.py)
# ==============================================================================


def test_catalytic_current() -> None:
    """Tests catalytic limiting current for EC' mechanism."""
    n = 1
    area = 1.0
    D = 1e-5
    c_bulk = 1e-6
    k_cat = 1e3
    c_cat = 1e-6

    expected = n * FARADAY * area * c_bulk * np.sqrt(D * k_cat * c_cat)
    result = catalytic_current(
        n=n, area=area, D=D, c_bulk=c_bulk, k_cat=k_cat, c_cat=c_cat
    )

    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-12)

    # Vectorized over c_cat
    c_cat_arr = np.array([1e-6, 4e-6, 9e-6])
    res_arr = catalytic_current(
        n=n, area=area, D=D, c_bulk=c_bulk, k_cat=k_cat, c_cat=c_cat_arr
    )
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape == (3,)
    assert np.isclose(res_arr[2] / res_arr[0], np.sqrt(9.0), rtol=1e-12)
