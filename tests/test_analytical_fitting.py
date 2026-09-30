"""Unit and benchmark tests for parameter fitting utilities (softpotato.analytical.fitting)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.fitting import (
    ButlerVolmerFitResult,
    CottrellFitResult,
    FitResult,
    KouteckyLevichFitResult,
    LevichFitResult,
    MicrodiscFitResult,
    RandlesSevcikFitResult,
    SECMApproachFitResult,
    TafelFitResult,
    fit_butler_volmer,
    fit_cottrell,
    fit_curve,
    fit_koutecky_levich,
    fit_levich,
    fit_linear,
    fit_microdisc_transient,
    fit_randles_sevcik,
    fit_secm_approach,
    fit_tafel,
)
from softpotato.analytical.hydrodynamics import levich
from softpotato.analytical.kinetics import butler_volmer
from softpotato.analytical.microelectrodes import microdisc_transient
from softpotato.analytical.secm import (
    secm_approach_negative_feedback,
    secm_approach_positive_feedback,
)
from softpotato.analytical.step import cottrell
from softpotato.analytical.voltammetry import randles_sevcik
from softpotato.constants import FARADAY

# ==============================================================================
# 1. Core Fitting Engine Tests (fit_linear & fit_curve)
# ==============================================================================


def test_fit_linear_basic():
    """Verify linear regression extracts exact slope and intercept on noise-free data."""
    x = np.linspace(1.0, 10.0, 10)
    true_m = 2.5
    true_q = -1.2
    y = true_m * x + true_q

    res = fit_linear(x, y, fit_intercept=True)
    assert isinstance(res, FitResult)
    assert math.isclose(res.params["slope"], true_m, rel_tol=1e-7)
    assert math.isclose(res.params["intercept"], true_q, rel_tol=1e-7)
    assert math.isclose(res.r_squared, 1.0, rel_tol=1e-7)
    assert res.rmse < 1e-10
    assert res.success is True

    # Check confidence intervals encapsulate true values
    ci_m = res.confidence_intervals["slope"]
    assert ci_m[0] <= true_m <= ci_m[1]
    ci_q = res.confidence_intervals["intercept"]
    assert ci_q[0] <= true_q <= ci_q[1]


def test_fit_linear_no_intercept():
    """Verify linear regression through the origin."""
    x = np.linspace(1.0, 5.0, 5)
    true_m = 3.14
    y = true_m * x

    res = fit_linear(x, y, fit_intercept=False)
    assert math.isclose(res.params["slope"], true_m, rel_tol=1e-7)
    assert "intercept" not in res.params
    assert math.isclose(res.r_squared, 1.0, rel_tol=1e-7)


def test_fit_curve_nonlinear():
    """Verify fit_curve optimizes exponential decay with bounds and errors."""
    x = np.linspace(0.1, 2.0, 30)
    true_a = 5.0
    true_b = 1.5

    def exp_model(x_val, a, b):
        return a * np.exp(-b * x_val)

    y = exp_model(x, true_a, true_b)

    res = fit_curve(
        exp_model,
        x,
        y,
        p0=[1.0, 1.0],
        bounds=([0.0, 0.0], [10.0, 10.0]),
        param_names=["a", "b"],
    )

    assert isinstance(res, FitResult)
    assert math.isclose(res.params["a"], true_a, rel_tol=1e-4)
    assert math.isclose(res.params["b"], true_b, rel_tol=1e-4)
    assert res.r_squared > 0.9999
    assert res.covariance.shape == (2, 2)
    assert res.correlation.shape == (2, 2)


def test_fit_validation_errors():
    """Verify input validation handles mismatched lengths and non-finite values."""
    with pytest.raises(ValueError, match="Array length mismatch"):
        fit_linear([1, 2, 3], [1, 2])

    with pytest.raises(ValueError, match="at least 2 data points"):
        fit_linear([1], [1])

    with pytest.raises(ValueError, match="finite numbers"):
        fit_linear([1.0, np.nan], [2.0, 3.0])


# ==============================================================================
# 2. Chronoamperometry Fitting (fit_cottrell)
# ==============================================================================


def test_fit_cottrell_linear():
    """Verify fit_cottrell recovers diffusion coefficient via linear method."""
    t = np.linspace(0.1, 5.0, 20)
    true_d = 2.5e-5
    true_c = 1e-3
    area = 0.0707
    n = 1

    i_theo = cottrell(t, D=true_d, c_bulk=true_c, area=area, n=n)

    res = fit_cottrell(
        t, i_theo, method="linear", n=n, area=area, c_bulk=true_c, fit_background=False
    )

    assert isinstance(res, CottrellFitResult)
    assert isinstance(res, FitResult)
    assert math.isclose(res.d, true_d, rel_tol=1e-6)
    assert res.d_stderr >= 0.0
    assert math.isclose(res.r_squared, 1.0, rel_tol=1e-6)
    assert res.slope is not None and res.slope > 0


def test_fit_cottrell_nonlinear_with_background():
    """Verify fit_cottrell recovers D and background offset simultaneously via nonlinear method."""
    t = np.linspace(0.05, 2.0, 40)
    true_d = 1.0e-5
    true_bg = 5.0e-7
    area = 1.0
    c = 1e-3

    i_data = cottrell(t, D=true_d, c_bulk=c, area=area, n=1) + true_bg

    res = fit_cottrell(
        t,
        i_data,
        method="nonlinear",
        n=1,
        area=area,
        c_bulk=c,
        fit_background=True,
        p0=[1e-5, 0.0],
    )

    assert math.isclose(res.d, true_d, rel_tol=1e-4)
    assert math.isclose(res.i_bg, true_bg, rel_tol=1e-4)
    assert res.r_squared > 0.9999


def test_fit_cottrell_invalid_time():
    """Verify non-positive time points are rejected."""
    with pytest.raises(ValueError, match="strictly positive"):
        fit_cottrell([0.0, 1.0], [1.0, 0.5])


# ==============================================================================
# 3. Voltammetry Fitting (fit_randles_sevcik)
# ==============================================================================


def test_fit_randles_sevcik_reversible():
    """Verify fit_randles_sevcik recovers D from peak current scan rate series."""
    scan_rates = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    true_d = 1e-5
    area = 0.1
    c = 1e-3
    n = 1

    ip_theo = randles_sevcik(scan_rates, D=true_d, area=area, c_bulk=c, n=n)

    res = fit_randles_sevcik(
        scan_rates,
        ip_theo,
        method="linear",
        n=n,
        area=area,
        c_bulk=c,
        fit_intercept=True,
    )

    assert isinstance(res, RandlesSevcikFitResult)
    assert math.isclose(res.d, true_d, rel_tol=1e-6)
    assert math.isclose(res.intercept, 0.0, abs_tol=1e-9)
    assert res.r_squared > 0.9999


def test_fit_randles_sevcik_nonlinear_irreversible():
    """Verify fit_randles_sevcik in irreversible mode."""
    scan_rates = np.array([0.02, 0.05, 0.1, 0.2])
    true_d = 7.5e-6
    alpha = 0.45
    area = 1.0

    ip_theo = sp.randles_sevcik_irreversible(
        scan_rates, D=true_d, area=area, c_bulk=1e-3, alpha=alpha, n=1
    )

    res = fit_randles_sevcik(
        scan_rates,
        ip_theo,
        method="nonlinear",
        model="irreversible",
        alpha=alpha,
        area=area,
        c_bulk=1e-3,
    )

    assert math.isclose(res.d, true_d, rel_tol=1e-4)


# ==============================================================================
# 4. Hydrodynamic Fitting (fit_levich & fit_koutecky_levich)
# ==============================================================================


def test_fit_levich_linear_and_nonlinear():
    """Verify fit_levich extracts Levich slope B and diffusion coefficient D."""
    rpm_vals = np.array([400, 900, 1600, 2500, 3600])
    true_d = 1.2e-5
    nu = 0.01
    c = 1e-3
    area = 0.196

    i_lim = levich(rpm=rpm_vals, D=true_d, nu=nu, c_bulk=c, area=area, n=1)

    # Linear method
    res_lin = fit_levich(
        i_lim, rpm=rpm_vals, method="linear", nu=nu, area=area, c_bulk=c
    )
    assert isinstance(res_lin, LevichFitResult)
    assert math.isclose(res_lin.d, true_d, rel_tol=1e-5)
    assert res_lin.levich_constant > 0
    assert res_lin.r_squared > 0.9999

    # Non-linear method
    res_nonlin = fit_levich(
        i_lim, rpm=rpm_vals, method="nonlinear", area=area, c_bulk=c
    )
    assert math.isclose(res_nonlin.d, true_d, rel_tol=1e-4)


def test_fit_koutecky_levich_linear_and_nonlinear():
    """Verify fit_koutecky_levich separates kinetic from mass-transport limits."""
    omega_vals = np.array([50.0, 100.0, 200.0, 400.0])
    true_d = 1.0e-5
    true_k0 = 2.0e-3
    area = 1.0
    c = 1e-3
    n = 1
    nu = 0.01

    i_k_true = n * FARADAY * area * true_k0 * c
    b_true = (
        0.620 * n * FARADAY * area * (true_d ** (2.0 / 3.0)) * (nu ** (-1.0 / 6.0)) * c
    )
    # 1/I = 1/IK + 1/(B*sqrt(w))
    i_measured = (1.0 / i_k_true + 1.0 / (b_true * np.sqrt(omega_vals))) ** (-1.0)

    # Linear fit
    res_lin = fit_koutecky_levich(
        i_measured, omega=omega_vals, method="linear", area=area, c_bulk=c, n=n, nu=nu
    )
    assert isinstance(res_lin, KouteckyLevichFitResult)
    assert math.isclose(res_lin.i_k, i_k_true, rel_tol=1e-5)
    assert math.isclose(res_lin.k_rate, true_k0, rel_tol=1e-5)
    assert math.isclose(res_lin.d, true_d, rel_tol=1e-5)

    # Non-linear fit directly on (I, omega)
    res_nonlin = fit_koutecky_levich(
        i_measured,
        omega=omega_vals,
        method="nonlinear",
        area=area,
        c_bulk=c,
        n=n,
        nu=nu,
    )
    assert math.isclose(res_nonlin.i_k, i_k_true, rel_tol=1e-4)
    assert math.isclose(res_nonlin.k_rate, true_k0, rel_tol=1e-4)
    assert math.isclose(res_nonlin.d, true_d, rel_tol=1e-4)


# ==============================================================================
# 5. Kinetics Fitting (fit_tafel & fit_butler_volmer)
# ==============================================================================


def test_fit_tafel_anodic():
    """Verify fit_tafel extracts exchange current and Tafel slope."""
    eta = np.linspace(0.12, 0.25, 20)
    true_i0 = 1e-4
    true_alpha = 0.5
    i_vals = butler_volmer(eta, i0=true_i0, alpha=true_alpha, n=1)

    res = fit_tafel(eta, i_vals, branch="anodic")
    assert isinstance(res, TafelFitResult)
    assert math.isclose(res.i0, true_i0, rel_tol=0.02)
    assert math.isclose(res.alpha, true_alpha, abs_tol=1e-2)
    assert res.slope > 0
    assert res.r_squared > 0.999


def test_fit_butler_volmer_full_range():
    """Verify fit_butler_volmer simultaneously fits I0 and alpha across overpotential domain."""
    eta = np.linspace(-0.2, 0.2, 50)
    true_i0 = 2.5e-4
    true_alpha = 0.6

    i_vals = butler_volmer(eta, i0=true_i0, alpha=true_alpha, n=1)

    res = fit_butler_volmer(
        eta,
        i_vals,
        n=1,
        p0=[1e-4, 0.5],
        bounds=([0.0, 0.05], [1.0, 0.95]),
    )

    assert isinstance(res, ButlerVolmerFitResult)
    assert math.isclose(res.i0, true_i0, rel_tol=1e-4)
    assert math.isclose(res.alpha, true_alpha, rel_tol=1e-4)
    assert res.r_squared > 0.9999


def test_fit_butler_volmer_with_potential_offset():
    """Verify fit_butler_volmer extracts non-zero equilibrium potential offset."""
    eta = np.linspace(-0.25, 0.25, 60)
    true_i0 = 1.0e-4
    true_alpha = 0.5
    true_e_eq = 0.02  # 20 mV offset

    f_const = FARADAY / (sp.GAS_CONSTANT * sp.STANDARD_TEMPERATURE)
    i_vals = true_i0 * (
        np.exp((1.0 - true_alpha) * f_const * (eta - true_e_eq))
        - np.exp(-true_alpha * f_const * (eta - true_e_eq))
    )

    res = fit_butler_volmer(
        eta,
        i_vals,
        fit_e_eq=True,
        p0=[1e-4, 0.5, 0.0],
    )

    assert math.isclose(res.i0, true_i0, rel_tol=1e-3)
    assert math.isclose(res.alpha, true_alpha, rel_tol=1e-3)
    assert res.e_eq is not None and math.isclose(res.e_eq, true_e_eq, abs_tol=1e-4)


# ==============================================================================
# 6. Microelectrode Fitting (fit_microdisc_transient)
# ==============================================================================


def test_fit_microdisc_transient_fit_d():
    """Verify fit_microdisc_transient recovers D for known microdisc radius."""
    t = np.geomspace(1e-4, 1.0, 30)
    a = 12.5e-6
    true_d = 1.0e-5
    c = 1e-3

    i_theo = microdisc_transient(t, radius=a, D=true_d, c_bulk=c, n=1)

    res = fit_microdisc_transient(t, i_theo, target="D", radius=a, c_bulk=c, p0=5e-6)

    assert isinstance(res, MicrodiscFitResult)
    assert math.isclose(res.d, true_d, rel_tol=1e-4)
    assert math.isclose(res.a, a, rel_tol=1e-7)
    assert res.r_squared > 0.9999


def test_fit_microdisc_transient_fit_a():
    """Verify fit_microdisc_transient recovers radius a for known D."""
    t = np.geomspace(1e-4, 1.0, 30)
    true_a = 25e-6
    d = 1.0e-5
    c = 1e-3

    i_theo = microdisc_transient(t, radius=true_a, D=d, c_bulk=c, n=1)

    res = fit_microdisc_transient(t, i_theo, target="radius", D=d, c_bulk=c, p0=10e-6)

    assert math.isclose(res.a, true_a, rel_tol=5e-3)


# ==============================================================================
# 7. SECM Approach Curve Fitting (fit_secm_approach)
# ==============================================================================


def test_fit_secm_approach_positive_feedback():
    """Verify fit_secm_approach simultaneously fits RG and positioning offset d_offset."""
    a = 10e-6
    true_rg = 8.5
    true_offset = 2.0e-6  # 2 micrometers offset

    # Measured stage positions d from 3 um to 25 um
    d_stage = np.linspace(3e-6, 25e-6, 25)
    l_true = (d_stage - true_offset) / a
    it_theo = secm_approach_positive_feedback(l_true, RG=true_rg)

    res = fit_secm_approach(
        d_stage,
        it_theo,
        substrate="conducting",
        a=a,
        fit_rg=True,
        fit_d_offset=True,
        p0=[5.0, 0.0],
    )

    assert isinstance(res, SECMApproachFitResult)
    assert math.isclose(res.rg, true_rg, rel_tol=1e-3)
    assert math.isclose(res.d_offset, true_offset, abs_tol=1e-7)
    assert res.r_squared > 0.9999


def test_fit_secm_approach_negative_feedback():
    """Verify fit_secm_approach on negative feedback over insulating substrate."""
    a = 15e-6
    true_rg = 5.0

    d_stage = np.linspace(2e-6, 30e-6, 20)
    l_true = d_stage / a
    it_theo = secm_approach_negative_feedback(l_true, RG=true_rg)

    res = fit_secm_approach(
        d_stage,
        it_theo,
        substrate="insulating",
        a=a,
        fit_rg=True,
        fit_d_offset=False,
        d_offset=0.0,
        p0=[3.0],
    )

    assert math.isclose(res.rg, true_rg, rel_tol=1e-3)
    assert res.r_squared > 0.9999


# ==============================================================================
# 8. Re-export and Package Integration Tests
# ==============================================================================


def test_package_exports():
    """Verify all fitting functions and classes are accessible via softpotato top-level and analytical."""
    assert hasattr(sp, "fit_cottrell")
    assert hasattr(sp, "fit_randles_sevcik")
    assert hasattr(sp, "fit_levich")
    assert hasattr(sp, "fit_koutecky_levich")
    assert hasattr(sp, "fit_tafel")
    assert hasattr(sp, "fit_butler_volmer")
    assert hasattr(sp, "fit_microdisc_transient")
    assert hasattr(sp, "fit_secm_approach")
    assert hasattr(sp, "FitResult")
    assert hasattr(sp, "CottrellFitResult")
    assert hasattr(sp, "LevichFitResult")
    assert hasattr(sp, "KouteckyLevichFitResult")
    assert hasattr(sp, "TafelFitResult")
    assert hasattr(sp, "ButlerVolmerFitResult")
    assert hasattr(sp, "MicrodiscFitResult")
    assert hasattr(sp, "SECMApproachFitResult")

    # analytical submodule
    assert hasattr(sp.analytical, "fit_cottrell")
    assert hasattr(sp.analytical, "fit_curve")
    assert hasattr(sp.analytical, "fit_linear")
