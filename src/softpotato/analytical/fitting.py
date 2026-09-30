r"""Parameter fitting and experimental data analysis utilities.

This submodule provides physically bounded linear and non-linear regression
utilities for extracting transport, kinetic, and geometric parameters from
experimental or simulated electrochemical data.

These utilities wrap :func:`scipy.optimize.curve_fit` and linear least-squares
algorithms with:
1. Physically realistic parameter boundaries (:math:`D > 0`, :math:`0 < \alpha < 1`,
   :math:`k^0 \ge 0`, :math:`c^* > 0`, :math:`A > 0`, :math:`RG \ge 1.0`).
2. Rigorous uncertainty quantification: parameter standard errors, covariance
   and correlation matrices, and Student's :math:`t` confidence intervals.
3. Goodness-of-fit statistics: :math:`R^2`, adjusted :math:`R^2`, Root-Mean-Square
   Error (RMSE), and reduced chi-squared (:math:`\chi_\nu^2`).
4. High-level fitting routines for standard electroanalytical techniques:
   - Cottrell chronoamperometry (:func:`fit_cottrell`)
   - Randles–Ševčík cyclic voltammetry (:func:`fit_randles_sevcik`)
   - Rotating Disk Electrode (RDE) Levich analysis (:func:`fit_levich`)
   - Koutecký–Levich mixed kinetic/diffusion analysis (:func:`fit_koutecky_levich`)
   - High-field Tafel kinetics (:func:`fit_tafel`)
   - Full-range Butler–Volmer kinetics (:func:`fit_butler_volmer`)
   - Microelectrode transient chronoamperometry (:func:`fit_microdisc_transient`)
   - SECM approach curve feedback modeling (:func:`fit_secm_approach`)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import optimize as opt
from scipy import stats

from softpotato.analytical.hydrodynamics import _resolve_rotation
from softpotato.analytical.microelectrodes import (
    mahon_oldham_transient,
    microdisc_transient,
)
from softpotato.analytical.secm import (
    secm_approach_curve,
    secm_approach_negative_feedback,
    secm_approach_positive_feedback,
)
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# ==============================================================================
# Base and Specialized Dataclasses
# ==============================================================================


@dataclass(frozen=True, kw_only=True)
class FitResult:
    r"""General container for regression analysis results.

    Parameters
    ----------
    params : dict[str, float]
        Dictionary of optimal fitted parameter values.
    stderr : dict[str, float]
        Asymptotic standard errors for each fitted parameter.
    covariance : numpy.ndarray
        Estimated parameter variance-covariance matrix :math:`C`.
    correlation : numpy.ndarray
        Normalized parameter correlation matrix :math:`R_{ij} = C_{ij} / (\sigma_i \sigma_j)`.
    confidence_intervals : dict[str, tuple[float, float]]
        Two-tailed confidence intervals at the requested confidence level.
    r_squared : float
        Coefficient of determination :math:`R^2`.
    r_squared_adj : float
        Adjusted coefficient of determination :math:`R_{\text{adj}}^2`.
    rmse : float
        Root-Mean-Square Error of the residuals.
    chi2_reduced : float
        Reduced chi-squared statistic :math:`\chi_\nu^2 = \text{SS}_{\text{res}} / \nu`.
    residuals : numpy.ndarray
        Array of residuals :math:`y_{\text{data}} - y_{\text{fit}}`.
    y_fit : numpy.ndarray
        Model evaluated at the input coordinates with optimal parameters.
    success : bool
        Whether the optimization algorithm converged successfully.
    message : str
        Termination status or message from the optimizer.
    nfev : int
        Number of function evaluations performed.
    """

    params: dict[str, float]
    stderr: dict[str, float]
    covariance: np.ndarray
    correlation: np.ndarray
    confidence_intervals: dict[str, tuple[float, float]]
    r_squared: float
    r_squared_adj: float
    rmse: float
    chi2_reduced: float
    residuals: np.ndarray
    y_fit: np.ndarray
    success: bool
    message: str
    nfev: int


@dataclass(frozen=True, kw_only=True)
class CottrellFitResult(FitResult):
    r"""Diagnostic regression results from Cottrell chronoamperometry analysis.

    Parameters
    ----------
    d : float
        Extracted diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}`.
    d_stderr : float
        Standard error of the extracted diffusion coefficient.
    i_bg : float
        Extracted background / capacitive current offset in Amperes.
    slope : float or None, default None
        Linear regression slope of :math:`I` vs. :math:`t^{-1/2}` if linear method was used.
    """

    d: float
    d_stderr: float
    i_bg: float
    slope: float | None = None


@dataclass(frozen=True, kw_only=True)
class RandlesSevcikFitResult(FitResult):
    r"""Diagnostic regression results from Randles–Ševčík voltammetry analysis.

    Parameters
    ----------
    d : float
        Extracted diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}`.
    d_stderr : float
        Standard error of the extracted diffusion coefficient.
    slope : float
        Regression slope of :math:`I_p` vs. :math:`v^{1/2}`.
    intercept : float
        Regression intercept (current offset at zero scan rate) in Amperes.
    """

    d: float
    d_stderr: float
    slope: float
    intercept: float


@dataclass(frozen=True, kw_only=True)
class LevichFitResult(FitResult):
    r"""Diagnostic regression results from RDE Levich analysis.

    Parameters
    ----------
    levich_constant : float
        Extracted Levich slope :math:`B` in :math:`\text{A}\cdot\text{rad}^{-1/2}\cdot\text{s}^{1/2}`.
    d : float
        Extracted diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}`.
    d_stderr : float
        Standard error of the extracted diffusion coefficient.
    slope : float or None, default None
        Linear regression slope of :math:`I_L` vs. :math:`\omega^{1/2}`.
    """

    levich_constant: float
    d: float
    d_stderr: float
    slope: float | None = None


@dataclass(frozen=True, kw_only=True)
class KouteckyLevichFitResult(FitResult):
    r"""Diagnostic regression results from Koutecký–Levich mixed kinetics analysis.

    Parameters
    ----------
    i_k : float
        Extracted kinetic current :math:`I_K` in Amperes.
    k_rate : float
        Extracted apparent heterogeneous rate constant :math:`k^0`.
    levich_constant : float
        Extracted Levich constant :math:`B`.
    d : float
        Estimated diffusion coefficient :math:`D`.
    slope : float or None, default None
        Regression slope :math:`1/B` of :math:`I^{-1}` vs. :math:`\omega^{-1/2}`.
    intercept : float or None, default None
        Regression intercept :math:`1/I_K` of :math:`I^{-1}` vs. :math:`\omega^{-1/2}`.
    """

    i_k: float
    k_rate: float
    levich_constant: float
    d: float
    slope: float | None = None
    intercept: float | None = None


@dataclass(frozen=True, kw_only=True)
class TafelFitResult(FitResult):
    r"""Diagnostic regression results from Tafel analysis.

    Parameters
    ----------
    i0 : float
        Extracted exchange current :math:`I_0` in Amperes.
    slope : float
        Tafel slope :math:`b` in Volts per decade (:math:`\text{V}/\text{dec}`).
    slope_mv : float
        Tafel slope :math:`b` in millivolts per decade (:math:`\text{mV}/\text{dec}`).
    alpha : float
        Extracted apparent charge-transfer coefficient :math:`\alpha`.
    k0 : float or None, default None
        Extracted standard heterogeneous rate constant :math:`k^0`.
    """

    i0: float
    slope: float
    slope_mv: float
    alpha: float
    k0: float | None = None


@dataclass(frozen=True, kw_only=True)
class ButlerVolmerFitResult(FitResult):
    r"""Diagnostic regression results from non-linear Butler–Volmer fitting.

    Parameters
    ----------
    i0 : float
        Extracted exchange current :math:`I_0` in Amperes.
    alpha : float
        Extracted charge-transfer coefficient :math:`\alpha`.
    k0 : float or None, default None
        Extracted standard heterogeneous rate constant :math:`k^0`.
    e_eq : float or None, default None
        Extracted equilibrium potential offset in Volts.
    """

    i0: float
    alpha: float
    k0: float | None = None
    e_eq: float | None = None


@dataclass(frozen=True, kw_only=True)
class MicrodiscFitResult(FitResult):
    r"""Diagnostic regression results from microdisc transient fitting.

    Parameters
    ----------
    a : float
        Microdisc electrode radius in meters.
    d : float
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}`.
    """

    a: float
    d: float


@dataclass(frozen=True, kw_only=True)
class SECMApproachFitResult(FitResult):
    r"""Diagnostic regression results from SECM approach curve fitting.

    Parameters
    ----------
    rg : float
        Extracted insulator radius ratio :math:`RG = r_g / a`.
    d_offset : float
        Extracted zero-distance vertical positioning offset in meters.
    k_sub : float or None, default None
        Extracted apparent dimensionless substrate rate constant :math:`\kappa`.
    """

    rg: float
    d_offset: float
    k_sub: float | None = None


# ==============================================================================
# Helper Functions
# ==============================================================================


def _validate_arrays(
    x: ArrayLike, y: ArrayLike, min_points: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 1D coordinate and observation arrays."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"Array length mismatch: x has {len(x_arr)} points, but y has {len(y_arr)}."
        )
    if len(x_arr) < min_points:
        raise ValueError(
            f"Fitting requires at least {min_points} data points; got {len(x_arr)}."
        )
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError(
            "Data arrays must contain only finite numbers (no NaN or Inf)."
        )
    return x_arr, y_arr


def _compute_statistics(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    param_names: Sequence[str],
    confidence_level: float = 0.95,
    sigma: np.ndarray | None = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    np.ndarray,
    dict[str, tuple[float, float]],
    float,
    float,
    float,
    float,
    np.ndarray,
]:
    """Compute parameter errors, covariance, correlation, and goodness-of-fit metrics."""
    n_pts = len(y_data)
    n_params = len(popt)
    dof = max(n_pts - n_params, 1)

    residuals = y_data - y_fit
    ss_res = float(np.sum(residuals**2))
    mean_y = float(np.mean(y_data))
    ss_tot = float(np.sum((y_data - mean_y) ** 2))

    # R^2 and adjusted R^2
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    if n_pts > n_params and ss_tot > 0:
        r_squared_adj = 1.0 - (ss_res / dof) / (ss_tot / (n_pts - 1))
    else:
        r_squared_adj = r_squared

    rmse = float(np.sqrt(ss_res / n_pts))

    if sigma is not None and np.all(sigma > 0):
        chi2_reduced = float(np.sum((residuals / sigma) ** 2) / dof)
    else:
        chi2_reduced = float(ss_res / dof)

    # Standard errors
    if pcov is not None and np.all(np.isfinite(pcov)):
        diag = np.maximum(np.diag(pcov), 0.0)
        perr = np.sqrt(diag)
        # Correlation matrix
        denom = np.outer(perr, perr)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.where(denom > 0, pcov / denom, np.eye(n_params))
            correlation = np.clip(correlation, -1.0, 1.0)
    else:
        perr = np.full(n_params, np.nan)
        pcov = np.full((n_params, n_params), np.nan)
        correlation = np.full((n_params, n_params), np.nan)

    # Confidence intervals via Student's t
    alpha_ci = 1.0 - confidence_level
    if dof > 0:
        t_val = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df=dof))
    else:
        t_val = float(stats.norm.ppf(1.0 - alpha_ci / 2.0))

    params_dict = {name: float(popt[i]) for i, name in enumerate(param_names)}
    stderr_dict = {name: float(perr[i]) for i, name in enumerate(param_names)}
    ci_dict = {}
    for i, name in enumerate(param_names):
        err = perr[i]
        if np.isfinite(err):
            ci_dict[name] = (float(popt[i] - t_val * err), float(popt[i] + t_val * err))
        else:
            ci_dict[name] = (float("-inf"), float("inf"))

    return (
        params_dict,
        stderr_dict,
        correlation,
        ci_dict,
        float(r_squared),
        float(r_squared_adj),
        rmse,
        chi2_reduced,
        residuals,
    )


# ==============================================================================
# Core Generic Regression Engines
# ==============================================================================


def fit_curve(
    model: Callable[..., float | np.ndarray],
    xdata: ArrayLike,
    ydata: ArrayLike,
    *,
    p0: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    param_names: Sequence[str] | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    confidence_level: float = 0.95,
    **curve_fit_kwargs: Any,
) -> FitResult:
    r"""Perform non-linear least-squares curve fitting with physical bounds and uncertainty.

    Wraps :func:`scipy.optimize.curve_fit` to estimate model parameters, computing
    asymptotic standard errors, covariance and correlation matrices, Student's :math:`t`
    confidence intervals, and goodness-of-fit metrics (:math:`R^2`, adjusted :math:`R^2`,
    RMSE, :math:`\chi_\nu^2`).

    Parameters
    ----------
    model : callable
        Model function ``f(x, *params)`` returning predicted dependent variable values.
    xdata : array-like
        Independent variable data array.
    ydata : array-like
        Dependent variable observed data array.
    p0 : sequence of float, optional
        Initial parameter guesses. If None, SciPy defaults are used.
    bounds : 2-tuple of sequence of float, optional
        Lower and upper parameter bounds ``(lower_bounds, upper_bounds)``.
    param_names : sequence of str, optional
        Human-readable names for the fitted parameters. If omitted, named ``p0, p1, ...``.
    sigma : array-like, optional
        Measurement uncertainties (standard deviations) of ``ydata``.
    absolute_sigma : bool, default False
        If True, ``sigma`` represents absolute uncertainties and ``pcov`` is not scaled.
    confidence_level : float, default 0.95
        Confidence level for two-tailed Student's :math:`t` confidence intervals (:math:`0 < \text{level} < 1`).
    **curve_fit_kwargs : Any
        Additional keyword arguments forwarded to :func:`scipy.optimize.curve_fit`.

    Returns
    -------
    FitResult
        Container holding fitted parameters, standard errors, correlation,
        confidence intervals, residuals, and goodness-of-fit metrics.
    """
    x_arr, y_arr = _validate_arrays(xdata, ydata, min_points=2)
    sig_arr = np.asarray(sigma, dtype=float).ravel() if sigma is not None else None
    if sig_arr is not None and len(sig_arr) != len(y_arr):
        raise ValueError("Length of sigma must match length of ydata.")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be between 0 and 1, got {confidence_level}."
        )

    # Default bounds and method
    cf_bounds = bounds if bounds is not None else (-np.inf, np.inf)

    kwargs = dict(curve_fit_kwargs)
    if bounds is not None:
        if "ftol" not in kwargs:
            kwargs["ftol"] = 1e-15
        if "xtol" not in kwargs:
            kwargs["xtol"] = 1e-15
        if "gtol" not in kwargs:
            kwargs["gtol"] = 1e-15
        if "x_scale" not in kwargs:
            kwargs["x_scale"] = "jac"

    try:
        popt, pcov = opt.curve_fit(
            model,
            x_arr,
            y_arr,
            p0=p0,
            bounds=cf_bounds,
            sigma=sig_arr,
            absolute_sigma=absolute_sigma,
            **kwargs,
        )
        success = True
        message = "Optimal parameters found."
    except Exception as exc:
        raise RuntimeError(f"Curve fitting failed to converge: {exc}") from exc

    if param_names is None:
        param_names = [f"p{i}" for i in range(len(popt))]
    elif len(param_names) != len(popt):
        raise ValueError(
            f"Number of param_names ({len(param_names)}) must match number of parameters ({len(popt)})."
        )

    y_fit = np.asarray(model(x_arr, *popt), dtype=float)
    (
        params_dict,
        stderr_dict,
        correlation,
        ci_dict,
        r2,
        r2_adj,
        rmse,
        chi2_red,
        residuals,
    ) = _compute_statistics(
        y_data=y_arr,
        y_fit=y_fit,
        popt=popt,
        pcov=pcov,
        param_names=param_names,
        confidence_level=confidence_level,
        sigma=sig_arr,
    )

    return FitResult(
        params=params_dict,
        stderr=stderr_dict,
        covariance=pcov,
        correlation=correlation,
        confidence_intervals=ci_dict,
        r_squared=r2,
        r_squared_adj=r2_adj,
        rmse=rmse,
        chi2_reduced=chi2_red,
        residuals=residuals,
        y_fit=y_fit,
        success=success,
        message=message,
        nfev=0,
    )


def fit_linear(
    xdata: ArrayLike,
    ydata: ArrayLike,
    *,
    fit_intercept: bool = True,
    param_names: tuple[str, str] = ("slope", "intercept"),
    confidence_level: float = 0.95,
) -> FitResult:
    r"""Perform ordinary linear regression with complete statistical indicators.

    Fits :math:`y = m x + q` (or :math:`y = m x` if ``fit_intercept=False``) using
    linear least squares, computing exact analytical standard errors and covariance.

    Parameters
    ----------
    xdata : array-like
        Independent variable coordinates.
    ydata : array-like
        Dependent variable observed coordinates.
    fit_intercept : bool, default True
        Whether to calculate the intercept :math:`q` or force regression through zero.
    param_names : tuple of (str, str), default ('slope', 'intercept')
        Names for the slope and intercept parameters.
    confidence_level : float, default 0.95
        Confidence level for two-tailed Student's :math:`t` confidence intervals.

    Returns
    -------
    FitResult
        Container holding fitted slope and intercept with errors and metrics.
    """
    x_arr, y_arr = _validate_arrays(xdata, ydata, min_points=2)
    n_pts = len(x_arr)

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be between 0 and 1, got {confidence_level}."
        )

    if fit_intercept:
        # Design matrix [x, 1]
        A = np.column_stack([x_arr, np.ones(n_pts)])
        popt, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        names = [param_names[0], param_names[1]]
        n_params = 2
    else:
        A = x_arr[:, np.newaxis]
        popt, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        names = [param_names[0]]
        n_params = 1

    y_fit = A @ popt
    dof = max(n_pts - n_params, 1)
    res = y_arr - y_fit
    s_sq = np.sum(res**2) / dof

    # Covariance matrix: s^2 * (A^T A)^(-1)
    try:
        inv_ata = np.linalg.inv(A.T @ A)
        pcov = s_sq * inv_ata
    except np.linalg.LinAlgError:
        pcov = np.full((n_params, n_params), np.nan)

    (
        params_dict,
        stderr_dict,
        correlation,
        ci_dict,
        r2,
        r2_adj,
        rmse,
        chi2_red,
        residuals,
    ) = _compute_statistics(
        y_data=y_arr,
        y_fit=y_fit,
        popt=popt,
        pcov=pcov,
        param_names=names,
        confidence_level=confidence_level,
    )

    return FitResult(
        params=params_dict,
        stderr=stderr_dict,
        covariance=pcov,
        correlation=correlation,
        confidence_intervals=ci_dict,
        r_squared=r2,
        r_squared_adj=r2_adj,
        rmse=rmse,
        chi2_reduced=chi2_red,
        residuals=residuals,
        y_fit=y_fit,
        success=True,
        message="Linear least squares solved successfully.",
        nfev=1,
    )


# ==============================================================================
# Technique-Specific Fitters
# ==============================================================================


def fit_cottrell(
    time: ArrayLike,
    current: ArrayLike,
    *,
    method: Literal["linear", "nonlinear"] = "linear",
    n: float = 1,
    area: float = 1.0,
    c_bulk: float = 1e-3,
    fit_background: bool = False,
    p0: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    confidence_level: float = 0.95,
    F: float = FARADAY,
) -> CottrellFitResult:
    r"""Fit chronoamperometric current transient data to the Cottrell equation.

    Extracts the diffusion coefficient :math:`D` and optional background current offset
    :math:`I_{\text{bg}}` via either linear regression of :math:`I` vs. :math:`t^{-1/2}`
    or non-linear curve fitting of :math:`I(t)`.

    .. math::

        I(t) = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}} + I_{\text{bg}}

    Parameters
    ----------
    time : array-like
        Array of time points :math:`t` in seconds (:math:`t > 0`).
    current : array-like
        Measured Faradaic currents :math:`I` in Amperes.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression methodology:
        - ``'linear'``: Fits :math:`I` vs. :math:`t^{-1/2}`.
        - ``'nonlinear'``: Directly optimizes :math:`I(t) = n F A c^* \sqrt{D / (\pi t)} + I_{\text{bg}}`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    area : float, default 1.0
        Electrode surface area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3` (:math:`c^* > 0`).
    fit_background : bool, default False
        Whether to fit a constant background current offset :math:`I_{\text{bg}}`.
    p0 : sequence of float, optional
        Initial guesses for non-linear fit ``[D]`` or ``[D, i_bg]``.
    bounds : tuple, optional
        Parameter bounds for non-linear fitting. Defaults to :math:`D \in (10^{-16}, 1.0)`.
    confidence_level : float, default 0.95
        Confidence level for parameter intervals (:math:`0 < \text{level} < 1`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}\cdot\text{mol}^{-1}`.

    Returns
    -------
    CottrellFitResult
        Container holding extracted :math:`D`, :math:`I_{\text{bg}}`, standard errors,
        confidence intervals, and regression metrics.
    """
    t_arr, i_arr = _validate_arrays(time, current, min_points=2)
    if np.any(t_arr <= 0):
        raise ValueError("All time coordinates must be strictly positive (t > 0).")
    if n <= 0 or area <= 0 or c_bulk <= 0 or F <= 0:
        raise ValueError("Parameters n, area, c_bulk, and F must be strictly positive.")

    prefactor = (n * F * area * c_bulk) / np.sqrt(np.pi)

    if method == "linear":
        x_inv_sqrt_t = 1.0 / np.sqrt(t_arr)
        lin_res = fit_linear(
            x_inv_sqrt_t,
            i_arr,
            fit_intercept=fit_background,
            param_names=("slope", "i_bg"),
            confidence_level=confidence_level,
        )
        slope = lin_res.params["slope"]
        slope_err = lin_res.stderr.get("slope", 0.0)
        i_bg = lin_res.params.get("i_bg", 0.0)

        # D = (slope / prefactor)^2
        d_val = float((slope / prefactor) ** 2)
        # Error propagation: sigma_D = 2 * (slope / prefactor^2) * sigma_slope = 2 * D * (sigma_slope / slope)
        d_err = (
            float(2.0 * d_val * abs(slope_err / slope))
            if slope != 0 and np.isfinite(slope_err)
            else float("nan")
        )

        params = {"D": d_val, "i_bg": i_bg}
        stderr = {"D": d_err, "i_bg": lin_res.stderr.get("i_bg", 0.0)}

        # CI for D
        ci_slope = lin_res.confidence_intervals.get("slope", (0.0, 0.0))
        d_ci_low = float((max(ci_slope[0], 0.0) / prefactor) ** 2)
        d_ci_high = float((max(ci_slope[1], 0.0) / prefactor) ** 2)
        ci_dict = {
            "D": (d_ci_low, d_ci_high),
            "i_bg": lin_res.confidence_intervals.get("i_bg", (0.0, 0.0)),
        }

        return CottrellFitResult(
            d=d_val,
            d_stderr=d_err,
            i_bg=i_bg,
            slope=slope,
            params=params,
            stderr=stderr,
            covariance=lin_res.covariance,
            correlation=lin_res.correlation,
            confidence_intervals=ci_dict,
            r_squared=lin_res.r_squared,
            r_squared_adj=lin_res.r_squared_adj,
            rmse=lin_res.rmse,
            chi2_reduced=lin_res.chi2_reduced,
            residuals=lin_res.residuals,
            y_fit=lin_res.y_fit,
            success=lin_res.success,
            message=lin_res.message,
            nfev=lin_res.nfev,
        )

    elif method == "nonlinear":
        model_fn: Callable[..., Any]
        if fit_background:

            def model_fn_bg(t, d_param, bg_param):
                return (
                    prefactor * np.sqrt(np.maximum(d_param, 0.0)) / np.sqrt(t)
                    + bg_param
                )

            model_fn = model_fn_bg
            p_init = p0 if p0 is not None else [1e-5, 0.0]
            b_bounds = (
                bounds if bounds is not None else ([1e-18, -np.inf], [1.0, np.inf])
            )
            p_names = ["D", "i_bg"]
        else:

            def model_fn_nobg(t, d_param):
                return prefactor * np.sqrt(np.maximum(d_param, 0.0)) / np.sqrt(t)

            model_fn = model_fn_nobg
            p_init = p0 if p0 is not None else [1e-5]
            b_bounds = bounds if bounds is not None else ([1e-18], [1.0])
            p_names = ["D"]

        res = fit_curve(
            model_fn,
            t_arr,
            i_arr,
            p0=p_init,
            bounds=b_bounds,
            param_names=p_names,
            confidence_level=confidence_level,
        )
        d_val = res.params["D"]
        d_err = res.stderr.get("D", 0.0)
        i_bg = res.params.get("i_bg", 0.0)

        return CottrellFitResult(
            d=d_val,
            d_stderr=d_err,
            i_bg=i_bg,
            slope=None,
            params=res.params,
            stderr=res.stderr,
            covariance=res.covariance,
            correlation=res.correlation,
            confidence_intervals=res.confidence_intervals,
            r_squared=res.r_squared,
            r_squared_adj=res.r_squared_adj,
            rmse=res.rmse,
            chi2_reduced=res.chi2_reduced,
            residuals=res.residuals,
            y_fit=res.y_fit,
            success=res.success,
            message=res.message,
            nfev=res.nfev,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'; must be 'linear' or 'nonlinear'."
        )


def fit_randles_sevcik(
    scan_rate: ArrayLike,
    peak_current: ArrayLike,
    *,
    method: Literal["linear", "nonlinear"] = "linear",
    model: Literal["reversible", "irreversible"] = "reversible",
    n: float = 1,
    area: float = 1.0,
    c_bulk: float = 1e-3,
    alpha: float = 0.5,
    n_alpha: float = 1,
    T: float = STANDARD_TEMPERATURE,
    fit_intercept: bool = True,
    confidence_level: float = 0.95,
    F: float = FARADAY,
    R: float = GAS_CONSTANT,
) -> RandlesSevcikFitResult:
    r"""Fit peak voltammetric currents to the Randles–Ševčík equation.

    Regresses peak current :math:`I_p` against scan rate :math:`v` to determine the
    diffusion coefficient :math:`D` for reversible or irreversible electron transfer.

    .. math::

        I_p = K \cdot n F A c^* \sqrt{\frac{F D}{R T}} \cdot v^{1/2} + I_{\text{offset}}

    Parameters
    ----------
    scan_rate : array-like
        Array of scan rates :math:`v` in :math:`\text{V}/\text{s}` (:math:`v > 0`).
    peak_current : array-like
        Array of peak currents :math:`I_p` in Amperes.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression methodology:
        - ``'linear'``: Fits :math:`I_p` vs. :math:`v^{1/2}`.
        - ``'nonlinear'``: Directly optimizes :math:`I_p(v)`.
    model : {'reversible', 'irreversible'}, default 'reversible'
        Electron transfer reversibility regime:
        - ``'reversible'``: Uses prefactor :math:`0.4463 \sqrt{n}`.
        - ``'irreversible'``: Uses prefactor :math:`0.4958 \sqrt{\alpha n_\alpha}`.
    n : float, default 1
        Total number of electrons transferred (:math:`n > 0`).
    area : float, default 1.0
        Electrode area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3` (:math:`c^* > 0`).
    alpha : float, default 0.5
        Charge-transfer coefficient for irreversible systems (:math:`0 < \alpha < 1`).
    n_alpha : float, default 1
        Number of electrons transferred up to the rate-determining step.
    T : float, default STANDARD_TEMPERATURE
        Absolute temperature in Kelvin (:math:`T > 0`).
    fit_intercept : bool, default True
        Whether to include a non-zero current offset intercept.
    confidence_level : float, default 0.95
        Confidence level for two-tailed Student's :math:`t` confidence intervals.
    F : float, default FARADAY
        Faraday constant.
    R : float, default GAS_CONSTANT
        Molar gas constant.

    Returns
    -------
    RandlesSevcikFitResult
        Container holding extracted :math:`D`, slope, intercept, standard errors, and metrics.
    """
    v_arr, ip_arr = _validate_arrays(scan_rate, peak_current, min_points=2)
    if np.any(v_arr <= 0):
        raise ValueError("All scan rates must be strictly positive (v > 0).")

    if model == "reversible":
        c_factor = 0.4463 * n * F * area * c_bulk * np.sqrt(n * F / (R * T))
    elif model == "irreversible":
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        c_factor = (
            0.4958 * n * F * area * c_bulk * np.sqrt(alpha * n_alpha * F / (R * T))
        )
    else:
        raise ValueError(
            f"Unsupported model '{model}'; must be 'reversible' or 'irreversible'."
        )

    if method == "linear":
        sqrt_v = np.sqrt(v_arr)
        lin_res = fit_linear(
            sqrt_v,
            ip_arr,
            fit_intercept=fit_intercept,
            param_names=("slope", "intercept"),
            confidence_level=confidence_level,
        )
        slope = lin_res.params["slope"]
        intercept = lin_res.params.get("intercept", 0.0)
        slope_err = lin_res.stderr.get("slope", 0.0)

        # D = (slope / c_factor)^2
        d_val = float((slope / c_factor) ** 2)
        d_err = (
            float(2.0 * d_val * abs(slope_err / slope))
            if slope != 0 and np.isfinite(slope_err)
            else float("nan")
        )

        ci_slope = lin_res.confidence_intervals.get("slope", (0.0, 0.0))
        d_ci_low = float((max(ci_slope[0], 0.0) / c_factor) ** 2)
        d_ci_high = float((max(ci_slope[1], 0.0) / c_factor) ** 2)

        params = {"D": d_val, "intercept": intercept}
        stderr = {"D": d_err, "intercept": lin_res.stderr.get("intercept", 0.0)}
        ci_dict = {
            "D": (d_ci_low, d_ci_high),
            "intercept": lin_res.confidence_intervals.get("intercept", (0.0, 0.0)),
        }

        return RandlesSevcikFitResult(
            d=d_val,
            d_stderr=d_err,
            slope=slope,
            intercept=intercept,
            params=params,
            stderr=stderr,
            covariance=lin_res.covariance,
            correlation=lin_res.correlation,
            confidence_intervals=ci_dict,
            r_squared=lin_res.r_squared,
            r_squared_adj=lin_res.r_squared_adj,
            rmse=lin_res.rmse,
            chi2_reduced=lin_res.chi2_reduced,
            residuals=lin_res.residuals,
            y_fit=lin_res.y_fit,
            success=lin_res.success,
            message=lin_res.message,
            nfev=lin_res.nfev,
        )
    elif method == "nonlinear":
        model_fn: Callable[..., Any]
        if fit_intercept:

            def model_fn_int(v, d_param, off_param):
                return (
                    c_factor * np.sqrt(np.maximum(d_param, 0.0)) * np.sqrt(v)
                    + off_param
                )

            model_fn = model_fn_int
            p_init = [1e-5, 0.0]
            b_bounds = ([1e-18, -np.inf], [1.0, np.inf])
            p_names = ["D", "intercept"]
        else:

            def model_fn_noint(v, d_param):
                return c_factor * np.sqrt(np.maximum(d_param, 0.0)) * np.sqrt(v)

            model_fn = model_fn_noint
            p_init = [1e-5]
            b_bounds = ([1e-18], [1.0])
            p_names = ["D"]

        res = fit_curve(
            model_fn,
            v_arr,
            ip_arr,
            p0=p_init,
            bounds=b_bounds,
            param_names=p_names,
            confidence_level=confidence_level,
        )
        d_val = res.params["D"]
        d_err = res.stderr.get("D", 0.0)
        intercept = res.params.get("intercept", 0.0)

        return RandlesSevcikFitResult(
            d=d_val,
            d_stderr=d_err,
            slope=float(c_factor * np.sqrt(d_val)),
            intercept=intercept,
            params=res.params,
            stderr=res.stderr,
            covariance=res.covariance,
            correlation=res.correlation,
            confidence_intervals=res.confidence_intervals,
            r_squared=res.r_squared,
            r_squared_adj=res.r_squared_adj,
            rmse=res.rmse,
            chi2_reduced=res.chi2_reduced,
            residuals=res.residuals,
            y_fit=res.y_fit,
            success=res.success,
            message=res.message,
            nfev=res.nfev,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'; must be 'linear' or 'nonlinear'."
        )


def fit_levich(
    current: ArrayLike,
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    method: Literal["linear", "nonlinear"] = "linear",
    n: float = 1,
    area: float = 1.0,
    c_bulk: float = 1e-3,
    nu: float = 0.01,
    fit_intercept: bool = False,
    confidence_level: float = 0.95,
    F: float = FARADAY,
) -> LevichFitResult:
    r"""Fit Rotating Disk Electrode limiting current data to the Levich equation.

    Extracts the Levich constant :math:`B` and diffusion coefficient :math:`D`
    from convective limiting currents :math:`I_L` measured across angular rotation rates :math:`\omega`.

    .. math::

        I_L = 0.620\, n F A D^{2/3} \nu^{-1/6} c^* \omega^{1/2}

    Parameters
    ----------
    current : array-like
        Measured limiting currents :math:`I_L` in Amperes.
    omega : array-like, optional
        Angular rotation velocities in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega > 0`).
    rpm : array-like, optional
        Rotation rates in RPM. Either ``omega`` or ``rpm`` must be supplied.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression methodology:
        - ``'linear'``: Linear regression of :math:`I_L` vs. :math:`\omega^{1/2}`.
        - ``'nonlinear'``: Non-linear regression on :math:`I_L(\omega)`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    area : float, default 1.0
        Electrode area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3` (:math:`c^* > 0`).
    nu : float, default 0.01
        Kinematic viscosity in :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}` (:math:`\nu > 0`).
    fit_intercept : bool, default False
        Whether to include a non-zero current offset intercept.
    confidence_level : float, default 0.95
        Confidence level for parameter intervals.
    F : float, default FARADAY
        Faraday constant.

    Returns
    -------
    LevichFitResult
        Container holding extracted Levich constant :math:`B`, :math:`D`, standard errors, and metrics.
    """
    w_arr, _ = _resolve_rotation(omega, rpm)
    w_arr, i_arr = _validate_arrays(w_arr, current, min_points=2)
    if np.any(w_arr <= 0):
        raise ValueError("All rotation speeds must be positive (omega > 0).")

    # B = 0.620 * n * F * area * c_bulk * nu^(-1/6) * D^(2/3)
    k_factor = 0.620 * n * F * area * c_bulk * (nu ** (-1.0 / 6.0))

    if method == "linear":
        sqrt_w = np.sqrt(w_arr)
        lin_res = fit_linear(
            sqrt_w,
            i_arr,
            fit_intercept=fit_intercept,
            param_names=("B", "intercept"),
            confidence_level=confidence_level,
        )
        b_val = lin_res.params["B"]
        b_err = lin_res.stderr.get("B", 0.0)

        # D = (B / k_factor)^(3/2)
        if b_val > 0 and k_factor > 0:
            d_val = float((b_val / k_factor) ** 1.5)
            # sigma_D = 1.5 * D * (sigma_B / B)
            d_err = (
                float(1.5 * d_val * abs(b_err / b_val))
                if np.isfinite(b_err)
                else float("nan")
            )
        else:
            d_val = float("nan")
            d_err = float("nan")

        ci_b = lin_res.confidence_intervals.get("B", (0.0, 0.0))
        d_ci_low = float((max(ci_b[0], 0.0) / k_factor) ** 1.5)
        d_ci_high = float((max(ci_b[1], 0.0) / k_factor) ** 1.5)

        params = {"B": b_val, "D": d_val}
        stderr = {"B": b_err, "D": d_err}
        ci_dict = {"B": ci_b, "D": (d_ci_low, d_ci_high)}

        return LevichFitResult(
            levich_constant=b_val,
            d=d_val,
            d_stderr=d_err,
            slope=b_val,
            params=params,
            stderr=stderr,
            covariance=lin_res.covariance,
            correlation=lin_res.correlation,
            confidence_intervals=ci_dict,
            r_squared=lin_res.r_squared,
            r_squared_adj=lin_res.r_squared_adj,
            rmse=lin_res.rmse,
            chi2_reduced=lin_res.chi2_reduced,
            residuals=lin_res.residuals,
            y_fit=lin_res.y_fit,
            success=lin_res.success,
            message=lin_res.message,
            nfev=lin_res.nfev,
        )
    elif method == "nonlinear":

        def model_fn(w, d_param):
            return k_factor * (np.maximum(d_param, 0.0) ** (2.0 / 3.0)) * np.sqrt(w)

        res = fit_curve(
            model_fn,
            w_arr,
            i_arr,
            p0=[1e-5],
            bounds=([1e-18], [1.0]),
            param_names=["D"],
            confidence_level=confidence_level,
        )
        d_val = res.params["D"]
        d_err = res.stderr.get("D", 0.0)
        b_val = float(k_factor * (d_val ** (2.0 / 3.0)))

        params = {"B": b_val, "D": d_val}
        stderr = {
            "B": float(1.5 * b_val * (d_err / d_val)) if d_val > 0 else 0.0,
            "D": d_err,
        }

        return LevichFitResult(
            levich_constant=b_val,
            d=d_val,
            d_stderr=d_err,
            slope=b_val,
            params=params,
            stderr=stderr,
            covariance=res.covariance,
            correlation=res.correlation,
            confidence_intervals=res.confidence_intervals,
            r_squared=res.r_squared,
            r_squared_adj=res.r_squared_adj,
            rmse=res.rmse,
            chi2_reduced=res.chi2_reduced,
            residuals=res.residuals,
            y_fit=res.y_fit,
            success=res.success,
            message=res.message,
            nfev=res.nfev,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'; must be 'linear' or 'nonlinear'."
        )


def fit_koutecky_levich(
    current: ArrayLike,
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    method: Literal["linear", "nonlinear"] = "linear",
    n: float = 1,
    area: float = 1.0,
    c_bulk: float = 1e-3,
    nu: float = 0.01,
    confidence_level: float = 0.95,
    F: float = FARADAY,
) -> KouteckyLevichFitResult:
    r"""Fit RDE series to the Koutecký–Levich equation to extract kinetic and transport parameters.

    Separates charge-transfer limitation (:math:`I_K`) from mass-transfer limitation (:math:`I_L`)
    to determine the heterogeneous rate constant :math:`k^0` and diffusion coefficient :math:`D`.

    .. math::

        \frac{1}{I} = \frac{1}{I_K} + \frac{1}{B} \omega^{-1/2}

    Parameters
    ----------
    current : array-like
        Measured Faradaic currents :math:`I` in Amperes at each rotation velocity (:math:`I > 0`).
    omega : array-like, optional
        Angular rotation velocities in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega > 0`).
    rpm : array-like, optional
        Rotation rates in RPM. Either ``omega`` or ``rpm`` must be supplied.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression methodology:
        - ``'linear'``: Fits :math:`1/I` vs. :math:`\omega^{-1/2}`.
        - ``'nonlinear'``: Directly fits :math:`I(\omega) = \frac{I_K B \sqrt{\omega}}{I_K + B \sqrt{\omega}}`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    area : float, default 1.0
        Electrode area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3` (:math:`c^* > 0`).
    nu : float, default 0.01
        Kinematic viscosity in :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}` (:math:`\nu > 0`).
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` intervals.
    F : float, default FARADAY
        Faraday constant.

    Returns
    -------
    KouteckyLevichFitResult
        Container holding extracted :math:`I_K`, :math:`k^0`, :math:`B`, :math:`D`, errors, and metrics.
    """
    w_arr, _ = _resolve_rotation(omega, rpm)
    w_arr, i_arr = _validate_arrays(w_arr, current, min_points=2)
    if np.any(w_arr <= 0) or np.any(i_arr <= 0):
        raise ValueError(
            "All rotation rates and currents in Koutecký–Levich analysis must be strictly positive."
        )

    k_factor = 0.620 * n * F * area * c_bulk * (nu ** (-1.0 / 6.0))

    if method == "linear":
        inv_sqrt_w = 1.0 / np.sqrt(w_arr)
        inv_i = 1.0 / i_arr

        lin_res = fit_linear(
            inv_sqrt_w,
            inv_i,
            fit_intercept=True,
            param_names=("slope", "intercept"),
            confidence_level=confidence_level,
        )
        slope = lin_res.params["slope"]
        intercept = lin_res.params["intercept"]

        b_val = float(1.0 / slope) if slope > 0 else float("nan")
        i_k = float(1.0 / intercept) if intercept > 0 else float("inf")
        k_rate = (
            float(i_k / (n * F * area * c_bulk))
            if not math.isinf(i_k) and c_bulk > 0
            else float("nan")
        )
        d_val = (
            float((b_val / k_factor) ** 1.5)
            if np.isfinite(b_val) and b_val > 0
            else float("nan")
        )

        params = {"i_k": i_k, "k_rate": k_rate, "B": b_val, "D": d_val}
        # Invert errors
        intercept_err = lin_res.stderr.get("intercept", 0.0)
        ik_err = (
            float(abs(intercept_err / (intercept**2)))
            if intercept > 0 and np.isfinite(intercept_err)
            else float("nan")
        )
        stderr = {"i_k": ik_err}

        return KouteckyLevichFitResult(
            i_k=i_k,
            k_rate=k_rate,
            levich_constant=b_val,
            d=d_val,
            slope=slope,
            intercept=intercept,
            params=params,
            stderr=stderr,
            covariance=lin_res.covariance,
            correlation=lin_res.correlation,
            confidence_intervals=lin_res.confidence_intervals,
            r_squared=lin_res.r_squared,
            r_squared_adj=lin_res.r_squared_adj,
            rmse=lin_res.rmse,
            chi2_reduced=lin_res.chi2_reduced,
            residuals=lin_res.residuals,
            y_fit=lin_res.y_fit,
            success=lin_res.success,
            message=lin_res.message,
            nfev=lin_res.nfev,
        )
    elif method == "nonlinear":

        def model_fn(w, ik_param, b_param):
            return (ik_param * b_param * np.sqrt(w)) / (ik_param + b_param * np.sqrt(w))

        # Initial guesses from endpoints
        b_guess = float(np.mean(i_arr / np.sqrt(w_arr)))
        ik_guess = float(np.max(i_arr) * 2.0)

        res = fit_curve(
            model_fn,
            w_arr,
            i_arr,
            p0=[ik_guess, b_guess],
            bounds=([1e-12, 1e-12], [np.inf, np.inf]),
            param_names=["i_k", "B"],
            confidence_level=confidence_level,
        )
        i_k = res.params["i_k"]
        b_val = res.params["B"]
        k_rate = float(i_k / (n * F * area * c_bulk)) if c_bulk > 0 else float("nan")
        d_val = (
            float((b_val / k_factor) ** 1.5)
            if np.isfinite(b_val) and b_val > 0
            else float("nan")
        )

        params = {"i_k": i_k, "k_rate": k_rate, "B": b_val, "D": d_val}

        return KouteckyLevichFitResult(
            i_k=i_k,
            k_rate=k_rate,
            levich_constant=b_val,
            d=d_val,
            slope=float(1.0 / b_val) if b_val > 0 else None,
            intercept=float(1.0 / i_k) if i_k > 0 else None,
            params=params,
            stderr=res.stderr,
            covariance=res.covariance,
            correlation=res.correlation,
            confidence_intervals=res.confidence_intervals,
            r_squared=res.r_squared,
            r_squared_adj=res.r_squared_adj,
            rmse=res.rmse,
            chi2_reduced=res.chi2_reduced,
            residuals=res.residuals,
            y_fit=res.y_fit,
            success=res.success,
            message=res.message,
            nfev=res.nfev,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'; must be 'linear' or 'nonlinear'."
        )


def fit_tafel(
    overpotential: ArrayLike,
    current: ArrayLike,
    *,
    branch: Literal["anodic", "cathodic"] = "anodic",
    fit_range: tuple[float, float] | None = None,
    area: float | None = None,
    c_bulk: float | None = None,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    confidence_level: float = 0.95,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> TafelFitResult:
    r"""Perform linear Tafel regression on high-overpotential current-potential data.

    .. math::

        \log_{10}|I| = \log_{10}(I_0) + \frac{1}{b} \eta

    Parameters
    ----------
    overpotential : array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts.
    current : array-like
        Measured Faradaic currents :math:`I` in Amperes.
    branch : {'anodic', 'cathodic'}, default 'anodic'
        Kinetic branch to isolate for regression.
    fit_range : tuple of (float, float), optional
        Overpotential interval ``(eta_min, eta_max)`` over which to fit.
    area : float, optional
        Electrode area in :math:`\text{m}^2` or :math:`\text{cm}^2`.
    c_bulk : float, optional
        Bulk concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin.
    confidence_level : float, default 0.95
        Confidence level for intervals.
    R : float, default GAS_CONSTANT
        Molar gas constant.
    F : float, default FARADAY
        Faraday constant.

    Returns
    -------
    TafelFitResult
        Container holding extracted :math:`I_0`, Tafel slope :math:`b`, :math:`\alpha`, :math:`k^0`.
    """
    eta_arr, i_arr = _validate_arrays(overpotential, current, min_points=3)

    if branch == "anodic":
        mask = eta_arr > 0
    elif branch == "cathodic":
        mask = eta_arr < 0
    else:
        raise ValueError(
            f"Unsupported branch '{branch}'; must be 'anodic' or 'cathodic'."
        )

    if fit_range is not None:
        low, high = sorted(fit_range)
        mask = mask & (eta_arr >= low) & (eta_arr <= high)

    eta_sub = eta_arr[mask]
    i_sub = np.abs(i_arr[mask])

    # Filter out non-positive currents
    pos_mask = i_sub > 0
    eta_sub = eta_sub[pos_mask]
    i_sub = i_sub[pos_mask]

    if len(eta_sub) < 3:
        raise ValueError(
            f"Tafel fit requires at least 3 valid points in the selected range, got {len(eta_sub)}."
        )

    log_i = np.log10(i_sub)
    lin_res = fit_linear(
        eta_sub,
        log_i,
        fit_intercept=True,
        param_names=("dlog_deta", "intercept"),
        confidence_level=confidence_level,
    )

    dlog_deta = lin_res.params["dlog_deta"]
    intercept = lin_res.params["intercept"]
    i0 = float(10.0**intercept)
    b_val = float(1.0 / dlog_deta) if dlog_deta != 0 else float("nan")
    b_mv = float(b_val * 1000.0)

    # alpha extraction:
    # anodic: dlog_deta = (1 - alpha) * n * F / (ln(10) * R * T)
    # cathodic: dlog_deta = - alpha * n * F / (ln(10) * R * T)
    factor = np.log(10.0) * R * T / (n * F)
    if branch == "anodic":
        alpha = float(1.0 - dlog_deta * factor)
    else:
        alpha = float(-dlog_deta * factor)

    k0 = (
        float(i0 / (n * F * area * c_bulk))
        if area is not None and c_bulk is not None and area > 0 and c_bulk > 0
        else None
    )

    params = {"i0": i0, "slope": b_val, "alpha": alpha}
    if k0 is not None:
        params["k0"] = k0

    return TafelFitResult(
        i0=i0,
        slope=b_val,
        slope_mv=b_mv,
        alpha=alpha,
        k0=k0,
        params=params,
        stderr=lin_res.stderr,
        covariance=lin_res.covariance,
        correlation=lin_res.correlation,
        confidence_intervals=lin_res.confidence_intervals,
        r_squared=lin_res.r_squared,
        r_squared_adj=lin_res.r_squared_adj,
        rmse=lin_res.rmse,
        chi2_reduced=lin_res.chi2_reduced,
        residuals=lin_res.residuals,
        y_fit=lin_res.y_fit,
        success=lin_res.success,
        message=lin_res.message,
        nfev=lin_res.nfev,
    )


def fit_butler_volmer(
    overpotential: ArrayLike,
    current: ArrayLike,
    *,
    area: float | None = None,
    c_bulk: float | None = None,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    p0: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    fit_e_eq: bool = False,
    confidence_level: float = 0.95,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> ButlerVolmerFitResult:
    r"""Fit full-range current-overpotential data to the non-linear Butler–Volmer equation.

    Simultaneously estimates exchange current :math:`I_0`, transfer coefficient :math:`\alpha`,
    and optional equilibrium potential offset :math:`\Delta E`.

    .. math::

        I(\eta) = I_0 \left[\exp\left(\frac{(1 - \alpha) n F (\eta - \eta_0)}{R T}\right) - \exp\left(-\frac{\alpha n F (\eta - \eta_0)}{R T}\right)\right]

    Parameters
    ----------
    overpotential : array-like
        Array of overpotentials :math:`\eta` in Volts.
    current : array-like
        Measured Faradaic currents in Amperes.
    area : float, optional
        Electrode area in :math:`\text{m}^2` or :math:`\text{cm}^2`.
    c_bulk : float, optional
        Bulk concentration in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Absolute temperature in Kelvin.
    p0 : sequence of float, optional
        Initial guesses for ``[i0, alpha]`` or ``[i0, alpha, e_eq]``.
    bounds : tuple, optional
        Parameter bounds. Defaults to :math:`I_0 \ge 0`, :math:`0.01 \le \alpha \le 0.99`.
    fit_e_eq : bool, default False
        Whether to optimize a potential offset :math:`\eta_0`.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` intervals.
    R : float, default GAS_CONSTANT
        Molar gas constant.
    F : float, default FARADAY
        Faraday constant.

    Returns
    -------
    ButlerVolmerFitResult
        Container holding extracted :math:`I_0`, :math:`\alpha`, :math:`k^0`, and metrics.
    """
    eta_arr, i_arr = _validate_arrays(overpotential, current, min_points=3)
    f_const = (n * F) / (R * T)

    bv_model: Callable[..., Any]
    if fit_e_eq:

        def bv_model_eeq(eta, i0_param, alpha_param, e_eq_param):
            eta_eff = eta - e_eq_param
            return i0_param * (
                np.exp(np.clip((1.0 - alpha_param) * f_const * eta_eff, -700, 700))
                - np.exp(np.clip(-alpha_param * f_const * eta_eff, -700, 700))
            )

        bv_model = bv_model_eeq
        p_init = (
            p0 if p0 is not None else [float(np.max(np.abs(i_arr)) * 0.1), 0.5, 0.0]
        )
        b_bounds = (
            bounds if bounds is not None else ([0.0, 0.01, -0.5], [np.inf, 0.99, 0.5])
        )
        p_names = ["i0", "alpha", "e_eq"]
    else:

        def bv_model_no_eeq(eta, i0_param, alpha_param):
            return i0_param * (
                np.exp(np.clip((1.0 - alpha_param) * f_const * eta, -700, 700))
                - np.exp(np.clip(-alpha_param * f_const * eta, -700, 700))
            )

        bv_model = bv_model_no_eeq
        p_init = p0 if p0 is not None else [float(np.max(np.abs(i_arr)) * 0.1), 0.5]
        b_bounds = bounds if bounds is not None else ([0.0, 0.01], [np.inf, 0.99])
        p_names = ["i0", "alpha"]

    res = fit_curve(
        bv_model,
        eta_arr,
        i_arr,
        p0=p_init,
        bounds=b_bounds,
        param_names=p_names,
        confidence_level=confidence_level,
    )

    i0 = res.params["i0"]
    alpha = res.params["alpha"]
    e_eq = res.params.get("e_eq", None)

    k0 = (
        float(i0 / (n * F * area * c_bulk))
        if area is not None and c_bulk is not None and area > 0 and c_bulk > 0
        else None
    )

    params = dict(res.params)
    if k0 is not None:
        params["k0"] = k0

    return ButlerVolmerFitResult(
        i0=i0,
        alpha=alpha,
        k0=k0,
        e_eq=e_eq,
        params=params,
        stderr=res.stderr,
        covariance=res.covariance,
        correlation=res.correlation,
        confidence_intervals=res.confidence_intervals,
        r_squared=res.r_squared,
        r_squared_adj=res.r_squared_adj,
        rmse=res.rmse,
        chi2_reduced=res.chi2_reduced,
        residuals=res.residuals,
        y_fit=res.y_fit,
        success=res.success,
        message=res.message,
        nfev=res.nfev,
    )


def fit_microdisc_transient(
    time: ArrayLike,
    current: ArrayLike,
    *,
    target: Literal["D", "radius", "a"] = "D",
    radius: float | None = None,
    a: float | None = None,
    D: float | None = None,
    c_bulk: float = 1e-3,
    n: float = 1,
    model: Literal["shoup_szabo", "mahon_oldham"] = "shoup_szabo",
    p0: float | None = None,
    bounds: tuple[float, float] | None = None,
    confidence_level: float = 0.95,
    F: float = FARADAY,
) -> MicrodiscFitResult:
    r"""Fit chronoamperometric transient data at an inlaid microdisc electrode.

    Employs the Shoup & Szabo or Mahon & Oldham analytical approximations to estimate
    either diffusion coefficient :math:`D` (with known radius :math:`a`) or radius :math:`a`
    (with known :math:`D`).

    Parameters
    ----------
    time : array-like
        Array of time points in seconds (:math:`t > 0`).
    current : array-like
        Measured Faradaic currents in Amperes.
    target : {'D', 'radius', 'a'}, default 'D'
        Parameter to optimize while holding the other constant:
        - ``'D'``: Fits diffusion coefficient :math:`D` (radius must be provided).
        - ``'radius'`` or ``'a'``: Fits microdisc radius :math:`a` (:math:`D` must be provided).
    radius : float, optional
        Microdisc electrode radius in meters (:math:`\text{radius} > 0`).
    a : float, optional
        Alias for ``radius``.
    D : float, optional
        Diffusion coefficient in :math:`\text{m}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration in :math:`\text{mol}/\text{m}^3` (:math:`c^* > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    model : {'shoup_szabo', 'mahon_oldham'}, default 'shoup_szabo'
        Microdisc transient analytical approximation model.
    p0 : float, optional
        Initial parameter guess.
    bounds : tuple of (float, float), optional
        Lower and upper bounds for the target parameter.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` intervals.
    F : float, default FARADAY
        Faraday constant.

    Returns
    -------
    MicrodiscFitResult
        Container holding extracted :math:`a`, :math:`D`, standard errors, and metrics.
    """
    t_arr, i_arr = _validate_arrays(time, current, min_points=2)
    if np.any(t_arr <= 0):
        raise ValueError("All time coordinates must be strictly positive (t > 0).")

    rad_val = radius if radius is not None else a
    transient_fn = (
        microdisc_transient if model == "shoup_szabo" else mahon_oldham_transient
    )

    b_bounds: tuple[Sequence[float], Sequence[float]]
    if target == "D":
        if rad_val is None or rad_val <= 0:
            raise ValueError(
                "When fitting D, microdisc radius ('radius' or 'a') must be provided and positive."
            )

        def fit_model(t, d_param):
            return transient_fn(t, radius=rad_val, D=d_param, c_bulk=c_bulk, n=n, F=F)

        p_init = [p0 if p0 is not None else 1e-5]
        b_bounds = (
            ([bounds[0]], [bounds[1]]) if bounds is not None else ([1e-16], [1.0])
        )
        p_names = ["D"]
    elif target in ("a", "radius"):
        if D is None or D <= 0:
            raise ValueError(
                "When fitting radius/a, diffusion coefficient 'D' must be provided and positive."
            )

        def fit_model(t, a_param):
            return transient_fn(t, radius=a_param, D=D, c_bulk=c_bulk, n=n, F=F)

        p_init = [p0 if p0 is not None else 10e-6]
        b_bounds = ([bounds[0]], [bounds[1]]) if bounds is not None else ([1e-9], [1.0])
        p_names = ["a"]
    else:
        raise ValueError(f"target must be 'D', 'radius', or 'a', got '{target}'.")

    res = fit_curve(
        fit_model,
        t_arr,
        i_arr,
        p0=p_init,
        bounds=b_bounds,
        param_names=p_names,
        confidence_level=confidence_level,
    )

    d_val = float(res.params["D"]) if target == "D" else float(D)  # type: ignore[arg-type]
    a_val = float(res.params["a"]) if target in ("a", "radius") else float(rad_val)  # type: ignore[arg-type]

    params = {"a": a_val, "D": d_val}

    return MicrodiscFitResult(
        a=a_val,
        d=d_val,
        params=params,
        stderr=res.stderr,
        covariance=res.covariance,
        correlation=res.correlation,
        confidence_intervals=res.confidence_intervals,
        r_squared=res.r_squared,
        r_squared_adj=res.r_squared_adj,
        rmse=res.rmse,
        chi2_reduced=res.chi2_reduced,
        residuals=res.residuals,
        y_fit=res.y_fit,
        success=res.success,
        message=res.message,
        nfev=res.nfev,
    )


def fit_secm_approach(
    distance: ArrayLike,
    normalized_current: ArrayLike,
    *,
    substrate: Literal["conducting", "insulating", "kinetic"] = "conducting",
    a: float = 10e-6,
    rg: float | None = None,
    d_offset: float = 0.0,
    fit_rg: bool = True,
    fit_d_offset: bool = True,
    p0: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    confidence_level: float = 0.95,
) -> SECMApproachFitResult:
    r"""Fit SECM approach curves to Lefrou and Cornut analytical models.

    Extracts the insulator radius ratio :math:`RG = r_g / a`, zero-distance vertical positioning
    offset :math:`d_{\text{offset}}`, and optional normalized substrate rate constant :math:`\kappa`.

    .. math::

        I_T(d) = I_T^{\text{model}}\left(\frac{d - d_{\text{offset}}}{a}, \, RG\right)

    Parameters
    ----------
    distance : array-like
        Experimental probe position / stage coordinate :math:`d` in meters.
    normalized_current : array-like
        Experimental normalized tip current :math:`I_T = i_T / i_{T,\infty}`.
    substrate : {'conducting', 'insulating', 'kinetic'}, default 'conducting'
        Substrate boundary condition model:
        - ``'conducting'``: Diffusion-controlled regeneration (positive feedback).
        - ``'insulating'``: Hindered diffusion (negative feedback).
        - ``'kinetic'``: Substrate kinetics with finite rate parameter :math:`\kappa`.
    a : float, default 10e-6
        Microdisc tip electrode radius in meters (:math:`a > 0`).
    rg : float, optional
        Fixed insulator radius ratio if ``fit_rg=False`` (:math:`RG \ge 1.02`).
    d_offset : float, default 0.0
        Fixed distance offset if ``fit_d_offset=False``.
    fit_rg : bool, default True
        Whether to fit the insulator ratio :math:`RG`.
    fit_d_offset : bool, default True
        Whether to fit the vertical distance zero-offset :math:`d_{\text{offset}}`.
    p0 : sequence of float, optional
        Initial parameter guesses.
    bounds : tuple, optional
        Parameter bounds for optimization.
    confidence_level : float, default 0.95
        Confidence level for parameter intervals.

    Returns
    -------
    SECMApproachFitResult
        Container holding extracted :math:`RG`, :math:`d_{\text{offset}}`, :math:`\kappa`, and metrics.
    """
    d_arr, it_arr = _validate_arrays(distance, normalized_current, min_points=3)
    if a <= 0:
        raise ValueError(f"Tip radius a must be strictly positive (a > 0), got {a}.")

    p_names: list[str] = []
    p_init: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    if fit_rg:
        p_names.append("rg")
        p_init.append(10.0 if rg is None else rg)
        lb.append(1.02)
        ub.append(1000.0)

    if fit_d_offset:
        p_names.append("d_offset")
        p_init.append(d_offset)
        # Offset shouldn't exceed minimum distance
        lb.append(-np.inf)
        ub.append(float(np.min(d_arr)))

    if substrate == "kinetic":
        p_names.append("k_sub")
        p_init.append(1.0)
        lb.append(0.0)
        ub.append(1e6)

    if not p_names:
        raise ValueError(
            "At least one parameter (rg, d_offset, or k_sub) must be selected for fitting."
        )

    if p0 is not None:
        p_init = list(p0)
    if bounds is not None:
        b_bounds = bounds
    else:
        b_bounds = (lb, ub)

    def secm_model(d, *params):
        param_dict = dict(zip(p_names, params))
        rg_val = param_dict.get("rg", rg)
        if rg_val is None:
            rg_val = 10.0
        offset_val = param_dict.get("d_offset", d_offset)
        l_norm = np.maximum((d - offset_val) / a, 1e-5)

        if substrate == "conducting":
            return secm_approach_positive_feedback(l_norm, RG=rg_val)
        elif substrate == "insulating":
            return secm_approach_negative_feedback(l_norm, RG=rg_val)
        elif substrate == "kinetic":
            k_val = param_dict.get("k_sub", 1.0)
            return secm_approach_curve(l_norm, RG=rg_val, kappa=k_val)
        else:
            raise ValueError(f"Unknown substrate '{substrate}'.")

    res = fit_curve(
        secm_model,
        d_arr,
        it_arr,
        p0=p_init,
        bounds=b_bounds,
        param_names=p_names,
        confidence_level=confidence_level,
    )

    rg_opt = float(res.params.get("rg", rg if rg is not None else 10.0))
    offset_opt = float(res.params.get("d_offset", d_offset))
    k_sub_opt = float(res.params["k_sub"]) if "k_sub" in res.params else None

    params = {"rg": rg_opt, "d_offset": offset_opt}
    if k_sub_opt is not None:
        params["k_sub"] = k_sub_opt

    return SECMApproachFitResult(
        rg=rg_opt,
        d_offset=offset_opt,
        k_sub=k_sub_opt,
        params=params,
        stderr=res.stderr,
        covariance=res.covariance,
        correlation=res.correlation,
        confidence_intervals=res.confidence_intervals,
        r_squared=res.r_squared,
        r_squared_adj=res.r_squared_adj,
        rmse=res.rmse,
        chi2_reduced=res.chi2_reduced,
        residuals=res.residuals,
        y_fit=res.y_fit,
        success=res.success,
        message=res.message,
        nfev=res.nfev,
    )


__all__ = [
    "ButlerVolmerFitResult",
    "CottrellFitResult",
    "FitResult",
    "KouteckyLevichFitResult",
    "LevichFitResult",
    "MicrodiscFitResult",
    "RandlesSevcikFitResult",
    "SECMApproachFitResult",
    "TafelFitResult",
    "fit_butler_volmer",
    "fit_cottrell",
    "fit_curve",
    "fit_koutecky_levich",
    "fit_levich",
    "fit_linear",
    "fit_microdisc_transient",
    "fit_randles_sevcik",
    "fit_secm_approach",
    "fit_tafel",
]
