"""
Analytical solutions for voltammetry techniques.
"""

from __future__ import annotations

import numpy as np

from softpotato.core.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE


def _validate_positive(val: float, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(val).__name__}.")
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}.")
    return float(val)


def _validate_non_negative(val: float, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(val).__name__}.")
    if val < 0:
        raise ValueError(f"{name} must be non-negative, got {val}.")
    return float(val)


def _validate_n_electrons(n: int, name: str = "n") -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(
            f"Number of electrons {name} must be an integer, got {type(n).__name__}."
        )
    if n < 1:
        raise ValueError(f"Number of electrons {name} must be >= 1, got {n}.")
    return n


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Transfer coefficient alpha must be strictly between 0 and 1, got {alpha}."
        )
    return float(alpha)


def _validate_scan_rate(
    scan_rate: float | np.ndarray,
) -> tuple[np.ndarray, bool]:
    if isinstance(scan_rate, bool) or not isinstance(
        scan_rate, (int, float, list, tuple, np.ndarray)
    ):
        raise TypeError(
            f"scan_rate must be a numeric scalar or array, got {type(scan_rate).__name__}."
        )
    v_arr = np.asarray(scan_rate, dtype=float)
    is_scalar = v_arr.ndim == 0
    if np.any(v_arr <= 0):
        raise ValueError("scan_rate must be strictly positive (v > 0).")
    return v_arr, is_scalar


def randles_sevcik(
    n: int,
    area: float,
    D: float,
    c_bulk: float,
    scan_rate: float | np.ndarray,
    T: float = STANDARD_TEMPERATURE,
) -> float | np.ndarray:
    r"""
    Calculates the peak current for a reversible electron transfer via the Randles-Sevcik equation.

    Mathematical expression:

    .. math::

        I_p = 0.4463 \, n F A C^0 \left( \frac{n F v D}{R T} \right)^{1/2}

    Parameters
    ----------
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    area : float
        Electrode geometric area in :math:`\text{cm}^2` (:math:`A > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).
    scan_rate : float or numpy.ndarray
        Potential scan rate :math:`v` in :math:`\text{V}/\text{s}` (:math:`v > 0`).
    T : float, optional
        Thermodynamic temperature in Kelvin (:math:`T > 0`). Defaults to 298.15 K.

    Returns
    -------
    float or numpy.ndarray
        Peak current :math:`I_p` in Amperes (:math:`\text{A}`). Matches the shape of `scan_rate`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `n < 1`, `area <= 0`, `D <= 0`, `c_bulk < 0`, `scan_rate <= 0`, or `T <= 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    t_val = _validate_positive(T, "T")
    v_arr, is_scalar = _validate_scan_rate(scan_rate)

    factor = (n_val * FARADAY * v_arr * d_val) / (GAS_CONSTANT * t_val)
    ip = 0.4463 * n_val * FARADAY * area_val * c_val * np.sqrt(factor)
    return float(ip) if is_scalar else ip


def randles_sevcik_irreversible(
    n: int,
    area: float,
    D: float,
    c_bulk: float,
    scan_rate: float | np.ndarray,
    alpha: float = 0.5,
    T: float = STANDARD_TEMPERATURE,
) -> float | np.ndarray:
    r"""
    Calculates the peak current for a totally irreversible electron transfer.

    Mathematical expression:

    .. math::

        I_p = 0.4958 \, n F A C^0 \sqrt{D} \left( \frac{\alpha F v}{R T} \right)^{1/2}

    Parameters
    ----------
    n : int
        Total number of electrons transferred (:math:`n \ge 1`).
    area : float
        Electrode geometric area in :math:`\text{cm}^2` (:math:`A > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).
    scan_rate : float or numpy.ndarray
        Potential scan rate :math:`v` in :math:`\text{V}/\text{s}` (:math:`v > 0`).
    alpha : float, optional
        Charge transfer coefficient (:math:`0 < \alpha < 1`). Defaults to 0.5.
    T : float, optional
        Thermodynamic temperature in Kelvin (:math:`T > 0`). Defaults to 298.15 K.

    Returns
    -------
    float or numpy.ndarray
        Peak current :math:`I_p` in Amperes (:math:`\text{A}`). Matches the shape of `scan_rate`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `n < 1`, `area <= 0`, `D <= 0`, `c_bulk < 0`, `scan_rate <= 0`,
        `alpha <= 0`, `alpha >= 1`, or `T <= 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    alpha_val = _validate_alpha(alpha)
    t_val = _validate_positive(T, "T")
    v_arr, is_scalar = _validate_scan_rate(scan_rate)

    factor = (alpha_val * FARADAY * v_arr) / (GAS_CONSTANT * t_val)
    ip = 0.4958 * n_val * FARADAY * area_val * c_val * np.sqrt(d_val) * np.sqrt(factor)
    return float(ip) if is_scalar else ip


def peak_potential_irreversible(
    e0: float,
    D: float,
    k0: float,
    scan_rate: float | np.ndarray,
    alpha: float = 0.5,
    n_alpha: int = 1,
    T: float = STANDARD_TEMPERATURE,
) -> float | np.ndarray:
    r"""
    Calculates the peak potential for a totally irreversible electron transfer.

    Mathematical expression:

    .. math::

        E_p = E^{0'} - \frac{R T}{\alpha n_\alpha F} \left[
            0.780 + \ln \left( \frac{\sqrt{D}}{k^0} \right)
            + \ln \left( \sqrt{\frac{\alpha n_\alpha F v}{R T}} \right)
        \right]

    Parameters
    ----------
    e0 : float
        Formal reduction potential :math:`E^{0'}` in Volts (:math:`\text{V}`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    k0 : float
        Standard heterogeneous rate constant :math:`k^0` in :math:`\text{cm/s}` (:math:`k^0 > 0`).
    scan_rate : float or numpy.ndarray
        Potential scan rate :math:`v` in :math:`\text{V}/\text{s}` (:math:`v > 0`).
    alpha : float, optional
        Transfer coefficient (:math:`0 < \alpha < 1`). Defaults to 0.5.
    n_alpha : int, optional
        Number of electrons transferred in the rate-determining step (:math:`n_\alpha \ge 1`).
        Defaults to 1.
    T : float, optional
        Thermodynamic temperature in Kelvin (:math:`T > 0`). Defaults to 298.15 K.

    Returns
    -------
    float or numpy.ndarray
        Peak potential :math:`E_p` in Volts (:math:`\text{V}`). Matches the shape of `scan_rate`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `D <= 0`, `k0 <= 0`, `scan_rate <= 0`, `alpha <= 0`, `alpha >= 1`,
        `n_alpha < 1`, or `T <= 0`.
    """
    if isinstance(e0, bool) or not isinstance(e0, (int, float)):
        raise TypeError(f"e0 must be a real number, got {type(e0).__name__}.")
    e0_val = float(e0)
    d_val = _validate_positive(D, "D")
    k0_val = _validate_positive(k0, "k0")
    alpha_val = _validate_alpha(alpha)
    n_a_val = _validate_n_electrons(n_alpha, "n_alpha")
    t_val = _validate_positive(T, "T")
    v_arr, is_scalar = _validate_scan_rate(scan_rate)

    rt_over_anf = (GAS_CONSTANT * t_val) / (alpha_val * n_a_val * FARADAY)
    term1 = 0.780
    term2 = np.log(np.sqrt(d_val) / k0_val)
    term3 = np.log(
        np.sqrt((alpha_val * n_a_val * FARADAY * v_arr) / (GAS_CONSTANT * t_val))
    )

    ep = e0_val - rt_over_anf * (term1 + term2 + term3)
    return float(ep) if is_scalar else ep


__all__ = [
    "peak_potential_irreversible",
    "randles_sevcik",
    "randles_sevcik_irreversible",
]
