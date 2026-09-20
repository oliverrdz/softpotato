"""
Analytical solutions for potential step techniques and chronocoulometry.
"""

from __future__ import annotations

import numpy as np

from softpotato.core.constants import FARADAY


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


def _validate_n_electrons(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(
            f"Number of electrons n must be an integer, got {type(n).__name__}."
        )
    if n < 1:
        raise ValueError(f"Number of electrons n must be >= 1, got {n}.")
    return n


def _validate_time_array(
    t: float | np.ndarray,
    allow_zero: bool = False,
) -> tuple[np.ndarray, bool]:
    if isinstance(t, bool) or not isinstance(t, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            f"Time t must be a numeric scalar or array, got {type(t).__name__}."
        )
    t_arr = np.asarray(t, dtype=float)
    is_scalar = t_arr.ndim == 0
    if allow_zero:
        if np.any(t_arr < 0):
            raise ValueError("Time t must be non-negative (t >= 0).")
    else:
        if np.any(t_arr <= 0):
            raise ValueError("Time t must be strictly positive (t > 0).")
    return t_arr, is_scalar


def cottrell(
    t: float | np.ndarray,
    n: int,
    area: float,
    D: float,
    c_bulk: float,
) -> float | np.ndarray:
    r"""
    Calculates the Cottrell current transient for planar diffusion.

    Mathematical expression:

    .. math::

        I(t) = \frac{n F A C^0 \sqrt{D}}{\sqrt{\pi t}}

    Parameters
    ----------
    t : float or numpy.ndarray
        Time after the potential step in seconds (:math:`t > 0`).
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    area : float
        Electrode geometric area in :math:`\text{cm}^2` (:math:`A > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration of electroactive species in :math:`\text{mol}/\text{cm}^3`
        (:math:`C^0 \ge 0`, note :math:`1\text{ mM} = 10^{-6}\text{ mol}/\text{cm}^3`).

    Returns
    -------
    float or numpy.ndarray
        Current in Amperes (:math:`\text{A}`). Matches the shape of `t`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `t <= 0`, `n < 1`, `area <= 0`, `D <= 0`, or `c_bulk < 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    t_arr, is_scalar = _validate_time_array(t, allow_zero=False)

    current = (n_val * FARADAY * area_val * c_val * np.sqrt(d_val)) / np.sqrt(
        np.pi * t_arr
    )
    return float(current) if is_scalar else current


def anson(
    t: float | np.ndarray,
    n: int,
    area: float,
    D: float,
    c_bulk: float,
    q_dl: float = 0.0,
    q_ads: float = 0.0,
) -> float | np.ndarray:
    r"""
    Calculates cumulative charge transient via the Anson equation (chronocoulometry).

    Mathematical expression:

    .. math::

        Q(t) = \frac{2 n F A C^0 \sqrt{D t}}{\sqrt{\pi}} + Q_{dl} + Q_{ads}

    Parameters
    ----------
    t : float or numpy.ndarray
        Time after the potential step in seconds (:math:`t \ge 0`).
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    area : float
        Electrode geometric area in :math:`\text{cm}^2` (:math:`A > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).
    q_dl : float, optional
        Double-layer capacitive charge in Coulombs (:math:`\text{C}`). Defaults to 0.0.
    q_ads : float, optional
        Faradaic charge from adsorbed species in Coulombs (:math:`\text{C}`). Defaults to 0.0.

    Returns
    -------
    float or numpy.ndarray
        Charge in Coulombs (:math:`\text{C}`). Matches the shape of `t`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `t < 0`, `n < 1`, `area <= 0`, `D <= 0`, or `c_bulk < 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    if isinstance(q_dl, bool) or not isinstance(q_dl, (int, float)):
        raise TypeError(f"q_dl must be a real number, got {type(q_dl).__name__}.")
    if isinstance(q_ads, bool) or not isinstance(q_ads, (int, float)):
        raise TypeError(f"q_ads must be a real number, got {type(q_ads).__name__}.")
    q_dl_val = float(q_dl)
    q_ads_val = float(q_ads)
    t_arr, is_scalar = _validate_time_array(t, allow_zero=True)

    charge = (
        (2.0 * n_val * FARADAY * area_val * c_val * np.sqrt(d_val * t_arr))
        / np.sqrt(np.pi)
        + q_dl_val
        + q_ads_val
    )
    return float(charge) if is_scalar else charge


def spherical_cottrell(
    t: float | np.ndarray,
    n: int,
    r0: float,
    D: float,
    c_bulk: float,
) -> float | np.ndarray:
    r"""
    Calculates the Cottrell current transient at a spherical electrode.

    Mathematical expression:

    .. math::

        I(t) = n F A C^0 D \left( \frac{1}{\sqrt{\pi D t}} + \frac{1}{r_0} \right)
             = 4 \pi n F D C^0 r_0 \left( 1 + \frac{r_0}{\sqrt{\pi D t}} \right)

    The electrode area :math:`A = 4 \pi r_0^2` is computed directly from the radius :math:`r_0`.

    Parameters
    ----------
    t : float or numpy.ndarray
        Time after the potential step in seconds (:math:`t > 0`).
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    r0 : float
        Spherical electrode radius in :math:`\text{cm}` (:math:`r_0 > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).

    Returns
    -------
    float or numpy.ndarray
        Current in Amperes (:math:`\text{A}`). Matches the shape of `t`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `t <= 0`, `n < 1`, `r0 <= 0`, `D <= 0`, or `c_bulk < 0`.
    """
    n_val = _validate_n_electrons(n)
    r0_val = _validate_positive(r0, "r0")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    t_arr, is_scalar = _validate_time_array(t, allow_zero=False)

    area = 4.0 * np.pi * (r0_val**2)
    current = (
        n_val
        * FARADAY
        * area
        * c_val
        * d_val
        * (1.0 / np.sqrt(np.pi * d_val * t_arr) + 1.0 / r0_val)
    )
    return float(current) if is_scalar else current


__all__ = [
    "anson",
    "cottrell",
    "spherical_cottrell",
]
