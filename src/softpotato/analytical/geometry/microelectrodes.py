"""
Analytical solutions for ultramicroelectrode (UME) geometries.
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


def _validate_radius_array(
    r: float | np.ndarray,
    name: str = "a",
) -> tuple[np.ndarray, bool]:
    if isinstance(r, bool) or not isinstance(r, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            f"{name} must be a numeric scalar or array, got {type(r).__name__}."
        )
    r_arr = np.asarray(r, dtype=float)
    is_scalar = r_arr.ndim == 0
    if np.any(r_arr <= 0):
        raise ValueError(f"{name} must be strictly positive ({name} > 0).")
    return r_arr, is_scalar


def steady_state_microdisc(
    n: int,
    a: float | np.ndarray | None = None,
    D: float = 0.0,
    c_bulk: float = 0.0,
    *,
    radius: float | np.ndarray | None = None,
) -> float | np.ndarray:
    r"""
    Calculates the steady-state diffusion-limiting current at an inlaid microdisc electrode.

    Mathematical expression (Saito equation):

    .. math::

        I_{ss} = 4 \, n F D C^0 a

    Parameters
    ----------
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    a : float or numpy.ndarray, optional
        Microdisc electrode radius in :math:`\text{cm}` (:math:`a > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).
    radius : float or numpy.ndarray, optional
        Alternative keyword argument for radius `a` for backward compatibility.

    Returns
    -------
    float or numpy.ndarray
        Steady-state limiting current :math:`I_{ss}` in Amperes (:math:`\text{A}`).
        Matches the shape of `a`.

    Raises
    ------
    TypeError
        If parameter types are invalid or if neither `a` nor `radius` is provided.
    ValueError
        If `n < 1`, `a <= 0`, `D <= 0`, or `c_bulk < 0`.
    """
    if a is None:
        if radius is not None:
            a = radius
        else:
            raise TypeError("steady_state_microdisc missing required argument: 'a'")

    n_val = _validate_n_electrons(n)
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    a_arr, is_scalar = _validate_radius_array(a, "a")

    iss = 4.0 * n_val * FARADAY * d_val * c_val * a_arr
    return float(iss) if is_scalar else iss


def steady_state_microhemisphere(
    n: int,
    r0: float | np.ndarray | None = None,
    D: float = 0.0,
    c_bulk: float = 0.0,
    *,
    radius: float | np.ndarray | None = None,
) -> float | np.ndarray:
    r"""
    Calculates the steady-state diffusion-limiting current at a microhemisphere electrode.

    Mathematical expression:

    .. math::

        I_{ss} = 2 \pi n F D C^0 r_0

    Parameters
    ----------
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    r0 : float or numpy.ndarray, optional
        Microhemisphere electrode radius in :math:`\text{cm}` (:math:`r_0 > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float
        Bulk concentration in :math:`\text{mol}/\text{cm}^3` (:math:`C^0 \ge 0`).
    radius : float or numpy.ndarray, optional
        Alternative keyword argument for radius `r0` for backward compatibility.

    Returns
    -------
    float or numpy.ndarray
        Steady-state limiting current :math:`I_{ss}` in Amperes (:math:`\text{A}`).
        Matches the shape of `r0`.

    Raises
    ------
    TypeError
        If parameter types are invalid or if neither `r0` nor `radius` is provided.
    ValueError
        If `n < 1`, `r0 <= 0`, `D <= 0`, or `c_bulk < 0`.
    """
    if r0 is None:
        if radius is not None:
            r0 = radius
        else:
            raise TypeError(
                "steady_state_microhemisphere missing required argument: 'r0'"
            )

    n_val = _validate_n_electrons(n)
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    r_arr, is_scalar = _validate_radius_array(r0, "r0")

    iss = 2.0 * np.pi * n_val * FARADAY * d_val * c_val * r_arr
    return float(iss) if is_scalar else iss


__all__ = [
    "steady_state_microdisc",
    "steady_state_microhemisphere",
]
