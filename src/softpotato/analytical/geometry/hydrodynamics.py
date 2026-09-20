"""
Analytical hydrodynamic solutions for rotating disk electrodes (RDE).
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


def _validate_omega(
    omega: float | np.ndarray,
) -> tuple[np.ndarray, bool]:
    if isinstance(omega, bool) or not isinstance(
        omega, (int, float, list, tuple, np.ndarray)
    ):
        raise TypeError(
            f"omega must be a numeric scalar or array, got {type(omega).__name__}."
        )
    w_arr = np.asarray(omega, dtype=float)
    is_scalar = w_arr.ndim == 0
    if np.any(w_arr <= 0):
        raise ValueError("omega must be strictly positive (omega > 0).")
    return w_arr, is_scalar


def levich(
    n: int,
    area: float,
    D: float,
    c_bulk: float,
    omega: float | np.ndarray,
    nu: float = 0.01,
) -> float | np.ndarray:
    r"""
    Calculates the Levich mass-transport limiting current at a rotating disk electrode (RDE).

    Mathematical expression:

    .. math::

        I_L = 0.62 \, n F A D^{2/3} \omega^{1/2} \nu^{-1/6} C^0

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
    omega : float or numpy.ndarray
        Angular rotation velocity in :math:`\text{rad/s}` (:math:`\omega > 0`,
        note :math:`\omega = 2 \pi f` where :math:`f = \text{rpm}/60`).
    nu : float, optional
        Kinematic viscosity of the solution in :math:`\text{cm}^2/\text{s}` (:math:`\nu > 0`).
        Defaults to 0.01 (approximate value for dilute aqueous solutions at 25 °C).

    Returns
    -------
    float or numpy.ndarray
        Levich limiting current :math:`I_L` in Amperes (:math:`\text{A}`).
        Matches the shape of `omega`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `n < 1`, `area <= 0`, `D <= 0`, `c_bulk < 0`, `omega <= 0`, or `nu <= 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    nu_val = _validate_positive(nu, "nu")
    w_arr, is_scalar = _validate_omega(omega)

    il = (
        0.62
        * n_val
        * FARADAY
        * area_val
        * (d_val ** (2.0 / 3.0))
        * np.sqrt(w_arr)
        * (nu_val ** (-1.0 / 6.0))
        * c_val
    )
    return float(il) if is_scalar else il


def koutecky_levich(
    n: int,
    area: float,
    D: float,
    c_bulk: float,
    omega: float | np.ndarray,
    k_f: float,
    nu: float = 0.01,
) -> float | np.ndarray:
    r"""
    Calculates the net steady-state current at an RDE via the Koutecký-Levich equation.

    Mathematical expression:

    .. math::

        \frac{1}{I} = \frac{1}{I_k} + \frac{1}{I_L}
                    = \frac{1}{n F A k_f C^0} + \frac{1}{0.62 n F A D^{2/3} \omega^{1/2} \nu^{-1/6} C^0}

    Solving for net current :math:`I`:

    .. math::

        I = \frac{I_k \cdot I_L}{I_k + I_L}

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
    omega : float or numpy.ndarray
        Angular rotation velocity in :math:`\text{rad/s}` (:math:`\omega > 0`).
    k_f : float
        Forward heterogeneous rate constant at the given potential in :math:`\text{cm/s}`
        (:math:`k_f > 0`).
    nu : float, optional
        Kinematic viscosity in :math:`\text{cm}^2/\text{s}` (:math:`\nu > 0`). Defaults to 0.01.

    Returns
    -------
    float or numpy.ndarray
        Net current :math:`I` in Amperes (:math:`\text{A}`). Matches the shape of `omega`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `n < 1`, `area <= 0`, `D <= 0`, `c_bulk < 0`, `omega <= 0`, `k_f <= 0`, or `nu <= 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")
    c_val = _validate_non_negative(c_bulk, "c_bulk")
    kf_val = _validate_positive(k_f, "k_f")
    nu_val = _validate_positive(nu, "nu")
    w_arr, is_scalar = _validate_omega(omega)

    # Kinetic current: I_k = n * F * A * k_f * C^0
    ik = n_val * FARADAY * area_val * kf_val * c_val

    # Levich limiting current: I_L = 0.62 * n * F * A * D^(2/3) * w^(1/2) * nu^(-1/6) * C^0
    il = (
        0.62
        * n_val
        * FARADAY
        * area_val
        * (d_val ** (2.0 / 3.0))
        * np.sqrt(w_arr)
        * (nu_val ** (-1.0 / 6.0))
        * c_val
    )

    if c_val == 0.0:
        current = np.zeros_like(w_arr)
    else:
        current = (ik * il) / (ik + il)

    return float(current) if is_scalar else current


__all__ = [
    "koutecky_levich",
    "levich",
]
