"""
Analytical solutions for coupled chemical reaction mechanisms.
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


def catalytic_current(
    n: int,
    area: float,
    D: float,
    c_bulk: float | np.ndarray,
    k_cat: float | np.ndarray,
    c_cat: float | np.ndarray,
) -> float | np.ndarray:
    r"""
    Calculates the steady-state catalytic limiting current for an EC' mechanism.

    Mathematical expression:

    .. math::

        I_{cat} = n F A C^0 \sqrt{D k_{cat} C_{cat}}

    Parameters
    ----------
    n : int
        Number of electrons transferred (:math:`n \ge 1`).
    area : float
        Electrode geometric area in :math:`\text{cm}^2` (:math:`A > 0`).
    D : float
        Diffusion coefficient in :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float or numpy.ndarray
        Bulk concentration of substrate (:math:`C^0 \ge 0`) in :math:`\text{mol}/\text{cm}^3`.
    k_cat : float or numpy.ndarray
        Second-order catalytic rate constant in :math:`\text{cm}^3/(\text{mol}\cdot\text{s})`
        (:math:`k_{cat} > 0`).
    c_cat : float or numpy.ndarray
        Bulk concentration of catalyst (:math:`C_{cat} \ge 0`) in :math:`\text{mol}/\text{cm}^3`.

    Returns
    -------
    float or numpy.ndarray
        Catalytic current :math:`I_{cat}` in Amperes (:math:`\text{A}`).

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `n < 1`, `area <= 0`, `D <= 0`, `c_bulk < 0`, `k_cat <= 0`, or `c_cat < 0`.
    """
    n_val = _validate_n_electrons(n)
    area_val = _validate_positive(area, "area")
    d_val = _validate_positive(D, "D")

    # Validate numeric types and bounds for concentrations and rate constant
    c_bulk_arr = np.asarray(c_bulk, dtype=float)
    k_cat_arr = np.asarray(k_cat, dtype=float)
    c_cat_arr = np.asarray(c_cat, dtype=float)

    if np.any(c_bulk_arr < 0):
        raise ValueError("c_bulk must be non-negative (c_bulk >= 0).")
    if np.any(k_cat_arr <= 0):
        raise ValueError("k_cat must be strictly positive (k_cat > 0).")
    if np.any(c_cat_arr < 0):
        raise ValueError("c_cat must be non-negative (c_cat >= 0).")

    is_scalar = (
        (c_bulk_arr.ndim == 0) and (k_cat_arr.ndim == 0) and (c_cat_arr.ndim == 0)
    )

    icat = (
        n_val * FARADAY * area_val * c_bulk_arr * np.sqrt(d_val * k_cat_arr * c_cat_arr)
    )
    return float(icat) if is_scalar else icat


__all__ = [
    "catalytic_current",
]
