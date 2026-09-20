"""
Analytical solutions for electrochemical reversibility diagnostics.
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


def _validate_n_electrons(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(
            f"Number of electrons n must be an integer, got {type(n).__name__}."
        )
    if n < 1:
        raise ValueError(f"Number of electrons n must be >= 1, got {n}.")
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


def nicholson_psi(
    k0: float,
    D_O: float,
    scan_rate: float | np.ndarray,
    n: int = 1,
    D_R: float | None = None,
    alpha: float = 0.5,
    T: float = STANDARD_TEMPERATURE,
) -> float | np.ndarray:
    r"""
    Calculates Nicholson's dimensionless kinetic reversibility parameter :math:`\Psi`.

    Mathematical expression:

    .. math::

        \Psi = \frac{k^0 \left( D_O / D_R \right)^{\alpha/2}}{\sqrt{\pi D_O \frac{n F v}{R T}}}

    When :math:`D_R` is omitted, it is assumed equal to :math:`D_O`, simplifying the numerator to :math:`k^0`.

    Parameters
    ----------
    k0 : float
        Standard heterogeneous electron transfer rate constant in :math:`\text{cm/s}` (:math:`k^0 > 0`).
    D_O : float
        Diffusion coefficient of the oxidized species in :math:`\text{cm}^2/\text{s}` (:math:`D_O > 0`).
    scan_rate : float or numpy.ndarray
        Potential scan rate :math:`v` in :math:`\text{V}/\text{s}` (:math:`v > 0`).
    n : int, optional
        Number of electrons transferred (:math:`n \ge 1`). Defaults to 1.
    D_R : float, optional
        Diffusion coefficient of the reduced species in :math:`\text{cm}^2/\text{s}` (:math:`D_R > 0`).
        Defaults to `None` (assumed equal to `D_O`).
    alpha : float, optional
        Transfer coefficient (:math:`0 < \alpha < 1`). Defaults to 0.5.
    T : float, optional
        Thermodynamic temperature in Kelvin (:math:`T > 0`). Defaults to 298.15 K.

    Returns
    -------
    float or numpy.ndarray
        Nicholson's dimensionless parameter :math:`\Psi`. Matches the shape of `scan_rate`.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If `k0 <= 0`, `D_O <= 0`, `D_R <= 0`, `scan_rate <= 0`, `n < 1`, `alpha <= 0`,
        `alpha >= 1`, or `T <= 0`.
    """
    k0_val = _validate_positive(k0, "k0")
    do_val = _validate_positive(D_O, "D_O")
    dr_val = _validate_positive(D_R, "D_R") if D_R is not None else do_val
    n_val = _validate_n_electrons(n)
    alpha_val = _validate_alpha(alpha)
    t_val = _validate_positive(T, "T")
    v_arr, is_scalar = _validate_scan_rate(scan_rate)

    numerator = k0_val * ((do_val / dr_val) ** (alpha_val / 2.0))
    denominator = np.sqrt(
        np.pi * do_val * (n_val * FARADAY * v_arr) / (GAS_CONSTANT * t_val)
    )

    psi = numerator / denominator
    return float(psi) if is_scalar else psi


__all__ = [
    "nicholson_psi",
]
