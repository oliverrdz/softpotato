"""Analytical and empirical equations for hydrodynamic electrochemical systems.

This module provides closed-form solutions, asymptotic expansions, and empirical
models for convective-diffusion mass transport at Rotating Disk Electrodes (RDE)
and Rotating Ring-Disk Electrodes (RRDE).

Functions & Classes
-------------------
rpm_to_rad_s
    Convert rotational speed from revolutions per minute (RPM) to angular velocity (rad/s).
rad_s_to_rpm
    Convert angular velocity from rad/s to rotational speed in RPM.
levich
    Levich equation for convective mass-transport limiting current at an RDE.
levich_constant
    Compute the theoretical Levich slope parameter B.
nernst_diffusion_layer
    Compute the hydrodynamic stagnant diffusion layer thickness delta.
koutecky_levich
    Koutecký–Levich equation for mixed kinetic and mass-transfer limitation.
koutecky_levich_analysis
    Perform linear regression on Koutecký–Levich data to extract kinetic and transport parameters.
KouteckyLevichResult
    Container dataclass for extracted Koutecký–Levich diagnostic regression parameters.
collection_efficiency
    Exact Albery–Bruckenstein theoretical collection efficiency N for an RRDE.
ring_limiting_current
    Convective mass-transport limiting current at the ring electrode (unshielded or shielded).
ring_collection_current
    Ring current resulting from collection of intermediate species generated at the disk.
shielding_factor
    Theoretical geometric shielding factor for an RRDE.
rotating_ring_disk
    Unified solver evaluating RRDE geometric parameters, collection efficiency, and limiting currents.
RRDEResult
    Container dataclass for comprehensive RRDE hydrodynamic and geometric parameters.

Conventions & Units
-------------------
All equations are dimensionally consistent in either standard SI or standard
electrochemical CGS units:

- **SI**:
  :math:`D` in :math:`\\text{m}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{m}^{-3}`,
  :math:`A` in :math:`\\text{m}^2`,
  :math:`\\nu` in :math:`\\text{m}^2\\cdot\\text{s}^{-1}` (water at 25 °C :math:`\\approx 10^{-6}`),
  :math:`\\omega` in :math:`\\text{rad}\\cdot\\text{s}^{-1}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}`
  :math:`\\implies I` in :math:`\\text{A}`.
- **CGS**:
  :math:`D` in :math:`\\text{cm}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{cm}^{-3}`,
  :math:`A` in :math:`\\text{cm}^2`,
  :math:`\\nu` in :math:`\\text{cm}^2\\cdot\\text{s}^{-1}` (water at 25 °C :math:`\\approx 0.01`),
  :math:`\\omega` in :math:`\\text{rad}\\cdot\\text{s}^{-1}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}`
  :math:`\\implies I` in :math:`\\text{A}`.

Standard IUPAC sign conventions are applied:
- Cathodic (reduction) currents are positive (:math:`I > 0`).
- Anodic (oxidation) currents are negative (:math:`I < 0`).

References
----------
- V. G. Levich, *Physicochemical Hydrodynamics*, Prentice-Hall, Englewood Cliffs, NJ, 1962.
- A. J. Bard, L. R. Faulkner, H. S. White, *Electrochemical Methods:
  Fundamentals and Applications*, 3rd ed., John Wiley & Sons, 2022, Chapter 9.
- W. J. Albery, S. Bruckenstein, "Ring-disc electrodes. Part 2.—Theoretical and
  experimental collection efficiencies", *Trans. Faraday Soc.*, 62 (1966) 1920–1931.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from softpotato.constants import FARADAY

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def rpm_to_rad_s(rpm: float | ArrayLike) -> float | np.ndarray:
    r"""Convert rotational frequency in RPM to angular velocity in radians per second.

    .. math::

        \omega = \frac{2 \pi}{60} \times \text{rpm}

    Parameters
    ----------
    rpm : float or array-like
        Rotational frequency in revolutions per minute (:math:`\text{rpm} \ge 0`).

    Returns
    -------
    float or numpy.ndarray
        Angular velocity :math:`\omega` in :math:`\text{rad}\cdot\text{s}^{-1}`.
    """
    arr = np.asarray(rpm, dtype=float)
    if np.any(arr < 0):
        raise ValueError(
            f"Rotational frequency rpm must be non-negative (rpm >= 0), got {rpm!r}."
        )
    val = (2.0 * np.pi / 60.0) * arr
    return float(val.item()) if arr.ndim == 0 else val


def rad_s_to_rpm(omega: float | ArrayLike) -> float | np.ndarray:
    r"""Convert angular velocity in radians per second to rotational frequency in RPM.

    .. math::

        \text{rpm} = \frac{60}{2 \pi} \times \omega

    Parameters
    ----------
    omega : float or array-like
        Angular velocity in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega \ge 0`).

    Returns
    -------
    float or numpy.ndarray
        Rotational frequency in revolutions per minute (:math:`\text{rpm}`).
    """
    arr = np.asarray(omega, dtype=float)
    if np.any(arr < 0):
        raise ValueError(
            f"Angular velocity omega must be non-negative (omega >= 0), got {omega!r}."
        )
    val = (60.0 / (2.0 * np.pi)) * arr
    return float(val.item()) if arr.ndim == 0 else val


def _resolve_rotation(
    omega: float | ArrayLike | None,
    rpm: float | ArrayLike | None,
) -> tuple[np.ndarray, bool]:
    """Validate and resolve rotation speed given either omega or rpm."""
    if omega is None and rpm is None:
        raise ValueError(
            "Either angular velocity 'omega' (rad/s) or rotational frequency 'rpm' must be provided."
        )
    if omega is not None and rpm is not None:
        raise ValueError(
            "Specify either 'omega' or 'rpm', but not both simultaneously."
        )

    if omega is not None:
        arr = np.asarray(omega, dtype=float)
        is_scalar = arr.ndim == 0
        if np.any(arr < 0):
            raise ValueError(
                f"Angular velocity omega must be non-negative (omega >= 0), got {omega!r}."
            )
        return arr, is_scalar
    else:
        assert rpm is not None
        arr_rpm = np.asarray(rpm, dtype=float)
        is_scalar = arr_rpm.ndim == 0
        if np.any(arr_rpm < 0):
            raise ValueError(
                f"Rotational frequency rpm must be non-negative (rpm >= 0), got {rpm!r}."
            )
        return (2.0 * np.pi / 60.0) * arr_rpm, is_scalar


def _validate_hydrodynamic_params(
    n: float,
    D: float,
    c_bulk: float,
    area: float,
    nu: float,
    F: float,
) -> None:
    """Validate common physical parameters for hydrodynamic equations."""
    if n <= 0:
        raise ValueError(f"Number of electrons n must be positive (n > 0), got {n}.")
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive (D > 0), got {D}.")
    if c_bulk < 0:
        raise ValueError(
            f"Bulk concentration c_bulk must be non-negative (c_bulk >= 0), got {c_bulk}."
        )
    if area <= 0:
        raise ValueError(f"Electrode area must be positive (area > 0), got {area}.")
    if nu <= 0:
        raise ValueError(f"Kinematic viscosity nu must be positive (nu > 0), got {nu}.")
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


def levich(
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    nu: float = 0.01,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute convective mass-transport limiting current at a Rotating Disk Electrode (RDE).

    The Levich equation calculates the steady-state diffusion-limited Faradaic
    current at a planar disk electrode rotating in a Newtonian fluid under
    laminar hydrodynamic flow conditions:

    .. math::

        I_L = 0.620 \, n F A D^{2/3} \omega^{1/2} \nu^{-1/6} c^*

    Parameters
    ----------
    omega : float or array-like, optional
        Angular rotation speed in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega \ge 0`).
        Either ``omega`` or ``rpm`` must be supplied.
    rpm : float or array-like, optional
        Rotational frequency in revolutions per minute (:math:`\text{rpm} \ge 0`).
    n : float, default 1
        Number of electrons transferred per molecule of electroactive species (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient of electroactive species (:math:`D > 0`, :math:`\text{cm}^2\cdot\text{s}^{-1}` or :math:`\text{m}^2\cdot\text{s}^{-1}`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species (:math:`c^* \ge 0`, :math:`\text{mol}\cdot\text{cm}^{-3}` or :math:`\text{mol}\cdot\text{m}^{-3}`).
    area : float, default 1.0
        Geometric surface area of the disk electrode (:math:`A > 0`, :math:`\text{cm}^2` or :math:`\text{m}^2`).
    nu : float, default 0.01
        Kinematic viscosity of the electrolyte solution (:math:`\nu > 0`, :math:`\text{cm}^2\cdot\text{s}^{-1}` or :math:`\text{m}^2\cdot\text{s}^{-1}`).
        Default is 0.01 :math:`\text{cm}^2\cdot\text{s}^{-1}` (water at 20–25 °C in CGS).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}\cdot\text{mol}^{-1}` (:math:`F > 0`).

    Returns
    -------
    float or numpy.ndarray
        Convective mass-transport limiting current :math:`I_L` in Amperes (A).

    Raises
    ------
    ValueError
        If both or neither ``omega`` and ``rpm`` are provided, or if any input is non-physical.

    Examples
    --------
    >>> import softpotato as sp
    >>> # Standard 5 mm OD disk (area = 0.1963 cm^2) at 1600 rpm
    >>> i_lim = sp.levich(rpm=1600, n=1, D=1e-5, c_bulk=1e-6, area=0.1963, nu=0.01)
    >>> round(i_lim * 1e6, 2)  # microamperes
    144.13
    """
    _validate_hydrodynamic_params(n=n, D=D, c_bulk=c_bulk, area=area, nu=nu, F=F)
    w_arr, is_scalar = _resolve_rotation(omega, rpm)

    b_const = 0.620 * n * F * area * (D ** (2.0 / 3.0)) * (nu ** (-1.0 / 6.0)) * c_bulk
    i_lim = b_const * np.sqrt(w_arr)
    return _format_output(i_lim, is_scalar)


def levich_constant(
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    nu: float = 0.01,
    F: float = FARADAY,
) -> float:
    r"""Compute the theoretical Levich constant (slope parameter) B.

    .. math::

        B = 0.620 \, n F A D^{2/3} \nu^{-1/6} c^*

    such that the Levich limiting current is :math:`I_L = B \sqrt{\omega}`.

    Parameters
    ----------
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    area : float, default 1.0
        Electrode geometric area (:math:`A > 0`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    float
        Levich slope constant :math:`B` in :math:`\text{A}\cdot\text{rad}^{-1/2}\cdot\text{s}^{1/2}`.
    """
    _validate_hydrodynamic_params(n=n, D=D, c_bulk=c_bulk, area=area, nu=nu, F=F)
    return float(
        0.620 * n * F * area * (D ** (2.0 / 3.0)) * (nu ** (-1.0 / 6.0)) * c_bulk
    )


def nernst_diffusion_layer(
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    D: float = 1e-5,
    nu: float = 0.01,
) -> float | np.ndarray:
    r"""Compute the Nernst stagnant diffusion layer thickness delta for an RDE.

    According to Levich hydrodynamic boundary layer theory:

    .. math::

        \delta = 1.610 \, D^{1/3} \nu^{1/6} \omega^{-1/2}

    Parameters
    ----------
    omega : float or array-like, optional
        Angular rotation speed in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega > 0`).
    rpm : float or array-like, optional
        Rotational frequency in revolutions per minute (:math:`\text{rpm} > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`, :math:`\text{cm}^2\cdot\text{s}^{-1}` or :math:`\text{m}^2\cdot\text{s}^{-1}`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`, :math:`\text{cm}^2\cdot\text{s}^{-1}` or :math:`\text{m}^2\cdot\text{s}^{-1}`).

    Returns
    -------
    float or numpy.ndarray
        Diffusion boundary layer thickness :math:`\delta` (:math:`\text{cm}` or :math:`\text{m}`).
    """
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive (D > 0), got {D}.")
    if nu <= 0:
        raise ValueError(f"Kinematic viscosity nu must be positive (nu > 0), got {nu}.")

    w_arr, is_scalar = _resolve_rotation(omega, rpm)

    # Note: at omega = 0, delta -> infinity
    with np.errstate(divide="ignore"):
        delta = 1.610 * (D ** (1.0 / 3.0)) * (nu ** (1.0 / 6.0)) / np.sqrt(w_arr)
    return _format_output(delta, is_scalar)


def koutecky_levich(
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    i_k: float | ArrayLike | None = None,
    k: float | ArrayLike | None = None,
    i_lim: float | ArrayLike | None = None,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    nu: float = 0.01,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute total current under mixed kinetic and convective mass-transfer control.

    The Koutecký–Levich equation describes the steady-state current response
    in the presence of simultaneous interfacial charge-transfer and mass-transport
    limitations:

    .. math::

        \frac{1}{I} = \frac{1}{I_K} + \frac{1}{I_L} \implies I = \frac{I_K I_L}{I_K + I_L}

    where :math:`I_K = n F A k(E) c^*` is the potential-dependent kinetic current,
    and :math:`I_L = 0.620 \, n F A D^{2/3} \omega^{1/2} \nu^{-1/6} c^*` is the
    convective mass-transport limiting current.

    Parameters
    ----------
    omega : float or array-like, optional
        Angular rotation speed in :math:`\text{rad}\cdot\text{s}^{-1}`.
    rpm : float or array-like, optional
        Rotational frequency in revolutions per minute.
    i_k : float or array-like, optional
        Kinetic current :math:`I_K` in Amperes (A). Either ``i_k`` or ``k`` must be provided.
    k : float or array-like, optional
        Heterogeneous charge-transfer rate constant (:math:`\text{cm}\cdot\text{s}^{-1}` or :math:`\text{m}\cdot\text{s}^{-1}`).
        If provided, :math:`I_K = n F A k c^*`.
    i_lim : float or array-like, optional
        Precomputed mass-transport limiting current :math:`I_L` in Amperes (A).
        If provided, rotation speed (``omega`` or ``rpm``) is optional.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    float or numpy.ndarray
        Observed total current :math:`I` in Amperes (A).

    Raises
    ------
    ValueError
        If neither ``i_k`` nor ``k`` is supplied, or if neither ``i_lim`` nor rotation
        speed (``omega`` or ``rpm``) is provided.
    """
    _validate_hydrodynamic_params(n=n, D=D, c_bulk=c_bulk, area=area, nu=nu, F=F)

    # Resolve kinetic current
    if i_k is None and k is None:
        raise ValueError("Either kinetic current 'i_k' or rate constant 'k' must be provided.")
    if i_k is not None and k is not None:
        raise ValueError("Provide either 'i_k' or 'k', but not both.")

    if i_k is not None:
        arr_ik = np.asarray(i_k, dtype=float)
        ik_is_scalar = arr_ik.ndim == 0
    else:
        assert k is not None
        arr_k = np.asarray(k, dtype=float)
        ik_is_scalar = arr_k.ndim == 0
        if np.any(arr_k < 0):
            raise ValueError(f"Rate constant k must be non-negative (k >= 0), got {k}.")
        arr_ik = n * F * area * arr_k * c_bulk

    # Resolve limiting current
    if i_lim is not None:
        arr_il = np.asarray(i_lim, dtype=float)
        il_is_scalar = arr_il.ndim == 0
    else:
        arr_il_out = levich(omega=omega, rpm=rpm, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu, F=F)
        arr_il = np.asarray(arr_il_out, dtype=float)
        il_is_scalar = arr_il.ndim == 0

    is_scalar = ik_is_scalar and il_is_scalar

    # Calculate combined current I = (I_K * I_L) / (I_K + I_L)
    # Handle infinite or zero cases robustly
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = arr_ik + arr_il
        # If denom is 0 (e.g. both are 0), result is 0
        i_tot = np.where(denom == 0.0, 0.0, (arr_ik * arr_il) / denom)
        # If I_K is infinite, I = I_L; if I_L is infinite, I = I_K
        i_tot = np.where(np.isinf(arr_ik), arr_il, i_tot)
        i_tot = np.where(np.isinf(arr_il), arr_ik, i_tot)

    return _format_output(i_tot, is_scalar)


@dataclass(frozen=True)
class KouteckyLevichResult:
    r"""Diagnostic regression results from Koutecký–Levich analysis.

    Parameters
    ----------
    i_k : float
        Extracted kinetic current :math:`I_K` in Amperes (A).
    k_rate : float
        Extracted apparent heterogeneous rate constant in :math:`\text{cm}\cdot\text{s}^{-1}` or :math:`\text{m}\cdot\text{s}^{-1}`.
    levich_constant : float
        Extracted experimental Levich constant :math:`B` in :math:`\text{A}\cdot\text{rad}^{-1/2}\cdot\text{s}^{1/2}`.
    d_estimated : float
        Diffusion coefficient estimated from the experimental slope :math:`B`.
    slope : float
        Regression slope :math:`m = 1 / B` of :math:`I^{-1}` vs. :math:`\omega^{-1/2}`.
    intercept : float
        Regression intercept :math:`q = 1 / I_K` of :math:`I^{-1}` vs. :math:`\omega^{-1/2}`.
    r_squared : float
        Coefficient of determination :math:`R^2` of the linear regression fit.
    """

    i_k: float
    k_rate: float
    levich_constant: float
    d_estimated: float
    slope: float
    intercept: float
    r_squared: float


def koutecky_levich_analysis(
    omega: float | ArrayLike | None = None,
    current: float | ArrayLike = 1.0,
    *,
    rpm: float | ArrayLike | None = None,
    area: float = 1.0,
    c_bulk: float = 1e-3,
    n: float = 1,
    nu: float = 0.01,
    F: float = FARADAY,
) -> KouteckyLevichResult:
    r"""Extract kinetic rate constant and Levich slope from experimental RDE series.

    Performs an ordinary least-squares linear regression on the linearized
    Koutecký–Levich equation:

    .. math::

        \frac{1}{I} = \frac{1}{I_K} + \frac{1}{B} \omega^{-1/2}

    Parameters
    ----------
    omega : array-like, optional
        Series of angular rotation velocities in :math:`\text{rad}\cdot\text{s}^{-1}` (:math:`\omega > 0`).
    current : array-like
        Series of measured currents :math:`I` in Amperes at the corresponding rotation speeds (:math:`I > 0`).
    rpm : array-like, optional
        Series of rotation rates in RPM. Either ``omega`` or ``rpm`` must be supplied.
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    c_bulk : float, default 1e-3
        Bulk reactant concentration (:math:`c^* > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    KouteckyLevichResult
        Regression analysis parameters including extracted :math:`I_K`, rate constant :math:`k`,
        Levich slope :math:`B`, estimated :math:`D`, and goodness of fit :math:`R^2`.

    Raises
    ------
    ValueError
        If fewer than 2 data points are provided, if arrays have mismatched lengths,
        or if non-positive currents/rotation rates are given.
    """
    _validate_hydrodynamic_params(n=n, D=1e-5, c_bulk=c_bulk, area=area, nu=nu, F=F)
    if c_bulk <= 0:
        raise ValueError(f"Bulk concentration c_bulk must be strictly positive (c_bulk > 0), got {c_bulk}.")

    w_arr, _ = _resolve_rotation(omega, rpm)
    i_arr = np.asarray(current, dtype=float)

    if w_arr.ndim == 0 or i_arr.ndim == 0 or len(w_arr) < 2 or len(i_arr) < 2:
        raise ValueError(
            f"Koutecky-Levich regression requires at least 2 data points; got shapes {w_arr.shape} and {i_arr.shape}."
        )
    if len(w_arr) != len(i_arr):
        raise ValueError(
            f"Length mismatch: rotation speeds has {len(w_arr)} points, but current has {len(i_arr)}."
        )
    if np.any(w_arr <= 0):
        raise ValueError("All rotation speeds in Koutecký–Levich analysis must be positive (omega > 0).")
    if np.any(i_arr <= 0):
        raise ValueError("All measured currents in Koutecký–Levich analysis must be positive (I > 0).")

    x_vals = 1.0 / np.sqrt(w_arr)
    y_vals = 1.0 / i_arr

    # Linear regression y = slope * x + intercept
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    # Goodness of fit R^2
    y_pred = slope * x_vals + intercept
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    ss_res = float(np.sum((y_vals - y_pred) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    i_k = float(1.0 / intercept) if intercept > 0 else float("inf")
    k_rate = float(i_k / (n * F * area * c_bulk)) if not math.isinf(i_k) else float("inf")
    b_levich = float(1.0 / slope) if slope > 0 else float("nan")

    # Estimate D from Levich constant B = 0.620 * n * F * A * D^(2/3) * nu^(-1/6) * c*
    # D^(2/3) = B / [0.620 * n * F * A * nu^(-1/6) * c*]
    prefactor = 0.620 * n * F * area * (nu ** (-1.0 / 6.0)) * c_bulk
    if b_levich > 0 and prefactor > 0:
        d_est = float((b_levich / prefactor) ** 1.5)
    else:
        d_est = float("nan")

    return KouteckyLevichResult(
        i_k=i_k,
        k_rate=k_rate,
        levich_constant=b_levich,
        d_estimated=d_est,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
    )


def _albery_f(theta: float | np.ndarray) -> float | np.ndarray:
    r"""Evaluate the Albery–Bruckenstein hydrodynamic integral function F(theta).

    .. math::

        F(\theta) = \frac{\sqrt{3}}{4\pi} \ln\left[\frac{(1 + \theta^{1/3})^3}{1 + \theta}\right]
        + \frac{3}{2\pi} \arctan\left(\frac{2\theta^{1/3} - 1}{\sqrt{3}}\right) + \frac{1}{4}

    where :math:`F(0) = 0` and :math:`F(\infty) = 1`.
    """
    arr = np.asarray(theta, dtype=float)
    is_scalar = arr.ndim == 0

    x = np.cbrt(arr)
    # Compute ((1+x)^3) / (1+x^3) = ((1+x)^2) / (1 - x + x^2)
    # Using the factored form avoids precision loss and handles 0 cleanly
    ratio = ((1.0 + x) ** 2.0) / (1.0 - x + x**2.0)
    term1 = (np.sqrt(3.0) / (4.0 * np.pi)) * np.log(ratio)
    term2 = (1.5 / np.pi) * np.arctan((2.0 * x - 1.0) / np.sqrt(3.0)) + 0.25

    res = term1 + term2
    return float(res.item()) if is_scalar else res


def _validate_rrde_radii(r1: float, r2: float, r3: float) -> None:
    """Validate geometric radii ordering for rotating ring-disk electrodes."""
    if r1 <= 0:
        raise ValueError(f"Disk radius r1 must be strictly positive (r1 > 0), got {r1}.")
    if r2 <= r1:
        raise ValueError(
            f"Ring inner radius r2 must be strictly greater than disk radius r1 (r2 > r1 = {r1}), got {r2}."
        )
    if r3 <= r2:
        raise ValueError(
            f"Ring outer radius r3 must be strictly greater than ring inner radius r2 (r3 > r2 = {r2}), got {r3}."
        )


def collection_efficiency(
    r1: float,
    r2: float,
    r3: float,
) -> float:
    r"""Compute theoretical collection efficiency N for a Rotating Ring-Disk Electrode (RRDE).

    The collection efficiency :math:`N` is the fraction of an electroactive
    intermediate generated at the central disk that reaches and reacts at the
    concentric ring electrode under steady-state convective diffusion.

    Evaluated using the exact closed-form solution derived by W. J. Albery and
    S. Bruckenstein (1966):

    .. math::

        N = 1 - F\left(\frac{\alpha}{\beta}\right) + \beta^{2/3}\left[1 - F(\alpha)\right]
        - (1 + \alpha + \beta)^{2/3}\left\{1 - F\left[\frac{\alpha}{\beta}(1 + \alpha + \beta)\right]\right\}

    where:

    .. math::

        \alpha = \left(\frac{r_2}{r_1}\right)^3 - 1, \qquad \beta = \left(\frac{r_3}{r_1}\right)^3 - \left(\frac{r_2}{r_1}\right)^3

    and :math:`F(\theta)` is the Albery hydrodynamic integral function.

    Parameters
    ----------
    r1 : float
        Radius of the central disk electrode (:math:`r_1 > 0`).
    r2 : float
        Inner radius of the concentric ring electrode (:math:`r_2 > r_1`).
    r3 : float
        Outer radius of the concentric ring electrode (:math:`r_3 > r_2`).

    Returns
    -------
    float
        Theoretical collection efficiency :math:`N \in (0, 1)`.

    Raises
    ------
    ValueError
        If radii do not strictly satisfy :math:`0 < r_1 < r_2 < r_3`.

    Examples
    --------
    >>> import softpotato as sp
    >>> # Standard Pine commercial tip (5.0 mm disk OD, 6.5 mm ring ID, 7.5 mm ring OD)
    >>> n_eff = sp.collection_efficiency(r1=0.25, r2=0.325, r3=0.375)
    >>> round(n_eff, 4)
    0.2555
    """
    _validate_rrde_radii(r1, r2, r3)

    r2_over_r1_cubed = (r2 / r1) ** 3.0
    r3_over_r1_cubed = (r3 / r1) ** 3.0

    alpha = r2_over_r1_cubed - 1.0
    beta = r3_over_r1_cubed - r2_over_r1_cubed

    arg1 = alpha / beta
    arg2 = alpha
    arg3 = (alpha / beta) * (1.0 + alpha + beta)

    f1 = float(_albery_f(arg1))  # type: ignore[arg-type]
    f2 = float(_albery_f(arg2))  # type: ignore[arg-type]
    f3 = float(_albery_f(arg3))  # type: ignore[arg-type]

    term1 = 1.0 - f1
    term2 = (beta ** (2.0 / 3.0)) * (1.0 - f2)
    term3 = ((1.0 + alpha + beta) ** (2.0 / 3.0)) * (1.0 - f3)

    n_val = term1 + term2 - term3
    return float(np.clip(n_val, 0.0, 1.0))


def ring_limiting_current(
    omega: float | ArrayLike | None = None,
    *,
    rpm: float | ArrayLike | None = None,
    r2: float,
    r3: float,
    r1: float | None = None,
    shielded: bool = False,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    nu: float = 0.01,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute convective mass-transport limiting current at an RRDE ring electrode.

    Calculates either the unshielded limiting ring current (when the disk is at
    open-circuit potential) or the shielded limiting ring current (when the disk
    is simultaneously consuming the bulk reactant at its limiting plateau):

    - **Unshielded Limiting Current** (:math:`I_{R, L}`):

      .. math::

          I_{R, L} = 0.620 \, n F \pi (r_3^3 - r_2^3)^{2/3} D^{2/3} \omega^{1/2} \nu^{-1/6} c^*

    - **Shielded Limiting Current** (:math:`I_{R, \text{sh}}`):

      .. math::

          I_{R, \text{sh}} = I_{R, L} - N I_{D, L}

    Parameters
    ----------
    omega : float or array-like, optional
        Angular rotation speed in :math:`\text{rad}\cdot\text{s}^{-1}`.
    rpm : float or array-like, optional
        Rotational frequency in revolutions per minute.
    r2 : float
        Inner radius of the ring electrode (:math:`r_2 > 0`).
    r3 : float
        Outer radius of the ring electrode (:math:`r_3 > r_2`).
    r1 : float, optional
        Radius of the central disk electrode (:math:`0 < r_1 < r_2`).
        Required if ``shielded=True``.
    shielded : bool, default False
        If True, returns the shielded ring limiting current accounting for disk depletion.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    float or numpy.ndarray
        Limiting ring current :math:`I_{R}` in Amperes (A).

    Raises
    ------
    ValueError
        If ``shielded=True`` but disk radius ``r1`` is not specified, or if radii are invalid.
    """
    _validate_hydrodynamic_params(n=n, D=D, c_bulk=c_bulk, area=1.0, nu=nu, F=F)
    if r2 <= 0:
        raise ValueError(f"Ring inner radius r2 must be positive (r2 > 0), got {r2}.")
    if r3 <= r2:
        raise ValueError(f"Ring outer radius r3 must be greater than r2 (r3 > r2 = {r2}), got {r3}.")

    w_arr, is_scalar = _resolve_rotation(omega, rpm)

    # Equivalent ring area factor: pi * (r3^3 - r2^3)^(2/3)
    ring_area_factor = np.pi * ((r3**3.0 - r2**3.0) ** (2.0 / 3.0))
    b_ring = 0.620 * n * F * ring_area_factor * (D ** (2.0 / 3.0)) * (nu ** (-1.0 / 6.0)) * c_bulk
    i_ring_unshielded = b_ring * np.sqrt(w_arr)

    if not shielded:
        return _format_output(i_ring_unshielded, is_scalar)

    if r1 is None:
        raise ValueError("Disk radius 'r1' must be provided to compute shielded ring current.")
    _validate_rrde_radii(r1, r2, r3)

    n_eff = collection_efficiency(r1, r2, r3)
    disk_area = np.pi * (r1**2.0)
    i_disk_lim = levich(omega=w_arr, n=n, D=D, c_bulk=c_bulk, area=disk_area, nu=nu, F=F)

    i_ring_shielded = i_ring_unshielded - (n_eff * np.asarray(i_disk_lim, dtype=float))
    return _format_output(i_ring_shielded, is_scalar)


def ring_collection_current(
    i_disk: float | ArrayLike,
    N: float | None = None,
    *,
    r1: float | None = None,
    r2: float | None = None,
    r3: float | None = None,
    n_ring: float = 1,
    n_disk: float = 1,
) -> float | np.ndarray:
    r"""Compute the ring current resulting from collection of species generated at the disk.

    .. math::

        I_R = -N \left(\frac{n_R}{n_D}\right) I_D

    where :math:`I_D` is the Faradaic current at the disk electrode, and :math:`N`
    is the RRDE collection efficiency.

    Parameters
    ----------
    i_disk : float or array-like
        Current measured at the disk electrode in Amperes (A).
    N : float, optional
        Collection efficiency (:math:`0 < N < 1`). If not provided, computed from
        geometric radii ``r1``, ``r2``, ``r3``.
    r1 : float, optional
        Radius of disk electrode.
    r2 : float, optional
        Inner radius of ring electrode.
    r3 : float, optional
        Outer radius of ring electrode.
    n_ring : float, default 1
        Number of electrons transferred in the ring reaction (:math:`n_R > 0`).
    n_disk : float, default 1
        Number of electrons transferred in the disk reaction (:math:`n_D > 0`).

    Returns
    -------
    float or numpy.ndarray
        Ring collection current :math:`I_R` in Amperes (A).
    """
    if n_ring <= 0 or n_disk <= 0:
        raise ValueError("Electron numbers n_ring and n_disk must be positive.")

    if N is None:
        if r1 is None or r2 is None or r3 is None:
            raise ValueError(
                "Either collection efficiency 'N' or electrode radii (r1, r2, r3) must be provided."
            )
        N = collection_efficiency(r1, r2, r3)
    elif not (0 < N <= 1):
        raise ValueError(f"Collection efficiency N must be in (0, 1], got {N}.")

    arr_id = np.asarray(i_disk, dtype=float)
    is_scalar = arr_id.ndim == 0

    i_ring = -N * (n_ring / n_disk) * arr_id
    return _format_output(i_ring, is_scalar)


def shielding_factor(
    r1: float,
    r2: float,
    r3: float,
    N: float | None = None,
) -> float:
    r"""Compute the theoretical geometric shielding factor S for an RRDE.

    The shielding factor :math:`S` characterizes the relative decrease in ring
    limiting current when the disk electrode is active:

    .. math::

        S = 1 - \frac{I_{R, \text{sh}}}{I_{R, L}} = N \left[\frac{r_1^3}{r_3^3 - r_2^3}\right]^{2/3}

    Parameters
    ----------
    r1 : float
        Radius of disk electrode (:math:`r_1 > 0`).
    r2 : float
        Inner radius of ring electrode (:math:`r_2 > r_1`).
    r3 : float
        Outer radius of ring electrode (:math:`r_3 > r_2`).
    N : float, optional
        Collection efficiency. If not provided, computed from ``r1``, ``r2``, ``r3``.

    Returns
    -------
    float
        Geometric shielding factor :math:`S \in (0, 1)`.
    """
    _validate_rrde_radii(r1, r2, r3)
    if N is None:
        N = collection_efficiency(r1, r2, r3)
    elif not (0 < N <= 1):
        raise ValueError(f"Collection efficiency N must be in (0, 1], got {N}.")

    geometric_ratio = (r1**3.0) / (r3**3.0 - r2**3.0)
    s_val = N * (geometric_ratio ** (2.0 / 3.0))
    return float(s_val)


@dataclass(frozen=True)
class RRDEResult:
    r"""Comprehensive parameters and limiting currents for an RRDE system.

    Parameters
    ----------
    r1 : float
        Disk radius.
    r2 : float
        Ring inner radius.
    r3 : float
        Ring outer radius.
    area_disk : float
        Geometric area of disk electrode: :math:`\pi r_1^2`.
    area_ring : float
        Geometric area of ring electrode: :math:`\pi (r_3^2 - r_2^2)`.
    N : float
        Theoretical collection efficiency.
    shielding_factor : float
        Theoretical geometric shielding factor.
    i_disk_lim : float or numpy.ndarray or None
        Limiting current at the disk electrode (A).
    i_ring_lim_unshielded : float or numpy.ndarray or None
        Unshielded limiting current at the ring electrode (A).
    i_ring_lim_shielded : float or numpy.ndarray or None
        Shielded limiting current at the ring electrode (A).
    """

    r1: float
    r2: float
    r3: float
    area_disk: float
    area_ring: float
    N: float
    shielding_factor: float
    i_disk_lim: float | np.ndarray | None
    i_ring_lim_unshielded: float | np.ndarray | None
    i_ring_lim_shielded: float | np.ndarray | None


def rotating_ring_disk(
    r1: float,
    r2: float,
    r3: float,
    *,
    omega: float | ArrayLike | None = None,
    rpm: float | ArrayLike | None = None,
    n_disk: float = 1,
    n_ring: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    nu: float = 0.01,
    F: float = FARADAY,
) -> RRDEResult:
    r"""Evaluate geometric parameters, collection efficiency, and limiting currents for an RRDE.

    Parameters
    ----------
    r1 : float
        Disk radius (:math:`r_1 > 0`).
    r2 : float
        Ring inner radius (:math:`r_2 > r_1`).
    r3 : float
        Ring outer radius (:math:`r_3 > r_2`).
    omega : float or array-like, optional
        Angular rotation speed in :math:`\text{rad}\cdot\text{s}^{-1}`.
    rpm : float or array-like, optional
        Rotational frequency in revolutions per minute.
    n_disk : float, default 1
        Number of electrons transferred at the disk (:math:`n_D > 0`).
    n_ring : float, default 1
        Number of electrons transferred at the ring (:math:`n_R > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    nu : float, default 0.01
        Kinematic viscosity (:math:`\nu > 0`).
    F : float, default FARADAY
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    RRDEResult
        Container dataclass with collection efficiency, shielding factor, areas,
        and (if rotation speed was supplied) disk and ring limiting currents.

    Examples
    --------
    >>> import softpotato as sp
    >>> res = sp.rotating_ring_disk(r1=0.25, r2=0.325, r3=0.375, rpm=1600, c_bulk=1e-6)
    >>> round(res.N, 4)
    0.2555
    >>> round(res.shielding_factor, 4)
    0.1691
    """
    _validate_rrde_radii(r1, r2, r3)

    area_d = np.pi * (r1**2.0)
    area_r = np.pi * (r3**2.0 - r2**2.0)

    n_eff = collection_efficiency(r1, r2, r3)
    sf = shielding_factor(r1, r2, r3, N=n_eff)

    i_d_lim: float | np.ndarray | None = None
    i_r_unshielded: float | np.ndarray | None = None
    i_r_shielded: float | np.ndarray | None = None

    if omega is not None or rpm is not None:
        _validate_hydrodynamic_params(n=n_disk, D=D, c_bulk=c_bulk, area=area_d, nu=nu, F=F)
        if n_ring <= 0:
            raise ValueError(f"Number of ring electrons must be positive (n_ring > 0), got {n_ring}.")

        w_arr, _ = _resolve_rotation(omega, rpm)

        i_d_lim = levich(omega=w_arr, n=n_disk, D=D, c_bulk=c_bulk, area=area_d, nu=nu, F=F)
        i_r_unshielded = ring_limiting_current(
            omega=w_arr,
            r2=r2,
            r3=r3,
            n=n_ring,
            D=D,
            c_bulk=c_bulk,
            nu=nu,
            F=F,
            shielded=False,
        )
        i_r_shielded = ring_limiting_current(
            omega=w_arr,
            r1=r1,
            r2=r2,
            r3=r3,
            n=n_ring,
            D=D,
            c_bulk=c_bulk,
            nu=nu,
            F=F,
            shielded=True,
        )

    return RRDEResult(
        r1=r1,
        r2=r2,
        r3=r3,
        area_disk=area_d,
        area_ring=area_r,
        N=n_eff,
        shielding_factor=sf,
        i_disk_lim=i_d_lim,
        i_ring_lim_unshielded=i_r_unshielded,
        i_ring_lim_shielded=i_r_shielded,
    )


__all__ = [
    "KouteckyLevichResult",
    "RRDEResult",
    "collection_efficiency",
    "koutecky_levich",
    "koutecky_levich_analysis",
    "levich",
    "levich_constant",
    "nernst_diffusion_layer",
    "rad_s_to_rpm",
    "ring_collection_current",
    "ring_limiting_current",
    "rotating_ring_disk",
    "rpm_to_rad_s",
    "shielding_factor",
]
