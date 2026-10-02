r"""Analytical equations for Scanning Electrochemical Microscopy (SECM) approach curves.

This module provides closed-form analytical approximations for steady-state feedback
approach curves at an inlaid circular microdisk ultramicroelectrode (UME) tip
surrounded by an insulating sheath (insulator-to-electrode radius ratio :math:`RG = r_g / a`).

The analytical approximations were developed by Christine Lefrou and Renaud Cornut
and describe:

- Pure positive feedback over conductive substrates (diffusion-controlled mediator regeneration).
- Pure negative feedback over insulating substrates (hindered diffusion).
- Approach curves over substrates with finite/irreversible first-order redox kinetics
  (:math:`\kappa = \Lambda = k a / D`), seamlessly interpolating between negative and
  positive feedback.
- Steady-state diffusion-limited current at infinite distance (:math:`i_{T,\infty}`) and
  dimensional tip current (:math:`i_T`) in Amperes.

Functions
---------
secm_approach_positive_feedback
    Normalized steady-state tip current over conductive substrates (positive feedback).
secm_approach_negative_feedback
    Normalized steady-state tip current over insulating substrates (negative feedback).
secm_approach_curve
    Normalized steady-state tip current for arbitrary substrate kinetics (:math:`\kappa \ge 0`).
secm_limiting_current_infinite
    Saito steady-state diffusion-limited current at infinite distance (:math:`i_{T,\infty}`).
secm_tip_current
    Dimensional steady-state tip current in Amperes (:math:`i_T = I_T \cdot i_{T,\infty}`).

Conventions & Units
-------------------
- **Normalized distance**: :math:`L = d / a` (:math:`L > 0`), where :math:`d` is the tip-substrate
  separation distance and :math:`a` is the electroactive microdisk radius.
- **Insulator ratio**: :math:`RG = r_g / a` (:math:`RG > 1`), where :math:`r_g` is the total radius
  including the surrounding glass insulator sheath.
- **Normalized current**: :math:`I_T = i_T / i_{T,\infty}`, where :math:`i_{T,\infty} = 4 n F D c^* a`
  is the steady-state diffusion-limited current in bulk solution (:math:`L \to \infty`).
- **Substrate kinetics parameter**: :math:`\kappa = \Lambda = \frac{k a}{D}` (:math:`\kappa \ge 0`),
  where :math:`k` is the heterogeneous first-order rate constant at the substrate, :math:`a` is the
  microdisk radius, and :math:`D` is the diffusion coefficient of the mediator.

Standard IUPAC sign conventions are applied for dimensional current:
- Cathodic (reduction) currents are negative (:math:`I < 0`).
- Anodic (oxidation) currents are positive (:math:`I > 0`).

References
----------
- R. Cornut, C. Lefrou, "New analytical approximation of feedback approach curves
  with a microdisk SECM tip and irreversible kinetic reaction at the substrate",
  *J. Electroanal. Chem.*, 621 (2008) 178–184.
- R. Cornut, C. Lefrou, "A unified new analytical approximation for negative
  feedback currents with a microdisk SECM tip", *J. Electroanal. Chem.*, 608 (2007) 59–66.
- C. Lefrou, "A unified new analytical approximation for positive feedback
  currents with a microdisk SECM tip", *J. Electroanal. Chem.*, 592 (2006) 103–112.
- C. Lefrou, R. Cornut, "Analytical Expressions for Quantitative Scanning
  Electrochemical Microscopy (SECM)", *ChemPhysChem*, 11 (2010) 547–556.
- Y. Saito, "A theoretical study on the diffusion current at the stationary
  spherical and disc electrodes", *Rev. Polarogr.*, 15 (1968) 177–187.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from softpotato.constants import FARADAY

_LN2: float = math.log(2.0)


def _validate_positive(val: float | ArrayLike, name: str) -> None:
    """Validate that value(s) are strictly positive (> 0)."""
    if np.any(np.asarray(val) <= 0):
        raise ValueError(f"{name} must be strictly positive ({name} > 0), got {val!r}.")


def _validate_non_negative(val: float | ArrayLike, name: str) -> None:
    """Validate that value(s) are non-negative (>= 0)."""
    if np.any(np.asarray(val) < 0):
        raise ValueError(f"{name} must be non-negative ({name} >= 0), got {val!r}.")


def _validate_rg(rg: float | ArrayLike) -> None:
    """Validate that insulator radius ratio RG is strictly greater than 1."""
    if np.any(np.asarray(rg) <= 1.0):
        raise ValueError(
            f"Insulator ratio RG = r_g / a must be strictly greater than 1 (RG > 1), got {rg!r}."
        )


def _validate_common_params(
    n: float,
    D: float,
    c_bulk: float,
    F: float,
) -> None:
    """Validate physical parameters for dimensional current calculations."""
    if n <= 0:
        raise ValueError(f"Number of electrons n must be positive (n > 0), got {n}.")
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive (D > 0), got {D}.")
    if c_bulk < 0:
        raise ValueError(
            f"Bulk concentration c_bulk must be non-negative (c_bulk >= 0), got {c_bulk}."
        )
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


def _compute_alpha_beta(
    rg_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute geometric intermediate parameters :math:`\alpha(RG)` and :math:`\beta(RG)`.

    Formulas from Cornut & Lefrou (2008), Eqs. 18b–18c:

    .. math::

        \theta(RG) = \frac{2}{\pi} \arccos\left(\frac{1}{RG}\right)

        \alpha(RG) = \ln(2) + \ln(2)(1 - \theta) - \ln(2)(1 - \theta^2)

        \beta(RG) = 1 + 0.639(1 - \theta) - 0.186(1 - \theta^2)
    """
    inv_rg = 1.0 / rg_arr
    theta = (2.0 / np.pi) * np.arccos(inv_rg)
    one_minus_theta = 1.0 - theta
    one_minus_theta_sq = 1.0 - theta * theta

    alpha = _LN2 + _LN2 * one_minus_theta - _LN2 * one_minus_theta_sq
    beta = 1.0 + 0.639 * one_minus_theta - 0.186 * one_minus_theta_sq

    return alpha, beta


def secm_approach_positive_feedback(
    L: float | ArrayLike,
    RG: float | ArrayLike = 10.0,
) -> float | np.ndarray:
    r"""Compute normalized steady-state tip current for positive feedback SECM.

    Calculates the normalized steady-state tip current :math:`I_T^{\text{cond}}(L, RG)`
    at an inlaid microdisk ultramicroelectrode approaching a conductive substrate under
    pure diffusion-controlled mediator regeneration, using the analytical approximation
    of Cornut and Lefrou (2008):

    .. math::

        I_T^{\text{cond}}(L, RG) = \alpha(RG) + \frac{\pi}{4 \beta(RG) \arctan(L)}
        + \left(1 - \alpha(RG) - \frac{1}{2\beta(RG)}\right) \frac{2}{\pi} \arctan(L)

    where :math:`\alpha(RG)` and :math:`\beta(RG)` are geometric coefficients defined by:

    .. math::

        \theta(RG) &= \frac{2}{\pi} \arccos\left(\frac{1}{RG}\right) \\
        \alpha(RG) &= \ln(2) + \ln(2)(1 - \theta) - \ln(2)(1 - \theta^2) \\
        \beta(RG) &= 1 + 0.639(1 - \theta) - 0.186(1 - \theta^2)

    As :math:`L \to \infty`, :math:`I_T^{\text{cond}} \to 1.0` exactly for all :math:`RG`.
    As :math:`L \to 0`, :math:`I_T^{\text{cond}} \sim \frac{\pi}{4 \beta(RG) L} \to \infty`.

    Parameters
    ----------
    L : float or array-like
        Normalized tip-to-substrate distance :math:`L = d / a` (:math:`L > 0`).
    RG : float or array-like, default 10.0
        Insulator radius ratio :math:`RG = r_g / a` (:math:`RG > 1`).

    Returns
    ----------
    float or np.ndarray
        Normalized tip current :math:`I_T = i_T / i_{T,\infty}` (dimensionless, :math:`I_T > 1`).

    Raises
    ------
    ValueError
        If any element of ``L`` is :math:`\le 0` or ``RG`` is :math:`\le 1`.

    Examples
    --------
    >>> from softpotato.analytical.secm import secm_approach_positive_feedback
    >>> # Normalized current at L = 1.0, RG = 10.0
    >>> round(float(secm_approach_positive_feedback(1.0, RG=10.0)), 4)
    1.5628
    >>> # Asymptotic approach to bulk current at large L
    >>> round(float(secm_approach_positive_feedback(100.0, RG=10.0)), 4)
    1.0041
    """
    L_arr = np.asarray(L, dtype=float)
    RG_arr = np.asarray(RG, dtype=float)

    is_scalar = (L_arr.ndim == 0) and (RG_arr.ndim == 0)

    _validate_positive(L_arr, "Normalized distance L")
    _validate_rg(RG_arr)

    L_bc, RG_bc = np.broadcast_arrays(L_arr, RG_arr)

    alpha, beta = _compute_alpha_beta(RG_bc)
    atan_L = np.arctan(L_bc)

    # I_T^cond = alpha + pi / (4 * beta * arctan(L)) + (1 - alpha - 1 / (2 * beta)) * (2 / pi) * arctan(L)
    term1 = alpha
    term2 = np.pi / (4.0 * beta * atan_L)
    term3 = (1.0 - alpha - 0.5 / beta) * (2.0 / np.pi) * atan_L

    res = term1 + term2 + term3
    return _format_output(res, is_scalar)


def secm_approach_negative_feedback(
    L: float | ArrayLike,
    RG: float | ArrayLike = 10.0,
) -> float | np.ndarray:
    r"""Compute normalized steady-state tip current for negative feedback SECM.

    Calculates the normalized steady-state tip current :math:`I_T^{\text{ins}}(L, RG)`
    at an inlaid microdisk ultramicroelectrode approaching an inert insulating substrate
    under hindered diffusion, using the analytical approximation of Cornut and Lefrou (2007, 2008):

    .. math::

        I_T^{\text{ins}}(L, RG) = \frac{\frac{2.08}{RG^{0.358}} \left(L - \frac{0.145}{RG}\right) + 1.585}
        {\frac{2.08}{RG^{0.358}} (L + 0.0023 RG) + 1.57 + \frac{\ln(RG)}{L} + \frac{2}{\pi RG} \ln\left(1 + \frac{\pi RG}{2 L}\right)}

    As :math:`L \to \infty`, :math:`I_T^{\text{ins}} \to 1.0` for all :math:`RG`.
    As :math:`L \to 0`, :math:`I_T^{\text{ins}} \to 0.0`.

    Parameters
    ----------
    L : float or array-like
        Normalized tip-to-substrate distance :math:`L = d / a` (:math:`L > 0`).
    RG : float or array-like, default 10.0
        Insulator radius ratio :math:`RG = r_g / a` (:math:`RG > 1`).

    Returns
    ----------
    float or np.ndarray
        Normalized tip current :math:`I_T = i_T / i_{T,\infty}` (dimensionless, :math:`0 < I_T < 1`).

    Raises
    ------
    ValueError
        If any element of ``L`` is :math:`\le 0` or ``RG`` is :math:`\le 1`.

    Examples
    --------
    >>> from softpotato.analytical.secm import secm_approach_negative_feedback
    >>> # Normalized current at L = 1.0, RG = 10.0
    >>> round(float(secm_approach_negative_feedback(1.0, RG=10.0)), 4)
    0.4983
    >>> # Strong hindered diffusion near substrate (L = 0.05)
    >>> round(float(secm_approach_negative_feedback(0.05, RG=10.0)), 4)
    0.0337
    """
    L_arr = np.asarray(L, dtype=float)
    RG_arr = np.asarray(RG, dtype=float)

    is_scalar = (L_arr.ndim == 0) and (RG_arr.ndim == 0)

    _validate_positive(L_arr, "Normalized distance L")
    _validate_rg(RG_arr)

    L_bc, RG_bc = np.broadcast_arrays(L_arr, RG_arr)

    c_rg = 2.08 / (RG_bc**0.358)
    num = c_rg * (L_bc - 0.145 / RG_bc) + 1.585
    den = (
        c_rg * (L_bc + 0.0023 * RG_bc)
        + 1.57
        + np.log(RG_bc) / L_bc
        + (2.0 / (np.pi * RG_bc)) * np.log(1.0 + (np.pi * RG_bc) / (2.0 * L_bc))
    )

    res = num / den
    return _format_output(res, is_scalar)


def secm_approach_curve(
    L: float | ArrayLike,
    RG: float | ArrayLike = 10.0,
    kappa: float | ArrayLike = np.inf,
) -> float | np.ndarray:
    r"""Compute normalized steady-state tip current for arbitrary substrate kinetics.

    Calculates the normalized steady-state tip current :math:`I_T(L, RG, \kappa)`
    at an inlaid microdisk ultramicroelectrode for first-order substrate reaction kinetics
    characterized by the dimensionless kinetic parameter :math:`\kappa = \Lambda = \frac{k a}{D}`,
    using the unified analytical approximation of Cornut and Lefrou (2008, Eq. 22):

    .. math::

        I_T(L, RG, \kappa) = I_T^{\text{cond}}\left(L + \frac{1}{\kappa}, RG\right)
        + \frac{I_T^{\text{ins}}(L, RG) - 1}{\left(1 + 2.47 RG^{0.31} L \kappa\right)
        \left(1 + L^{0.006 RG + 0.113} \kappa^{0.0236 RG + 0.91}\right)}

    Special limiting cases:

    - Pure positive feedback (:math:`\kappa \to \infty`): reduces to
      :func:`secm_approach_positive_feedback` (:math:`I_T^{\text{cond}}`).
    - Pure negative feedback (:math:`\kappa = 0`): reduces to
      :func:`secm_approach_negative_feedback` (:math:`I_T^{\text{ins}}`).
    - Finite substrate kinetics (:math:`0 < \kappa < \infty`): smoothly interpolates
      between the negative and positive feedback curves.

    Parameters
    ----------
    L : float or array-like
        Normalized tip-to-substrate distance :math:`L = d / a` (:math:`L > 0`).
    RG : float or array-like, default 10.0
        Insulator radius ratio :math:`RG = r_g / a` (:math:`RG > 1`).
    kappa : float or array-like, default np.inf
        Dimensionless heterogeneous first-order substrate rate constant
        :math:`\kappa = \Lambda = \frac{k a}{D}` (:math:`\kappa \ge 0`).
        Set to ``np.inf`` for diffusion-controlled positive feedback, and ``0.0``
        for an inert insulating substrate.

    Returns
    ----------
    float or np.ndarray
        Normalized tip current :math:`I_T = i_T / i_{T,\infty}` (dimensionless).

    Raises
    ------
    ValueError
        If any element of ``L`` is :math:`\le 0`, ``RG`` is :math:`\le 1`, or ``kappa`` is :math:`< 0`.

    Examples
    --------
    >>> from softpotato.analytical.secm import secm_approach_curve
    >>> import numpy as np
    >>> # Positive feedback (kappa = inf)
    >>> round(float(secm_approach_curve(1.0, RG=10.0, kappa=np.inf)), 4)
    1.5628
    >>> # Negative feedback (kappa = 0)
    >>> round(float(secm_approach_curve(1.0, RG=10.0, kappa=0.0)), 4)
    0.4983
    >>> # Intermediate substrate kinetics (kappa = 1.0)
    >>> round(float(secm_approach_curve(1.0, RG=10.0, kappa=1.0)), 4)
    1.2064
    """
    L_arr = np.asarray(L, dtype=float)
    RG_arr = np.asarray(RG, dtype=float)
    kappa_arr = np.asarray(kappa, dtype=float)

    is_scalar = (L_arr.ndim == 0) and (RG_arr.ndim == 0) and (kappa_arr.ndim == 0)

    _validate_positive(L_arr, "Normalized distance L")
    _validate_rg(RG_arr)
    _validate_non_negative(kappa_arr, "Substrate kinetics parameter kappa")

    L_bc, RG_bc, k_bc = np.broadcast_arrays(L_arr, RG_arr, kappa_arr)

    # Base feedback bounds
    i_ins = secm_approach_negative_feedback(L_bc, RG=RG_bc)
    i_cond = secm_approach_positive_feedback(L_bc, RG=RG_bc)

    # Initialize output array matching broadcast shape
    res = np.zeros_like(L_bc, dtype=float)

    # Mask regimes: pure positive, pure negative, finite kinetics
    is_pos = np.isinf(k_bc)
    is_neg = k_bc == 0.0
    is_finite = ~is_pos & ~is_neg

    if np.any(is_pos):
        res[is_pos] = np.asarray(i_cond)[is_pos]

    if np.any(is_neg):
        res[is_neg] = np.asarray(i_ins)[is_neg]

    if np.any(is_finite):
        l_fin = L_bc[is_finite]
        rg_fin = RG_bc[is_finite]
        k_fin = k_bc[is_finite]
        ins_fin = np.asarray(i_ins)[is_finite]

        # Effective distance for conductive part: L + 1 / kappa
        l_eff = l_fin + 1.0 / k_fin
        cond_eff = secm_approach_positive_feedback(l_eff, RG=rg_fin)

        term_den1 = 1.0 + 2.47 * (rg_fin**0.31) * l_fin * k_fin
        pow_l = 0.006 * rg_fin + 0.113
        pow_k = 0.0236 * rg_fin + 0.91
        term_den2 = 1.0 + (l_fin**pow_l) * (k_fin**pow_k)

        kinetic_correction = (ins_fin - 1.0) / (term_den1 * term_den2)
        res[is_finite] = cond_eff + kinetic_correction

    return _format_output(res, is_scalar)


def secm_limiting_current_infinite(
    radius: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute steady-state diffusion-limited current at infinite distance from substrate.

    Calculates the Saito steady-state diffusion-limited Faradaic current at an inlaid
    circular microdisk electrode of radius :math:`a` in bulk solution (:math:`L \to \infty`):

    .. math::

        i_{T,\infty} = 4 n F D c^* a

    Parameters
    ----------
    radius : float or array-like
        Radius of the microdisk tip :math:`a` in meters or cm (:math:`a > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of mediator :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    ----------
    float or np.ndarray
        Steady-state limiting current :math:`i_{T,\infty}` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``radius``, ``D``, ``n``, or ``F`` is :math:`\le 0`, or ``c_bulk`` is :math:`< 0`.
    """
    _validate_common_params(n=n, D=D, c_bulk=c_bulk, F=F)
    _validate_positive(radius, "Electrode radius")

    r_arr = np.asarray(radius, dtype=float)
    is_scalar = r_arr.ndim == 0

    i_inf = 4.0 * n * F * D * c_bulk * r_arr
    return _format_output(i_inf, is_scalar)


def secm_tip_current(
    L: float | ArrayLike,
    RG: float | ArrayLike = 10.0,
    kappa: float | ArrayLike = np.inf,
    radius: float = 1e-5,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
    reduction: bool = True,
) -> float | np.ndarray:
    r"""Compute dimensional steady-state tip current in Amperes.

    Scales the normalized SECM approach curve :math:`I_T(L, RG, \kappa)` by the Saito
    steady-state limiting current at infinite distance :math:`i_{T,\infty} = 4 n F D c^* a`:

    .. math::

        i_T = I_T(L, RG, \kappa) \cdot 4 n F D c^* a

    Parameters
    ----------
    L : float or array-like
        Normalized tip-to-substrate distance :math:`L = d / a` (:math:`L > 0`).
    RG : float or array-like, default 10.0
        Insulator radius ratio :math:`RG = r_g / a` (:math:`RG > 1`).
    kappa : float or array-like, default np.inf
        Dimensionless heterogeneous first-order substrate rate constant
        :math:`\kappa = \Lambda = \frac{k a}{D}` (:math:`\kappa \ge 0`).
    radius : float, default 1e-5
        Active microdisk radius :math:`a` in meters or cm (:math:`a > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of mediator :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).
    reduction : bool, default True
        Direction of reaction. Cathodic reduction (True) yields negative dimensional tip current
        (:math:`i_T < 0`), while anodic oxidation (False) yields positive current (:math:`i_T > 0`).

    Returns
    ----------
    float or np.ndarray
        Dimensional tip current :math:`i_T` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If any input parameter violates physical bounds.
    """
    i_inf = secm_limiting_current_infinite(radius=radius, n=n, D=D, c_bulk=c_bulk, F=F)
    i_norm = secm_approach_curve(L=L, RG=RG, kappa=kappa)

    sign = -1.0 if reduction else 1.0
    res = i_norm * i_inf * sign

    if isinstance(res, np.ndarray) and res.ndim == 0:
        return float(res.item())
    return res
