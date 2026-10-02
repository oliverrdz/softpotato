"""Analytical and empirical equations for electrochemical step techniques.

This module provides closed-form solutions, asymptotic expansions, and empirical
models for transient potential-step (chronoamperometry, chronocoulometry) and
current-step (chronopotentiometry) electrochemical experiments under semi-infinite
diffusion at planar, spherical, and cylindrical electrode geometries.

Functions
---------
cottrell
    Planar Cottrell equation for chronoamperometric current transients.
cottrell_spherical
    Cottrell equation for spherical and hemispherical electrodes.
cottrell_step
    Double potential step chronoamperometry reversal transient.
anson
    Anson equation for chronocoulometric cumulative charge transients.
sand_transition_time
    Sand equation for chronopotentiometric transition time.
sand
    Surface concentration transient under constant-current step.
sand_potential
    Potential-time transient for a reversible chronopotentiometric wave.
cottrell_cylinder
    Chronoamperometric current transient for cylindrical wire/fiber electrodes.
step_concentration_profile
    Exact spatial concentration distribution :math:`c(x, t)` for planar potential step.
step_flux_profile
    Exact spatial diffusion flux distribution :math:`J(x, t)` for planar potential step.

Conventions & Units
-------------------
All equations are dimensionally consistent in either standard SI or standard
electrochemical CGS units:

- **SI**: :math:`t` in :math:`\\text{s}`, :math:`D` in :math:`\\text{m}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{m}^{-3}`, :math:`A` in :math:`\\text{m}^2`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}` :math:`\\implies I` in :math:`\\text{A}`.
- **CGS**: :math:`t` in :math:`\\text{s}`, :math:`D` in :math:`\\text{cm}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{cm}^{-3}`, :math:`A` in :math:`\\text{cm}^2`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}` :math:`\\implies I` in :math:`\\text{A}`.

Standard IUPAC sign conventions are applied:
- Cathodic (reduction) currents are negative (:math:`I < 0`).
- Anodic (oxidation) currents are positive (:math:`I > 0`).

References
----------
- A. J. Bard, L. R. Faulkner, H. S. White, *Electrochemical Methods:
  Fundamentals and Applications*, 3rd ed., John Wiley & Sons, 2022.
- K. Aoki, K. Honda, K. Tokuda, H. Matsuda, "Voltammetry at microcylinder
  electrodes: Part II. Chronoamperometry", *J. Electroanal. Chem.*, 186 (1985) 79–86.
- K. B. Oldham, "Analytical expressions for the transient current at a
  microcylinder electrode", *J. Electroanal. Chem.*, 224 (1987) 229–232.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf

from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

_EULER_GAMMA: float = 0.5772156649015329


def _validate_common_params(
    n: float,
    D: float,
    c_bulk: float,
    area: float,
    F: float,
) -> None:
    """Validate standard physical parameters."""
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
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


def cottrell(
    t: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute planar Cottrell chronoamperometric current transient.

    Calculates the Faradaic diffusion-controlled current response at a planar electrode
    subject to an instantaneous potential step into the diffusion-limited regime
    (:math:`c(0, t) = 0`):

    .. math::

        I(t) = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}}

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the potential step in seconds (:math:`t \ge 0`).
    n : int or float, default 1
        Number of electrons transferred per ion/molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient of the electroactive species in :math:`\text{m}^2/\text{s}`
        or :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of the electroactive species in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    area : float, default 1.0
        Electrode surface area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F \approx 96485.332\,\text{C}/\text{mol}`).

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`). At :math:`t = 0`,
        returns ``np.inf``.

    Raises
    ------
    ValueError
        If any element of ``t`` is negative, or if physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import cottrell
    >>> # Cottrell current at t = 1.0 s
    >>> current = cottrell(1.0, n=1, D=1e-5, c_bulk=1e-3, area=1e-4)
    >>> round(current * 1e6, 3)  # current in microamperes
    17.214
    >>> # At t = 0, current approaches infinity
    >>> cottrell(0.0)
    inf
    """
    _validate_common_params(n, D, c_bulk, area, F)
    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    with np.errstate(divide="ignore"):
        inv_sqrt_pi_t = np.where(t_arr == 0.0, np.inf, 1.0 / np.sqrt(np.pi * t_arr))

    i_val = n * F * area * math.sqrt(D) * c_bulk * inv_sqrt_pi_t
    return _format_output(i_val, is_scalar)


def cottrell_spherical(
    t: float | ArrayLike,
    r: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float | None = None,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute Cottrell chronoamperometric current at a spherical electrode.

    Calculates current at a spherical electrode, hanging mercury drop, or spherical
    microelectrode of radius :math:`r`:

    .. math::

        I(t) = n F A D c^* \left(\frac{1}{\sqrt{\pi D t}} + \frac{1}{r}\right)
             = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}} + \frac{n F A D c^*}{r}

    If ``area is None``, defaults to the surface area of a complete sphere
    (:math:`A = 4 \pi r^2`), yielding:

    .. math::

        I(t) = 4 \pi n F D c^* r \left(1 + \frac{r}{\sqrt{\pi D t}}\right)

    As :math:`t \to \infty`, the current reaches the steady-state limiting current
    :math:`I_{\text{ss}} = 4 \pi n F D c^* r`. As :math:`t \to 0` (or :math:`r \to \infty`),
    planar Cottrell behavior is recovered.

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the potential step in seconds (:math:`t \ge 0`).
    r : float
        Radius of the spherical electrode (:math:`r > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    area : float or None, default None
        Electrode area. If None, defaults to full sphere area :math:`4 \pi r^2`.
        For a hemisphere on an insulating plane, set ``area = 2 * np.pi * r**2``.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``r <= 0``, ``t < 0``, or physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import cottrell_spherical
    >>> # Steady-state limiting current at long time (r = 10 um)
    >>> r = 1e-5
    >>> i_steady = cottrell_spherical(1e6, r=r, n=1, D=1e-9, c_bulk=1.0)
    >>> expected_steady = 4 * np.pi * 96485.332 * 1e-9 * 1.0 * r
    >>> bool(np.isclose(i_steady, expected_steady, rtol=1e-4))
    True

    """
    if r <= 0:
        raise ValueError(f"Electrode radius r must be positive (r > 0), got {r}.")

    area_val = 4.0 * math.pi * (r**2) if area is None else float(area)
    _validate_common_params(n, D, c_bulk, area_val, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    with np.errstate(divide="ignore"):
        term_transient = np.where(
            t_arr == 0.0, np.inf, 1.0 / np.sqrt(np.pi * D * t_arr)
        )

    i_val = n * F * area_val * D * c_bulk * (term_transient + 1.0 / r)
    return _format_output(i_val, is_scalar)


def cottrell_step(
    t: float | ArrayLike,
    tau: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    D_red: float | None = None,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute double potential step chronoamperometry reversal transient.

    Calculates current during a double-step potential experiment for a reversible
    redox couple :math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`.

    - **Forward step** (:math:`0 < t \le \tau`): Potential is stepped to a value where
      :math:`\text{Ox}` is reduced at the diffusion-controlled rate (:math:`c_{\text{Ox}}(0, t) = 0`),
      yielding negative cathodic current:

      .. math::

          I_f(t) = -\frac{n F A \sqrt{D_{\text{Ox}}} c^*_{\text{Ox}}}{\sqrt{\pi t}}

    - **Reversal step** (:math:`t > \tau`): Potential is stepped to re-oxidize
      :math:`\text{Red}` back to :math:`\text{Ox}` at the diffusion-controlled rate
      (:math:`c_{\text{Red}}(0, t) = 0`), yielding positive anodic current:

      .. math::

          I_r(t) = \frac{n F A \sqrt{D_{\text{Red}}} c^*_{\text{Ox}}}{\sqrt{\pi}}
                   \left[ \frac{1}{\sqrt{t - \tau}} - \frac{1}{\sqrt{t}} \right]

    When :math:`D_{\text{Ox}} = D_{\text{Red}} = D`, the reversal current satisfies the
    classic diagnostic ratio:

    .. math::

        \frac{-I_r(t)}{I_f(t - \tau)} = 1 - \sqrt{\frac{t - \tau}{t}}

    At :math:`t = 2\tau`, :math:`-I(2\tau) / I(\tau) = 1 - 1/\sqrt{2} \approx 0.292893`.

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the initial step in seconds (:math:`t \ge 0`).
    tau : float
        Forward pulse duration in seconds (:math:`\tau > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient of reactant :math:`\text{Ox}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of reactant :math:`\text{Ox}` (:math:`c^*_{\text{Ox}} \ge 0`).
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    D_red : float or None, default None
        Diffusion coefficient of electrogenerated product :math:`\text{Red}`.
        If None, assumes :math:`D_{\text{Red}} = D_{\text{Ox}} = D`.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`). Negative for
        cathodic reduction (:math:`t \le \tau`) and positive for anodic oxidation (:math:`t > \tau`).

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``t < 0``, or physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import cottrell_step
    >>> tau = 1.0
    >>> # Evaluate forward and reversal currents
    >>> i_fwd = cottrell_step(tau, tau=tau)
    >>> i_rev = cottrell_step(2.0 * tau, tau=tau)
    >>> ratio = -i_rev / i_fwd
    >>> round(ratio, 6)
    0.292893
    """
    if tau <= 0:
        raise ValueError(f"Step duration tau must be positive (tau > 0), got {tau}.")

    d_red_val = D if D_red is None else float(D_red)
    _validate_common_params(n, D, c_bulk, area, F)
    if d_red_val <= 0:
        raise ValueError(
            f"Product diffusion coefficient D_red must be positive (D_red > 0), got {d_red_val}."
        )

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    result = np.empty_like(t_arr, dtype=float)
    fwd_mask = t_arr <= tau
    rev_mask = ~fwd_mask

    coeff_fwd = n * F * area * math.sqrt(D) * c_bulk / math.sqrt(math.pi)
    coeff_rev = n * F * area * math.sqrt(d_red_val) * c_bulk / math.sqrt(math.pi)

    with np.errstate(divide="ignore", invalid="ignore"):
        if np.any(fwd_mask):
            t_fwd = t_arr[fwd_mask]
            inv_sqrt_t = np.where(t_fwd == 0.0, np.inf, 1.0 / np.sqrt(t_fwd))
            result[fwd_mask] = -coeff_fwd * inv_sqrt_t

        if np.any(rev_mask):
            t_rev = t_arr[rev_mask]
            dt_rev = t_rev - tau
            term1 = np.where(dt_rev == 0.0, np.inf, 1.0 / np.sqrt(dt_rev))
            term2 = 1.0 / np.sqrt(t_rev)
            result[rev_mask] = coeff_rev * (term1 - term2)

    return _format_output(result, is_scalar)


def anson(
    t: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    Q_dl: float = 0.0,
    gamma: float = 0.0,
    tau: float | None = None,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute cumulative charge transient using the Anson chronocoulometry equation.

    For single potential-step chronocoulometry:

    .. math::

        Q(t) = Q_{\text{diff}}(t) + Q_{\text{dl}} + Q_{\text{ads}}
             = \frac{2 n F A \sqrt{D} c^* \sqrt{t}}{\sqrt{\pi}} + Q_{\text{dl}} + n F A \Gamma

    where:

    - :math:`Q_{\text{diff}}(t)` is the Faradaic charge from diffusing species.
    - :math:`Q_{\text{dl}}` is the double-layer capacitive charge.
    - :math:`Q_{\text{ads}} = n F A \Gamma` is the Faradaic charge from adsorbed species
      with surface excess :math:`\Gamma`.

    An **Anson plot** of :math:`Q(t)` versus :math:`\sqrt{t}` yields:

    - Slope: :math:`\dfrac{2 n F A \sqrt{D} c^*}{\sqrt{\pi}}`
    - Intercept (:math:`t = 0`): :math:`Q_{\text{dl}} + n F A \Gamma`

    When ``tau`` is provided (double-step chronocoulometry), the cumulative charge
    for :math:`t > \tau` accounts for re-oxidation:

    .. math::

        Q(t > \tau) = \frac{2 n F A \sqrt{D} c^*}{\sqrt{\pi}} \left[ \sqrt{t} - \sqrt{t - \tau} \right]
                      + Q_{\text{dl}} + n F A \Gamma

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the step in seconds (:math:`t \ge 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    Q_dl : float, default 0.0
        Double-layer capacitive charge in Coulombs (:math:`\text{C}`).
    gamma : float, default 0.0
        Surface excess of adsorbed reactant :math:`\Gamma` in :math:`\text{mol}/\text{m}^2`
        or :math:`\text{mol}/\text{cm}^2` (:math:`\Gamma \ge 0`).
    tau : float or None, default None
        Forward pulse duration in seconds (:math:`\tau > 0`). If None, single-step
        chronocoulometry is evaluated.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float or np.ndarray
        Cumulative charge :math:`Q(t)` in Coulombs (:math:`\text{C}`).

    Raises
    ------
    ValueError
        If ``t < 0``, ``gamma < 0``, ``tau <= 0``, or physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import anson
    >>> # Intercept at t = 0 matches Q_dl + n*F*A*gamma
    >>> q_0 = anson(0.0, Q_dl=5e-6, gamma=1e-10, area=1e-4)
    >>> round(q_0 * 1e6, 6)
    5.000965
    """
    _validate_common_params(n, D, c_bulk, area, F)
    if gamma < 0:
        raise ValueError(
            f"Surface excess gamma must be non-negative (gamma >= 0), got {gamma}."
        )
    if tau is not None and tau <= 0:
        raise ValueError(f"Step duration tau must be positive (tau > 0), got {tau}.")

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    q_const = Q_dl + n * F * area * gamma
    slope = 2.0 * n * F * area * math.sqrt(D) * c_bulk / math.sqrt(math.pi)

    if tau is None:
        q_diff = slope * np.sqrt(t_arr)
        q_val = q_diff + q_const
    else:
        q_val = np.empty_like(t_arr, dtype=float)
        fwd_mask = t_arr <= tau
        rev_mask = ~fwd_mask

        if np.any(fwd_mask):
            q_val[fwd_mask] = slope * np.sqrt(t_arr[fwd_mask]) + q_const
        if np.any(rev_mask):
            t_rev = t_arr[rev_mask]
            q_val[rev_mask] = slope * (np.sqrt(t_rev) - np.sqrt(t_rev - tau)) + q_const

    return _format_output(q_val, is_scalar)


def sand_transition_time(
    I: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    F: float = FARADAY,
) -> float:
    r"""Compute Sand equation transition time :math:`\tau` for chronopotentiometry.

    Calculates the transition time :math:`\tau` at which surface concentration
    drops to zero (:math:`c(0, \tau) = 0`) under a constant applied current :math:`I`:

    .. math::

        |I| \sqrt{\tau} = \frac{n F A \sqrt{\pi D} c^*}{2}
        \implies \tau = \frac{\pi D (n F A c^*)^2}{4 I^2}

    Parameters
    ----------
    I : float
        Constant applied current in Amperes (:math:`I \neq 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* > 0`).
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float
        Transition time :math:`\tau` in seconds (:math:`\text{s}`).

    Raises
    ------
    ValueError
        If ``I == 0``, ``c_bulk <= 0``, or physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import sand_transition_time
    >>> tau = sand_transition_time(I=1e-4, n=1, D=1e-5, c_bulk=1e-3, area=1e-4)
    >>> round(tau, 4)
    0.0731
    """
    if I == 0.0:
        raise ValueError("Applied current I cannot be zero in the Sand equation.")
    if c_bulk <= 0.0:
        raise ValueError(
            f"Bulk concentration c_bulk must be strictly positive (c_bulk > 0), got {c_bulk}."
        )
    _validate_common_params(n, D, c_bulk, area, F)

    numerator = math.pi * D * ((n * F * area * c_bulk) ** 2)
    denominator = 4.0 * (I**2)
    return float(numerator / denominator)


def sand(
    t: float | ArrayLike,
    I: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    area: float = 1.0,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute surface concentration transient under constant-current step.

    Calculates the time-dependent electrode surface concentration :math:`c(0, t)`
    under galvanostatic (chronopotentiometric) conditions:

    .. math::

        c(0, t) = c^* - \frac{2 |I| \sqrt{t}}{n F A \sqrt{\pi D}}
                = c^* \left(1 - \sqrt{\frac{t}{\tau}}\right)

    where :math:`\tau` is the Sand transition time. For :math:`t \ge \tau`, the surface
    concentration is depleted (:math:`c(0, t) = 0`).

    Parameters
    ----------
    t : float or array-like
        Time in seconds (:math:`t \ge 0`).
    I : float
        Constant applied current in Amperes (:math:`I \neq 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    area : float, default 1.0
        Electrode area (:math:`A > 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float or np.ndarray
        Surface concentration :math:`c(0, t)` in concentration units (:math:`c \ge 0`).

    Raises
    ------
    ValueError
        If ``I == 0``, ``t < 0``, or physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.step import sand, sand_transition_time
    >>> tau = sand_transition_time(I=1e-4, n=1, D=1e-5, c_bulk=1e-3, area=1e-4)
    >>> # At t = 0, surface concentration is c_bulk
    >>> sand(0.0, I=1e-4, n=1, D=1e-5, c_bulk=1e-3, area=1e-4)
    0.001
    >>> # At t = tau, surface concentration is completely depleted
    >>> round(sand(tau, I=1e-4, n=1, D=1e-5, c_bulk=1e-3, area=1e-4), 6)
    0.0

    """
    if I == 0.0:
        raise ValueError("Applied current I cannot be zero in the Sand equation.")
    _validate_common_params(n, D, c_bulk, area, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    depletion_rate = (2.0 * abs(I)) / (n * F * area * math.sqrt(math.pi * D))
    c_surf = c_bulk - depletion_rate * np.sqrt(t_arr)
    c_surf = np.maximum(0.0, c_surf)

    return _format_output(c_surf, is_scalar)


def sand_potential(
    t: float | ArrayLike,
    tau: float,
    E_half: float = 0.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute potential-time curve for a reversible chronopotentiometric wave.

    Calculates the potential transient for a reversible electron transfer process
    (:math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`) under constant applied current:

    .. math::

        E(t) = E_{\tau/4} + \frac{R T}{n F} \ln\left( \frac{\sqrt{\tau} - \sqrt{t}}{\sqrt{t}} \right)

    where :math:`E_{\tau/4} = E_{1/2}` is the quarter-wave potential (the potential at
    :math:`t = \tau/4`).

    Parameters
    ----------
    t : float or array-like
        Time in seconds (:math:`0 < t < \tau`).
    tau : float
        Sand transition time in seconds (:math:`\tau > 0`).
    E_half : float, default 0.0
        Quarter-wave (half-wave) reversible potential :math:`E_{1/2}` in Volts.
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}/(\text{mol}\cdot\text{K})`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}/\text{mol}`).

    Returns
    -------
    float or np.ndarray
        Electrode potential :math:`E(t)` in Volts (:math:`\text{V}`). At :math:`t \ge \tau`,
        returns ``-np.inf``.

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``t < 0``, ``T <= 0``, ``n <= 0``, ``R <= 0``, or ``F <= 0``.

    Examples
    --------
    >>> from softpotato.analytical.step import sand_potential
    >>> # At quarter-transition time (t = tau / 4), E(t) equals E_half
    >>> tau = 4.0
    >>> sand_potential(1.0, tau=tau, E_half=0.250)
    0.25
    """
    if tau <= 0:
        raise ValueError(f"Transition time tau must be positive (tau > 0), got {tau}.")
    if n <= 0:
        raise ValueError(f"Number of electrons n must be positive (n > 0), got {n}.")
    if T <= 0:
        raise ValueError(f"Temperature T must be positive (T > 0), got {T}.")
    if R <= 0:
        raise ValueError(f"Gas constant R must be positive (R > 0), got {R}.")
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    sqrt_tau = math.sqrt(tau)
    vt = (R * T) / (n * F)

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_t = np.sqrt(t_arr)
        ratio = (sqrt_tau - sqrt_t) / sqrt_t
        e_pot = np.where(
            t_arr == 0.0,
            np.inf,
            np.where(t_arr >= tau, -np.inf, E_half + vt * np.log(ratio)),
        )

    return _format_output(e_pot, is_scalar)


def cottrell_cylinder(
    t: float | ArrayLike,
    r0: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    length: float = 1.0,
    area: float | None = None,
    method: Literal["auto", "aoki", "oldham", "short_time", "long_time"] = "auto",
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute chronoamperometric current transient at a cylindrical wire/fiber electrode.

    Calculates current for a cylindrical microelectrode of radius :math:`r_0` and
    length :math:`l` (:math:`A = 2 \pi r_0 l`):

    .. math::

        I(t) = \frac{n F A D c^*}{r_0} \Phi(\theta)

    where :math:`\theta = \dfrac{D t}{r_0^2}` is dimensionless time.

    Approximations
    --------------
    - **Aoki approximation** (``method="aoki"`` or ``"auto"``) (Aoki et al., 1985):
      Valid across :math:`0 < \theta \le 10^6` with error within 1%:

      .. math::

          \Phi(\theta) = \frac{1}{\sqrt{\pi \theta}} + 0.422 - 0.0675 \log_{10}(\theta)
                         \pm 0.0058 \{\log_{10}(\theta) - 1.47\}^2

      where :math:`\pm` is :math:`+` for :math:`\log_{10}(\theta) \ge 1.47` and
      :math:`-` for :math:`\log_{10}(\theta) < 1.47`.

    - **Oldham rational Padé approximation** (``method="oldham"``) (Oldham, 1987):
      Valid across all :math:`\theta > 0` with maximum error :math:`< 0.4\%`:

      .. math::

          \Phi(\theta) = \frac{1}{\sqrt{\pi \theta}}
                         \left( \frac{1 + 0.36074 \sqrt{\theta} + 0.36502 \theta}
                                     {1 + 0.28173 \sqrt{\theta} + 0.17887 \theta} \right)

    - **Short-time asymptotic expansion** (``method="short_time"``, :math:`\theta \le 1`):

      .. math::

          \Phi(\theta) = \frac{1}{\sqrt{\pi \theta}} + \frac{1}{2}
                         - \frac{1}{4} \sqrt{\frac{\theta}{\pi}} + \frac{1}{8} \theta

    - **Long-time asymptotic expansion** (``method="long_time"``, :math:`\theta \gg 1`):

      .. math::

          \Phi(\theta) = \frac{2}{u} - \frac{2 \gamma}{u^2}, \quad u = \ln(4\theta) - 2\gamma

      where :math:`\gamma \approx 0.57721566` is Euler's constant.

    Parameters
    ----------
    t : float or array-like
        Time in seconds (:math:`t \ge 0`).
    r0 : float
        Electrode cylinder radius in meters or cm (:math:`r_0 > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).
    length : float, default 1.0
        Length of the cylinder wire (:math:`l > 0`).
    area : float or None, default None
        Electrode surface area. If None, defaults to :math:`A = 2 \pi r_0 l`.
    method : {'auto', 'aoki', 'oldham', 'short_time', 'long_time'}, default 'auto'
        Approximation formula to evaluate.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``r0 <= 0``, ``length <= 0``, ``t < 0``, or method is unsupported.

    References
    ----------
    - K. Aoki, K. Honda, K. Tokuda, H. Matsuda, "Voltammetry at microcylinder
      electrodes: Part II. Chronoamperometry", *J. Electroanal. Chem.*, 186 (1985) 79–86.
    - K. B. Oldham, "Analytical expressions for the transient current at a
      microcylinder electrode", *J. Electroanal. Chem.*, 224 (1987) 229–232.
    """

    if r0 <= 0:
        raise ValueError(f"Cylinder radius r0 must be positive (r0 > 0), got {r0}.")
    if length <= 0:
        raise ValueError(
            f"Cylinder length must be positive (length > 0), got {length}."
        )

    area_val = 2.0 * math.pi * r0 * length if area is None else float(area)
    _validate_common_params(n, D, c_bulk, area_val, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (D * t_arr) / (r0**2)

        if method in ("auto", "aoki"):
            # Aoki approximation
            log10_theta = np.log10(theta)
            diff_log = log10_theta - 1.47
            sign_term = np.where(log10_theta >= 1.47, 1.0, -1.0)
            phi = (
                1.0 / np.sqrt(np.pi * theta)
                + 0.422
                - 0.0675 * log10_theta
                + sign_term * 0.0058 * (diff_log**2)
            )
        elif method == "oldham":
            # Oldham Padé approximation
            sqrt_th = np.sqrt(theta)
            num = 1.0 + 0.36074 * sqrt_th + 0.36502 * theta
            den = 1.0 + 0.28173 * sqrt_th + 0.17887 * theta
            phi = (1.0 / np.sqrt(np.pi * theta)) * (num / den)
        elif method == "short_time":
            # Short-time expansion
            sqrt_th = np.sqrt(theta)
            phi = (
                1.0 / np.sqrt(np.pi * theta)
                + 0.5
                - 0.25 * np.sqrt(theta / np.pi)
                + 0.125 * theta
            )
        elif method == "long_time":
            # Long-time expansion
            u = np.log(4.0 * theta) - 2.0 * _EULER_GAMMA
            phi = (2.0 / u) - (2.0 * _EULER_GAMMA / (u**2))
        else:
            raise ValueError(
                f"Unknown method '{method}'. Valid options are 'auto', 'aoki', 'oldham', 'short_time', 'long_time'."
            )

        # Handle t = 0 exactly
        phi = np.where(t_arr == 0.0, np.inf, phi)

    prefactor = (n * F * area_val * D * c_bulk) / r0
    current = prefactor * phi
    return _format_output(current, is_scalar)


def step_concentration_profile(
    x: float | ArrayLike,
    t: float,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
) -> float | np.ndarray:
    r"""Compute exact spatial concentration profile :math:`c(x, t)` for a planar potential step.

    Calculates the exact error-function analytical solution to 1D linear semi-infinite
    diffusion following a step to zero surface concentration (:math:`c(0, t) = 0`):

    .. math::

        c(x, t) = c^* \text{erf}\left(\frac{x}{2 \sqrt{D t}}\right)

    Parameters
    ----------
    x : float or array-like
        Distance from electrode surface in meters or cm (:math:`x \ge 0`).
    t : float
        Time elapsed since the potential step in seconds (:math:`t \ge 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).

    Returns
    -------
    float or np.ndarray
        Concentration at spatial coordinate(s) :math:`x` at time :math:`t`.

    Raises
    ------
    ValueError
        If ``t < 0``, any ``x < 0``, ``D <= 0``, or ``c_bulk < 0``.

    Examples
    --------
    >>> from softpotato.analytical.step import step_concentration_profile
    >>> # Surface concentration is zero for t > 0
    >>> step_concentration_profile(0.0, t=1.0)
    0.0
    >>> # Bulk concentration is reached at large distance
    >>> step_concentration_profile(1.0, t=1e-3, D=1e-5, c_bulk=1.0)
    1.0
    """
    if t < 0:
        raise ValueError(f"Time t must be non-negative (t >= 0), got {t}.")
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive (D > 0), got {D}.")
    if c_bulk < 0:
        raise ValueError(
            f"Bulk concentration c_bulk must be non-negative (c_bulk >= 0), got {c_bulk}."
        )

    is_scalar = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)

    if np.any(x_arr < 0.0):
        raise ValueError("Spatial distance x must be non-negative (x >= 0).")

    if t == 0.0:
        # At t = 0, c(0, 0) = 0 (or undefined) and c(x > 0, 0) = c_bulk
        res = np.where(x_arr == 0.0, 0.0, c_bulk)
    else:
        arg = x_arr / (2.0 * math.sqrt(D * t))
        res = c_bulk * erf(arg)

    return _format_output(res, is_scalar)


def step_flux_profile(
    x: float | ArrayLike,
    t: float,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
) -> float | np.ndarray:
    r"""Compute exact spatial diffusion flux profile :math:`J(x, t)` for a planar potential step.

    Calculates the spatial diffusion flux :math:`J(x, t) = -D \dfrac{\partial c}{\partial x}`:

    .. math::

        J(x, t) = -\frac{\sqrt{D} c^*}{\sqrt{\pi t}} \exp\left(-\frac{x^2}{4 D t}\right)

    At the electrode surface (:math:`x = 0`), the interfacial flux is:

    .. math::

        J(0, t) = -\frac{\sqrt{D} c^*}{\sqrt{\pi t}}

    corresponding to the Cottrell current :math:`I(t) = -n F A J(0, t)`.

    Parameters
    ----------
    x : float or array-like
        Distance from electrode surface in meters or cm (:math:`x \ge 0`).
    t : float
        Time elapsed since the potential step in seconds (:math:`t > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration (:math:`c^* \ge 0`).

    Returns
    -------
    float or np.ndarray
        Diffusion flux :math:`J(x, t)` in :math:`\text{mol}/(\text{m}^2\cdot\text{s})`.

    Raises
    ------
    ValueError
        If ``t <= 0``, any ``x < 0``, ``D <= 0``, or ``c_bulk < 0``.

    Examples
    --------
    >>> from softpotato.analytical.step import step_flux_profile
    >>> flux = step_flux_profile(0.0, t=1.0, D=1e-5, c_bulk=1.0)
    >>> round(flux, 6)
    -0.001784
    """
    if t <= 0:
        raise ValueError(f"Time t must be strictly positive (t > 0), got {t}.")
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive (D > 0), got {D}.")
    if c_bulk < 0:
        raise ValueError(
            f"Bulk concentration c_bulk must be non-negative (c_bulk >= 0), got {c_bulk}."
        )

    is_scalar = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)

    if np.any(x_arr < 0.0):
        raise ValueError("Spatial distance x must be non-negative (x >= 0).")

    prefactor = -(math.sqrt(D) * c_bulk) / math.sqrt(math.pi * t)
    decay = np.exp(-(x_arr**2) / (4.0 * D * t))
    flux = prefactor * decay

    return _format_output(flux, is_scalar)


__all__ = [
    "anson",
    "cottrell",
    "cottrell_cylinder",
    "cottrell_spherical",
    "cottrell_step",
    "sand",
    "sand_potential",
    "sand_transition_time",
    "step_concentration_profile",
    "step_flux_profile",
]
