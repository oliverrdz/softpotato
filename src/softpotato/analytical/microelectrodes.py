"""Analytical and empirical equations for microelectrodes and ultramicroelectrodes (UMEs).

This module provides closed-form steady-state limiting currents, asymptotic expansions,
and full-time transient chronoamperometric models for common microelectrode geometries:
inlaid microdiscs, spherical microelectrodes, hemispherical microelectrodes, and inlaid
microbands.

Functions
---------
microdisc_limiting_current
    Saito steady-state diffusion-limited current for an inlaid circular microdisc.
microdisc_transient
    Shoup & Szabo chronoamperometric transient for an inlaid circular microdisc.
mahon_oldham_transient
    Mahon & Oldham high-precision chronoamperometric transient for an inlaid microdisc.
microhemisphere_limiting_current
    Steady-state diffusion-limited current for a hemispherical microelectrode on an insulator.
microsphere_limiting_current
    Steady-state diffusion-limited current for a spherical microelectrode in bulk solution.
microband_limiting_current
    Quasi-steady-state diffusion-limited current for an inlaid microband electrode.

Conventions & Units
-------------------
All equations are dimensionally consistent in either standard SI or standard
electrochemical CGS units:

- **SI**: :math:`t` in :math:`\\text{s}`, :math:`D` in :math:`\\text{m}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{m}^{-3}`, lengths/radii in :math:`\\text{m}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}` :math:`\\implies I` in :math:`\\text{A}`.
- **CGS**: :math:`t` in :math:`\\text{s}`, :math:`D` in :math:`\\text{cm}^2\\cdot\\text{s}^{-1}`,
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{cm}^{-3}`, lengths/radii in :math:`\\text{cm}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}` :math:`\\implies I` in :math:`\\text{A}`.

Standard IUPAC sign conventions are applied:
- Both cathodic (reduction) and anodic (oxidation) currents are positive (:math:`I > 0`).

References
----------
- Y. Saito, "A theoretical study on the diffusion current at the stationary spherical
  and disc electrodes", *Rev. Polarogr.*, 15 (1968) 177–187.
- J. Newman, "Resistance for flow of current to a disk", *J. Electrochem. Soc.*,
  113 (1966) 501–502.
- D. Shoup, A. Szabo, "Chronoamperometric current at finite disk electrodes",
  *J. Electroanal. Chem.*, 140 (1982) 237–245.
- P. J. Mahon, K. B. Oldham, "A analytical expression for the transient current at a
  microdisk electrode", *Anal. Chem.*, 77 (2005) 6100–6101.
- A. J. Bard, L. R. Faulkner, H. S. White, *Electrochemical Methods: Fundamentals
  and Applications*, 3rd ed., John Wiley & Sons, 2022.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from softpotato.constants import FARADAY


def _validate_common_params(
    n: float,
    D: float,
    c_bulk: float,
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
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


def microdisc_limiting_current(
    radius: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute steady-state diffusion-limited current at an inlaid circular microdisc.

    Calculates the exact steady-state Faradaic current at an inlaid circular microdisc
    electrode of radius :math:`a` flush with an infinite insulating coplanar plane,
    derived by Saito (1968) and Newman (1966):

    .. math::

        I_{\text{lim}} = 4 n F D c^* a

    Parameters
    ----------
    radius : float or array-like
        Radius of the microdisc electrode :math:`a` in meters or cm (:math:`a > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Steady-state limiting current :math:`I_{\text{lim}}` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``radius <= 0`` or any physical parameter is invalid.

    References
    ----------
    - Y. Saito, "A theoretical study on the diffusion current at the stationary spherical
      and disc electrodes", *Rev. Polarogr.*, 15 (1968) 177–187.
    - J. Newman, "Resistance for flow of current to a disk", *J. Electrochem. Soc.*,
      113 (1966) 501–502.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.microelectrodes import microdisc_limiting_current
    >>> # 10 um radius microdisc, 1 mM bulk concentration, D = 1e-5 cm^2/s
    >>> i_ss = microdisc_limiting_current(10e-4, n=1, D=1e-5, c_bulk=1e-6)
    >>> round(i_ss * 1e9, 4)
    3.8594
    """
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(radius)
    r_arr = np.asarray(radius, dtype=float)

    if np.any(r_arr <= 0.0):
        raise ValueError(
            f"Electrode radius must be positive (radius > 0), got {radius!r}."
        )

    i_val = 4.0 * n * F * D * c_bulk * r_arr
    return _format_output(i_val, is_scalar)


def microhemisphere_limiting_current(
    radius: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute steady-state diffusion-limited current at a hemispherical microelectrode.

    Calculates the exact steady-state Faradaic current at a hemispherical microelectrode
    of radius :math:`r` mounted on an insulating substrate:

    .. math::

        I_{\text{lim}} = 2 \pi n F D c^* r

    Parameters
    ----------
    radius : float or array-like
        Radius of the hemispherical electrode :math:`r` in meters or cm (:math:`r > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Steady-state limiting current :math:`I_{\text{lim}}` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``radius <= 0`` or any physical parameter is invalid.

    Examples
    --------
    >>> import math
    >>> from softpotato.analytical.microelectrodes import microhemisphere_limiting_current
    >>> i_ss = microhemisphere_limiting_current(1e-5, n=1, D=1e-9, c_bulk=1.0)
    >>> expected = 2 * math.pi * 96485.332 * 1e-9 * 1.0 * 1e-5
    >>> bool(math.isclose(i_ss, expected, rel_tol=1e-5))
    True
    """
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(radius)
    r_arr = np.asarray(radius, dtype=float)

    if np.any(r_arr <= 0.0):
        raise ValueError(
            f"Electrode radius must be positive (radius > 0), got {radius!r}."
        )

    i_val = 2.0 * math.pi * n * F * D * c_bulk * r_arr
    return _format_output(i_val, is_scalar)


def microsphere_limiting_current(
    radius: float | ArrayLike,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute steady-state diffusion-limited current at a spherical microelectrode.

    Calculates the exact steady-state Faradaic current at a spherical microelectrode
    of radius :math:`r` suspended in semi-infinite bulk solution:

    .. math::

        I_{\text{lim}} = 4 \pi n F D c^* r

    Parameters
    ----------
    radius : float or array-like
        Radius of the spherical electrode :math:`r` in meters or cm (:math:`r > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Steady-state limiting current :math:`I_{\text{lim}}` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``radius <= 0`` or any physical parameter is invalid.

    Examples
    --------
    >>> import math
    >>> from softpotato.analytical.microelectrodes import microsphere_limiting_current
    >>> i_ss = microsphere_limiting_current(1e-5, n=1, D=1e-9, c_bulk=1.0)
    >>> expected = 4 * math.pi * 96485.332 * 1e-9 * 1.0 * 1e-5
    >>> bool(math.isclose(i_ss, expected, rel_tol=1e-5))
    True
    """
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(radius)
    r_arr = np.asarray(radius, dtype=float)

    if np.any(r_arr <= 0.0):
        raise ValueError(
            f"Electrode radius must be positive (radius > 0), got {radius!r}."
        )

    i_val = 4.0 * math.pi * n * F * D * c_bulk * r_arr
    return _format_output(i_val, is_scalar)


def microdisc_transient(
    t: float | ArrayLike,
    radius: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute chronoamperometric current transient at an inlaid microdisc (Shoup & Szabo).

    Calculates the Faradaic diffusion-limited current response at an inlaid circular
    microdisc of radius :math:`a` following an instantaneous potential step, using
    the Shoup and Szabo (1982) empirical model:

    .. math::

        I(t) = 4 n F D c^* a \cdot f(\tau)

    where :math:`\tau = \dfrac{4 D t}{a^2}` is the dimensionless time parameter, and
    :math:`f(\tau)` is given by:

    .. math::

        f(\tau) = 0.7854 + 0.8862 \tau^{-1/2} + 0.2146 \exp(-0.7823 \tau^{-1/2})

    Asymptotic Limits
    -----------------
    - **Short times** (:math:`\tau \to 0`): :math:`f(\tau) \approx 0.8862 \tau^{-1/2}`,
      reproducing the planar Cottrell current across disk area :math:`A = \pi a^2`:

      .. math::

          I(t) \to \frac{n F (\pi a^2) \sqrt{D} c^*}{\sqrt{\pi t}}

    - **Long times** (:math:`\tau \to \infty`): :math:`f(\tau) \to 0.7854 + 0.2146 = 1.0000`,
      reproducing Saito's steady-state limiting current:

      .. math::

          I(t) \to 4 n F D c^* a

    The maximum relative error of this approximation is less than :math:`0.6\%`
    across the entire range :math:`0 < \tau < \infty`.

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the potential step in seconds (:math:`t \ge 0`).
    radius : float
        Radius of the inlaid microdisc :math:`a` in meters or cm (:math:`a > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`). At :math:`t = 0`,
        returns ``np.inf``.

    Raises
    ------
    ValueError
        If ``radius <= 0``, ``t < 0``, or any physical parameter is invalid.

    References
    ----------
    - D. Shoup, A. Szabo, "Chronoamperometric current at finite disk electrodes",
      *J. Electroanal. Chem.*, 140 (1982) 237–245.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.microelectrodes import (
    ...     microdisc_limiting_current,
    ...     microdisc_transient,
    ... )
    >>> # Steady-state convergence at long times (a = 10 um)
    >>> a = 1e-5
    >>> i_trans = microdisc_transient(1e4, radius=a, n=1, D=1e-9, c_bulk=1.0)
    >>> i_steady = microdisc_limiting_current(radius=a, n=1, D=1e-9, c_bulk=1.0)
    >>> bool(np.isclose(i_trans, i_steady, rtol=1e-3))
    True
    """
    if radius <= 0:
        raise ValueError(
            f"Microdisc radius must be positive (radius > 0), got {radius}."
        )
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (4.0 * D * t_arr) / (radius**2)
        inv_sqrt_tau = 1.0 / np.sqrt(tau)
        f_tau = 0.7854 + 0.8862 * inv_sqrt_tau + 0.2146 * np.exp(-0.7823 * inv_sqrt_tau)
        # Handle t = 0 explicitly
        f_tau = np.where(t_arr == 0.0, np.inf, f_tau)

    prefactor = 4.0 * n * F * D * c_bulk * radius
    current = prefactor * f_tau
    return _format_output(current, is_scalar)


def mahon_oldham_transient(
    t: float | ArrayLike,
    radius: float,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    branch: Literal["auto", "short_time", "long_time"] = "auto",
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute high-precision chronoamperometric transient at a microdisc (Mahon & Oldham).

    Calculates the Faradaic diffusion-controlled current at an inlaid circular microdisc
    electrode of radius :math:`a` across all time regimes using the Mahon and Oldham (2005)
    analytical piecewise model:

    .. math::

        I(t) = \pi n F D c^* a \cdot f(\sigma)

    where :math:`\sigma = \dfrac{D t}{a^2}` is dimensionless time, and :math:`f(\sigma)` is:

    .. math::

        f(\sigma) = \begin{cases}
        \dfrac{1}{\sqrt{\pi \sigma}} + 1 + \sqrt{\dfrac{\sigma}{4\pi}} - \dfrac{3\sigma}{25} + \dfrac{3\sigma^{3/2}}{226}, & \sigma \le 1.281 \\
        \dfrac{4}{\pi} + \dfrac{8}{\sqrt{\pi^5 \sigma}} + \dfrac{25\sigma^{-3/2}}{2792} - \dfrac{\sigma^{-5/2}}{3880} - \dfrac{\sigma^{-7/2}}{4500}, & \sigma \ge 1.281
        \end{cases}

    Precision & Error Bounds
    ------------------------
    - Maximum relative error is below :math:`0.02\%` for all intermediate and transition times.
    - As :math:`\sigma \to 0` (short times), :math:`f(\sigma) \to \dfrac{1}{\sqrt{\pi \sigma}}`,
      matching exact planar Cottrell behavior: :math:`I(t) \to \dfrac{n F (\pi a^2) \sqrt{D} c^*}{\sqrt{\pi t}}`.
    - As :math:`\sigma \to \infty` (long times), :math:`f(\sigma) \to \dfrac{4}{\pi}`,
      matching exact Saito steady-state limiting current: :math:`I(t) \to 4 n F D c^* a`.

    Parameters
    ----------
    t : float or array-like
        Time elapsed since the potential step in seconds (:math:`t \ge 0`).
    radius : float
        Radius of the inlaid microdisc :math:`a` in meters or cm (:math:`a > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    branch : {'auto', 'short_time', 'long_time'}, default 'auto'
        Branch formula to evaluate:

        - ``'auto'``: Automatically selects short-time (:math:`\sigma \le 1.281`) or
          long-time (:math:`\sigma \ge 1.281`) branch element-wise.
        - ``'short_time'``: Evaluates the short-time asymptotic series expansion.
        - ``'long_time'``: Evaluates the long-time asymptotic series expansion.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(t)` in Amperes (:math:`\text{A}`). At :math:`t = 0`,
        returns ``np.inf``.

    Raises
    ------
    ValueError
        If ``radius <= 0``, ``t < 0``, ``branch`` is invalid, or physical parameters are invalid.

    References
    ----------
    - P. J. Mahon, K. B. Oldham, "A analytical expression for the transient current at a
      microdisk electrode", *Anal. Chem.*, 77 (2005) 6100–6101.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.microelectrodes import (
    ...     mahon_oldham_transient,
    ...     microdisc_limiting_current,
    ... )
    >>> # Steady-state convergence at long times (a = 10 um)
    >>> a = 1e-5
    >>> i_trans = mahon_oldham_transient(1e4, radius=a, n=1, D=1e-9, c_bulk=1.0)
    >>> i_steady = microdisc_limiting_current(radius=a, n=1, D=1e-9, c_bulk=1.0)
    >>> bool(np.isclose(i_trans, i_steady, rtol=1e-4))
    True
    """
    if radius <= 0:
        raise ValueError(
            f"Microdisc radius must be positive (radius > 0), got {radius}."
        )
    if branch not in ("auto", "short_time", "long_time"):
        raise ValueError(
            f"Unknown branch '{branch}'. Valid options are 'auto', 'short_time', 'long_time'."
        )
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr < 0.0):
        raise ValueError("Time t must be non-negative (t >= 0).")

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (D * t_arr) / (radius**2)

        def _eval_short(s: np.ndarray) -> np.ndarray:
            return (
                1.0 / np.sqrt(math.pi * s)
                + 1.0
                + np.sqrt(s / (4.0 * math.pi))
                - 3.0 * s / 25.0
                + 3.0 * (s**1.5) / 226.0
            )

        def _eval_long(s: np.ndarray) -> np.ndarray:
            return (
                4.0 / math.pi
                + 8.0 / np.sqrt((math.pi**5) * s)
                + 25.0 * (s ** (-1.5)) / 2792.0
                - (s ** (-2.5)) / 3880.0
                - (s ** (-3.5)) / 4500.0
            )

        if branch == "short_time":
            f_sigma = _eval_short(sigma)
        elif branch == "long_time":
            f_sigma = _eval_long(sigma)
        else:  # "auto"
            short_mask = sigma <= 1.281
            f_sigma = np.empty_like(sigma, dtype=float)

            # Avoid division by zero warnings when evaluating masked branches
            if np.any(short_mask):
                f_sigma[short_mask] = _eval_short(sigma[short_mask])
            if np.any(~short_mask):
                f_sigma[~short_mask] = _eval_long(sigma[~short_mask])

        # Exact handling at t = 0
        f_sigma = np.where(t_arr == 0.0, np.inf, f_sigma)

    prefactor = math.pi * n * F * D * c_bulk * radius
    current = prefactor * f_sigma
    return _format_output(current, is_scalar)


def microband_limiting_current(
    t: float | ArrayLike,
    width: float,
    length: float = 1.0,
    n: float = 1,
    D: float = 1e-5,
    c_bulk: float = 1e-3,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute quasi-steady-state diffusion-limited current at an inlaid microband.

    Calculates the Faradaic diffusion-limited current at an inlaid microband electrode
    of width :math:`w` and length :math:`l` (:math:`l \gg w`) in the quasi-steady-state
    regime:

    .. math::

        I_{\text{lim}}(t) = \frac{2 \pi n F D c^* l}{\ln\left(\dfrac{0.64 D t}{w^2}\right)}

    Validity Domain
    ---------------
    Because diffusion to a band electrode is two-dimensional without a true time-independent
    steady state, the current decays logarithmically with time. The logarithmic approximation
    is physically meaningful for long dimensionless times:

    .. math::

        \frac{0.64 D t}{w^2} > 1 \implies t > \frac{w^2}{0.64 D}

    Parameters
    ----------
    t : float or array-like
        Time in seconds (:math:`t > \dfrac{w^2}{0.64 D}`).
    width : float
        Width of the microband :math:`w` in meters or cm (:math:`w > 0`).
    length : float, default 1.0
        Length of the microband :math:`l` in meters or cm (:math:`l > 0`).
    n : int or float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    D : float, default 1e-5
        Diffusion coefficient :math:`D` in :math:`\text{m}^2/\text{s}` or
        :math:`\text{cm}^2/\text{s}` (:math:`D > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* \ge 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float or np.ndarray
        Quasi-steady-state limiting current :math:`I_{\text{lim}}(t)` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If ``width <= 0``, ``length <= 0``, ``t <= w**2 / (0.64 * D)``, or physical
        parameters are invalid.

    References
    ----------
    - A. Szabo, D. K. Cope, D. E. Tallman, P. M. Kovach, R. M. Wightman, "Chronoamperometric
      current at hemicylinder and band microelectrodes: Theory and experiment",
      *J. Electroanal. Chem.*, 217 (1987) 417–423.
    - K. Aoki, K. Tokuda, H. Matsuda, "Derivation of an approximate equation for
      chronoamperometric curves at microband electrodes and its experimental verification",
      *J. Electroanal. Chem.*, 230 (1987) 61–67.

    Examples
    --------
    >>> import math
    >>> from softpotato.analytical.microelectrodes import microband_limiting_current
    >>> # w = 10 um, length = 1 mm, D = 1e-9 m^2/s, t = 1.0 s
    >>> i_band = microband_limiting_current(1.0, width=1e-5, length=1e-3, n=1, D=1e-9, c_bulk=1.0)
    >>> i_band > 0
    True
    """
    if width <= 0:
        raise ValueError(f"Microband width must be positive (width > 0), got {width}.")
    if length <= 0:
        raise ValueError(
            f"Microband length must be positive (length > 0), got {length}."
        )
    _validate_common_params(n, D, c_bulk, F)

    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)

    if np.any(t_arr <= 0.0):
        raise ValueError("Time t must be strictly positive (t > 0).")

    arg = (0.64 * D * t_arr) / (width**2)
    if np.any(arg <= 1.0):
        min_arg = float(np.min(arg))
        raise ValueError(
            f"Quasi-steady-state microband equation requires 0.64 * D * t / width**2 > 1.0 "
            f"(got minimum argument {min_arg:.4e}). For shorter times t <= width**2 / (0.64 * D), "
            f"use planar Cottrell linear diffusion."
        )

    log_term = np.log(arg)
    numerator = 2.0 * math.pi * n * F * D * c_bulk * length
    current = numerator / log_term
    return _format_output(current, is_scalar)
