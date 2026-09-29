"""Analytical and empirical equations for thermodynamics and interfacial kinetics.

This module provides closed-form equations, asymptotic expansions, and regression
analysis tools for electrochemical thermodynamics (Nernst equation) and heterogeneous
interfacial charge-transfer kinetics (Butler–Volmer equation and Tafel approximations).

Functions & Classes
-------------------
nernst
    Compute Nernst equilibrium electrode potential from concentration or concentration ratio.
nernst_potential
    Alias for :func:`nernst`.
nernst_ratio
    Compute equilibrium concentration ratio :math:`c_{\\text{Ox}} / c_{\\text{Red}}` from potential.
nernst_equilibrium_concentrations
    Compute equilibrium species concentrations from potential and total concentration.
exchange_current_density
    Compute exchange current density :math:`j_0` from standard rate constant and concentrations.
exchange_current
    Compute exchange current :math:`I_0` from standard rate constant, concentrations, and area.
charge_transfer_resistance
    Compute charge-transfer resistance :math:`R_{\\text{ct}}` from exchange current.
butler_volmer
    Compute Faradaic current or current density using the Butler–Volmer equation.
butler_volmer_current_density
    Compute Faradaic current density using the Butler–Volmer equation.
butler_volmer_linear
    Compute low-overpotential linear approximation of the Butler–Volmer equation.
tafel_slope
    Compute theoretical Tafel slope :math:`b` in V/decade.
tafel
    Compute high-overpotential Tafel current approximation.
tafel_overpotential
    Compute overpotential :math:`\\eta` from current using the Tafel approximation.
TafelResult
    Container dataclass for Tafel diagnostic linear regression parameters.
tafel_analysis
    Perform linear regression on overpotential-current data to extract Tafel parameters.

Conventions & Units
-------------------
All equations are dimensionally consistent in either standard SI or standard
electrochemical CGS units:

- **SI**:
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{m}^{-3}`,
  :math:`A` in :math:`\\text{m}^2`,
  :math:`k^0` in :math:`\\text{m}\\cdot\\text{s}^{-1}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}`
  :math:`\\implies I` in :math:`\\text{A}`, :math:`j` in :math:`\\text{A}\\cdot\\text{m}^{-2}`.
- **CGS**:
  :math:`c^*` in :math:`\\text{mol}\\cdot\\text{cm}^{-3}`,
  :math:`A` in :math:`\\text{cm}^2`,
  :math:`k^0` in :math:`\\text{cm}\\cdot\\text{s}^{-1}`,
  :math:`F` in :math:`\\text{C}\\cdot\\text{mol}^{-1}`
  :math:`\\implies I` in :math:`\\text{A}`, :math:`j` in :math:`\\text{A}\\cdot\\text{cm}^{-2}`.

Standard IUPAC sign conventions are applied by default:
- Anodic (oxidation) currents are positive (:math:`I > 0`, :math:`j > 0`) for positive overpotentials (:math:`\\eta > 0`).
- Cathodic (reduction) currents are negative (:math:`I < 0`, :math:`j < 0`) for negative overpotentials (:math:`\\eta < 0`).
- The toggle ``anodic_positive=False`` can be specified to follow the polarographic / US convention (cathodic positive).

References
----------
- A. J. Bard, L. R. Faulkner, H. S. White, *Electrochemical Methods:
  Fundamentals and Applications*, 3rd ed., John Wiley & Sons, 2022, Chapter 3.
- J. O'M. Bockris, A. K. N. Reddy, M. Gamboa-Aldeco, *Modern Electrochemistry 2A:
  Fundamentals of Electrodics*, 2nd ed., Kluwer Academic / Plenum Publishers, 2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import expit
from scipy.stats import linregress

from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

_LN10: float = 2.302585092994046


def _validate_n(n: float) -> None:
    """Validate number of electrons transferred."""
    if n <= 0:
        raise ValueError(f"Number of electrons n must be positive (n > 0), got {n}.")


def _validate_temp(T: float) -> None:
    """Validate thermodynamic temperature."""
    if T <= 0:
        raise ValueError(f"Temperature T must be positive (T > 0 in Kelvin), got {T}.")


def _validate_faraday(F: float) -> None:
    """Validate Faraday constant."""
    if F <= 0:
        raise ValueError(f"Faraday constant F must be positive (F > 0), got {F}.")


def _validate_gas_constant(R: float) -> None:
    """Validate molar gas constant."""
    if R <= 0:
        raise ValueError(f"Molar gas constant R must be positive (R > 0), got {R}.")


def _validate_alpha(alpha: float) -> None:
    """Validate transfer coefficient alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Transfer coefficient alpha must be strictly between 0 and 1 (0 < alpha < 1), got {alpha}."
        )


def _validate_positive(val: float, name: str) -> None:
    """Validate that a physical parameter is strictly positive."""
    if val <= 0:
        raise ValueError(f"{name} must be positive ({name} > 0), got {val}.")


def _validate_non_negative(val: float, name: str) -> None:
    """Validate that a physical parameter is non-negative."""
    if val < 0:
        raise ValueError(f"{name} must be non-negative ({name} >= 0), got {val}.")


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


# ==============================================================================
# 1. Thermodynamics: Nernst Equation
# ==============================================================================


def nernst(
    c_ox: float | ArrayLike,
    c_red: float | ArrayLike | None = None,
    E0: float = 0.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute Nernst equilibrium electrode potential.

    Calculates the reversible thermodynamic electrode potential :math:`E_{\text{eq}}`
    for the redox reaction :math:`\text{Ox} + n e^- \rightleftharpoons \text{Red}`:

    .. math::

        E_{\text{eq}} = E^{0\prime} + \frac{R T}{n F} \ln\left( \frac{c_{\text{Ox}}}{c_{\text{Red}}} \right)

    If ``c_red`` is omitted (set to ``None``), ``c_ox`` is interpreted as the
    precomputed concentration ratio :math:`r = c_{\text{Ox}} / c_{\text{Red}}`.

    Parameters
    ----------
    c_ox : float or array-like
        Concentration of oxidized species :math:`c_{\text{Ox}}` (:math:`\ge 0`),
        or concentration ratio :math:`c_{\text{Ox}} / c_{\text{Red}}` if ``c_red`` is ``None``.
    c_red : float or array-like, optional
        Concentration of reduced species :math:`c_{\text{Red}}` (:math:`\ge 0`).
        If ``None``, ``c_ox`` is treated as the ratio :math:`c_{\text{Ox}} / c_{\text{Red}}`.
    E0 : float, default 0.0
        Standard or formal electrode potential :math:`E^{0\prime}` in Volts (V).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    float or np.ndarray
        Equilibrium electrode potential :math:`E_{\text{eq}}` in Volts (V).
        Returns ``-np.inf`` if :math:`c_{\text{Ox}} = 0` (and :math:`c_{\text{Red}} > 0`),
        and ``np.inf`` if :math:`c_{\text{Red}} = 0` (and :math:`c_{\text{Ox}} > 0`).

    Raises
    ------
    ValueError
        If concentrations are negative, if both :math:`c_{\text{Ox}}` and :math:`c_{\text{Red}}`
        are zero, or if physical parameters :math:`n, T, R, F` are non-positive.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import nernst
    >>> # Equal concentrations yield formal potential E0
    >>> nernst(1.0, 1.0, E0=0.25)
    0.25
    >>> # 10-fold excess of Ox at 298.15 K shifts potential by ~59.16 mV (n=1)
    >>> round(nernst(10.0, 1.0, E0=0.0) * 1000, 2)
    59.16
    >>> # Direct ratio input
    >>> round(nernst(10.0, E0=0.0) * 1000, 2)
    59.16
    """
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    f_thermal = (R * T) / (n * F)

    if c_red is None:
        is_scalar = np.isscalar(c_ox)
        ratio_arr = np.asarray(c_ox, dtype=float)
        if np.any(ratio_arr < 0.0):
            raise ValueError(
                "Concentration ratio (c_ox / c_red) must be non-negative (>= 0)."
            )
        with np.errstate(divide="ignore"):
            e_eq = E0 + f_thermal * np.log(ratio_arr)
        return _format_output(e_eq, is_scalar)

    is_scalar = np.isscalar(c_ox) and np.isscalar(c_red)
    ox_arr = np.asarray(c_ox, dtype=float)
    red_arr = np.asarray(c_red, dtype=float)

    if np.any(ox_arr < 0.0) or np.any(red_arr < 0.0):
        raise ValueError("Species concentrations must be non-negative (>= 0).")

    both_zero = np.logical_and(ox_arr == 0.0, red_arr == 0.0)
    if np.any(both_zero):
        raise ValueError(
            "Concentrations of Ox and Red cannot both be zero simultaneously."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ox_arr / red_arr
        e_eq = E0 + f_thermal * np.log(ratio)

    return _format_output(e_eq, is_scalar)


nernst_potential = nernst


def nernst_ratio(
    E: float | ArrayLike,
    E0: float = 0.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute equilibrium concentration ratio :math:`c_{\text{Ox}} / c_{\text{Red}}` from potential.

    Inverts the Nernst equation to determine the equilibrium surface concentration
    ratio corresponding to an applied electrode potential :math:`E`:

    .. math::

        \theta = \frac{c_{\text{Ox}}}{c_{\text{Red}}} = \exp\left( \frac{n F (E - E^{0\prime})}{R T} \right)

    Parameters
    ----------
    E : float or array-like
        Electrode potential :math:`E` in Volts (V).
    E0 : float, default 0.0
        Standard or formal electrode potential :math:`E^{0\prime}` in Volts (V).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    float or np.ndarray
        Equilibrium concentration ratio :math:`c_{\text{Ox}} / c_{\text{Red}}` (dimensionless).

    Raises
    ------
    ValueError
        If :math:`n, T, R, F` are non-positive.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import nernst_ratio
    >>> # At formal potential E = E0, ratio is unity
    >>> nernst_ratio(0.25, E0=0.25)
    1.0
    >>> # At E = E0 + 0.05916 V, ratio is ~10.0 (n=1, 298.15 K)
    >>> round(nernst_ratio(0.05916, E0=0.0), 1)
    10.0
    """
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    is_scalar = np.isscalar(E)
    e_arr = np.asarray(E, dtype=float)

    z = (n * F / (R * T)) * (e_arr - E0)
    with np.errstate(over="ignore"):
        ratio = np.exp(z)

    return _format_output(ratio, is_scalar)


def nernst_equilibrium_concentrations(
    E: float | ArrayLike,
    c_total: float | ArrayLike = 1.0,
    E0: float = 0.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    r"""Compute equilibrium species concentrations from potential and total concentration.

    Evaluates the equilibrium concentrations of both oxidized (:math:`c_{\text{Ox}}`)
    and reduced (:math:`c_{\text{Red}}`) species given total electroactive concentration
    :math:`c_{\text{total}} = c_{\text{Ox}} + c_{\text{Red}}`:

    .. math::

        c_{\text{Ox}} = c_{\text{total}} \cdot \frac{1}{1 + \exp\left(-\frac{n F (E - E^{0\prime})}{R T}\right)}

    .. math::

        c_{\text{Red}} = c_{\text{total}} \cdot \frac{1}{1 + \exp\left(\frac{n F (E - E^{0\prime})}{R T}\right)}

    The calculation uses numerically stable logistic evaluation (:func:`scipy.special.expit`)
    to prevent floating-point overflow across extreme potentials.

    Parameters
    ----------
    E : float or array-like
        Electrode potential :math:`E` in Volts (V).
    c_total : float or array-like, default 1.0
        Total concentration :math:`c_{\text{total}} = c_{\text{Ox}} + c_{\text{Red}}`
        in :math:`\text{mol}/\text{m}^3` or :math:`\text{mol}/\text{cm}^3` (:math:`c_{\text{total}} \ge 0`).
    E0 : float, default 0.0
        Standard or formal electrode potential :math:`E^{0\prime}` in Volts (V).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    tuple of (float or np.ndarray, float or np.ndarray)
        Tuple of ``(c_ox, c_red)`` representing equilibrium concentrations.

    Raises
    ------
    ValueError
        If :math:`c_{\text{total}} < 0` or if physical parameters :math:`n, T, R, F` are non-positive.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import nernst_equilibrium_concentrations
    >>> # At formal potential, both species have equal 50% fraction
    >>> c_ox, c_red = nernst_equilibrium_concentrations(0.0, c_total=2.0, E0=0.0)
    >>> (round(c_ox, 2), round(c_red, 2))
    (1.0, 1.0)
    """
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    is_scalar = np.isscalar(E) and np.isscalar(c_total)
    e_arr = np.asarray(E, dtype=float)
    ctot_arr = np.asarray(c_total, dtype=float)

    if np.any(ctot_arr < 0.0):
        raise ValueError(
            f"Total concentration c_total must be non-negative (>= 0), got {c_total!r}."
        )

    z = (n * F / (R * T)) * (e_arr - E0)
    frac_ox = expit(z)
    frac_red = expit(-z)

    c_ox_out = ctot_arr * frac_ox
    c_red_out = ctot_arr * frac_red

    if is_scalar:
        return float(c_ox_out.item()), float(c_red_out.item())
    return c_ox_out, c_red_out


# ==============================================================================
# 2. Interfacial Charge-Transfer: Butler–Volmer Kinetics
# ==============================================================================


def exchange_current_density(
    k0: float,
    c_ox: float = 1.0,
    c_red: float = 1.0,
    alpha: float = 0.5,
    n: float = 1,
    F: float = FARADAY,
) -> float:
    r"""Compute exchange current density :math:`j_0`.

    Calculates the equilibrium exchange current density for the single-step
    electron transfer reaction :math:`\text{Red} \rightleftharpoons \text{Ox} + n e^-`:

    .. math::

        j_0 = n F k^0 (c_{\text{Ox}}^*)^{1-\alpha} (c_{\text{Red}}^*)^\alpha

    Parameters
    ----------
    k0 : float
        Standard heterogeneous charge-transfer rate constant in :math:`\text{m}/\text{s}`
        or :math:`\text{cm}/\text{s}` (:math:`k^0 \ge 0`).
    c_ox : float, default 1.0
        Bulk concentration of oxidized species in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c_{\text{Ox}}^* \ge 0`).
    c_red : float, default 1.0
        Bulk concentration of reduced species in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c_{\text{Red}}^* \ge 0`).
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float
        Exchange current density :math:`j_0` in :math:`\text{A}/\text{m}^2`
        or :math:`\text{A}/\text{cm}^2`.

    Raises
    ------
    ValueError
        If :math:`k^0 < 0`, :math:`c^* < 0`, :math:`\alpha \notin (0, 1)`, or :math:`n, F \le 0`.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import exchange_current_density
    >>> # k0 = 1e-2 cm/s, c_bulk = 1e-3 mol/cm^3, n=1
    >>> j0 = exchange_current_density(1e-2, c_ox=1e-3, c_red=1e-3, alpha=0.5, n=1)
    >>> round(j0, 2)
    0.96
    """
    _validate_non_negative(k0, "Standard rate constant k0")
    _validate_non_negative(c_ox, "Oxidized concentration c_ox")
    _validate_non_negative(c_red, "Reduced concentration c_red")
    _validate_alpha(alpha)
    _validate_n(n)
    _validate_faraday(F)

    return float(n * F * k0 * (c_ox ** (1.0 - alpha)) * (c_red**alpha))


def exchange_current(
    k0: float,
    c_ox: float = 1.0,
    c_red: float = 1.0,
    alpha: float = 0.5,
    area: float = 1.0,
    n: float = 1,
    F: float = FARADAY,
) -> float:
    r"""Compute total exchange current :math:`I_0`.

    .. math::

        I_0 = j_0 A = n F A k^0 (c_{\text{Ox}}^*)^{1-\alpha} (c_{\text{Red}}^*)^\alpha

    Parameters
    ----------
    k0 : float
        Standard heterogeneous charge-transfer rate constant in :math:`\text{m}/\text{s}`
        or :math:`\text{cm}/\text{s}` (:math:`k^0 \ge 0`).
    c_ox : float, default 1.0
        Bulk concentration of oxidized species (:math:`c_{\text{Ox}}^* \ge 0`).
    c_red : float, default 1.0
        Bulk concentration of reduced species (:math:`c_{\text{Red}}^* \ge 0`).
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    area : float, default 1.0
        Electrode surface area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}` (:math:`F > 0`).

    Returns
    -------
    float
        Total exchange current :math:`I_0` in Amperes (A).

    Raises
    ------
    ValueError
        If parameters are physically invalid.
    """
    _validate_positive(area, "Electrode area")
    j0 = exchange_current_density(k0, c_ox=c_ox, c_red=c_red, alpha=alpha, n=n, F=F)
    return float(j0 * area)


def charge_transfer_resistance(
    i0: float,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float:
    r"""Compute charge-transfer resistance :math:`R_{\text{ct}}`.

    Evaluates the activation polarization resistance at the equilibrium potential
    (:math:`\eta = 0`):

    .. math::

        R_{\text{ct}} = \left( \frac{\partial \eta}{\partial I} \right)_{\eta=0} = \frac{R T}{n F I_0}

    If an exchange current density :math:`j_0` is supplied in place of :math:`I_0`,
    the function returns the area-specific charge-transfer resistance
    :math:`R_{\text{ct,spec}} = R_{\text{ct}} \cdot A` in :math:`\Omega\cdot\text{m}^2`
    or :math:`\Omega\cdot\text{cm}^2`.

    Parameters
    ----------
    i0 : float
        Exchange current :math:`I_0` in Amperes (A), or exchange current density
        :math:`j_0` (:math:`I_0 > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    float
        Charge-transfer resistance :math:`R_{\text{ct}}` in Ohms (:math:`\Omega`),
        or :math:`\Omega\cdot\text{area}` if :math:`j_0` was passed.

    Raises
    ------
    ValueError
        If :math:`I_0 \le 0` or if physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import charge_transfer_resistance
    >>> # At I0 = 1 mA, n=1, 298.15 K
    >>> r_ct = charge_transfer_resistance(1e-3, n=1)
    >>> round(r_ct, 2)
    25.69
    """
    _validate_positive(i0, "Exchange current i0")
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    return float((R * T) / (n * F * i0))


def butler_volmer(
    eta: float | ArrayLike,
    i0: float = 1.0,
    *,
    alpha: float = 0.5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    c_ox_surf: float | ArrayLike | None = None,
    c_red_surf: float | ArrayLike | None = None,
    c_ox_bulk: float = 1.0,
    c_red_bulk: float = 1.0,
    anodic_positive: bool = True,
) -> float | np.ndarray:
    r"""Compute Faradaic current or current density using the Butler–Volmer equation.

    Under pure activation control (:math:`c_i(0) = c_i^*`), calculates the net
    Faradaic current response as a function of electrode overpotential
    :math:`\eta = E - E_{\text{eq}}`:

    .. math::

        I(\eta) = I_0 \left[ \exp\left( \frac{(1 - \alpha) n F \eta}{R T} \right) - \exp\left( -\frac{\alpha n F \eta}{R T} \right) \right]

    When interfacial surface concentrations :math:`c_{\text{Ox}}(0)` or
    :math:`c_{\text{Red}}(0)` are supplied, evaluates the mass-transport-corrected
    generalized Butler–Volmer equation:

    .. math::

        I(\eta) = I_0 \left[ \frac{c_{\text{Red}}(0)}{c_{\text{Red}}^*} \exp\left( \frac{(1 - \alpha) n F \eta}{R T} \right) - \frac{c_{\text{Ox}}(0)}{c_{\text{Ox}}^*} \exp\left( -\frac{\alpha n F \eta}{R T} \right) \right]

    Parameters
    ----------
    eta : float or array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts (V).
    i0 : float, default 1.0
        Exchange current :math:`I_0` in Amperes (A), or exchange current density
        :math:`j_0` (:math:`I_0 \ge 0`). If set to 1.0, returns normalized current :math:`I / I_0`.
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).
    c_ox_surf : float or array-like, optional
        Surface concentration of oxidized species :math:`c_{\text{Ox}}(0) \ge 0`.
    c_red_surf : float or array-like, optional
        Surface concentration of reduced species :math:`c_{\text{Red}}(0) \ge 0`.
    c_ox_bulk : float, default 1.0
        Bulk concentration of oxidized species :math:`c_{\text{Ox}}^* > 0`.
    c_red_bulk : float, default 1.0
        Bulk concentration of reduced species :math:`c_{\text{Red}}^* > 0`.
    anodic_positive : bool, default True
        If ``True`` (default IUPAC convention), anodic oxidation currents are positive
        (:math:`I > 0` for :math:`\eta > 0`) and cathodic reduction currents are negative
        (:math:`I < 0` for :math:`\eta < 0`). If ``False`` (polarographic convention),
        cathodic currents are positive.

    Returns
    -------
    float or np.ndarray
        Faradaic current :math:`I(\eta)` in Amperes (A) or current density in
        :math:`\text{A}/\text{area}`.

    Raises
    ------
    ValueError
        If parameters are physically invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import butler_volmer
    >>> # Zero overpotential gives zero current
    >>> butler_volmer(0.0, i0=1e-3)
    0.0
    >>> # Symmetric curve at alpha = 0.5
    >>> i_anodic = butler_volmer(0.05, i0=1.0)
    >>> i_cathodic = butler_volmer(-0.05, i0=1.0)
    >>> round(i_anodic + i_cathodic, 6)
    0.0
    """
    _validate_non_negative(i0, "Exchange current i0")
    _validate_alpha(alpha)
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    is_scalar = np.isscalar(eta)
    eta_arr = np.asarray(eta, dtype=float)

    f_factor = (n * F) / (R * T)

    # Evaluate mass-transport correction factors
    gamma_red: float | np.ndarray = 1.0
    if c_red_surf is not None:
        _validate_positive(c_red_bulk, "Bulk reduced concentration c_red_bulk")
        red_surf_arr = np.asarray(c_red_surf, dtype=float)
        if np.any(red_surf_arr < 0.0):
            raise ValueError(
                "Surface concentration c_red_surf must be non-negative (>= 0)."
            )
        gamma_red = red_surf_arr / c_red_bulk

    gamma_ox: float | np.ndarray = 1.0
    if c_ox_surf is not None:
        _validate_positive(c_ox_bulk, "Bulk oxidized concentration c_ox_bulk")
        ox_surf_arr = np.asarray(c_ox_surf, dtype=float)
        if np.any(ox_surf_arr < 0.0):
            raise ValueError(
                "Surface concentration c_ox_surf must be non-negative (>= 0)."
            )
        gamma_ox = ox_surf_arr / c_ox_bulk

    # Exponential terms
    arg_anodic = (1.0 - alpha) * f_factor * eta_arr
    arg_cathodic = -alpha * f_factor * eta_arr

    with np.errstate(over="ignore"):
        exp_anodic = np.exp(arg_anodic)
        exp_cathodic = np.exp(arg_cathodic)

    current_anodic = gamma_red * exp_anodic
    current_cathodic = gamma_ox * exp_cathodic

    if anodic_positive:
        net_current = i0 * (current_anodic - current_cathodic)
    else:
        net_current = i0 * (current_cathodic - current_anodic)

    return _format_output(net_current, is_scalar)


def butler_volmer_current_density(
    eta: float | ArrayLike,
    j0: float = 1.0,
    *,
    alpha: float = 0.5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    c_ox_surf: float | ArrayLike | None = None,
    c_red_surf: float | ArrayLike | None = None,
    c_ox_bulk: float = 1.0,
    c_red_bulk: float = 1.0,
    anodic_positive: bool = True,
) -> float | np.ndarray:
    r"""Compute Faradaic current density :math:`j(\eta)` using the Butler–Volmer equation.

    Convenience wrapper around :func:`butler_volmer` explicitly denoting current density.

    Parameters
    ----------
    eta : float or array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts (V).
    j0 : float, default 1.0
        Exchange current density :math:`j_0` in :math:`\text{A}/\text{m}^2`
        or :math:`\text{A}/\text{cm}^2` (:math:`j_0 \ge 0`).
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).
    c_ox_surf : float or array-like, optional
        Surface concentration of oxidized species :math:`c_{\text{Ox}}(0) \ge 0`.
    c_red_surf : float or array-like, optional
        Surface concentration of reduced species :math:`c_{\text{Red}}(0) \ge 0`.
    c_ox_bulk : float, default 1.0
        Bulk concentration of oxidized species :math:`c_{\text{Ox}}^* > 0`.
    c_red_bulk : float, default 1.0
        Bulk concentration of reduced species :math:`c_{\text{Red}}^* > 0`.
    anodic_positive : bool, default True
        If ``True``, anodic oxidation currents are positive.

    Returns
    -------
    float or np.ndarray
        Current density :math:`j(\eta)` in :math:`\text{A}/\text{m}^2`
        or :math:`\text{A}/\text{cm}^2`.
    """
    return butler_volmer(
        eta,
        i0=j0,
        alpha=alpha,
        n=n,
        T=T,
        R=R,
        F=F,
        c_ox_surf=c_ox_surf,
        c_red_surf=c_red_surf,
        c_ox_bulk=c_ox_bulk,
        c_red_bulk=c_red_bulk,
        anodic_positive=anodic_positive,
    )


def butler_volmer_linear(
    eta: float | ArrayLike,
    i0: float = 1.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    anodic_positive: bool = True,
) -> float | np.ndarray:
    r"""Compute low-overpotential linear approximation of the Butler–Volmer equation.

    When overpotential is small compared to the thermal voltage (:math:`|\eta| \ll \frac{R T}{n F}`):

    .. math::

        I(\eta) \approx I_0 \frac{n F \eta}{R T} = \frac{\eta}{R_{\text{ct}}}

    This linear relationship is valid within :math:`< 1\%` error for :math:`|\eta| \lesssim 10\,\text{mV}`.

    Parameters
    ----------
    eta : float or array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts (V).
    i0 : float, default 1.0
        Exchange current :math:`I_0` in Amperes (A) or current density (:math:`I_0 \ge 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).
    anodic_positive : bool, default True
        If ``True``, anodic current is positive for :math:`\eta > 0`.

    Returns
    -------
    float or np.ndarray
        Linearized current approximation in Amperes (A) or current density.

    Raises
    ------
    ValueError
        If parameters are physically invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import butler_volmer_linear, butler_volmer
    >>> # At eta = 2 mV, linear approximation closely matches full Butler-Volmer
    >>> i_full = butler_volmer(0.002, i0=1.0)
    >>> i_lin = butler_volmer_linear(0.002, i0=1.0)
    >>> round(abs(i_full - i_lin) / i_full * 100, 3) < 0.1  # error < 0.1%
    True
    """
    _validate_non_negative(i0, "Exchange current i0")
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    is_scalar = np.isscalar(eta)
    eta_arr = np.asarray(eta, dtype=float)

    slope = (n * F) / (R * T)
    sign = 1.0 if anodic_positive else -1.0
    val = sign * i0 * slope * eta_arr

    return _format_output(val, is_scalar)


# ==============================================================================
# 3. High-Overpotential Kinetics: Tafel Equation & Analysis
# ==============================================================================


def tafel_slope(
    alpha: float = 0.5,
    n: float = 1,
    branch: Literal["anodic", "cathodic"] = "anodic",
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    signed: bool = True,
) -> float:
    r"""Compute theoretical Tafel slope :math:`b` in Volts per decade (:math:`\text{V}/\text{dec}`).

    Calculates the theoretical logarithmic voltage-current sensitivity parameter:

    - **Anodic branch** (:math:`\eta \gg 0`):

      .. math::

          b_a = \frac{\ln(10) R T}{(1 - \alpha) n F}

    - **Cathodic branch** (:math:`\eta \ll 0`):

      .. math::

          b_c = -\frac{\ln(10) R T}{\alpha n F} \quad (\text{if signed=True})

    Parameters
    ----------
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    branch : {'anodic', 'cathodic'}, default 'anodic'
        Kinetic polarization branch to evaluate.
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).
    signed : bool, default True
        If ``True``, cathodic Tafel slope is returned as negative (:math:`b_c < 0`).
        If ``False``, the absolute magnitude :math:`|b_c|` is returned.

    Returns
    -------
    float
        Theoretical Tafel slope :math:`b` in :math:`\text{V}/\text{decade}`.

    Raises
    ------
    ValueError
        If ``branch`` is not ``'anodic'`` or ``'cathodic'``, or if parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import tafel_slope
    >>> # Standard 1-electron transfer at 25 °C with alpha = 0.5: ~118.3 mV/dec
    >>> round(tafel_slope(alpha=0.5, n=1, branch="anodic") * 1000, 1)
    118.3
    >>> round(tafel_slope(alpha=0.5, n=1, branch="cathodic", signed=True) * 1000, 1)
    -118.3
    """
    _validate_alpha(alpha)
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    branch_norm = branch.lower().strip()
    if branch_norm == "anodic":
        return float((_LN10 * R * T) / ((1.0 - alpha) * n * F))
    elif branch_norm == "cathodic":
        mag = (_LN10 * R * T) / (alpha * n * F)
        return float(-mag if signed else mag)
    else:
        raise ValueError(
            f"Invalid branch '{branch}'. Must be either 'anodic' or 'cathodic'."
        )


def tafel(
    eta: float | ArrayLike,
    i0: float = 1.0,
    alpha: float = 0.5,
    branch: Literal["auto", "anodic", "cathodic"] = "auto",
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    anodic_positive: bool = True,
) -> float | np.ndarray:
    r"""Compute high-overpotential Tafel current approximation.

    At sufficiently high overpotentials (:math:`|\eta| > \frac{R T}{n F}`), the
    opposing reaction direction becomes negligible, reducing the Butler–Volmer equation
    to a single exponential branch:

    - **Anodic** (:math:`\eta \gg 0`):

      .. math::

          I_a = I_0 \exp\left( \frac{(1 - \alpha) n F \eta}{R T} \right)

    - **Cathodic** (:math:`\eta \ll 0`):

      .. math::

          |I_c| = I_0 \exp\left( -\frac{\alpha n F \eta}{R T} \right)

    Parameters
    ----------
    eta : float or array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts (V).
    i0 : float, default 1.0
        Exchange current :math:`I_0` in Amperes (A) or current density (:math:`I_0 \ge 0`).
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    branch : {'auto', 'anodic', 'cathodic'}, default 'auto'
        Branch to evaluate. If ``'auto'``, evaluates the anodic branch for :math:`\eta \ge 0`
        and the cathodic branch for :math:`\eta < 0`.
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).
    anodic_positive : bool, default True
        If ``True``, anodic current is positive and cathodic current is negative.

    Returns
    -------
    float or np.ndarray
        Tafel current approximation in Amperes (A) or current density.

    Raises
    ------
    ValueError
        If ``branch`` is invalid or if physical parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import tafel
    >>> # Overpotential of 118.2 mV increases current by 1 decade above I0
    >>> round(tafel(0.11822, i0=1.0, alpha=0.5, n=1, branch="anodic"), 1)
    10.0
    """
    _validate_non_negative(i0, "Exchange current i0")
    _validate_alpha(alpha)
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    is_scalar = np.isscalar(eta)
    eta_arr = np.asarray(eta, dtype=float)

    f_factor = (n * F) / (R * T)
    branch_norm = branch.lower().strip()

    sign_a = 1.0 if anodic_positive else -1.0
    sign_c = -1.0 if anodic_positive else 1.0

    if branch_norm == "anodic":
        with np.errstate(over="ignore"):
            val = sign_a * i0 * np.exp((1.0 - alpha) * f_factor * eta_arr)
    elif branch_norm == "cathodic":
        with np.errstate(over="ignore"):
            val = sign_c * i0 * np.exp(-alpha * f_factor * eta_arr)
    elif branch_norm == "auto":
        with np.errstate(over="ignore"):
            i_a = sign_a * i0 * np.exp((1.0 - alpha) * f_factor * eta_arr)
            i_c = sign_c * i0 * np.exp(-alpha * f_factor * eta_arr)
        val = np.where(eta_arr >= 0.0, i_a, i_c)
    else:
        raise ValueError(
            f"Invalid branch '{branch}'. Must be 'auto', 'anodic', or 'cathodic'."
        )

    return _format_output(val, is_scalar)


def tafel_overpotential(
    current: float | ArrayLike,
    i0: float = 1.0,
    alpha: float = 0.5,
    branch: Literal["anodic", "cathodic"] = "anodic",
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute overpotential :math:`\eta` from current using the Tafel approximation.

    Inverts the Tafel relationship to determine the activation overpotential
    required to sustain a given Faradaic current :math:`|I|`:

    .. math::

        \eta = b \log_{10}\left( \frac{|I|}{I_0} \right)

    Parameters
    ----------
    current : float or array-like
        Faradaic current or current density (:math:`|I| > 0`).
    i0 : float, default 1.0
        Exchange current :math:`I_0 > 0` in matching units.
    alpha : float, default 0.5
        Apparent cathodic charge-transfer coefficient (:math:`0 < \alpha < 1`).
    branch : {'anodic', 'cathodic'}, default 'anodic'
        Polarization branch ('anodic' or 'cathodic').
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    float or np.ndarray
        Activation overpotential :math:`\eta` in Volts (V).

    Raises
    ------
    ValueError
        If current or :math:`I_0` are non-positive, or if parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.kinetics import tafel_overpotential
    >>> # 1 decade of current above I0 requires ~118.3 mV overpotential (n=1, alpha=0.5)
    >>> eta = tafel_overpotential(10.0, i0=1.0, alpha=0.5, branch="anodic")
    >>> round(eta * 1000, 1)
    118.3
    """
    _validate_positive(i0, "Exchange current i0")
    b = tafel_slope(alpha=alpha, n=n, branch=branch, T=T, R=R, F=F, signed=True)

    is_scalar = np.isscalar(current)
    cur_arr = np.asarray(current, dtype=float)

    if np.any(cur_arr <= 0.0):
        # Allow negative signed currents by taking absolute value, but reject zeros
        abs_cur = np.abs(cur_arr)
        if np.any(abs_cur == 0.0):
            raise ValueError(
                "Current must be non-zero to evaluate Tafel overpotential."
            )
    else:
        abs_cur = cur_arr

    eta_out = b * np.log10(abs_cur / i0)
    return _format_output(eta_out, is_scalar)


@dataclass(frozen=True)
class TafelResult:
    r"""Container for Tafel diagnostic linear regression parameters.

    Parameters
    ----------
    branch : str
        Tafel polarization branch analyzed (``'anodic'`` or ``'cathodic'``).
    i0 : float
        Extracted exchange current :math:`I_0` in Amperes (A) or current density
        in :math:`\text{A}/\text{area}`.
    slope : float
        Fitted Tafel slope :math:`b = \partial \eta / \partial \log_{10}|I|`
        in :math:`\text{V}/\text{decade}`.
    slope_mv : float
        Fitted Tafel slope in :math:`\text{mV}/\text{decade}` (:math:`b \times 1000`).
    alpha : float
        Extracted apparent charge-transfer coefficient :math:`\alpha` (dimensionless).
    intercept : float
        Regression intercept :math:`a` of :math:`\eta = a + b \log_{10}|I|` in Volts (V).
    r_squared : float
        Coefficient of determination :math:`R^2` of the linear regression fit.
    stderr : float
        Standard error of the regression slope :math:`\partial \log_{10}|I| / \partial \eta`
        in :math:`\text{V}^{-1}`.
    intercept_stderr : float
        Standard error of the regression intercept in :math:`\log_{10}(\text{A})`.
    k0 : float or None, default None
        Extracted apparent standard heterogeneous rate constant :math:`k^0`
        in :math:`\text{m}/\text{s}` or :math:`\text{cm}/\text{s}`, computed if ``area``
        and ``c_bulk`` were provided.
    """

    branch: str
    i0: float
    slope: float
    slope_mv: float
    alpha: float
    intercept: float
    r_squared: float
    stderr: float
    intercept_stderr: float
    k0: float | None = None


def tafel_analysis(
    overpotential: float | ArrayLike,
    current: float | ArrayLike,
    *,
    branch: Literal["anodic", "cathodic"] = "anodic",
    fit_range: tuple[float, float] | None = None,
    area: float | None = None,
    c_bulk: float | None = None,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> TafelResult:
    r"""Perform linear Tafel regression on overpotential-current data.

    Applies linear regression to the high-overpotential logarithmic relationship:

    .. math::

        \log_{10}|I| = \log_{10}(I_0) + \frac{1}{b} \eta

    to extract:

    1. The exchange current :math:`I_0 = 10^{\text{intercept}}`.
    2. The experimental Tafel slope :math:`b = 1 / \text{slope}` in :math:`\text{V}/\text{decade}`
       and :math:`\text{mV}/\text{decade}`.
    3. The apparent charge-transfer coefficient :math:`\alpha`.
    4. The standard rate constant :math:`k^0` (if ``area`` and ``c_bulk`` are supplied).

    Parameters
    ----------
    overpotential : array-like
        Electrode overpotential :math:`\eta = E - E_{\text{eq}}` in Volts (V).
    current : array-like
        Measured Faradaic current in Amperes (A) or current density.
    branch : {'anodic', 'cathodic'}, default 'anodic'
        Kinetic branch to isolate for regression:
        - ``'anodic'``: Fits positive overpotentials (:math:`\eta > 0`).
        - ``'cathodic'``: Fits negative overpotentials (:math:`\eta < 0`).
    fit_range : tuple of (float, float), optional
        Potential window ``(eta_min, eta_max)`` over which to execute the linear fit.
        Recommended to select the activation-dominated high-field region
        (:math:`|\eta| \in [0.10, 0.30]\,\text{V}`) while avoiding mass-transfer limitations.
    area : float, optional
        Electrode surface area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    c_bulk : float, optional
        Bulk electroactive species concentration in :math:`\text{mol}/\text{m}^3`
        or :math:`\text{mol}/\text{cm}^3` (:math:`c^* > 0`).
    n : int or float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default STANDARD_TEMPERATURE
        Temperature in Kelvin (:math:`T > 0`, default 298.15 K).
    R : float, default GAS_CONSTANT
        Molar gas constant (:math:`R \approx 8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`).
    F : float, default FARADAY
        Faraday constant (:math:`F \approx 96485.332\,\text{C}\cdot\text{mol}^{-1}`).

    Returns
    -------
    TafelResult
        Container holding fitted kinetic and regression parameters.

    Raises
    ------
    ValueError
        If fewer than 3 valid data points exist in the selected fit window,
        or if physical parameters are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.analytical.kinetics import butler_volmer, tafel_analysis
    >>> # Generate synthetic Butler-Volmer data with I0 = 1e-4 A, alpha = 0.5
    >>> eta_pts = np.linspace(0.12, 0.25, 50)
    >>> i_pts = butler_volmer(eta_pts, i0=1e-4, alpha=0.5, n=1)
    >>> res = tafel_analysis(eta_pts, i_pts, branch="anodic")
    >>> round(res.i0 * 1e4, 2)
    0.99
    >>> round(res.alpha, 2)
    0.5
    >>> round(res.slope_mv, 1)
    118.0
    """
    _validate_n(n)
    _validate_temp(T)
    _validate_gas_constant(R)
    _validate_faraday(F)

    eta_arr = np.asarray(overpotential, dtype=float).ravel()
    i_arr = np.asarray(current, dtype=float).ravel()

    if len(eta_arr) != len(i_arr):
        raise ValueError(
            f"Length mismatch: overpotential has {len(eta_arr)} elements, "
            f"current has {len(i_arr)} elements."
        )

    branch_norm = branch.lower().strip()
    if branch_norm not in ("anodic", "cathodic"):
        raise ValueError(
            f"Invalid branch '{branch}'. Must be either 'anodic' or 'cathodic'."
        )

    # Filter by branch default direction
    if branch_norm == "anodic":
        mask = eta_arr > 0.0
    else:
        mask = eta_arr < 0.0

    # Apply optional user fit_range
    if fit_range is not None:
        eta_min, eta_max = fit_range
        if eta_min >= eta_max:
            raise ValueError(
                f"fit_range must satisfy eta_min < eta_max, got ({eta_min}, {eta_max})."
            )
        mask = mask & (eta_arr >= eta_min) & (eta_arr <= eta_max)

    # Filter positive currents for logarithm
    abs_i = np.abs(i_arr)
    mask = mask & (abs_i > 0.0) & np.isfinite(eta_arr) & np.isfinite(abs_i)

    eta_fit = eta_arr[mask]
    abs_i_fit = abs_i[mask]

    if len(eta_fit) < 3:
        raise ValueError(
            f"At least 3 valid data points are required for Tafel linear regression, "
            f"found {len(eta_fit)} in the specified branch/fit_range."
        )

    log_i_fit = np.log10(abs_i_fit)

    # Regress log10|I| vs eta: log10|I| = m * eta + q
    res = linregress(eta_fit, log_i_fit)
    m = float(res.slope)
    q = float(res.intercept)
    r_val = float(res.rvalue)
    stderr_m = float(res.stderr) if res.stderr is not None else 0.0
    stderr_q = float(res.intercept_stderr) if res.intercept_stderr is not None else 0.0

    # Physical derivations
    i0_extracted = float(10.0**q)
    b_slope = float(1.0 / m)
    b_slope_mv = float(b_slope * 1000.0)

    # Tafel slope equation:
    # b_a = ln(10)*R*T / ((1 - alpha)*n*F) => (1 - alpha) = ln(10)*R*T*m / (n*F)
    # b_c = -ln(10)*R*T / (alpha*n*F)     => alpha = -ln(10)*R*T*m / (n*F)
    f_factor = (n * F) / (_LN10 * R * T)

    if branch_norm == "anodic":
        alpha_extracted = float(1.0 - (m / f_factor))
    else:
        alpha_extracted = float(-m / f_factor)

    # Intercept in eta = a + b * log10|I| formulation:
    # eta = (1/m)*log10|I| - (q/m) => a = -q/m = -b * log10(i0)
    intercept_eta = float(-q / m)

    # Standard rate constant extraction if area and bulk concentration are provided
    k0_extracted: float | None = None
    if area is not None and c_bulk is not None:
        _validate_positive(area, "Electrode area")
        _validate_positive(c_bulk, "Bulk concentration c_bulk")
        # j0 = n * F * k0 * c_bulk => k0 = I0 / (n * F * area * c_bulk)
        k0_extracted = float(i0_extracted / (n * F * area * c_bulk))

    return TafelResult(
        branch=branch_norm,
        i0=i0_extracted,
        slope=b_slope,
        slope_mv=b_slope_mv,
        alpha=alpha_extracted,
        intercept=intercept_eta,
        r_squared=float(r_val**2),
        stderr=stderr_m,
        intercept_stderr=stderr_q,
        k0=k0_extracted,
    )


__all__ = [
    "TafelResult",
    "butler_volmer",
    "butler_volmer_current_density",
    "butler_volmer_linear",
    "charge_transfer_resistance",
    "exchange_current",
    "exchange_current_density",
    "nernst",
    "nernst_equilibrium_concentrations",
    "nernst_potential",
    "nernst_ratio",
    "tafel",
    "tafel_analysis",
    "tafel_overpotential",
    "tafel_slope",
]
