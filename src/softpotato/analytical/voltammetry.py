r"""Analytical and empirical equations for voltammetry and kinetic diagnostics.

This module provides closed-form equations, asymptotic expansions, empirical approximations,
and diagnostic tools for cyclic voltammetry (CV) and linear sweep voltammetry (LSV)
at planar electrodes under semi-infinite linear diffusion.

Functions & Classes
-------------------
randles_sevcik
    Reversible electron transfer peak current (Randles–Ševčík equation).
randles_sevcik_irreversible
    Totally irreversible electron transfer peak current.
randles_sevcik_quasi
    Quasi-reversible electron transfer peak current and shape factor approximations.
nicholson_psi
    Nicholson method relating peak potential separation (Delta Ep) to kinetic parameter Psi and k0.
nicholson_delta_ep
    Forward Nicholson approximation predicting peak separation from Psi.
nicholson_rate_constant
    Direct extraction of standard rate constant k0 from peak potential separation.
matsuda_ayabe
    Reversibility criteria classification based on the dimensionless Matsuda–Ayabe parameter Lambda.
matsuda_ayabe_lambda
    Compute the dimensionless kinetic parameter Lambda.
peak_potential_irreversible
    Peak potential and scan-rate dependence for totally irreversible systems.
NicholsonResult
    Container dataclass for Nicholson cyclic voltammetry kinetic diagnostic parameters.
MatsudaAyabeResult
    Container dataclass for Matsuda–Ayabe reversibility classification and diagnostic parameters.

Conventions & Units
-------------------
All equations are dimensionally consistent in either standard SI or standard
electrochemical CGS units:

- **SI**:
  :math:`v` in :math:`\text{V}\cdot\text{s}^{-1}`,
  :math:`c^*` in :math:`\text{mol}\cdot\text{m}^{-3}`,
  :math:`A` in :math:`\text{m}^2`,
  :math:`D` in :math:`\text{m}^2\cdot\text{s}^{-1}`,
  :math:`k^0` in :math:`\text{m}\cdot\text{s}^{-1}`,
  :math:`F` in :math:`\text{C}\cdot\text{mol}^{-1}`
  :math:`\implies I_p` in :math:`\text{A}`.
- **CGS**:
  :math:`v` in :math:`\text{V}\cdot\text{s}^{-1}`,
  :math:`c^*` in :math:`\text{mol}\cdot\text{cm}^{-3}`,
  :math:`A` in :math:`\text{cm}^2`,
  :math:`D` in :math:`\text{cm}^2\cdot\text{s}^{-1}`,
  :math:`k^0` in :math:`\text{cm}\cdot\text{s}^{-1}`,
  :math:`F` in :math:`\text{C}\cdot\text{mol}^{-1}`
  :math:`\implies I_p` in :math:`\text{A}`.

Standard IUPAC sign conventions are applied:
- Cathodic (reduction) peak currents are positive (:math:`I_p > 0`).
- Anodic (oxidation) peak currents are negative (:math:`I_p < 0`) when ``scan_direction="anodic"``.

References
----------
- A. J. Bard, L. R. Faulkner, H. S. White, *Electrochemical Methods:
  Fundamentals and Applications*, 3rd ed., John Wiley & Sons, 2022, Chapter 6.
- J. E. B. Randles, "A cathode ray polarograph. Part II. The current-voltage curves",
  *Trans. Faraday Soc.*, 44 (1948) 327–338.
- A. Ševčík, "Oscillographic polarography with periodical triangular voltage",
  *Collect. Czech. Chem. Commun.*, 13 (1948) 349–377.
- H. Matsuda, Y. Ayabe, "Zur Theorie der Randles-Sevcikschen Kathodenstrahl-Polarographie",
  *Z. Elektrochem.*, 59 (1955) 494–503.
- R. S. Nicholson, "Theory and application of cyclic voltammetry for measurement of
  electrode reaction kinetics", *Anal. Chem.*, 37 (1965) 1351–1355.
- I. Lavagnini, R. Antiochia, F. Magno, "An extended method for the practical evaluation
  of the standard rate constant from cyclic voltammetric data", *Electroanalysis*, 16 (2004) 505–506.
- T. W. Swaddle, "Homogeneous versus heterogeneous self-exchange electron transfer reactions
  of metal complexes: Insights from pressure effects", *Chem. Rev.*, 105 (2005) 2573–2608.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Tabulated Nicholson (1965) Table 1 data: psi vs. n * Delta_Ep (mV)
_NICHOLSON_PSI_TABLE = np.array(
    [20.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.50, 0.35, 0.25, 0.10],
    dtype=float,
)
_NICHOLSON_DELTA_EP_MV_TABLE = np.array(
    [61.0, 63.0, 64.0, 65.0, 66.0, 68.0, 72.0, 84.0, 92.0, 105.0, 121.0, 141.0, 212.0],
    dtype=float,
)

# Monotonic interpolators in log(psi) space
_PCHIP_DELTA_EP_TO_LOG_PSI = PchipInterpolator(
    _NICHOLSON_DELTA_EP_MV_TABLE, np.log(_NICHOLSON_PSI_TABLE)
)
_PCHIP_LOG_PSI_TO_DELTA_EP = PchipInterpolator(
    np.log(_NICHOLSON_PSI_TABLE)[::-1], _NICHOLSON_DELTA_EP_MV_TABLE[::-1]
)


def _validate_positive(val: float, name: str) -> None:
    """Validate that a physical parameter is strictly positive."""
    if val <= 0:
        raise ValueError(f"{name} must be positive ({name} > 0), got {val}.")


def _validate_non_negative(val: float, name: str) -> None:
    """Validate that a physical parameter is non-negative."""
    if val < 0:
        raise ValueError(f"{name} must be non-negative ({name} >= 0), got {val}.")


def _validate_alpha(alpha: float) -> None:
    """Validate transfer coefficient alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Transfer coefficient alpha must be strictly between 0 and 1 (0 < alpha < 1), got {alpha}."
        )


def _format_output(arr: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    """Format NumPy array output as float if input was scalar."""
    if is_scalar:
        return float(arr.item())
    return arr


# ==============================================================================
# Container Dataclasses
# ==============================================================================


@dataclass(frozen=True)
class NicholsonResult:
    r"""Container for Nicholson cyclic voltammetry kinetic diagnostic parameters.

    Parameters
    ----------
    psi : float
        Dimensionless Nicholson kinetic parameter :math:`\Psi`.
    delta_ep : float
        Peak potential separation :math:`\Delta E_p` in Volts.
    delta_ep_mv : float
        Peak potential separation :math:`\Delta E_p` in millivolts.
    k0 : float or None, default None
        Extracted heterogeneous standard electron transfer rate constant :math:`k^0`
        in :math:`\text{m}\cdot\text{s}^{-1}` (or :math:`\text{cm}\cdot\text{s}^{-1}`).
    method : str, default "swaddle"
        Method used for the evaluation (``"swaddle"``, ``"lavagnini"``, or ``"table"``).
    n : float, default 1.0
        Number of electrons transferred.
    """

    psi: float
    delta_ep: float
    delta_ep_mv: float
    k0: float | None = None
    method: str = "swaddle"
    n: float = 1.0


@dataclass(frozen=True)
class MatsudaAyabeResult:
    r"""Container for Matsuda–Ayabe reversibility classification and diagnostics.

    Parameters
    ----------
    lambda_param : float
        Dimensionless Matsuda–Ayabe kinetic parameter :math:`\Lambda`.
    classification : {"reversible", "quasi-reversible", "irreversible"}
        Electrochemical reversibility classification zone.
    k0 : float
        Standard heterogeneous rate constant :math:`k^0`.
    v : float
        Potential scan rate :math:`v`.
    D : float
        Diffusion coefficient :math:`D`.
    is_reversible : bool
        True if system is reversible (:math:`\Lambda \ge 15`).
    is_quasi_reversible : bool
        True if system is quasi-reversible (:math:`10^{-3} < \Lambda < 15`).
    is_irreversible : bool
        True if system is irreversible (:math:`\Lambda \le 10^{-3}`).
    """

    lambda_param: float
    classification: Literal["reversible", "quasi-reversible", "irreversible"]
    k0: float
    v: float
    D: float
    is_reversible: bool
    is_quasi_reversible: bool
    is_irreversible: bool


# ==============================================================================
# 1. Randles–Ševčík Reversible Peak Current
# ==============================================================================


def randles_sevcik(
    v: float | ArrayLike,
    c_bulk: float = 1e-3,
    D: float = 1e-5,
    area: float = 1.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    scan_direction: Literal["cathodic", "anodic"] = "cathodic",
) -> float | np.ndarray:
    r"""Compute reversible electron transfer peak current using the Randles–Ševčík equation.

    Calculates the Faradaic peak current :math:`I_p` for a diffusion-controlled,
    electrochemically reversible (Nernstian) single-step electron transfer at a planar
    electrode under semi-infinite linear diffusion:

    .. math::

        I_p = 0.4463 \, n F A c^* \sqrt{\frac{n F D v}{R T}}

    At standard temperature (:math:`T = 298.15\,\text{K}`):

    .. math::

        I_p \approx (2.686 \times 10^5) \, n^{3/2} A D^{1/2} c^* v^{1/2}

    Parameters
    ----------
    v : float or array-like
        Potential sweep scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}\cdot\text{m}^{-3}`
        or :math:`\text{mol}\cdot\text{cm}^{-3}` (:math:`c^* \ge 0`).
    D : float, default 1e-5
        Diffusion coefficient of electroactive species in :math:`\text{m}^2\cdot\text{s}^{-1}`
        or :math:`\text{cm}^2\cdot\text{s}^{-1}` (:math:`D > 0`).
    area : float, default 1.0
        Electrode geometric surface area :math:`A` in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    n : float, default 1
        Number of electrons transferred per molecule (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant in :math:`\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}` (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant in :math:`\text{C}\cdot\text{mol}^{-1}` (:math:`F > 0`).
    scan_direction : {"cathodic", "anodic"}, default "cathodic"
        Direction of potential sweep. Cathodic (reduction) yields positive current (:math:`I_p > 0`),
        while anodic (oxidation) yields negative current (:math:`I_p < 0`).

    Returns
    -------
    float or numpy.ndarray
        Peak current :math:`I_p` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If physical parameters are non-positive or ``scan_direction`` is invalid.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import randles_sevcik
    >>> # Reversible 1-electron reduction at 0.1 V/s, 1 mM, 0.07 cm^2, D = 1e-5 cm^2/s
    >>> ip = randles_sevcik(0.1, c_bulk=1e-6, D=1e-5, area=0.07)
    >>> round(ip * 1e6, 2)  # Current in microamperes
    18.81
    """
    _validate_non_negative(c_bulk, "Bulk concentration c_bulk")
    _validate_positive(D, "Diffusion coefficient D")
    _validate_positive(area, "Electrode area")
    _validate_positive(n, "Number of electrons n")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")

    if scan_direction not in ("cathodic", "anodic"):
        raise ValueError(
            f"Invalid scan_direction: {scan_direction!r}. Must be 'cathodic' or 'anodic'."
        )

    v_arr = np.asarray(v, dtype=float)
    if np.any(v_arr <= 0):
        raise ValueError(f"Scan rate v must be strictly positive (v > 0), got {v!r}.")

    prefactor = 0.4463 * n * F * area * c_bulk * np.sqrt((n * F * D * v_arr) / (R * T))
    if scan_direction == "anodic":
        prefactor = -prefactor

    return _format_output(prefactor, np.ndim(v) == 0)


# ==============================================================================
# 2. Randles–Ševčík Irreversible Peak Current
# ==============================================================================


def randles_sevcik_irreversible(
    v: float | ArrayLike,
    alpha: float = 0.5,
    n_alpha: float = 1,
    c_bulk: float = 1e-3,
    D: float = 1e-5,
    area: float = 1.0,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    scan_direction: Literal["cathodic", "anodic"] = "cathodic",
) -> float | np.ndarray:
    r"""Compute totally irreversible electron transfer peak current.

    Calculates the Faradaic peak current :math:`I_p` for a totally irreversible
    electron transfer at a planar electrode under semi-infinite linear diffusion:

    .. math::

        I_p = 0.4958 \, n F A c^* \sqrt{\frac{\alpha n_\alpha F D v}{R T}}

    Parameters
    ----------
    v : float or array-like
        Potential sweep scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    alpha : float, default 0.5
        Charge transfer coefficient (:math:`0 < \alpha < 1`).
    n_alpha : float, default 1
        Number of electrons transferred in the rate-determining step (:math:`n_\alpha > 0`).
    c_bulk : float, default 1e-3
        Bulk concentration of electroactive species :math:`c^*` in :math:`\text{mol}\cdot\text{m}^{-3}`
        or :math:`\text{mol}\cdot\text{cm}^{-3}` (:math:`c^* \ge 0`).
    D : float, default 1e-5
        Diffusion coefficient in :math:`\text{m}^2\cdot\text{s}^{-1}` or :math:`\text{cm}^2\cdot\text{s}^{-1}` (:math:`D > 0`).
    area : float, default 1.0
        Electrode geometric surface area :math:`A` in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    n : float, default 1
        Total number of electrons transferred per reactant molecule (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant in :math:`\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}` (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant in :math:`\text{C}\cdot\text{mol}^{-1}` (:math:`F > 0`).
    scan_direction : {"cathodic", "anodic"}, default "cathodic"
        Direction of potential sweep. Cathodic (reduction) yields positive current (:math:`I_p > 0`),
        while anodic (oxidation) yields negative current (:math:`I_p < 0`).

    Returns
    -------
    float or numpy.ndarray
        Totally irreversible peak current :math:`I_p` in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If parameters are invalid or outside physical bounds.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import randles_sevcik_irreversible
    >>> ip_irrev = randles_sevcik_irreversible(0.1, alpha=0.5, c_bulk=1e-6, D=1e-5, area=0.07)
    >>> round(ip_irrev * 1e6, 2)
    14.77
    """
    _validate_alpha(alpha)
    _validate_positive(n_alpha, "Number of electrons n_alpha")
    _validate_non_negative(c_bulk, "Bulk concentration c_bulk")
    _validate_positive(D, "Diffusion coefficient D")
    _validate_positive(area, "Electrode area")
    _validate_positive(n, "Number of electrons n")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")

    if scan_direction not in ("cathodic", "anodic"):
        raise ValueError(
            f"Invalid scan_direction: {scan_direction!r}. Must be 'cathodic' or 'anodic'."
        )

    v_arr = np.asarray(v, dtype=float)
    if np.any(v_arr <= 0):
        raise ValueError(f"Scan rate v must be strictly positive (v > 0), got {v!r}.")

    prefactor = (
        0.4958
        * n
        * F
        * area
        * c_bulk
        * np.sqrt((alpha * n_alpha * F * D * v_arr) / (R * T))
    )
    if scan_direction == "anodic":
        prefactor = -prefactor

    return _format_output(prefactor, np.ndim(v) == 0)


# ==============================================================================
# 3. Randles–Ševčík Quasi-Reversible Peak Current
# ==============================================================================


def randles_sevcik_quasi(
    v: float | ArrayLike,
    k0: float | None = None,
    *,
    lambda_param: float | ArrayLike | None = None,
    c_bulk: float = 1e-3,
    D: float = 1e-5,
    area: float = 1.0,
    alpha: float = 0.5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    scan_direction: Literal["cathodic", "anodic"] = "cathodic",
) -> float | np.ndarray:
    r"""Compute quasi-reversible electron transfer peak current and shape factor.

    Approximates the peak current in the intermediate quasi-reversible regime
    (:math:`10^{-3} < \Lambda < 15`) using the Matsuda–Ayabe current function
    correction factor :math:`K(\Lambda, \alpha)`:

    .. math::

        I_{p,\text{quasi}} = I_{p,\text{rev}} \cdot K(\Lambda, \alpha)

    where :math:`I_{p,\text{rev}}` is the reversible Randles–Ševčík peak current,
    and :math:`K(\Lambda, \alpha)` transitions smoothly between the reversible limit
    (:math:`K \to 1.0` as :math:`\Lambda \to \infty`) and the totally irreversible limit
    (:math:`K \to \frac{0.4958}{0.4463} \sqrt{\alpha} \approx 1.1109 \sqrt{\alpha}` as :math:`\Lambda \to 0`).

    Parameters
    ----------
    v : float or array-like
        Potential sweep scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    k0 : float, optional
        Standard heterogeneous rate constant :math:`k^0` in :math:`\text{m}\cdot\text{s}^{-1}`
        or :math:`\text{cm}\cdot\text{s}^{-1}` (:math:`k^0 > 0`).
        Either ``k0`` or ``lambda_param`` must be supplied.
    lambda_param : float or array-like, optional
        Dimensionless Matsuda–Ayabe kinetic parameter :math:`\Lambda > 0`.
    c_bulk : float, default 1e-3
        Bulk concentration in :math:`\text{mol}\cdot\text{m}^{-3}` or :math:`\text{mol}\cdot\text{cm}^{-3}` (:math:`c^* \ge 0`).
    D : float, default 1e-5
        Diffusion coefficient in :math:`\text{m}^2\cdot\text{s}^{-1}` or :math:`\text{cm}^2\cdot\text{s}^{-1}` (:math:`D > 0`).
    area : float, default 1.0
        Electrode geometric area in :math:`\text{m}^2` or :math:`\text{cm}^2` (:math:`A > 0`).
    alpha : float, default 0.5
        Charge transfer coefficient (:math:`0 < \alpha < 1`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).
    scan_direction : {"cathodic", "anodic"}, default "cathodic"
        Direction of sweep ("cathodic" or "anodic").

    Returns
    -------
    float or numpy.ndarray
        Quasi-reversible peak current in Amperes (:math:`\text{A}`).

    Raises
    ------
    ValueError
        If neither or both ``k0`` and ``lambda_param`` are provided, or parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import randles_sevcik_quasi
    >>> ip_quasi = randles_sevcik_quasi(0.1, k0=1e-3, c_bulk=1e-6, D=1e-5, area=0.07)
    >>> round(ip_quasi * 1e6, 2)
    15.3
    """
    _validate_alpha(alpha)
    _validate_non_negative(c_bulk, "Bulk concentration c_bulk")
    _validate_positive(D, "Diffusion coefficient D")
    _validate_positive(area, "Electrode area")
    _validate_positive(n, "Number of electrons n")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")

    if k0 is None and lambda_param is None:
        raise ValueError("Either 'k0' or 'lambda_param' must be provided.")

    v_arr = np.asarray(v, dtype=float)
    if np.any(v_arr <= 0):
        raise ValueError(f"Scan rate v must be strictly positive (v > 0), got {v!r}.")

    if lambda_param is not None:
        lam = np.asarray(lambda_param, dtype=float)
        if np.any(lam <= 0):
            raise ValueError(
                f"Dimensionless parameter lambda_param must be positive (lambda > 0), got {lambda_param!r}."
            )
    else:
        assert k0 is not None
        _validate_positive(k0, "Standard rate constant k0")
        lam = k0 / np.sqrt((D * n * F * v_arr) / (R * T))

    # Compute reversible peak current baseline
    i_rev = randles_sevcik(
        v=v_arr,
        c_bulk=c_bulk,
        D=D,
        area=area,
        n=n,
        T=T,
        R=R,
        F=F,
        scan_direction=scan_direction,
    )

    # Matsuda-Ayabe rational shape factor connecting irrev and rev limits
    # As Lambda -> 0: K -> (0.4958 / 0.4463) * sqrt(alpha)
    # As Lambda -> inf: K -> 1.0
    k_irrev = (0.4958 / 0.4463) * math.sqrt(alpha)
    c1 = 0.77
    k_factor = (k_irrev + c1 * lam + lam**2) / (1.0 + c1 * lam + lam**2)

    val = i_rev * k_factor
    is_scalar = (np.ndim(v) == 0) and (
        lambda_param is None or np.ndim(lambda_param) == 0
    )
    return _format_output(np.asarray(val), is_scalar)


# ==============================================================================
# 4. Nicholson Kinetic Diagnostics (nicholson_psi, etc.)
# ==============================================================================


def nicholson_psi(
    delta_ep: float | ArrayLike | None = None,
    *,
    psi: float | ArrayLike | None = None,
    k0: float | None = None,
    v: float | None = None,
    D: float = 1e-5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    in_volts: bool = True,
    method: Literal["swaddle", "lavagnini", "table"] = "swaddle",
) -> float | np.ndarray | NicholsonResult:
    r"""Relate peak potential separation to Nicholson kinetic parameter and standard rate constant.

    Implements the Nicholson method relating peak-to-peak potential separation
    :math:`\Delta E_p = |E_{pa} - E_{pc}|` to the dimensionless kinetic parameter :math:`\Psi`:

    .. math::

        \Psi = \frac{k^0}{\sqrt{\pi D \frac{n F v}{R T}}}

    Supports both forward (:math:`\Psi \to \Delta E_p`) and inverse
    (:math:`\Delta E_p \to \Psi`) evaluations, with optional extraction of the standard
    rate constant :math:`k^0` when scan rate :math:`v` is provided.

    Available Approximation Methods
    --------------------------------
    - ``"swaddle"`` (default): T. W. Swaddle (2005) empirical rational formulation:

      .. math::

          \ln \Psi = 3.69 - 1.16 \ln(n \Delta E_p[\text{mV}] - 59)

    - ``"lavagnini"``: I. Lavagnini et al. (2004) empirical formula:

      .. math::

          \Psi = \frac{-0.6288 + 0.0021 (n \Delta E_p[\text{mV}])}{1 - 0.017 (n \Delta E_p[\text{mV}])}

    - ``"table"``: High-precision monotonic Pchip interpolation against the original
      tabulated values from Nicholson (1965) Table 1.

    Parameters
    ----------
    delta_ep : float or array-like, optional
        Peak potential separation :math:`\Delta E_p`. In Volts if ``in_volts=True`` (default),
        or in millivolts if ``in_volts=False``.
    psi : float or array-like, optional
        Dimensionless Nicholson kinetic parameter :math:`\Psi > 0`. If supplied, performs
        forward calculation to predict :math:`\Delta E_p`.
    k0 : float, optional
        Standard rate constant :math:`k^0` in :math:`\text{m}\cdot\text{s}^{-1}` or :math:`\text{cm}\cdot\text{s}^{-1}`.
    v : float, optional
        Potential scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`). When provided alongside
        ``delta_ep``, enables extraction of :math:`k^0`.
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).
    in_volts : bool, default True
        If True, ``delta_ep`` input/output is in Volts. If False, in millivolts (mV).
    method : {"swaddle", "lavagnini", "table"}, default "swaddle"
        Approximation model for the Nicholson working curve.

    Returns
    -------
    float, numpy.ndarray, or NicholsonResult
        - If ``psi`` is provided: returns predicted :math:`\Delta E_p` (in V or mV).
        - If ``delta_ep`` is provided without ``v``: returns dimensionless :math:`\Psi`.
        - If ``delta_ep`` and ``v`` are both provided (scalar): returns a :class:`NicholsonResult`
          containing :math:`\Psi`, :math:`\Delta E_p`, and extracted :math:`k^0`.

    Raises
    ------
    ValueError
        If neither or both ``delta_ep`` and ``psi`` are supplied, or parameters are invalid.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import nicholson_psi
    >>> # Evaluate Psi from Delta_Ep = 72 mV for n = 1
    >>> psi_val = nicholson_psi(delta_ep=0.072, in_volts=True, method="swaddle")
    >>> round(float(psi_val), 2)
    2.04
    """
    _validate_positive(n, "Number of electrons n")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")
    _validate_positive(D, "Diffusion coefficient D")

    if method not in ("swaddle", "lavagnini", "table"):
        raise ValueError(
            f"Invalid method: {method!r}. Must be 'swaddle', 'lavagnini', or 'table'."
        )

    if delta_ep is None and psi is None:
        raise ValueError("Either 'delta_ep' or 'psi' must be provided.")
    if delta_ep is not None and psi is not None:
        raise ValueError("Cannot provide both 'delta_ep' and 'psi'.")

    # Forward direction: psi -> delta_ep
    if psi is not None:
        psi_arr = np.asarray(psi, dtype=float)
        if np.any(psi_arr <= 0):
            raise ValueError(
                f"Nicholson parameter psi must be strictly positive (psi > 0), got {psi!r}."
            )

        if method == "swaddle":
            # ln(n*Delta_Ep - 59) = (3.69 - ln(psi)) / 1.16
            n_dep_mv = 59.0 + np.exp((3.69 - np.log(psi_arr)) / 1.16)
        elif method == "lavagnini":
            # Inverting Lavagnini: psi * (1 - 0.017*X) = -0.6288 + 0.0021*X
            # => X = (psi + 0.6288) / (0.017*psi + 0.0021)
            n_dep_mv = (psi_arr + 0.6288) / (0.017 * psi_arr + 0.0021)
        else:  # "table"
            # Use Pchip on log(psi)
            log_p = np.clip(
                np.log(psi_arr),
                np.log(_NICHOLSON_PSI_TABLE[-1]),
                np.log(_NICHOLSON_PSI_TABLE[0]),
            )
            n_dep_mv = _PCHIP_LOG_PSI_TO_DELTA_EP(log_p)

        dep_mv = n_dep_mv / n
        result_dep = dep_mv / 1000.0 if in_volts else dep_mv
        return _format_output(result_dep, np.ndim(psi) == 0)

    # Inverse direction: delta_ep -> psi
    assert delta_ep is not None
    dep_arr = np.asarray(delta_ep, dtype=float)
    if np.any(dep_arr <= 0):
        raise ValueError(
            f"Peak potential separation delta_ep must be positive, got {delta_ep!r}."
        )

    dep_mv = dep_arr * 1000.0 if in_volts else dep_arr
    n_dep = n * dep_mv

    if method == "swaddle":
        if np.any(n_dep <= 59.0):
            raise ValueError(
                f"For the Swaddle method, n * delta_ep (mV) must be strictly greater than 59.0 mV. Got {n_dep!r}."
            )
        psi_computed = np.exp(3.69 - 1.16 * np.log(n_dep - 59.0))
    elif method == "lavagnini":
        denom = 1.0 - 0.017 * n_dep
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError(
                f"Singularity in Lavagnini formula near n * delta_ep = {1.0 / 0.017:.2f} mV."
            )
        psi_computed = (-0.6288 + 0.0021 * n_dep) / denom
        if np.any(psi_computed <= 0):
            raise ValueError(
                f"Lavagnini approximation produced non-positive psi for n * delta_ep = {n_dep!r}."
            )
    else:  # "table"
        clamped_ndep = np.clip(
            n_dep, _NICHOLSON_DELTA_EP_MV_TABLE[0], _NICHOLSON_DELTA_EP_MV_TABLE[-1]
        )
        psi_computed = np.exp(_PCHIP_DELTA_EP_TO_LOG_PSI(clamped_ndep))

    # If v is supplied and input is scalar, return a rich NicholsonResult
    if v is not None and np.ndim(delta_ep) == 0:
        _validate_positive(v, "v")
        psi_val = float(psi_computed.item())
        # k0 = psi * sqrt(pi * D * n * F * v / (R * T))
        k0_val = psi_val * math.sqrt(math.pi * D * n * F * v / (R * T))
        dep_v = float(dep_arr.item()) if in_volts else float(dep_arr.item()) / 1000.0
        return NicholsonResult(
            psi=psi_val,
            delta_ep=dep_v,
            delta_ep_mv=dep_v * 1000.0,
            k0=k0_val,
            method=method,
            n=float(n),
        )

    return _format_output(psi_computed, np.ndim(delta_ep) == 0)


def nicholson_delta_ep(
    psi: float | ArrayLike,
    n: float = 1,
    in_volts: bool = True,
    method: Literal["swaddle", "lavagnini", "table"] = "swaddle",
) -> float | np.ndarray:
    r"""Predict peak potential separation from Nicholson dimensionless parameter Psi.

    Parameters
    ----------
    psi : float or array-like
        Dimensionless Nicholson kinetic parameter :math:`\Psi > 0`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    in_volts : bool, default True
        If True, returns :math:`\Delta E_p` in Volts. If False, in millivolts.
    method : {"swaddle", "lavagnini", "table"}, default "swaddle"
        Approximation formula.

    Returns
    -------
    float or numpy.ndarray
        Peak potential separation :math:`\Delta E_p`.
    """
    res = nicholson_psi(psi=psi, n=n, in_volts=in_volts, method=method)
    assert not isinstance(res, NicholsonResult)
    return res


def nicholson_rate_constant(
    delta_ep: float,
    v: float,
    D: float = 1e-5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    in_volts: bool = True,
    method: Literal["swaddle", "lavagnini", "table"] = "swaddle",
) -> float:
    r"""Extract heterogeneous standard rate constant k0 from peak potential separation.

    .. math::

        k^0 = \Psi(\Delta E_p) \cdot \sqrt{\pi D \frac{n F v}{R T}}

    Parameters
    ----------
    delta_ep : float
        Measured peak potential separation :math:`\Delta E_p`.
    v : float
        Potential sweep scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).
    in_volts : bool, default True
        If True, ``delta_ep`` is in Volts. If False, in millivolts.
    method : {"swaddle", "lavagnini", "table"}, default "swaddle"
        Nicholson model.

    Returns
    -------
    float
        Standard rate constant :math:`k^0` in :math:`\text{m}\cdot\text{s}^{-1}` or :math:`\text{cm}\cdot\text{s}^{-1}`.
    """
    res = nicholson_psi(
        delta_ep=delta_ep,
        v=v,
        D=D,
        n=n,
        T=T,
        R=R,
        F=F,
        in_volts=in_volts,
        method=method,
    )
    assert isinstance(res, NicholsonResult) and res.k0 is not None
    return res.k0


# ==============================================================================
# 5. Matsuda–Ayabe Reversibility Diagnostics
# ==============================================================================


def matsuda_ayabe_lambda(
    k0: float | ArrayLike,
    v: float | ArrayLike,
    D: float = 1e-5,
    n: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> float | np.ndarray:
    r"""Compute the dimensionless Matsuda–Ayabe reversibility parameter Lambda.

    .. math::

        \Lambda = \frac{k^0}{\sqrt{D \frac{n F v}{R T}}} = \sqrt{\pi} \Psi

    Parameters
    ----------
    k0 : float or array-like
        Standard rate constant :math:`k^0 > 0`.
    v : float or array-like
        Scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    float or numpy.ndarray
        Dimensionless kinetic parameter :math:`\Lambda`.
    """
    _validate_positive(D, "Diffusion coefficient D")
    _validate_positive(n, "Number of electrons n")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")

    k0_arr = np.asarray(k0, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if np.any(k0_arr <= 0):
        raise ValueError(
            f"Rate constant k0 must be strictly positive (k0 > 0), got {k0!r}."
        )
    if np.any(v_arr <= 0):
        raise ValueError(f"Scan rate v must be strictly positive (v > 0), got {v!r}.")

    denom = np.sqrt((D * n * F * v_arr) / (R * T))
    val = k0_arr / denom
    is_scalar = (np.ndim(k0) == 0) and (np.ndim(v) == 0)
    return _format_output(val, is_scalar)


def matsuda_ayabe(
    k0: float,
    v: float,
    D: float = 1e-5,
    n: float = 1,
    alpha: float = 0.5,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
) -> MatsudaAyabeResult:
    r"""Classify voltammetric reversibility according to the Matsuda–Ayabe criteria.

    Classifies the redox process based on the dimensionless kinetic parameter
    :math:`\Lambda = \frac{k^0}{\sqrt{D \frac{n F v}{R T}}}` into one of three regimes:

    - **Reversible (Nernstian)**: :math:`\Lambda \ge 15`
      Electron transfer is fast relative to diffusion; :math:`\Delta E_p \approx 59/n\,\text{mV}`.
    - **Quasi-reversible**: :math:`10^{-3} < \Lambda < 15`
      Both kinetics and diffusion control the response; :math:`\Delta E_p` depends on :math:`v`.
    - **Totally Irreversible**: :math:`\Lambda \le 10^{-3}`
      Kinetics are sluggish; the reverse reaction is negligible at the peak potential.

    Parameters
    ----------
    k0 : float
        Standard heterogeneous rate constant :math:`k^0 > 0`.
    v : float
        Scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    alpha : float, default 0.5
        Charge transfer coefficient (:math:`0 < \alpha < 1`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).

    Returns
    -------
    MatsudaAyabeResult
        Classification container with :math:`\Lambda` and boolean flags.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import matsuda_ayabe
    >>> res = matsuda_ayabe(k0=0.1, v=0.1, D=1e-5)
    >>> res.classification
    'reversible'
    >>> res.is_reversible
    True
    """
    _validate_alpha(alpha)
    lam = float(matsuda_ayabe_lambda(k0=k0, v=v, D=D, n=n, T=T, R=R, F=F))

    if lam >= 15.0:
        cls: Literal["reversible", "quasi-reversible", "irreversible"] = "reversible"
        is_rev, is_quasi, is_irrev = True, False, False
    elif lam <= 1e-3:
        cls = "irreversible"
        is_rev, is_quasi, is_irrev = False, False, True
    else:
        cls = "quasi-reversible"
        is_rev, is_quasi, is_irrev = False, True, False

    return MatsudaAyabeResult(
        lambda_param=lam,
        classification=cls,
        k0=k0,
        v=v,
        D=D,
        is_reversible=is_rev,
        is_quasi_reversible=is_quasi,
        is_irreversible=is_irrev,
    )


# ==============================================================================
# 6. Irreversible Peak Potential & Diagnostics
# ==============================================================================


def peak_potential_irreversible(
    v: float | ArrayLike,
    k0: float,
    E0_prime: float = 0.0,
    D: float = 1e-5,
    alpha: float = 0.5,
    n_alpha: float = 1,
    T: float = STANDARD_TEMPERATURE,
    R: float = GAS_CONSTANT,
    F: float = FARADAY,
    scan_direction: Literal["cathodic", "anodic"] = "cathodic",
) -> float | np.ndarray:
    r"""Compute peak potential for totally irreversible electron transfer.

    Calculates the peak potential :math:`E_p` as a function of scan rate :math:`v`
    under totally irreversible charge-transfer kinetics (Bard & Faulkner Eq. 6.3.18):

    .. math::

        E_p = E^{0\prime} \mp \frac{R T}{\alpha n_\alpha F} \left[ 0.780 + \frac{1}{2} \ln\left( \frac{\alpha n_\alpha F D v}{R T (k^0)^2} \right) \right]

    (negative sign for cathodic reduction, positive sign for anodic oxidation with
    :math:`(1-\alpha)`).

    The diagnostic shift per decade of scan rate is:

    .. math::

        \frac{\partial E_p}{\partial \log_{10} v} = \mp \frac{2.3026 R T}{2 \alpha n_\alpha F} \approx \mp \frac{29.6}{\alpha n_\alpha}\,\text{mV/decade at } 25^\circ\text{C}

    Parameters
    ----------
    v : float or array-like
        Potential sweep scan rate in :math:`\text{V}\cdot\text{s}^{-1}` (:math:`v > 0`).
    k0 : float
        Heterogeneous standard rate constant :math:`k^0 > 0`.
    E0_prime : float, default 0.0
        Formal electrode potential :math:`E^{0\prime}` in Volts.
    D : float, default 1e-5
        Diffusion coefficient (:math:`D > 0`).
    alpha : float, default 0.5
        Charge transfer coefficient (:math:`0 < \alpha < 1`).
    n_alpha : float, default 1
        Number of electrons in rate-determining step (:math:`n_\alpha > 0`).
    T : float, default 298.15
        Thermodynamic temperature in Kelvin (:math:`T > 0`).
    R : float, default 8.31446
        Molar gas constant (:math:`R > 0`).
    F : float, default 96485.332
        Faraday constant (:math:`F > 0`).
    scan_direction : {"cathodic", "anodic"}, default "cathodic"
        Direction of sweep ("cathodic" or "anodic").

    Returns
    -------
    float or numpy.ndarray
        Peak potential :math:`E_p` in Volts.

    Raises
    ------
    ValueError
        If parameters are non-positive or outside physical limits.

    Examples
    --------
    >>> from softpotato.analytical.voltammetry import peak_potential_irreversible
    >>> ep = peak_potential_irreversible(0.1, k0=1e-4, E0_prime=0.0)
    >>> round(ep, 4)
    -0.2347
    """
    _validate_positive(k0, "Standard rate constant k0")
    _validate_positive(D, "Diffusion coefficient D")
    _validate_alpha(alpha)
    _validate_positive(n_alpha, "Number of electrons n_alpha")
    _validate_positive(T, "Temperature T")
    _validate_positive(R, "Gas constant R")
    _validate_positive(F, "Faraday constant F")

    if scan_direction not in ("cathodic", "anodic"):
        raise ValueError(
            f"Invalid scan_direction: {scan_direction!r}. Must be 'cathodic' or 'anodic'."
        )

    v_arr = np.asarray(v, dtype=float)
    if np.any(v_arr <= 0):
        raise ValueError(f"Scan rate v must be strictly positive (v > 0), got {v!r}.")

    eff_alpha = alpha if scan_direction == "cathodic" else (1.0 - alpha)
    shift_arg = (eff_alpha * n_alpha * F * D * v_arr) / (R * T * (k0**2))
    bracket = 0.780 + 0.5 * np.log(shift_arg)
    slope_term = (R * T) / (eff_alpha * n_alpha * F)

    if scan_direction == "cathodic":
        ep = E0_prime - slope_term * bracket
    else:
        ep = E0_prime + slope_term * bracket

    return _format_output(ep, np.ndim(v) == 0)


__all__ = [
    "MatsudaAyabeResult",
    "NicholsonResult",
    "matsuda_ayabe",
    "matsuda_ayabe_lambda",
    "nicholson_delta_ep",
    "nicholson_psi",
    "nicholson_rate_constant",
    "peak_potential_irreversible",
    "randles_sevcik",
    "randles_sevcik_irreversible",
    "randles_sevcik_quasi",
]
