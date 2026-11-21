"""Post-processing and electrochemical diagnostics for technique simulations.

Provides automated voltammetric peak detection (:func:`find_peaks`) and turnkey
diagnostic workflows (:func:`analyze_randles_sevcik`, :func:`analyze_nicholson`,
:func:`analyze_levich`, :func:`analyze_koutecky_levich`, :func:`analyze_tafel`,
:func:`analyze_cottrell`, :func:`analyze_sand`) built directly on
:class:`~softpotato.techniques.TechniqueResult` and :mod:`softpotato.analytical`.

All functions follow IUPAC polarity conventions: anodic (oxidation) currents are
positive (:math:`I > 0`) and cathodic (reduction) currents are negative
(:math:`I < 0`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks

from softpotato.analytical.fitting import (
    CottrellFitResult,
    KouteckyLevichFitResult,
    LevichFitResult,
    RandlesSevcikFitResult,
    TafelFitResult,
    fit_cottrell,
    fit_koutecky_levich,
    fit_levich,
    fit_randles_sevcik,
    fit_tafel,
)
from softpotato.analytical.step import sand_transition_time
from softpotato.analytical.voltammetry import (
    MatsudaAyabeResult,
    NicholsonResult,
    matsuda_ayabe,
    nicholson_psi,
)
from softpotato.constants import FARADAY, STANDARD_TEMPERATURE

from .base import _validate_finite_real, _validate_positive_finite
from .chronoamperometry import Chronoamperometry
from .chronopotentiometry import Chronopotentiometry
from .pulse import DifferentialPulseVoltammetry, SquareWaveVoltammetry
from .rde import RDESweepResult
from .result import TechniqueResult
from .voltammetry import CyclicVoltammetry

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

PeakPolarity = Literal["anodic", "cathodic"]

_DEFAULT_KINEMATIC_VISCOSITY: float = 1e-6


# ==============================================================================
# Result Dataclasses
# ==============================================================================


@dataclass(frozen=True)
class PeakInfo:
    r"""Metadata for a single detected voltammetric current peak.

    Attributes
    ----------
    E_peak : float
        Peak potential :math:`E_p` in Volts (parabolically refined when enabled).
    I_peak : float
        Peak current :math:`I_p` in Amperes under IUPAC polarity conventions
        (positive for anodic, negative for cathodic).
    index : int
        Discrete sample index of the detected local extremum in the analyzed
        potential/current arrays.
    E_half_peak : float or None
        Half-peak potential :math:`E_{p/2}` in Volts on the rising (pre-peak)
        flank where :math:`I = I_p / 2`, or ``None`` if not crossed.
    polarity : {'anodic', 'cathodic'}
        Peak polarity: ``'anodic'`` (:math:`I_p > 0`) or ``'cathodic'``
        (:math:`I_p < 0`).
    """

    E_peak: float
    I_peak: float
    index: int
    E_half_peak: float | None
    polarity: PeakPolarity


@dataclass(frozen=True)
class CVDiagnostics:
    r"""Summary voltammetric peak diagnostics for a CV or LSV trace.

    Attributes
    ----------
    anodic_peak : PeakInfo or None
        Dominant anodic (positive current) peak, or ``None`` if not detected.
    cathodic_peak : PeakInfo or None
        Dominant cathodic (negative current) peak, or ``None`` if not detected.
    delta_Ep : float or None
        Peak-to-peak separation :math:`\Delta E_p = |E_{pa} - E_{pc}|` in Volts
        when both peaks are detected, otherwise ``None``.
    E_half : float or None
        Midpoint potential :math:`E_{1/2} = \frac{1}{2}(E_{pa} + E_{pc})` in
        Volts when both peaks are detected, otherwise ``None``.
    peak_current_ratio : float or None
        Magnitude ratio :math:`|I_{pa} / I_{pc}|` when both peaks are detected
        and :math:`I_{pc} \neq 0`, otherwise ``None``.
    """

    anodic_peak: PeakInfo | None
    cathodic_peak: PeakInfo | None
    delta_Ep: float | None
    E_half: float | None
    peak_current_ratio: float | None


@dataclass(frozen=True)
class NicholsonDiagnostics:
    r"""Combined Nicholson and Matsuda–Ayabe kinetic diagnostics for CV.

    Attributes
    ----------
    delta_Ep : float
        Peak-to-peak potential separation :math:`\Delta E_p` in Volts.
    delta_Ep_mv : float
        Peak-to-peak potential separation :math:`\Delta E_p` in millivolts.
    psi : float
        Dimensionless Nicholson kinetic parameter :math:`\Psi`.
    k0 : float
        Extracted standard heterogeneous rate constant :math:`k^0` in units
        consistent with :math:`D` (e.g., :math:`\text{m}/\text{s}` or
        :math:`\text{cm}/\text{s}`).
    lambda_param : float
        Dimensionless Matsuda–Ayabe reversibility parameter
        :math:`\Lambda = \sqrt{\pi}\,\Psi`.
    regime : {'reversible', 'quasi-reversible', 'irreversible'}
        Matsuda–Ayabe kinetic reversibility classification.
    nicholson : NicholsonResult
        Underlying :class:`~softpotato.analytical.voltammetry.NicholsonResult`.
    matsuda_ayabe : MatsudaAyabeResult
        Underlying :class:`~softpotato.analytical.voltammetry.MatsudaAyabeResult`.
    peaks : CVDiagnostics or None
        Detected voltammetric peaks when analyzed from a :class:`TechniqueResult`
        or ``(E, I)`` arrays, or ``None`` when a scalar ``delta_Ep`` was supplied.
    """

    delta_Ep: float
    delta_Ep_mv: float
    psi: float
    k0: float
    lambda_param: float
    regime: Literal["reversible", "quasi-reversible", "irreversible"]
    nicholson: NicholsonResult
    matsuda_ayabe: MatsudaAyabeResult
    peaks: CVDiagnostics | None = None


@dataclass(frozen=True)
class SandDiagnostics:
    r"""Diagnostic summary for galvanostatic chronopotentiometry (Sand analysis).

    Attributes
    ----------
    tau : float or None
        Detected transition time :math:`\tau` in seconds, or ``None`` if no
        depletion/inflection transition occurred.
    tau_theoretical : float or None
        Theoretical Sand transition time from
        :func:`~softpotato.analytical.step.sand_transition_time` when ``D`` is
        supplied, otherwise ``None``.
    relative_error : float or None
        Relative error :math:`|\tau - \tau_{\text{th}}| / \tau_{\text{th}}`
        when both ``tau`` and ``tau_theoretical`` are available, else ``None``.
    sand_constant : float or None
        Experimental Sand product :math:`|I|\sqrt{\tau}` in
        :math:`\text{A}\cdot\text{s}^{1/2}` when ``tau`` is detected.
    D_estimated : float or None
        Diffusion coefficient :math:`D = \frac{4 I^2 \tau}{\pi (n F A c^*)^2}`
        estimated from the detected transition time.
    current : float
        Applied galvanostatic current :math:`I` in Amperes (signed IUPAC).
    detection_method : {'concentration', 'potential', 'none'}
        Signal source used to detect :math:`\tau`.
    """

    tau: float | None
    tau_theoretical: float | None
    relative_error: float | None
    sand_constant: float | None
    D_estimated: float | None
    current: float
    detection_method: Literal["concentration", "potential", "none"]


# ==============================================================================
# Internal Helpers
# ==============================================================================


def _validate_1d_pair(
    x: ArrayLike,
    y: ArrayLike,
    x_name: str,
    y_name: str,
    *,
    min_points: int = 2,
    require_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two matching 1D numeric arrays."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError(f"'{x_name}' and '{y_name}' must be 1D arrays.")
    if x_arr.size != y_arr.size:
        raise ValueError(
            f"Length mismatch: '{x_name}' has {x_arr.size} points, "
            f"'{y_name}' has {y_arr.size}."
        )
    if x_arr.size < min_points:
        raise ValueError(
            f"At least {min_points} points are required; got {x_arr.size}."
        )
    if require_finite and not (
        np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))
    ):
        raise ValueError(f"All values in '{x_name}' and '{y_name}' must be finite.")
    return x_arr, y_arr


def _extract_potential_current(
    result_or_E: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    cycle: int | None = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1D ``(E, I)`` arrays from a TechniqueResult or raw array pair."""
    if isinstance(result_or_E, TechniqueResult):
        if current is not None:
            raise ValueError(
                "Do not pass 'current' when the first argument is a TechniqueResult."
            )
        tech = result_or_E.technique
        if isinstance(tech, CyclicVoltammetry) and cycle is not None:
            res = result_or_E.cycles[cycle]
            e_arr, i_arr = res.potential, res.current
        elif isinstance(tech, SquareWaveVoltammetry):
            e_arr, i_arr = result_or_E.E_step, result_or_E.i_net
        elif isinstance(tech, DifferentialPulseVoltammetry):
            e_arr, i_arr = result_or_E.E_base, result_or_E.i_diff
        else:
            e_arr, i_arr = result_or_E.potential, result_or_E.current
        return _validate_1d_pair(e_arr, i_arr, "potential", "current", min_points=2)

    if current is None:
        raise ValueError(
            "'current' must be provided when the first argument is a potential array."
        )
    return _validate_1d_pair(result_or_E, current, "E", "current", min_points=2)


def _parabolic_refine(
    e_arr: np.ndarray,
    i_arr: np.ndarray,
    idx: int,
    polarity: PeakPolarity,
) -> tuple[float, float]:
    """Sub-sample 3-point quadratic vertex refinement around discrete index ``idx``."""
    e_center = float(e_arr[idx])
    i_center = float(i_arr[idx])
    if idx <= 0 or idx >= e_arr.size - 1:
        return e_center, i_center

    x0 = float(e_arr[idx - 1]) - e_center
    x2 = float(e_arr[idx + 1]) - e_center
    # Require strictly monotonic potential across the 3-point stencil (not a vertex)
    if x0 == 0.0 or x2 == 0.0 or x0 * x2 >= 0.0:
        return e_center, i_center

    y0 = float(i_arr[idx - 1])
    y1 = i_center
    y2 = float(i_arr[idx + 1])

    denom = x0 * x2 * (x0 - x2)
    if denom == 0.0:
        return e_center, i_center

    a = (x2 * (y0 - y1) - x0 * (y2 - y1)) / denom
    b = ((y2 - y1) - a * (x2**2)) / x2

    # Require concave down for anodic maximum, concave up for cathodic minimum
    if polarity == "anodic" and a >= 0.0:
        return e_center, i_center
    if polarity == "cathodic" and a <= 0.0:
        return e_center, i_center

    x_star = -b / (2.0 * a)
    x_lo, x_hi = (x0, x2) if x0 < x2 else (x2, x0)
    if not (x_lo < x_star < x_hi):
        return e_center, i_center

    e_peak = e_center + x_star
    i_peak = a * (x_star**2) + b * x_star + y1
    return float(e_peak), float(i_peak)


def _find_half_peak_potential(
    e_arr: np.ndarray,
    i_arr: np.ndarray,
    idx: int,
    i_peak: float,
    polarity: PeakPolarity,
) -> float | None:
    """Interpolate half-peak potential E_{p/2} on the rising flank before ``idx``."""
    if idx <= 0:
        return None

    half_i = 0.5 * i_peak
    # Determine scan direction entering the peak so we don't walk across a switching vertex
    de_enter = float(e_arr[idx] - e_arr[idx - 1])

    for k in range(idx - 1, -1, -1):
        de_step = float(e_arr[k + 1] - e_arr[k])
        if de_enter != 0.0 and de_step != 0.0 and (de_enter * de_step < 0.0):
            break

        ik = float(i_arr[k])
        ik1 = float(i_arr[k + 1])
        if polarity == "anodic":
            crossed = ik <= half_i <= ik1
        else:
            crossed = ik >= half_i >= ik1

        if crossed:
            if ik1 == ik:
                return float(e_arr[k])
            frac = (half_i - ik) / (ik1 - ik)
            return float(e_arr[k] + frac * (e_arr[k + 1] - e_arr[k]))

    return None


def _interpolate_current_at_potential(
    e_arr: np.ndarray,
    i_arr: np.ndarray,
    potential: float,
) -> float:
    """Evaluate I(E) at ``potential`` via linear interpolation on the primary sweep leg."""
    e_val = _validate_finite_real(potential, "potential")
    if e_arr.size < 2:
        raise ValueError("At least 2 points are required to interpolate I(E).")

    # If non-monotonic (e.g. CV), isolate the first monotonic sweep leg
    de = np.diff(e_arr)
    nonzero = np.flatnonzero(de != 0.0)
    if nonzero.size == 0:
        if math.isclose(float(e_arr[0]), e_val, rel_tol=1e-9, abs_tol=1e-12):
            return float(i_arr[-1])
        raise ValueError(
            f"Constant potential trace at E={e_arr[0]!r} does not contain potential={e_val!r}."
        )

    first_sign = 1.0 if de[nonzero[0]] > 0.0 else -1.0
    reversals = np.flatnonzero(de * first_sign < 0.0)
    if reversals.size > 0:
        end_idx = int(reversals[0]) + 1
        e_leg = e_arr[:end_idx]
        i_leg = i_arr[:end_idx]
    else:
        e_leg = e_arr
        i_leg = i_arr

    e_min = float(np.min(e_leg))
    e_max = float(np.max(e_leg))
    if e_val < e_min or e_val > e_max:
        raise ValueError(
            f"Requested potential {e_val!r} V lies outside the sweep range "
            f"[{e_min!r}, {e_max!r}] V."
        )

    if first_sign < 0.0:
        e_leg = e_leg[::-1]
        i_leg = i_leg[::-1]

    return float(np.interp(e_val, e_leg, i_leg))


def _resolve_kinematic_viscosity(
    kinematic_viscosity: float | None,
    techniques: Sequence[Any] = (),
) -> float:
    """Resolve positive kinematic viscosity from explicit arg, technique metadata, or default."""
    if kinematic_viscosity is not None:
        return _validate_positive_finite(kinematic_viscosity, "kinematic_viscosity")
    for tech in techniques:
        for attr in ("kinematic_viscosity", "nu"):
            val = getattr(tech, attr, None)
            if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                fval = float(val)
                if math.isfinite(fval) and fval > 0.0:
                    return fval
    return _DEFAULT_KINEMATIC_VISCOSITY


def _extract_rde_series(
    results: RDESweepResult | Sequence[TechniqueResult] | Mapping[Any, TechniqueResult],
    *,
    potential: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    """Extract (omegas, currents, techniques) from RDESweepResult or TechniqueResult collection."""
    if isinstance(results, RDESweepResult):
        omegas = np.asarray(results.omegas, dtype=float)
        if potential is None:
            currents = np.asarray(results.limiting_currents, dtype=float)
        else:
            curr_list: list[float] = []
            for tech, sr in zip(results.techniques, results.results):
                e_arr = np.asarray(tech.potential(sr.t), dtype=float)
                i_arr = np.asarray(sr.currents[results.species], dtype=float)
                curr_list.append(
                    _interpolate_current_at_potential(e_arr, i_arr, potential)
                )
            currents = np.asarray(curr_list, dtype=float)
        return omegas, currents, results.techniques

    items: list[tuple[Any, TechniqueResult]]
    if isinstance(results, Mapping):
        if len(results) == 0:
            raise ValueError("'results' mapping must not be empty.")
        items = list(results.items())
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        if len(results) == 0:
            raise ValueError("'results' sequence must not be empty.")
        items = [(None, r) for r in results]
    else:
        raise TypeError(
            "results must be an RDESweepResult, sequence/mapping of TechniqueResult, "
            f"or raw omega array, got {type(results).__name__}."
        )

    omegas_list: list[float] = []
    currents_list: list[float] = []
    techs_list: list[Any] = []

    for key, res in items:
        if not isinstance(res, TechniqueResult):
            raise TypeError(
                f"Expected TechniqueResult entries, got {type(res).__name__}."
            )
        res_tech = res.technique
        techs_list.append(res_tech)
        omega_val = getattr(res_tech, "omega", None)
        if omega_val is None:
            if isinstance(key, (int, float, np.number)) and not isinstance(key, bool):
                omega_val = float(key)
            else:
                raise ValueError(
                    "Could not infer rotation rate 'omega' from technique "
                    f"{type(res_tech).__name__}. Use RotatingDiskElectrode or a numeric mapping key."
                )
        omegas_list.append(_validate_positive_finite(omega_val, "omega"))

        if potential is None:
            currents_list.append(float(res.current[-1]))
        else:
            currents_list.append(
                _interpolate_current_at_potential(res.potential, res.current, potential)
            )

    return (
        np.asarray(omegas_list, dtype=float),
        np.asarray(currents_list, dtype=float),
        tuple(techs_list),
    )


def _detect_potential_inflection(t: np.ndarray, potential: np.ndarray) -> float | None:
    """Detect chronopotentiometric transition time tau from potential E(t) inflection."""
    finite_mask = np.isfinite(potential)
    if not np.any(finite_mask):
        return None

    # If potential diverges to +/-inf (e.g. Nernstian depletion at t >= tau),
    # the earliest non-finite sample after finite samples marks tau.
    finite_idx = np.flatnonzero(finite_mask)
    first_finite = int(finite_idx[0])
    nonfinite_after = np.flatnonzero(~finite_mask[first_finite:])
    if nonfinite_after.size > 0:
        div_idx = first_finite + int(nonfinite_after[0])
        if div_idx > 0 and np.isfinite(potential[div_idx - 1]):
            return float(0.5 * (t[div_idx - 1] + t[div_idx]))
        return float(t[div_idx])

    if t.size < 3:
        return None

    # Finite chronopotentiogram: find maximum |dE/dt| on interior points
    de_dt = np.abs(np.gradient(potential, t))
    interior = de_dt[1:-1]
    if interior.size == 0 or np.max(interior) == 0.0:
        return None
    best_idx = int(np.argmax(interior)) + 1
    return float(t[best_idx])


# ==============================================================================
# 1. Peak Detection
# ==============================================================================


@overload
def find_peaks(
    result_or_E: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    cycle: int | None = -1,
    parabolic: bool = True,
    prominence: float | None = None,
    min_current: float = 0.0,
    distance: int | None = None,
    polarity: Literal["both", "anodic", "cathodic"] = "both",
    return_all: Literal[False] = False,
) -> CVDiagnostics: ...


@overload
def find_peaks(
    result_or_E: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    cycle: int | None = -1,
    parabolic: bool = True,
    prominence: float | None = None,
    min_current: float = 0.0,
    distance: int | None = None,
    polarity: Literal["both", "anodic", "cathodic"] = "both",
    return_all: Literal[True],
) -> list[PeakInfo]: ...


def find_peaks(
    result_or_E: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    cycle: int | None = -1,
    parabolic: bool = True,
    prominence: float | None = None,
    min_current: float = 0.0,
    distance: int | None = None,
    polarity: Literal["both", "anodic", "cathodic"] = "both",
    return_all: bool = False,
) -> CVDiagnostics | list[PeakInfo]:
    r"""Detect anodic and cathodic voltammetric peaks under IUPAC polarity conventions.

    Identifies interior local maxima (:math:`I > 0`, anodic) and local minima
    (:math:`I < 0`, cathodic) in a voltammogram, optionally refining peak
    potential :math:`E_p` and peak current :math:`I_p` via 3-point parabolic
    interpolation and computing the rising-flank half-peak potential
    :math:`E_{p/2}`.

    Parameters
    ----------
    result_or_E : TechniqueResult or array-like
        Either a :class:`~softpotato.techniques.TechniqueResult` (CV, LSV, SWV,
        DPV, etc.) or a 1D array of electrode potentials :math:`E` in Volts.
    current : array-like, optional
        1D array of Faradaic currents :math:`I` in Amperes (required when
        ``result_or_E`` is a raw potential array; must be ``None`` when a
        :class:`TechniqueResult` is passed).
    cycle : int or None, default -1
        Cycle index to analyze when ``result_or_E`` is a
        :class:`~softpotato.techniques.CyclicVoltammetry` result (``-1`` selects
        the last cycle; ``None`` analyzes the full multi-cycle trace).
    parabolic : bool, default True
        If ``True``, refine :math:`E_p` and :math:`I_p` using 3-point quadratic
        interpolation around the discrete grid extremum.
    prominence : float, optional
        Minimum peak prominence (in Amperes) forwarded to
        :func:`scipy.signal.find_peaks`.
    min_current : float, default 0.0
        Minimum absolute current threshold (:math:`|I_p| > \text{min\_current}`)
        for a peak to be accepted.
    distance : int, optional
        Minimum sample separation between adjacent peaks of the same polarity.
    polarity : {'both', 'anodic', 'cathodic'}, default 'both'
        Which peak polarities to search for.
    return_all : bool, default False
        If ``False`` (default), returns a :class:`CVDiagnostics` summary
        containing the dominant anodic and cathodic peaks plus
        :math:`\Delta E_p`, :math:`E_{1/2}`, and :math:`|I_{pa}/I_{pc}|`.
        If ``True``, returns a list of all detected :class:`PeakInfo` objects
        ordered by sample index.

    Returns
    -------
    CVDiagnostics or list[PeakInfo]
        Voltammetric peak diagnostics summary or full list of detected peaks.
    """
    if polarity not in ("both", "anodic", "cathodic"):
        raise ValueError(
            f"polarity must be 'both', 'anodic', or 'cathodic', got {polarity!r}."
        )
    if min_current < 0.0 or not math.isfinite(min_current):
        raise ValueError(
            f"min_current must be finite and non-negative, got {min_current!r}."
        )

    e_arr, i_arr = _extract_potential_current(result_or_E, current, cycle=cycle)

    peak_kwargs: dict[str, Any] = {}
    if prominence is not None:
        peak_kwargs["prominence"] = _validate_positive_finite(prominence, "prominence")
    if distance is not None:
        peak_kwargs["distance"] = distance

    anodic_peaks: list[PeakInfo] = []
    cathodic_peaks: list[PeakInfo] = []

    if e_arr.size >= 3:
        if polarity in ("both", "anodic"):
            an_indices, _ = _scipy_find_peaks(i_arr, height=min_current, **peak_kwargs)
            for raw_idx in an_indices:
                idx = int(raw_idx)
                if float(i_arr[idx]) <= min_current:
                    continue
                if parabolic:
                    e_p, i_p = _parabolic_refine(e_arr, i_arr, idx, "anodic")
                else:
                    e_p, i_p = float(e_arr[idx]), float(i_arr[idx])
                e_half_p = _find_half_peak_potential(e_arr, i_arr, idx, i_p, "anodic")
                anodic_peaks.append(
                    PeakInfo(
                        E_peak=e_p,
                        I_peak=i_p,
                        index=idx,
                        E_half_peak=e_half_p,
                        polarity="anodic",
                    )
                )

        if polarity in ("both", "cathodic"):
            cat_indices, _ = _scipy_find_peaks(
                -i_arr, height=min_current, **peak_kwargs
            )
            for raw_idx in cat_indices:
                idx = int(raw_idx)
                if float(-i_arr[idx]) <= min_current:
                    continue
                if parabolic:
                    e_p, i_p = _parabolic_refine(e_arr, i_arr, idx, "cathodic")
                else:
                    e_p, i_p = float(e_arr[idx]), float(i_arr[idx])
                e_half_p = _find_half_peak_potential(e_arr, i_arr, idx, i_p, "cathodic")
                cathodic_peaks.append(
                    PeakInfo(
                        E_peak=e_p,
                        I_peak=i_p,
                        index=idx,
                        E_half_peak=e_half_p,
                        polarity="cathodic",
                    )
                )

    if return_all:
        all_peaks = anodic_peaks + cathodic_peaks
        all_peaks.sort(key=lambda p: p.index)
        return all_peaks

    best_anodic = max(anodic_peaks, key=lambda p: p.I_peak) if anodic_peaks else None
    best_cathodic = (
        min(cathodic_peaks, key=lambda p: p.I_peak) if cathodic_peaks else None
    )

    if best_anodic is not None and best_cathodic is not None:
        delta_ep: float | None = float(abs(best_anodic.E_peak - best_cathodic.E_peak))
        e_half: float | None = float(0.5 * (best_anodic.E_peak + best_cathodic.E_peak))
        ratio: float | None = (
            float(abs(best_anodic.I_peak / best_cathodic.I_peak))
            if best_cathodic.I_peak != 0.0
            else None
        )
    else:
        delta_ep = None
        e_half = None
        ratio = None

    return CVDiagnostics(
        anodic_peak=best_anodic,
        cathodic_peak=best_cathodic,
        delta_Ep=delta_ep,
        E_half=e_half,
        peak_current_ratio=ratio,
    )


# ==============================================================================
# 2. Randles–Ševčík Analysis
# ==============================================================================


def analyze_randles_sevcik(
    results: Sequence[TechniqueResult] | Mapping[Any, TechniqueResult] | ArrayLike,
    peak_currents: ArrayLike | None = None,
    *,
    n: float,
    area: float,
    c_bulk: float,
    temperature: float = STANDARD_TEMPERATURE,
    polarity: PeakPolarity = "cathodic",
    cycle: int | None = -1,
    parabolic: bool = True,
    method: Literal["linear", "nonlinear"] = "linear",
    model: Literal["reversible", "irreversible"] = "reversible",
    alpha: float = 0.5,
    n_alpha: float = 1,
    fit_intercept: bool = True,
    confidence_level: float = 0.95,
) -> RandlesSevcikFitResult:
    r"""Perform multi-scan-rate Randles–Ševčík regression to extract diffusivity :math:`D`.

    Extracts peak currents :math:`I_p` and scan rates :math:`v` from a series of
    CV or LSV :class:`~softpotato.techniques.TechniqueResult` objects (or accepts
    raw ``(scan_rates, peak_currents)`` arrays directly) and delegates to
    :func:`~softpotato.analytical.fitting.fit_randles_sevcik`.

    Parameters
    ----------
    results : Sequence[TechniqueResult], Mapping[Any, TechniqueResult], or array-like
        Collection of CV/LSV simulation results (e.g., from
        :func:`~softpotato.techniques.simulate_sweep`) or a 1D array of scan
        rates :math:`v` in V/s when ``peak_currents`` is supplied.
    peak_currents : array-like, optional
        1D array of peak currents :math:`I_p` in Amperes when ``results`` is a
        raw array of scan rates.
    n : float
        Number of electrons transferred (:math:`n > 0`).
    area : float
        Electrode surface area :math:`A > 0` in :math:`\text{m}^2` or
        :math:`\text{cm}^2`.
    c_bulk : float
        Bulk reactant concentration :math:`c^* > 0` in matching volume units.
    temperature : float, default 298.15
        Absolute temperature :math:`T > 0` in Kelvin.
    polarity : {'cathodic', 'anodic'}, default 'cathodic'
        Peak polarity to extract from each voltammogram.
    cycle : int or None, default -1
        CV cycle index to inspect (ignored for LSV or raw arrays).
    parabolic : bool, default True
        Whether to use parabolic sub-sample peak refinement.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression method passed to :func:`fit_randles_sevcik`.
    model : {'reversible', 'irreversible'}, default 'reversible'
        Reversibility model passed to :func:`fit_randles_sevcik`.
    alpha : float, default 0.5
        Charge-transfer coefficient for irreversible model.
    n_alpha : float, default 1
        Electrons transferred up to the rate-determining step (irreversible model).
    fit_intercept : bool, default True
        Whether to fit a non-zero current intercept.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` parameter intervals.

    Returns
    -------
    RandlesSevcikFitResult
        Fitted diffusion coefficient :math:`D`, slope, intercept, standard
        errors, and goodness-of-fit metrics.
    """
    if polarity not in ("cathodic", "anodic"):
        raise ValueError(f"polarity must be 'cathodic' or 'anodic', got {polarity!r}.")

    if peak_currents is not None:
        v_arr, ip_arr = _validate_1d_pair(
            results,  # type: ignore[arg-type]
            peak_currents,
            "scan_rates",
            "peak_currents",
            min_points=2,
        )
    else:
        items: list[tuple[Any, Any]]
        if isinstance(results, Mapping):
            if len(results) == 0:
                raise ValueError("'results' mapping must not be empty.")
            items = list(results.items())
        elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
            if len(results) == 0:
                raise ValueError("'results' sequence must not be empty.")
            items = [(None, r) for r in results]
        else:
            raise TypeError(
                "results must be a sequence/mapping of TechniqueResult objects "
                "or a scan-rate array paired with 'peak_currents'."
            )

        v_list: list[float] = []
        ip_list: list[float] = []
        for idx, (key, res) in enumerate(items):
            if not isinstance(res, TechniqueResult):
                raise TypeError(
                    f"Expected TechniqueResult at index {idx}, got {type(res).__name__}."
                )
            v_val = getattr(res.technique, "scan_rate", None)
            if v_val is None:
                if isinstance(key, (int, float, np.number)) and not isinstance(
                    key, bool
                ):
                    v_val = float(key)
                else:
                    raise ValueError(
                        f"Could not infer 'scan_rate' from technique "
                        f"{type(res.technique).__name__} at index {idx}."
                    )
            v_list.append(_validate_positive_finite(v_val, "scan_rate"))

            diag = find_peaks(res, cycle=cycle, parabolic=parabolic, polarity=polarity)
            peak = diag.cathodic_peak if polarity == "cathodic" else diag.anodic_peak
            if peak is None:
                raise ValueError(
                    f"No {polarity} peak detected in result at index {idx} "
                    f"(scan_rate={v_val!r} V/s)."
                )
            ip_list.append(peak.I_peak)

        v_arr = np.asarray(v_list, dtype=float)
        ip_arr = np.asarray(ip_list, dtype=float)

    return fit_randles_sevcik(
        scan_rate=v_arr,
        peak_current=ip_arr,
        method=method,
        model=model,
        n=n,
        area=area,
        c_bulk=c_bulk,
        alpha=alpha,
        n_alpha=n_alpha,
        T=temperature,
        fit_intercept=fit_intercept,
        confidence_level=confidence_level,
    )


# ==============================================================================
# 3. Nicholson & Matsuda–Ayabe Analysis
# ==============================================================================


def analyze_nicholson(
    result: TechniqueResult | ArrayLike | float,
    current: ArrayLike | None = None,
    *,
    D: float,
    scan_rate: float | None = None,
    n: float = 1,
    temperature: float = STANDARD_TEMPERATURE,
    alpha: float = 0.5,
    cycle: int | None = -1,
    parabolic: bool = True,
    method: Literal["swaddle", "lavagnini", "table"] = "swaddle",
) -> NicholsonDiagnostics:
    r"""Extract heterogeneous rate constant :math:`k^0` via Nicholson analysis.

    Detects the anodic and cathodic peaks of a CV trace, computes
    :math:`\Delta E_p = |E_{pa} - E_{pc}|`, evaluates the Nicholson kinetic
    parameter :math:`\Psi` and standard rate constant :math:`k^0` via
    :func:`~softpotato.analytical.voltammetry.nicholson_psi`, and classifies the
    kinetic regime via :func:`~softpotato.analytical.voltammetry.matsuda_ayabe`.

    Parameters
    ----------
    result : TechniqueResult, array-like, or float
        A CV :class:`~softpotato.techniques.TechniqueResult`, a 1D potential
        array (paired with ``current``), or a scalar peak-to-peak separation
        :math:`\Delta E_p` in Volts.
    current : array-like, optional
        1D Faradaic current array in Amperes when ``result`` is a potential array.
    D : float
        Diffusion coefficient (:math:`D > 0`).
    scan_rate : float, optional
        Potential scan rate :math:`v > 0` in V/s. Inferred automatically from
        ``result.technique.scan_rate`` when ``result`` is a CV
        :class:`TechniqueResult`.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    temperature : float, default 298.15
        Absolute temperature :math:`T > 0` in Kelvin.
    alpha : float, default 0.5
        Charge-transfer coefficient :math:`0 < \alpha < 1` forwarded to
        :func:`~softpotato.analytical.voltammetry.matsuda_ayabe`.
    cycle : int or None, default -1
        CV cycle index to inspect when ``result`` is a :class:`TechniqueResult`.
    parabolic : bool, default True
        Whether to use 3-point parabolic interpolation for peak refinement.
    method : {'swaddle', 'lavagnini', 'table'}, default 'swaddle'
        Nicholson working curve approximation passed to :func:`nicholson_psi`.

    Returns
    -------
    NicholsonDiagnostics
        Container with :math:`\Delta E_p`, :math:`\Psi`, :math:`k^0`,
        :math:`\Lambda`, Matsuda–Ayabe regime classification, and detected peaks.
    """
    peaks: CVDiagnostics | None = None
    if isinstance(result, TechniqueResult):
        if current is not None:
            raise ValueError(
                "Do not pass 'current' when the first argument is a TechniqueResult."
            )
        if scan_rate is None:
            inferred_v = getattr(result.technique, "scan_rate", None)
            if inferred_v is None:
                raise ValueError(
                    "Could not infer 'scan_rate' from technique "
                    f"{type(result.technique).__name__}; pass 'scan_rate' explicitly."
                )
            v_val = _validate_positive_finite(inferred_v, "scan_rate")
        else:
            v_val = _validate_positive_finite(scan_rate, "scan_rate")

        peaks = find_peaks(result, cycle=cycle, parabolic=parabolic)
        if peaks.delta_Ep is None:
            raise ValueError(
                "Nicholson analysis requires both an anodic and a cathodic peak; "
                f"detected anodic={peaks.anodic_peak is not None}, "
                f"cathodic={peaks.cathodic_peak is not None}."
            )
        delta_ep = peaks.delta_Ep
    elif current is not None:
        if scan_rate is None:
            raise ValueError(
                "'scan_rate' must be provided when analyzing raw (E, I) arrays."
            )
        v_val = _validate_positive_finite(scan_rate, "scan_rate")
        peaks = find_peaks(result, current, parabolic=parabolic)
        if peaks.delta_Ep is None:
            raise ValueError(
                "Nicholson analysis requires both an anodic and a cathodic peak."
            )
        delta_ep = peaks.delta_Ep
    elif isinstance(result, (int, float, np.number)) and not isinstance(result, bool):
        if scan_rate is None:
            raise ValueError(
                "'scan_rate' must be provided when passing scalar 'delta_Ep'."
            )
        v_val = _validate_positive_finite(scan_rate, "scan_rate")
        delta_ep = _validate_positive_finite(float(result), "delta_Ep")
    else:
        raise TypeError(
            "result must be a TechniqueResult, a potential array paired with 'current', "
            f"or a scalar delta_Ep in Volts, got {type(result).__name__}."
        )

    nich_res = nicholson_psi(
        delta_ep=delta_ep,
        v=v_val,
        D=D,
        n=n,
        T=temperature,
        in_volts=True,
        method=method,
    )
    assert isinstance(nich_res, NicholsonResult) and nich_res.k0 is not None

    ma_res = matsuda_ayabe(
        k0=nich_res.k0,
        v=v_val,
        D=D,
        n=n,
        alpha=alpha,
        T=temperature,
    )

    return NicholsonDiagnostics(
        delta_Ep=nich_res.delta_ep,
        delta_Ep_mv=nich_res.delta_ep_mv,
        psi=nich_res.psi,
        k0=nich_res.k0,
        lambda_param=ma_res.lambda_param,
        regime=ma_res.classification,
        nicholson=nich_res,
        matsuda_ayabe=ma_res,
        peaks=peaks,
    )


# ==============================================================================
# 4. Levich & Koutecký–Levich Analysis (RDE)
# ==============================================================================


def analyze_levich(
    results: RDESweepResult
    | Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | ArrayLike,
    limiting_currents: ArrayLike | None = None,
    *,
    n: float,
    area: float,
    c_bulk: float,
    kinematic_viscosity: float | None = None,
    limiting_potential: float | None = None,
    method: Literal["linear", "nonlinear"] = "linear",
    fit_intercept: bool = False,
    confidence_level: float = 0.95,
) -> LevichFitResult:
    r"""Perform Levich regression (:math:`I_L` vs. :math:`\omega^{1/2}`) on RDE sweeps.

    Accepts an :class:`~softpotato.techniques.RDESweepResult`, a sequence or
    mapping of RDE :class:`~softpotato.techniques.TechniqueResult` objects, or
    raw ``(omegas, limiting_currents)`` arrays, and delegates to
    :func:`~softpotato.analytical.fitting.fit_levich`.

    Parameters
    ----------
    results : RDESweepResult, Sequence[TechniqueResult], Mapping, or array-like
        Multi-rotation-rate RDE simulation results or 1D array of angular
        rotation velocities :math:`\omega` in rad/s (paired with
        ``limiting_currents``).
    limiting_currents : array-like, optional
        1D array of limiting currents :math:`I_L` in Amperes when ``results`` is
        a raw array of angular velocities.
    n : float
        Number of electrons transferred (:math:`n > 0`).
    area : float
        Disk electrode surface area :math:`A > 0` in :math:`\text{m}^2` or
        :math:`\text{cm}^2`.
    c_bulk : float
        Bulk reactant concentration :math:`c^* > 0` in matching volume units.
    kinematic_viscosity : float, optional
        Electrolyte kinematic viscosity :math:`\nu > 0` in
        :math:`\text{m}^2/\text{s}` or :math:`\text{cm}^2/\text{s}`. If omitted,
        inferred from the RDE technique metadata when available, defaulting to
        ``1e-6`` :math:`\text{m}^2/\text{s}`.
    limiting_potential : float, optional
        Optional electrode potential (in Volts) at which to sample :math:`I_L`
        from each RDE sweep. When ``None``, uses the steady-state terminal
        current of each sweep.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression method forwarded to :func:`fit_levich`.
    fit_intercept : bool, default False
        Whether to include a non-zero current offset intercept.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` parameter intervals.

    Returns
    -------
    LevichFitResult
        Extracted Levich constant :math:`B`, diffusion coefficient :math:`D`,
        standard errors, and goodness-of-fit statistics.
    """
    if limiting_currents is not None:
        w_arr, il_arr = _validate_1d_pair(
            results,  # type: ignore[arg-type]
            limiting_currents,
            "omegas",
            "limiting_currents",
            min_points=2,
        )
        nu_val = _resolve_kinematic_viscosity(kinematic_viscosity)
    else:
        w_arr, il_arr, techs = _extract_rde_series(
            results,  # type: ignore[arg-type]
            potential=limiting_potential,
        )
        nu_val = _resolve_kinematic_viscosity(kinematic_viscosity, techs)

    return fit_levich(
        current=il_arr,
        omega=w_arr,
        method=method,
        n=n,
        area=area,
        c_bulk=c_bulk,
        nu=nu_val,
        fit_intercept=fit_intercept,
        confidence_level=confidence_level,
    )


def analyze_koutecky_levich(
    results: RDESweepResult
    | Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | ArrayLike,
    currents: ArrayLike | None = None,
    *,
    potential: float | None = None,
    n: float,
    area: float,
    c_bulk: float,
    kinematic_viscosity: float | None = None,
    method: Literal["linear", "nonlinear"] = "linear",
    confidence_level: float = 0.95,
) -> KouteckyLevichFitResult:
    r"""Perform Koutecký–Levich regression (:math:`|I|^{-1}` vs. :math:`\omega^{-1/2}`).

    Evaluates :math:`I(E, \omega)` at the chosen electrode ``potential`` across a
    series of RDE sweeps (or accepts raw ``(omegas, currents)`` arrays) and
    delegates to :func:`~softpotato.analytical.fitting.fit_koutecky_levich` to
    separate kinetic current :math:`I_K` and mass-transport Levich constant
    :math:`B`. Handles both anodic (:math:`I > 0`) and cathodic (:math:`I < 0`)
    IUPAC currents cleanly.

    Parameters
    ----------
    results : RDESweepResult, Sequence[TechniqueResult], Mapping, or array-like
        Multi-rotation-rate RDE simulation results or 1D array of angular
        rotation velocities :math:`\omega` in rad/s (paired with ``currents``).
    currents : array-like, optional
        1D array of Faradaic currents :math:`I` in Amperes at each rotation rate
        when ``results`` is a raw array of angular velocities.
    potential : float, optional
        Electrode potential :math:`E` in Volts at which to evaluate
        :math:`I(E, \omega)` across RDE sweeps. Required when ``results`` is an
        :class:`RDESweepResult` or collection of :class:`TechniqueResult`.
    n : float
        Number of electrons transferred (:math:`n > 0`).
    area : float
        Disk electrode surface area :math:`A > 0` in :math:`\text{m}^2` or
        :math:`\text{cm}^2`.
    c_bulk : float
        Bulk reactant concentration :math:`c^* > 0` in matching volume units.
    kinematic_viscosity : float, optional
        Electrolyte kinematic viscosity :math:`\nu > 0`. Inferred from RDE
        technique metadata when available, defaulting to ``1e-6``.
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression method forwarded to :func:`fit_koutecky_levich`.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` parameter intervals.

    Returns
    -------
    KouteckyLevichFitResult
        Extracted kinetic current :math:`I_K`, apparent rate constant
        :math:`k(E)`, Levich constant :math:`B`, diffusion coefficient :math:`D`,
        and goodness-of-fit statistics.
    """
    if currents is not None:
        w_arr, i_arr = _validate_1d_pair(
            results,  # type: ignore[arg-type]
            currents,
            "omegas",
            "currents",
            min_points=2,
        )
        nu_val = _resolve_kinematic_viscosity(kinematic_viscosity)
    else:
        if potential is None:
            raise ValueError(
                "'potential' must be provided when analyzing RDE simulation results."
            )
        w_arr, i_arr, techs = _extract_rde_series(
            results,  # type: ignore[arg-type]
            potential=potential,
        )
        nu_val = _resolve_kinematic_viscosity(kinematic_viscosity, techs)

    if np.any(i_arr == 0.0):
        raise ValueError("All currents in Koutecký–Levich analysis must be non-zero.")
    if not (np.all(i_arr > 0.0) or np.all(i_arr < 0.0)):
        raise ValueError(
            "All currents in Koutecký–Levich analysis must share the same polarity "
            "(all anodic > 0 or all cathodic < 0)."
        )

    return fit_koutecky_levich(
        current=np.abs(i_arr),
        omega=w_arr,
        method=method,
        n=n,
        area=area,
        c_bulk=c_bulk,
        nu=nu_val,
        confidence_level=confidence_level,
    )


# ==============================================================================
# 5. Tafel Analysis
# ==============================================================================


def analyze_tafel(
    result: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    E0: float,
    overpotential_window: tuple[float, float] | None = None,
    branch: PeakPolarity = "cathodic",
    temperature: float = STANDARD_TEMPERATURE,
    n: float = 1,
    area: float | None = None,
    c_bulk: float | None = None,
    cycle: int | None = -1,
    confidence_level: float = 0.95,
) -> TafelFitResult:
    r"""Extract Tafel slope, transfer coefficient, and exchange current.

    Computes overpotential :math:`\eta = E - E^{0\prime}` and delegates to
    :func:`~softpotato.analytical.fitting.fit_tafel` on the selected kinetic
    branch (``'cathodic'`` for :math:`\eta < 0`, ``'anodic'`` for :math:`\eta > 0`).

    Parameters
    ----------
    result : TechniqueResult or array-like
        A voltammetric :class:`~softpotato.techniques.TechniqueResult` or a 1D
        electrode potential array :math:`E` in Volts (paired with ``current``).
    current : array-like, optional
        1D Faradaic current array :math:`I` in Amperes when ``result`` is a
        potential array.
    E0 : float
        Formal / equilibrium potential :math:`E^{0\prime}` in Volts used to
        compute :math:`\eta = E - E^{0\prime}`.
    overpotential_window : tuple of (float, float), optional
        Signed overpotential interval ``(eta_min, eta_max)`` in Volts over which
        to fit (e.g., ``(-0.20, -0.06)`` for the cathodic branch or
        ``(0.06, 0.20)`` for the anodic branch).
    branch : {'cathodic', 'anodic'}, default 'cathodic'
        Kinetic polarization branch to fit.
    temperature : float, default 298.15
        Absolute temperature :math:`T > 0` in Kelvin.
    n : float, default 1
        Number of electrons transferred (:math:`n > 0`).
    area : float, optional
        Electrode surface area :math:`A > 0` (if provided alongside ``c_bulk``,
        extracts :math:`k^0`).
    c_bulk : float, optional
        Bulk concentration :math:`c^* > 0` (if provided alongside ``area``,
        extracts :math:`k^0`).
    cycle : int or None, default -1
        CV cycle index to inspect when ``result`` is a CV
        :class:`TechniqueResult`.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` parameter intervals.

    Returns
    -------
    TafelFitResult
        Extracted exchange current :math:`I_0`, Tafel slope :math:`b` (in V/dec
        and mV/dec), charge-transfer coefficient :math:`\alpha`, optional
        :math:`k^0`, and goodness-of-fit statistics.
    """
    e0_val = _validate_finite_real(E0, "E0")
    e_arr, i_arr = _extract_potential_current(result, current, cycle=cycle)

    # For CV TechniqueResult, isolate the forward leg so reverse-scan diffusion
    # depletion does not contaminate the Tafel regression at overlapping potentials.
    if isinstance(result, TechniqueResult) and isinstance(
        result.technique, CyclicVoltammetry
    ):
        de = np.diff(e_arr)
        nonzero = np.flatnonzero(de != 0.0)
        if nonzero.size > 0:
            first_sign = 1.0 if de[nonzero[0]] > 0.0 else -1.0
            reversals = np.flatnonzero(de * first_sign < 0.0)
            if reversals.size > 0:
                end_idx = int(reversals[0]) + 1
                e_arr = e_arr[:end_idx]
                i_arr = i_arr[:end_idx]

    eta = e_arr - e0_val
    return fit_tafel(
        overpotential=eta,
        current=i_arr,
        branch=branch,
        fit_range=overpotential_window,
        area=area,
        c_bulk=c_bulk,
        n=n,
        T=temperature,
        confidence_level=confidence_level,
    )


# ==============================================================================
# 6. Cottrell Analysis (Chronoamperometry)
# ==============================================================================


def analyze_cottrell(
    result: TechniqueResult | ArrayLike,
    current: ArrayLike | None = None,
    *,
    n: float,
    area: float,
    c_bulk: float,
    t_min: float | None = None,
    t_max: float | None = None,
    method: Literal["linear", "nonlinear"] = "linear",
    fit_background: bool = False,
    confidence_level: float = 0.95,
) -> CottrellFitResult:
    r"""Perform Cottrell analysis (:math:`I` vs. :math:`t^{-1/2}`) on a CA transient.

    Filters the transient to strictly positive times within ``[t_min, t_max]``
    and delegates to :func:`~softpotato.analytical.fitting.fit_cottrell` to
    recover the diffusion coefficient :math:`D`.

    Parameters
    ----------
    result : TechniqueResult or array-like
        A chronoamperometric :class:`~softpotato.techniques.TechniqueResult` or
        a 1D time array :math:`t` in seconds (paired with ``current``).
    current : array-like, optional
        1D Faradaic current array :math:`I` in Amperes when ``result`` is a time
        array.
    n : float
        Number of electrons transferred (:math:`n > 0`).
    area : float
        Electrode surface area :math:`A > 0` in :math:`\text{m}^2` or
        :math:`\text{cm}^2`.
    c_bulk : float
        Bulk reactant concentration :math:`c^* > 0` in matching volume units.
    t_min : float, optional
        Lower time cutoff in seconds (inclusive, strictly positive).
    t_max : float, optional
        Upper time cutoff in seconds (inclusive).
    method : {'linear', 'nonlinear'}, default 'linear'
        Regression method forwarded to :func:`fit_cottrell`.
    fit_background : bool, default False
        Whether to fit a constant background current offset :math:`I_{\text{bg}}`.
    confidence_level : float, default 0.95
        Confidence level for Student's :math:`t` parameter intervals.

    Returns
    -------
    CottrellFitResult
        Extracted diffusion coefficient :math:`D`, standard error, slope,
        background current, and goodness-of-fit statistics.
    """
    t_offset = 0.0
    if isinstance(result, TechniqueResult):
        if current is not None:
            raise ValueError(
                "Do not pass 'current' when the first argument is a TechniqueResult."
            )
        t_arr, i_arr = _validate_1d_pair(
            result.time, result.current, "time", "current", min_points=2
        )
        if (
            isinstance(result.technique, Chronoamperometry)
            and result.technique.t_quiet > 0.0
        ):
            t_offset = result.technique.t_quiet
    else:
        if current is None:
            raise ValueError(
                "'current' must be provided when the first argument is a time array."
            )
        t_arr, i_arr = _validate_1d_pair(result, current, "t", "current", min_points=2)

    if t_min is not None:
        t_min_val = _validate_finite_real(t_min, "t_min")
    else:
        t_min_val = None
    if t_max is not None:
        t_max_val = _validate_finite_real(t_max, "t_max")
    else:
        t_max_val = None
    if t_min_val is not None and t_max_val is not None and t_min_val >= t_max_val:
        raise ValueError(
            f"t_min ({t_min_val!r}) must be strictly less than t_max ({t_max_val!r})."
        )

    t_eff = t_arr - t_offset
    mask = t_eff > 0.0
    if t_min_val is not None:
        mask &= t_arr >= t_min_val
    if t_max_val is not None:
        mask &= t_arr <= t_max_val

    t_sub = t_eff[mask]
    i_sub = i_arr[mask]
    if t_sub.size < 2:
        raise ValueError(
            f"Cottrell analysis requires at least 2 valid points with t > 0 in the "
            f"selected window; got {t_sub.size}."
        )

    return fit_cottrell(
        time=t_sub,
        current=i_sub,
        method=method,
        n=n,
        area=area,
        c_bulk=c_bulk,
        fit_background=fit_background,
        confidence_level=confidence_level,
    )


# ==============================================================================
# 7. Sand Analysis (Chronopotentiometry)
# ==============================================================================


def analyze_sand(
    result: TechniqueResult | ArrayLike,
    signal: ArrayLike | None = None,
    *,
    n: float,
    area: float,
    c_bulk: float,
    threshold: float = 1e-3,
    D: float | None = None,
    current: float | None = None,
    species: str | None = None,
    signal_kind: Literal["auto", "concentration", "potential"] = "auto",
    relative: bool = True,
    F: float = FARADAY,
) -> SandDiagnostics:
    r"""Detect chronopotentiometric transition time :math:`\tau` and run Sand analysis.

    Detects the transition time :math:`\tau` from reactant surface concentration
    depletion (via :meth:`~softpotato.techniques.Chronopotentiometry.detect_transition_time`)
    or from the potential inflection point :math:`\max |dE/dt|`, computes the
    experimental Sand product :math:`|I|\sqrt{\tau}` and estimated diffusion
    coefficient :math:`D`, and optionally compares against
    :func:`~softpotato.analytical.step.sand_transition_time`.

    Parameters
    ----------
    result : TechniqueResult or array-like
        A chronopotentiometric :class:`~softpotato.techniques.TechniqueResult`
        or a 1D time array :math:`t` in seconds (paired with ``signal``).
    signal : array-like, optional
        1D surface concentration :math:`c(0, t)` or electrode potential
        :math:`E(t)` array when ``result`` is a raw time array.
    n : float
        Number of electrons transferred (:math:`n > 0`).
    area : float
        Electrode surface area :math:`A > 0` in :math:`\text{m}^2` or
        :math:`\text{cm}^2`.
    c_bulk : float
        Bulk reactant concentration :math:`c^* > 0` in matching volume units.
    threshold : float, default 1e-3
        Surface concentration depletion threshold forwarded to
        :meth:`Chronopotentiometry.detect_transition_time`.
    D : float, optional
        Known diffusion coefficient (:math:`D > 0`) for computing theoretical
        Sand transition time :math:`\tau_{\text{th}}` and relative error.
    current : float, optional
        Applied constant current :math:`I \neq 0` in Amperes. Inferred
        automatically from ``result`` when a :class:`TechniqueResult` is passed;
        required when raw arrays are passed.
    species : str, optional
        Reactant species name to inspect in ``result.concentrations``. Defaults
        to ``result.metadata['reactant_species']`` or the first species with
        positive initial surface concentration.
    signal_kind : {'auto', 'concentration', 'potential'}, default 'auto'
        Signal source used for transition time detection.
    relative : bool, default True
        Whether ``threshold`` is relative to initial surface concentration.
    F : float, default FARADAY
        Faraday constant in :math:`\text{C}/\text{mol}`.

    Returns
    -------
    SandDiagnostics
        Detected :math:`\tau`, theoretical :math:`\tau` (if ``D`` supplied),
        relative error, Sand constant :math:`|I|\sqrt{\tau}`, estimated
        :math:`D`, applied current, and detection method.
    """
    n_val = _validate_positive_finite(n, "n")
    area_val = _validate_positive_finite(area, "area")
    c_bulk_val = _validate_positive_finite(c_bulk, "c_bulk")
    f_val = _validate_positive_finite(F, "F")
    if D is not None:
        d_val: float | None = _validate_positive_finite(D, "D")
    else:
        d_val = None

    if signal_kind not in ("auto", "concentration", "potential"):
        raise ValueError(
            "signal_kind must be 'auto', 'concentration', or 'potential', "
            f"got {signal_kind!r}."
        )

    tau: float | None = None
    detection_method: Literal["concentration", "potential", "none"] = "none"

    if isinstance(result, TechniqueResult):
        if signal is not None:
            raise ValueError(
                "Do not pass 'signal' when the first argument is a TechniqueResult."
            )
        t_arr = np.asarray(result.time, dtype=float)
        if current is not None:
            i_val = _validate_finite_real(current, "current")
        elif isinstance(result.technique, Chronopotentiometry):
            nonzero_i = [v for v in result.technique.currents if v != 0.0]
            if not nonzero_i:
                raise ValueError(
                    "Applied current cannot be zero in Sand chronopotentiometry analysis."
                )
            i_val = float(nonzero_i[0])
        else:
            nz = result.current[result.current != 0.0]
            if nz.size == 0:
                raise ValueError(
                    "Applied current cannot be zero in Sand chronopotentiometry analysis."
                )
            i_val = float(nz[0])

        if i_val == 0.0:
            raise ValueError("Applied current I cannot be zero in the Sand equation.")

        # 1. Surface concentration depletion detection
        if signal_kind in ("auto", "concentration"):
            sp_name = species or result.metadata.get("reactant_species")
            if sp_name is None:
                for cand in result.species:
                    if float(result.concentrations[cand][0, 0]) > 0.0:
                        sp_name = cand
                        break
            if sp_name is not None and sp_name in result.concentrations:
                c_surf = np.asarray(result.concentrations[sp_name][:, 0], dtype=float)
                if not relative or c_surf[0] > 0.0:
                    tau = Chronopotentiometry.detect_transition_time(
                        t_arr, c_surf, threshold=threshold, relative=relative
                    )
                    if tau is not None:
                        detection_method = "concentration"

        # 2. Fallback or explicit potential inflection detection
        if tau is None and signal_kind in ("auto", "potential"):
            pot_arr = np.asarray(result.potential, dtype=float)
            tau = _detect_potential_inflection(t_arr, pot_arr)
            if tau is not None:
                detection_method = "potential"
    else:
        if signal is None:
            raise ValueError(
                "'signal' must be provided when the first argument is a time array."
            )
        if current is None:
            raise ValueError(
                "'current' must be provided when analyzing raw (t, signal) arrays."
            )
        i_val = _validate_finite_real(current, "current")
        if i_val == 0.0:
            raise ValueError("Applied current I cannot be zero in the Sand equation.")

        if signal_kind == "potential":
            t_arr, pot_arr = _validate_1d_pair(
                result, signal, "t", "potential", min_points=2, require_finite=False
            )
            if not np.all(np.isfinite(t_arr)) or np.any(np.diff(t_arr) <= 0.0):
                raise ValueError("'t' must be finite and strictly increasing.")
            tau = _detect_potential_inflection(t_arr, pot_arr)
            if tau is not None:
                detection_method = "potential"
        else:
            t_arr, c_surf = _validate_1d_pair(
                result, signal, "t", "c_surface", min_points=2
            )
            tau = Chronopotentiometry.detect_transition_time(
                t_arr, c_surf, threshold=threshold, relative=relative
            )
            if tau is not None:
                detection_method = "concentration"

    tau_th = (
        sand_transition_time(
            I=i_val, n=n_val, D=d_val, c_bulk=c_bulk_val, area=area_val, F=f_val
        )
        if d_val is not None
        else None
    )

    if tau is not None and tau > 0.0:
        sand_const: float | None = float(abs(i_val) * math.sqrt(tau))
        denom = math.pi * ((n_val * f_val * area_val * c_bulk_val) ** 2)
        d_est: float | None = float(4.0 * (i_val**2) * tau / denom)
        rel_err: float | None = (
            float(abs(tau - tau_th) / tau_th) if tau_th is not None else None
        )
    else:
        sand_const = None
        d_est = None
        rel_err = None

    return SandDiagnostics(
        tau=tau,
        tau_theoretical=tau_th,
        relative_error=rel_err,
        sand_constant=sand_const,
        D_estimated=d_est,
        current=i_val,
        detection_method=detection_method,
    )


__all__ = [
    "CVDiagnostics",
    "NicholsonDiagnostics",
    "PeakInfo",
    "PeakPolarity",
    "SandDiagnostics",
    "analyze_cottrell",
    "analyze_koutecky_levich",
    "analyze_levich",
    "analyze_nicholson",
    "analyze_randles_sevcik",
    "analyze_sand",
    "analyze_tafel",
    "find_peaks",
]
