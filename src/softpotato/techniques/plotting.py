"""Declarative Matplotlib plotting suite for electrochemical technique simulations.

Provides a unified, technique-aware plotting interface (:func:`plot_result`,
:func:`plot_summary`, :func:`plot_sweep`), focused single-plot renderers
(:func:`plot_voltammogram`, :func:`plot_current_vs_time`,
:func:`plot_potential_vs_time`, :func:`plot_concentration_profile`,
:func:`plot_charge_vs_time`, :func:`plot_dpv`, :func:`plot_swv`,
:func:`plot_scv`), and electrochemical diagnostic plotting helpers
(:func:`plot_peaks`, :func:`plot_randles_sevcik`, :func:`plot_cottrell`,
:func:`plot_levich`, :func:`plot_koutecky_levich`, :func:`plot_tafel`).

Every function follows standard Matplotlib conventions:
- Accepts an optional ``ax`` parameter to draw onto existing axes.
- Returns ``(fig, ax)`` (or ``(fig, axes)`` for :func:`plot_summary`) without
  calling ``plt.show()``.
- Uses SI units and IUPAC current polarity conventions (anodic positive,
  cathodic negative).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from softpotato.analytical.fitting import (
    CottrellFitResult,
    KouteckyLevichFitResult,
    LevichFitResult,
    RandlesSevcikFitResult,
    TafelFitResult,
)

from .base import PlotSpec, _validate_finite_real, resolve_plot_kind
from .chronopotentiometry import Chronopotentiometry
from .diagnostics import (
    CVDiagnostics,
    PeakInfo,
    PeakPolarity,
    _extract_potential_current,
    _extract_rde_series,
    find_peaks,
)
from .pulse import (
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
)
from .rde import RDESweepResult
from .result import TechniqueResult
from .voltammetry import CyclicVoltammetry

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

_PlotRenderer = Callable[..., tuple["Figure", "Axes"]]


# ==============================================================================
# Internal Helpers
# ==============================================================================


def _resolve_axes(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure/axes pair when ``ax`` is ``None``."""
    if ax is None:
        fig, created_ax = plt.subplots(figsize=figsize)
        return fig, created_ax
    fig = cast("Figure", ax.figure)
    return fig, ax


def _apply_labels(
    ax: Axes,
    spec: PlotSpec,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    """Apply axis labels, title, and optional grid to ``ax``."""
    ax.set_xlabel(spec.x_label if xlabel is None else xlabel)
    ax.set_ylabel(spec.y_label if ylabel is None else ylabel)
    resolved_title = spec.title if title is None else title
    if resolved_title:
        ax.set_title(resolved_title)
    if grid:
        ax.grid(True, alpha=0.3)


def _get_spec(result: TechniqueResult, canonical_kind: str) -> PlotSpec:
    """Retrieve the technique's PlotSpec for ``canonical_kind`` or raise ValueError."""
    specs = result.technique.plot_specs
    if canonical_kind not in specs:
        available = ", ".join(repr(k) for k in specs)
        raise ValueError(
            f"Plot kind {canonical_kind!r} is not available for "
            f"{type(result.technique).__name__}. Available plots: {available}."
        )
    return specs[canonical_kind]


def _distance_axis_label(result: TechniqueResult, spec: PlotSpec) -> str:
    """Choose radial vs. Cartesian distance label based on electrode geometry."""
    geom_name = str(result.metadata.get("geometry", "")).lower()
    if geom_name in ("spherical", "cylindrical"):
        return "Radial distance, $r$ / m"
    return spec.x_label


def _overlay_peaks_on_axes(
    ax: Axes,
    peaks: CVDiagnostics | PeakInfo | Sequence[PeakInfo],
) -> None:
    """Draw peak markers onto ``ax`` from CVDiagnostics or PeakInfo sequence."""
    peak_list: list[PeakInfo] = []
    if isinstance(peaks, CVDiagnostics):
        if peaks.anodic_peak is not None:
            peak_list.append(peaks.anodic_peak)
        if peaks.cathodic_peak is not None:
            peak_list.append(peaks.cathodic_peak)
    elif isinstance(peaks, PeakInfo):
        peak_list.append(peaks)
    else:
        peak_list.extend(peaks)

    for peak in peak_list:
        is_anodic = peak.polarity == "anodic"
        marker = "^" if is_anodic else "v"
        color = "tab:red" if is_anodic else "tab:blue"
        tag = "Anodic" if is_anodic else "Cathodic"
        label = f"{tag} peak ($E_p$ = {peak.E_peak:.3g} V, $I_p$ = {peak.I_peak:.3g} A)"
        ax.scatter(
            [peak.E_peak],
            [peak.I_peak],
            marker=marker,
            color=color,
            s=55,
            zorder=5,
            label=label,
        )


def _rde_sweep_to_technique_results(
    sweep_result: RDESweepResult,
) -> list[tuple[str, TechniqueResult]]:
    """Convert an RDESweepResult into labeled TechniqueResult objects for plotting."""
    items: list[tuple[str, TechniqueResult]] = []
    for rpm, tech, sr in zip(
        sweep_result.rpms, sweep_result.techniques, sweep_result.results
    ):
        e_arr = np.asarray(tech.potential(sr.t), dtype=float)
        i_arr = np.asarray(sr.currents[sweep_result.species], dtype=float)
        tr = TechniqueResult(
            technique=tech,
            solver_result=sr,
            time=sr.t,
            potential=e_arr,
            current=i_arr,
            metadata={"geometry": "planar", "rpm": float(rpm)},
        )
        items.append((f"{rpm:.0f} rpm", tr))
    return items


# ==============================================================================
# Core Single-Result Plot Renderers
# ==============================================================================


def plot_voltammogram(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    cycle: int | Literal["all"] | None = None,
    peaks: bool | CVDiagnostics | PeakInfo | Sequence[PeakInfo] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot Faradaic current vs. electrode potential (:math:`I` vs. :math:`E`).

    Parameters
    ----------
    result : TechniqueResult
        Simulation result to plot.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    cycle : int, 'all', or None, default None
        For :class:`CyclicVoltammetry`, select a specific cycle index (e.g.
        ``-1`` for the final cycle), ``'all'`` to plot each cycle as a separate
        labeled trace, or ``None`` to plot the continuous trace.
    peaks : bool, CVDiagnostics, PeakInfo, or Sequence[PeakInfo], optional
        If ``True``, runs :func:`find_peaks` and overlays detected peaks.
        If a :class:`CVDiagnostics` or :class:`PeakInfo` (sequence) is given,
        overlays those peaks directly.
    title, xlabel, ylabel : str, optional
        Optional overrides for the default :class:`PlotSpec` labels.
    grid : bool, default True
        Whether to show a light background grid.
    figsize : tuple[float, float], optional
        Figure size passed to ``plt.subplots`` when ``ax`` is ``None``.
    **plot_kwargs : Any
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the voltammogram.
    """
    spec = result.technique.plot_specs.get(
        "voltammogram",
        PlotSpec(
            kind="voltammogram",
            title="Voltammogram",
            x_label="Potential, $E$ / V",
            y_label="Current, $I$ / A",
        ),
    )
    fig, ax = _resolve_axes(ax, figsize=figsize)

    if isinstance(result.technique, CyclicVoltammetry) and cycle is not None:
        if cycle == "all":
            for idx, cyc in enumerate(result.cycles):
                kw = dict(plot_kwargs)
                kw.setdefault("label", f"Cycle {idx + 1}")
                ax.plot(cyc.potential, cyc.current, **kw)
            ax.legend()
        elif isinstance(cycle, int) and not isinstance(cycle, bool):
            cyc = result.cycles[cycle]
            ax.plot(cyc.potential, cyc.current, **plot_kwargs)
        else:
            raise ValueError(
                f"cycle must be an int, 'all', or None, got {cycle!r}."
            )
    else:
        ax.plot(result.potential, result.current, **plot_kwargs)

    if peaks is True:
        peak_cycle = cycle if isinstance(cycle, int) else -1
        detected = find_peaks(result, cycle=peak_cycle)
        _overlay_peaks_on_axes(ax, detected)
        ax.legend()
    elif peaks is not None and peaks is not False:
        _overlay_peaks_on_axes(ax, peaks)
        ax.legend()

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_current_vs_time(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    step: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot net Faradaic current vs. time (:math:`I(t)` vs. :math:`t`).

    Handles both 1D current arrays and 2D :class:`SampledCurrentVoltammetry`
    step families (using ``step`` to select a single step or plotting all steps).
    """
    spec = result.technique.plot_specs["i_vs_t"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    if result.current.ndim == 2:
        if step is not None:
            kw = dict(plot_kwargs)
            if isinstance(result.technique, SampledCurrentVoltammetry):
                kw.setdefault("label", f"$E$ = {result.technique.potentials[step]:.3g} V")
            ax.plot(result.time, result.current[step], **kw)
            if "label" in kw:
                ax.legend()
        else:
            potentials = (
                result.technique.potentials
                if isinstance(result.technique, SampledCurrentVoltammetry)
                else None
            )
            for idx in range(result.current.shape[0]):
                kw = dict(plot_kwargs)
                if potentials is not None:
                    kw.setdefault("label", f"$E$ = {potentials[idx]:.3g} V")
                else:
                    kw.setdefault("label", f"Step {idx + 1}")
                ax.plot(result.time, result.current[idx], **kw)
            ax.legend()
    else:
        ax.plot(result.time, result.current, **plot_kwargs)

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_potential_vs_time(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    step: int | None = None,
    show_transition_time: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot electrode potential vs. time (:math:`E(t)` vs. :math:`t`).

    For :class:`Chronopotentiometry`, optionally marks the detected transition
    time :math:`\\tau` with a vertical dashed reference line when
    ``show_transition_time=True``.
    """
    spec = result.technique.plot_specs["E_vs_t"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    if result.potential.ndim == 2:
        if step is not None:
            ax.plot(result.time, result.potential[step], **plot_kwargs)
        else:
            potentials = (
                result.technique.potentials
                if isinstance(result.technique, SampledCurrentVoltammetry)
                else None
            )
            for idx in range(result.potential.shape[0]):
                kw = dict(plot_kwargs)
                if potentials is not None:
                    kw.setdefault("label", f"$E$ = {potentials[idx]:.3g} V")
                else:
                    kw.setdefault("label", f"Step {idx + 1}")
                ax.plot(result.time, result.potential[idx], **kw)
            ax.legend()
    else:
        ax.plot(result.time, result.potential, **plot_kwargs)

    if (
        show_transition_time
        and isinstance(result.technique, Chronopotentiometry)
        and result.transition_time is not None
    ):
        tau = result.transition_time
        ax.axvline(
            tau,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=rf"$\tau$ = {tau:.4g} s",
        )
        ax.legend()

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_concentration_profile(
    result: TechniqueResult,
    *,
    t: float | Sequence[float] | None = None,
    species: str | Sequence[str] | None = None,
    step: int | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot species concentration vs. spatial distance at specific time(s).

    Parameters
    ----------
    result : TechniqueResult
        Simulation result containing spatial concentration profiles.
    t : float or Sequence[float], optional
        Target timestamp(s) in seconds. Defaults to the final simulated time
        ``result.time[-1]``. Each requested timestamp is snapped to the nearest
        simulated evaluation point.
    species : str or Sequence[str], optional
        Species name(s) to plot. Defaults to all species in ``result.species``.
    step : int, optional
        For :class:`SampledCurrentVoltammetry`, select which step's
        concentration profile to plot from ``result.metadata['step_results']``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    title, xlabel, ylabel : str, optional
        Optional overrides for the default :class:`PlotSpec` labels.
    grid : bool, default True
        Whether to show a light background grid.
    figsize : tuple[float, float], optional
        Figure size passed to ``plt.subplots`` when ``ax`` is ``None``.
    **plot_kwargs : Any
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the concentration profile(s).
    """
    target_result = result
    if step is not None and isinstance(result.technique, SampledCurrentVoltammetry):
        step_results = result.metadata.get("step_results")
        if step_results is not None:
            target_result = step_results[step]

    spec = target_result.technique.plot_specs["c_vs_x"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    # Resolve species list
    if species is None:
        selected_species = list(target_result.species)
    elif isinstance(species, str):
        selected_species = [species]
    else:
        selected_species = list(species)
    if not selected_species:
        raise ValueError("species must not be empty.")
    for sp_name in selected_species:
        if sp_name not in target_result:
            raise KeyError(
                f"Species {sp_name!r} not found in result. "
                f"Available species: {target_result.species}."
            )

    # Resolve target time(s)
    t_arr = target_result.time
    if t is None:
        t_targets = [float(t_arr[-1])]
    elif np.isscalar(t):
        t_targets = [_validate_finite_real(t, "t")]
    else:
        if isinstance(t, (str, bytes)):
            raise TypeError("'t' must be a float or sequence of floats.")
        t_targets = [_validate_finite_real(val, "t") for val in t]
        if not t_targets:
            raise ValueError("'t' must not be empty.")

    t_min = float(t_arr[0])
    t_max = float(t_arr[-1])
    tol = 1e-12 * max(1.0, abs(t_max - t_min))
    snapped_indices: list[int] = []
    for t_val in t_targets:
        if t_val < t_min - tol or t_val > t_max + tol:
            raise ValueError(
                f"Requested time t={t_val!r} s lies outside simulated range "
                f"[{t_min!r}, {t_max!r}] s."
            )
        snapped_indices.append(int(np.argmin(np.abs(t_arr - t_val))))

    x_grid = target_result.x
    multi_time = len(snapped_indices) > 1
    for t_idx in snapped_indices:
        t_actual = float(t_arr[t_idx])
        for sp_name in selected_species:
            c_prof = target_result[sp_name][t_idx]
            kw = dict(plot_kwargs)
            if multi_time:
                kw.setdefault("label", f"{sp_name} ($t$ = {t_actual:.4g} s)")
            else:
                kw.setdefault("label", sp_name)
            ax.plot(x_grid, c_prof, **kw)

    ax.legend()

    resolved_title = title
    if resolved_title is None:
        if not multi_time:
            t_actual = float(t_arr[snapped_indices[0]])
            resolved_title = f"{spec.title} ($t$ = {t_actual:.4g} s)"
        else:
            resolved_title = spec.title

    resolved_xlabel = (
        _distance_axis_label(target_result, spec) if xlabel is None else xlabel
    )
    _apply_labels(
        ax,
        spec,
        title=resolved_title,
        xlabel=resolved_xlabel,
        ylabel=ylabel,
        grid=grid,
    )
    return fig, ax


def plot_charge_vs_time(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    step: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot cumulative Faradaic charge vs. time (:math:`Q(t)` vs. :math:`t`)."""
    spec = result.technique.plot_specs.get(
        "Q_vs_t",
        PlotSpec(
            kind="Q_vs_t",
            title="Chronocoulogram",
            x_label="Time, $t$ / s",
            y_label="Charge, $Q$ / C",
        ),
    )
    fig, ax = _resolve_axes(ax, figsize=figsize)

    q_arr = result.Q
    if q_arr.ndim == 2:
        if step is not None:
            ax.plot(result.time, q_arr[step], **plot_kwargs)
        else:
            for idx in range(q_arr.shape[0]):
                kw = dict(plot_kwargs)
                kw.setdefault("label", f"Step {idx + 1}")
                ax.plot(result.time, q_arr[idx], **kw)
            ax.legend()
    else:
        ax.plot(result.time, q_arr, **plot_kwargs)

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_dpv(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    show_components: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a Differential Pulse Voltammogram (:math:`\\Delta I` vs. :math:`E_{\\mathrm{base}}`)."""
    if not isinstance(result.technique, DifferentialPulseVoltammetry):
        raise TypeError(
            "plot_dpv requires a DifferentialPulseVoltammetry TechniqueResult, "
            f"got {type(result.technique).__name__}."
        )
    spec = result.technique.plot_specs["dpv"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    kw = dict(plot_kwargs)
    kw.setdefault("label", r"$\Delta I = I_{\mathrm{pulse}} - I_{\mathrm{pre}}$")
    ax.plot(result.E_base, result.i_diff, **kw)

    if show_components:
        ax.plot(
            result.E_base,
            result.i_pulse,
            linestyle="--",
            alpha=0.75,
            label=r"$I_{\mathrm{pulse}}$",
        )
        ax.plot(
            result.E_base,
            result.i_pre,
            linestyle=":",
            alpha=0.75,
            label=r"$I_{\mathrm{pre}}$",
        )
        ax.legend()
    elif "label" in plot_kwargs:
        ax.legend()

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_swv(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    show_components: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a Square-Wave Voltammogram (:math:`I_{\\mathrm{net}}`, :math:`I_f`, :math:`I_r` vs. :math:`E_{\\mathrm{step}}`)."""
    if not isinstance(result.technique, SquareWaveVoltammetry):
        raise TypeError(
            "plot_swv requires a SquareWaveVoltammetry TechniqueResult, "
            f"got {type(result.technique).__name__}."
        )
    spec = result.technique.plot_specs["swv"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    kw = dict(plot_kwargs)
    kw.setdefault("label", r"$I_{\mathrm{net}}$")
    ax.plot(result.E_step, result.i_net, **kw)

    if show_components:
        ax.plot(
            result.E_step,
            result.i_forward,
            linestyle="--",
            alpha=0.75,
            label=r"$I_{\mathrm{forward}}$",
        )
        ax.plot(
            result.E_step,
            result.i_reverse,
            linestyle=":",
            alpha=0.75,
            label=r"$I_{\mathrm{reverse}}$",
        )
        ax.legend()
    elif "label" in plot_kwargs:
        ax.legend()

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


def plot_scv(
    result: TechniqueResult,
    *,
    ax: Axes | None = None,
    sample_times: float | Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a Sampled-Current Voltammogram (:math:`I(\\tau)` vs. :math:`E_{\\mathrm{step}}`)."""
    if not isinstance(result.technique, SampledCurrentVoltammetry):
        raise TypeError(
            "plot_scv requires a SampledCurrentVoltammetry TechniqueResult, "
            f"got {type(result.technique).__name__}."
        )
    spec = result.technique.plot_specs["scv"]
    fig, ax = _resolve_axes(ax, figsize=figsize)

    sample_times_arg: float | list[float] | tuple[float, ...] | np.ndarray | None
    if sample_times is None or isinstance(sample_times, (float, int, list, tuple, np.ndarray)):
        sample_times_arg = (
            float(sample_times) if isinstance(sample_times, int) else sample_times
        )
    else:
        sample_times_arg = list(sample_times)

    e_steps, i_sampled = result.sample_currents(sample_times=sample_times_arg)
    if i_sampled.ndim == 1:
        kw = dict(plot_kwargs)
        kw.setdefault("marker", "o")
        ax.plot(e_steps, i_sampled, **kw)
    else:
        times_used: tuple[float, ...]
        if sample_times is None:
            raw_times = result.technique.sample_times
            times_used = () if raw_times is None else tuple(float(x) for x in raw_times)
        elif np.isscalar(sample_times):
            times_used = (float(sample_times),)  # type: ignore[arg-type]
        else:
            times_used = tuple(float(x) for x in sample_times)  # type: ignore[union-attr]
        for idx in range(i_sampled.shape[0]):
            kw = dict(plot_kwargs)
            kw.setdefault("marker", "o")
            if idx < len(times_used):
                kw.setdefault("label", rf"$\tau$ = {times_used[idx]:.4g} s")
            else:
                kw.setdefault("label", f"Sample {idx + 1}")
            ax.plot(e_steps, i_sampled[idx], **kw)
        ax.legend()

    _apply_labels(ax, spec, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    return fig, ax


# ==============================================================================
# Renderer Registry & Central Dispatcher
# ==============================================================================


_PLOT_RENDERERS: dict[str, _PlotRenderer] = {
    "voltammogram": plot_voltammogram,
    "i_vs_t": plot_current_vs_time,
    "E_vs_t": plot_potential_vs_time,
    "c_vs_x": plot_concentration_profile,
    "Q_vs_t": plot_charge_vs_time,
    "dpv": plot_dpv,
    "swv": plot_swv,
    "scv": plot_scv,
}


def plot_result(
    result: TechniqueResult,
    kind: str | None = None,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Dispatch a plot for ``result`` using the technique's declarative plot specs.

    Parameters
    ----------
    result : TechniqueResult
        Output from :func:`~softpotato.techniques.simulate`.
    kind : str, optional
        Concise plot key (e.g. ``'voltammogram'``, ``'i_vs_t'``, ``'E_vs_t'``,
        ``'c_vs_x'``, ``'Q_vs_t'``, ``'dpv'``, ``'swv'``, ``'scv'``) or verbose
        alias (e.g. ``'current_vs_potential'``, ``'current_vs_time'``,
        ``'potential_vs_time'``, ``'concentration_profile'``). Defaults to
        ``result.default_plot``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    **kwargs : Any
        Additional keyword arguments forwarded to the plot renderer.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the rendered plot.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`TechniqueResult` or ``kind`` is not a string.
    ValueError
        If ``kind`` is not supported by ``result.technique``.
    """
    if not isinstance(result, TechniqueResult):
        raise TypeError(
            f"result must be a TechniqueResult instance, got {type(result).__name__}."
        )

    canonical = (
        result.default_plot if kind is None else resolve_plot_kind(kind)
    )
    _get_spec(result, canonical)
    renderer = _PLOT_RENDERERS[canonical]
    return renderer(result, ax=ax, **kwargs)


# ==============================================================================
# Multi-Panel Summary & Parameter Sweep Overlays
# ==============================================================================


def plot_summary(
    result: TechniqueResult,
    *,
    t: float | Sequence[float] | None = None,
    species: str | Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Render a multi-panel summary figure for a :class:`TechniqueResult`.

    Panels include the technique's default plot together with the universal
    views (``'i_vs_t'``, ``'E_vs_t'``, ``'c_vs_x'``), deduplicated in order
    (producing a 1x3 layout when the default plot is already ``'i_vs_t'`` or
    ``'E_vs_t'``, or a 2x2 layout otherwise).

    Parameters
    ----------
    result : TechniqueResult
        Simulation result to summarize.
    t : float or Sequence[float], optional
        Target time(s) in seconds passed to the ``'c_vs_x'`` panel.
    species : str or Sequence[str], optional
        Species subset passed to the ``'c_vs_x'`` panel.
    figsize : tuple[float, float], optional
        Optional figure size override.
    **kwargs : Any
        Additional keyword arguments forwarded to each panel's :func:`plot_result`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and array of Axes objects.
    """
    if not isinstance(result, TechniqueResult):
        raise TypeError(
            f"result must be a TechniqueResult instance, got {type(result).__name__}."
        )

    panel_kinds: list[str] = []
    for k in (result.default_plot, "i_vs_t", "E_vs_t", "c_vs_x"):
        if k not in panel_kinds:
            panel_kinds.append(k)

    if len(panel_kinds) == 3:
        nrows, ncols = 1, 3
        default_size = (13.5, 4.0)
    else:
        nrows, ncols = 2, 2
        default_size = (10.5, 8.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=default_size if figsize is None else figsize,
    )
    axes_arr = np.asarray(axes)
    flat_axes = axes_arr.ravel()

    for ax_item, panel_kind in zip(flat_axes, panel_kinds):
        panel_kwargs = dict(kwargs)
        if panel_kind == "c_vs_x":
            if t is not None:
                panel_kwargs["t"] = t
            if species is not None:
                panel_kwargs["species"] = species
        plot_result(result, kind=panel_kind, ax=ax_item, **panel_kwargs)

    fig.tight_layout()
    return fig, axes_arr


def plot_sweep(
    results: Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | RDESweepResult,
    kind: str | None = None,
    *,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    legend_title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Overlay a series of simulation results on a single set of axes.

    Accepts output from :func:`~softpotato.techniques.simulate_sweep` (either
    a ``Sequence[TechniqueResult]`` or ``Mapping[Any, TechniqueResult]``) or
    :class:`~softpotato.techniques.RDESweepResult`.

    Parameters
    ----------
    results : Sequence[TechniqueResult], Mapping[Any, TechniqueResult], or RDESweepResult
        Collection of simulation results to overlay.
    kind : str, optional
        Concise plot key or verbose alias. Defaults to the first result's
        ``default_plot``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    labels : Sequence[str], optional
        Custom legend labels matching the length of ``results``.
    legend_title : str, optional
        Optional title for the legend.
    figsize : tuple[float, float], optional
        Figure size passed to ``plt.subplots`` when ``ax`` is ``None``.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`plot_result`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the overlaid sweep traces.
    """
    labeled_items: list[tuple[str, TechniqueResult]] = []
    if isinstance(results, RDESweepResult):
        labeled_items = _rde_sweep_to_technique_results(results)
    elif isinstance(results, Mapping):
        if len(results) == 0:
            raise ValueError("'results' mapping must not be empty.")
        for key, res in results.items():
            if not isinstance(res, TechniqueResult):
                raise TypeError(
                    f"Expected TechniqueResult values, got {type(res).__name__}."
                )
            labeled_items.append((str(key), res))
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        if len(results) == 0:
            raise ValueError("'results' sequence must not be empty.")
        for idx, res in enumerate(results):
            if not isinstance(res, TechniqueResult):
                raise TypeError(
                    f"Expected TechniqueResult entries, got {type(res).__name__}."
                )
            tech = res.technique
            if hasattr(tech, "scan_rate"):
                auto_label = f"$v$ = {tech.scan_rate:.4g} V/s"
            elif hasattr(tech, "rpm"):
                auto_label = f"{tech.rpm:.0f} rpm"
            elif hasattr(tech, "omega"):
                auto_label = rf"$\omega$ = {tech.omega:.4g} rad/s"
            else:
                auto_label = f"Run {idx + 1}"
            labeled_items.append((auto_label, res))
    else:
        raise TypeError(
            "results must be a Sequence/Mapping of TechniqueResult or an "
            f"RDESweepResult, got {type(results).__name__}."
        )

    if labels is not None:
        if len(labels) != len(labeled_items):
            raise ValueError(
                f"Length of 'labels' ({len(labels)}) must match number of "
                f"results ({len(labeled_items)})."
            )
        labeled_items = [
            (str(lbl), res) for lbl, (_, res) in zip(labels, labeled_items)
        ]

    fig, ax = _resolve_axes(ax, figsize=figsize)
    resolved_kind = kind if kind is not None else labeled_items[0][1].default_plot

    for lbl, res in labeled_items:
        item_kwargs = dict(kwargs)
        item_kwargs["label"] = lbl
        # Avoid cluttering sweep overlays with multiple component lines unless requested
        canonical = resolve_plot_kind(resolved_kind)
        if canonical == "swv":
            item_kwargs.setdefault("show_components", False)
        plot_result(res, kind=resolved_kind, ax=ax, **item_kwargs)

    ax.legend(title=legend_title)
    return fig, ax


# ==============================================================================
# Electrochemical Diagnostic Plotting Helpers
# ==============================================================================


def plot_peaks(
    result: TechniqueResult,
    peaks: CVDiagnostics | PeakInfo | Sequence[PeakInfo] | None = None,
    *,
    cycle: int | None = -1,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a voltammogram with annotated anodic and cathodic peak diagnostics.

    Parameters
    ----------
    result : TechniqueResult
        Voltammetric simulation result (CV, LSV, SWV, DPV).
    peaks : CVDiagnostics, PeakInfo, or Sequence[PeakInfo], optional
        Pre-computed peak diagnostics from :func:`find_peaks`. If ``None``,
        automatically runs ``find_peaks(result, cycle=cycle)``.
    cycle : int or None, default -1
        Cycle index for :class:`CyclicVoltammetry` (``-1`` selects the final
        cycle; ``None`` plots the full trace).
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    title : str, optional
        Optional title override. When ``None`` and ``peaks`` is a
        :class:`CVDiagnostics` with ``delta_Ep``, includes :math:`\\Delta E_p`
        in the title.
    grid : bool, default True
        Whether to display a grid.
    figsize : tuple[float, float], optional
        Figure size when ``ax`` is ``None``.
    **plot_kwargs : Any
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes with the voltammogram and peak markers.
    """
    e_arr, i_arr = _extract_potential_current(result, cycle=cycle)
    resolved_peaks = find_peaks(result, cycle=cycle) if peaks is None else peaks

    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.plot(e_arr, i_arr, **plot_kwargs)
    _overlay_peaks_on_axes(ax, resolved_peaks)
    ax.legend()

    resolved_title = title
    if resolved_title is None:
        base_title = "Voltammetric Peak Diagnostics"
        if isinstance(resolved_peaks, CVDiagnostics) and resolved_peaks.delta_Ep is not None:
            resolved_title = (
                rf"{base_title} ($\Delta E_p$ = {resolved_peaks.delta_Ep * 1e3:.1f} mV)"
            )
        else:
            resolved_title = base_title

    spec = PlotSpec(
        kind="voltammogram",
        title=resolved_title,
        x_label="Potential, $E$ / V",
        y_label="Current, $I$ / A",
    )
    _apply_labels(ax, spec, title=resolved_title, grid=grid)
    return fig, ax


def plot_randles_sevcik(
    results_or_v: Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | ArrayLike,
    fit_result: RandlesSevcikFitResult,
    *,
    peak_currents: ArrayLike | None = None,
    polarity: PeakPolarity = "cathodic",
    cycle: int | None = -1,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a Randles–Ševčík diagnostic regression (:math:`|I_p|` vs. :math:`v^{1/2}`).

    Parameters
    ----------
    results_or_v : Sequence[TechniqueResult], Mapping[Any, TechniqueResult], or array-like
        Collection of CV/LSV results or raw 1D array of scan rates :math:`v` (V/s).
    fit_result : RandlesSevcikFitResult
        Fit output from :func:`~softpotato.techniques.analyze_randles_sevcik`
        or :func:`~softpotato.analytical.fitting.fit_randles_sevcik`.
    peak_currents : array-like, optional
        Optional raw peak current array when ``results_or_v`` is a scan-rate array.
        If omitted, reconstructed from ``fit_result.y_fit + fit_result.residuals``.
    polarity : {'cathodic', 'anodic'}, default 'cathodic'
        Peak polarity label used when extracting from ``TechniqueResult`` objects.
    cycle : int or None, default -1
        Unused placeholder for API parity with :func:`analyze_randles_sevcik`.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    title : str, optional
        Optional plot title override.
    grid : bool, default True
        Whether to display a background grid.
    figsize : tuple[float, float], optional
        Figure size when ``ax`` is ``None``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the Randles–Ševčík plot.
    """
    _ = (polarity, cycle)
    if isinstance(results_or_v, Mapping):
        v_list: list[float] = []
        for key, res in results_or_v.items():
            sr = getattr(res.technique, "scan_rate", None)
            v_list.append(float(sr if sr is not None else key))
        v_arr = np.asarray(v_list, dtype=float)
    elif (
        isinstance(results_or_v, Sequence)
        and not isinstance(results_or_v, (str, bytes))
        and len(results_or_v) > 0
        and isinstance(results_or_v[0], TechniqueResult)
    ):
        tr_seq = [r for r in results_or_v if isinstance(r, TechniqueResult)]
        v_arr = np.asarray(
            [float(getattr(r.technique, "scan_rate", np.nan)) for r in tr_seq],
            dtype=float,
        )
    else:
        v_arr = np.asarray(results_or_v, dtype=float)

    sqrt_v = np.sqrt(v_arr)
    y_fit = np.asarray(fit_result.y_fit, dtype=float)
    y_data = (
        np.asarray(peak_currents, dtype=float)
        if peak_currents is not None
        else y_fit + np.asarray(fit_result.residuals, dtype=float)
    )
    order = np.argsort(sqrt_v)

    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.scatter(sqrt_v, y_data, color="tab:blue", zorder=4, label="Data")
    ax.plot(
        sqrt_v[order],
        y_fit[order],
        color="tab:red",
        linewidth=1.5,
        label=rf"Fit ($D$ = {fit_result.d:.3e}, $R^2$ = {fit_result.r_squared:.4f})",
    )
    ax.legend()

    spec = PlotSpec(
        kind="randles_sevcik",
        title="Randles–Ševčík Analysis",
        x_label=r"$\sqrt{v}$ / $\mathrm{V^{1/2}\,s^{-1/2}}$",
        y_label=r"Peak Current, $I_p$ / A",
    )
    _apply_labels(ax, spec, title=title, grid=grid)
    return fig, ax


def plot_cottrell(
    result_or_t: TechniqueResult | ArrayLike,
    fit_result: CottrellFitResult,
    *,
    current: ArrayLike | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a Cottrell diagnostic regression (:math:`I` vs. :math:`t^{-1/2}`).

    Parameters
    ----------
    result_or_t : TechniqueResult or array-like
        Chronoamperometry :class:`TechniqueResult` or 1D time array :math:`t` (s).
    fit_result : CottrellFitResult
        Fit output from :func:`~softpotato.techniques.analyze_cottrell` or
        :func:`~softpotato.analytical.fitting.fit_cottrell`.
    current : array-like, optional
        Optional raw current array when ``result_or_t`` is a time array.
        If omitted, reconstructed from ``fit_result.y_fit + fit_result.residuals``.
    t_min, t_max : float, optional
        Time window bounds matching those passed to :func:`analyze_cottrell`.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw onto.
    title : str, optional
        Optional plot title override.
    grid : bool, default True
        Whether to display a background grid.
    figsize : tuple[float, float], optional
        Figure size when ``ax`` is ``None``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and Axes containing the Cottrell plot.
    """
    if isinstance(result_or_t, TechniqueResult):
        t_arr = np.asarray(result_or_t.time, dtype=float)
    else:
        t_arr = np.asarray(result_or_t, dtype=float)

    mask = t_arr > 0.0
    if t_min is not None:
        mask &= t_arr >= t_min
    if t_max is not None:
        mask &= t_arr <= t_max
    t_sel = t_arr[mask]

    inv_sqrt_t = 1.0 / np.sqrt(t_sel)
    y_fit = np.asarray(fit_result.y_fit, dtype=float)
    if current is not None:
        curr_arr = np.asarray(current, dtype=float)
        y_data = curr_arr[mask] if curr_arr.size == t_arr.size else curr_arr
    else:
        y_data = y_fit + np.asarray(fit_result.residuals, dtype=float)

    order = np.argsort(inv_sqrt_t)
    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.scatter(inv_sqrt_t, y_data, s=18, color="tab:blue", zorder=4, label="Data")
    ax.plot(
        inv_sqrt_t[order],
        y_fit[order],
        color="tab:red",
        linewidth=1.5,
        label=rf"Fit ($D$ = {fit_result.d:.3e}, $R^2$ = {fit_result.r_squared:.4f})",
    )
    ax.legend()

    spec = PlotSpec(
        kind="cottrell",
        title="Cottrell Analysis",
        x_label=r"$t^{-1/2}$ / $\mathrm{s^{-1/2}}$",
        y_label="Current, $I$ / A",
    )
    _apply_labels(ax, spec, title=title, grid=grid)
    return fig, ax


def plot_levich(
    results_or_omega: RDESweepResult
    | Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | ArrayLike,
    fit_result: LevichFitResult,
    *,
    currents: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a Levich RDE diagnostic regression (:math:`I_L` vs. :math:`\omega^{1/2}`)."""
    if isinstance(results_or_omega, (RDESweepResult, Mapping)) or (
        isinstance(results_or_omega, Sequence)
        and not isinstance(results_or_omega, (str, bytes))
        and len(results_or_omega) > 0
        and isinstance(results_or_omega[0], TechniqueResult)
    ):
        omega_arr, _, _ = _extract_rde_series(results_or_omega)  # type: ignore[arg-type]
    else:
        omega_arr = np.asarray(results_or_omega, dtype=float)

    sqrt_omega = np.sqrt(omega_arr)
    y_fit = np.asarray(fit_result.y_fit, dtype=float)
    y_data = (
        np.asarray(currents, dtype=float)
        if currents is not None
        else y_fit + np.asarray(fit_result.residuals, dtype=float)
    )
    order = np.argsort(sqrt_omega)

    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.scatter(sqrt_omega, y_data, color="tab:blue", zorder=4, label="Data")
    ax.plot(
        sqrt_omega[order],
        y_fit[order],
        color="tab:red",
        linewidth=1.5,
        label=rf"Fit ($D$ = {fit_result.d:.3e}, $R^2$ = {fit_result.r_squared:.4f})",
    )
    ax.legend()

    spec = PlotSpec(
        kind="levich",
        title="Levich Analysis",
        x_label=r"$\omega^{1/2}$ / $\mathrm{rad^{1/2}\,s^{-1/2}}$",
        y_label=r"Limiting Current, $I_L$ / A",
    )
    _apply_labels(ax, spec, title=title, grid=grid)
    return fig, ax


def plot_koutecky_levich(
    results_or_omega: RDESweepResult
    | Sequence[TechniqueResult]
    | Mapping[Any, TechniqueResult]
    | ArrayLike,
    fit_result: KouteckyLevichFitResult,
    *,
    currents: ArrayLike | None = None,
    potential: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a Koutecký–Levich diagnostic regression (:math:`I^{-1}` vs. :math:`\omega^{-1/2}`)."""
    if isinstance(results_or_omega, (RDESweepResult, Mapping)) or (
        isinstance(results_or_omega, Sequence)
        and not isinstance(results_or_omega, (str, bytes))
        and len(results_or_omega) > 0
        and isinstance(results_or_omega[0], TechniqueResult)
    ):
        omega_arr, _, _ = _extract_rde_series(
            results_or_omega, potential=potential  # type: ignore[arg-type]
        )
    else:
        omega_arr = np.asarray(results_or_omega, dtype=float)

    inv_sqrt_omega = 1.0 / np.sqrt(omega_arr)
    y_fit = np.asarray(fit_result.y_fit, dtype=float)
    if currents is not None:
        y_data = 1.0 / np.abs(np.asarray(currents, dtype=float))
    else:
        y_data = y_fit + np.asarray(fit_result.residuals, dtype=float)
    order = np.argsort(inv_sqrt_omega)

    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.scatter(inv_sqrt_omega, y_data, color="tab:blue", zorder=4, label="Data")
    ax.plot(
        inv_sqrt_omega[order],
        y_fit[order],
        color="tab:red",
        linewidth=1.5,
        label=rf"Fit ($I_K$ = {fit_result.i_k:.3e} A, $R^2$ = {fit_result.r_squared:.4f})",
    )
    ax.legend()

    spec = PlotSpec(
        kind="koutecky_levich",
        title="Koutecký–Levich Analysis",
        x_label=r"$\omega^{-1/2}$ / $\mathrm{rad^{-1/2}\,s^{1/2}}$",
        y_label=r"Inverse Current, $|I|^{-1}$ / $\mathrm{A^{-1}}$",
    )
    _apply_labels(ax, spec, title=title, grid=grid)
    return fig, ax


def plot_tafel(
    overpotential_or_result: TechniqueResult | ArrayLike,
    fit_result: TafelFitResult,
    *,
    current: ArrayLike | None = None,
    E0: float | None = None,
    overpotential_window: tuple[float, float] | None = None,
    branch: Literal["cathodic", "anodic"] = "cathodic",
    cycle: int | None = -1,
    ax: Axes | None = None,
    title: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a Tafel diagnostic regression (:math:`\eta` vs. :math:`\log_{10}|I|`)."""
    if isinstance(overpotential_or_result, TechniqueResult):
        if E0 is None:
            raise ValueError(
                "'E0' must be provided when plotting Tafel analysis from a TechniqueResult."
            )
        e_arr, i_arr = _extract_potential_current(overpotential_or_result, cycle=cycle)
        eta_all = e_arr - float(E0)
        if branch == "anodic":
            mask = (eta_all > 0.0) & (i_arr > 0.0)
        else:
            mask = (eta_all < 0.0) & (i_arr < 0.0)
        if overpotential_window is not None:
            lo, hi = overpotential_window
            mask &= (eta_all >= lo) & (eta_all <= hi)
        log_i = np.log10(np.abs(i_arr[mask]))
    else:
        if current is None:
            raise ValueError(
                "'current' must be provided when the first argument is an overpotential array."
            )
        eta_all = np.asarray(overpotential_or_result, dtype=float)
        i_arr = np.asarray(current, dtype=float)
        mask = np.isfinite(eta_all) & np.isfinite(i_arr) & (np.abs(i_arr) > 0.0)
        if overpotential_window is not None:
            lo, hi = overpotential_window
            mask &= (eta_all >= lo) & (eta_all <= hi)
        log_i = np.log10(np.abs(i_arr[mask]))

    y_fit = np.asarray(fit_result.y_fit, dtype=float)
    y_data = y_fit + np.asarray(fit_result.residuals, dtype=float)
    if log_i.size != y_fit.size:
        raise ValueError(
            f"Filtered data length ({log_i.size}) does not match fit_result.y_fit "
            f"length ({y_fit.size}). Ensure E0, branch, and overpotential_window "
            "match those used in analyze_tafel()."
        )

    order = np.argsort(log_i)
    fig, ax = _resolve_axes(ax, figsize=figsize)
    ax.scatter(log_i, y_data, s=18, color="tab:blue", zorder=4, label="Data")
    slope_mv = (
        fit_result.slope_mv
        if math.isfinite(fit_result.slope_mv)
        else fit_result.slope * 1e3
    )
    ax.plot(
        log_i[order],
        y_fit[order],
        color="tab:red",
        linewidth=1.5,
        label=rf"Fit ($b$ = {slope_mv:.1f} mV/dec, $I_0$ = {fit_result.i0:.2e} A)",
    )
    ax.legend()

    spec = PlotSpec(
        kind="tafel",
        title="Tafel Analysis",
        x_label=r"$\log_{10}(|I| / \mathrm{A})$",
        y_label=r"Overpotential, $\eta$ / V",
    )
    _apply_labels(ax, spec, title=title, grid=grid)
    return fig, ax
