"""High-level simulation pipeline orchestrator (``simulate`` and ``simulate_sweep``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np

from softpotato.constants import FARADAY
from softpotato.geometry import Geometry, get_geometry
from softpotato.grid import Mesh1D, create_diffusion_mesh
from softpotato.reactions.heterogeneous import (
    HeterogeneousKinetics,
    NernstianEquilibrium,
)
from softpotato.reactions.mechanisms import ElectrochemicalStep, Mechanism
from softpotato.reactions.species import Species, SpeciesCollection
from softpotato.solver.base import (
    BaseSolver,
    BoundaryCondition,
    DirichletBC,
    NeumannBC,
    SolverResult,
    get_solver,
)

from .base import BaseTechnique
from .chronopotentiometry import Chronopotentiometry
from .pulse import SampledCurrentVoltammetry
from .result import TechniqueResult

MechanismInput = Mechanism | SpeciesCollection | Sequence[Species] | Species


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_mechanism(
    mechanism: MechanismInput,
    *,
    mode: Literal["potentiostatic", "galvanostatic"],
    kinetics: HeterogeneousKinetics | NernstianEquilibrium | None = None,
) -> Mechanism:
    """Normalize a Mechanism, SpeciesCollection, Species sequence, or single Species."""
    if isinstance(mechanism, Mechanism):
        return mechanism

    if isinstance(mechanism, Species):
        species_col = SpeciesCollection([mechanism])
    elif isinstance(mechanism, SpeciesCollection):
        species_col = mechanism
    elif isinstance(mechanism, Sequence) and not isinstance(mechanism, (str, bytes)):
        if len(mechanism) == 0:
            raise ValueError("species sequence must not be empty.")
        if not all(isinstance(sp, Species) for sp in mechanism):
            raise TypeError(
                "All items in a species sequence must be Species instances."
            )
        species_col = SpeciesCollection(mechanism)
    else:
        raise TypeError(
            "mechanism must be a Mechanism, SpeciesCollection, Species, or sequence of "
            f"Species, got {type(mechanism).__name__}."
        )

    steps: list[ElectrochemicalStep] = []
    if len(species_col) >= 2:
        sp_list = list(species_col.values())
        ox_sp, red_sp = sp_list[0], sp_list[1]
        if kinetics is not None:
            steps.append(ElectrochemicalStep(ox_sp, red_sp, kinetics))
        elif mode == "potentiostatic":
            e0 = (
                ox_sp.standard_potential
                if ox_sp.standard_potential is not None
                else (
                    red_sp.standard_potential
                    if red_sp.standard_potential is not None
                    else 0.0
                )
            )
            n_val = abs(ox_sp.n_electrons) if ox_sp.n_electrons else 1
            steps.append(
                ElectrochemicalStep(ox_sp, red_sp, NernstianEquilibrium(E0=e0, n=n_val))
            )
    elif kinetics is not None:
        raise ValueError(
            "kinetics requires at least two species (oxidized and reduced) when "
            "constructing a Mechanism from Species."
        )

    return Mechanism(species=species_col, electrochemical_steps=steps)


def _resolve_geometry(geometry: Geometry | str) -> Geometry:
    """Resolve a Geometry instance or registered geometry alias."""
    if isinstance(geometry, Geometry):
        return geometry
    if isinstance(geometry, str):
        return get_geometry(geometry)
    raise TypeError(
        f"geometry must be a Geometry instance or str alias, got {type(geometry).__name__}."
    )


def _resolve_solver(
    solver: BaseSolver | str,
    solver_kwargs: Mapping[str, Any] | None,
) -> tuple[BaseSolver, dict[str, Any]]:
    """Resolve solver instance and runtime kwargs to forward to ``solver.solve``."""
    opts = dict(solver_kwargs) if solver_kwargs is not None else {}
    if isinstance(solver, str):
        return get_solver(solver, **opts), {}
    if isinstance(solver, BaseSolver):
        return solver, opts
    raise TypeError(
        f"solver must be a BaseSolver instance or str alias, got {type(solver).__name__}."
    )


def _resolve_nernst_layer_thickness(
    technique: BaseTechnique,
    d_max: float,
) -> float | None:
    """Detect optional hydrodynamic Nernst diffusion layer thickness (e.g. RDE)."""
    for attr in ("diffusion_layer_thickness", "nernst_layer_thickness", "delta_N"):
        member = getattr(technique, attr, None)
        if member is None:
            continue
        if callable(member):
            val = float(member(d_max))
        elif isinstance(member, (int, float, np.number)) and not isinstance(
            member, bool
        ):
            val = float(member)
        else:
            continue
        if math.isfinite(val) and val > 0.0:
            return val
    return None


def _build_auto_grid(
    technique: BaseTechnique,
    mechanism: Mechanism,
    geometry: Geometry,
    mesh_kwargs: Mapping[str, Any] | None = None,
) -> Mesh1D:
    """Build a physics-informed 1D mesh sized to diffusion or Nernst layer thickness."""
    d_max = max(sp.diffusivity for sp in mechanism.species.values())
    x_min = (
        float(getattr(geometry, "r0", 0.0))
        if geometry.p > 0 and hasattr(geometry, "r0")
        else 0.0
    )

    opts: dict[str, Any] = dict(mesh_kwargs) if mesh_kwargs is not None else {}
    opts.setdefault("kind", "geometric")
    opts.setdefault("x_min", x_min)

    if "factor" not in opts:
        delta_n = _resolve_nernst_layer_thickness(technique, d_max)
        if delta_n is not None:
            opts["factor"] = delta_n / math.sqrt(d_max * technique.duration)

    return create_diffusion_mesh(d_max, technique.duration, **opts)


def _validate_t_eval(
    t_eval: np.ndarray | Sequence[float] | None,
    technique: BaseTechnique,
) -> np.ndarray:
    """Resolve and validate evaluation timestamps."""
    if t_eval is None:
        return technique.time_grid()
    arr = np.asarray(t_eval, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 timestamps.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("All timestamps in t_eval must be finite.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("t_eval must be strictly monotonically increasing.")
    tol = 1e-12 * max(1.0, technique.duration)
    if arr[0] < -tol or arr[-1] > technique.duration + tol:
        raise ValueError(
            f"t_eval [{arr[0]}, {arr[-1]}] must lie within [0, {technique.duration}]."
        )
    return arr


def _make_segment_callable(
    technique: BaseTechnique,
    seg_start: float,
    seg_end: float,
    *,
    is_last: bool,
) -> Callable[[float], float]:
    """Wrap technique excitation for smooth evaluation on ``[seg_start, seg_end]``.

    At an interior event boundary ``seg_end`` (``not is_last``), clamps to the
    left-hand limit ``np.nextafter(seg_end, seg_start)`` so right-continuous
    step/pulse discontinuities never leak into the preceding segment.
    """
    end_eval = seg_end if is_last else float(np.nextafter(seg_end, seg_start))

    def _eval(t: float) -> float:
        if t <= seg_start:
            return float(technique.excitation(seg_start))
        if t >= seg_end:
            return float(technique.excitation(end_eval))
        return float(technique.excitation(t))

    return _eval


def _electrode_area_for_side(
    geometry: Geometry,
    side: Literal["left", "right"],
    grid: np.ndarray | Mesh1D,
) -> float:
    """Compute active electrode surface area at ``side``."""
    if side == "left":
        return float(geometry.area)
    # Delegate to geometry.current(flux=1/(1*F), n_electrons=1, side="right", grid=grid)
    # so SphericalGeometry / CylindricalGeometry right-boundary areas are exact.
    return float(
        geometry.current(1.0 / FARADAY, n_electrons=1, side="right", grid=grid)
    )


def _infer_problem_n_electrons(
    mechanism: Mechanism,
    n_electrons: int | Mapping[str, int] | None,
) -> int | dict[str, int] | None:
    """Infer species n_electrons map from electrochemical steps when not set on Species."""
    if n_electrons is not None:
        return dict(n_electrons) if isinstance(n_electrons, Mapping) else n_electrons

    if not mechanism.electrochemical_steps:
        return None

    inferred: dict[str, int] = {}
    for name, sp in mechanism.species.items():
        if sp.n_electrons is not None:
            inferred[name] = sp.n_electrons

    for step in mechanism.electrochemical_steps:
        n_step = round(step.kinetics.n)
        inferred.setdefault(step.ox_name, n_step)
        inferred.setdefault(step.red_name, -n_step)

    for name in mechanism.species_names:
        inferred.setdefault(name, 0)

    return inferred


def _resolve_galvanostatic_pair(
    mechanism: Mechanism,
    electrodes: Literal["left", "right", "both"],
) -> tuple[
    str, str | None, float, Literal["left", "right"], ElectrochemicalStep | None
]:
    """Identify (ox_name, red_name, n_electrons, side, step) for galvanostatic BCs."""
    if mechanism.electrochemical_steps:
        step = mechanism.electrochemical_steps[0]
        side: Literal["left", "right"] = (
            electrodes if electrodes in ("left", "right") else step.electrode_side
        )
        return step.ox_name, step.red_name, float(step.kinetics.n), side, step

    side = "right" if electrodes == "right" else "left"
    sp_list = list(mechanism.species.values())
    if len(sp_list) == 1:
        sp = sp_list[0]
        n_val = float(abs(sp.n_electrons)) if sp.n_electrons else 1.0
        return sp.name, None, n_val, side, None

    ox_candidates = [sp for sp in sp_list if (sp.n_electrons or 0) > 0]
    red_candidates = [sp for sp in sp_list if (sp.n_electrons or 0) < 0]
    ox_sp = ox_candidates[0] if ox_candidates else sp_list[0]
    red_sp = (
        red_candidates[0]
        if red_candidates
        else next(sp for sp in sp_list if sp.name != ox_sp.name)
    )
    n_val = float(abs(ox_sp.n_electrons)) if ox_sp.n_electrons else 1.0
    return ox_sp.name, red_sp.name, n_val, side, None


def _build_galvanostatic_bcs(
    current_func: Callable[[float], float],
    mechanism: Mechanism,
    geometry: Geometry,
    grid: np.ndarray | Mesh1D,
    *,
    electrodes: Literal["left", "right", "both"],
    bulk_bc: Literal["dirichlet", "neumann"],
    bc_overrides: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]] | None,
) -> tuple[
    dict[str, tuple[BoundaryCondition, BoundaryCondition]],
    str,
    str | None,
    float,
    Literal["left", "right"],
    float,
    ElectrochemicalStep | None,
]:
    """Build matched Neumann interfacial flux BCs for galvanostatic techniques."""
    ox_name, red_name, n_val, side, step = _resolve_galvanostatic_pair(
        mechanism, electrodes
    )
    area = _electrode_area_for_side(geometry, side, grid)
    denom = n_val * FARADAY * area

    # Left boundary inward normal is +x: J_left = -D * dc/dx => dc_ox/dx = -J / D_ox.
    # Right boundary inward normal is -x: J_right = +D * dc/dx => dc_ox/dx = +J / D_ox.
    side_sign = -1.0 if side == "left" else 1.0

    d_ox = mechanism.species[ox_name].diffusivity
    bcs: dict[str, tuple[BoundaryCondition, BoundaryCondition]] = {}

    def _ox_grad(t: float, _d: float = d_ox) -> float:
        return side_sign * (current_func(t) / denom) / _d

    ox_elec = NeumannBC(_ox_grad)
    ox_far = (
        DirichletBC(mechanism.species[ox_name].bulk_concentration)
        if bulk_bc == "dirichlet"
        else NeumannBC(0.0)
    )
    bcs[ox_name] = (ox_elec, ox_far) if side == "left" else (ox_far, ox_elec)

    if red_name is not None:
        d_red = mechanism.species[red_name].diffusivity

        def _red_grad(t: float, _d: float = d_red) -> float:
            return -side_sign * (current_func(t) / denom) / _d

        red_elec = NeumannBC(_red_grad)
        red_far = (
            DirichletBC(mechanism.species[red_name].bulk_concentration)
            if bulk_bc == "dirichlet"
            else NeumannBC(0.0)
        )
        bcs[red_name] = (red_elec, red_far) if side == "left" else (red_far, red_elec)

    for sp_name, sp_obj in mechanism.species.items():
        if sp_name in bcs:
            continue
        elec_bc = NeumannBC(0.0)
        far_bc = (
            DirichletBC(sp_obj.bulk_concentration)
            if bulk_bc == "dirichlet"
            else NeumannBC(0.0)
        )
        bcs[sp_name] = (elec_bc, far_bc) if side == "left" else (far_bc, elec_bc)

    if bc_overrides:
        bcs.update(bc_overrides)

    return bcs, ox_name, red_name, n_val, side, area, step


def _stitch_solver_results(
    segments: list[tuple[SolverResult, np.ndarray]],
    mesh: Mesh1D | None,
) -> SolverResult:
    """Concatenate time-sliced SolverResult segments into a single SolverResult."""
    first_res = segments[0][0]
    species_names = first_res.species

    t_parts = [res.t[idx] for res, idx in segments]
    t_all = np.concatenate(t_parts)

    conc_all = {
        sp: np.concatenate(
            [res.concentrations[sp][idx] for res, idx in segments], axis=0
        )
        for sp in species_names
    }
    flux_all = {
        sp: np.concatenate([res.fluxes[sp][idx] for res, idx in segments], axis=0)
        for sp in species_names
        if sp in first_res.fluxes
    }
    rflux_all = {
        sp: np.concatenate([res.right_fluxes[sp][idx] for res, idx in segments], axis=0)
        for sp in species_names
        if sp in first_res.right_fluxes
    }
    curr_all = {
        sp: np.concatenate([res.currents[sp][idx] for res, idx in segments], axis=0)
        for sp in species_names
        if sp in first_res.currents
    }
    lcurr_all = {
        sp: np.concatenate(
            [res.left_currents[sp][idx] for res, idx in segments], axis=0
        )
        for sp in species_names
        if sp in first_res.left_currents
    }
    rcurr_all = {
        sp: np.concatenate(
            [res.right_currents[sp][idx] for res, idx in segments], axis=0
        )
        for sp in species_names
        if sp in first_res.right_currents
    }

    raw_outputs = [res.raw_output for res, _ in segments]
    return SolverResult(
        t=t_all,
        x=first_res.x.copy(),
        concentrations=conc_all,
        fluxes=flux_all,
        right_fluxes=rflux_all,
        currents=curr_all,
        left_currents=lcurr_all,
        right_currents=rcurr_all,
        success=all(res.success for res, _ in segments),
        message=segments[-1][0].message,
        raw_output=raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs,
        mesh=mesh if mesh is not None else first_res.mesh,
    )


def _extract_net_faradaic_current(
    solver_result: SolverResult,
    mechanism: Mechanism,
    geometry: Geometry,
    electrodes: Literal["left", "right", "both"],
) -> np.ndarray:
    """Extract net IUPAC Faradaic current I(t) without double-counting redox pairs."""
    n_times = solver_result.t.size
    if mechanism.electrochemical_steps:
        total = np.zeros(n_times, dtype=float)
        for step in mechanism.electrochemical_steps:
            side: Literal["left", "right"] = (
                electrodes if electrodes in ("left", "right") else step.electrode_side
            )
            flux_ox = (
                solver_result.fluxes[step.ox_name]
                if side == "left"
                else solver_result.right_fluxes[step.ox_name]
            )
            n_val = float(step.kinetics.n)
            if n_val.is_integer():
                step_i = np.asarray(
                    geometry.current(
                        flux_ox,
                        n_electrons=round(n_val),
                        side=side,
                        grid=solver_result.x,
                    ),
                    dtype=float,
                )
            else:
                area = _electrode_area_for_side(geometry, side, solver_result.x)
                step_i = n_val * FARADAY * area * np.asarray(flux_ox, dtype=float)
            total += step_i
        return total

    if not solver_result.currents:
        return np.zeros(n_times, dtype=float)

    if len(solver_result.currents) == 1:
        return np.asarray(
            next(iter(solver_result.currents.values())), dtype=float
        ).copy()

    # Prefer species with positive n_electrons to avoid double-counting (Ox=+n, Red=-n)
    pos_species = [
        name
        for name, sp in mechanism.species.items()
        if sp.n_electrons is not None
        and sp.n_electrons > 0
        and name in solver_result.currents
    ]
    selected = pos_species if pos_species else list(solver_result.currents.keys())
    total = np.zeros(n_times, dtype=float)
    for name in selected:
        total += np.asarray(solver_result.currents[name], dtype=float)
    return total


def _recover_galvanostatic_potential(
    technique: BaseTechnique,
    solver_result: SolverResult,
    mechanism: Mechanism,
    current_arr: np.ndarray,
    *,
    ox_name: str,
    red_name: str | None,
    n_val: float,
    side: Literal["left", "right"],
    area: float,
    step: ElectrochemicalStep | None,
) -> np.ndarray:
    """Recover electrode surface potential E(t) for galvanostatic simulations."""
    if red_name is None:
        return np.full(solver_result.t.size, np.nan, dtype=float)

    idx_surf = 0 if side == "left" else -1
    c_ox_surf = solver_result.concentrations[ox_name][:, idx_surf]
    c_red_surf = solver_result.concentrations[red_name][:, idx_surf]

    cp_helper = (
        technique
        if isinstance(technique, Chronopotentiometry)
        else Chronopotentiometry(current=1.0, duration=technique.duration)
    )

    if step is not None:
        kin = step.kinetics
        if isinstance(kin, HeterogeneousKinetics):
            return np.asarray(
                cp_helper.recover_potential(
                    c_ox_surf,
                    c_red_surf,
                    kinetics=kin,
                    current=current_arr,
                    area=area,
                ),
                dtype=float,
            )
        return np.asarray(
            cp_helper.recover_potential(
                c_ox_surf,
                c_red_surf,
                E0=kin.E0,
                n=kin.n,
                T=kin.T,
                R=kin.R,
                F=kin.F,
            ),
            dtype=float,
        )

    ox_sp = mechanism.species[ox_name]
    red_sp = mechanism.species[red_name]
    e0 = (
        ox_sp.standard_potential
        if ox_sp.standard_potential is not None
        else (
            red_sp.standard_potential if red_sp.standard_potential is not None else 0.0
        )
    )
    return np.asarray(
        cp_helper.recover_potential(c_ox_surf, c_red_surf, E0=e0, n=n_val),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Public API: simulate & simulate_sweep
# ---------------------------------------------------------------------------


def simulate(
    technique: BaseTechnique,
    mechanism: MechanismInput,
    geometry: Geometry | str = "planar",
    grid: np.ndarray | Mesh1D | None = None,
    solver: BaseSolver | str = "scipy_ivp",
    t_eval: np.ndarray | Sequence[float] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    *,
    mesh_kwargs: Mapping[str, Any] | None = None,
    bc_kwargs: Mapping[str, Any] | None = None,
    bc_overrides: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
    | None = None,
    electrodes: Literal["left", "right", "both"] = "left",
    bulk_bc: Literal["dirichlet", "neumann"] = "dirichlet",
    n_electrons: int | Mapping[str, int] | None = None,
    kinetics: HeterogeneousKinetics | NernstianEquilibrium | None = None,
) -> TechniqueResult:
    r"""Run a complete electrochemical technique simulation in a single call.

    Orchestrates physics-informed spatial meshing, boundary condition assembly
    (both potentiostatic and galvanostatic), event-aware piecewise numerical
    integration across waveform discontinuities/switching vertices, and
    Faradaic current / cumulative charge extraction.

    Parameters
    ----------
    technique : BaseTechnique
        Electrochemical excitation waveform (e.g., :class:`Chronoamperometry`,
        :class:`LinearSweepVoltammetry`, :class:`CyclicVoltammetry`,
        :class:`DifferentialPulseVoltammetry`, :class:`SquareWaveVoltammetry`,
        :class:`SampledCurrentVoltammetry`, :class:`Chronopotentiometry`, or
        :class:`CustomWaveform`).
    mechanism : Mechanism, SpeciesCollection, Sequence[Species], or Species
        Chemical/electrochemical reaction mechanism or species specification.
        When a :class:`SpeciesCollection` or sequence of :class:`Species` is
        passed directly, a :class:`Mechanism` is constructed automatically
        (using ``kinetics`` if given, or Nernstian equilibrium for a redox pair
        in potentiostatic mode).
    geometry : Geometry or str, default ``'planar'``
        Coordinate geometry instance or registered alias (``'planar'``,
        ``'spherical'``, ``'cylindrical'``).
    grid : np.ndarray or Mesh1D, optional
        1D spatial grid or mesh. When ``None``, automatically constructs a
        physics-informed mesh via :func:`~softpotato.grid.create_diffusion_mesh`
        respecting curvilinear ``r0`` and hydrodynamic Nernst layer thickness.
    solver : BaseSolver or str, default ``'scipy_ivp'``
        Numerical diffusion solver instance or registered alias (``'scipy_ivp'``,
        ``'crank_nicolson'``, ``'implicit'``, ``'explicit'``).
    t_eval : np.ndarray or Sequence[float], optional
        Strictly increasing 1D evaluation timestamps in ``[0, technique.duration]``.
        Defaults to ``technique.time_grid()``.
    solver_kwargs : Mapping[str, Any], optional
        Optional keyword arguments passed to the solver (e.g. ``method='Radau'``,
        ``rtol=1e-6``, ``atol=1e-8``, ``dt=...``).
    mesh_kwargs : Mapping[str, Any], optional
        Optional keyword arguments forwarded to :func:`~softpotato.grid.create_diffusion_mesh`
        when ``grid=None``.
    bc_kwargs : Mapping[str, Any], optional
        Optional keyword arguments forwarded to :meth:`Mechanism.create_problem`
        when constructing potentiostatic boundary conditions.
    bc_overrides : Mapping[str, tuple[BoundaryCondition, BoundaryCondition]], optional
        Optional per-species boundary condition overrides.
    electrodes : {'left', 'right', 'both'}, default ``'left'``
        Active electrode boundary for Faradaic current evaluation.
    bulk_bc : {'dirichlet', 'neumann'}, default ``'dirichlet'``
        Boundary condition type applied at the bulk (non-electrode) boundary.
    n_electrons : int or Mapping[str, int], optional
        Optional electron transfer count overrides.
    kinetics : HeterogeneousKinetics or NernstianEquilibrium, optional
        Optional interfacial kinetics used when ``mechanism`` is supplied as
        a :class:`SpeciesCollection` or sequence of :class:`Species`.

    Returns
    -------
    TechniqueResult
        Structured simulation result containing ``time``, ``potential``,
        ``current``, ``charge``, ``solver_result``, species profiles, and
        technique-specific views.
    """
    if not isinstance(technique, BaseTechnique):
        raise TypeError(
            f"technique must be a BaseTechnique instance, got {type(technique).__name__}."
        )

    resolved_mech = _resolve_mechanism(
        mechanism, mode=technique.mode, kinetics=kinetics
    )
    resolved_geom = _resolve_geometry(geometry)
    resolved_solver, run_kwargs = _resolve_solver(solver, solver_kwargs)
    resolved_t_eval = _validate_t_eval(t_eval, technique)
    resolved_grid: np.ndarray | Mesh1D = (
        grid
        if grid is not None
        else _build_auto_grid(technique, resolved_mech, resolved_geom, mesh_kwargs)
    )
    mesh_obj = resolved_grid if isinstance(resolved_grid, Mesh1D) else None

    # Special handling for SampledCurrentVoltammetry (family of CA steps)
    if isinstance(technique, SampledCurrentVoltammetry):
        step_results: list[TechniqueResult] = []
        for step_ca in technique.steps:
            step_res = simulate(
                technique=step_ca,
                mechanism=resolved_mech,
                geometry=resolved_geom,
                grid=resolved_grid,
                solver=resolved_solver,
                t_eval=resolved_t_eval,
                solver_kwargs=run_kwargs if run_kwargs else None,
                bc_kwargs=bc_kwargs,
                bc_overrides=bc_overrides,
                electrodes=electrodes,
                bulk_bc=bulk_bc,
                n_electrons=n_electrons,
            )
            step_results.append(step_res)

        pot_2d = np.asarray(technique.excitation(resolved_t_eval), dtype=float)
        curr_2d = np.vstack([r.current for r in step_results])
        last_sr = step_results[-1].solver_result
        agg_sr = SolverResult(
            t=last_sr.t,
            x=last_sr.x,
            concentrations=last_sr.concentrations,
            fluxes=last_sr.fluxes,
            right_fluxes=last_sr.right_fluxes,
            currents=last_sr.currents,
            left_currents=last_sr.left_currents,
            right_currents=last_sr.right_currents,
            success=all(r.success for r in step_results),
            message=(
                "All SCV steps completed successfully."
                if all(r.success for r in step_results)
                else next(r.message for r in step_results if not r.success)
            ),
            raw_output=[r.solver_result.raw_output for r in step_results],
            mesh=mesh_obj,
        )
        return TechniqueResult(
            technique=technique,
            solver_result=agg_sr,
            time=resolved_t_eval,
            potential=pot_2d,
            current=curr_2d,
            metadata={
                "solver": type(resolved_solver).__name__,
                "geometry": resolved_geom.name,
                "n_steps": technique.n_steps,
                "step_results": tuple(step_results),
            },
        )

    # Partition [0, t_end] at strictly interior event_times
    t_end_target = float(resolved_t_eval[-1])
    tol = 1e-12 * max(1.0, technique.duration)
    interior_events = [
        float(ev)
        for ev in technique.event_times
        if tol < float(ev) < t_end_target - tol
    ]
    boundaries = [0.0, *interior_events, t_end_target]
    n_segments = len(boundaries) - 1

    problem_ne = _infer_problem_n_electrons(resolved_mech, n_electrons)
    extra_bc_kwargs = dict(bc_kwargs) if bc_kwargs is not None else {}

    prev_c: dict[str, np.ndarray] | None = None
    stitched_segments: list[tuple[SolverResult, np.ndarray]] = []
    galv_info: (
        tuple[
            str,
            str | None,
            float,
            Literal["left", "right"],
            float,
            ElectrochemicalStep | None,
        ]
        | None
    ) = None

    for seg_idx in range(n_segments):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]
        is_last = seg_idx == n_segments - 1

        seg_callable = _make_segment_callable(
            technique, seg_start, seg_end, is_last=is_last
        )

        # Select requested evaluation points falling inside this segment
        if seg_idx == 0:
            mask = (resolved_t_eval >= seg_start - tol) & (
                resolved_t_eval <= seg_end + tol
            )
        else:
            mask = (resolved_t_eval > seg_start + tol) & (
                resolved_t_eval <= seg_end + tol
            )
        req_times = resolved_t_eval[mask]

        # Build solver t_eval for this segment (must span [seg_start, seg_end] with >= 2 pts)
        seg_points = [seg_start, *req_times.tolist(), seg_end]
        seg_t_eval_list: list[float] = []
        for pt in sorted(seg_points):
            clamped_pt = min(max(pt, seg_start), seg_end)
            if not seg_t_eval_list or (clamped_pt - seg_t_eval_list[-1]) > tol:
                seg_t_eval_list.append(clamped_pt)
        if seg_t_eval_list[-1] != seg_end:
            seg_t_eval_list[-1] = seg_end
        if len(seg_t_eval_list) < 2:
            seg_t_eval_list = [seg_start, seg_end]
        seg_t_eval = np.asarray(seg_t_eval_list, dtype=float)

        # Build DiffusionProblem for this segment
        if technique.mode == "galvanostatic":
            (
                galv_bcs,
                ox_name,
                red_name,
                n_val,
                side,
                area,
                step_obj,
            ) = _build_galvanostatic_bcs(
                seg_callable,
                resolved_mech,
                resolved_geom,
                resolved_grid,
                electrodes=electrodes,
                bulk_bc=bulk_bc,
                bc_overrides=bc_overrides,
            )
            galv_info = (ox_name, red_name, n_val, side, area, step_obj)
            problem = resolved_mech.create_problem(
                resolved_grid,
                boundary_conditions=galv_bcs,
                geometry=resolved_geom,
                initial_conditions=prev_c,
                n_electrons=problem_ne,
                electrodes=electrodes,
            )
        else:
            problem = resolved_mech.create_problem(
                resolved_grid,
                potential=seg_callable,
                geometry=resolved_geom,
                initial_conditions=prev_c,
                n_electrons=problem_ne,
                electrodes=electrodes,
                bulk_bc=bulk_bc,
                bc_overrides=bc_overrides,
                **extra_bc_kwargs,
            )

        seg_res = resolved_solver.solve(
            problem,
            t_span=(seg_start, seg_end),
            t_eval=seg_t_eval,
            **run_kwargs,
        )

        if not seg_res.success:
            empty_arr = np.full(resolved_t_eval.size, np.nan, dtype=float)
            return TechniqueResult(
                technique=technique,
                solver_result=seg_res,
                time=resolved_t_eval,
                potential=empty_arr.copy(),
                current=empty_arr.copy(),
                charge=empty_arr.copy(),
                metadata={
                    "solver": type(resolved_solver).__name__,
                    "geometry": resolved_geom.name,
                    "n_segments": n_segments,
                    "failed_segment": seg_idx,
                },
            )

        # Carry terminal concentration profile into the next segment
        prev_c = {
            sp: seg_res.concentrations[sp][-1].copy()
            for sp in resolved_mech.species_names
        }

        if req_times.size > 0:
            # Map each requested timestamp to its index in seg_res.t
            idx = np.searchsorted(seg_res.t, req_times)
            idx = np.clip(idx, 0, seg_res.t.size - 1)
            # Pick nearest neighbor in case of floating-point rounding
            left_idx = np.clip(idx - 1, 0, seg_res.t.size - 1)
            use_left = np.abs(seg_res.t[left_idx] - req_times) < np.abs(
                seg_res.t[idx] - req_times
            )
            chosen_idx = np.where(use_left, left_idx, idx)
            stitched_segments.append((seg_res, chosen_idx))

    combined_sr = _stitch_solver_results(stitched_segments, mesh_obj)
    # Preserve exact requested time stamps
    combined_sr.t = resolved_t_eval.copy()

    metadata: dict[str, Any] = {
        "solver": type(resolved_solver).__name__,
        "geometry": resolved_geom.name,
        "n_segments": n_segments,
    }

    if technique.mode == "galvanostatic":
        assert galv_info is not None
        ox_name, red_name, n_val, side, area, step_obj = galv_info
        current_arr = np.asarray(technique.current(resolved_t_eval), dtype=float)
        potential_arr = _recover_galvanostatic_potential(
            technique,
            combined_sr,
            resolved_mech,
            current_arr,
            ox_name=ox_name,
            red_name=red_name,
            n_val=n_val,
            side=side,
            area=area,
            step=step_obj,
        )
        if isinstance(technique, Chronopotentiometry):
            idx_surf = 0 if side == "left" else -1
            first_nonzero = next(
                (i_val for i_val in technique.currents if i_val != 0.0), 0.0
            )
            reactant_name = (
                red_name if (first_nonzero > 0.0 and red_name is not None) else ox_name
            )
            c_reactant = combined_sr.concentrations[reactant_name][:, idx_surf]
            tau = (
                Chronopotentiometry.detect_transition_time(resolved_t_eval, c_reactant)
                if c_reactant[0] > 0.0
                else None
            )
            metadata["transition_time"] = tau
            metadata["reactant_species"] = reactant_name
    else:
        potential_arr = np.asarray(technique.potential(resolved_t_eval), dtype=float)
        current_arr = _extract_net_faradaic_current(
            combined_sr, resolved_mech, resolved_geom, electrodes
        )

    return TechniqueResult(
        technique=technique,
        solver_result=combined_sr,
        time=resolved_t_eval,
        potential=potential_arr,
        current=current_arr,
        metadata=metadata,
    )


@overload
def simulate_sweep(
    techniques: Sequence[BaseTechnique],
    mechanism: MechanismInput,
    *,
    values: None = None,
    geometry: Geometry | str = "planar",
    grid: np.ndarray | Mesh1D | None = None,
    solver: BaseSolver | str = "scipy_ivp",
    t_eval: np.ndarray | Sequence[float] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> list[TechniqueResult]: ...


@overload
def simulate_sweep(
    techniques: Mapping[Any, BaseTechnique],
    mechanism: MechanismInput,
    *,
    values: None = None,
    geometry: Geometry | str = "planar",
    grid: np.ndarray | Mesh1D | None = None,
    solver: BaseSolver | str = "scipy_ivp",
    t_eval: np.ndarray | Sequence[float] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[Any, TechniqueResult]: ...


@overload
def simulate_sweep(
    techniques: Callable[[Any], BaseTechnique],
    mechanism: MechanismInput,
    *,
    values: Sequence[Any],
    geometry: Geometry | str = "planar",
    grid: np.ndarray | Mesh1D | None = None,
    solver: BaseSolver | str = "scipy_ivp",
    t_eval: np.ndarray | Sequence[float] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[Any, TechniqueResult]: ...


def simulate_sweep(
    techniques: Sequence[BaseTechnique]
    | Mapping[Any, BaseTechnique]
    | Callable[[Any], BaseTechnique],
    mechanism: MechanismInput,
    *,
    values: Sequence[Any] | None = None,
    geometry: Geometry | str = "planar",
    grid: np.ndarray | Mesh1D | None = None,
    solver: BaseSolver | str = "scipy_ivp",
    t_eval: np.ndarray | Sequence[float] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> list[TechniqueResult] | dict[Any, TechniqueResult]:
    """Run a parameter sweep (e.g., scan-rate or rotation-rate series) via :func:`simulate`.

    Supports three input modes:

    1. **Sequence of techniques**: ``simulate_sweep([cv1, cv2, ...], mechanism)``
       returns ``list[TechniqueResult]``.
    2. **Mapping of parameter labels to techniques**:
       ``simulate_sweep({0.05: cv1, 0.1: cv2}, mechanism)`` returns
       ``dict[Any, TechniqueResult]``.
    3. **Factory callable + parameter values**:
       ``simulate_sweep(lambda v: CyclicVoltammetry(..., scan_rate=v), mechanism, values=[0.05, 0.1, 0.5])``
       returns ``dict[Any, TechniqueResult]`` keyed by each value in ``values``.

    Parameters
    ----------
    techniques : Sequence[BaseTechnique], Mapping[Any, BaseTechnique], or Callable
        Collection of techniques to simulate or a factory callable mapping each
        parameter value in ``values`` to a :class:`BaseTechnique`.
    mechanism : Mechanism, SpeciesCollection, Sequence[Species], or Species
        Chemical/electrochemical mechanism passed to :func:`simulate`.
    values : Sequence[Any], optional
        Parameter values to sweep when ``techniques`` is a factory callable.
    geometry : Geometry or str, default ``'planar'``
        Coordinate geometry forwarded to :func:`simulate`.
    grid : np.ndarray or Mesh1D, optional
        Optional fixed spatial grid forwarded to :func:`simulate`. Leave ``None``
        so each technique auto-sizes its diffusion mesh to its own duration.
    solver : BaseSolver or str, default ``'scipy_ivp'``
        Numerical solver forwarded to :func:`simulate`.
    t_eval : np.ndarray or Sequence[float], optional
        Optional fixed evaluation timestamps forwarded to :func:`simulate`.
    solver_kwargs : Mapping[str, Any], optional
        Optional solver options forwarded to :func:`simulate`.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`simulate`.

    Returns
    -------
    list[TechniqueResult] or dict[Any, TechniqueResult]
        Ordered list or mapping of simulation results ready for diagnostics.
    """
    if isinstance(techniques, BaseTechnique):
        raise TypeError(
            "techniques must be a sequence/mapping of BaseTechnique instances or "
            "a factory callable paired with 'values', not a single BaseTechnique."
        )

    if isinstance(techniques, Mapping):
        if values is not None:
            raise ValueError(
                "'values' cannot be specified when 'techniques' is a mapping."
            )
        if len(techniques) == 0:
            raise ValueError("'techniques' mapping must not be empty.")
        return {
            key: simulate(
                tech,
                mechanism,
                geometry=geometry,
                grid=grid,
                solver=solver,
                t_eval=t_eval,
                solver_kwargs=solver_kwargs,
                **kwargs,
            )
            for key, tech in techniques.items()
        }

    if isinstance(techniques, Sequence) and not isinstance(techniques, (str, bytes)):
        if values is not None:
            raise ValueError(
                "'values' cannot be specified when 'techniques' is a sequence."
            )
        if len(techniques) == 0:
            raise ValueError("'techniques' sequence must not be empty.")
        return [
            simulate(
                tech,
                mechanism,
                geometry=geometry,
                grid=grid,
                solver=solver,
                t_eval=t_eval,
                solver_kwargs=solver_kwargs,
                **kwargs,
            )
            for tech in techniques
        ]

    if callable(techniques):
        if values is None:
            raise ValueError(
                "'values' must be provided when 'techniques' is a factory callable."
            )
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(
                "'values' must be a non-empty sequence of parameter values."
            )
        if len(values) == 0:
            raise ValueError("'values' must not be empty.")
        results: dict[Any, TechniqueResult] = {}
        for val in values:
            tech = techniques(val)
            results[val] = simulate(
                tech,
                mechanism,
                geometry=geometry,
                grid=grid,
                solver=solver,
                t_eval=t_eval,
                solver_kwargs=solver_kwargs,
                **kwargs,
            )
        return results

    raise TypeError(
        "techniques must be a sequence/mapping of BaseTechnique instances or a "
        f"factory callable, got {type(techniques).__name__}."
    )
