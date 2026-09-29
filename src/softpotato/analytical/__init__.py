"""Analytical and empirical electrochemical equations.

This subpackage provides closed-form solutions, asymptotic expansions, and empirical
models for electrochemical systems. These analytical equations serve as:
1. Exact ground-truth benchmarks for numerical PDE solvers (:mod:`softpotato.solver`).
2. Rapid parameter estimation tools for experimental and simulated data.
3. Diagnostic baselines for transient, steady-state, and voltammetric responses.

Submodules
----------
hydrodynamics : Convection and hydrodynamic equations for RDE and RRDE.
step : Potential and current step equations (Cottrell, Anson, Sand, etc.).
"""

from . import hydrodynamics, step
from .hydrodynamics import (
    KouteckyLevichResult,
    RRDEResult,
    collection_efficiency,
    koutecky_levich,
    koutecky_levich_analysis,
    levich,
    levich_constant,
    nernst_diffusion_layer,
    rad_s_to_rpm,
    ring_collection_current,
    ring_limiting_current,
    rotating_ring_disk,
    rpm_to_rad_s,
    shielding_factor,
)
from .step import (
    anson,
    cottrell,
    cottrell_cylinder,
    cottrell_spherical,
    cottrell_step,
    sand,
    sand_potential,
    sand_transition_time,
    step_concentration_profile,
    step_flux_profile,
)

__all__ = [
    "KouteckyLevichResult",
    "RRDEResult",
    "anson",
    "collection_efficiency",
    "cottrell",
    "cottrell_cylinder",
    "cottrell_spherical",
    "cottrell_step",
    "hydrodynamics",
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
    "sand",
    "sand_potential",
    "sand_transition_time",
    "shielding_factor",
    "step",
    "step_concentration_profile",
    "step_flux_profile",
]
