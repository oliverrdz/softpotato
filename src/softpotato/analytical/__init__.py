"""Analytical and empirical electrochemical equations.

This subpackage provides closed-form solutions, asymptotic expansions, and empirical
models for electrochemical systems. These analytical equations serve as:
1. Exact ground-truth benchmarks for numerical PDE solvers (:mod:`softpotato.solver`).
2. Rapid parameter estimation tools for experimental and simulated data.
3. Diagnostic baselines for transient, steady-state, and voltammetric responses.

Submodules
----------
step : Potential and current step equations (Cottrell, Anson, Sand, etc.).
"""

from . import step
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
    "anson",
    "cottrell",
    "cottrell_cylinder",
    "cottrell_spherical",
    "cottrell_step",
    "sand",
    "sand_potential",
    "sand_transition_time",
    "step",
    "step_concentration_profile",
    "step_flux_profile",
]
