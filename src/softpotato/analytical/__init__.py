"""Analytical and empirical electrochemical equations.

This subpackage provides closed-form solutions, asymptotic expansions, and empirical
models for electrochemical systems. These analytical equations serve as:
1. Exact ground-truth benchmarks for numerical PDE solvers (:mod:`softpotato.solver`).
2. Rapid parameter estimation tools for experimental and simulated data.
3. Diagnostic baselines for transient, steady-state, and voltammetric responses.

Submodules
----------
hydrodynamics : Convection and hydrodynamic equations for RDE and RRDE.
kinetics : Thermodynamics and interfacial kinetics (Nernst, Butler–Volmer, Tafel).
step : Potential and current step equations (Cottrell, Anson, Sand, etc.).
voltammetry : Voltammetry and kinetic diagnostics (Randles–Ševčík, Nicholson, Matsuda–Ayabe).
"""

from . import hydrodynamics, kinetics, step, voltammetry
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
from .kinetics import (
    TafelResult,
    butler_volmer,
    butler_volmer_current_density,
    butler_volmer_linear,
    charge_transfer_resistance,
    exchange_current,
    exchange_current_density,
    nernst,
    nernst_equilibrium_concentrations,
    nernst_potential,
    nernst_ratio,
    tafel,
    tafel_analysis,
    tafel_overpotential,
    tafel_slope,
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
from .voltammetry import (
    MatsudaAyabeResult,
    NicholsonResult,
    matsuda_ayabe,
    matsuda_ayabe_lambda,
    nicholson_delta_ep,
    nicholson_psi,
    nicholson_rate_constant,
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
    randles_sevcik_quasi,
)

__all__ = [
    "KouteckyLevichResult",
    "MatsudaAyabeResult",
    "NicholsonResult",
    "RRDEResult",
    "TafelResult",
    "anson",
    "butler_volmer",
    "butler_volmer_current_density",
    "butler_volmer_linear",
    "charge_transfer_resistance",
    "collection_efficiency",
    "cottrell",
    "cottrell_cylinder",
    "cottrell_spherical",
    "cottrell_step",
    "exchange_current",
    "exchange_current_density",
    "hydrodynamics",
    "kinetics",
    "koutecky_levich",
    "koutecky_levich_analysis",
    "levich",
    "levich_constant",
    "matsuda_ayabe",
    "matsuda_ayabe_lambda",
    "nernst",
    "nernst_diffusion_layer",
    "nernst_equilibrium_concentrations",
    "nernst_potential",
    "nernst_ratio",
    "nicholson_delta_ep",
    "nicholson_psi",
    "nicholson_rate_constant",
    "peak_potential_irreversible",
    "rad_s_to_rpm",
    "randles_sevcik",
    "randles_sevcik_irreversible",
    "randles_sevcik_quasi",
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
    "tafel",
    "tafel_analysis",
    "tafel_overpotential",
    "tafel_slope",
    "voltammetry",
]
