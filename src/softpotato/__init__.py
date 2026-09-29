"""Soft Potato: Electrochemical simulation and analysis toolkit in Python.

Examples
--------
Simulating 1D chemical diffusion using the SciPy Method of Lines solver:

>>> import numpy as np
>>> import softpotato as sp
>>> from softpotato.solver import DiffusionProblem, DirichletBC, get_solver
>>>
>>> # Define 1D grid and problem
>>> grid = np.linspace(0.0, 1.0, 50)
>>> problem = DiffusionProblem(
...     grid=grid,
...     diffusivity={"A": 1e-4},
...     boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
...     initial_conditions={"A": 1.0},
... )
>>>
>>> # Select solver and integrate
>>> solver = get_solver("scipy_ivp")
>>> result = solver.solve(problem, t_span=(0.0, 1.0))
>>> result.success
True
>>> result["A"].shape
(101, 50)
"""

from . import analytical, constants, solver
from .analytical import (
    KouteckyLevichResult,
    RRDEResult,
    anson,
    collection_efficiency,
    cottrell,
    cottrell_cylinder,
    cottrell_spherical,
    cottrell_step,
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
    sand,
    sand_potential,
    sand_transition_time,
    shielding_factor,
    step_concentration_profile,
    step_flux_profile,
)
from .constants import (
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    FARADAY,
    GAS_CONSTANT,
    STANDARD_TEMPERATURE,
    T_STD,
    VACUUM_PERMITTIVITY,
    F,
    R,
)
from .solver import (
    BaseSolver,
    BoundaryCondition,
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    SolverResult,
    get_solver,
    list_solvers,
)

__version__ = "3.0.0"

__all__ = [
    "AVOGADRO",
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "FARADAY",
    "GAS_CONSTANT",
    "STANDARD_TEMPERATURE",
    "T_STD",
    "VACUUM_PERMITTIVITY",
    "BaseSolver",
    "BoundaryCondition",
    "DiffusionProblem",
    "DirichletBC",
    "F",
    "KouteckyLevichResult",
    "NeumannBC",
    "R",
    "RRDEResult",
    "SolverResult",
    "__version__",
    "analytical",
    "anson",
    "collection_efficiency",
    "constants",
    "cottrell",
    "cottrell_cylinder",
    "cottrell_spherical",
    "cottrell_step",
    "get_solver",
    "koutecky_levich",
    "koutecky_levich_analysis",
    "levich",
    "levich_constant",
    "list_solvers",
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
    "solver",
    "step_concentration_profile",
    "step_flux_profile",
]
