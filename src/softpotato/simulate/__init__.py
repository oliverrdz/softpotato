"""
Numerical PDE solvers for electrochemical mass transport.
"""

from .efd import explicit_diffuse_step
from .ifd import crank_nicolson_step
from .solver import SimulationResult, Solver, solve_1d, solve_homogeneous_1d

__all__ = [
    "SimulationResult",
    "Solver",
    "crank_nicolson_step",
    "explicit_diffuse_step",
    "solve_1d",
    "solve_homogeneous_1d",
]
