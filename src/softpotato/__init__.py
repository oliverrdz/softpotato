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

from . import solver
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

__version__ = "3.0.0a1"

__all__ = [
    "__version__",
    "solver",
    "BaseSolver",
    "BoundaryCondition",
    "DiffusionProblem",
    "DirichletBC",
    "NeumannBC",
    "SolverResult",
    "get_solver",
    "list_solvers",
]
