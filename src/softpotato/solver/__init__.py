"""Numerical solvers for chemical species diffusion and reaction-diffusion problems.

This module provides a unified framework for solving 1D multi-species diffusion equations:

.. math::

    \\frac{\\partial c_i}{\\partial t} = D_i \\frac{\\partial^2 c_i}{\\partial x^2} + R_i(\\mathbf{c}, x, t)

Available built-in solvers include:
- ``"scipy_ivp"`` / ``"solve_ivp"``: Method of Lines using ``scipy.integrate.solve_ivp`` (Radau/BDF/RK45).
- ``"crank_nicolson"``: 2nd-order unconditionally stable Crank-Nicolson finite difference.
- ``"implicit"`` / ``"btcs"``: Unconditionally stable Backward-Time Central-Space.
- ``"explicit"`` / ``"ftcs"``: Forward-Time Central-Space with CFL stability enforcement.

Examples
--------
Solving a 1D diffusion problem and comparing two solvers:

>>> import numpy as np
>>> from softpotato.solver import (
...     DiffusionProblem,
...     DirichletBC,
...     NeumannBC,
...     get_solver,
...     list_solvers,
... )
>>>
>>> list_solvers()
['btcs', 'crank-nicolson', 'crank_nicolson', 'explicit', 'ftcs', 'implicit', 'scipy_ivp', 'solve_ivp']
>>>
>>> grid = np.linspace(0.0, 1.0, 41)
>>> problem = DiffusionProblem(
...     grid=grid,
...     diffusivity={"c": 0.01},
...     boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
...     initial_conditions={"c": lambda x: np.sin(np.pi * x)},
... )
>>>
>>> t_eval = np.linspace(0.0, 0.5, 6)
>>> solver_mol = get_solver("scipy_ivp")
>>> res_mol = solver_mol.solve(problem, t_span=(0.0, 0.5), t_eval=t_eval)
>>> res_mol.success
True
>>>
>>> solver_cn = get_solver("crank_nicolson", dt=0.01)
>>> res_cn = solver_cn.solve(problem, t_span=(0.0, 0.5), t_eval=t_eval)
>>> res_cn.success
True
"""

from .base import (
    BaseSolver,
    BoundaryCondition,
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    SolverResult,
    get_solver,
    list_solvers,
    register_solver,
)
from .crank_nicolson import CrankNicolson
from .explicit import ExplicitFiniteDifference
from .implicit import ImplicitFiniteDifference
from .scipy_ivp import ScipyIVPSolver

__all__ = [
    "BaseSolver",
    "BoundaryCondition",
    "CrankNicolson",
    "DiffusionProblem",
    "DirichletBC",
    "ExplicitFiniteDifference",
    "ImplicitFiniteDifference",
    "NeumannBC",
    "ScipyIVPSolver",
    "SolverResult",
    "get_solver",
    "list_solvers",
    "register_solver",
]
