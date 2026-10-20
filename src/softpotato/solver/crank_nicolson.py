"""Crank-Nicolson finite difference diffusion solver."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from softpotato.geometry import Geometry

from .base import BaseSolver, DiffusionProblem, SolverResult


class CrankNicolson(BaseSolver, name="crank_nicolson"):
    r"""Crank-Nicolson (:math:`\theta = 0.5`) finite difference diffusion solver.

    Second-order accurate in both time and space: :math:`\mathcal{O}(\Delta t^2 + \Delta x^2)`.
    Unconditionally stable. Solves the coupled linear system at each time step:

    .. math::

        \frac{c_i^{n+1} - c_i^n}{\Delta t} = \frac{D}{2} \left[ \frac{c_{i+1}^{n+1} - 2c_i^{n+1} + c_{i-1}^{n+1}}{\Delta x^2} + \frac{c_{i+1}^n - 2c_i^n + c_{i-1}^n}{\Delta x^2} \right] + R_i^{n+1/2}

    Employs ``scipy.linalg.solve_banded`` to invert the tridiagonal system in :math:`\mathcal{O}(N)` time at each step.

    Parameters
    ----------
    dt : float, optional
        Fixed time step size :math:`\Delta t`.
        If None (default), automatically selects 100 uniform steps over the integration
        interval: :math:`\Delta t = (t_{\text{end}} - t_{\text{start}}) / 100`.
    **options : Any
        Additional solver options stored in ``self.options``.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.solver import CrankNicolson, DiffusionProblem, DirichletBC
    >>>
    >>> grid = np.linspace(0.0, 1.0, 41)
    >>> problem = DiffusionProblem(
    ...     grid=grid,
    ...     diffusivity={"c": 0.02},
    ...     boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(0.0))},
    ...     initial_conditions={"c": lambda x: np.sin(np.pi * x)},
    ... )
    >>> # 2nd-order accurate time marching
    >>> solver = CrankNicolson(dt=0.01)
    >>> result = solver.solve(problem, t_span=(0.0, 0.2))
    >>> result.success
    True
    >>> result["c"].shape[1]
    41
    """

    def __init__(self, dt: float | None = None, **options: Any) -> None:
        """Initialize Crank-Nicolson solver."""
        super().__init__(dt=dt, **options)

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        geom = problem.geometry
        assert isinstance(geom, Geometry)
        dx = problem.dx if problem.is_uniform else None
        t_start, t_end = t_span

        user_dt = (
            kwargs.get("dt")
            if kwargs.get("dt") is not None
            else self.options.get("dt", None)
        )
        if user_dt is not None:
            base_dt = float(user_dt)
        else:
            base_dt = (t_end - t_start) / 100.0

        if t_eval is None:
            n_steps = max(2, int(np.ceil((t_end - t_start) / base_dt)) + 1)
            t_eval = np.linspace(t_start, t_end, n_steps)
        else:
            t_eval = np.asarray(t_eval, dtype=float)

        n_times = len(t_eval)
        N = problem.n_points
        curr_c = {sp: problem.get_initial_profile(sp) for sp in problem.species}

        # If t_eval starts after t_start, advance from t_start to t_eval[0]
        t_current = t_start
        if t_eval[0] > t_start:
            interval = t_eval[0] - t_start
            substeps = max(1, int(np.ceil(interval / base_dt)))
            dt = interval / substeps
            for _ in range(substeps):
                curr_c = self._step_crank_nicolson(problem, curr_c, t_current, dt, dx)
                t_current += dt

        records = {sp: np.zeros((n_times, N), dtype=float) for sp in problem.species}
        for sp in problem.species:
            records[sp][0, :] = curr_c[sp]

        # Time marching with substepping
        for k in range(n_times - 1):
            t_k = t_eval[k]
            t_next = t_eval[k + 1]
            interval = t_next - t_k
            substeps = max(1, int(np.ceil(interval / base_dt)))
            dt = interval / substeps

            t_current = t_k
            for _ in range(substeps):
                curr_c = self._step_crank_nicolson(problem, curr_c, t_current, dt, dx)
                t_current += dt

            for sp in problem.species:
                records[sp][k + 1, :] = curr_c[sp]

        return SolverResult(
            t=t_eval,
            x=problem.grid,
            concentrations=records,
            success=True,
            message="Crank-Nicolson solution completed successfully.",
        )

    def _step_crank_nicolson(
        self,
        problem: DiffusionProblem,
        c_dict: dict[str, np.ndarray],
        t: float,
        dt: float,
        dx: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Advance all species profiles by one Crank-Nicolson time step."""
        reactions = (
            problem.reactions(c_dict, problem.grid, t) if problem.reactions else None
        )
        geom = problem.geometry
        assert isinstance(geom, Geometry)
        new_c: dict[str, np.ndarray] = {}
        N = problem.n_points
        t_next = t + dt
        theta = 0.5

        for sp in problem.species:
            c = c_dict[sp]
            D = problem.diffusivity[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]

            # Assemble banded Laplacian L and boundary scaling from geometry
            l_banded, bc_scale = geom.laplacian_matrix(
                problem.grid, bc_left=bc_left, bc_right=bc_right
            )

            ab = np.zeros((3, N), dtype=float)
            rhs = np.zeros(N, dtype=float)

            # Interior rows (1 <= i <= N - 2)
            ab[1, 1:-1] = 1.0 - theta * dt * D * l_banded[1, 1:-1]
            ab[0, 2:] = -theta * dt * D * l_banded[0, 2:]
            ab[2, :-2] = -theta * dt * D * l_banded[2, :-2]

            l_dot_c = (
                l_banded[2, :-2] * c[:-2]
                + l_banded[1, 1:-1] * c[1:-1]
                + l_banded[0, 2:] * c[2:]
            )
            rhs[1:-1] = c[1:-1] + (1.0 - theta) * dt * D * l_dot_c
            if reactions and sp in reactions:
                rhs[1:-1] += dt * reactions[sp][1:-1]

            # Left boundary (i = 0)
            if bc_left.btype == "dirichlet":
                ab[1, 0] = 1.0
                ab[0, 1] = 0.0
                rhs[0] = bc_left.evaluate(t_next)
            else:  # Neumann
                ab[1, 0] = 1.0 - theta * dt * D * l_banded[1, 0]
                ab[0, 1] = -theta * dt * D * l_banded[0, 1]
                g_left_curr = bc_left.evaluate(t)
                g_left_next = bc_left.evaluate(t_next)
                g_left_eff = theta * g_left_next + (1.0 - theta) * g_left_curr
                l_dot_c_0 = l_banded[1, 0] * c[0] + l_banded[0, 1] * c[1]
                rhs[0] = (
                    c[0]
                    + (1.0 - theta) * dt * D * l_dot_c_0
                    + dt * D * bc_scale[0] * g_left_eff
                )
                if reactions and sp in reactions:
                    rhs[0] += dt * reactions[sp][0]

            # Right boundary (i = N - 1)
            if bc_right.btype == "dirichlet":
                ab[1, -1] = 1.0
                ab[2, -2] = 0.0
                rhs[-1] = bc_right.evaluate(t_next)
            else:  # Neumann
                ab[1, -1] = 1.0 - theta * dt * D * l_banded[1, -1]
                ab[2, -2] = -theta * dt * D * l_banded[2, -2]
                g_right_curr = bc_right.evaluate(t)
                g_right_next = bc_right.evaluate(t_next)
                g_right_eff = theta * g_right_next + (1.0 - theta) * g_right_curr
                l_dot_c_end = l_banded[1, -1] * c[-1] + l_banded[2, -2] * c[-2]
                rhs[-1] = (
                    c[-1]
                    + (1.0 - theta) * dt * D * l_dot_c_end
                    + dt * D * bc_scale[-1] * g_right_eff
                )
                if reactions and sp in reactions:
                    rhs[-1] += dt * reactions[sp][-1]

            new_c[sp] = solve_banded((1, 1), ab, rhs)

        return new_c


# Also register the alias 'crank-nicolson'
BaseSolver._registry["crank-nicolson"] = CrankNicolson
