"""Crank-Nicolson finite difference diffusion solver."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.linalg import solve_banded

from .base import BaseSolver, DiffusionProblem, SolverResult


class CrankNicolson(BaseSolver, name="crank_nicolson"):
    r"""Crank-Nicolson (:math:`\theta = 0.5`) finite difference diffusion solver.

    Second-order accurate in both time and space: :math:`\mathcal{O}(\Delta t^2 + \Delta x^2)`.
    Unconditionally stable. Solves the coupled linear system at each time step:

    .. math::

        \frac{c_i^{n+1} - c_i^n}{\Delta t} = \frac{D}{2} \left[ \frac{c_{i+1}^{n+1} - 2c_i^{n+1} + c_{i-1}^{n+1}}{\Delta x^2} + \frac{c_{i+1}^n - 2c_i^n + c_{i-1}^n}{\Delta x^2} \right] + R_i^{n+1/2}

    Employs ``scipy.linalg.solve_banded`` to invert the tridiagonal system in :math:`\mathcal{O}(N)` time at each step.

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

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        if not problem.is_uniform:
            raise NotImplementedError("CrankNicolson requires a uniform spatial grid.")

        dx = problem.dx
        t_start, t_end = t_span

        user_dt = self.options.get("dt", kwargs.get("dt", None))
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
        dx: float,
    ) -> dict[str, np.ndarray]:
        """Advance all species profiles by one Crank-Nicolson time step."""
        reactions = problem.reactions(c_dict, problem.grid, t) if problem.reactions else None
        new_c: dict[str, np.ndarray] = {}
        N = problem.n_points
        t_next = t + dt

        for sp in problem.species:
            c = c_dict[sp]
            D = problem.diffusivity[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]
            r = D * dt / (2.0 * dx**2)

            ab = np.zeros((3, N), dtype=float)
            rhs = np.zeros(N, dtype=float)

            # Interior rows
            ab[1, 1:-1] = 1.0 + 2.0 * r
            ab[0, 2:] = -r       # A[i, i+1]
            ab[2, :-2] = -r      # A[i, i-1]
            rhs[1:-1] = r * c[:-2] + (1.0 - 2.0 * r) * c[1:-1] + r * c[2:]
            if reactions and sp in reactions:
                rhs[1:-1] += dt * reactions[sp][1:-1]

            # Left boundary
            if bc_left.btype == "dirichlet":
                ab[1, 0] = 1.0
                ab[0, 1] = 0.0
                rhs[0] = bc_left.evaluate(t_next)
            else:  # Neumann
                ab[1, 0] = 1.0 + 2.0 * r
                ab[0, 1] = -2.0 * r
                g_left_curr = bc_left.evaluate(t)
                g_left_next = bc_left.evaluate(t_next)
                rhs[0] = (
                    (1.0 - 2.0 * r) * c[0]
                    + 2.0 * r * c[1]
                    - 2.0 * r * dx * (g_left_curr + g_left_next)
                )
                if reactions and sp in reactions:
                    rhs[0] += dt * reactions[sp][0]

            # Right boundary
            if bc_right.btype == "dirichlet":
                ab[1, -1] = 1.0
                ab[2, -2] = 0.0
                rhs[-1] = bc_right.evaluate(t_next)
            else:  # Neumann
                ab[1, -1] = 1.0 + 2.0 * r
                ab[2, -2] = -2.0 * r
                g_right_curr = bc_right.evaluate(t)
                g_right_next = bc_right.evaluate(t_next)
                rhs[-1] = (
                    (1.0 - 2.0 * r) * c[-1]
                    + 2.0 * r * c[-2]
                    + 2.0 * r * dx * (g_right_curr + g_right_next)
                )
                if reactions and sp in reactions:
                    rhs[-1] += dt * reactions[sp][-1]

            new_c[sp] = solve_banded((1, 1), ab, rhs)

        return new_c


# Also register the alias 'crank-nicolson'
BaseSolver._registry["crank-nicolson"] = CrankNicolson

