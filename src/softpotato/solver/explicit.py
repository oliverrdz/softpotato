"""Explicit finite difference (Forward-Time Central-Space) diffusion solver."""

from __future__ import annotations

from typing import Any
import numpy as np

from .base import BaseSolver, DiffusionProblem, SolverResult


class ExplicitFiniteDifference(BaseSolver, name="explicit"):
    r"""Explicit Forward-Time Central-Space (FTCS) finite difference diffusion solver.

    Numerically solves 1D multi-species diffusion equations:

    .. math::

        c_i^{n+1} = c_i^n + \Delta t \left[ D \frac{c_{i+1}^n - 2c_i^n + c_{i-1}^n}{\Delta x^2} + R_i^n \right]

    Enforces the Courant-Friedrichs-Lewy (CFL) stability criterion:

    .. math::

        \Delta t \le \frac{\Delta x^2}{2 \max_i(D_i)}

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.solver import ExplicitFiniteDifference, DiffusionProblem, DirichletBC
    >>>
    >>> grid = np.linspace(0.0, 1.0, 21)
    >>> problem = DiffusionProblem(
    ...     grid=grid,
    ...     diffusivity={"A": 0.01},
    ...     boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
    ...     initial_conditions={"A": 0.0},
    ... )
    >>> # Use automatic CFL-safe time stepping
    >>> solver = ExplicitFiniteDifference()
    >>> result = solver.solve(problem, t_span=(0.0, 0.1))
    >>> result.success
    True
    >>> result["A"].shape[1]
    21
    """

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        if not problem.is_uniform:
            raise NotImplementedError("ExplicitFiniteDifference requires a uniform spatial grid.")

        dx = problem.dx
        max_D = max(problem.diffusivity.values())
        cfl_limit = (dx**2) / (2.0 * max_D)

        user_dt = self.options.get("dt", kwargs.get("dt", None))
        if user_dt is not None:
            dt_req = float(user_dt)
            if dt_req > cfl_limit:
                raise ValueError(
                    f"Specified time step dt={dt_req:.3e} exceeds the CFL stability limit "
                    f"dx²/(2*D_max)={cfl_limit:.3e}."
                )
            base_dt = dt_req
        else:
            # Default to a safe sub-critical time step (45% of stability limit)
            base_dt = 0.45 * cfl_limit

        t_start, t_end = t_span
        if t_eval is None:
            # Generate evaluation time grid
            n_steps = max(2, int(np.ceil((t_end - t_start) / base_dt)) + 1)
            t_eval = np.linspace(t_start, t_end, n_steps)
        else:
            t_eval = np.asarray(t_eval, dtype=float)

        # Initialize current state
        curr_c = {sp: problem.get_initial_profile(sp) for sp in problem.species}

        # Solution records: shape (n_times, n_x) per species
        n_times = len(t_eval)
        records = {
            sp: np.zeros((n_times, problem.n_points), dtype=float) for sp in problem.species
        }
        for sp in problem.species:
            records[sp][0, :] = curr_c[sp]

        # Time marching across t_eval intervals with adaptive substepping
        for k in range(n_times - 1):
            t_k = t_eval[k]
            t_next = t_eval[k + 1]
            interval = t_next - t_k
            substeps = max(1, int(np.ceil(interval / base_dt)))
            dt = interval / substeps

            t_current = t_k
            for _ in range(substeps):
                curr_c = self._step_explicit(problem, curr_c, t_current, dt, dx)
                t_current += dt

            for sp in problem.species:
                records[sp][k + 1, :] = curr_c[sp]

        return SolverResult(
            t=t_eval,
            x=problem.grid,
            concentrations=records,
            success=True,
            message="Explicit finite difference solution completed successfully.",
        )

    def _step_explicit(
        self,
        problem: DiffusionProblem,
        c_dict: dict[str, np.ndarray],
        t: float,
        dt: float,
        dx: float,
    ) -> dict[str, np.ndarray]:
        """Advance all species concentration profiles by one time step dt."""
        reactions = problem.reactions(c_dict, problem.grid, t) if problem.reactions else None
        new_c: dict[str, np.ndarray] = {}
        t_next = t + dt

        for sp in problem.species:
            c = c_dict[sp]
            D = problem.diffusivity[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]

            d2c = np.empty_like(c)
            # Interior 2nd-derivative
            d2c[1:-1] = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (dx**2)

            # Left boundary
            if bc_left.btype == "neumann":
                g_left = bc_left.evaluate(t)
                d2c[0] = (2.0 * c[1] - 2.0 * c[0] - 2.0 * dx * g_left) / (dx**2)
            else:
                d2c[0] = 0.0

            # Right boundary
            if bc_right.btype == "neumann":
                g_right = bc_right.evaluate(t)
                d2c[-1] = (2.0 * c[-2] - 2.0 * c[-1] + 2.0 * dx * g_right) / (dx**2)
            else:
                d2c[-1] = 0.0

            # Advance state
            sp_reaction = reactions[sp] if reactions and sp in reactions else 0.0
            next_sp_c = c + dt * (D * d2c + sp_reaction)

            # Enforce Dirichlet boundaries at t_next
            if bc_left.btype == "dirichlet":
                next_sp_c[0] = bc_left.evaluate(t_next)
            if bc_right.btype == "dirichlet":
                next_sp_c[-1] = bc_right.evaluate(t_next)

            new_c[sp] = next_sp_c

        return new_c


# Also register the alias 'ftcs'
BaseSolver._registry["ftcs"] = ExplicitFiniteDifference

