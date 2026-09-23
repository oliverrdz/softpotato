"""Implicit finite difference (Backward-Time Central-Space) diffusion solver."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.linalg import solve_banded

from .base import BaseSolver, DiffusionProblem, SolverResult


class ImplicitFiniteDifference(BaseSolver, name="implicit"):
    r"""Implicit Backward-Time Central-Space (BTCS) finite difference diffusion solver.

    Numerically solves 1D multi-species diffusion equations:

    .. math::

        \frac{c_i^{n+1} - c_i^n}{\Delta t} = D \frac{c_{i+1}^{n+1} - 2c_i^{n+1} + c_{i-1}^{n+1}}{\Delta x^2} + R_i^{n+1}

    Unconditionally stable for arbitrary time steps :math:`\Delta t`. Solves the resulting
    tridiagonal system using ``scipy.linalg.solve_banded`` in :math:`\mathcal{O}(N)` time per step.

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
    >>> from softpotato.solver import ImplicitFiniteDifference, DiffusionProblem, DirichletBC
    >>>
    >>> grid = np.linspace(0.0, 1.0, 31)
    >>> problem = DiffusionProblem(
    ...     grid=grid,
    ...     diffusivity={"A": 0.05},
    ...     boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
    ...     initial_conditions={"A": 0.0},
    ... )
    >>> # Implicit schemes remain stable even with large time steps
    >>> solver = ImplicitFiniteDifference(dt=0.05)
    >>> result = solver.solve(problem, t_span=(0.0, 0.5))
    >>> result.success
    True
    >>> result["A"].shape[1]
    31
    """

    def __init__(self, dt: float | None = None, **options: Any) -> None:
        """Initialize Implicit Finite Difference (BTCS) solver."""
        super().__init__(dt=dt, **options)

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        if not problem.is_uniform:
            raise NotImplementedError("ImplicitFiniteDifference requires a uniform spatial grid.")

        dx = problem.dx
        t_start, t_end = t_span

        user_dt = kwargs.get("dt") if kwargs.get("dt") is not None else self.options.get("dt", None)
        if user_dt is not None:
            base_dt = float(user_dt)
        else:
            # Default to dividing total duration into 100 steps
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
                curr_c = self._step_implicit(problem, curr_c, t_current, dt, dx)
                t_current += dt

            for sp in problem.species:
                records[sp][k + 1, :] = curr_c[sp]

        return SolverResult(
            t=t_eval,
            x=problem.grid,
            concentrations=records,
            success=True,
            message="Implicit finite difference solution completed successfully.",
        )

    def _step_implicit(
        self,
        problem: DiffusionProblem,
        c_dict: dict[str, np.ndarray],
        t: float,
        dt: float,
        dx: float,
    ) -> dict[str, np.ndarray]:
        """Advance all species profiles by one BTCS implicit time step."""
        reactions = problem.reactions(c_dict, problem.grid, t) if problem.reactions else None
        new_c: dict[str, np.ndarray] = {}
        N = problem.n_points
        t_next = t + dt

        for sp in problem.species:
            c = c_dict[sp]
            D = problem.diffusivity[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]
            r = D * dt / (dx**2)

            # Assemble banded matrix ab of shape (3, N)
            # ab[0, 1:] = upper diagonal
            # ab[1, :]  = main diagonal
            # ab[2, :-1] = lower diagonal
            ab = np.zeros((3, N), dtype=float)
            rhs = np.copy(c)
            if reactions and sp in reactions:
                rhs += dt * reactions[sp]

            # Interior rows
            ab[1, 1:-1] = 1.0 + 2.0 * r
            ab[0, 2:] = -r       # A[i, i+1]
            ab[2, :-2] = -r      # A[i, i-1]

            # Left boundary (row 0)
            if bc_left.btype == "dirichlet":
                ab[1, 0] = 1.0
                ab[0, 1] = 0.0
                rhs[0] = bc_left.evaluate(t_next)
            else:  # Neumann
                ab[1, 0] = 1.0 + 2.0 * r
                ab[0, 1] = -2.0 * r
                g_left = bc_left.evaluate(t_next)
                rhs[0] -= 2.0 * r * dx * g_left

            # Right boundary (row N-1)
            if bc_right.btype == "dirichlet":
                ab[1, -1] = 1.0
                ab[2, -2] = 0.0
                rhs[-1] = bc_right.evaluate(t_next)
            else:  # Neumann
                ab[1, -1] = 1.0 + 2.0 * r
                ab[2, -2] = -2.0 * r
                g_right = bc_right.evaluate(t_next)
                rhs[-1] += 2.0 * r * dx * g_right

            # Solve tridiagonal system in O(N)
            new_c[sp] = solve_banded((1, 1), ab, rhs)

        return new_c


# Also register the alias 'btcs'
BaseSolver._registry["btcs"] = ImplicitFiniteDifference

