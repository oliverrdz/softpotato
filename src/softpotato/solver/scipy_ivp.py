"""Method of Lines diffusion solver using scipy.integrate.solve_ivp."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .base import BaseSolver, DiffusionProblem, SolverResult


class ScipyIVPSolver(BaseSolver, name="scipy_ivp"):
    r"""Diffusion solver using Method of Lines (MOL) and ``scipy.integrate.solve_ivp``.

    Discretizes the spatial diffusion operator into a coupled system of ODEs:

    .. math::

        \frac{dc_i}{dt} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(\mathbf{c}, x, t)

    Integrates the resulting stiff or non-stiff system using SciPy's ODE suite.
    Defaults to the stiff L-stable ``'Radau'`` integrator, suitable for diffusion-reaction systems.

    Parameters
    ----------
    method : {'Radau', 'BDF', 'RK45', 'RK23', 'DOP853', 'LSODA'}, default 'Radau'
        ODE integration algorithm passed to ``scipy.integrate.solve_ivp``.
        Recommended options:

        - ``'Radau'`` (default): 5th-order implicit Runge-Kutta; highly recommended for stiff
          diffusion and fast reaction-diffusion equations.
        - ``'BDF'``: Variable-order multi-step Backward Differentiation Formula; well-suited
          for large stiff systems.
        - ``'RK45'``: Explicit Runge-Kutta order 4(5); suitable for non-stiff pure diffusion.
    rtol : float, default 1e-6
        Relative error tolerance for adaptive time stepping.
    atol : float, default 1e-8
        Absolute error tolerance for adaptive time stepping.
    **options : Any
        Additional solver options passed to ``scipy.integrate.solve_ivp``.

    Examples
    --------
    Simulating reaction-diffusion with first-order decay:

    >>> import numpy as np
    >>> from softpotato.solver import ScipyIVPSolver, DiffusionProblem, NeumannBC
    >>>
    >>> grid = np.linspace(0.0, 1.0, 31)
    >>> problem = DiffusionProblem(
    ...     grid=grid,
    ...     diffusivity={"A": 0.01},
    ...     boundary_conditions={"A": (NeumannBC(0.0), NeumannBC(0.0))},
    ...     initial_conditions={"A": 1.0},
    ...     reactions=lambda c_dict, x, t: {"A": -0.5 * c_dict["A"]},
    ... )
    >>> solver = ScipyIVPSolver(method="Radau", rtol=1e-5)
    >>> result = solver.solve(problem, t_span=(0.0, 1.0))
    >>> result.success
    True
    >>> result["A"].shape[1]
    31
    """

    def __init__(
        self,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        **options: Any,
    ) -> None:
        """Initialize SciPy Method of Lines IVP solver."""
        super().__init__(method=method, rtol=rtol, atol=atol, **options)

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        grid = problem.grid
        N = problem.n_points
        t_start, t_end = t_span

        # Determine unknown indices per species based on boundary conditions
        species_info: dict[str, dict[str, Any]] = {}
        total_unknowns = 0

        for sp in problem.species:
            bc_left, bc_right = problem.boundary_conditions[sp]
            idx_start = 1 if bc_left.btype == "dirichlet" else 0
            idx_end = N - 1 if bc_right.btype == "dirichlet" else N
            size = idx_end - idx_start

            species_info[sp] = {
                "idx_start": idx_start,
                "idx_end": idx_end,
                "size": size,
                "offset": total_unknowns,
                "left_dirichlet": bc_left.btype == "dirichlet",
                "right_dirichlet": bc_right.btype == "dirichlet",
            }
            total_unknowns += size

        # Assemble initial state vector y0
        y0 = np.empty(total_unknowns, dtype=float)
        for sp in problem.species:
            info = species_info[sp]
            ic_full = problem.get_initial_profile(sp)
            y0[info["offset"] : info["offset"] + info["size"]] = ic_full[
                info["idx_start"] : info["idx_end"]
            ]

        # Precompute spatial discretization weights
        is_uniform = problem.is_uniform
        if is_uniform:
            dx = problem.dx
            inv_dx2 = 1.0 / (dx**2)
        else:
            h = np.diff(grid)  # h[i] = x[i+1] - x[i]

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """Compute ODE system time derivatives."""
            # 1. Unpack y into full concentration profiles
            c_dict: dict[str, np.ndarray] = {}
            for sp in problem.species:
                info = species_info[sp]
                c_full = np.empty(N, dtype=float)
                bc_left, bc_right = problem.boundary_conditions[sp]

                c_full[info["idx_start"] : info["idx_end"]] = y[
                    info["offset"] : info["offset"] + info["size"]
                ]
                if info["left_dirichlet"]:
                    c_full[0] = bc_left.evaluate(t)
                if info["right_dirichlet"]:
                    c_full[-1] = bc_right.evaluate(t)

                c_dict[sp] = c_full

            # 2. Evaluate chemical reaction rates if specified
            reactions = (
                problem.reactions(c_dict, grid, t) if problem.reactions else None
            )

            # 3. Compute spatial derivatives and fill dy
            dy = np.empty(total_unknowns, dtype=float)
            for sp in problem.species:
                info = species_info[sp]
                c_full = c_dict[sp]
                D = problem.diffusivity[sp]
                bc_left, bc_right = problem.boundary_conditions[sp]

                d2c = np.empty(N, dtype=float)
                if is_uniform:
                    # Interior points
                    d2c[1:-1] = (
                        c_full[2:] - 2.0 * c_full[1:-1] + c_full[:-2]
                    ) * inv_dx2

                    # Boundary points if Neumann
                    if not info["left_dirichlet"]:
                        g_left = bc_left.evaluate(t)
                        d2c[0] = (
                            2.0 * c_full[1] - 2.0 * c_full[0] - 2.0 * dx * g_left
                        ) * inv_dx2

                    if not info["right_dirichlet"]:
                        g_right = bc_right.evaluate(t)
                        d2c[-1] = (
                            2.0 * c_full[-2] - 2.0 * c_full[-1] + 2.0 * dx * g_right
                        ) * inv_dx2
                else:
                    # General 3-point stencil for non-uniform grid
                    # d²c/dx² ~ 2 / (h_{i-1} + h_i) * [ (c_{i+1}-c_i)/h_i - (c_i-c_{i-1})/h_{i-1} ]
                    hi_m1 = h[:-1]
                    hi = h[1:]
                    d2c[1:-1] = (2.0 / (hi_m1 + hi)) * (
                        (c_full[2:] - c_full[1:-1]) / hi
                        - (c_full[1:-1] - c_full[:-2]) / hi_m1
                    )
                    if not info["left_dirichlet"]:
                        g_left = bc_left.evaluate(t)
                        h0 = h[0]
                        d2c[0] = (2.0 / h0) * ((c_full[1] - c_full[0]) / h0 - g_left)
                    if not info["right_dirichlet"]:
                        g_right = bc_right.evaluate(t)
                        hm1 = h[-1]
                        d2c[-1] = (2.0 / hm1) * (
                            g_right - (c_full[-1] - c_full[-2]) / hm1
                        )

                sp_rate = reactions[sp] if reactions and sp in reactions else 0.0
                sp_dcdt = D * d2c + sp_rate

                dy[info["offset"] : info["offset"] + info["size"]] = sp_dcdt[
                    info["idx_start"] : info["idx_end"]
                ]

            return dy

        # Solver options
        method = (
            kwargs.get("method")
            if kwargs.get("method") is not None
            else self.options.get("method", "Radau")
        )
        rtol = (
            kwargs.get("rtol")
            if kwargs.get("rtol") is not None
            else self.options.get("rtol", 1e-6)
        )
        atol = (
            kwargs.get("atol")
            if kwargs.get("atol") is not None
            else self.options.get("atol", 1e-8)
        )

        if t_eval is None:
            # Default to 101 points across t_span
            t_eval = np.linspace(t_start, t_end, 101)
        else:
            t_eval = np.asarray(t_eval, dtype=float)

        # Call scipy.integrate.solve_ivp
        ode_res = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

        if not ode_res.success:
            return SolverResult(
                t=t_eval,
                x=grid,
                concentrations={},
                success=False,
                message=f"solve_ivp failed: {ode_res.message}",
                raw_output=ode_res,
            )

        # Unpack solution into (n_times, N) per species
        n_times = len(ode_res.t)
        records: dict[str, np.ndarray] = {}

        for sp in problem.species:
            info = species_info[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]
            sol_sp = np.empty((n_times, N), dtype=float)

            # Insert solved unknowns
            y_sp = ode_res.y[info["offset"] : info["offset"] + info["size"], :]
            sol_sp[:, info["idx_start"] : info["idx_end"]] = y_sp.T

            # Fill Dirichlet boundaries at evaluated times
            if info["left_dirichlet"]:
                sol_sp[:, 0] = [bc_left.evaluate(ti) for ti in ode_res.t]
            if info["right_dirichlet"]:
                sol_sp[:, -1] = [bc_right.evaluate(ti) for ti in ode_res.t]

            records[sp] = sol_sp

        return SolverResult(
            t=ode_res.t,
            x=grid,
            concentrations=records,
            success=True,
            message="SciPy solve_ivp completed successfully.",
            raw_output=ode_res,
        )


# Also register the alias 'solve_ivp'
BaseSolver._registry["solve_ivp"] = ScipyIVPSolver
