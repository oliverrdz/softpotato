"""Method of Lines diffusion solver using scipy.integrate.solve_ivp."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from softpotato.geometry import Geometry

from .base import BaseSolver, DiffusionProblem, SolverResult

_IMPLICIT_JAC_METHODS = frozenset({"Radau", "BDF", "LSODA"})


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
    use_jacobian : bool, default True
        When ``True`` and using an implicit method (``'Radau'``, ``'BDF'``,
        ``'LSODA'``), automatically assemble and pass the coupled sparse
        analytical Jacobian :math:`\mathbf{J}_{\text{diffusion}} + \mathbf{J}_{\text{reactions}}`
        whenever ``problem.reactions`` provides a kinetic Jacobian (e.g.
        ``ReactionNetwork``, ``HomogeneousReaction``, ``Mechanism``, or a
        custom object implementing ``local_jacobian`` / ``jacobian``). Set to
        ``False`` to force SciPy's numerical finite-difference Jacobian.
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
        use_jacobian: bool = True,
        **options: Any,
    ) -> None:
        """Initialize SciPy Method of Lines IVP solver."""
        super().__init__(
            method=method,
            rtol=rtol,
            atol=atol,
            use_jacobian=use_jacobian,
            **options,
        )

    @staticmethod
    def _build_diffusion_jacobian(
        geom: Geometry,
        grid: np.ndarray,
        problem: DiffusionProblem,
        species_info: dict[str, dict[str, Any]],
    ) -> sparse.csc_matrix:
        """Assemble constant sparse block-tridiagonal diffusion Jacobian J_diff."""
        n_pts = len(grid)
        blocks: list[sparse.csc_matrix] = []
        for sp in problem.species:
            info = species_info[sp]
            bc_left, bc_right = problem.boundary_conditions[sp]
            l_banded, _ = geom.laplacian_matrix(
                grid, bc_left=bc_left, bc_right=bc_right
            )
            l_full = sparse.diags(
                [l_banded[2, :-1], l_banded[1, :], l_banded[0, 1:]],
                offsets=[-1, 0, 1],
                shape=(n_pts, n_pts),
                format="csc",
            )
            d_val = problem.diffusivity[sp]
            sub = (d_val * l_full)[
                info["idx_start"] : info["idx_end"],
                info["idx_start"] : info["idx_end"],
            ]
            blocks.append(sub.tocsc())
        return sparse.block_diag(blocks, format="csc")

    @staticmethod
    def _assemble_reaction_jacobian_from_local(
        local_jac: np.ndarray,
        species_names: list[str],
        species_info: dict[str, dict[str, Any]],
        n_pts: int,
        total_unknowns: int,
    ) -> sparse.csc_matrix:
        """Map node-wise local Jacobian (S, S, N) onto unknown state indices."""
        arr = np.asarray(local_jac, dtype=float)
        if arr.ndim == 2:
            arr = np.broadcast_to(arr[:, :, np.newaxis], (*arr.shape, n_pts))

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        for i, sp_i in enumerate(species_names):
            info_i = species_info[sp_i]
            start_i = info_i["idx_start"]
            end_i = info_i["idx_end"]
            offset_i = info_i["offset"]
            for k, sp_k in enumerate(species_names):
                info_k = species_info[sp_k]
                j_start = max(start_i, info_k["idx_start"])
                j_end = min(end_i, info_k["idx_end"])
                if j_end <= j_start:
                    continue
                vals = arr[i, k, j_start:j_end]
                if np.any(vals != 0.0):
                    j_idx = np.arange(j_start, j_end)
                    rows.append(offset_i + (j_idx - start_i))
                    cols.append(info_k["offset"] + (j_idx - info_k["idx_start"]))
                    data.append(vals)

        if not data:
            return sparse.csc_matrix((total_unknowns, total_unknowns), dtype=float)
        return sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(total_unknowns, total_unknowns),
        ).tocsc()

    @staticmethod
    def _assemble_reaction_jacobian_from_dict(
        jac_dict: Mapping[tuple[str, str], np.ndarray],
        species_info: dict[str, dict[str, Any]],
        n_pts: int,
        total_unknowns: int,
    ) -> sparse.csc_matrix:
        """Map {(sp_i, sp_k): dR_i/dc_k} dict onto unknown state indices."""
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        for (sp_i, sp_k), block in jac_dict.items():
            if sp_i not in species_info or sp_k not in species_info:
                continue
            info_i = species_info[sp_i]
            info_k = species_info[sp_k]
            j_start = max(info_i["idx_start"], info_k["idx_start"])
            j_end = min(info_i["idx_end"], info_k["idx_end"])
            if j_end <= j_start:
                continue
            arr = np.broadcast_to(np.asarray(block, dtype=float), (n_pts,))
            vals = arr[j_start:j_end]
            if np.any(vals != 0.0):
                j_idx = np.arange(j_start, j_end)
                rows.append(info_i["offset"] + (j_idx - info_i["idx_start"]))
                cols.append(info_k["offset"] + (j_idx - info_k["idx_start"]))
                data.append(vals)

        if not data:
            return sparse.csc_matrix((total_unknowns, total_unknowns), dtype=float)
        return sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(total_unknowns, total_unknowns),
        ).tocsc()

    def _run_solver(
        self,
        problem: DiffusionProblem,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
        **kwargs: Any,
    ) -> SolverResult:
        geom = problem.geometry
        assert isinstance(geom, Geometry)
        grid = np.asarray(problem.grid, dtype=float)
        N = problem.n_points
        t_start, t_end = t_span
        species_names = list(problem.species)

        # Determine unknown indices per species based on boundary conditions
        species_info: dict[str, dict[str, Any]] = {}
        total_unknowns = 0

        for sp in species_names:
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
        for sp in species_names:
            info = species_info[sp]
            ic_full = problem.get_initial_profile(sp)
            y0[info["offset"] : info["offset"] + info["size"]] = ic_full[
                info["idx_start"] : info["idx_end"]
            ]

        def unpack_state(t: float, y: np.ndarray) -> dict[str, np.ndarray]:
            """Unpack flat unknown state vector y into full concentration profiles."""
            c_dict: dict[str, np.ndarray] = {}
            for sp in species_names:
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
            return c_dict

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """Compute ODE system time derivatives."""
            # 1. Unpack y into full concentration profiles
            c_dict = unpack_state(t, y)

            # 2. Evaluate chemical reaction rates if specified
            reactions = (
                problem.reactions(c_dict, grid, t) if problem.reactions else None
            )

            # 3. Compute spatial derivatives and fill dy
            dy = np.empty(total_unknowns, dtype=float)
            for sp in species_names:
                info = species_info[sp]
                c_full = c_dict[sp]
                D = problem.diffusivity[sp]
                bc_left, bc_right = problem.boundary_conditions[sp]

                lap = geom.laplacian(
                    c_full, grid, bc_left=bc_left, bc_right=bc_right, t=t
                )
                sp_rate = reactions[sp] if reactions and sp in reactions else 0.0
                sp_dcdt = D * lap + sp_rate

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
        use_jacobian = (
            kwargs.get("use_jacobian")
            if kwargs.get("use_jacobian") is not None
            else self.options.get("use_jacobian", True)
        )
        explicit_jac = (
            kwargs.get("jac") if "jac" in kwargs else self.options.get("jac", None)
        )

        # Assemble coupled analytical Jacobian for stiff implicit methods
        jac_arg: Any = None
        if method in _IMPLICIT_JAC_METHODS:
            if explicit_jac is not None:
                jac_arg = explicit_jac
            elif use_jacobian and problem.reactions is not None:
                rxn_obj = problem.reactions
                local_jac_fn: Any = getattr(rxn_obj, "local_jacobian", None)
                rxn_jac_fn: Any = getattr(rxn_obj, "jacobian", None)
                has_local_jac = callable(local_jac_fn)
                has_jac = callable(rxn_jac_fn)

                if has_local_jac or has_jac:
                    dense_jac = method == "LSODA"
                    j_diff = self._build_diffusion_jacobian(
                        geom, grid, problem, species_info
                    )

                    if has_local_jac:

                        def jac_from_local(t: float, y: np.ndarray) -> Any:
                            c_dict = unpack_state(t, y)
                            local_j = local_jac_fn(
                                c_dict, grid, t, species=species_names
                            )
                            j_rxn = self._assemble_reaction_jacobian_from_local(
                                local_j, species_names, species_info, N, total_unknowns
                            )
                            j_tot = j_diff + j_rxn
                            return j_tot.toarray() if dense_jac else j_tot

                        jac_arg = jac_from_local
                    else:
                        active_indices = np.concatenate(
                            [
                                i * N
                                + np.arange(
                                    species_info[sp]["idx_start"],
                                    species_info[sp]["idx_end"],
                                )
                                for i, sp in enumerate(species_names)
                            ]
                        )

                        def jac_from_method(t: float, y: np.ndarray) -> Any:
                            c_dict = unpack_state(t, y)
                            try:
                                raw_j = rxn_jac_fn(
                                    c_dict, grid, t, species=species_names
                                )
                            except TypeError:
                                raw_j = rxn_jac_fn(c_dict, grid, t)

                            if isinstance(raw_j, Mapping):
                                j_rxn = self._assemble_reaction_jacobian_from_dict(
                                    raw_j, species_info, N, total_unknowns
                                )
                            else:
                                mat_csc = sparse.csc_matrix(raw_j)
                                j_rxn = mat_csc[active_indices, :][:, active_indices]
                            j_tot = j_diff + j_rxn
                            return j_tot.toarray() if dense_jac else j_tot

                        jac_arg = jac_from_method

        if t_eval is None:
            # Default to 101 points across t_span
            t_eval = np.linspace(t_start, t_end, 101)
        else:
            t_eval = np.asarray(t_eval, dtype=float)

        # Call scipy.integrate.solve_ivp
        ivp_kwargs: dict[str, Any] = {
            "fun": rhs,
            "t_span": t_span,
            "y0": y0,
            "method": method,
            "t_eval": t_eval,
            "rtol": rtol,
            "atol": atol,
        }
        if jac_arg is not None:
            ivp_kwargs["jac"] = jac_arg

        ode_res = solve_ivp(**ivp_kwargs)

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
