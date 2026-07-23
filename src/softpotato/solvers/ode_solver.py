from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from softpotato.core.abcs import (
    BaseBoundaryCondition,
    BaseDiscretizer,
    BaseMesh,
    BaseModel,
    BaseSolver,
)


class SimulationResult(NamedTuple):
    """Container for time-series simulation outputs."""

    t: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    current: np.ndarray


class ODESolver(BaseSolver):
    """
    Time integration engine using SciPy `solve_ivp`.
    """

    def __init__(
        self,
        mesh: BaseMesh,
        model: BaseModel,
        discretizer: BaseDiscretizer,
        bc: BaseBoundaryCondition,
        method: str = "BDF",
        atol: float = 1e-8,
        rtol: float = 1e-6,
    ) -> None:
        self.mesh = mesh
        self.model = model
        self.discretizer = discretizer
        self.bc = bc
        self.method = method
        self.atol = atol
        self.rtol = rtol

        self.A = self.discretizer.build_system_matrix(self.mesh, self.model)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Evaluate semi-discrete ODE Right-Hand Side dy/dt = f(t, y).
        """
        y_bc = self.bc.apply(y, t, self.mesh, self.model)
        dydt = self.A @ y_bc

        # Zero out boundary node time derivatives (nodes fixed by BCs)
        N = self.mesh.num_nodes
        dydt[0] = 0.0
        dydt[N - 1] = 0.0
        dydt[N] = 0.0
        dydt[2 * N - 1] = 0.0

        return dydt

    def solve(
        self,
        t_span: tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray | None = None,
    ) -> SimulationResult:
        """
        Execute time integration across target time interval `t_span`.
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=y0,
            method=self.method,
            t_eval=t_eval,
            atol=self.atol,
            rtol=self.rtol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        potentials = np.zeros_like(sol.t)
        currents = np.zeros_like(sol.t)

        technique = getattr(self.bc, "technique", None)

        for idx, (t, y_step) in enumerate(zip(sol.t, sol.y.T)):
            if technique is not None:
                potentials[idx] = technique(t)

            y_bc = self.bc.apply(y_step, t, self.mesh, self.model)
            currents[idx] = self.bc.calculate_current(y_bc, self.mesh, model=self.model)

        return SimulationResult(
            t=sol.t, y=sol.y, potential=potentials, current=currents
        )
