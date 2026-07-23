import numpy as np

from softpotato.core.abcs import (
    BaseBoundaryCondition,
    BaseMesh,
    BaseModel,
    BaseTechnique,
)


class NernstianEquilibriumBC(BaseBoundaryCondition):
    """
    Reversible surface Nernstian equilibrium boundary condition for electroactive species $R$ and $O$.
    """

    FARADAY_CONSTANT = 96485.3321  # C / mol
    GAS_CONSTANT = 8.3144626  # J / (mol K)

    def __init__(
        self,
        technique: BaseTechnique,
        E0: float = 0.0,
        n: int = 1,
        T: float = 298.15,
        A: float = 1.0,
    ) -> None:
        self.technique = technique
        self.E0 = float(E0)
        self.n = int(n)
        self.T = float(T)
        self.A = float(A)

        self.f = (self.n * self.FARADAY_CONSTANT) / (self.GAS_CONSTANT * self.T)

    def get_surface_ratio(self, t: float) -> float:
        """Evaluate Nernst surface ratio $\theta(t) = C_O(0,t) / C_R(0,t)$."""
        E_t = self.technique(t)
        exponent = np.clip(self.f * (E_t - self.E0), -500.0, 500.0)
        return float(np.exp(exponent))

    def apply(
        self, state: np.ndarray, t: float, mesh: BaseMesh, model: BaseModel
    ) -> np.ndarray:
        """
        Enforce 2nd-order surface Nernstian equilibrium at node 0 and bulk concentrations at node N-1.
        """
        N = mesh.num_nodes
        C_R = state[:N]
        C_O = state[N:]

        diff_coeffs = model.get_diffusion_coefficients()
        species_names = model.species_names
        D_R = diff_coeffs[species_names[0]]
        D_O = diff_coeffs[species_names[1]]

        theta = self.get_surface_ratio(t)

        # 2nd-order 3-point forward flux balance at x=0
        numerator = 4.0 * (D_R * C_R[1] + D_O * C_O[1]) - (D_R * C_R[2] + D_O * C_O[2])
        denominator = 3.0 * (D_R + D_O * theta)

        C_R_surf = numerator / denominator
        C_O_surf = theta * C_R_surf

        C_R[0] = C_R_surf
        C_O[0] = C_O_surf

        # Bulk boundary conditions at node N-1 (x=L)
        ics = model.get_initial_conditions(mesh.x)
        C_R[-1] = ics[species_names[0]][-1]
        C_O[-1] = ics[species_names[1]][-1]

        return np.concatenate([C_R, C_O])

    def calculate_current(
        self,
        state: np.ndarray,
        mesh: BaseMesh,
        model: BaseModel | None = None,
        A: float | None = None,
    ) -> float:
        """
        Calculate Faradaic oxidation current $i(t)$ using second-order 3-point forward finite difference:

        $$i(t) = -n F A D_O \\left. \\frac{\\partial C_O}{\\partial x} \\right|_{x=0}
               \\approx -n F A D_O \\frac{-3 C_{O,0} + 4 C_{O,1} - C_{O,2}}{2 \\Delta x}$$
        """
        N = mesh.num_nodes
        dx = float(mesh.dx)
        C_O = state[N:]

        dC_O_dx = (-3.0 * C_O[0] + 4.0 * C_O[1] - C_O[2]) / (2.0 * dx)

        D_O = 1e-9
        if model is not None:
            diff_coeffs = model.get_diffusion_coefficients()
            D_O = diff_coeffs[model.species_names[1]]

        area = self.A if A is None else float(A)

        i_faradaic = -self.n * self.FARADAY_CONSTANT * area * D_O * dC_O_dx
        return float(i_faradaic)
