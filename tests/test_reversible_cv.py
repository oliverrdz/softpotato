import numpy as np

from softpotato.discretizers.fdm_1d import FDM1DDiscretizer
from softpotato.mesh.uniform_1d import Uniform1DMesh
from softpotato.physics.nernst_bc import NernstianEquilibriumBC
from softpotato.physics.species import TwoSpeciesModel
from softpotato.solvers.ode_solver import ODESolver
from softpotato.techniques import CyclicVoltammetry


def randles_sevcik_peak_current(
    n: int, A: float, C_bulk: float, D: float, v: float, T: float = 298.15
) -> float:
    """
    Calculate theoretical peak current using Randles-Sevcik equation:
    i_p = 0.4463 * n * F * A * C^* * sqrt( (n * F * v * D) / (R * T) )
    """
    F = 96485.3321
    R = 8.3144626
    constant = 0.4463
    sigma = (n * F * v * D) / (R * T)
    i_p = constant * n * F * A * C_bulk * np.sqrt(sigma)
    return float(i_p)


def test_randles_sevcik_peak_current_validation():
    """
    Validate simulated CV peak current against Randles-Sevcik analytical formula.
    """
    n = 1
    T = 298.15
    D_R = 1e-9  # m^2/s
    D_O = 1e-9  # m^2/s
    C_bulk = 1.0  # mol/m^3 (1 mM)
    A = 1e-4  # m^2 (1 cm^2)
    v = 0.1  # V/s (100 mV/s)
    E0 = 0.0  # V

    x_max = 6.0 * np.sqrt(D_R * (1.0 / v))
    mesh = Uniform1DMesh(x_min=0.0, x_max=x_max, num_nodes=201)

    model = TwoSpeciesModel(D_R=D_R, D_O=D_O, C_R_bulk=C_bulk, C_O_bulk=0.0)
    technique = CyclicVoltammetry(E_start=-0.2, E_vertex1=0.3, scan_rate=v)
    bc = NernstianEquilibriumBC(technique=technique, E0=E0, n=n, T=T, A=A)

    discretizer = FDM1DDiscretizer()
    solver = ODESolver(
        mesh=mesh,
        model=model,
        discretizer=discretizer,
        bc=bc,
        method="BDF",
        atol=1e-8,
        rtol=1e-6,
    )

    y0 = model.get_initial_state_vector(mesh.x)
    result = solver.solve(t_span=technique.t_span, y0=y0)

    # Extract forward anodic peak current
    t_vertex1 = technique.t_leg1_first
    forward_indices = np.where(result.t <= t_vertex1)[0]

    i_forward = result.current[forward_indices]
    i_p_simulated = np.max(i_forward)

    i_p_theoretical = randles_sevcik_peak_current(
        n=n, A=A, C_bulk=C_bulk, D=D_R, v=v, T=T
    )

    relative_error = abs(i_p_simulated - i_p_theoretical) / i_p_theoretical
    assert relative_error < 0.015
