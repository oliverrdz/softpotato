import matplotlib.pyplot as plt
import numpy as np

from softpotato.discretizers import FDM1DDiscretizer
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import ButlerVolmerBC, TwoSpeciesModel
from softpotato.solvers import ODESolver
from softpotato.techniques import LinearSweepVoltammetry


def run_linear_sweep_voltammetry_simulation():
    print("Setting up Linear Sweep Voltammetry (LSV) simulation...")

    # 1. Physical and System Parameters
    n = 1  # Number of electrons transferred
    T = 298.15  # Temperature (K)
    A = 1e-4  # Electrode area (m^2) [1 cm^2]
    D_R = 1e-9  # Diffusivity of species R (m^2/s)
    D_O = 1e-9  # Diffusivity of species O (m^2/s)
    C_R_bulk = 1.0  # Bulk concentration of species R (mol/m^3) [1 mM]
    C_O_bulk = 0.0  # Bulk concentration of species O (mol/m^3)

    # 2. Heterogeneous Kinetics Parameters (Butler-Volmer)
    E0 = 0.0  # Formal potential (V)
    k0 = 1e-4  # Standard rate constant (m/s)
    alpha = 0.5  # Charge transfer coefficient

    # 3. Excitation Waveform (Linear Sweep Voltammetry)
    technique = LinearSweepVoltammetry(
        E_start=-0.4,  # Starting potential (V)
        E_end=0.4,  # End potential (V)
        scan_rate=0.1,  # Scan rate v = 0.1 V/s (100 mV/s)
    )

    # 4. Spatial Domain Mesh Setup
    # x_max ≈ 6 * sqrt(D * t_total) to avoid boundary effects
    x_max = 6.0 * np.sqrt(D_R * technique.t_total)
    mesh = Uniform1DMesh(x_min=0.0, x_max=x_max, num_nodes=250)

    # 5. Assemble Physical Transport Model & Boundary Conditions
    model = TwoSpeciesModel(D_R=D_R, D_O=D_O, C_R_bulk=C_R_bulk, C_O_bulk=C_O_bulk)
    bc = ButlerVolmerBC(technique=technique, E0=E0, k0=k0, alpha=alpha, n=n, T=T, A=A)

    # 6. Discretizer & ODE Solver Engine
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

    # 7. Run Simulation
    print(f"Sweeping potential from {technique.E_start} V to {technique.E_end} V...")
    y0 = model.get_initial_state_vector(mesh.x)
    result = solver.solve(t_span=technique.t_span, y0=y0)
    print("Simulation completed successfully!")

    # 8. Visualization & Output
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Excitation Signal E(t) vs Time t
    ax1.plot(result.t, result.potential, color="tab:red", lw=2)
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel("Applied Potential $E$ (V)")
    ax1.set_title("Potential Ramp $E(t)$")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Linear Sweep Voltammogram i(E) vs Applied Potential E
    i_microamps = result.current * 1e6
    ax2.plot(result.potential, i_microamps, color="tab:blue", lw=2)
    ax2.axhline(0, color="gray", linestyle=":", lw=1)
    ax2.set_xlabel("Applied Potential $E$ (V)")
    ax2.set_ylabel(r"Faradaic Current $i$ ($\mu$A)")
    ax2.set_title("Linear Sweep Voltammogram")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_linear_sweep_voltammetry_simulation()
