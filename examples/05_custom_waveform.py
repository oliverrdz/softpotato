import matplotlib.pyplot as plt
import numpy as np

from softpotato.discretizers import FDM1DDiscretizer
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import ButlerVolmerBC, TwoSpeciesModel
from softpotato.solvers import ODESolver
from softpotato.techniques import (
    Chronoamperometry,
    CustomWaveform,
    LinearSweepVoltammetry,
)


def run_custom_waveform_simulation():
    print("Setting up Custom Waveform simulation...")

    # 1. System and Kinetics Parameters
    n = 1
    T = 298.15
    A = 1e-4  # 1 cm^2
    D_R = 1e-9
    D_O = 1e-9
    C_R_bulk = 1.0  # 1 mM
    C_O_bulk = 0.0

    E0 = 0.0
    k0 = 1e-4
    alpha = 0.5

    # 2. Build Chained Sequence: CA -> LSV -> CA
    ca_pre = Chronoamperometry(E_init=0.0, E_step1=0.0, t_step1=2.0)
    lsv_sweep = LinearSweepVoltammetry(E_start=-0.3, E_end=0.5, scan_rate=0.1)
    ca_post = Chronoamperometry(E_init=0.5, E_step1=-0.1, t_step1=3.0)

    technique = CustomWaveform(techniques=[ca_pre, lsv_sweep, ca_post])

    print(f"Total waveform duration: {technique.t_total:.2f} s")

    # 3. Mesh & Model Assembly
    x_max = 6.0 * np.sqrt(D_R * technique.t_total)
    mesh = Uniform1DMesh(x_min=0.0, x_max=x_max, num_nodes=250)

    model = TwoSpeciesModel(D_R=D_R, D_O=D_O, C_R_bulk=C_R_bulk, C_O_bulk=C_O_bulk)
    bc = ButlerVolmerBC(technique=technique, E0=E0, k0=k0, alpha=alpha, n=n, T=T, A=A)

    # 4. Discretization and Solver Setup
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

    # 5. Execute Simulation
    print("Integrating system across custom multi-technique sequence...")
    y0 = model.get_initial_state_vector(mesh.x)
    result = solver.solve(t_span=technique.t_span, y0=y0)
    print("Simulation completed successfully!")

    # 6. Visualization
    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    # Signal E(t)
    ax1.plot(result.t, result.potential, color="tab:purple", lw=2)
    ax1.set_ylabel("Potential $E$ (V)")
    ax1.set_title("Custom Chained Waveform (CA $\\rightarrow$ LSV $\\rightarrow$ CA)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Faradaic Current Transient i(t)
    i_microamps = result.current * 1e6
    ax2.plot(result.t, i_microamps, color="tab:green", lw=2)
    ax2.axhline(0, color="gray", linestyle=":", lw=1)
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel(r"Faradaic Current $i$ ($\mu$A)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_custom_waveform_simulation()
