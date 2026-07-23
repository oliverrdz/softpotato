"""
Example 01: Reversible 1D Cyclic Voltammetry Simulation

Demonstrates assembling Soft Potato's modular components to simulate
a reversible single-electron oxidation (R -> O + e-) and plot the voltammogram.
"""

import matplotlib

matplotlib.use("TkAgg")  # or "QtAgg"

import matplotlib.pyplot as plt
import numpy as np

from softpotato.discretizers import FDM1DDiscretizer
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import NernstianEquilibriumBC, TwoSpeciesModel
from softpotato.solvers import ODESolver
from softpotato.techniques import CyclicVoltammetry


def run_cv_example():
    # -------------------------------------------------------------------------
    # 1. Spatial Grid (1D Mesh)
    # -------------------------------------------------------------------------
    # Set maximum boundary x_max based on estimated diffusion layer thickness
    D_R = 1e-9  # m^2/s
    scan_rate = 0.1  # V/s (100 mV/s)
    x_max = 6.0 * np.sqrt(D_R * (1.0 / scan_rate))  # ~1.9 mm

    mesh = Uniform1DMesh(x_min=0.0, x_max=x_max, num_nodes=201)

    # -------------------------------------------------------------------------
    # 2. Physical & Kinetic Model
    # -------------------------------------------------------------------------
    model = TwoSpeciesModel(
        D_R=D_R,
        D_O=1e-9,  # m^2/s
        C_R_bulk=1.0,  # mol/m^3 (1 mM)
        C_O_bulk=0.0,  # mol/m^3
    )

    # -------------------------------------------------------------------------
    # 3. Experimental Technique (Triangular Waveform)
    # -------------------------------------------------------------------------
    technique = CyclicVoltammetry(
        E_start=-0.2,  # Start potential (V)
        E_vertex1=0.3,  # Reversal potential (V)
        scan_rate=scan_rate,  # 100 mV/s
        n_cycles=1,  # Single cycle
    )

    # -------------------------------------------------------------------------
    # 4. Surface Boundary Conditions (Reversible Nernstian Equilibrium)
    # -------------------------------------------------------------------------
    bc = NernstianEquilibriumBC(
        technique=technique,
        E0=0.0,  # Formal reduction potential (V)
        n=1,  # 1-electron process
        T=298.15,  # Temperature (K)
        A=1e-4,  # Electrode area (1 cm^2)
    )

    # -------------------------------------------------------------------------
    # 5. Spatial Discretizer & ODE Integration Engine
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 6. Run Simulation
    # -------------------------------------------------------------------------
    print("Running Cyclic Voltammetry simulation...")
    y0 = model.get_initial_state_vector(mesh.x)
    result = solver.solve(t_span=technique.t_span, y0=y0)
    print("Simulation complete!")

    # -------------------------------------------------------------------------
    # 7. Visualization
    # -------------------------------------------------------------------------
    potentials_mV = result.potential * 1e3  # Convert V to mV
    currents_uA = result.current * 1e6  # Convert A to µA

    plt.figure(figsize=(7, 5))
    plt.plot(potentials_mV, currents_uA, label="Simulated CV", color="navy", lw=2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    plt.xlabel("Potential $E$ (mV vs $E^0$)")
    plt.ylabel(r"Faradaic Current $i$ ($\mu$A)")
    plt.title("Reversible Cyclic Voltammogram ($R \\rightarrow O + e^-$)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cv_example()
