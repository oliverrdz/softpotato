"""User Guide: Linear Sweep (LSV) and Cyclic Voltammetry (CV).

Demonstrates:
1. LinearSweepVoltammetry with simulate() and plot_result()
2. Multi-cycle CyclicVoltammetry with Butler-Volmer kinetics
3. Multi-scan-rate parameter sweep with simulate_sweep() and plot_sweep()
"""

import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    CyclicVoltammetry,
    LinearSweepVoltammetry,
    plot_result,
    plot_sweep,
    simulate,
    simulate_sweep,
)

# ==============================================================================
# 1. Physical System (SI units: m^2/s, mol/m^3 = mM, m^2, m/s)
# ==============================================================================
E0 = 0.0  # Formal potential [V]
D = 1.0e-9  # Diffusion coefficient [m^2/s]
c_bulk = 1.0  # Bulk concentration of Ox [mol/m^3] (= 1.0 mM)
k0 = 1.0e-3  # Heterogeneous rate constant [m/s]
alpha = 0.5  # Transfer coefficient [-]
area = 7.07e-6  # Electrode area [m^2] (3 mm diameter disk)

# Species must be ordered [ox, red] for Ox + e- -> Red
ox = Species(
    "Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0, n_electrons=1
)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)
kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1)
geometry = PlanarGeometry(area)

# ==============================================================================
# 2. Linear Sweep Voltammetry (LSV)
# ==============================================================================
lsv = LinearSweepVoltammetry(E_start=0.4, E_end=-0.4, scan_rate=0.1, dE=0.002)
lsv_result = simulate(lsv, [ox, red], geometry=geometry, kinetics=kinetics)
print(
    f"LSV: duration = {lsv.duration:.2f} s, "
    f"peak cathodic current = {lsv_result.current.min() * 1e6:.2f} uA"
)

fig_lsv, ax_lsv = plot_result(lsv_result)
ax_lsv.set_title("Linear Sweep Voltammetry (v = 0.1 V/s)")

# ==============================================================================
# 3. Multi-Cycle Cyclic Voltammetry (CV)
# ==============================================================================
cv = CyclicVoltammetry(
    E_start=0.4,
    E_vertex1=-0.4,
    E_vertex2=0.4,
    scan_rate=0.1,
    n_cycles=2,
    dE=0.002,
)
cv_result = simulate(cv, [ox, red], geometry=geometry, kinetics=kinetics)
print(
    f"CV (2 cycles): duration = {cv.duration:.2f} s, "
    f"switching times = {cv.event_times}"
)

fig_cv, ax_cv = plot_result(cv_result)
ax_cv.set_title("Cyclic Voltammetry (2 cycles, v = 0.1 V/s)")

# ==============================================================================
# 4. Multi-Scan-Rate Sweep with simulate_sweep()
# ==============================================================================
scan_rates = [0.02, 0.05, 0.1, 0.2, 0.5]
sweep_results = simulate_sweep(
    lambda v: CyclicVoltammetry(
        E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v, dE=0.004
    ),
    [ox, red],
    values=scan_rates,
    geometry=geometry,
    kinetics=kinetics,
)

for v, res in sweep_results.items():
    print(
        f"  v = {v:>4.2f} V/s -> i_pc = {res.current.min() * 1e6:>7.2f} uA, "
        f"i_pa = {res.current.max() * 1e6:>6.2f} uA"
    )

fig_sweep, ax_sweep = plot_sweep(
    sweep_results,
    labels=[f"{v} V/s" for v in scan_rates],
    legend_title="Scan Rate",
)
ax_sweep.set_title("Scan-Rate Series (Cyclic Voltammetry)")

plt.show()
