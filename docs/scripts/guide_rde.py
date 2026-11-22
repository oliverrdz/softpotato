"""User Guide: Rotating Disk Electrode (RDE) & Hydrodynamic Sweeps.

Demonstrates:
1. Single-rate RotatingDiskElectrode simulation via simulate()
2. Multi-rotation-rate sweep with simulate_rde_sweep()
3. Comparison of simulated limiting currents against the analytical Levich equation
"""

import matplotlib.pyplot as plt

from softpotato.analytical.hydrodynamics import levich
from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    RotatingDiskElectrode,
    plot_result,
    plot_sweep,
    simulate,
    simulate_rde_sweep,
)

# ==============================================================================
# 1. Physical System (SI units: m^2/s, mol/m^3 = mM, m^2, m/s)
# ==============================================================================
E0 = 0.0  # Formal potential [V]
D = 1.0e-9  # Diffusion coefficient [m^2/s]
c_bulk = 1.0  # Bulk concentration of Ox [mol/m^3] (= 1.0 mM)
nu = 1.0e-6  # Electrolyte kinematic viscosity [m^2/s] (water at 298 K)
k0 = 1.0e-3  # Heterogeneous rate constant [m/s]
alpha = 0.5  # Transfer coefficient [-]
area = 1.96e-5  # Disk electrode area [m^2] (5 mm diameter disk)

ox = Species(
    "Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0, n_electrons=1
)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)
kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1)
geometry = PlanarGeometry(area)

# ==============================================================================
# 2. Single Rotation Rate RDE via simulate()
# ==============================================================================
rde = RotatingDiskElectrode(
    E_start=0.4,
    E_end=-0.4,
    scan_rate=0.05,
    rpm=1600,
    kinematic_viscosity=nu,
    model="nernst_layer",
)
res_rde = simulate(rde, [ox, red], geometry=geometry, kinetics=kinetics)
delta_n = rde.diffusion_layer_thickness(D)
print(
    f"RDE (1600 rpm): omega = {rde.omega:.2f} rad/s, "
    f"delta_N = {delta_n * 1e6:.2f} um, "
    f"I_lim(sim) = {abs(res_rde.current[-1]) * 1e6:.2f} uA"
)

fig_rde, ax_rde = plot_result(res_rde)
ax_rde.set_title("RDE Linear Sweep Voltammogram (1600 rpm)")

# ==============================================================================
# 3. Multi-Rotation-Rate Sweep with simulate_rde_sweep() & Butler-Volmer
# ==============================================================================
rpms = [400, 900, 1600, 2500]
sweep_res = simulate_rde_sweep(
    E_start=0.4,
    E_end=-0.4,
    scan_rate=0.05,
    rpms=rpms,
    kinematic_viscosity=nu,
    model="convective_diffusion",
    D=D,
    c_bulk=c_bulk,
    kinetics=kinetics,
    area=area,
    solver_kwargs={"rtol": 1e-6, "atol": 1e-9},
)

for rpm, omega, i_lim_sim in zip(rpms, sweep_res.omegas, sweep_res.limiting_currents):
    i_lim_levich = levich(omega=omega, n=1, D=D, c_bulk=c_bulk, area=area, nu=nu)
    print(
        f"  {rpm:>4d} rpm: |I_lim(sim)| = {abs(i_lim_sim) * 1e6:>7.2f} uA, "
        f"Levich = {i_lim_levich * 1e6:>7.2f} uA"
    )

fig_sweep, ax_sweep = plot_sweep(sweep_res, legend_title="Rotation Rate")
ax_sweep.set_title("RDE Rotation-Rate Series (Convective-Diffusion, Butler-Volmer)")

plt.show()
