"""User Guide: Step Techniques — Chronoamperometry (CA) & Chronopotentiometry (CP).

Demonstrates:
1. Single-step and double-step Chronoamperometry (potentiostatic)
2. Single-step and current-reversal Chronopotentiometry (galvanostatic)
3. Automatic Sand transition time detection in CP metadata
"""

import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    Chronoamperometry,
    Chronopotentiometry,
    plot_result,
    simulate,
)

# ==============================================================================
# 1. Physical System (SI units: m^2/s, mol/m^3 = mM, m^2, m/s)
# ==============================================================================
E0 = 0.0  # Formal potential [V]
D = 1.0e-9  # Diffusion coefficient [m^2/s]
c_bulk = 1.0  # Bulk concentration of Ox [mol/m^3] (= 1.0 mM)
k0 = 1.0e-3  # Heterogeneous rate constant [m/s]
alpha = 0.5  # Transfer coefficient [-]
area = 7.07e-6  # Electrode area [m^2]

ox = Species(
    "Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0, n_electrons=1
)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)
kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1)
geometry = PlanarGeometry(area)

# ==============================================================================
# 2. Single-Step & Double-Step Chronoamperometry (CA)
# ==============================================================================
ca_single = Chronoamperometry(E_step=-0.4, duration=5.0)
res_ca = simulate(ca_single, [ox, red], geometry=geometry, kinetics=kinetics)
print(
    f"Single-step CA: i(0.5 s) = {res_ca.current[len(res_ca.time) // 10] * 1e6:.2f} uA, "
    f"i(5.0 s) = {res_ca.current[-1] * 1e6:.2f} uA"
)

# Double-step CA with an initial 0.5 s quiet hold at +0.3 V
ca_double = Chronoamperometry(
    E_step1=-0.4,
    E_step2=0.4,
    t_step1=2.0,
    t_step2=2.0,
    E_quiet=0.3,
    t_quiet=0.5,
)
res_dca = simulate(ca_double, [ox, red], geometry=geometry, kinetics=kinetics)
print(f"Double-step CA: step transitions at t = {ca_double.event_times} s")

fig_ca, ax_ca = plot_result(res_dca)
ax_ca.set_title("Double-Step Chronoamperometry (with Quiet Hold)")

# ==============================================================================
# 3. Single-Step & Current-Reversal Chronopotentiometry (CP)
# ==============================================================================
# Negative current = cathodic (reduction) under IUPAC convention
cp_single = Chronopotentiometry(current=-1.5e-5, duration=4.0)
res_cp = simulate(cp_single, [ox, red], geometry=geometry, kinetics=kinetics)
tau = res_cp.metadata.get("transition_time")
print(f"Single-step CP: Sand transition time tau = {tau:.3f} s" if tau else "No transition")

fig_cp, ax_cp = plot_result(res_cp)
ax_cp.set_title("Constant-Current Chronopotentiometry")

# Current-reversal CP: reduce for 1.5 s, then oxidize for 1.5 s
cp_rev = Chronopotentiometry(currents=[-1.5e-5, 1.5e-5], durations=[1.5, 1.5])
res_rcp = simulate(cp_rev, [ox, red], geometry=geometry, kinetics=kinetics)
print(f"Current-reversal CP: reversal at t = {cp_rev.event_times} s")

fig_rcp, ax_rcp = plot_result(res_rcp)
ax_rcp.set_title("Current-Reversal Chronopotentiometry")

plt.show()
