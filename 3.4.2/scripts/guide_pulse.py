"""User Guide: Pulse & Sampled-Current Voltammetry (SCV, DPV, SWV).

Demonstrates:
1. SampledCurrentVoltammetry (SCV) across a potential window
2. DifferentialPulseVoltammetry (DPV)
3. SquareWaveVoltammetry (SWV)
"""

import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
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
# 2. Sampled-Current Voltammetry (SCV)
# ==============================================================================
scv = SampledCurrentVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    dE=0.05,
    duration=2.0,
    sample_times=[0.5, 1.0, 2.0],
)
res_scv = simulate(scv, [ox, red], geometry=geometry, kinetics=kinetics)
sampled = res_scv.sample_currents()
print(f"SCV: {scv.n_steps} steps, sampled at tau = {scv.sample_times} s")

fig_scv, ax_scv = plot_result(res_scv)
ax_scv.set_title("Sampled-Current Voltammetry (SCV)")

# ==============================================================================
# 3. Differential Pulse Voltammetry (DPV)
# ==============================================================================
dpv = DifferentialPulseVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    step_height=0.02,
    pulse_amplitude=-0.025,
    pulse_width=0.05,
    pulse_period=0.5,
)
res_dpv = simulate(dpv, [ox, red], geometry=geometry, kinetics=kinetics)
E_base, _, _, i_diff_dpv = dpv.sample_currents(res_dpv.time, res_dpv.current)
print(
    f"DPV: {dpv.n_steps} pulses, peak |dI| = {abs(i_diff_dpv).max() * 1e6:.2f} uA "
    f"at E = {E_base[abs(i_diff_dpv).argmax()]:.3f} V"
)

fig_dpv, ax_dpv = plot_result(res_dpv)
ax_dpv.set_title("Differential Pulse Voltammetry (DPV)")

# ==============================================================================
# 4. Square-Wave Voltammetry (SWV)
# ==============================================================================
swv = SquareWaveVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    step_height=0.02,
    amplitude=0.025,
    frequency=10.0,
)
res_swv = simulate(swv, [ox, red], geometry=geometry, kinetics=kinetics)
E_step, _, _, i_net_swv = swv.sample_currents(res_swv.time, res_swv.current)
print(
    f"SWV: {swv.n_steps} cycles (f = {swv.frequency} Hz), "
    f"peak |dI| = {abs(i_net_swv).max() * 1e6:.2f} uA "
    f"at E = {E_step[abs(i_net_swv).argmax()]:.3f} V"
)

fig_swv, ax_swv = plot_result(res_swv)
ax_swv.set_title("Square-Wave Voltammetry (SWV)")

plt.show()
