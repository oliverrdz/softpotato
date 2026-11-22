"""5-Minute First Simulation: Cyclic Voltammetry with simulate() and plot_result().

Edit only the physical and waveform parameters below to simulate a cyclic
voltammogram and plot it in one call—no manual mesh or PDE setup required.
"""

import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import CyclicVoltammetry, plot_result, simulate

# ==============================================================================
# CONFIGURATION — Edit physical & waveform parameters here
# ==============================================================================
# Redox couple & electrode parameters (SI units: m^2/s, mol/m^3 = mM, m^2)
E0 = 0.0               # Formal potential [V]
D = 1.0e-9             # Diffusion coefficient [m^2/s] (= 1.0e-5 cm^2/s)
c_bulk = 1.0           # Bulk concentration of Ox [mol/m^3] (= 1.0 mM)
k0 = 1.0e-3            # Heterogeneous rate constant [m/s] (= 0.1 cm/s)
alpha = 0.5            # Transfer coefficient [-]
area = 7.07e-6         # Electrode area [m^2] (= 0.0707 cm^2, 3 mm disk)

# Cyclic voltammetry waveform
E_start = 0.4  # Initial potential [V]
E_vertex1 = -0.4  # First switching potential [V]
E_vertex2 = 0.4  # Second switching potential [V]
v = 0.1  # Scan rate [V/s]

# ==============================================================================
# Run Simulation & Plot
# ==============================================================================
ox = Species(
    "Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0, n_electrons=1
)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)

cv = CyclicVoltammetry(
    E_start=E_start, E_vertex1=E_vertex1, E_vertex2=E_vertex2, scan_rate=v
)
kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1)
geometry = PlanarGeometry(area)

# Species must be ordered [ox, red] according to the reduction reaction: Ox + e- -> Red
result = simulate(cv, [ox, red], geometry=geometry, kinetics=kinetics)

print(f"Simulated {len(result.time)} time points over {result.time[-1]:.2f} s")
print(f"Peak cathodic current: {result.current.min() * 1e6:.2f} uA")
print(f"Peak anodic current:   {result.current.max() * 1e6:.2f} uA")

fig, ax = plot_result(result)
ax.set_title(f"Cyclic Voltammogram (v = {v} V/s, k0 = {k0:.1e} m/s)")
plt.show()
