"""5-Minute Analytical Baselines: Cottrell, Levich, Randles-Sevcik, and SECM.

Edit only the physical parameters in the CONFIGURATION section below to compute
exact theoretical currents and approach curves without setting up any PDEs or meshes.
"""

import matplotlib.pyplot as plt
import numpy as np

from softpotato import analytical as spa

# ==============================================================================
# CONFIGURATION — Edit physical parameters here
# ==============================================================================
n = 1                  # Number of electrons transferred
D = 6.7e-6             # Diffusion coefficient [cm^2/s] (e.g. ferrocyanide in water)
c_bulk = 1.0e-6        # Bulk concentration [mol/cm^3] (1.0e-6 mol/cm^3 = 1.0 mM)
area = 0.0707          # Electrode area [cm^2] (3 mm diameter disk)
nu = 0.01              # Kinematic viscosity [cm^2/s] (aqueous ~ 0.01 cm^2/s)
T = 298.15             # Temperature [K]

# SECM tip geometry & kinetics
RG = 5.0               # Insulator-to-electrode radius ratio (r_g / a)
kappa = 2.0            # Dimensionless substrate rate constant (k * a / D)

# ==============================================================================
# 1. Cottrell Equation (Chronoamperometry)
# ==============================================================================
t = np.linspace(0.05, 5.0, 200)  # Time after step [s]
i_cottrell = spa.cottrell(t, n=n, D=D, c_bulk=c_bulk, area=area)

# ==============================================================================
# 2. Levich Equation (Rotating Disk Electrode)
# ==============================================================================
rpm = np.linspace(100, 2500, 100)  # Rotation rate [rpm]
i_levich = spa.levich(rpm=rpm, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)

# ==============================================================================
# 3. Randles-Sevcik Equation (Cyclic Voltammetry Peak Current)
# ==============================================================================
v = np.linspace(0.01, 1.0, 100)  # Scan rate [V/s]
i_peak = spa.randles_sevcik(v, c_bulk=c_bulk, D=D, area=area, n=n, T=T)

# ==============================================================================
# 4. SECM Approach Curves (Conducting, Insulating, and Finite Kinetics)
# ==============================================================================
L = np.linspace(0.15, 5.0, 200)  # Normalized tip-substrate distance (d / a)
i_cond = spa.secm_approach_positive_feedback(L, RG=RG)
i_ins = spa.secm_approach_negative_feedback(L, RG=RG)
i_kin = spa.secm_approach_curve(L, RG=RG, kappa=kappa)

# ==============================================================================
# Print Summary & Plot
# ==============================================================================
print(f"Cottrell current at t = 1.0 s:    {spa.cottrell(1.0, n=n, D=D, c_bulk=c_bulk, area=area) * 1e6:.2f} uA")
print(f"Levich current at 1600 rpm:       {spa.levich(rpm=1600, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu) * 1e6:.2f} uA")
print(f"Randles-Sevcik peak at 0.1 V/s:   {spa.randles_sevcik(0.1, c_bulk=c_bulk, D=D, area=area, n=n, T=T) * 1e6:.2f} uA")
print(f"SECM finite-kinetics at L = 0.5:  {spa.secm_approach_curve(0.5, RG=RG, kappa=kappa):.3f} (normalized)")

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

axes[0, 0].plot(t, i_cottrell * 1e6)
axes[0, 0].set(title="Cottrell Transient", xlabel="Time / s", ylabel="Current / uA")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(rpm, i_levich * 1e6)
axes[0, 1].set(title="Levich Limiting Current", xlabel="Rotation Rate / rpm", ylabel="Limiting Current / uA")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(np.sqrt(v), i_peak * 1e6)
axes[1, 0].set(title="Randles-Sevcik Peak Current", xlabel=r"$v^{1/2}$ / $(V/s)^{1/2}$", ylabel="Peak Current / uA")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(L, i_cond, label="Conducting (pos. feedback)")
axes[1, 1].plot(L, i_kin, label=f"Kinetic (kappa={kappa})")
axes[1, 1].plot(L, i_ins, label="Insulating (neg. feedback)")
axes[1, 1].set(title=f"SECM Approach Curves (RG={RG})", xlabel="Normalized Distance L = d/a", ylabel="Normalized Current $I_T$")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
