"""5-Minute Experimental Data Fitting: Cottrell, Randles-Sevcik, and SECM.

Demonstrates how to extract physical parameters (D, RG, d_offset) with R^2
and 95% confidence intervals using softpotato.analytical.fitting.
Replace the synthetic arrays below with your own experimental measurements.
"""

import matplotlib.pyplot as plt
import numpy as np

from softpotato import analytical as spa

rng = np.random.default_rng(42)

# ==============================================================================
# KNOWN EXPERIMENTAL CONSTANTS — Edit to match your cell
# ==============================================================================
n = 1  # Electrons transferred
c_bulk = 1.0e-6  # Bulk concentration [mol/cm^3] (1.0 mM)
area = 0.0707  # Macroelectrode area [cm^2]
T = 298.15  # Temperature [K]
a_ume = 12.5e-6  # SECM UME radius [m] (25 um diameter)

# ==============================================================================
# 1. Fit Cottrell Chronoamperometry -> Extract D
# ==============================================================================
t_exp = np.linspace(0.05, 4.0, 80)
i_true_ca = spa.cottrell(t_exp, n=n, D=6.7e-6, c_bulk=c_bulk, area=area)
i_exp_ca = i_true_ca + rng.normal(0.0, 2.5e-7, size=t_exp.size)

ca_fit = spa.fit_cottrell(
    t_exp,
    i_exp_ca,
    n=n,
    area=area,
    c_bulk=c_bulk,
    confidence_level=0.95,
)
ci_d_ca = ca_fit.confidence_intervals["D"]
print("=== Cottrell Fit ===")
print(
    f"  D      = {ca_fit.d:.3e} cm^2/s  (95% CI: [{ci_d_ca[0]:.3e}, {ci_d_ca[1]:.3e}])"
)
print(f"  R^2    = {ca_fit.r_squared:.5f}")

# ==============================================================================
# 2. Fit Randles-Sevcik Peak Currents -> Extract D
# ==============================================================================
v_exp = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ip_true = spa.randles_sevcik(v_exp, c_bulk=c_bulk, D=6.7e-6, area=area, n=n, T=T)
ip_exp = ip_true + rng.normal(0.0, 1.2e-6, size=v_exp.size)

rs_fit = spa.fit_randles_sevcik(
    v_exp,
    ip_exp,
    n=n,
    area=area,
    c_bulk=c_bulk,
    T=T,
    confidence_level=0.95,
)
ci_d_rs = rs_fit.confidence_intervals["D"]
print("\n=== Randles-Sevcik Fit ===")
print(
    f"  D      = {rs_fit.d:.3e} cm^2/s  (95% CI: [{ci_d_rs[0]:.3e}, {ci_d_rs[1]:.3e}])"
)
print(f"  R^2    = {rs_fit.r_squared:.5f}")

# ==============================================================================
# 3. Fit SECM Approach Curve -> Extract RG and d_offset
# ==============================================================================
d_exp = np.linspace(3.0e-6, 50.0e-6, 60)
L_true = (d_exp - 0.8e-6) / a_ume
it_true = spa.secm_approach_positive_feedback(L_true, RG=5.2)
it_exp = it_true + rng.normal(0.0, 0.03, size=d_exp.size)

secm_fit = spa.fit_secm_approach(
    d_exp,
    it_exp,
    substrate="conducting",
    a=a_ume,
    confidence_level=0.95,
)
ci_rg = secm_fit.confidence_intervals["rg"]
ci_d0 = secm_fit.confidence_intervals["d_offset"]
print("\n=== SECM Approach Fit ===")
print(f"  RG       = {secm_fit.rg:.2f}  (95% CI: [{ci_rg[0]:.2f}, {ci_rg[1]:.2f}])")
print(
    f"  d_offset = {secm_fit.d_offset * 1e6:.2f} um  (95% CI: [{ci_d0[0] * 1e6:.2f}, {ci_d0[1] * 1e6:.2f}] um)"
)
print(f"  R^2      = {secm_fit.r_squared:.5f}")

# ==============================================================================
# Plot Data vs. Fits
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].scatter(t_exp, i_exp_ca * 1e6, s=16, alpha=0.7, label="Data")
axes[0].plot(
    t_exp,
    ca_fit.y_fit * 1e6,
    color="C1",
    lw=2,
    label=f"Fit ($R^2={ca_fit.r_squared:.4f}$)",
)
axes[0].set(title="Cottrell Fit", xlabel="Time / s", ylabel="Current / uA")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(np.sqrt(v_exp), ip_exp * 1e6, s=28, alpha=0.8, label="Data")
axes[1].plot(
    np.sqrt(v_exp),
    rs_fit.y_fit * 1e6,
    color="C1",
    lw=2,
    label=f"Fit ($R^2={rs_fit.r_squared:.4f}$)",
)
axes[1].set(
    title="Randles-Sevcik Fit",
    xlabel=r"$v^{1/2}$ / $(V/s)^{1/2}$",
    ylabel="Peak Current / uA",
)
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].scatter(d_exp * 1e6, it_exp, s=16, alpha=0.7, label="Data")
axes[2].plot(
    d_exp * 1e6,
    secm_fit.y_fit,
    color="C1",
    lw=2,
    label=f"Fit ($R^2={secm_fit.r_squared:.4f}$)",
)
axes[2].set(
    title="SECM Approach Fit",
    xlabel="Stage Position d / um",
    ylabel="Normalized Current $I_T$",
)
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.tight_layout()
plt.show()
