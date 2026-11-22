"""User Guide: Declarative & Diagnostic Plotting Suite.

Demonstrates all plotting functions in softpotato.techniques on TechniqueResult objects:
1. plot_result() — technique-aware default and explicit kind dispatch
2. plot_voltammogram() — voltammograms with multi-cycle and peak overlay options
3. plot_peaks() — annotated anodic/cathodic peak diagnostics
4. plot_summary() — multi-panel summary figure
5. plot_sweep() — overlaying CV and RDE parameter sweeps
6. plot_randles_sevcik(), plot_tafel(), plot_levich(), plot_koutecky_levich()
7. plot_dpv(), plot_swv(), plot_scv() — pulse and sampled-current voltammograms
"""

import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, NernstianEquilibrium, Species
from softpotato.techniques import (
    CyclicVoltammetry,
    DifferentialPulseVoltammetry,
    LinearSweepVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
    analyze_koutecky_levich,
    analyze_levich,
    analyze_randles_sevcik,
    analyze_tafel,
    plot_dpv,
    plot_koutecky_levich,
    plot_levich,
    plot_peaks,
    plot_randles_sevcik,
    plot_result,
    plot_scv,
    plot_summary,
    plot_sweep,
    plot_swv,
    plot_tafel,
    plot_voltammogram,
    simulate,
    simulate_rde_sweep,
    simulate_sweep,
)

# ==============================================================================
# 1. Physical System (SI units)
# ==============================================================================
E0 = 0.0
D = 1.0e-9
c_bulk = 1.0
area = 7.07e-6
geometry = PlanarGeometry(area)

ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)
nernst = NernstianEquilibrium(E0=E0, n=1)
bv_quasi = ButlerVolmer(k0=5.0e-5, alpha=0.5, E0=E0, n=1)
bv_slow = ButlerVolmer(k0=1.0e-6, alpha=0.5, E0=E0, n=1)

# ==============================================================================
# 2. Core Single-Result & Summary Plots
# ==============================================================================
cv = CyclicVoltammetry(
    E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=0.1, n_cycles=2, dE=0.004
)
cv_result = simulate(cv, [ox, red], geometry=geometry, kinetics=bv_quasi)

fig1, ax1 = plot_result(cv_result)
ax1.set_title("plot_result() Default (CV)")

fig2, ax2 = plot_voltammogram(cv_result, cycle="all")
ax2.set_title("plot_voltammogram(cycle='all')")

fig3, ax3 = plot_peaks(cv_result)
fig_sum, axes_sum = plot_summary(cv_result)

# ==============================================================================
# 3. Parameter Sweeps & Diagnostic Regression Plots
# ==============================================================================
scan_rates = [0.02, 0.05, 0.1, 0.2]
cv_sweep = simulate_sweep(
    lambda v: CyclicVoltammetry(
        E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v, dE=0.005
    ),
    [ox, red],
    values=scan_rates,
    geometry=geometry,
    kinetics=nernst,
)
rs_fit = analyze_randles_sevcik(cv_sweep, n=1, area=area, c_bulk=c_bulk)

fig_swp, ax_swp = plot_sweep(cv_sweep, legend_title="Scan Rate")
fig_rs, ax_rs = plot_randles_sevcik(cv_sweep, rs_fit)

# Tafel plot on LSV
lsv = LinearSweepVoltammetry(E_start=0.0, E_end=-0.35, scan_rate=0.05, dE=0.002)
lsv_result = simulate(lsv, [ox, red], geometry=geometry, kinetics=bv_slow)
tafel_fit = analyze_tafel(
    lsv_result,
    E0=E0,
    overpotential_window=(-0.18, -0.07),
    branch="cathodic",
    n=1,
    area=area,
    c_bulk=c_bulk,
)
fig_taf, ax_taf = plot_tafel(
    lsv_result,
    tafel_fit,
    E0=E0,
    overpotential_window=(-0.18, -0.07),
    branch="cathodic",
)

# Levich & Koutecky-Levich plots on RDE sweeps
rpms = [400.0, 900.0, 1600.0, 2500.0]
rde_sweep = simulate_rde_sweep(
    rpms=rpms,
    E_start=0.3,
    E_end=-0.4,
    scan_rate=0.05,
    D=D,
    c_bulk=c_bulk,
    area=area,
    kinetics=bv_quasi,
    dE=0.005,
)
lev_fit = analyze_levich(rde_sweep, n=1, area=area, c_bulk=c_bulk)
kl_fit = analyze_koutecky_levich(
    rde_sweep, potential=-0.15, n=1, area=area, c_bulk=c_bulk
)

fig_lev, ax_lev = plot_levich(rde_sweep, lev_fit)
fig_kl, ax_kl = plot_koutecky_levich(rde_sweep, kl_fit, potential=-0.15)

# ==============================================================================
# 4. Pulse & Sampled-Current Voltammetry Plots
# ==============================================================================
dpv = DifferentialPulseVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    step_height=0.01,
    pulse_amplitude=-0.05,
    pulse_width=0.05,
    pulse_period=0.2,
)
dpv_result = simulate(dpv, [ox, red], geometry=geometry, kinetics=nernst)
fig_dpv, ax_dpv = plot_dpv(dpv_result, show_components=True)

swv = SquareWaveVoltammetry(
    E_start=0.3, E_end=-0.3, step_height=0.01, amplitude=0.025, frequency=15.0
)
swv_result = simulate(swv, [ox, red], geometry=geometry, kinetics=nernst)
fig_swv, ax_swv = plot_swv(swv_result, show_components=True)

scv = SampledCurrentVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    dE=0.05,
    duration=1.0,
    sample_times=[0.2, 0.5, 1.0],
)
scv_result = simulate(scv, [ox, red], geometry=geometry, kinetics=nernst)
fig_scv, ax_scv = plot_scv(scv_result)

plt.show()
