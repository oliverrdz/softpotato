"""User Guide: Post-Processing & Electrochemical Diagnostics.

Demonstrates all diagnostic functions in softpotato.techniques on TechniqueResult objects:
1. find_peaks() — anodic/cathodic peak detection, parabolic refinement, delta_Ep, E_half
2. analyze_randles_sevcik() — multi-scan-rate D extraction
3. analyze_nicholson() — k0 extraction and Matsuda-Ayabe regime classification
4. analyze_cottrell() — chronoamperometric D extraction
5. analyze_levich() — RDE mass-transport D extraction
6. analyze_koutecky_levich() — RDE mixed kinetic/mass-transport I_K and D extraction
7. analyze_sand() — chronopotentiometric transition time tau and D extraction
8. analyze_tafel() — Tafel slope, alpha, exchange current I0, and k0 extraction
"""

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, NernstianEquilibrium, Species
from softpotato.techniques import (
    Chronoamperometry,
    Chronopotentiometry,
    CyclicVoltammetry,
    LinearSweepVoltammetry,
    analyze_cottrell,
    analyze_koutecky_levich,
    analyze_levich,
    analyze_nicholson,
    analyze_randles_sevcik,
    analyze_sand,
    analyze_tafel,
    find_peaks,
    simulate,
    simulate_rde_sweep,
    simulate_sweep,
)

# ==============================================================================
# 1. Physical System (SI units: m^2/s, mol/m^3 = mM, m^2, m/s)
# ==============================================================================
E0 = 0.0
D = 1.0e-9
c_bulk = 1.0
area = 7.07e-6
geometry = PlanarGeometry(area)

ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)

# ==============================================================================
# 2. Peak Detection (find_peaks) & Nicholson Analysis (analyze_nicholson)
# ==============================================================================
bv_quasi = ButlerVolmer(k0=5.0e-5, alpha=0.5, E0=E0, n=1)
cv = CyclicVoltammetry(
    E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=0.1, dE=0.002
)
cv_result = simulate(cv, [ox, red], geometry=geometry, kinetics=bv_quasi)

peaks = find_peaks(cv_result)
all_peaks = find_peaks(cv_result, return_all=True)
print("=== find_peaks() ===")
print(
    f"  Cathodic peak: E_pc = {peaks.cathodic_peak.E_peak:.4f} V, "
    f"I_pc = {peaks.cathodic_peak.I_peak * 1e6:.2f} uA"
)
print(
    f"  Anodic peak:   E_pa = {peaks.anodic_peak.E_peak:.4f} V, "
    f"I_pa = {peaks.anodic_peak.I_peak * 1e6:.2f} uA"
)
print(
    f"  delta_Ep = {peaks.delta_Ep * 1e3:.1f} mV, "
    f"E_half = {peaks.E_half:.4f} V, "
    f"|I_pa/I_pc| = {peaks.peak_current_ratio:.3f} "
    f"(total peaks found: {len(all_peaks)})"
)

nich = analyze_nicholson(cv_result, D=D, n=1)
print("\n=== analyze_nicholson() ===")
print(
    f"  delta_Ep = {nich.delta_Ep_mv:.1f} mV, Psi = {nich.psi:.3f}, "
    f"k0 = {nich.k0:.2e} m/s, Lambda = {nich.lambda_param:.2f}, regime = {nich.regime!r}"
)

# ==============================================================================
# 3. Multi-Scan-Rate Randles-Sevcik Analysis (analyze_randles_sevcik)
# ==============================================================================
nernst = NernstianEquilibrium(E0=E0, n=1)
scan_rates = [0.02, 0.05, 0.1, 0.2, 0.5]
cv_sweep = simulate_sweep(
    lambda v: CyclicVoltammetry(
        E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v, dE=0.004
    ),
    [ox, red],
    values=scan_rates,
    geometry=geometry,
    kinetics=nernst,
)
rs_fit = analyze_randles_sevcik(cv_sweep, n=1, area=area, c_bulk=c_bulk)
print("\n=== analyze_randles_sevcik() ===")
print(f"  D = {rs_fit.d:.3e} m^2/s (true: {D:.3e}), R^2 = {rs_fit.r_squared:.5f}")

# ==============================================================================
# 4. Cottrell Chronoamperometry Analysis (analyze_cottrell)
# ==============================================================================
ca = Chronoamperometry(E_step=-0.4, duration=2.0)
ca_result = simulate(ca, [ox, red], geometry=geometry, kinetics=nernst)
cott_fit = analyze_cottrell(ca_result, n=1, area=area, c_bulk=c_bulk, t_min=0.05)
print("\n=== analyze_cottrell() ===")
print(f"  D = {cott_fit.d:.3e} m^2/s (true: {D:.3e}), R^2 = {cott_fit.r_squared:.5f}")

# ==============================================================================
# 5. Levich & Koutecky-Levich RDE Analysis
# ==============================================================================
rpms = [400.0, 900.0, 1600.0, 2500.0]
rde_rev = simulate_rde_sweep(
    rpms=rpms,
    E_start=0.3,
    E_end=-0.4,
    scan_rate=0.05,
    D=D,
    c_bulk=c_bulk,
    area=area,
    kinetics=nernst,
    dE=0.005,
)
levich_fit = analyze_levich(rde_rev, n=1, area=area, c_bulk=c_bulk)
print("\n=== analyze_levich() ===")
print(
    f"  D = {levich_fit.d:.3e} m^2/s (true: {D:.3e}), "
    f"B = {levich_fit.levich_constant:.3e}, R^2 = {levich_fit.r_squared:.5f}"
)

rde_kin = simulate_rde_sweep(
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
kl_fit = analyze_koutecky_levich(
    rde_kin, potential=-0.15, n=1, area=area, c_bulk=c_bulk
)
print("\n=== analyze_koutecky_levich() ===")
print(
    f"  I_K = {kl_fit.i_k * 1e6:.2f} uA, k_rate = {kl_fit.k_rate:.2e} m/s, "
    f"D = {kl_fit.d:.3e} m^2/s, R^2 = {kl_fit.r_squared:.5f}"
)

# ==============================================================================
# 6. Sand Chronopotentiometry Analysis (analyze_sand)
# ==============================================================================
cp = Chronopotentiometry(current=-1.5e-5, duration=4.0)
cp_result = simulate(cp, [ox, red], geometry=geometry, kinetics=bv_quasi)
sand_diag = analyze_sand(cp_result, n=1, area=area, c_bulk=c_bulk, D=D)
print("\n=== analyze_sand() ===")
print(
    f"  tau = {sand_diag.tau:.4f} s (theoretical: {sand_diag.tau_theoretical:.4f} s), "
    f"D_est = {sand_diag.D_estimated:.3e} m^2/s, method = {sand_diag.detection_method!r}"
)

# ==============================================================================
# 7. Tafel Kinetic Analysis (analyze_tafel)
# ==============================================================================
bv_slow = ButlerVolmer(k0=1.0e-6, alpha=0.5, E0=E0, n=1)
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
print("\n=== analyze_tafel() ===")
print(
    f"  Tafel slope = {tafel_fit.slope_mv:.1f} mV/dec, alpha = {tafel_fit.alpha:.3f}, "
    f"I0 = {tafel_fit.i0:.2e} A, k0 = {tafel_fit.k0:.2e} m/s, "
    f"R^2 = {tafel_fit.r_squared:.5f}"
)
