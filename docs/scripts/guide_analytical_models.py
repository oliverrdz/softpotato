"""User Guide: Analytical Models across softpotato.analytical Submodules.

Demonstrates closed-form equations in:
1. step: cottrell, anson, sand_transition_time
2. hydrodynamics: levich, koutecky_levich, collection_efficiency
3. kinetics: nernst, butler_volmer, tafel_slope
4. voltammetry: randles_sevcik, nicholson_psi, matsuda_ayabe
5. microelectrodes: microdisc_limiting_current, microdisc_transient
6. secm: secm_approach_positive_feedback, secm_approach_negative_feedback, secm_approach_curve
"""

import numpy as np

from softpotato.analytical import (
    hydrodynamics,
    kinetics,
    microelectrodes,
    secm,
    step,
    voltammetry,
)

# Consistent SI units
n = 1
D = 1.0e-9  # m^2/s
c_bulk = 1.0  # mol/m^3 (= 1 mM)
area = 7.07e-6  # m^2 (3 mm disk)
nu = 1.0e-6  # m^2/s (water)
T = 298.15  # K

# 1. Step techniques
t = np.array([0.1, 0.5, 1.0, 2.0])
i_cott = step.cottrell(t, n=n, D=D, c_bulk=c_bulk, area=area)
q_anson = step.anson(t, n=n, D=D, c_bulk=c_bulk, area=area)
tau_sand = step.sand_transition_time(I=4.0e-5, n=n, D=D, c_bulk=c_bulk, area=area)
print(
    f"Step: Cottrell I(1s) = {i_cott[2] * 1e6:.2f} uA, "
    f"Anson Q(1s) = {q_anson[2] * 1e6:.2f} uC, Sand tau = {tau_sand:.3f} s"
)

# 2. Hydrodynamics (RDE & RRDE)
i_lev = hydrodynamics.levich(rpm=1600, n=n, D=D, c_bulk=c_bulk, area=area, nu=nu)
i_kl = hydrodynamics.koutecky_levich(
    omega=hydrodynamics.rpm_to_rad_s(1600),
    n=n,
    D=D,
    c_bulk=c_bulk,
    area=area,
    nu=nu,
    i_k=5.0e-4,
)
n_coll = hydrodynamics.collection_efficiency(r1=2.5e-3, r2=2.75e-3, r3=3.75e-3)
print(
    f"Hydrodynamics: I_L(1600 rpm) = {i_lev * 1e6:.2f} uA, "
    f"I_KL = {i_kl * 1e6:.2f} uA, RRDE N = {n_coll:.3f}"
)

# 3. Interfacial Kinetics
e_eq = kinetics.nernst(E0=0.0, n=n, c_ox=1.0, c_red=0.1, T=T)
i_bv = kinetics.butler_volmer(eta=-0.1, i0=1.0e-5, alpha=0.5, n=n, T=T)
b_tafel = kinetics.tafel_slope(alpha=0.5, n=n, T=T)
print(
    f"Kinetics: E_eq = {e_eq:.4f} V, I_BV(-0.1 V) = {i_bv * 1e6:.2f} uA, "
    f"Tafel slope = {b_tafel * 1e3:.1f} mV/dec"
)

# 4. Voltammetry
i_rs = voltammetry.randles_sevcik(v=0.1, c_bulk=c_bulk, D=D, area=area, n=n, T=T)
nich = voltammetry.nicholson_psi(delta_ep=0.085, v=0.1, D=D, n=n, T=T, in_volts=True)
ma = voltammetry.matsuda_ayabe(k0=nich.k0, v=0.1, D=D, n=n, alpha=0.5, T=T)
print(
    f"Voltammetry: I_p = {i_rs * 1e6:.2f} uA, "
    f"Nicholson k0 = {nich.k0:.2e} m/s ({ma.classification})"
)

# 5. Microelectrodes (UME)
a_ume = 12.5e-6  # 25 um diameter disk
i_ss = microelectrodes.microdisc_limiting_current(radius=a_ume, n=n, D=D, c_bulk=c_bulk)
i_tr = microelectrodes.microdisc_transient(
    t=0.01, radius=a_ume, n=n, D=D, c_bulk=c_bulk
)
print(
    f"Microelectrodes: I_ss = {i_ss * 1e9:.2f} nA, "
    f"I_transient(10 ms) = {i_tr * 1e9:.2f} nA"
)

# 6. SECM Approach Curves
L = np.array([0.2, 0.5, 1.0, 2.0])
it_pos = secm.secm_approach_positive_feedback(L, RG=5.0)
it_neg = secm.secm_approach_negative_feedback(L, RG=5.0)
it_kin = secm.secm_approach_curve(L, RG=5.0, kappa=2.0)
print(
    f"SECM (L=0.5): I_T(+) = {it_pos[1]:.3f}, "
    f"I_T(kappa=2) = {it_kin[1]:.3f}, I_T(-) = {it_neg[1]:.3f}"
)
