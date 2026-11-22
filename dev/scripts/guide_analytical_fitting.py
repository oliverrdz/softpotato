"""User Guide: Curve Fitting & Parameter Estimation.

Demonstrates softpotato.analytical.fitting:
1. fit_cottrell() — extract D from chronoamperometry transients
2. fit_randles_sevcik() — extract D from CV peak current vs. scan rate
3. fit_secm_approach() — extract RG and d_offset from SECM approach curves
4. fit_curve() — general-purpose bounded nonlinear least-squares regression
"""

import numpy as np

from softpotato import analytical as spa

rng = np.random.default_rng(7)

n = 1
D_true = 1.0e-9  # m^2/s
c_bulk = 1.0  # mol/m^3
area = 7.07e-6  # m^2
T = 298.15  # K

# 1. fit_cottrell()
t_ca = np.linspace(0.05, 3.0, 60)
i_ca = spa.cottrell(t_ca, n=n, D=D_true, c_bulk=c_bulk, area=area) + rng.normal(
    0.0, 2.0e-7, size=t_ca.size
)
ca_fit = spa.fit_cottrell(t_ca, i_ca, n=n, area=area, c_bulk=c_bulk)
print(
    f"fit_cottrell:      D = {ca_fit.d:.3e} +/- {ca_fit.d_stderr:.2e} m^2/s, "
    f"R^2 = {ca_fit.r_squared:.5f}"
)

# 2. fit_randles_sevcik()
v_cv = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ip_cv = spa.randles_sevcik(v_cv, c_bulk=c_bulk, D=D_true, area=area, n=n, T=T)
ip_cv += rng.normal(0.0, 1.0e-6, size=v_cv.size)
rs_fit = spa.fit_randles_sevcik(v_cv, ip_cv, n=n, area=area, c_bulk=c_bulk, T=T)
print(
    f"fit_randles_sevcik: D = {rs_fit.d:.3e} +/- {rs_fit.d_stderr:.2e} m^2/s, "
    f"R^2 = {rs_fit.r_squared:.5f}"
)

# 3. fit_secm_approach()
a_ume = 12.5e-6
d_secm = np.linspace(3.0e-6, 45.0e-6, 50)
L_true = (d_secm - 0.6e-6) / a_ume
it_secm = spa.secm_approach_positive_feedback(L_true, RG=5.0) + rng.normal(
    0.0, 0.02, size=d_secm.size
)
secm_fit = spa.fit_secm_approach(d_secm, it_secm, substrate="conducting", a=a_ume)
print(
    f"fit_secm_approach: RG = {secm_fit.rg:.2f}, "
    f"d_offset = {secm_fit.d_offset * 1e6:.2f} um, R^2 = {secm_fit.r_squared:.5f}"
)


# 4. fit_curve() — custom bounded nonlinear model (exponential RC decay)
def rc_decay(t_arr: np.ndarray, i0: float, tau: float) -> np.ndarray:
    return i0 * np.exp(-t_arr / tau)


t_rc = np.linspace(0.0, 0.05, 80)
i_rc = rc_decay(t_rc, 5.0e-4, 0.012) + rng.normal(0.0, 5.0e-6, size=t_rc.size)
rc_fit = spa.fit_curve(
    rc_decay,
    t_rc,
    i_rc,
    p0=[4.0e-4, 0.01],
    bounds=([0.0, 1e-6], [1.0, 1.0]),
    param_names=["I0", "tau"],
)
print(
    f"fit_curve:         I0 = {rc_fit.params['I0'] * 1e6:.1f} uA, "
    f"tau = {rc_fit.params['tau'] * 1e3:.2f} ms, R^2 = {rc_fit.r_squared:.5f}"
)
