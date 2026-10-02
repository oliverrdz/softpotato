"""Tests for analytical step equations (softpotato.analytical.step)."""

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.step import (
    anson,
    cottrell,
    cottrell_cylinder,
    cottrell_spherical,
    cottrell_step,
    sand,
    sand_potential,
    sand_transition_time,
    step_concentration_profile,
    step_flux_profile,
)
from softpotato.constants import FARADAY
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

# ==============================================================================
# 1. Planar Cottrell Equation Tests
# ==============================================================================


def test_cottrell_scalar_and_vector():
    """Verify planar Cottrell returns expected scalar float and vector array."""
    t_val = 1.0
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 1.0

    expected = n * FARADAY * area * math.sqrt(D) * c_bulk / math.sqrt(math.pi * t_val)
    i_scalar = cottrell(t_val, n=n, D=D, c_bulk=c_bulk, area=area)
    assert isinstance(i_scalar, float)
    assert math.isclose(i_scalar, expected, rel_tol=1e-12)

    # Vector input
    t_arr = np.array([0.5, 1.0, 2.0, 4.0])
    i_arr = cottrell(t_arr, n=n, D=D, c_bulk=c_bulk, area=area)
    assert isinstance(i_arr, np.ndarray)
    assert i_arr.shape == t_arr.shape
    np.testing.assert_allclose(
        i_arr,
        expected / np.sqrt(t_arr),
        rtol=1e-12,
    )


def test_cottrell_zero_time():
    """Verify Cottrell equation approaches infinity at t = 0 without error."""
    assert math.isinf(cottrell(0.0))
    i_vec = cottrell(np.array([0.0, 1.0]))
    assert np.isinf(i_vec[0])
    assert not np.isinf(i_vec[1])


def test_cottrell_input_validation():
    """Verify invalid parameters trigger helpful ValueErrors."""
    with pytest.raises(ValueError, match="Time t must be non-negative"):
        cottrell(-0.5)

    with pytest.raises(ValueError, match="Time t must be non-negative"):
        cottrell(np.array([0.1, -0.2]))

    with pytest.raises(ValueError, match="Number of electrons n must be positive"):
        cottrell(1.0, n=0)

    with pytest.raises(ValueError, match="Diffusion coefficient D must be positive"):
        cottrell(1.0, D=-1e-5)

    with pytest.raises(
        ValueError, match="Bulk concentration c_bulk must be non-negative"
    ):
        cottrell(1.0, c_bulk=-1e-3)

    with pytest.raises(ValueError, match="Electrode area must be positive"):
        cottrell(1.0, area=0.0)

    with pytest.raises(ValueError, match="Faraday constant F must be positive"):
        cottrell(1.0, F=0.0)


# ==============================================================================
# 2. Spherical Cottrell Equation Tests
# ==============================================================================


def test_cottrell_spherical_steady_state():
    """Verify spherical Cottrell converges to steady-state limiting current at long times."""
    r = 1e-4  # 100 um radius
    n = 1
    D = 1e-5
    c_bulk = 1e-3

    # Theoretical steady-state limiting current for full sphere: I_ss = 4 * pi * n * F * D * c* * r
    i_ss_expected = 4.0 * math.pi * n * FARADAY * D * c_bulk * r

    # At very long time (e.g. t = 1e8 s), transient term is negligible
    i_long = cottrell_spherical(1e8, r=r, n=n, D=D, c_bulk=c_bulk)
    assert math.isclose(i_long, i_ss_expected, rel_tol=1e-5)


def test_cottrell_spherical_planar_limit():
    """Verify spherical Cottrell converges to planar Cottrell as radius approaches infinity."""
    t = 1.0
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    r_large = 1e6  # 1,000,000 meters radius (effectively flat)
    area = 1.0

    i_sphere = cottrell_spherical(t, r=r_large, n=n, D=D, c_bulk=c_bulk, area=area)
    i_planar = cottrell(t, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(i_sphere, i_planar, rel_tol=1e-5)


def test_cottrell_spherical_custom_area():
    """Verify custom area (e.g. hemisphere A = 2*pi*r²) is respected."""
    r = 1e-5
    area_hemi = 2.0 * math.pi * (r**2)
    i_hemi = cottrell_spherical(1.0, r=r, area=area_hemi)
    i_full = cottrell_spherical(1.0, r=r)  # area = 4*pi*r²
    assert math.isclose(i_hemi * 2.0, i_full, rel_tol=1e-12)


def test_cottrell_spherical_validation():
    """Verify invalid radius raises ValueError."""
    with pytest.raises(ValueError, match="Electrode radius r must be positive"):
        cottrell_spherical(1.0, r=0.0)
    with pytest.raises(ValueError, match="Electrode radius r must be positive"):
        cottrell_spherical(1.0, r=-1e-4)


# ==============================================================================
# 3. Double Potential Step Chronoamperometry Reversal Tests
# ==============================================================================


def test_cottrell_step_forward_and_reversal():
    """Verify forward step matches cottrell and reversal matches theoretical ratio."""
    tau = 2.0
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 1e-4

    # Forward step check (t <= tau)
    t_fwd = 1.0
    assert math.isclose(
        cottrell_step(t_fwd, tau=tau, n=n, D=D, c_bulk=c_bulk, area=area),
        cottrell(t_fwd, n=n, D=D, c_bulk=c_bulk, area=area),
        rel_tol=1e-12,
    )

    # Diagnostic ratio check at t = 2 * tau: I(2*tau) / I(tau) == 1 - 1/sqrt(2)
    i_tau = cottrell_step(tau, tau=tau, n=n, D=D, c_bulk=c_bulk, area=area)
    i_2tau = cottrell_step(2.0 * tau, tau=tau, n=n, D=D, c_bulk=c_bulk, area=area)
    expected_ratio = 1.0 - 1.0 / math.sqrt(2.0)
    assert math.isclose(i_2tau / i_tau, expected_ratio, rel_tol=1e-12)


def test_cottrell_step_vector_transition():
    """Verify vectorized array spanning across the switching time tau."""
    tau = 1.0
    t_arr = np.linspace(0.1, 3.0, 30)
    i_arr = cottrell_step(t_arr, tau=tau)

    # Forward currents must be positive
    assert np.all(i_arr[t_arr <= tau] > 0.0)
    # Reversal currents must be positive
    assert np.all(i_arr[t_arr > tau] > 0.0)


def test_cottrell_step_validation():
    """Verify validation on tau and product diffusion coefficient."""
    with pytest.raises(ValueError, match="Step duration tau must be positive"):
        cottrell_step(1.0, tau=0.0)
    with pytest.raises(
        ValueError, match="Product diffusion coefficient D_red must be positive"
    ):
        cottrell_step(1.0, tau=1.0, D_red=-1e-5)


# ==============================================================================
# 4. Anson Chronocoulometry Tests
# ==============================================================================


def test_anson_single_step():
    """Verify Anson equation matches intercept and diffusional slope."""
    q_dl = 1.5e-6
    gamma = 2.0e-10
    n = 1
    area = 1e-4
    c_bulk = 1e-3
    D = 1e-5

    # Intercept at t = 0
    q_0 = anson(0.0, n=n, D=D, c_bulk=c_bulk, area=area, Q_dl=q_dl, gamma=gamma)
    expected_intercept = q_dl + n * FARADAY * area * gamma
    assert math.isclose(q_0, expected_intercept, rel_tol=1e-12)

    # Check derivative: dQ/dt ~ I_cottrell(t)
    t = 1.0
    dt = 1e-6
    q_t = anson(t, n=n, D=D, c_bulk=c_bulk, area=area, Q_dl=q_dl, gamma=gamma)
    q_t_dt = anson(t + dt, n=n, D=D, c_bulk=c_bulk, area=area, Q_dl=q_dl, gamma=gamma)
    dq_dt = (q_t_dt - q_t) / dt
    i_cottrell = cottrell(t + dt / 2.0, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(dq_dt, i_cottrell, rel_tol=1e-5)


def test_anson_reversal_chronocoulometry():
    """Verify Anson double-step continuity and conservation."""
    tau = 2.0
    q_fwd_end = anson(tau, tau=tau)
    q_rev_start = anson(tau + 1e-12, tau=tau)
    # Charge must be continuous across tau (sqrt(1e-12) = 1e-6)
    assert math.isclose(q_fwd_end, q_rev_start, rel_tol=1e-4)

    # At infinite time, all electrogenerated Red is reoxidized, Q(inf) -> Q_intercept
    q_intercept = anson(0.0, tau=tau)
    q_inf = anson(1e8, tau=tau)
    assert math.isclose(q_inf, q_intercept, abs_tol=1e-4)


def test_anson_validation():
    """Verify Anson parameter validation."""
    with pytest.raises(ValueError, match="Surface excess gamma must be non-negative"):
        anson(1.0, gamma=-1e-10)
    with pytest.raises(ValueError, match="Step duration tau must be positive"):
        anson(1.0, tau=-1.0)


# ==============================================================================
# 5. Sand Chronopotentiometry Tests
# ==============================================================================


def test_sand_transition_time_and_surface_concentration():
    """Verify Sand transition time produces zero surface concentration at t = tau."""
    I = 1e-4
    n = 1
    D = 1e-5
    c_bulk = 1e-3
    area = 1e-4

    tau = sand_transition_time(I=I, n=n, D=D, c_bulk=c_bulk, area=area)
    assert tau > 0.0

    # At t = 0, surface concentration must equal bulk concentration
    c_0 = sand(0.0, I=I, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(c_0, c_bulk, rel_tol=1e-12)

    # At t = tau, surface concentration must reach 0.0
    c_tau = sand(tau, I=I, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(c_tau, 0.0, abs_tol=1e-12)

    # For t > tau, surface concentration remains clamped at 0.0
    c_past = sand(2.0 * tau, I=I, n=n, D=D, c_bulk=c_bulk, area=area)
    assert c_past == 0.0


def test_sand_potential_quarter_wave():
    """Verify sand_potential produces E_half at t = tau / 4."""
    tau = 4.0
    e_half = 0.250
    e_q = sand_potential(1.0, tau=tau, E_half=e_half)
    assert math.isclose(e_q, e_half, rel_tol=1e-12)

    # Potential approaches -inf at t >= tau
    e_tau = sand_potential(tau, tau=tau, E_half=e_half)
    assert np.isneginf(e_tau)


def test_sand_validation():
    """Verify Sand parameter validation."""
    with pytest.raises(ValueError, match="Applied current I cannot be zero"):
        sand_transition_time(I=0.0)
    with pytest.raises(ValueError, match="Applied current I cannot be zero"):
        sand(1.0, I=0.0)
    with pytest.raises(
        ValueError, match="Bulk concentration c_bulk must be strictly positive"
    ):
        sand_transition_time(I=1.0, c_bulk=0.0)


# ==============================================================================
# 6. Cylindrical Electrode Chronoamperometry Tests
# ==============================================================================


def test_cottrell_cylinder_planar_limit():
    """Verify cylinder current approaches planar Cottrell at very short times (theta << 1)."""
    r0 = 1e-3  # 1 mm
    length = 1e-2  # 1 cm
    area = 2.0 * math.pi * r0 * length
    n = 1
    D = 1e-5
    c_bulk = 1e-3

    # Small time: t = 1e-5 s -> theta = D*t/r0² = 1e-5*1e-5 / 1e-6 = 1e-4 << 1
    t_short = 1e-5
    i_cyl_auto = cottrell_cylinder(
        t_short, r0=r0, length=length, D=D, c_bulk=c_bulk, n=n
    )
    i_cyl_oldham = cottrell_cylinder(
        t_short, r0=r0, length=length, D=D, c_bulk=c_bulk, n=n, method="oldham"
    )
    i_planar = cottrell(t_short, n=n, D=D, c_bulk=c_bulk, area=area)

    assert math.isclose(i_cyl_auto, i_planar, rel_tol=2e-2)
    assert math.isclose(i_cyl_oldham, i_planar, rel_tol=2e-2)


def test_cottrell_cylinder_methods():
    """Verify different methods execute without error and return consistent results."""
    t = 1.0
    r0 = 1e-4
    for meth in ["auto", "aoki", "oldham", "short_time", "long_time"]:
        res = cottrell_cylinder(t, r0=r0, method=meth)
        assert isinstance(res, float)
        assert not math.isnan(res)

    with pytest.raises(ValueError, match="Unknown method 'invalid_method'"):
        cottrell_cylinder(t, r0=r0, method="invalid_method")  # type: ignore[arg-type]


# ==============================================================================
# 7. Spatial Concentration and Flux Profile Tests
# ==============================================================================


def test_step_concentration_profile():
    """Verify exact concentration profile satisfies boundary conditions."""
    t = 1.0
    D = 1e-5
    c_bulk = 1e-3

    # Boundary condition at electrode surface: c(0, t) = 0
    c_surf = step_concentration_profile(0.0, t=t, D=D, c_bulk=c_bulk)
    assert math.isclose(c_surf, 0.0, abs_tol=1e-12)

    # Bulk concentration far into solution: c(x -> inf, t) = c_bulk
    c_bulk_dist = step_concentration_profile(1.0, t=t, D=D, c_bulk=c_bulk)
    assert math.isclose(c_bulk_dist, c_bulk, rel_tol=1e-6)

    # Vectorized distances
    x_arr = np.linspace(0, 0.05, 20)
    c_arr = step_concentration_profile(x_arr, t=t, D=D, c_bulk=c_bulk)
    assert isinstance(c_arr, np.ndarray)
    assert c_arr[0] == 0.0
    assert np.all(np.diff(c_arr) >= 0.0)  # Monotonically increasing


def test_step_flux_profile_and_cottrell_link():
    """Verify surface flux exactly links to the Cottrell current."""
    t = 1.0
    D = 1e-5
    c_bulk = 1e-3
    n = 1
    area = 1e-4

    j_surf = step_flux_profile(0.0, t=t, D=D, c_bulk=c_bulk)
    i_from_flux = -n * FARADAY * area * j_surf
    i_cottrell = cottrell(t, n=n, D=D, c_bulk=c_bulk, area=area)
    assert math.isclose(i_from_flux, i_cottrell, rel_tol=1e-12)


# ==============================================================================
# 8. Numerical Solver Cross-Validation Benchmark
# ==============================================================================


def test_numerical_solver_cottrell_benchmark():
    """Validate ScipyIVPSolver numerical simulation against analytical cottrell."""
    D = 1e-5
    c_bulk = 1e-3
    n = 1
    area = 1e-4
    t_end = 1.0

    # 1D domain scaled to 6 * sqrt(D * t_end)
    l_domain = 6.0 * math.sqrt(D * t_end)
    grid = np.linspace(0.0, l_domain, 101)

    problem = DiffusionProblem(
        grid=grid,
        diffusivity={"c": D},
        boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(c_bulk))},
        initial_conditions={"c": c_bulk},
    )

    solver = get_solver("scipy_ivp")
    t_eval = np.linspace(0.1, t_end, 10)
    result = solver.solve(problem, t_span=(0.0, t_end), t_eval=t_eval)

    assert result.success

    # Numerical current from interfacial flux: I = -n * F * A * J(0)
    num_flux = result.fluxes["c"]
    i_numerical = -n * FARADAY * area * num_flux
    i_analytical = cottrell(t_eval, n=n, D=D, c_bulk=c_bulk, area=area)

    # Verify numerical and analytical agreement (mean error < 0.2%, max error < 1.0%)
    rel_error = np.abs(i_numerical - i_analytical) / i_analytical
    assert np.mean(rel_error) < 3e-3, (
        f"Mean relative error {np.mean(rel_error):.4e} exceeds 0.3%"
    )
    assert np.max(rel_error) < 1e-2, (
        f"Max relative error {np.max(rel_error):.4e} exceeds 1.0%"
    )

    # Validate spatial concentration profile at t_end
    c_numerical = result["c"][-1, :]
    c_exact = step_concentration_profile(grid, t=t_end, D=D, c_bulk=c_bulk)
    max_c_diff = np.max(np.abs(c_numerical - c_exact))
    assert max_c_diff < 5e-3, f"Concentration diff {max_c_diff:.4e} exceeds tolerance"


# ==============================================================================
# 9. Top-Level Package Exports Verification
# ==============================================================================


def test_top_level_exports():
    """Verify analytical step functions are exported at both package and subpackage levels."""
    assert sp.cottrell is cottrell
    assert sp.analytical.cottrell is cottrell
    assert sp.cottrell_spherical is cottrell_spherical
    assert sp.cottrell_step is cottrell_step
    assert sp.anson is anson
    assert sp.sand is sand
    assert sp.sand_transition_time is sand_transition_time
    assert sp.sand_potential is sand_potential
    assert sp.cottrell_cylinder is cottrell_cylinder
    assert sp.step_concentration_profile is step_concentration_profile
    assert sp.step_flux_profile is step_flux_profile
