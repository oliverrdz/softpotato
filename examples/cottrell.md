# Diffusion-Controlled Chronoamperometry with Soft Potato

This tutorial shows how to model a diffusion-controlled chronoamperometric experiment using Soft Potato and compare the implemented solvers against the analytical Cotterell/Cottrell solution.

It is written for electrochemists who want to understand:

- how a potential-step experiment becomes a PDE;
- why the boundary conditions are physically meaningful;
- how the faradaic current is obtained from concentration gradients;
- how different numerical solvers behave in practice.

---

## 1. The electrochemical model

Consider the reduction of a dissolved species \(O\) at a planar macroelectrode:

\[
O + ne^- \rightarrow R
\]

After a sufficiently negative potential step, the electrode reaction becomes diffusion-controlled. The oxidised species is consumed at the electrode surface, and the remaining concentration field evolves by diffusion.

The governing concentration equation is:

\[
\frac{\partial C}{\partial t}
=
D \frac{\partial^2 C}{\partial x^2}
\]

for \(x > 0\), where:

- \(C(x,t)\) = concentration of \(O\)
- \(D\) = diffusion coefficient
- \(x\) = distance from the electrode
- \(t\) = time

For a planar macroelectrode, the electrochemical meaning of the boundary conditions is:

- at the electrode surface, \(x=0\): \(C(0,t)=0\)
- far from the electrode, \(x \to \infty\): \(C(\infty,t)=C^*\)

The first condition means the species is consumed immediately at the electrode. The second means the bulk solution remains at the bulk concentration.

Because we cannot simulate an infinite domain numerically, we approximate the far boundary by a large but finite distance \(L\):

\[
C(L,t)=C^*
\]

This gives a finite-domain approximation of the semi-infinite diffusion problem.

---

## 2. Initial and boundary conditions

The initial condition is usually uniform bulk concentration:

\[
C(x,0)=C^*
\]

The finite-domain boundary-value problem is:

\[
\frac{\partial C}{\partial t}=D\frac{\partial^2 C}{\partial x^2},
\qquad 0<x<L
\]

with

\[
C(x,0)=C^*,
\qquad
C(0,t)=0,
\qquad
C(L,t)=C^*.
\]

This is a classic diffusion-controlled potential-step problem.

In electrochemical terms:

- the electrode boundary is a sink for the reactant;
- the far boundary represents the reservoir or bulk concentration.

---

## 3. Analytical solution and Cotterell/Cottrell equation

For a semi-infinite diffusion field, the analytical solution is:

\[
C(x,t)=C^* \,\mathrm{erf}\left(\frac{x}{2\sqrt{Dt}}\right)
\]

The flux to the electrode is obtained from Fick's first law:

\[
J(t) = -D\left.\frac{\partial C}{\partial x}\right|_{x=0}
\]

Evaluating the derivative at the electrode gives:

\[
J(t)=C^*\sqrt{\frac{D}{\pi t}}
\]

The faradaic current is then:

\[
I(t)=nFAJ(t)
\]

so that

\[
\boxed{
I(t)=nFA C^* \sqrt{\frac{D}{\pi t}}
}
\]

This is the Cotterell/Cottrell equation.

It is often written as:

\[
I(t)\propto t^{-1/2}
\]

This is the fundamental signature of diffusion-controlled chronoamperometry on a planar electrode.

---

## 4. Mass-transfer coefficient and its time dependence

The current can also be written in terms of a mass-transfer coefficient:

\[
J(t)=k_m(t)\, C^*
\]

Comparing with the analytical flux:

\[
k_m(t)=\sqrt{\frac{D}{\pi t}}
\]

Therefore,

\[
\boxed{
k_m(t)=\sqrt{\frac{D}{\pi t}}
}
\]

and

\[
I(t)=nFA\,k_m(t)\,C^*
\]

This coefficient is time dependent because the diffusion layer grows with time.

A rough estimate of the diffusion layer thickness is:

\[
\delta(t)\sim \sqrt{Dt}
\]

and therefore

\[
k_m(t)\sim \frac{D}{\delta(t)}
\]

This explains the familiar decay of the current with time: as the diffusion layer thickens, the concentration gradient at the electrode becomes smaller, so the flux falls as \(t^{-1/2}\).

---

## 5. Install and import Soft Potato

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[dev,docs]"
```

Then:

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

import softpotato as sp
from softpotato.solver import (
    DiffusionProblem,
    DirichletBC,
    get_solver,
    list_solvers,
)

print(f"Soft Potato version: {sp.__version__}")
print(f"Available solvers: {list_solvers()}")
```

---

## 6. Define the chronoamperometry problem

Use realistic electrochemical units:

```python
# Physical parameters
D = 1.0e-5          # cm^2 s^-1
C_bulk = 1.0e-6     # mol cm^-3
n = 1               # electrons transferred
F = 96485.33212     # C mol^-1
A = 0.0707          # electrode area in cm^2

# Domain approximation of infinite solution
L = 0.10            # cm
Nx = 201
x = np.linspace(0.0, L, Nx)

# Time domain
t_end = 5.0
t_eval = np.linspace(0.0, t_end, 251)

problem = DiffusionProblem(
    grid=x,
    diffusivity={"O": D},
    initial_conditions={"O": C_bulk},
    boundary_conditions={
        "O": (
            DirichletBC(0.0),       # electrode: reactive sink
            DirichletBC(C_bulk),    # far boundary: bulk concentration
        )
    },
)

print(f"dx = {problem.dx:.3e} cm")
print(f"Diffusion length at t_end: {np.sqrt(D*t_end):.3e} cm")
```

---

## 7. Analytical functions

```python
def analytical_concentration(x, t, D, C_bulk):
    return C_bulk * erf(x / (2.0 * np.sqrt(D * t)))


def analytical_flux(t, D, C_bulk):
    return C_bulk * np.sqrt(D / (np.pi * t))


def analytical_current(t, n, F, A, D, C_bulk):
    return n * F * A * analytical_flux(t, D, C_bulk)


def analytical_km(t, D):
    return np.sqrt(D / (np.pi * t))
```

---

## 8. First simulation: explain the electrochemistry

Let us solve one chronoamperogram with the adaptive `scipy_ivp` solver, which is a good general-purpose choice for diffusion problems.

```python
solver = get_solver(
    "scipy_ivp",
    method="Radau",
    rtol=1e-7,
    atol=1e-10,
)

start = time.perf_counter()
result = solver.solve(
    problem,
    t_span=(0.0, t_end),
    t_eval=t_eval,
)
elapsed = time.perf_counter() - start

print(f"Success: {result.success}")
print(f"Wall-clock time: {elapsed:.3f} s")
print(f"Number of time points: {len(result.t)}")
```

The solver returns the concentration profile and the flux at the electrode surface. Soft Potato computes the surface flux numerically from the concentration gradient:

\[
\left.\frac{\partial C}{\partial x}\right|_{x=0}
\approx
\frac{-3C_0 + 4C_1 - C_2}{2\Delta x}
\]

and then converts to current:

\[
I(t) = nFA \left(-D\frac{\partial C}{\partial x}\bigg|_{x=0}\right)
\]

```python
flux = result.fluxes["O"]
current = n * F * A * flux

positive_times = result.t > 0.0
exact_current = analytical_current(
    result.t[positive_times],
    n=n,
    F=F,
    A=A,
    D=D,
    C_bulk=C_bulk,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(
    result.t[positive_times],
    current[positive_times] * 1e6,
    label="Soft Potato (Radau)",
)
ax[0].plot(
    result.t[positive_times],
    exact_current * 1e6,
    "k--",
    label="Cotterell analytical",
)
ax[0].set_xlabel("Time / s")
ax[0].set_ylabel("Current / µA")
ax[0].set_title("Chronoamperogram")
ax[0].grid(alpha=0.3)
ax[0].legend()

# Concentration profile at final time
final_index = -1
profile = result["O"][final_index]

ax[1].plot(x, profile, label=f"t = {result.t[final_index]:.2f} s")
ax[1].plot(
    x,
    analytical_concentration(x, result.t[final_index], D, C_bulk),
    "k--",
    label="Analytical solution",
)
ax[1].set_xlabel("Distance from electrode / cm")
ax[1].set_ylabel("Concentration / mol cm$^{-3}$")
ax[1].set_title("Diffusion layer")
ax[1].grid(alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.show()
```

This plot shows the key electrochemical idea:

- the concentration drops to zero at the electrode;
- the bulk concentration is recovered far from the surface;
- the current is controlled by the slope at the interface.

---

## 9. Compare solver outputs

Now we simulate the same problem with each solver available in Soft Potato:

- `scipy_ivp`
- `crank_nicolson`
- `implicit`
- `explicit`

```python
solver_specs = {
    "SciPy IVP (Radau)": (
        "scipy_ivp",
        {"method": "Radau", "rtol": 1e-7, "atol": 1e-10},
    ),
    "Crank-Nicolson": (
        "crank_nicolson",
        {"dt": 0.01},
    ),
    "Implicit BTCS": (
        "implicit",
        {"dt": 0.01},
    ),
    "Explicit FTCS": (
        "explicit",
        {},
    ),
}

results = {}
timings = {}

for label, (name, options) in solver_specs.items():
    solver = get_solver(name, **options)
    start = time.perf_counter()
    results[label] = solver.solve(problem, t_span=(0.0, t_end), t_eval=t_eval)
    timings[label] = time.perf_counter() - start

    print(f"{label:24s} solved in {timings[label]:.3f} s")
```

Now compare the currents:

```python
fig, ax = plt.subplots(figsize=(8, 5))

for label, res in results.items():
    current_num = n * F * A * res.fluxes["O"]
    ax.plot(
        res.t[res.t > 0.0],
        current_num[res.t > 0.0] * 1e6,
        label=label,
    )

ax.plot(
    result.t[positive_times],
    exact_current * 1e6,
    "k--",
    linewidth=2,
    label="Analytical Cotterell",
)

ax.set_xlabel("Time / s")
ax.set_ylabel("Current / µA")
ax.set_title("Solver comparison for diffusion-controlled chronoamperometry")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

This makes the numerical differences visible.

---

## 10. Error analysis against the analytical solution

Let us compute metrics such as:

- maximum absolute current error
- root mean square error
- relative error

Define:

\[
e_I(t) = |I_{\mathrm{num}}(t)-I_{\mathrm{analytical}}(t)|
\]

and use:

\[
L_\infty = \max_t e_I(t)
\]

\[
\mathrm{RMSE}= \sqrt{\frac{1}{N}\sum_i e_I(t_i)^2}
\]

```python
error_summary = {}

for label, res in results.items():
    current_num = n * F * A * res.fluxes["O"]
    times = res.t[res.t > 0.0]
    current_vals = current_num[res.t > 0.0]

    exact_vals = analytical_current(
        times,
        n=n,
        F=F,
        A=A,
        D=D,
        C_bulk=C_bulk,
    )

    error = np.abs(current_vals - exact_vals)
    error_summary[label] = {
        "max_error": np.max(error),
        "rmse": np.sqrt(np.mean(error**2)),
        "relative_max_error": np.max(error / np.abs(exact_vals)),
    }

print(f"{'Solver':24s} {'Time (s)':>10s} {'Max error (A)':>16s} {'RMSE (A)':>12s}")
for label, summary in error_summary.items():
    print(
        f"{label:24s} "
        f"{timings[label]:10.3f} "
        f"{summary['max_error']:16.3e} "
        f"{summary['rmse']:12.3e}"
    )
```

We can also plot the relative error as a function of time:

```python
fig, ax = plt.subplots(figsize=(8, 5))

for label, res in results.items():
    current_num = n * F * A * res.fluxes["O"]
    times = res.t[res.t > 0.0]
    exact_vals = analytical_current(
        times,
        n=n,
        F=F,
        A=A,
        D=D,
        C_bulk=C_bulk,
    )

    relative_error = (current_num[res.t > 0.0] - exact_vals) / exact_vals
    ax.plot(times, relative_error, label=label)

ax.axhline(0.0, color="k", linestyle="--")
ax.set_xlabel("Time / s")
ax.set_ylabel(r"Relative current error, $(I-I_{exact})/I_{exact}$")
ax.set_title("Current error relative to analytical solution")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

## 11. Numerical stability and the CFL condition

The explicit finite-difference solver requires the Courant–Friedrichs–Lewy condition:

\[
\Delta t \le \frac{\Delta x^2}{2D}
\]

This is important because if \(\Delta t\) is too large, the explicit scheme becomes unstable.

For the current grid:

```python
dx = problem.dx
dt_cfl = dx**2 / (2.0 * D)
print(f"dx = {dx:.3e} cm")
print(f"Explicit CFL limit = {dt_cfl:.3e} s")
```

This is why explicit diffusion solves become expensive on fine grids. Halving \(\Delta x\) reduces the stable \(\Delta t\) by a factor of four.

---

## 12. Demonstrate a problematic time step

Let us deliberately violate the CFL criterion for the explicit solver:

```python
bad_dt = 1.1 * dt_cfl

try:
    bad_solver = get_solver("explicit", dt=bad_dt)
    bad_solver.solve(problem, t_span=(0.0, t_end), t_eval=t_eval)
    print("Unexpectedly succeeded")
except ValueError as err:
    print("Explicit solver rejected the time step:")
    print(err)
```

This is good practice: the numerical method should fail explicitly rather than silently produce a misleading current.

---

## 13. Time-step convergence study

To investigate convergence, vary the time step and compare the current at a fixed time.

```python
t_probe = 1.0
dt_values = np.array([0.08, 0.04, 0.02, 0.01, 0.005])

convergence_errors = {
    "Explicit FTCS": [],
    "Implicit BTCS": [],
    "Crank-Nicolson": [],
}

for dt in dt_values:
    for label, solver_name in [
        ("Explicit FTCS", "explicit"),
        ("Implicit BTCS", "implicit"),
        ("Crank-Nicolson", "crank_nicolson"),
    ]:
        try:
            solver = get_solver(solver_name, dt=dt)
            res = solver.solve(
                problem,
                t_span=(0.0, t_probe),
                t_eval=np.array([0.0, t_probe]),
            )

            I_num = n * F * A * res.fluxes["O"][-1]
            I_exact = analytical_current(
                t_probe,
                n=n,
                F=F,
                A=A,
                D=D,
                C_bulk=C_bulk,
            )

            convergence_errors[label].append(abs(I_num - I_exact))

        except ValueError:
            convergence_errors[label].append(np.nan)
```

Plot the convergence:

```python
fig, ax = plt.subplots(figsize=(8, 5))

for label, errors in convergence_errors.items():
    ax.loglog(dt_values, errors, "o-", label=label)

ax.axvline(dt_cfl, color="k", linestyle="--", label="Explicit CFL limit")
ax.set_xlabel(r"Time step, $\Delta t$ / s")
ax.set_ylabel("Absolute current error / A")
ax.set_title("Time-step convergence")
ax.invert_xaxis()
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
```

The expected behaviour is:

- explicit FTCS: first-order in time, stable only when \(\Delta t\) satisfies the CFL condition
- implicit BTCS: first-order in time, unconditionally stable
- Crank–Nicolson: second-order in time, often more accurate but can oscillate with large \(\Delta t\)
- SciPy adaptive solver: near-optimal accuracy with error tolerances rather than fixed \(\Delta t\)

---

## 14. Summary of numerical behaviour

| Solver | Stability | Accuracy | Typical behaviour |
|---|---|---|---|
| `scipy_ivp` | Good for stiff diffusion | High, controlled by tolerance | Robust and efficient for many problems |
| `crank_nicolson` | Unconditionally stable for linear diffusion | Second-order in time | Accurate but may oscillate for large steps |
| `implicit` | Unconditionally stable | First-order in time | Robust and damped |
| `explicit` | Conditionally stable | First-order in time | Simple but can be computationally expensive |

---

## 15. Modelling interpretation for electrochemists

This tutorial illustrates a key point in electrochemical modelling:

- the measured current is not just a concentration value;
- it is the concentration gradient at the electrode surface;
- therefore, current is sensitive to numerical differentiation and boundary treatment;
- the boundary conditions encode the chemistry and transport physics.

For diffusion-controlled chronoamperometry, the boundary conditions are the physical core of the problem:

- \(C(0,t)=0\): reactant is consumed at the electrode
- \(C(L,t)=C^*\): bulk solution acts as a reservoir

The diffusion-mass-transfer coefficient

\[
k_m(t)=\sqrt{\frac{D}{\pi t}}
\]

captures the same physics as the Cottrell current, and it shows directly that transport is time dependent.

---

## 16. Final takeaway

The diffusion-controlled macroelectrode chronoamperogram is governed by the same PDE that underpins many electrochemical mass-transport models:

\[
\frac{\partial C}{\partial t}=D\frac{\partial^2 C}{\partial x^2}
\]

The analytical current is:

\[
\boxed{
I(t)=nFA C^* \sqrt{\frac{D}{\pi t}}
}
\]

which is the Cotterell/Cottrell equation.

Comparing the numerical solutions from different Soft Potato solvers shows that:

- the physical model is the same;
- the numerical method matters;
- stability, accuracy, convergence, and runtime must all be considered;
- a good electrochemical simulation is not only physically realistic, but also numerically reliable.

This is the key lesson: electrochemical models are not just equations; they are PDEs solved under specific physical assumptions, and the solver choice affects the interpretation of the experiment.