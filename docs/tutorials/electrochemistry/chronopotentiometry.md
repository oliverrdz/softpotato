# Simulating Chronopotentiometry & The Sand Equation

In **chronopotentiometry**, a constant current step $I_0$ is imposed on a working electrode, and the resulting potential transient $E(t)$ is monitored as electroactive species are depleted at the electrode surface.

When the surface concentration of the reactant drops to zero, the electrode can no longer support the prescribed current via that redox reaction alone, causing a sharp potential transition. The time at which this occurs is known as the **transition time** ($\tau$), governed by the classical **Sand equation**.

In this tutorial, you will learn how to:
1. Understand the theoretical foundations and the Sand equation.
2. Formulate 1D diffusion with constant-flux (Neumann) boundary conditions.
3. Solve the diffusion problem numerically using Python.
4. Benchmark the numerical surface concentration and transition time against analytical expressions from [`softpotato.analytical`][softpotato.analytical].
5. Analyze the potential transient $E(t)$ and extract the quarter-wave potential $E_{\tau/4}$.

---

## 1. Physical Model & The Sand Equation

Consider a semi-infinite planar electrode immersed in an unstirred solution containing species $\text{Ox}$ at bulk concentration $c^*$. At $t = 0$, a constant reduction current $I_0$ (cathodic current, $I_0 < 0$ by IUPAC convention) is applied:

$$\text{Ox} + n e^- \rightleftharpoons \text{Red}$$

### Boundary Conditions
Diffusion of species $\text{Ox}$ obeys Fick's second law:

$$\frac{\partial c(x, t)}{\partial t} = D \frac{\partial^2 c(x, t)}{\partial x^2}$$

Subject to:
- **Initial condition**: $c(x, 0) = c^*$
- **Bulk boundary condition**: $\lim_{x \to \infty} c(x, t) = c^*$
- **Surface flux condition (Neumann)**:
  $$\left. D \frac{\partial c}{\partial x} \right|_{x=0} = -\frac{j}{n F} = -\frac{I_0}{n F A}$$

### The Exact Surface Concentration
Integrating Fick's second law with this constant-flux boundary condition yields the surface concentration transient:

$$c(0, t) = c^* - \frac{2 I_0 \sqrt{t}}{n F A \sqrt{\pi D}}$$

### The Sand Transition Time
The transition time $\tau$ occurs when $c(0, \tau) = 0$:

$$\tau^{1/2} = \frac{n F A \sqrt{\pi D} c^*}{2 |I_0|}$$

Squaring both sides gives the celebrated **Sand equation**:

$$\tau = \frac{\pi D (n F A c^*)^2}{4 I_0^2}$$

---

## 2. Numerical Simulation with Python & Soft Potato

Let's set up a numerical simulation of chronopotentiometric diffusion using a spatial finite difference grid and solve the resulting system of ODEs.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from softpotato.analytical import (
    sand,
    sand_potential,
    sand_transition_time,
)

# 1. Physical Parameters
n = 1  # Number of electrons
F = 96485.332  # Faraday constant (C / mol)
A = 1e-4  # Electrode area (1 cm^2 in m^2)
D = 1e-9  # Diffusion coefficient (m^2 / s)
c_bulk = 1.0  # Bulk concentration (mol / m^3 = 1 mM)
I_app = -0.05  # Imposed cathodic current (A)

# Calculate theoretical transition time using Soft Potato
tau_analytical = sand_transition_time(
    current=abs(I_app),
    c_bulk=c_bulk,
    d=D,
    area=A,
    n=n,
)
print(f"Analytical Transition Time (tau): {tau_analytical:.4f} s")
```

### Discretizing the Spatial Domain

We discretize the diffusion layer $x \in [0, L_{\text{max}}]$ where $L_{\text{max}} \approx 6 \sqrt{D \tau}$:

```python
# Grid sizing
L_max = 6.0 * np.sqrt(D * tau_analytical)
Nx = 150
x = np.linspace(0, L_max, Nx)
dx = x[1] - x[0]

# Flux boundary term at the surface: D * dc/dx = - I / (n * F * A)
flux_surface = -I_app / (n * F * A)


def diffusion_ode(t, c):
    """Semi-discrete 1D diffusion ODE system with constant surface flux."""
    dcdt = np.zeros_like(c)

    # Interior nodes (central differences)
    dcdt[1:-1] = D * (c[2:] - 2 * c[1:-1] + c[:-2]) / (dx**2)

    # Surface node (x = 0): ghost-cell approach for Neumann condition
    # (c[1] - c[-1]) / (2 * dx) = flux_surface / D  ==>  c[-1] = c[1] - 2 * dx * flux_surface / D
    c_ghost = c[1] - 2 * dx * (flux_surface / D)
    dcdt[0] = D * (c[1] - 2 * c[0] + c_ghost) / (dx**2)

    # Bulk boundary (x = L_max): fixed bulk concentration
    dcdt[-1] = 0.0

    return dcdt


# Solve from t = 0 to 1.05 * tau
t_eval = np.linspace(0, 0.999 * tau_analytical, 200)
c_init = np.full(Nx, c_bulk)

sol = solve_ivp(
    diffusion_ode,
    (0, t_eval[-1]),
    c_init,
    t_eval=t_eval,
    method="BDF",
)
c_sim = sol.y  # Shape: (Nx, len(t_eval))
c_surface_sim = c_sim[0, :]
```

---

## 3. Benchmarking Against Analytical Theory

Now we compare the numerically simulated surface concentration against the exact analytical equation provided by `softpotato.analytical.sand`:

```python
# Compute analytical surface concentration
c_surface_exact = sand(
    time=t_eval,
    current=abs(I_app),
    c_bulk=c_bulk,
    d=D,
    area=A,
    n=n,
)

# Plot surface concentration depletion
plt.figure(figsize=(7, 4.5))
plt.plot(t_eval, c_surface_sim, "o", markevery=10, label="Numerical (BDF)")
plt.plot(t_eval, c_surface_exact, "-", label="Analytical Sand Equation")
plt.axvline(
    tau_analytical,
    color="crimson",
    linestyle="--",
    label=rf"$\tau = {tau_analytical:.3f}\,$s",
)

plt.xlabel("Time / s")
plt.ylabel("Surface Concentration / mol m$^{-3}$")
plt.title("Chronopotentiometry: Surface Concentration Depletion")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

!!! chip "Chip's Tip: Ghost Cells for Neumann Conditions"
    When implementing a constant-current (flux) boundary condition, using a **fictitious ghost cell** yields second-order spatial accuracy $\mathcal{O}(\Delta x^2)$ at the electrode surface! This prevents premature numerical depletion and ensures tight agreement with the Sand equation.

---

## 4. Potential Transient & The Quarter-Wave Potential

For a reversible electrochemical couple, the electrode potential varies with time according to the Karaoglanoff equation:

$$E(t) = E_{\tau/4} + \frac{R T}{n F} \ln\left( \frac{\tau^{1/2} - t^{1/2}}{t^{1/2}} \right)$$

Where $E_{\tau/4}$ is the **quarter-wave potential**, occurring at $t = \tau / 4$, which equals the formal potential $E^{0'}$ when diffusion coefficients of oxidized and reduced species are equal.

```python
E_formal = 0.200  # V vs ref
temperature = 298.15  # K

# Evaluate theoretical potential transient
E_transient = sand_potential(
    time=t_eval,
    tau=tau_analytical,
    e_formal=E_formal,
    temp_k=temperature,
    n=n,
)

plt.figure(figsize=(7, 4.5))
plt.plot(t_eval, E_transient, color="#FD8031", lw=2, label="E(t) Transient")
plt.axvline(
    tau_analytical / 4.0,
    color="navy",
    linestyle=":",
    label=rf"$t = \tau/4$ ($E = {E_formal:.3f}\,$V)",
)
plt.axvline(
    tau_analytical,
    color="crimson",
    linestyle="--",
    label=rf"$\tau = {tau_analytical:.3f}\,$s",
)

plt.xlabel("Time / s")
plt.ylabel("Electrode Potential $E$ / V")
plt.title("Chronopotentiogram: Reversible Potential Wave")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 5. Summary & Key Takeaways

- **Transition time $\tau$** scales inversely with current squared ($I_0^{-2}$), providing a robust experimental diagnostic for diffusion-controlled processes via the constant Sand product $I_0 \tau^{1/2}$.
- Soft Potato's [`softpotato.analytical.sand_transition_time`][softpotato.analytical] and [`softpotato.analytical.sand`][softpotato.analytical] allow instant validation of numerical simulation grids.
- At $t = \tau / 4$, the potential transient directly yields the formal redox potential $E^{0'}$.
