# Time-Stepping & Solvers

Once space is discretized onto a 1D mesh, the reaction–diffusion PDE becomes a coupled system of ordinary differential equations (ODEs) in time—one equation per species per interior grid node:

$$\frac{d\mathbf{c}}{dt} = \mathbf{L}\mathbf{c} + \mathbf{R}(\mathbf{c}, t) + \mathbf{b}(t)$$

where $\mathbf{L}$ is the discrete diffusion/advection operator, $\mathbf{R}$ is the homogeneous reaction term, and $\mathbf{b}(t)$ encodes the electrode and bulk boundary conditions. Soft Potato provides four built-in solvers in `softpotato.solver`.

---

## Built-In Solvers

### 1. Explicit Finite Difference (`ExplicitFiniteDifference`, `"explicit"` / `"ftcs"`)

Uses **Forward-Time Central-Space (FTCS)**: the future state $\mathbf{c}^{n+1}$ is computed directly from the current state $\mathbf{c}^n$:

$$\frac{\mathbf{c}^{n+1} - \mathbf{c}^n}{\Delta t} = \mathbf{L}\mathbf{c}^n + \mathbf{R}(\mathbf{c}^n, t_n)$$

- **Accuracy**: $\mathcal{O}(\Delta t, \Delta x^2)$.
- **Stability**: Conditionally stable. Diffusion requires the **CFL (Fourier) condition**:

  $$\text{Fo} = \frac{D \Delta t}{\Delta x_{\min}^2} \le \frac{1}{2}$$

  If $\Delta t$ violates this bound, numerical oscillations blow up exponentially. Soft Potato enforces this stability check automatically.

### 2. Implicit Finite Difference (`ImplicitFiniteDifference`, `"implicit"` / `"btcs"`)

Uses **Backward-Time Central-Space (BTCS)** (backward Euler), evaluating the right-hand side at the unknown next step $t_{n+1}$:

$$\frac{\mathbf{c}^{n+1} - \mathbf{c}^n}{\Delta t} = \mathbf{L}\mathbf{c}^{n+1} + \mathbf{R}(\mathbf{c}^{n+1}, t_{n+1})$$

- **Accuracy**: $\mathcal{O}(\Delta t, \Delta x^2)$ (first-order in time).
- **Stability**: **Unconditionally stable** and **L-stable**—it strongly damps high-frequency spatial noise, making it ideal right after sharp potential steps.

### 3. Crank–Nicolson (`CrankNicolson`, `"crank_nicolson"` / `"crank-nicolson"`)

Averages the explicit and implicit operators at $t_n$ and $t_{n+1}$ (trapezoidal rule in time):

$$\frac{\mathbf{c}^{n+1} - \mathbf{c}^n}{\Delta t} = \frac{1}{2}\left[\mathbf{L}\mathbf{c}^n + \mathbf{L}\mathbf{c}^{n+1}\right] + \dots$$

- **Accuracy**: $\mathcal{O}(\Delta t^2, \Delta x^2)$ (**second-order in both space and time**).
- **Stability**: Unconditionally stable (A-stable), though not L-stable: very large time steps across a discontinuous step change can introduce mild ringing near $x = 0$ unless $\Delta t$ is kept moderate.

### 4. Method of Lines (`ScipyIVPSolver`, `"scipy_ivp"` / `"solve_ivp"`)

Passes the semi-discrete ODE system to `scipy.integrate.solve_ivp` with adaptive time-stepping and automatic error control (`rtol`, `atol`).

- **Supported integrators**: Implicit stiff solvers (`"BDF"`, `"Radau"`) and explicit Runge–Kutta (`"RK45"`, `"DOP853"`).
- **Why it is the default in `simulate()`**: Electrochemical diffusion on stretched meshes is **stiff** (eigenvalues span many orders of magnitude between $\Delta x_0$ and $\Delta x_{\max}$), and homogeneous reactions add further stiffness. `"Radau"` / `"BDF"` automatically take tiny steps right after a potential jump and large steps once the profile smooths out, using sparse/analytical Jacobians from `ReactionNetwork` and `Mechanism`.

---

## Solver Comparison

| Solver | Registry Names | Time Order | Stability | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`ScipyIVPSolver`** | `"scipy_ivp"`, `"solve_ivp"` | Adaptive (up to 5th order) | Stiff-stable (`Radau` / `BDF`) | **Default choice** for CV, CA, pulse, RDE, and stiff `EC`/`EC'` mechanisms on expanding grids. |
| **`CrankNicolson`** | `"crank_nicolson"`, `"crank-nicolson"` | $\mathcal{O}(\Delta t^2)$ | Unconditionally stable | Fixed-step high-accuracy diffusion benchmarks and smooth waveforms. |
| **`ImplicitFiniteDifference`** | `"implicit"`, `"btcs"` | $\mathcal{O}(\Delta t)$ | Unconditionally stable (L-stable) | Step-edge robustness when damping high-frequency transients matters more than 2nd-order time accuracy. |
| **`ExplicitFiniteDifference`** | `"explicit"`, `"ftcs"` | $\mathcal{O}(\Delta t)$ | $\Delta t \le \frac{\Delta x_{\min}^2}{2 D}$ | Pedagogical comparisons and uniform grids with small $\Delta t$. |

---

## Step-Edge Events & Piecewise Integration

Electrochemical waveforms frequently contain **discontinuities** or **slope kinks**:

- **Potential steps** (`Chronoamperometry`, `SampledCurrentVoltammetry`, `DifferentialPulseVoltammetry`, `SquareWaveVoltammetry`): instantaneous jumps in $E(t)$.
- **Switching vertices** (`CyclicVoltammetry`): sudden reversal of $dE/dt$ from $-v$ to $+v$.
- **Multi-segment waveforms** (`CustomWaveform`): transitions between `ConstantSegment`, `LinearRampSegment`, etc.

If an adaptive ODE solver integrates blindly across a discontinuity, it either wastes steps hunting for the jump or smears the vertex. Every technique in `softpotato.techniques` therefore exposes its exact interior discontinuity timestamps via `technique.event_times`.

During `simulate()`, Soft Potato partitions $[0, t_{\text{end}}]$ at `event_times` into smooth sub-intervals $[t_k, t_{k+1}]$, integrating each sub-interval cleanly and restarting the solver at each event with the exact concentration state.

---

## Where to Go Next

- **User Guide**: [Techniques & Waveforms](../../user-guide/index.md) · [Pulse & Sampled-Current](../../user-guide/techniques/pulse.md) · [Custom Waveforms](../../user-guide/techniques/waveforms.md)
- **Tutorials**: [Cyclic Voltammetry Quickstart](../../tutorials/start/cv_quickstart.ipynb)
- **Numerics**: [Spatial Grids & Meshes](grids.md) · [Benchmarking & Validation](benchmarking.md)
