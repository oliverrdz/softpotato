# Techniques Module (`softpotato.techniques`)

The `softpotato.techniques` subpackage provides high-level electrochemical experimental protocols, excitation waveform generators, an automated simulation pipeline ([`simulate`][softpotato.techniques.simulate], [`simulate_sweep`][softpotato.techniques.simulate_sweep]), a structured result container ([`TechniqueResult`][softpotato.techniques.TechniqueResult]), a declarative Matplotlib plotting suite, and automated electrochemical diagnostics.

---

## Theoretical Background

### 1. Potentiostatic vs. Galvanostatic Excitation & Event-Aware Integration

Every experimental technique in `softpotato.techniques` subclasses [`BaseTechnique`][softpotato.techniques.BaseTechnique], encapsulating a time-dependent excitation signal over $t \in [0, t_{\max}]$ under either:

- **Potentiostatic control** (`mode="potentiostatic"`): prescribes the electrode potential $E(t)$ at the working electrode boundary, driving interfacial Nernstian, Butler–Volmer, or Tafel charge transfer.
- **Galvanostatic control** (`mode="galvanostatic"`): prescribes the net Faradaic current $I(t)$ (e.g. [`Chronopotentiometry`][softpotato.techniques.Chronopotentiometry]), enforcing a Neumann interfacial flux $J(t) = I(t) / (n F A)$ and recovering the electrode potential $E(t)$ from the resulting surface concentrations.

Techniques with sharp step edges or switching vertices (CA, CV, DPV, SWV, SCV, CP, `CustomWaveform`) expose exact interior timestamps via `event_times`. [`simulate`][softpotato.techniques.simulate] performs piecewise event-aware integration across each smooth sub-interval, preventing adaptive ODE solvers (`Radau`, `BDF`) from stepping over narrow pulses or switching vertices.

### 2. Sampled-Current Voltammetry (SCV)

[`SampledCurrentVoltammetry`][softpotato.techniques.SampledCurrentVoltammetry] models a family of $K$ independent constant-potential step experiments $\{E_1, E_2, \dots, E_K\}$, each held for identical duration $t_{\max}$ (with optional common quiet time $t_{\text{quiet}}$ at $E_{\text{quiet}}$). Sampling the resulting chronoamperometric transients $I_k(t)$ at one or more fixed sampling times $\tau \in (0, t_{\max}]$ yields a **sampled-current voltammogram** $I(\tau)$ vs. $E_k$ free of charging-current interference at sufficiently long $\tau$:

$$I_k(\tau) = \text{Cottrell / Butler–Volmer response at potential } E_k \text{ evaluated at } t = \tau$$

### 3. Pulse Voltammetry (DPV & SWV)

- **Differential Pulse Voltammetry** ([`DifferentialPulseVoltammetry`][softpotato.techniques.DifferentialPulseVoltammetry]): superimposes a constant-amplitude pulse $\Delta E_p$ onto a staircase ramp of step height $\Delta E_s$ and period $t_{\text{period}}$. Current is sampled just before pulse application ($I_{\text{pre}}$) and at the end of the pulse ($I_{\text{pulse}}$), yielding the differential peak-shaped signal $\Delta I = I_{\text{pulse}} - I_{\text{pre}}$ vs. $E_{\text{base}}$.
- **Square Wave Voltammetry** ([`SquareWaveVoltammetry`][softpotato.techniques.SquareWaveVoltammetry]): applies a symmetric square wave of amplitude $E_{\text{sw}}$ and frequency $f$ ($T = 1/f$) superimposed on a staircase ramp $\Delta E_s$. Sampling at the end of each forward and reverse half-cycle produces $I_{\text{forward}}$, $I_{\text{reverse}}$, and the net signal $I_{\text{net}} = I_{\text{forward}} - I_{\text{reverse}}$.

### 4. Rotating Disk Electrode (RDE) Hydrodynamics

[`RotatingDiskElectrode`][softpotato.techniques.RotatingDiskElectrode] couples any potentiostatic waveform with convective-diffusion mass transport at angular velocity $\omega$ ($\text{rad}\,\text{s}^{-1}$) or rotational speed $\text{rpm}$ in an electrolyte of kinematic viscosity $\nu$:

- **`"nernst_layer"`**: truncates the spatial domain at the steady-state Nernst diffusion layer thickness $\delta_N = 1.61\, D^{1/3} \omega^{-1/2} \nu^{1/6}$ with bulk Dirichlet conditions at $x = \delta_N$.
- **`"levich_convection"`**: solves the full 1D convective-diffusion PDE $\partial c_i / \partial t = D_i \partial^2 c_i / \partial x^2 - v_x(x) \partial c_i / \partial x + R_i$ using the Cochran axial velocity profile $v_x(x) = -0.510\, \omega^{3/2} \nu^{-1/2} x^2$ via [`RDEConvectionTerm`][softpotato.techniques.RDEConvectionTerm].

---

## API Overview

### Waveforms & Experimental Techniques

| Class | Mode | Key Parameters | Description |
| :--- | :--- | :--- | :--- |
| [`BaseTechnique`][softpotato.techniques.BaseTechnique] | Both | `mode`, `duration`, `event_times`, `plot_specs` | Abstract base class for all electrochemical excitation protocols. |
| [`Chronoamperometry`][softpotato.techniques.Chronoamperometry] | Potentiostatic | `E_step`, `duration`, `E_step1`/`E_step2`, `potentials`/`durations`, `E_quiet`, `t_quiet` | Single-, double-, and multi-step potential chronoamperometry. |
| [`LinearSweepVoltammetry`][softpotato.techniques.LinearSweepVoltammetry] | Potentiostatic | `E_start`, `E_end`, `scan_rate`, `dE`, `n_points` | Unidirectional linear potential sweep $E(t) = E_{\text{start}} \pm v t$. |
| [`CyclicVoltammetry`][softpotato.techniques.CyclicVoltammetry] | Potentiostatic | `E_start`, `E_vertex1`, `E_vertex2`, `scan_rate`, `E_end`, `n_cycles` | Single- or multi-cycle triangular potential waveform with switching vertices. |
| [`SampledCurrentVoltammetry`][softpotato.techniques.SampledCurrentVoltammetry] | Potentiostatic | `potentials` or (`E_start`, `E_end`, `dE`/`step_height`), `duration`, `sample_times` | Family of independent constant-potential steps sampled at fixed $\tau$. |
| [`DifferentialPulseVoltammetry`][softpotato.techniques.DifferentialPulseVoltammetry] | Potentiostatic | `E_start`, `E_end`, `step_height`, `pulse_amplitude`, `pulse_width`, `pulse_period`, `sample_fraction` | Staircase + pulse waveform with `sample_currents` returning `(E_base, i_pre, i_pulse, i_diff)`. |
| [`SquareWaveVoltammetry`][softpotato.techniques.SquareWaveVoltammetry] | Potentiostatic | `E_start`, `E_end`, `step_height`, `amplitude`, `frequency`, `sample_fraction` | Staircase + square-wave waveform with `sample_currents` returning `(E_step, i_forward, i_reverse, i_net)`. |
| [`Chronopotentiometry`][softpotato.techniques.Chronopotentiometry] | Galvanostatic | `current`, `duration`, `currents`/`durations`, `I_quiet`, `t_quiet` | Single- or multi-step constant-current protocol with potential recovery and transition time $\tau$ detection. |
| [`RotatingDiskElectrode`][softpotato.techniques.RotatingDiskElectrode] | Potentiostatic | `waveform`, `omega`/`rpm`, `kinematic_viscosity`, `model` | Hydrodynamic RDE wrapper (`'nernst_layer'` or `'levich_convection'`). |
| [`RDESweep`][softpotato.techniques.RDESweep] / [`simulate_rde_sweep`][softpotato.techniques.simulate_rde_sweep] | Potentiostatic | `omegas`/`rpms`, `waveform` or (`E_start`, `E_end`, `scan_rate`), `model` | Multi-rotation-rate RDE sweep builder and runner returning [`RDESweepResult`][softpotato.techniques.RDESweepResult]. |
| [`CustomWaveform`][softpotato.techniques.CustomWaveform] | Both | `segments`, `mode`, `n_points` | Composite excitation chaining [`ConstantSegment`][softpotato.techniques.ConstantSegment], [`LinearRampSegment`][softpotato.techniques.LinearRampSegment], [`SinusoidalSegment`][softpotato.techniques.SinusoidalSegment], and [`CallableSegment`][softpotato.techniques.CallableSegment]. |

### Simulation Pipeline & Result Container

| Symbol | Key Parameters / Attributes | Description |
| :--- | :--- | :--- |
| [`simulate`][softpotato.techniques.simulate] | `technique`, `mechanism`, `geometry`, `grid`, `solver`, `t_eval`, `kinetics` | End-to-end simulation orchestrator: builds physics-informed mesh, assembles BCs, integrates piecewise across `event_times`, and returns `TechniqueResult`. |
| [`simulate_sweep`][softpotato.techniques.simulate_sweep] | `techniques`, `mechanism`, `values`, `geometry`, `solver` | Runs parameter sweeps (sequence, mapping, or factory callable + `values`) via `simulate`. |
| [`TechniqueResult`][softpotato.techniques.TechniqueResult] | `time`, `potential`, `current`, `charge`, `solver_result`, `cycles`, `E_step`/`i_net`, `E_base`/`i_diff`, `transition_time`, `sample_currents()`, `plot()`, `plot_summary()` | Structured result container with dict-like species access (`result["Ox"]`), technique-specific views, and plotting facade. |

### Declarative Plotting Interfaces

Every plotting function accepts an optional Matplotlib `ax`, returns `(fig, ax)` (or `(fig, axes)` for `plot_summary`), uses SI units and IUPAC current polarity (anodic positive, cathodic negative), and never calls `plt.show()`.

| Function / Helper | Plot Kind Key | Description |
| :--- | :--- | :--- |
| [`PlotSpec`][softpotato.techniques.PlotSpec] / [`resolve_plot_kind`][softpotato.techniques.resolve_plot_kind] | `PLOT_ALIASES` | Declarative plot metadata and alias resolver mapping verbose names (e.g. `'current_vs_time'`) to canonical keys (`'i_vs_t'`). |
| [`plot_result`][softpotato.techniques.plot_result] | `kind=None` (uses `result.default_plot`) | Central technique-aware dispatcher (also exposed as `result.plot(kind=...)`). |
| [`plot_summary`][softpotato.techniques.plot_summary] | Multi-panel (`1x3` or `2x2`) | Renders default technique plot alongside `'i_vs_t'`, `'E_vs_t'`, and `'c_vs_x'` (also `result.plot_summary()`). |
| [`plot_sweep`][softpotato.techniques.plot_sweep] | Any supported `kind` | Overlays `Sequence[TechniqueResult]`, `Mapping[Any, TechniqueResult]`, or `RDESweepResult` on a single axes. |
| [`plot_voltammogram`][softpotato.techniques.plot_voltammogram] | `'voltammogram'` | $I$ vs. $E$ voltammogram with optional CV `cycle` selection and `peaks` overlay. |
| [`plot_current_vs_time`][softpotato.techniques.plot_current_vs_time] | `'i_vs_t'` | $I(t)$ chronoamperogram (supports 2D `SampledCurrentVoltammetry` step families via `step=`). |
| [`plot_potential_vs_time`][softpotato.techniques.plot_potential_vs_time] | `'E_vs_t'` | $E(t)$ waveform or chronopotentiogram with optional transition-time $\tau$ marker. |
| [`plot_concentration_profile`][softpotato.techniques.plot_concentration_profile] | `'c_vs_x'` | Spatial concentration profile $c_i(x, t)$ at one or more timestamps `t` and `species`. |
| [`plot_charge_vs_time`][softpotato.techniques.plot_charge_vs_time] | `'Q_vs_t'` | Cumulative charge $Q(t)$ chronocoulogram. |
| [`plot_scv`][softpotato.techniques.plot_scv] | `'scv'` | Sampled-current voltammogram $I(\tau)$ vs. $E_{\text{step}}$ for one or more `sample_times`. |
| [`plot_dpv`][softpotato.techniques.plot_dpv] | `'dpv'` | Differential pulse voltammogram ($\Delta I$ and optional `i_pre`, `i_pulse` components). |
| [`plot_swv`][softpotato.techniques.plot_swv] | `'swv'` | Square-wave voltammogram ($I_{\text{net}}$ and optional `i_forward`, `i_reverse` components). |
| [`plot_peaks`][softpotato.techniques.plot_peaks] | Diagnostic | Voltammogram with detected anodic/cathodic peak markers and annotations. |
| [`plot_randles_sevcik`][softpotato.techniques.plot_randles_sevcik] | Diagnostic | $I_p$ vs. $v^{1/2}$ Randles–Ševčík regression and fitted line. |
| [`plot_cottrell`][softpotato.techniques.plot_cottrell] | Diagnostic | $I$ vs. $t^{-1/2}$ Cottrell regression and fitted line. |
| [`plot_levich`][softpotato.techniques.plot_levich] | Diagnostic | $I_L$ vs. $\omega^{1/2}$ Levich regression and fitted line. |
| [`plot_koutecky_levich`][softpotato.techniques.plot_koutecky_levich] | Diagnostic | $1/I$ vs. $\omega^{-1/2}$ Koutecký–Levich regression and fitted line. |
| [`plot_tafel`][softpotato.techniques.plot_tafel] | Diagnostic | $\log_{10}|I|$ vs. $\eta$ Tafel plot and linear fit. |

### Post-Processing & Diagnostics

| Function / Dataclass | Inputs | Output | Description |
| :--- | :--- | :--- | :--- |
| [`find_peaks`][softpotato.techniques.find_peaks] | `TechniqueResult` or `(E, i)` | [`CVDiagnostics`][softpotato.techniques.CVDiagnostics] or `list[PeakInfo]` | Parabolic sub-grid peak detection ($E_{pa}, I_{pa}, E_{pc}, I_{pc}, \Delta E_p, |I_{pa}/I_{pc}|$). |
| [`analyze_randles_sevcik`][softpotato.techniques.analyze_randles_sevcik] | Multi-scan-rate results or `(v, I_p)` | `RandlesSevcikFitResult` | Extracts diffusivity $D$ from $I_p$ vs. $v^{1/2}$ regression. |
| [`analyze_nicholson`][softpotato.techniques.analyze_nicholson] | `TechniqueResult` or `(E, i)` | [`NicholsonDiagnostics`][softpotato.techniques.NicholsonDiagnostics] | Extracts $\Delta E_p$, $\Psi$, and standard rate constant $k^0$ via Nicholson's method. |
| [`analyze_levich`][softpotato.techniques.analyze_levich] | `RDESweepResult` or `results` | `LevichFitResult` | Extracts diffusivity $D$ and Levich constant $B$ from $I_L$ vs. $\omega^{1/2}$. |
| [`analyze_koutecky_levich`][softpotato.techniques.analyze_koutecky_levich] | `RDESweepResult` or `results` | `KouteckyLevichFitResult` | Separates kinetic current $I_k$ ($k^0$) and mass-transport slope $B$ ($D$). |
| [`analyze_tafel`][softpotato.techniques.analyze_tafel] | `TechniqueResult` or `(\eta, i)` | `TafelFitResult` | Extracts exchange current $I_0$, transfer coefficient $\alpha$, and Tafel slope $b$. |
| [`analyze_cottrell`][softpotato.techniques.analyze_cottrell] | `TechniqueResult` or `(t, i)` | `CottrellFitResult` | Extracts diffusivity $D$ from $I$ vs. $t^{-1/2}$ Cottrell regression. |
| [`analyze_sand`][softpotato.techniques.analyze_sand] | `TechniqueResult` or `(t, signal)` | [`SandDiagnostics`][softpotato.techniques.SandDiagnostics] | Detects transition time $\tau$, computes $I\tau^{1/2}$, and estimates $D$. |

---

## User Guide & Examples

### 1. Cyclic Voltammetry Sweep, Plotting & Diagnostics

```python
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    CyclicVoltammetry,
    analyze_nicholson,
    analyze_randles_sevcik,
    find_peaks,
    plot_randles_sevcik,
    plot_sweep,
    simulate,
    simulate_sweep,
)

ox = Species(
    "Ox",
    diffusivity=1e-9,
    bulk_concentration=1.0,
    standard_potential=0.0,
    n_electrons=1,
)
red = Species(
    "Red",
    diffusivity=1e-9,
    bulk_concentration=0.0,
    standard_potential=0.0,
    n_electrons=-1,
)
kinetics = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)

# Single CV simulation + declarative plotting
cv = CyclicVoltammetry(E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=0.1)
res = simulate(cv, [ox, red], kinetics=kinetics)
fig, ax = res.plot()  # Default plot ('voltammogram')
fig_sum, axes = res.plot_summary()  # 2x2 summary (voltammogram, i_vs_t, E_vs_t, c_vs_x)

# Multi-scan-rate sweep & diagnostics
results = simulate_sweep(
    lambda v: CyclicVoltammetry(
        E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v
    ),
    [ox, red],
    values=[0.05, 0.1, 0.2, 0.5],
    kinetics=kinetics,
)
plot_sweep(results, legend_title="Scan Rate")
rs_fit = analyze_randles_sevcik(results, n=1, area=1e-4, c_bulk=1.0)
plot_randles_sevcik(results, rs_fit)
nich = analyze_nicholson(res, D=1e-9)
```

### 2. Sampled-Current Voltammetry (SCV)

```python
from softpotato.reactions import NernstianEquilibrium, Species
from softpotato.techniques import SampledCurrentVoltammetry, plot_scv, simulate

ox = Species(
    "Ox",
    diffusivity=1e-9,
    bulk_concentration=1.0,
    standard_potential=0.0,
    n_electrons=1,
)
red = Species(
    "Red",
    diffusivity=1e-9,
    bulk_concentration=0.0,
    standard_potential=0.0,
    n_electrons=-1,
)

scv = SampledCurrentVoltammetry(
    E_start=0.3,
    E_end=-0.3,
    dE=0.05,
    duration=2.0,
    sample_times=[0.2, 0.5, 1.0, 2.0],
)
res_scv = simulate(scv, [ox, red], kinetics=NernstianEquilibrium(E0=0.0, n=1))

# Extract sampled voltammograms directly or plot with plot_scv / res_scv.plot('scv')
e_steps, i_sampled = res_scv.sample_currents()
fig, ax = plot_scv(res_scv)
```

### 3. Interactive Tutorials

For complete, runnable notebooks with plots and analytical validation, see:

- **Voltammetry & Diagnostics**: [`tutorials/techniques/cyclic_voltammetry.ipynb`](../tutorials/techniques/cyclic_voltammetry.ipynb) — LSV, multi-scan-rate CV sweeps, `plot_result`/`plot_summary`/`plot_sweep`, Randles–Ševčík, and Nicholson $k^0$ extraction.
- **Sampled-Current, Pulse Voltammetry & Chronopotentiometry**: [`tutorials/techniques/pulse_and_cp.ipynb`](../tutorials/techniques/pulse_and_cp.ipynb) — SCV (`plot_scv`), DPV (`plot_dpv`), SWV (`plot_swv`), and CP with Sand transition-time diagnostics.
- **Rotating Disk Electrode (RDE)**: [`tutorials/techniques/rde.ipynb`](../tutorials/techniques/rde.ipynb) — multi-rotation-rate RDE sweeps with Levich and Koutecký–Levich analysis.

---

## Module Contents

::: softpotato.techniques
