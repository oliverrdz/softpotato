# User Guide: Techniques & Waveforms

The `softpotato.techniques` module provides a high-level simulation pipeline that handles spatial meshing, boundary conditions, piecewise event-aware time integration, and Faradaic current extraction in a single `simulate()` call.

## The `simulate()` Pipeline

Every simulation follows the same three-step pattern:

1. **Define the chemical system**: create `Species` (ordered `[ox, red]` for $\text{Ox} + n e^- \rightarrow \text{Red}$) and optional electrode kinetics (`ButlerVolmer`, `MarcusHushChidsey`, or `NernstianEquilibrium`).
2. **Choose a technique**: instantiate a waveform generator such as `CyclicVoltammetry`, `Chronoamperometry`, `SquareWaveVoltammetry`, `RotatingDiskElectrode`, or `CustomWaveform`.
3. **Run and visualize**: call `simulate(technique, [ox, red], geometry=PlanarGeometry(area), kinetics=kinetics)` and pass the returned `TechniqueResult` to `plot_result(result)` or `plot_summary(result)`.

!!! note "Units Convention (SI)"
    All `softpotato.techniques` simulations use consistent SI units:

    - Length: $\text{m}$ (Area: $\text{m}^2$)
    - Time: $\text{s}$
    - Potential: $\text{V}$
    - Current: $\text{A}$
    - Concentration: $\text{mol}/\text{m}^3$ ($1\text{ mol}/\text{m}^3 = 1\text{ mM}$)
    - Diffusion coefficient: $\text{m}^2/\text{s}$ ($10^{-9}\text{ m}^2/\text{s} = 10^{-5}\text{ cm}^2/\text{s}$)
    - Heterogeneous rate constant: $\text{m}/\text{s}$ ($10^{-3}\text{ m}/\text{s} = 0.1\text{ cm}/\text{s}$)

## Parameter Sweeps with `simulate_sweep()`

To run a series of simulations across scan rates, step potentials, or any other parameter, pass a list, dictionary, or factory callable + `values` to `simulate_sweep()`:

```python
results = simulate_sweep(
    lambda v: CyclicVoltammetry(E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=v),
    [ox, red],
    values=[0.02, 0.05, 0.1, 0.2, 0.5],
    geometry=geometry,
    kinetics=kinetics,
)
```

Overlay the resulting traces with `plot_sweep(results)`.

## Technique Guides

| Guide | Covered Classes & Functions | Script |
| :--- | :--- | :--- |
| [Voltammetry (LSV & CV)](techniques/voltammetry.md) | `LinearSweepVoltammetry`, `CyclicVoltammetry`, `simulate_sweep` | [`guide_cv_lsv.py`](../scripts/guide_cv_lsv.py) |
| [Step Techniques (CA & CP)](techniques/step.md) | `Chronoamperometry`, `Chronopotentiometry` | [`guide_ca_cp.py`](../scripts/guide_ca_cp.py) |
| [Pulse & Sampled-Current Voltammetry](techniques/pulse.md) | `SampledCurrentVoltammetry`, `DifferentialPulseVoltammetry`, `SquareWaveVoltammetry` | [`guide_pulse.py`](../scripts/guide_pulse.py) |
| [Rotating Disk Electrode (RDE)](techniques/rde.md) | `RotatingDiskElectrode`, `simulate_rde_sweep`, `RDESweep` | [`guide_rde.py`](../scripts/guide_rde.py) |
| [Custom Waveforms & Segments](techniques/waveforms.md) | `CustomWaveform`, `ConstantSegment`, `LinearRampSegment`, `SinusoidalSegment`, `CallableSegment` | [`guide_custom_waveform.py`](../scripts/guide_custom_waveform.py) |

## Post-Processing & Analytical Guides

| Guide | Covered Functions & Submodules | Script |
| :--- | :--- | :--- |
| [Diagnostics & Post-Processing](postprocessing/diagnostics.md) | `find_peaks`, `analyze_randles_sevcik`, `analyze_nicholson`, `analyze_cottrell`, `analyze_levich`, `analyze_koutecky_levich`, `analyze_sand`, `analyze_tafel` | [`guide_diagnostics.py`](../scripts/guide_diagnostics.py) |
| [Plotting Suite](postprocessing/plotting.md) | `plot_result`, `plot_voltammogram`, `plot_peaks`, `plot_summary`, `plot_sweep`, `plot_randles_sevcik`, `plot_tafel`, `plot_levich`, `plot_koutecky_levich`, `plot_dpv`, `plot_swv`, `plot_scv` | [`guide_plotting.py`](../scripts/guide_plotting.py) |
| [Analytical Models](analytical/models.md) | `step`, `hydrodynamics`, `kinetics`, `voltammetry`, `microelectrodes`, `secm` | [`guide_analytical_models.py`](../scripts/guide_analytical_models.py) |
| [Curve Fitting & Regression](analytical/fitting.md) | `fit_cottrell`, `fit_randles_sevcik`, `fit_secm_approach`, `fit_curve` | [`guide_analytical_fitting.py`](../scripts/guide_analytical_fitting.py) |
