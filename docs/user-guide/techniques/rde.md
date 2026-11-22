# Rotating Disk Electrode (RDE)

`RotatingDiskElectrode` couples any potentiostatic waveform (defaulting to `LinearSweepVoltammetry`) with forced hydrodynamic convection at a rotating disk electrode.

## Hydrodynamic Models

Two mass-transport models are supported via the `model` parameter:

- **`"nernst_layer"` (default)**: Nernst stagnant diffusion layer approximation on $x \in [0, \delta_N]$ with a bulk Dirichlet boundary condition at $x = \delta_N$:

  $$\delta_N = 1.610 \, D^{1/3} \nu^{1/6} \omega^{-1/2}$$

- **`"convective_diffusion"`**: Full Levich axial convective-diffusion PDE on $x \in [0, k \delta_N]$ (default $k = 3.0$) including the von Kármán axial fluid velocity toward the disk:

  $$v_x(x) = -0.51023 \, \omega^{3/2} \nu^{-1/2} x^2$$

## Physical Inputs

### `RotatingDiskElectrode` & `simulate_rde_sweep`

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `waveform` | `BaseTechnique`, optional | — | Custom potentiostatic waveform (mutually exclusive with `E_start`, `E_end`, `scan_rate`). |
| `E_start`, `E_end`, `scan_rate` | `float`, optional | $\text{V}$, $\text{V}/\text{s}$ | Direct LSV parameters when `waveform` is omitted. |
| `omega` / `omegas` | `float` / `Sequence[float]` | $\text{rad}/\text{s}$ | Angular rotation velocity ($> 0$). Mutually exclusive with `rpm` / `rpms`. |
| `rpm` / `rpms` | `float` / `Sequence[float]` | $\text{rpm}$ | Rotational speed in revolutions per minute ($> 0$). Mutually exclusive with `omega` / `omegas`. |
| `kinematic_viscosity` | `float`, default `1e-6` | $\text{m}^2/\text{s}$ | Electrolyte kinematic viscosity $\nu > 0$ ($10^{-6}\text{ m}^2/\text{s}$ for water at $298\text{ K}$). |
| `model` | `'nernst_layer'` or `'convective_diffusion'` | — | Hydrodynamic transport model. |
| `domain_factor` | `float`, optional | — | Multiplier $k > 0$ on $\delta_N$ for mesh extent (defaults to `1.0` for Nernst layer, `3.0` for convective diffusion). |

`simulate_rde_sweep()` returns an `RDESweepResult` containing `.omegas`, `.rpms`, `.limiting_currents`, and `.results`, which can be passed directly to `plot_sweep()`, `analyze_levich()`, or `analyze_koutecky_levich()`.

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_rde.py"
```
