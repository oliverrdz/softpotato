# Voltammetry (LSV & CV)

Linear Sweep Voltammetry (`LinearSweepVoltammetry`) and Cyclic Voltammetry (`CyclicVoltammetry`) apply a linear potential ramp $E(t)$ at a constant scan rate $v = |dE/dt|$. Soft Potato automatically splits cyclic waveforms at their switching vertices (`event_times`) so the ODE/PDE solver integrates cleanly across slope discontinuities.

## Physical Inputs

### `LinearSweepVoltammetry`

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `E_start` | `float` | $\text{V}$ | Initial electrode potential. |
| `E_end` | `float` | $\text{V}$ | Final electrode potential (must differ from `E_start`). |
| `scan_rate` | `float` | $\text{V}/\text{s}$ | Positive scan rate magnitude $v > 0$. |
| `dE` | `float`, optional | $\text{V}$ | Potential step resolution for the evaluation grid (mutually exclusive with `n_points`). |
| `n_points` | `int`, optional | — | Number of evaluation time points $\ge 2$ (mutually exclusive with `dE`). |

### `CyclicVoltammetry`

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `E_start` | `float` | $\text{V}$ | Initial electrode potential. |
| `E_vertex1` | `float` | $\text{V}$ | First switching vertex potential $E_{\lambda 1}$ (must differ from `E_start`). |
| `E_vertex2` | `float` | $\text{V}$ | Second switching vertex potential $E_{\lambda 2}$ (must differ from `E_vertex1`). Set equal to `E_start` for a standard two-vertex triangular wave. |
| `scan_rate` | `float` | $\text{V}/\text{s}$ | Positive scan rate magnitude $v > 0$. |
| `E_end` | `float`, optional | $\text{V}$ | Final potential at the end of the last cycle (defaults to `E_start`). |
| `n_cycles` | `int`, default `1` | — | Number of complete voltammetric cycles ($\ge 1$). |
| `dE` / `n_points` | optional | $\text{V}$ / — | Evaluation grid resolution (mutually exclusive). |

### Multi-Scan-Rate Sweeps (`simulate_sweep`)

Pass a factory callable `lambda v: CyclicVoltammetry(..., scan_rate=v)` and `values=[...]` to `simulate_sweep()`. Each scan rate automatically constructs a physics-informed diffusion mesh sized to its own experiment duration $\sqrt{D t_{\max}}$.

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_cv_lsv.py"
```
