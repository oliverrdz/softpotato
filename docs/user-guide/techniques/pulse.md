# Pulse & Sampled-Current Voltammetry

Pulse and sampled-current techniques sample Faradaic current at specific windows after potential steps to enhance sensitivity and discriminate against capacitive background currents.

## Physical Inputs

### `SampledCurrentVoltammetry` (SCV)

Simulates a family of independent constant-potential steps across a potential window and samples $I(\tau)$ at fixed sampling time(s) $\tau \in (0, t_{\max}]$.

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `potentials` | `Sequence[float]`, optional | $\text{V}$ | Explicit list/array of step potentials (mutually exclusive with `E_start`/`E_end`/`dE`). |
| `E_start`, `E_end`, `dE` | `float`, optional | $\text{V}$ | Potential window bounds and positive step increment (`step_height` is an alias for `dE`). |
| `duration` | `float` | $\text{s}$ | Constant-potential hold duration ($> 0$) shared by all steps. |
| `sample_times` | `float` or `Sequence[float]`, optional | $\text{s}$ | Sampling time(s) $\tau \in (0, t_{\max}]$ used by `result.sample_currents()`. |
| `E_quiet`, `t_quiet` | optional | $\text{V}$, $\text{s}$ | Optional resting hold applied before every step. |

### `DifferentialPulseVoltammetry` (DPV)

Superimposes potential pulses of amplitude $\Delta E_p$ and width $t_p$ onto a staircase ramp with period $\tau$ and step height $\Delta E_s$.

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `E_start`, `E_end` | `float` | $\text{V}$ | Initial and final staircase base potentials. |
| `step_height` | `float` | $\text{V}$ | Staircase step magnitude $\Delta E_s > 0$. |
| `pulse_amplitude` | `float` | $\text{V}$ | Pulse amplitude $\Delta E_p \neq 0$ (negative for cathodic pulses, positive for anodic). |
| `pulse_width` | `float` | $\text{s}$ | Pulse duration $t_p > 0$. |
| `pulse_period` | `float` | $\text{s}$ | Total cycle period $\tau > t_p$. |
| `sample_fraction` | `float`, default `0.1` | — | Fraction $\in (0, 1]$ at the end of each base/pulse segment averaged by `sample_currents()`. |

### `SquareWaveVoltammetry` (SWV)

Superimposes a symmetric square wave of amplitude $E_{\text{sw}} > 0$ and frequency $f > 0$ (period $\tau = 1/f$, half-period $t_p = 1/(2f)$) onto a staircase ramp.

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `E_start`, `E_end` | `float` | $\text{V}$ | Initial and final staircase potentials. |
| `step_height` | `float` | $\text{V}$ | Staircase step magnitude $\Delta E_s > 0$. |
| `amplitude` | `float` | $\text{V}$ | Square-wave amplitude $E_{\text{sw}} > 0$. |
| `frequency` | `float` | $\text{Hz}$ | Square-wave frequency $f > 0$. |
| `sample_fraction` | `float`, default `0.1` | — | Fraction $\in (0, 1]$ at the end of each half-cycle averaged by `sample_currents()`. |

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_pulse.py"
```
