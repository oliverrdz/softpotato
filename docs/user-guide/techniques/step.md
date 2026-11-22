# Step Techniques (CA & CP)

Step techniques apply abrupt jumps in either electrode potential (**Chronoamperometry**, potentiostatic) or applied current (**Chronopotentiometry**, galvanostatic).

## Physical Inputs

### `Chronoamperometry` (Potentiostatic)

Supports three construction modes, each optionally preceded by a resting quiet hold `(E_quiet, t_quiet)`:

1. **Single-step**: `Chronoamperometry(E_step=..., duration=...)`
2. **Double-step**: `Chronoamperometry(E_step1=..., E_step2=..., t_step1=..., t_step2=...)`
3. **Multi-step**: `Chronoamperometry(potentials=[...], durations=[...])`

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `E_step` / `E_step1`, `E_step2` / `potentials` | `float` / `Sequence[float]` | $\text{V}$ | Applied step potential(s). |
| `duration` / `t_step1`, `t_step2` / `durations` | `float` / `Sequence[float]` | $\text{s}$ | Step duration(s) ($> 0$). |
| `E_quiet` | `float`, optional | $\text{V}$ | Initial resting potential applied before the first step. |
| `t_quiet` | `float`, default `0.0` | $\text{s}$ | Resting duration ($\ge 0$). Requires `E_quiet` when $> 0$. |

### `Chronopotentiometry` (Galvanostatic)

Supports single-step (`current`, `duration`) and multi-step / current-reversal (`currents`, `durations`) modes, with optional `(I_quiet, t_quiet)`.

Under **IUPAC polarity**, negative current ($I < 0$) is **cathodic** (reduction) and positive current ($I > 0$) is **anodic** (oxidation).

| Parameter | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `current` / `currents` | `float` / `Sequence[float]` | $\text{A}$ | Applied Faradaic current(s). |
| `duration` / `durations` | `float` / `Sequence[float]` | $\text{s}$ | Step duration(s) ($> 0$). |
| `I_quiet` | `float`, optional | $\text{A}$ | Initial resting current applied before the first step. |
| `t_quiet` | `float`, default `0.0` | $\text{s}$ | Resting duration ($\ge 0$). Requires `I_quiet` when $> 0$. |

!!! tip "Automatic Sand Transition Time ($\tau$)"
    When simulating `Chronopotentiometry` via `simulate()`, Soft Potato automatically detects the Sand transition time $\tau$ (when surface reactant concentration depletes to zero) and stores it in `result.metadata["transition_time"]`.

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_ca_cp.py"
```
