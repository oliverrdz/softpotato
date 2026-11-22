# Diagnostics & Post-Processing

The `softpotato.techniques` post-processing suite operates directly on `TechniqueResult` (and `RDESweepResult`) objects returned by `simulate()`, `simulate_sweep()`, and `simulate_rde_sweep()`, extracting peak metrics, diffusion coefficients, and heterogeneous rate constants without custom boilerplate.

## Available Diagnostic Functions

| Function | Input | Extracted Quantities | Return Type |
| :--- | :--- | :--- | :--- |
| `find_peaks` | CV / LSV / SWV / DPV `TechniqueResult` | $E_{pa}$, $I_{pa}$, $E_{pc}$, $I_{pc}$, $E_{p/2}$, $\Delta E_p$, $E_{1/2}$, $\|I_{pa}/I_{pc}\|$ | `CVDiagnostics` (or `list[PeakInfo]` with `return_all=True`) |
| `analyze_randles_sevcik` | Sequence/mapping of CV/LSV `TechniqueResult` | Diffusion coefficient $D$, slope, intercept, $R^2$ | `RandlesSevcikFitResult` |
| `analyze_nicholson` | Quasi-reversible CV `TechniqueResult` | $\Delta E_p$, Nicholson $\Psi$, $k^0$, Matsuda–Ayabe $\Lambda$ & regime | `NicholsonDiagnostics` |
| `analyze_cottrell` | `Chronoamperometry` `TechniqueResult` | Diffusion coefficient $D$, background $I_{\text{bg}}$, $R^2$ | `CottrellFitResult` |
| `analyze_levich` | `RDESweepResult` or RDE `TechniqueResult` list | Diffusion coefficient $D$, Levich constant $B$, $R^2$ | `LevichFitResult` |
| `analyze_koutecky_levich` | `RDESweepResult` at mixed-control `potential` | Kinetic current $I_K$, $k(E)$, Levich $B$, $D$, $R^2$ | `KouteckyLevichFitResult` |
| `analyze_sand` | `Chronopotentiometry` `TechniqueResult` | Transition time $\tau$, theoretical $\tau$, Sand product $\|I\|\sqrt{\tau}$, $D$ | `SandDiagnostics` |
| `analyze_tafel` | Voltammetric `TechniqueResult` + `E0` | Tafel slope $b$ (V/dec & mV/dec), $\alpha$, $I_0$, optional $k^0$, $R^2$ | `TafelFitResult` |

!!! tip "Raw Arrays Supported Too"
    Every diagnostic function also accepts raw NumPy arrays (for example `find_peaks(E, I)` or `analyze_cottrell(t, I, ...)`), so you can apply the exact same analysis pipeline to experimental potentiostat exports.

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_diagnostics.py"
```
