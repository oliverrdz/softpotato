# Plotting Suite

`softpotato.techniques` includes a declarative Matplotlib plotting suite. Every renderer accepts an optional `ax` parameter to draw onto existing axes and returns `(fig, ax)` (or `(fig, axes)` for `plot_summary()`) without calling `plt.show()`.

## Core & Technique-Specific Renderers

| Function | Target | Key Options |
| :--- | :--- | :--- |
| `plot_result` | Any `TechniqueResult` | Dispatches via `result.default_plot` or `kind` (`'voltammogram'`, `'i_vs_t'`, `'E_vs_t'`, `'c_vs_x'`, `'Q_vs_t'`, `'dpv'`, `'swv'`, `'scv'`). |
| `plot_voltammogram` | CV, LSV, RDE `TechniqueResult` | `cycle` (`int`, `'all'`, or `None`), `peaks` (`True`, `CVDiagnostics`, or `PeakInfo` list). |
| `plot_summary` | Any `TechniqueResult` | Multi-panel summary combining `default_plot`, `i_vs_t`, `E_vs_t`, and `c_vs_x`. |
| `plot_sweep` | `simulate_sweep()` or `RDESweepResult` | Overlays multiple traces with automatic labels (`labels`, `legend_title`). |
| `plot_dpv` | `DifferentialPulseVoltammetry` | `show_components=True` overlays $I_{\text{pulse}}$ and $I_{\text{pre}}$. |
| `plot_swv` | `SquareWaveVoltammetry` | `show_components=True` overlays $I_{\text{forward}}$ and $I_{\text{reverse}}$. |
| `plot_scv` | `SampledCurrentVoltammetry` | `sample_times` selects $\tau$ values for $I(\tau)$ vs. $E_{\text{step}}$. |

## Diagnostic Regression Plots

| Function | Paired Diagnostic | Plotted Axes |
| :--- | :--- | :--- |
| `plot_peaks` | `find_peaks()` | $I$ vs. $E$ with anodic/cathodic peak markers and $\Delta E_p$ title. |
| `plot_randles_sevcik` | `analyze_randles_sevcik()` | $I_p$ vs. $\sqrt{v}$ scatter + linear/nonlinear regression fit. |
| `plot_cottrell` | `analyze_cottrell()` | $I$ vs. $t^{-1/2}$ scatter + regression fit. |
| `plot_levich` | `analyze_levich()` | $I_L$ vs. $\omega^{1/2}$ scatter + regression fit. |
| `plot_koutecky_levich` | `analyze_koutecky_levich()` | $\|I\|^{-1}$ vs. $\omega^{-1/2}$ scatter + regression fit. |
| `plot_tafel` | `analyze_tafel()` | $\eta$ vs. $\log_{10}\|I\|$ scatter + Tafel line fit. |

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_plotting.py"
```
