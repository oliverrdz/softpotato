# Tutorials

Interactive, end-to-end Jupyter notebooks with saved outputs—each following a consistent **Goal → Physical Setup → Simulation → Diagnostics & Plots → Key Takeaways** workflow.

Every notebook can be run locally or opened in Google Colab (uncomment `# !pip install -q softpotato` in the first code cell).

---

## Starter Notebooks

New to Soft Potato? Start with these 5-minute quickstart notebooks:

| Notebook | Topic | Key APIs |
| :--- | :--- | :--- |
| [Chronoamperometry (CA) Quickstart](start/ca_quickstart.ipynb) | Potential-step simulation vs. Cottrell equation & $D$ fitting | `Chronoamperometry`, `spa.cottrell()`, `spa.fit_cottrell()` |
| [Cyclic Voltammetry (CV) Quickstart](start/cv_quickstart.ipynb) | Single CV, Randles–Ševčík check, and $k^0$ reversibility | `CyclicVoltammetry`, `ButlerVolmer`, `spa.randles_sevcik()` |

---

## By Technique

Step-by-step walkthroughs of Soft Potato's experimental technique generators and diagnostic pipelines:

| Notebook | Focus | Key APIs |
| :--- | :--- | :--- |
| [Chronoamperometry & the Cottrell Equation](techniques/cottrell.ipynb) | Planar potential step, $t^{-1/2}$ decay, and $D$ extraction | `Chronoamperometry`, `analyze_cottrell()`, `plot_cottrell()`, `fit_cottrell()` |
| [Chronoamperometry Across Geometries](techniques/cottrell_geometries.ipynb) | Planar vs. spherical vs. cylindrical potential-step transients | `PlanarGeometry`, `SphericalGeometry`, `CylindricalGeometry`, `cottrell_spherical()`, `cottrell_cylinder()` |
| [Cyclic Voltammetry & Multi-Scan-Rate Sweeps](techniques/cyclic_voltammetry.ipynb) | Multi-cycle CV, peak detection, summary panels, and scan-rate sweeps | `CyclicVoltammetry`, `simulate_sweep()`, `find_peaks()`, `plot_peaks()`, `plot_summary()` |
| [Pulse Voltammetry & Chronopotentiometry](techniques/pulse_and_cp.ipynb) | SCV, DPV, SWV, and constant-current CP with Sand transition time | `SampledCurrentVoltammetry`, `DifferentialPulseVoltammetry`, `SquareWaveVoltammetry`, `Chronopotentiometry`, `analyze_sand()` |
| [Rotating Disk Electrode (RDE)](techniques/rde.ipynb) | Rotation-rate sweeps, Levich, and Koutecký–Levich analysis | `simulate_rde_sweep()`, `analyze_levich()`, `plot_levich()`, `analyze_koutecky_levich()`, `plot_koutecky_levich()` |

---

## By Mechanism & Geometry

Focused studies on electrode geometry, interfacial kinetics, and coupled homogeneous reaction mechanisms:

| Notebook | Focus | Key APIs |
| :--- | :--- | :--- |
| [Reversible Planar CV (`NernstianEquilibrium`)](mechanisms/cv_nernst_planar.ipynb) | Nernstian CV signatures ($\Delta E_p$ approx 57 $\text{mV}$) and concentration profiles | `NernstianEquilibrium`, `find_peaks()`, `plot_concentration_profile()` |
| [Reversible Spherical CV](mechanisms/cv_nernst_spherical.ipynb) | Macro-to-microelectrode transition from peak-shaped to sigmoidal waves | `SphericalGeometry`, `microsphere_limiting_current()` |
| [Randles–Ševčík Analysis](mechanisms/cv_randles_sevcik.ipynb) | Multi-scan-rate $I_p$ vs. $v^{1/2}$ regression and $D$ confidence intervals | `simulate_sweep()`, `analyze_randles_sevcik()`, `plot_randles_sevcik()`, `fit_randles_sevcik()` |
| [Butler–Volmer Kinetics, Nicholson & Tafel](mechanisms/cv_butler_volmer.ipynb) | Reversibility regimes across $k^0$, Nicholson $\Psi$, and Tafel extraction | `ButlerVolmer`, `analyze_nicholson()`, `analyze_tafel()`, `plot_tafel()` |
| [The `EC` Mechanism](mechanisms/cv_ec.ipynb) | Coupled follow-up first-order reaction ($\text{Red} \xrightarrow{k_c} \text{P}$) | `Mechanism`, `ElectrochemicalStep`, `FirstOrderReaction` |
| [The `EE` Mechanism](mechanisms/cv_ee.ipynb) | Sequential two-electron transfer ($\text{A} \rightleftharpoons \text{B} \rightleftharpoons \text{C}$) and wave merging | `Mechanism`, `ElectrochemicalStep` |
| [The Catalytic `EC'` Mechanism](mechanisms/catalytic_ec.ipynb) | Homogeneous catalyst regeneration ($\text{Red} \xrightarrow{k_{\text{cat}}} \text{Ox}$) and plateau scaling | `Mechanism`, `ElectrochemicalStep`, `CatalyticReaction` |

---

## Related Documentation

- **[User Guide](../user-guide/index.md)**: Task-oriented guides for `simulate()`, diagnostics, plotting, and curve fitting.
- **[Concepts](../concepts/index.md)**: Theoretical and numerical foundations behind electrode kinetics, mass transport, and reaction mechanisms.
- **[API Reference](../api/index.md)**: Complete class and function signatures.
