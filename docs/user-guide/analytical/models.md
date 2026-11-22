# Analytical Models (`softpotato.analytical`)

The `softpotato.analytical` package provides closed-form solutions, asymptotic expansions, and empirical electrochemical models for rapid baseline calculations and solver benchmarking.

!!! note "Consistent Units"
    All functions in `softpotato.analytical` are unit-agnostic as long as length units are consistent across `D`, `area`, `c_bulk`, `nu`, and electrode radius `a` (for example, all SI: $\text{m}$, $\text{m}^2$, $\text{mol}/\text{m}^3$; or all CGS: $\text{cm}$, $\text{cm}^2$, $\text{mol}/\text{cm}^3$).

## Submodules Overview

### `softpotato.analytical.step` — Potential & Current Steps

- `cottrell`, `cottrell_spherical`, `cottrell_cylinder`, `cottrell_step`: Chronoamperometric current transients at planar, spherical, and cylindrical electrodes.
- `anson`: Chronocoulometric charge $Q(t)$ for planar potential steps.
- `sand`, `sand_transition_time`, `sand_potential`: Galvanostatic chronopotentiometry transition time $\tau$ and $E(t)$ response.
- `step_concentration_profile`, `step_flux_profile`: Exact error-function spatial concentration $c(x, t)$ and interfacial flux profiles.

### `softpotato.analytical.hydrodynamics` — RDE & RRDE

- `levich`, `levich_constant`, `nernst_diffusion_layer`: Levich limiting current $I_L$, Levich constant $B$, and Nernst diffusion layer thickness $\delta_N$.
- `koutecky_levich`, `koutecky_levich_analysis`: Mixed kinetic/mass-transport RDE current and regression.
- `rpm_to_rad_s`, `rad_s_to_rpm`: Angular velocity conversions ($\omega = 2\pi \cdot \text{rpm} / 60$).
- `collection_efficiency`, `ring_limiting_current`, `ring_collection_current`, `shielding_factor`, `rotating_ring_disk`: Albery–Bruckenstein rotating ring-disk electrode (RRDE) equations.

### `softpotato.analytical.kinetics` — Interfacial Kinetics

- `nernst`, `nernst_potential`, `nernst_ratio`, `nernst_equilibrium_concentrations`: Equilibrium potential and surface concentration ratios.
- `exchange_current`, `exchange_current_density`, `charge_transfer_resistance`: $I_0$, $j_0$, and $R_{\text{ct}}$.
- `butler_volmer`, `butler_volmer_current_density`, `butler_volmer_linear`: Full and low-overpotential linearized Butler–Volmer equations.
- `tafel`, `tafel_slope`, `tafel_overpotential`, `tafel_analysis`: High-field Tafel kinetics.

### `softpotato.analytical.voltammetry` — Voltammetric Diagnostics

- `randles_sevcik`, `randles_sevcik_irreversible`, `randles_sevcik_quasi`: Reversible, irreversible, and quasi-reversible CV peak currents.
- `nicholson_psi`, `nicholson_delta_ep`, `nicholson_rate_constant`: Nicholson working curve relating $\Delta E_p$ to $\Psi$ and $k^0$.
- `matsuda_ayabe`, `matsuda_ayabe_lambda`, `peak_potential_irreversible`: Dimensionless reversibility parameter $\Lambda$ and regime classification.

### `softpotato.analytical.microelectrodes` — Ultramicroelectrodes (UMEs)

- `microdisc_limiting_current`, `microhemisphere_limiting_current`, `microsphere_limiting_current`, `microband_limiting_current`: Steady-state UME limiting currents.
- `microdisc_transient`, `mahon_oldham_transient`: Shoup–Szabo and Mahon–Oldham microdisc chronoamperometric transients.

### `softpotato.analytical.secm` — Scanning Electrochemical Microscopy

- `secm_approach_positive_feedback`, `secm_approach_negative_feedback`, `secm_approach_curve`: Normalized SECM tip current $I_T(L)$ for conducting, insulating, and finite-kinetics substrates.
- `secm_limiting_current_infinite`, `secm_tip_current`: Steady-state tip current in bulk and dimensional approach currents.

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_analytical_models.py"
```
