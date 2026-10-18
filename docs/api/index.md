# API Reference

Welcome to the **Soft Potato** API Reference. The toolkit is organized into modular
subpackages and modules for physical constants, closed-form analytical electrochemistry,
and numerical PDE solvers.

## Constants Module

The [`softpotato.constants`][softpotato.constants] module provides standardized physical and electrochemical
constants defined according to CODATA 2018 / 2019 SI conventions [[1]](#ref-1).

| Constant / Identifier | Symbol | Value & Units |
| :--- | :--- | :--- |
| [`FARADAY`][softpotato.constants.FARADAY] (`F`) | $F$ | $96485.332 \,\text{C}\cdot\text{mol}^{-1}$ |
| [`GAS_CONSTANT`][softpotato.constants.GAS_CONSTANT] (`R`) | $R$ | $8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$ |
| [`STANDARD_TEMPERATURE`][softpotato.constants.STANDARD_TEMPERATURE] (`T_STD`) | $T^\circ$ | $298.15\,\text{K}$ ($25\,^\circ\text{C}$) |
| [`AVOGADRO`][softpotato.constants.AVOGADRO] | $N_{\text{A}}$ | $6.02214 \times 10^{23}\,\text{mol}^{-1}$ |
| [`BOLTZMANN`][softpotato.constants.BOLTZMANN] | $k_{\text{B}}$ | $1.38065 \times 10^{-23}\,\text{J}\cdot\text{K}^{-1}$ |
| [`ELEMENTARY_CHARGE`][softpotato.constants.ELEMENTARY_CHARGE] | $e$ | $1.60218 \times 10^{-19}\,\text{C}$ |
| [`VACUUM_PERMITTIVITY`][softpotato.constants.VACUUM_PERMITTIVITY] | $\varepsilon_0$ | $8.85419 \times 10^{-12}\,\text{F}\cdot\text{m}^{-1}$ |

## Analytical Module

The [`softpotato.analytical`][softpotato.analytical] subpackage provides closed-form analytical solutions,
asymptotic approximations, and diagnostic formulations across transient step techniques,
hydrodynamics, interfacial kinetics, voltammetry, microelectrodes, SECM, parameter fitting,
and numerical benchmark suites.

### Transient Step Techniques

The [`softpotato.analytical.step`][softpotato.analytical.step] module provides exact solutions for potentiostatic
and galvanostatic step experiments, including Cottrell, Anson, and Sand formulations.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`cottrell`][softpotato.analytical.step.cottrell] | $t, n, D, c^*, A$ | $I(t)$ | Planar chronoamperometric diffusion-limited current transient. | [2](#ref-2) |
| [`cottrell_step`][softpotato.analytical.step.cottrell_step] | $t, \tau, n, D, c^*, A, D_{\text{Red}}$ | $I(t)$ | Double potential step chronoamperometry reversal current. | [3](#ref-3) |
| [`cottrell_spherical`][softpotato.analytical.step.cottrell_spherical] | $t, r, n, D, c^*, A$ | $I(t)$ | Chronoamperometric current at a spherical electrode. | [3](#ref-3) |
| [`cottrell_cylinder`][softpotato.analytical.step.cottrell_cylinder] | $t, r_0, n, D, c^*, l, A$ | $I(t)$ | Chronoamperometric current at a cylindrical wire/fiber electrode. | [4](#ref-4), [5](#ref-5) |
| [`anson`][softpotato.analytical.step.anson] | $t, n, D, c^*, A, Q_{\text{dl}}, \Gamma, \tau$ | $Q(t)$ | Cumulative charge transient for chronocoulometry. | [6](#ref-6) |
| [`sand`][softpotato.analytical.step.sand] | $t, I, n, D, c^*, A$ | $c(0, t)$ | Surface reactant concentration transient under constant current. | [7](#ref-7) |
| [`sand_transition_time`][softpotato.analytical.step.sand_transition_time] | $I, n, D, c^*, A$ | $\tau$ | Transition time for galvanostatic chronopotentiometry. | [7](#ref-7) |
| [`sand_potential`][softpotato.analytical.step.sand_potential] | $t, \tau, E_{1/2}, n, T$ | $E(t)$ | Potential-time response curve for chronopotentiometry. | [7](#ref-7) |
| [`step_concentration_profile`][softpotato.analytical.step.step_concentration_profile] | $x, t, D, c^*$ | $c(x, t)$ | Exact spatial concentration profile following a potential step. | [8](#ref-8) |
| [`step_flux_profile`][softpotato.analytical.step.step_flux_profile] | $x, t, D, c^*$ | $J(x, t)$ | Exact spatial diffusion flux profile following a potential step. | [8](#ref-8) |

### Hydrodynamics & Convection Equations

The [`softpotato.analytical.hydrodynamics`][softpotato.analytical.hydrodynamics] module provides models for rotating disk
(RDE) and rotating ring-disk (RRDE) convective electrochemical systems.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`levich`][softpotato.analytical.hydrodynamics.levich] | $\omega, n, D, c^*, A, \nu$ | $I_{\text{L}}$ | Convective mass-transfer limiting current at an RDE. | [9](#ref-9) |
| [`levich_constant`][softpotato.analytical.hydrodynamics.levich_constant] | $n, D, c^*, A, \nu$ | $B$ | Theoretical Levich slope parameter for RDE rotation series. | [9](#ref-9) |
| [`koutecky_levich`][softpotato.analytical.hydrodynamics.koutecky_levich] | $\omega, I_{\text{k}}, k, I_{\text{lim}}, n, D, c^*, A, \nu$ | $I$ | Current under mixed kinetic and convective mass-transfer control. | [10](#ref-10) |
| [`koutecky_levich_analysis`][softpotato.analytical.hydrodynamics.koutecky_levich_analysis] | $\omega, I, A, c^*, n, \nu$ | $k, B$ | Kinetic rate constant and Levich slope extraction from RDE data. | [10](#ref-10) |
| [`nernst_diffusion_layer`][softpotato.analytical.hydrodynamics.nernst_diffusion_layer] | $\omega, D, \nu$ | $\delta$ | Steady-state stagnant diffusion layer thickness at an RDE. | [9](#ref-9) |
| [`collection_efficiency`][softpotato.analytical.hydrodynamics.collection_efficiency] | $r_1, r_2, r_3$ | $N$ | Theoretical collection efficiency for an RRDE geometry. | [11](#ref-11) |
| [`shielding_factor`][softpotato.analytical.hydrodynamics.shielding_factor] | $r_1, r_2, r_3, N$ | $S$ | Theoretical geometric shielding factor for an RRDE. | [11](#ref-11) |
| [`ring_collection_current`][softpotato.analytical.hydrodynamics.ring_collection_current] | $I_{\text{disk}}, N, n_{\text{ring}}, n_{\text{disk}}$ | $I_{\text{ring}}$ | Ring electrode current from disk product collection. | [11](#ref-11) |
| [`ring_limiting_current`][softpotato.analytical.hydrodynamics.ring_limiting_current] | $\omega, r_1, r_2, r_3, n, D, c^*, \nu$ | $I_{\text{R,lim}}$ | Convective mass-transfer limiting current at an RRDE ring. | [11](#ref-11) |
| [`rotating_ring_disk`][softpotato.analytical.hydrodynamics.rotating_ring_disk] | $r_1, r_2, r_3, \omega, n, D, c^*, \nu$ | $N, S, I_{\text{D,lim}}, I_{\text{R,lim}}$ | Comprehensive collection and limiting current metrics for RRDE. | [11](#ref-11) |
| [`rad_s_to_rpm`][softpotato.analytical.hydrodynamics.rad_s_to_rpm] | $\omega$ | $f_{\text{RPM}}$ | Angular velocity to rotational speed conversion. |  |
| [`rpm_to_rad_s`][softpotato.analytical.hydrodynamics.rpm_to_rad_s] | $f_{\text{RPM}}$ | $\omega$ | Rotational speed to angular velocity conversion. |  |

### Thermodynamics & Interfacial Kinetics

The [`softpotato.analytical.kinetics`][softpotato.analytical.kinetics] module provides thermodynamic equilibrium potentials
and electrode kinetic models, including Nernst, Butler–Volmer, and Tafel formulations.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`nernst`][softpotato.analytical.kinetics.nernst] | $E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T$ | $E$ | Reversible equilibrium reduction potential. | [12](#ref-12) |
| [`nernst_potential`][softpotato.analytical.kinetics.nernst] | $E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T$ | $E$ | Nernst equilibrium electrode potential calculation. | [12](#ref-12) |
| [`nernst_ratio`][softpotato.analytical.kinetics.nernst_ratio] | $E, E^\circ, n, T$ | $c_{\text{Ox}} / c_{\text{Red}}$ | Equilibrium concentration ratio from applied potential. | [12](#ref-12) |
| [`nernst_equilibrium_concentrations`][softpotato.analytical.kinetics.nernst_equilibrium_concentrations] | $E, E^\circ, c_{\text{total}}, n, T$ | $c_{\text{Ox}}, c_{\text{Red}}$ | Individual equilibrium species concentrations from total pool. | [12](#ref-12) |
| [`butler_volmer`][softpotato.analytical.kinetics.butler_volmer] | $\eta, I_0, \alpha, n, T$ | $I$ | Faradaic current from activation overpotential. | [13](#ref-13), [14](#ref-14) |
| [`butler_volmer_current_density`][softpotato.analytical.kinetics.butler_volmer_current_density] | $\eta, j_0, \alpha, n, T$ | $j$ | Faradaic current density from activation overpotential. | [13](#ref-13), [14](#ref-14) |
| [`butler_volmer_linear`][softpotato.analytical.kinetics.butler_volmer_linear] | $\eta, I_0, n, T$ | $I$ | Low-overpotential linearized Butler–Volmer current. | [13](#ref-13), [14](#ref-14) |
| [`charge_transfer_resistance`][softpotato.analytical.kinetics.charge_transfer_resistance] | $I_0, n, T$ | $R_{\text{ct}}$ | Charge-transfer resistance at equilibrium potential. | [13](#ref-13), [14](#ref-14) |
| [`exchange_current`][softpotato.analytical.kinetics.exchange_current] | $k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, A, \alpha, n$ | $I_0$ | Standard exchange current from rate constant and concentrations. | [13](#ref-13), [14](#ref-14) |
| [`exchange_current_density`][softpotato.analytical.kinetics.exchange_current_density] | $k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, \alpha, n$ | $j_0$ | Exchange current density from rate constant and concentrations. | [13](#ref-13), [14](#ref-14) |
| [`tafel`][softpotato.analytical.kinetics.tafel] | $\eta, I_0, \alpha, n, T$ | $I$ | High-overpotential unidirectional Tafel current approximation. | [15](#ref-15) |
| [`tafel_overpotential`][softpotato.analytical.kinetics.tafel_overpotential] | $I, I_0, \alpha, n, T$ | $\eta$ | Activation overpotential calculated from current. | [15](#ref-15) |
| [`tafel_slope`][softpotato.analytical.kinetics.tafel_slope] | $\alpha, n, T$ | $b$ | Theoretical Tafel slope in Volts per decade. | [15](#ref-15) |
| [`tafel_analysis`][softpotato.analytical.kinetics.tafel_analysis] | $\eta, I, A, c^*, n, T$ | $I_0, \alpha, b$ | Semi-logarithmic Tafel regression for kinetic parameter extraction. | [15](#ref-15) |

### Voltammetry & Kinetic Diagnostics

The [`softpotato.analytical.voltammetry`][softpotato.analytical.voltammetry] module provides peak current expressions and
reversibility diagnostics for cyclic and linear sweep voltammetry.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`randles_sevcik`][softpotato.analytical.voltammetry.randles_sevcik] | $v, c^*, D, A, n, T$ | $I_{\text{p}}$ | Reversible electron transfer peak voltammetric current. | [16](#ref-16), [17](#ref-17) |
| [`randles_sevcik_irreversible`][softpotato.analytical.voltammetry.randles_sevcik_irreversible] | $v, \alpha, n_\alpha, c^*, D, A, n, T$ | $I_{\text{p}}$ | Totally irreversible electron transfer peak current. | [18](#ref-18), [19](#ref-19) |
| [`randles_sevcik_quasi`][softpotato.analytical.voltammetry.randles_sevcik_quasi] | $v, k_0, \Lambda, c^*, D, A, \alpha, n, T$ | $I_{\text{p}}$ | Quasi-reversible voltammetric peak current and wave shape. | [19](#ref-19) |
| [`peak_potential_irreversible`][softpotato.analytical.voltammetry.peak_potential_irreversible] | $v, k_0, E^{\circ\prime}, D, \alpha, n_\alpha, T$ | $E_{\text{p}}$ | Peak potential shift for totally irreversible kinetics. | [18](#ref-18), [19](#ref-19) |
| [`nicholson_delta_ep`][softpotato.analytical.voltammetry.nicholson_delta_ep] | $\psi, n$ | $\Delta E_{\text{p}}$ | Peak potential separation from Nicholson parameter. | [20](#ref-20), [21](#ref-21) |
| [`nicholson_psi`][softpotato.analytical.voltammetry.nicholson_psi] | $\Delta E_{\text{p}}, k_0, v, D, n, T$ | $\psi$ | Nicholson dimensionless kinetic parameter calculation. | [20](#ref-20), [21](#ref-21) |
| [`nicholson_rate_constant`][softpotato.analytical.voltammetry.nicholson_rate_constant] | $\Delta E_{\text{p}}, v, D, n, T$ | $k_0$ | Standard heterogeneous rate constant extraction from peak split. | [20](#ref-20), [21](#ref-21) |
| [`matsuda_ayabe_lambda`][softpotato.analytical.voltammetry.matsuda_ayabe_lambda] | $k_0, v, D, n, T$ | $\Lambda$ | Dimensionless Matsuda–Ayabe reversibility parameter. | [19](#ref-19) |
| [`matsuda_ayabe`][softpotato.analytical.voltammetry.matsuda_ayabe] | $k_0, v, D, n, \alpha, T$ | $\Lambda, \text{zone}$ | Reversibility classification (reversible, quasi, irreversible). | [19](#ref-19) |

### Microelectrode Geometries

The [`softpotato.analytical.microelectrodes`][softpotato.analytical.microelectrodes] module provides steady-state and transient
diffusion-limited current expressions for ultramicroelectrode (UME) geometries.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`microdisc_limiting_current`][softpotato.analytical.microelectrodes.microdisc_limiting_current] | $a, n, D, c^*$ | $I_{\text{ss}}$ | Saito steady-state limiting current at an inlaid circular microdisc. | [22](#ref-22), [23](#ref-23) |
| [`microdisc_transient`][softpotato.analytical.microelectrodes.microdisc_transient] | $t, a, n, D, c^*$ | $I(t)$ | Shoup–Szabo full-time chronoamperometric transient at a microdisc. | [24](#ref-24) |
| [`mahon_oldham_transient`][softpotato.analytical.microelectrodes.mahon_oldham_transient] | $t, a, n, D, c^*$ | $I(t)$ | High-precision Mahon–Oldham chronoamperometric transient. | [25](#ref-25) |
| [`microsphere_limiting_current`][softpotato.analytical.microelectrodes.microsphere_limiting_current] | $r, n, D, c^*$ | $I_{\text{ss}}$ | Steady-state spherical diffusion-limited current. | [22](#ref-22) |
| [`microhemisphere_limiting_current`][softpotato.analytical.microelectrodes.microhemisphere_limiting_current] | $r, n, D, c^*$ | $I_{\text{ss}}$ | Steady-state hemispherical diffusion-limited current. | [22](#ref-22) |
| [`microband_limiting_current`][softpotato.analytical.microelectrodes.microband_limiting_current] | $t, w, l, n, D, c^*$ | $I(t)$ | Quasi-steady-state diffusion-limited current at an inlaid microband. | [26](#ref-26), [27](#ref-27) |

### Scanning Electrochemical Microscopy

The [`softpotato.analytical.secm`][softpotato.analytical.secm] module provides normalized approach curve models
and feedback current expressions for scanning electrochemical microscopy.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`secm_limiting_current_infinite`][softpotato.analytical.secm.secm_limiting_current_infinite] | $a, n, D, c^*$ | $I_{\text{T},\infty}$ | Unhindered bulk steady-state current at microelectrode tip. | [22](#ref-22) |
| [`secm_approach_curve`][softpotato.analytical.secm.secm_approach_curve] | $L, R_{\text{g}}, \kappa$ | $I_{\text{T}}^{\text{norm}}$ | Normalized tip approach curve over substrates of arbitrary kinetics. | [28](#ref-28), [29](#ref-29) |
| [`secm_approach_positive_feedback`][softpotato.analytical.secm.secm_approach_positive_feedback] | $L, R_{\text{g}}$ | $I_{\text{T}}^{\text{norm}}$ | Pure positive feedback approach curve over conductive substrate. | [28](#ref-28), [29](#ref-29) |
| [`secm_approach_negative_feedback`][softpotato.analytical.secm.secm_approach_negative_feedback] | $L, R_{\text{g}}$ | $I_{\text{T}}^{\text{norm}}$ | Pure negative feedback approach curve over insulating substrate. | [28](#ref-28), [29](#ref-29) |
| [`secm_tip_current`][softpotato.analytical.secm.secm_tip_current] | $L, R_{\text{g}}, \kappa, a, n, D, c^*$ | $I_{\text{T}}$ | Dimensional steady-state tip current in Amperes. | [28](#ref-28), [29](#ref-29) |

### Parameter Fitting & Regression

The [`softpotato.analytical.fitting`][softpotato.analytical.fitting] module provides non-linear regression and parameter
extraction routines for experimental electrochemical data.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`fit_linear`][softpotato.analytical.fitting.fit_linear] | $x, y$ | $m, c, R^2$ | Ordinary linear regression with complete error diagnostics. |  |
| [`fit_curve`][softpotato.analytical.fitting.fit_curve] | $f(x, p), x, y, p_0, \sigma$ | $p, \sigma_p, R^2$ | Physically bounded non-linear least-squares curve fitting. |  |
| [`fit_cottrell`][softpotato.analytical.fitting.fit_cottrell] | $t, I, n, A, c^*$ | $D, I_{\text{dl}}$ | Chronoamperometric Cottrell regression for diffusivity extraction. | [2](#ref-2) |
| [`fit_levich`][softpotato.analytical.fitting.fit_levich] | $\omega, I, n, A, c^*, \nu$ | $D, B$ | RDE Levich analysis for diffusion coefficient determination. | [9](#ref-9) |
| [`fit_koutecky_levich`][softpotato.analytical.fitting.fit_koutecky_levich] | $\omega, I, n, A, c^*, \nu$ | $I_{\text{k}}, k_0, D$ | Koutecký–Levich analysis for kinetic and transport parameters. | [10](#ref-10) |
| [`fit_tafel`][softpotato.analytical.fitting.fit_tafel] | $\eta, I, n, T$ | $I_0, \alpha, b$ | Tafel plot regression for exchange current and transfer coefficient. | [15](#ref-15) |
| [`fit_randles_sevcik`][softpotato.analytical.fitting.fit_randles_sevcik] | $v, I_{\text{p}}, n, A, c^*, T$ | $D$ | Voltammetric peak current regression for diffusion coefficient. | [16](#ref-16), [17](#ref-17) |
| [`fit_microdisc_transient`][softpotato.analytical.fitting.fit_microdisc_transient] | $t, I, r, c^*, n$ | $D$ | Microdisc transient current fitting to extract diffusivity or radius. | [24](#ref-24), [25](#ref-25) |
| [`fit_secm_approach`][softpotato.analytical.fitting.fit_secm_approach] | $d, I_{\text{norm}}, a$ | $R_{\text{g}}, d_{\text{offset}}$ | SECM approach curve fitting for sheath ratio and distance offset. | [28](#ref-28), [29](#ref-29) |

### Simulation Benchmarking Suite

The [`softpotato.analytical.benchmark`][softpotato.analytical.benchmark] module provides validation harnesses, exact
analytical benchmarks, and spatial/temporal order of convergence testing utilities.

| Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`compute_error_metrics`][softpotato.analytical.benchmark.compute_error_metrics] | $u_{\text{num}}, u_{\text{exact}}$ | $e_{\max}, \text{RMSE}, L_2$ | Standardized error norms between numerical and exact profiles. | [31](#ref-31) |
| [`estimate_convergence_order`][softpotato.analytical.benchmark.estimate_convergence_order] | $\Delta x, e$ | $p, R^2$ | Empirical discretization order calculation via log-log regression. | [31](#ref-31) |
| [`list_benchmark_cases`][softpotato.analytical.benchmark.list_benchmark_cases] | $\varnothing$ | $\text{cases}$ | Names of available standard benchmark cases. |  |
| [`run_benchmark`][softpotato.analytical.benchmark.run_benchmark] | $\text{case}, \text{solver}$ | $\text{result}$ | Unified runner executing a specified validation benchmark. |  |
| [`run_cottrell_benchmark`][softpotato.analytical.benchmark.run_cottrell_benchmark] | $\text{solver}, D, c^*, A$ | $\text{result}$ | Chronoamperometric potential step solver accuracy benchmark. | [2](#ref-2) |
| [`run_fourier_decay_benchmark`][softpotato.analytical.benchmark.run_fourier_decay_benchmark] | $\text{solver}, D, L, t_{\text{end}}$ | $\text{result}$ | 1D Fourier sine diffusion series analytical validation. | [30](#ref-30) |
| [`run_mass_conservation_benchmark`][softpotato.analytical.benchmark.run_mass_conservation_benchmark] | $\text{solver}, D, L, t$ | $\text{result}$ | Zero-flux total mass conservation test across simulation time. |  |
| [`run_benchmark_suite`][softpotato.analytical.benchmark.run_benchmark_suite] | $\text{solvers}, \text{cases}$ | $\text{suite}$ | Comprehensive benchmark suite execution across multiple solvers. |  |
| [`verify_spatial_convergence`][softpotato.analytical.benchmark.verify_spatial_convergence] | $\text{solver}, \Delta x_i, D, L$ | $p_x$ | Spatial grid refinement convergence order verification. | [31](#ref-31) |
| [`verify_temporal_convergence`][softpotato.analytical.benchmark.verify_temporal_convergence] | $\text{solver}, \Delta t_j, D, L$ | $p_t$ | Time-step refinement convergence order verification. | [31](#ref-31) |

## Solver Module

The [`softpotato.solver`][softpotato.solver] subpackage provides numerical partial differential equation (PDE)
solvers and boundary condition primitives for 1D multi-species diffusion and reaction-diffusion systems.

| Class / Function | Inputs | Output | Description | References |
| :--- | :--- | :--- | :--- | :--- |
| [`get_solver`][softpotato.solver.get_solver] | $\text{name}$ | $\text{solver}$ | Factory retrieving an instantiated numerical solver by alias. |  |
| [`list_solvers`][softpotato.solver.list_solvers] | $\varnothing$ | $\text{names}$ | Returns all registered solver names and aliases. |  |
| [`register_solver`][softpotato.solver.register_solver] | $\text{name}, \text{cls}$ | $\varnothing$ | Extends solver registry with a custom solver class. |  |
| [`DirichletBC`][softpotato.solver.DirichletBC] | $c_{\text{bc}}$ | $\text{BC}$ | Fixed concentration boundary condition specification. |  |
| [`NeumannBC`][softpotato.solver.NeumannBC] | $J_{\text{bc}}$ | $\text{BC}$ | Fixed diffusion flux boundary condition specification. |  |
| [`DiffusionProblem`][softpotato.solver.DiffusionProblem] | $x, D_i, \text{BC}_i, c_{i,0}, R_i$ | $\text{Problem}$ | Data container defining multi-species diffusion-reaction problem. |  |
| [`ExplicitFiniteDifference`][softpotato.solver.ExplicitFiniteDifference] | $\Delta t$ | $\text{solver}$ | Forward-Time Central-Space explicit finite difference solver. |  |
| [`ImplicitFiniteDifference`][softpotato.solver.ImplicitFiniteDifference] | $\Delta t$ | $\text{solver}$ | Backward-Time Central-Space unconditionally stable implicit solver. |  |
| [`CrankNicolson`][softpotato.solver.CrankNicolson] | $\Delta t$ | $\text{solver}$ | Second-order in space and time Crank–Nicolson solver. | [32](#ref-32) |
| [`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver] | $\text{method}, \text{rtol}, \text{atol}$ | $\text{solver}$ | Method of Lines solver wrapping SciPy adaptive ODE integrators. | [33](#ref-33) |
| [`solve`][softpotato.solver.BaseSolver.solve] | $\text{Problem}, (t_0, t_1)$ | $c_i(x, t)$ | Integrates problem across time span to compute concentration fields. |  |

## References

1. <a id="ref-1"></a>Tiesinga, E.; Mohr, P. J.; Newell, D. B.; Taylor, B. N. "CODATA recommended values of the fundamental physical constants: 2018." *Reviews of Modern Physics*, **93** (2), 025010 (2021). [doi:10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)
2. <a id="ref-2"></a>Cottrell, F. G. "Der Reststrom bei galvanischer Polarisation, betrachtet als Diffusionsproblem." *Zeitschrift für Physikalische Chemie*, **42**, 385–431 (1903). [doi:10.1515/zpch-1903-4229](https://doi.org/10.1515/zpch-1903-4229)
3. <a id="ref-3"></a>Bard, A. J.; Faulkner, L. R.; White, H. S. *Electrochemical Methods: Fundamentals and Applications*, 3rd ed.; John Wiley & Sons: Hoboken, NJ, 2022.
4. <a id="ref-4"></a>Aoki, K.; Honda, K.; Tokuda, K.; Matsuda, H. "Voltammetry at microcylinder electrodes: Part II. Chronoamperometry." *Journal of Electroanalytical Chemistry*, **186** (1–2), 79–86 (1985). [doi:10.1016/0368-1874(85)85756-3](https://doi.org/10.1016/0368-1874(85)85756-3)
5. <a id="ref-5"></a>Oldham, K. B. "Analytical expressions for the transient current at a microcylinder electrode." *Journal of Electroanalytical Chemistry*, **224** (1–2), 229–232 (1987). [doi:10.1016/0022-0728(87)85093-X](https://doi.org/10.1016/0022-0728(87)85093-X)
6. <a id="ref-6"></a>Anson, F. C. "Application of Potentiostatic Current Integration to the Study of the Adsorption of Cobalt(III)-(Ethylenedinitrilo)tetraacetate on Mercury Electrodes." *Analytical Chemistry*, **36** (5), 932–934 (1964). [doi:10.1021/ac60211a003](https://doi.org/10.1021/ac60211a003)
7. <a id="ref-7"></a>Sand, H. J. S. "On the concentration at the electrodes in a solution, with special reference to the liberation of hydrogen by electrolysis of a mixture of copper sulphate and sulphuric acid." *Philosophical Magazine*, **1** (1), 45–79 (1901). [doi:10.1080/14786440109462590](https://doi.org/10.1080/14786440109462590)
8. <a id="ref-8"></a>Crank, J. *The Mathematics of Diffusion*, 2nd ed.; Oxford University Press: Oxford, UK, 1975.
9. <a id="ref-9"></a>Levich, V. G. *Physicochemical Hydrodynamics*; Prentice-Hall: Englewood Cliffs, NJ, 1962.
10. <a id="ref-10"></a>Koutecký, J.; Levich, V. G. "Calculation of the kinetics of a heterogeneous reaction in a moving liquid." *Zhurnal Fizicheskoi Khimii*, **32**, 1565–1575 (1958).
11. <a id="ref-11"></a>Albery, W. J.; Bruckenstein, S. "Ring-disc electrodes. Part 2.—Theoretical analysis of collection efficiency and shielding." *Transactions of the Faraday Society*, **62**, 1920–1931 (1966). [doi:10.1039/TF9626201920](https://doi.org/10.1039/TF9626201920)
12. <a id="ref-12"></a>Nernst, W. "Die elektromotorische Wirksamkeit der Jonen." *Zeitschrift für Physikalische Chemie*, **4**, 129–181 (1889). [doi:10.1515/zpch-1889-0412](https://doi.org/10.1515/zpch-1889-0412)
13. <a id="ref-13"></a>Butler, J. A. V. "The studies of hydrogen overpotential." *Transactions of the Faraday Society*, **19**, 729–733 (1924). [doi:10.1039/TF9241900729](https://doi.org/10.1039/TF9241900729)
14. <a id="ref-14"></a>Erdey-Grúz, T.; Volmer, M. "Zur Theorie der Wasserstoffüberspannung." *Zeitschrift für Physikalische Chemie*, **150A**, 203–213 (1930). [doi:10.1515/zpch-1930-15020](https://doi.org/10.1515/zpch-1930-15020)
15. <a id="ref-15"></a>Tafel, J. "Über die Polarisation bei kathodischer Wasserstoffentwicklung." *Zeitschrift für Physikalische Chemie*, **50**, 641–712 (1905). [doi:10.1515/zpch-1905-5043](https://doi.org/10.1515/zpch-1905-5043)
16. <a id="ref-16"></a>Randles, J. E. B. "A cathode ray polarograph. Part II. The current-voltage curves." *Transactions of the Faraday Society*, **44**, 327–338 (1948). [doi:10.1039/TF9484400327](https://doi.org/10.1039/TF9484400327)
17. <a id="ref-17"></a>Ševčík, A. "Oscillographic polarography with periodical triangular voltage." *Collection of Czechoslovak Chemical Communications*, **13**, 349–377 (1948). [doi:10.1135/cccc19480349](https://doi.org/10.1135/cccc19480349)
18. <a id="ref-18"></a>Delahay, P. "Theory of Irreversible Polarographic Waves with Fixed and Diffusion-Controlled Surface Concentrations." *Journal of the American Chemical Society*, **75** (5), 1190–1196 (1953). [doi:10.1021/ja01101a052](https://doi.org/10.1021/ja01101a052)
19. <a id="ref-19"></a>Matsuda, H.; Ayabe, Y. "Zur Theorie der Randles-Sevčikschen Kathodenstrahl-Polarographie." *Zeitschrift für Elektrochemie*, **59** (6), 494–503 (1955). [doi:10.1002/bbpc.19550590605](https://doi.org/10.1002/bbpc.19550590605)
20. <a id="ref-20"></a>Nicholson, R. S. "Theory and Application of Cyclic Voltammetry for Measurement of Electrode Reaction Kinetics." *Analytical Chemistry*, **37** (11), 1351–1355 (1965). [doi:10.1021/ac60230a016](https://doi.org/10.1021/ac60230a016)
21. <a id="ref-21"></a>Lavagnini, I.; Antiochia, R.; Magno, F. "An Extended Method for the Practical Evaluation of the Standard Rate Constant from Cyclic Voltammetric Data." *Electroanalysis*, **16** (6), 505–506 (2004). [doi:10.1002/elan.200302851](https://doi.org/10.1002/elan.200302851)
22. <a id="ref-22"></a>Saito, Y. "A theoretical study on the diffusion current at the stationary spherical and disc electrodes." *Review of Polarography*, **15** (6), 177–187 (1968). [doi:10.5189/revpolarography.15.177](https://doi.org/10.5189/revpolarography.15.177)
23. <a id="ref-23"></a>Newman, J. "Resistance for Flow of Current to a Disk." *Journal of the Electrochemical Society*, **113** (5), 501–502 (1966). [doi:10.1149/1.2424003](https://doi.org/10.1149/1.2424003)
24. <a id="ref-24"></a>Shoup, D.; Szabo, A. "Chronoamperometric current at finite disk electrodes." *Journal of Electroanalytical Chemistry*, **140** (2), 237–245 (1982). [doi:10.1016/0368-1874(82)85294-1](https://doi.org/10.1016/0368-1874(82)85294-1)
25. <a id="ref-25"></a>Mahon, P. J.; Oldham, K. B. "An analytical expression for the transient current at a microdisk electrode." *Analytical Chemistry*, **77** (18), 6100–6101 (2005). [doi:10.1021/ac050965i](https://doi.org/10.1021/ac050965i)
26. <a id="ref-26"></a>Szabo, A.; Cope, D. K.; Tallman, D. E.; Kovach, P. M.; Wightman, R. M. "Chronoamperometric current at hemicylinder and band microelectrodes: Theory and experiment." *Journal of Electroanalytical Chemistry*, **217** (2), 417–423 (1987). [doi:10.1016/0022-0728(87)80234-2](https://doi.org/10.1016/0022-0728(87)80234-2)
27. <a id="ref-27"></a>Aoki, K.; Tokuda, K.; Matsuda, H. "Derivation of an approximate equation for chronoamperometric curves at microband electrodes and its experimental verification." *Journal of Electroanalytical Chemistry*, **225** (1–2), 19–32 (1987). [doi:10.1016/0022-0728(87)80003-3](https://doi.org/10.1016/0022-0728(87)80003-3)
28. <a id="ref-28"></a>Cornut, R.; Lefrou, C. "New Analytical Approximations for SECM with an Inlaid Microdisk Tip: Pure Positive Feedback and Insulating Substrate Feedback." *Journal of Electroanalytical Chemistry*, **604** (2), 91–100 (2007). [doi:10.1016/j.jelechem.2007.03.003](https://doi.org/10.1016/j.jelechem.2007.03.003)
29. <a id="ref-29"></a>Lefrou, C.; Cornut, R. "Analytical Approximations for Quantitative Approach Curves in SECM with a Flat and Non-Flat Inlaid Disk Tip under Finite Kinetics." *ChemPhysChem*, **9** (15), 2154–2160 (2008). [doi:10.1002/cphc.200800366](https://doi.org/10.1002/cphc.200800366)
30. <a id="ref-30"></a>Fourier, J. *Théorie analytique de la chaleur*; Firmin Didot: Paris, France, 1822.
31. <a id="ref-31"></a>Roache, P. J. *Verification and Validation in Computational Science and Engineering*; Hermosa Publishers: Albuquerque, NM, 1998.
32. <a id="ref-32"></a>Crank, J.; Nicolson, P. "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Mathematical Proceedings of the Cambridge Philosophical Society*, **43** (1), 50–67 (1947). [doi:10.1017/S0305004100023197](https://doi.org/10.1017/S0305004100023197)
33. <a id="ref-33"></a>Hairer, E.; Wanner, G. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed.; Springer-Verlag: Berlin, Germany, 1996. [doi:10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)
