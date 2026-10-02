API Reference
=============

Welcome to the **Soft Potato** API Reference. The toolkit is organized into modular
subpackages and modules for physical constants, closed-form analytical electrochemistry,
and numerical PDE solvers.

.. toctree::
   :hidden:
   :maxdepth: 2

   Constants Module <constants>
   Analytical Module <analytical/index>
   Solver Module <solver>


.. rubric:: Constants Module
   :class: module-heading

The :mod:`softpotato.constants` module provides standardized physical and electrochemical
constants defined according to CODATA 2018 / 2019 SI conventions [1].

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Constant / Identifier
     - Symbol
     - Value & Units
   * - :data:`~softpotato.constants.FARADAY` (``F``)
     - :math:`F`
     - :math:`96485.332 \,\text{C}\cdot\text{mol}^{-1}`
   * - :data:`~softpotato.constants.GAS_CONSTANT` (``R``)
     - :math:`R`
     - :math:`8.31446\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}`
   * - :data:`~softpotato.constants.STANDARD_TEMPERATURE` (``T_STD``)
     - :math:`T^\circ`
     - :math:`298.15\,\text{K}` (:math:`25\,^\circ\text{C}`)
   * - :data:`~softpotato.constants.AVOGADRO`
     - :math:`N_{\text{A}}`
     - :math:`6.02214 \times 10^{23}\,\text{mol}^{-1}`
   * - :data:`~softpotato.constants.BOLTZMANN`
     - :math:`k_{\text{B}}`
     - :math:`1.38065 \times 10^{-23}\,\text{J}\cdot\text{K}^{-1}`
   * - :data:`~softpotato.constants.ELEMENTARY_CHARGE`
     - :math:`e`
     - :math:`1.60218 \times 10^{-19}\,\text{C}`
   * - :data:`~softpotato.constants.VACUUM_PERMITTIVITY`
     - :math:`\varepsilon_0`
     - :math:`8.85419 \times 10^{-12}\,\text{F}\cdot\text{m}^{-1}`


.. rubric:: Analytical Module
   :class: module-heading

The :mod:`softpotato.analytical` subpackage provides closed-form analytical solutions,
asymptotic approximations, and diagnostic formulations across transient step techniques,
hydrodynamics, interfacial kinetics, voltammetry, microelectrodes, SECM, parameter fitting,
and numerical benchmark suites.

.. rubric:: Transient Step Techniques
   :class: submodule-heading

The :mod:`softpotato.analytical.step` module provides exact solutions for potentiostatic
and galvanostatic step experiments, including Cottrell, Anson, and Sand formulations.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.step.cottrell`
     - :math:`t, n, D, c^*, A`
     - :math:`I(t)`
     - Planar chronoamperometric diffusion-limited current transient.
     - [2]_
   * - :func:`~softpotato.analytical.step.cottrell_step`
     - :math:`t, \tau, n, D, c^*, A, D_{\text{Red}}`
     - :math:`I(t)`
     - Double potential step chronoamperometry reversal current.
     - [3]_
   * - :func:`~softpotato.analytical.step.cottrell_spherical`
     - :math:`t, r, n, D, c^*, A`
     - :math:`I(t)`
     - Chronoamperometric current at a spherical electrode.
     - [3]_
   * - :func:`~softpotato.analytical.step.cottrell_cylinder`
     - :math:`t, r_0, n, D, c^*, l, A`
     - :math:`I(t)`
     - Chronoamperometric current at a cylindrical wire/fiber electrode.
     - [4]_, [5]_
   * - :func:`~softpotato.analytical.step.anson`
     - :math:`t, n, D, c^*, A, Q_{\text{dl}}, \Gamma, \tau`
     - :math:`Q(t)`
     - Cumulative charge transient for chronocoulometry.
     - [6]_
   * - :func:`~softpotato.analytical.step.sand`
     - :math:`t, I, n, D, c^*, A`
     - :math:`c(0, t)`
     - Surface reactant concentration transient under constant current.
     - [7]_
   * - :func:`~softpotato.analytical.step.sand_transition_time`
     - :math:`I, n, D, c^*, A`
     - :math:`\tau`
     - Transition time for galvanostatic chronopotentiometry.
     - [7]_
   * - :func:`~softpotato.analytical.step.sand_potential`
     - :math:`t, \tau, E_{1/2}, n, T`
     - :math:`E(t)`
     - Potential-time response curve for chronopotentiometry.
     - [7]_
   * - :func:`~softpotato.analytical.step.step_concentration_profile`
     - :math:`x, t, D, c^*`
     - :math:`c(x, t)`
     - Exact spatial concentration profile following a potential step.
     - [8]_
   * - :func:`~softpotato.analytical.step.step_flux_profile`
     - :math:`x, t, D, c^*`
     - :math:`J(x, t)`
     - Exact spatial diffusion flux profile following a potential step.
     - [8]_

.. rubric:: Hydrodynamics & Convection Equations
   :class: submodule-heading

The :mod:`softpotato.analytical.hydrodynamics` module provides models for rotating disk
(RDE) and rotating ring-disk (RRDE) convective electrochemical systems.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.hydrodynamics.levich`
     - :math:`\omega, n, D, c^*, A, \nu`
     - :math:`I_{\text{L}}`
     - Convective mass-transfer limiting current at an RDE.
     - [9]_
   * - :func:`~softpotato.analytical.hydrodynamics.levich_constant`
     - :math:`n, D, c^*, A, \nu`
     - :math:`B`
     - Theoretical Levich slope parameter for RDE rotation series.
     - [9]_
   * - :func:`~softpotato.analytical.hydrodynamics.koutecky_levich`
     - :math:`\omega, I_{\text{k}}, k, I_{\text{lim}}, n, D, c^*, A, \nu`
     - :math:`I`
     - Current under mixed kinetic and convective mass-transfer control.
     - [10]_
   * - :func:`~softpotato.analytical.hydrodynamics.koutecky_levich_analysis`
     - :math:`\omega, I, A, c^*, n, \nu`
     - :math:`k, B`
     - Kinetic rate constant and Levich slope extraction from RDE data.
     - [10]_
   * - :func:`~softpotato.analytical.hydrodynamics.nernst_diffusion_layer`
     - :math:`\omega, D, \nu`
     - :math:`\delta`
     - Steady-state stagnant diffusion layer thickness at an RDE.
     - [9]_
   * - :func:`~softpotato.analytical.hydrodynamics.collection_efficiency`
     - :math:`r_1, r_2, r_3`
     - :math:`N`
     - Theoretical collection efficiency for an RRDE geometry.
     - [11]_
   * - :func:`~softpotato.analytical.hydrodynamics.shielding_factor`
     - :math:`r_1, r_2, r_3, N`
     - :math:`S`
     - Theoretical geometric shielding factor for an RRDE.
     - [11]_
   * - :func:`~softpotato.analytical.hydrodynamics.ring_collection_current`
     - :math:`I_{\text{disk}}, N, n_{\text{ring}}, n_{\text{disk}}`
     - :math:`I_{\text{ring}}`
     - Ring electrode current from disk product collection.
     - [11]_
   * - :func:`~softpotato.analytical.hydrodynamics.ring_limiting_current`
     - :math:`\omega, r_1, r_2, r_3, n, D, c^*, \nu`
     - :math:`I_{\text{R,lim}}`
     - Convective mass-transfer limiting current at an RRDE ring.
     - [11]_
   * - :func:`~softpotato.analytical.hydrodynamics.rotating_ring_disk`
     - :math:`r_1, r_2, r_3, \omega, n, D, c^*, \nu`
     - :math:`N, S, I_{\text{D,lim}}, I_{\text{R,lim}}`
     - Comprehensive collection and limiting current metrics for RRDE.
     - [11]_
   * - :func:`~softpotato.analytical.hydrodynamics.rad_s_to_rpm`
     - :math:`\omega`
     - :math:`f_{\text{RPM}}`
     - Angular velocity to rotational speed conversion.
     - 
   * - :func:`~softpotato.analytical.hydrodynamics.rpm_to_rad_s`
     - :math:`f_{\text{RPM}}`
     - :math:`\omega`
     - Rotational speed to angular velocity conversion.
     - 

.. rubric:: Thermodynamics & Interfacial Kinetics
   :class: submodule-heading

The :mod:`softpotato.analytical.kinetics` module provides thermodynamic equilibrium potentials
and electrode kinetic models, including Nernst, Butler–Volmer, and Tafel formulations.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.kinetics.nernst`
     - :math:`E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T`
     - :math:`E`
     - Reversible equilibrium reduction potential.
     - [12]_
   * - :func:`~softpotato.analytical.kinetics.nernst_potential`
     - :math:`E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T`
     - :math:`E`
     - Nernst equilibrium electrode potential calculation.
     - [12]_
   * - :func:`~softpotato.analytical.kinetics.nernst_ratio`
     - :math:`E, E^\circ, n, T`
     - :math:`c_{\text{Ox}} / c_{\text{Red}}`
     - Equilibrium concentration ratio from applied potential.
     - [12]_
   * - :func:`~softpotato.analytical.kinetics.nernst_equilibrium_concentrations`
     - :math:`E, E^\circ, c_{\text{total}}, n, T`
     - :math:`c_{\text{Ox}}, c_{\text{Red}}`
     - Individual equilibrium species concentrations from total pool.
     - [12]_
   * - :func:`~softpotato.analytical.kinetics.butler_volmer`
     - :math:`\eta, I_0, \alpha, n, T`
     - :math:`I`
     - Faradaic current from activation overpotential.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.butler_volmer_current_density`
     - :math:`\eta, j_0, \alpha, n, T`
     - :math:`j`
     - Faradaic current density from activation overpotential.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.butler_volmer_linear`
     - :math:`\eta, I_0, n, T`
     - :math:`I`
     - Low-overpotential linearized Butler–Volmer current.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.charge_transfer_resistance`
     - :math:`I_0, n, T`
     - :math:`R_{\text{ct}}`
     - Charge-transfer resistance at equilibrium potential.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.exchange_current`
     - :math:`k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, A, \alpha, n`
     - :math:`I_0`
     - Standard exchange current from rate constant and concentrations.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.exchange_current_density`
     - :math:`k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, \alpha, n`
     - :math:`j_0`
     - Exchange current density from rate constant and concentrations.
     - [13]_, [14]_
   * - :func:`~softpotato.analytical.kinetics.tafel`
     - :math:`\eta, I_0, \alpha, n, T`
     - :math:`I`
     - High-overpotential unidirectional Tafel current approximation.
     - [15]_
   * - :func:`~softpotato.analytical.kinetics.tafel_overpotential`
     - :math:`I, I_0, \alpha, n, T`
     - :math:`\eta`
     - Activation overpotential calculated from current.
     - [15]_
   * - :func:`~softpotato.analytical.kinetics.tafel_slope`
     - :math:`\alpha, n, T`
     - :math:`b`
     - Theoretical Tafel slope in Volts per decade.
     - [15]_
   * - :func:`~softpotato.analytical.kinetics.tafel_analysis`
     - :math:`\eta, I, A, c^*, n, T`
     - :math:`I_0, \alpha, b`
     - Semi-logarithmic Tafel regression for kinetic parameter extraction.
     - [15]_

.. rubric:: Voltammetry & Kinetic Diagnostics
   :class: submodule-heading

The :mod:`softpotato.analytical.voltammetry` module provides peak current expressions and
reversibility diagnostics for cyclic and linear sweep voltammetry.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik`
     - :math:`v, c^*, D, A, n, T`
     - :math:`I_{\text{p}}`
     - Reversible electron transfer peak voltammetric current.
     - [16]_, [17]_
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik_irreversible`
     - :math:`v, \alpha, n_\alpha, c^*, D, A, n, T`
     - :math:`I_{\text{p}}`
     - Totally irreversible electron transfer peak current.
     - [18]_, [19]_
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik_quasi`
     - :math:`v, k_0, \Lambda, c^*, D, A, \alpha, n, T`
     - :math:`I_{\text{p}}`
     - Quasi-reversible voltammetric peak current and wave shape.
     - [19]_
   * - :func:`~softpotato.analytical.voltammetry.peak_potential_irreversible`
     - :math:`v, k_0, E^{\circ\prime}, D, \alpha, n_\alpha, T`
     - :math:`E_{\text{p}}`
     - Peak potential shift for totally irreversible kinetics.
     - [18]_, [19]_
   * - :func:`~softpotato.analytical.voltammetry.nicholson_delta_ep`
     - :math:`\psi, n`
     - :math:`\Delta E_{\text{p}}`
     - Peak potential separation from Nicholson parameter.
     - [20]_, [21]_
   * - :func:`~softpotato.analytical.voltammetry.nicholson_psi`
     - :math:`\Delta E_{\text{p}}, k_0, v, D, n, T`
     - :math:`\psi`
     - Nicholson dimensionless kinetic parameter calculation.
     - [20]_, [21]_
   * - :func:`~softpotato.analytical.voltammetry.nicholson_rate_constant`
     - :math:`\Delta E_{\text{p}}, v, D, n, T`
     - :math:`k_0`
     - Standard heterogeneous rate constant extraction from peak split.
     - [20]_, [21]_
   * - :func:`~softpotato.analytical.voltammetry.matsuda_ayabe_lambda`
     - :math:`k_0, v, D, n, T`
     - :math:`\Lambda`
     - Dimensionless Matsuda–Ayabe reversibility parameter.
     - [19]_
   * - :func:`~softpotato.analytical.voltammetry.matsuda_ayabe`
     - :math:`k_0, v, D, n, \alpha, T`
     - :math:`\Lambda, \text{zone}`
     - Reversibility classification (reversible, quasi, irreversible).
     - [19]_

.. rubric:: Microelectrode Geometries
   :class: submodule-heading

The :mod:`softpotato.analytical.microelectrodes` module provides steady-state and transient
diffusion-limited current expressions for ultramicroelectrode (UME) geometries.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.microelectrodes.microdisc_limiting_current`
     - :math:`a, n, D, c^*`
     - :math:`I_{\text{ss}}`
     - Saito steady-state limiting current at an inlaid circular microdisc.
     - [22]_, [23]_
   * - :func:`~softpotato.analytical.microelectrodes.microdisc_transient`
     - :math:`t, a, n, D, c^*`
     - :math:`I(t)`
     - Shoup–Szabo full-time chronoamperometric transient at a microdisc.
     - [24]_
   * - :func:`~softpotato.analytical.microelectrodes.mahon_oldham_transient`
     - :math:`t, a, n, D, c^*`
     - :math:`I(t)`
     - High-precision Mahon–Oldham chronoamperometric transient.
     - [25]_
   * - :func:`~softpotato.analytical.microelectrodes.microsphere_limiting_current`
     - :math:`r, n, D, c^*`
     - :math:`I_{\text{ss}}`
     - Steady-state spherical diffusion-limited current.
     - [22]_
   * - :func:`~softpotato.analytical.microelectrodes.microhemisphere_limiting_current`
     - :math:`r, n, D, c^*`
     - :math:`I_{\text{ss}}`
     - Steady-state hemispherical diffusion-limited current.
     - [22]_
   * - :func:`~softpotato.analytical.microelectrodes.microband_limiting_current`
     - :math:`t, w, l, n, D, c^*`
     - :math:`I(t)`
     - Quasi-steady-state diffusion-limited current at an inlaid microband.
     - [26]_, [27]_

.. rubric:: Scanning Electrochemical Microscopy
   :class: submodule-heading

The :mod:`softpotato.analytical.secm` module provides normalized approach curve models
and feedback current expressions for scanning electrochemical microscopy.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.secm.secm_limiting_current_infinite`
     - :math:`a, n, D, c^*`
     - :math:`I_{\text{T},\infty}`
     - Unhindered bulk steady-state current at microelectrode tip.
     - [22]_
   * - :func:`~softpotato.analytical.secm.secm_approach_curve`
     - :math:`L, R_{\text{g}}, \kappa`
     - :math:`I_{\text{T}}^{\text{norm}}`
     - Normalized tip approach curve over substrates of arbitrary kinetics.
     - [28]_, [29]_
   * - :func:`~softpotato.analytical.secm.secm_approach_positive_feedback`
     - :math:`L, R_{\text{g}}`
     - :math:`I_{\text{T}}^{\text{norm}}`
     - Pure positive feedback approach curve over conductive substrate.
     - [28]_, [29]_
   * - :func:`~softpotato.analytical.secm.secm_approach_negative_feedback`
     - :math:`L, R_{\text{g}}`
     - :math:`I_{\text{T}}^{\text{norm}}`
     - Pure negative feedback approach curve over insulating substrate.
     - [28]_, [29]_
   * - :func:`~softpotato.analytical.secm.secm_tip_current`
     - :math:`L, R_{\text{g}}, \kappa, a, n, D, c^*`
     - :math:`I_{\text{T}}`
     - Dimensional steady-state tip current in Amperes.
     - [28]_, [29]_

.. rubric:: Parameter Fitting & Regression
   :class: submodule-heading

The :mod:`softpotato.analytical.fitting` module provides non-linear regression and parameter
extraction routines for experimental electrochemical data.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.fitting.fit_linear`
     - :math:`x, y`
     - :math:`m, c, R^2`
     - Ordinary linear regression with complete error diagnostics.
     - 
   * - :func:`~softpotato.analytical.fitting.fit_curve`
     - :math:`f(x, p), x, y, p_0, \sigma`
     - :math:`p, \sigma_p, R^2`
     - Physically bounded non-linear least-squares curve fitting.
     - 
   * - :func:`~softpotato.analytical.fitting.fit_cottrell`
     - :math:`t, I, n, A, c^*`
     - :math:`D, I_{\text{dl}}`
     - Chronoamperometric Cottrell regression for diffusivity extraction.
     - [2]_
   * - :func:`~softpotato.analytical.fitting.fit_levich`
     - :math:`\omega, I, n, A, c^*, \nu`
     - :math:`D, B`
     - RDE Levich analysis for diffusion coefficient determination.
     - [9]_
   * - :func:`~softpotato.analytical.fitting.fit_koutecky_levich`
     - :math:`\omega, I, n, A, c^*, \nu`
     - :math:`I_{\text{k}}, k_0, D`
     - Koutecký–Levich analysis for kinetic and transport parameters.
     - [10]_
   * - :func:`~softpotato.analytical.fitting.fit_tafel`
     - :math:`\eta, I, n, T`
     - :math:`I_0, \alpha, b`
     - Tafel plot regression for exchange current and transfer coefficient.
     - [15]_
   * - :func:`~softpotato.analytical.fitting.fit_randles_sevcik`
     - :math:`v, I_{\text{p}}, n, A, c^*, T`
     - :math:`D`
     - Voltammetric peak current regression for diffusion coefficient.
     - [16]_, [17]_
   * - :func:`~softpotato.analytical.fitting.fit_microdisc_transient`
     - :math:`t, I, r, c^*, n`
     - :math:`D`
     - Microdisc transient current fitting to extract diffusivity or radius.
     - [24]_, [25]_
   * - :func:`~softpotato.analytical.fitting.fit_secm_approach`
     - :math:`d, I_{\text{norm}}, a`
     - :math:`R_{\text{g}}, d_{\text{offset}}`
     - SECM approach curve fitting for sheath ratio and distance offset.
     - [28]_, [29]_

.. rubric:: Simulation Benchmarking Suite
   :class: submodule-heading

The :mod:`softpotato.analytical.benchmark` module provides validation harnesses, exact
analytical benchmarks, and spatial/temporal order of convergence testing utilities.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.analytical.benchmark.compute_error_metrics`
     - :math:`u_{\text{num}}, u_{\text{exact}}`
     - :math:`e_{\max}, \text{RMSE}, L_2`
     - Standardized error norms between numerical and exact profiles.
     - [31]_
   * - :func:`~softpotato.analytical.benchmark.estimate_convergence_order`
     - :math:`\Delta x, e`
     - :math:`p, R^2`
     - Empirical discretization order calculation via log-log regression.
     - [31]_
   * - :func:`~softpotato.analytical.benchmark.list_benchmark_cases`
     - :math:`\varnothing`
     - :math:`\text{cases}`
     - Names of available standard benchmark cases.
     - 
   * - :func:`~softpotato.analytical.benchmark.run_benchmark`
     - :math:`\text{case}, \text{solver}`
     - :math:`\text{result}`
     - Unified runner executing a specified validation benchmark.
     - 
   * - :func:`~softpotato.analytical.benchmark.run_cottrell_benchmark`
     - :math:`\text{solver}, D, c^*, A`
     - :math:`\text{result}`
     - Chronoamperometric potential step solver accuracy benchmark.
     - [2]_
   * - :func:`~softpotato.analytical.benchmark.run_fourier_decay_benchmark`
     - :math:`\text{solver}, D, L, t_{\text{end}}`
     - :math:`\text{result}`
     - 1D Fourier sine diffusion series analytical validation.
     - [30]_
   * - :func:`~softpotato.analytical.benchmark.run_mass_conservation_benchmark`
     - :math:`\text{solver}, D, L, t`
     - :math:`\text{result}`
     - Zero-flux total mass conservation test across simulation time.
     - 
   * - :func:`~softpotato.analytical.benchmark.run_benchmark_suite`
     - :math:`\text{solvers}, \text{cases}`
     - :math:`\text{suite}`
     - Comprehensive benchmark suite execution across multiple solvers.
     - 
   * - :func:`~softpotato.analytical.benchmark.verify_spatial_convergence`
     - :math:`\text{solver}, \Delta x_i, D, L`
     - :math:`p_x`
     - Spatial grid refinement convergence order verification.
     - [31]_
   * - :func:`~softpotato.analytical.benchmark.verify_temporal_convergence`
     - :math:`\text{solver}, \Delta t_j, D, L`
     - :math:`p_t`
     - Time-step refinement convergence order verification.
     - [31]_


.. rubric:: Solver Module
   :class: module-heading

The :mod:`softpotato.solver` subpackage provides numerical partial differential equation (PDE)
solvers and boundary condition primitives for 1D multi-species diffusion and reaction-diffusion systems.

.. list-table::
   :header-rows: 1
   :widths: 22 17 11 40 10

   * - Class / Function
     - Inputs
     - Output
     - Description
     - References
   * - :func:`~softpotato.solver.get_solver`
     - :math:`\text{name}`
     - :math:`\text{solver}`
     - Factory retrieving an instantiated numerical solver by alias.
     - 
   * - :func:`~softpotato.solver.list_solvers`
     - :math:`\varnothing`
     - :math:`\text{names}`
     - Returns all registered solver names and aliases.
     - 
   * - :func:`~softpotato.solver.register_solver`
     - :math:`\text{name}, \text{cls}`
     - :math:`\varnothing`
     - Extends solver registry with a custom solver class.
     - 
   * - :class:`~softpotato.solver.DirichletBC`
     - :math:`c_{\text{bc}}`
     - :math:`\text{BC}`
     - Fixed concentration boundary condition specification.
     - 
   * - :class:`~softpotato.solver.NeumannBC`
     - :math:`J_{\text{bc}}`
     - :math:`\text{BC}`
     - Fixed diffusion flux boundary condition specification.
     - 
   * - :class:`~softpotato.solver.DiffusionProblem`
     - :math:`x, D_i, \text{BC}_i, c_{i,0}, R_i`
     - :math:`\text{Problem}`
     - Data container defining multi-species diffusion-reaction problem.
     - 
   * - :class:`~softpotato.solver.ExplicitFiniteDifference`
     - :math:`\Delta t`
     - :math:`\text{solver}`
     - Forward-Time Central-Space explicit finite difference solver.
     - 
   * - :class:`~softpotato.solver.ImplicitFiniteDifference`
     - :math:`\Delta t`
     - :math:`\text{solver}`
     - Backward-Time Central-Space unconditionally stable implicit solver.
     - 
   * - :class:`~softpotato.solver.CrankNicolson`
     - :math:`\Delta t`
     - :math:`\text{solver}`
     - Second-order in space and time Crank–Nicolson solver.
     - [32]_
   * - :class:`~softpotato.solver.ScipyIVPSolver`
     - :math:`\text{method}, \text{rtol}, \text{atol}`
     - :math:`\text{solver}`
     - Method of Lines solver wrapping SciPy adaptive ODE integrators.
     - [33]_
   * - :meth:`~softpotato.solver.BaseSolver.solve`
     - :math:`\text{Problem}, (t_0, t_1)`
     - :math:`c_i(x, t)`
     - Integrates problem across time span to compute concentration fields.
     - 


.. rubric:: References
   :class: module-heading

.. [1] Tiesinga, E.; Mohr, P. J.; Newell, D. B.; Taylor, B. N. "CODATA recommended values of the fundamental physical constants: 2018." *Reviews of Modern Physics*, **93** (2), 025010 (2021). `doi:10.1103/RevModPhys.93.025010 <https://doi.org/10.1103/RevModPhys.93.025010>`_
.. [2] Cottrell, F. G. "Der Reststrom bei galvanischer Polarisation, betrachtet als Diffusionsproblem." *Zeitschrift für Physikalische Chemie*, **42**, 385–431 (1903). `doi:10.1515/zpch-1903-4229 <https://doi.org/10.1515/zpch-1903-4229>`_
.. [3] Bard, A. J.; Faulkner, L. R.; White, H. S. *Electrochemical Methods: Fundamentals and Applications*, 3rd ed.; John Wiley & Sons: Hoboken, NJ, 2022.
.. [4] Aoki, K.; Honda, K.; Tokuda, K.; Matsuda, H. "Voltammetry at microcylinder electrodes: Part II. Chronoamperometry." *Journal of Electroanalytical Chemistry*, **186** (1–2), 79–86 (1985). `doi:10.1016/0368-1874(85)85756-3 <https://doi.org/10.1016/0368-1874(85)85756-3>`_
.. [5] Oldham, K. B. "Analytical expressions for the transient current at a microcylinder electrode." *Journal of Electroanalytical Chemistry*, **224** (1–2), 229–232 (1987). `doi:10.1016/0022-0728(87)85093-X <https://doi.org/10.1016/0022-0728(87)85093-X>`_
.. [6] Anson, F. C. "Application of Potentiostatic Current Integration to the Study of the Adsorption of Cobalt(III)-(Ethylenedinitrilo)tetraacetate on Mercury Electrodes." *Analytical Chemistry*, **36** (5), 932–934 (1964). `doi:10.1021/ac60211a003 <https://doi.org/10.1021/ac60211a003>`_
.. [7] Sand, H. J. S. "On the concentration at the electrodes in a solution, with special reference to the liberation of hydrogen by electrolysis of a mixture of copper sulphate and sulphuric acid." *Philosophical Magazine*, **1** (1), 45–79 (1901). `doi:10.1080/14786440109462590 <https://doi.org/10.1080/14786440109462590>`_
.. [8] Crank, J. *The Mathematics of Diffusion*, 2nd ed.; Oxford University Press: Oxford, UK, 1975.
.. [9] Levich, V. G. *Physicochemical Hydrodynamics*; Prentice-Hall: Englewood Cliffs, NJ, 1962.
.. [10] Koutecký, J.; Levich, V. G. "Calculation of the kinetics of a heterogeneous reaction in a moving liquid." *Zhurnal Fizicheskoi Khimii*, **32**, 1565–1575 (1958).
.. [11] Albery, W. J.; Bruckenstein, S. "Ring-disc electrodes. Part 2.—Theoretical analysis of collection efficiency and shielding." *Transactions of the Faraday Society*, **62**, 1920–1931 (1966). `doi:10.1039/TF9626201920 <https://doi.org/10.1039/TF9626201920>`_
.. [12] Nernst, W. "Die elektromotorische Wirksamkeit der Jonen." *Zeitschrift für Physikalische Chemie*, **4**, 129–181 (1889). `doi:10.1515/zpch-1889-0412 <https://doi.org/10.1515/zpch-1889-0412>`_
.. [13] Butler, J. A. V. "The studies of hydrogen overpotential." *Transactions of the Faraday Society*, **19**, 729–733 (1924). `doi:10.1039/TF9241900729 <https://doi.org/10.1039/TF9241900729>`_
.. [14] Erdey-Grúz, T.; Volmer, M. "Zur Theorie der Wasserstoffüberspannung." *Zeitschrift für Physikalische Chemie*, **150A**, 203–213 (1930). `doi:10.1515/zpch-1930-15020 <https://doi.org/10.1515/zpch-1930-15020>`_
.. [15] Tafel, J. "Über die Polarisation bei kathodischer Wasserstoffentwicklung." *Zeitschrift für Physikalische Chemie*, **50**, 641–712 (1905). `doi:10.1515/zpch-1905-5043 <https://doi.org/10.1515/zpch-1905-5043>`_
.. [16] Randles, J. E. B. "A cathode ray polarograph. Part II. The current-voltage curves." *Transactions of the Faraday Society*, **44**, 327–338 (1948). `doi:10.1039/TF9484400327 <https://doi.org/10.1039/TF9484400327>`_
.. [17] Ševčík, A. "Oscillographic polarography with periodical triangular voltage." *Collection of Czechoslovak Chemical Communications*, **13**, 349–377 (1948). `doi:10.1135/cccc19480349 <https://doi.org/10.1135/cccc19480349>`_
.. [18] Delahay, P. "Theory of Irreversible Polarographic Waves with Fixed and Diffusion-Controlled Surface Concentrations." *Journal of the American Chemical Society*, **75** (5), 1190–1196 (1953). `doi:10.1021/ja01101a052 <https://doi.org/10.1021/ja01101a052>`_
.. [19] Matsuda, H.; Ayabe, Y. "Zur Theorie der Randles-Sevčikschen Kathodenstrahl-Polarographie." *Zeitschrift für Elektrochemie*, **59** (6), 494–503 (1955). `doi:10.1002/bbpc.19550590605 <https://doi.org/10.1002/bbpc.19550590605>`_
.. [20] Nicholson, R. S. "Theory and Application of Cyclic Voltammetry for Measurement of Electrode Reaction Kinetics." *Analytical Chemistry*, **37** (11), 1351–1355 (1965). `doi:10.1021/ac60230a016 <https://doi.org/10.1021/ac60230a016>`_
.. [21] Lavagnini, I.; Antiochia, R.; Magno, F. "An Extended Method for the Practical Evaluation of the Standard Rate Constant from Cyclic Voltammetric Data." *Electroanalysis*, **16** (6), 505–506 (2004). `doi:10.1002/elan.200302851 <https://doi.org/10.1002/elan.200302851>`_
.. [22] Saito, Y. "A theoretical study on the diffusion current at the stationary spherical and disc electrodes." *Review of Polarography*, **15** (6), 177–187 (1968). `doi:10.5189/revpolarography.15.177 <https://doi.org/10.5189/revpolarography.15.177>`_
.. [23] Newman, J. "Resistance for Flow of Current to a Disk." *Journal of the Electrochemical Society*, **113** (5), 501–502 (1966). `doi:10.1149/1.2424003 <https://doi.org/10.1149/1.2424003>`_
.. [24] Shoup, D.; Szabo, A. "Chronoamperometric current at finite disk electrodes." *Journal of Electroanalytical Chemistry*, **140** (2), 237–245 (1982). `doi:10.1016/0368-1874(82)85294-1 <https://doi.org/10.1016/0368-1874(82)85294-1>`_
.. [25] Mahon, P. J.; Oldham, K. B. "An analytical expression for the transient current at a microdisk electrode." *Analytical Chemistry*, **77** (18), 6100–6101 (2005). `doi:10.1021/ac050965i <https://doi.org/10.1021/ac050965i>`_
.. [26] Szabo, A.; Cope, D. K.; Tallman, D. E.; Kovach, P. M.; Wightman, R. M. "Chronoamperometric current at hemicylinder and band microelectrodes: Theory and experiment." *Journal of Electroanalytical Chemistry*, **217** (2), 417–423 (1987). `doi:10.1016/0022-0728(87)80234-2 <https://doi.org/10.1016/0022-0728(87)80234-2>`_
.. [27] Aoki, K.; Tokuda, K.; Matsuda, H. "Derivation of an approximate equation for chronoamperometric curves at microband electrodes and its experimental verification." *Journal of Electroanalytical Chemistry*, **225** (1–2), 19–32 (1987). `doi:10.1016/0022-0728(87)80003-3 <https://doi.org/10.1016/0022-0728(87)80003-3>`_
.. [28] Cornut, R.; Lefrou, C. "New Analytical Approximations for SECM with an Inlaid Microdisk Tip: Pure Positive Feedback and Insulating Substrate Feedback." *Journal of Electroanalytical Chemistry*, **604** (2), 91–100 (2007). `doi:10.1016/j.jelechem.2007.03.003 <https://doi.org/10.1016/j.jelechem.2007.03.003>`_
.. [29] Lefrou, C.; Cornut, R. "Analytical Approximations for Quantitative Approach Curves in SECM with a Flat and Non-Flat Inlaid Disk Tip under Finite Kinetics." *ChemPhysChem*, **9** (15), 2154–2160 (2008). `doi:10.1002/cphc.200800366 <https://doi.org/10.1002/cphc.200800366>`_
.. [30] Fourier, J. *Théorie analytique de la chaleur*; Firmin Didot: Paris, France, 1822.
.. [31] Roache, P. J. *Verification and Validation in Computational Science and Engineering*; Hermosa Publishers: Albuquerque, NM, 1998.
.. [32] Crank, J.; Nicolson, P. "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Mathematical Proceedings of the Cambridge Philosophical Society*, **43** (1), 50–67 (1947). `doi:10.1017/S0305004100023197 <https://doi.org/10.1017/S0305004100023197>`_
.. [33] Hairer, E.; Wanner, G. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed.; Springer-Verlag: Berlin, Germany, 1996. `doi:10.1007/978-3-642-05221-7 <https://doi.org/10.1007/978-3-642-05221-7>`_
