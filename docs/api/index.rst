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
constants defined according to CODATA 2018 / 2019 SI conventions.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Constant / Identifier
     - Symbol
     - Value & Units
   * - :data:`~softpotato.constants.FARADAY` (``F``)
     - :math:`F`
     - :math:`96485.332\,\text{C}\cdot\text{mol}^{-1}`
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
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.step.cottrell`
     - :math:`t, n, D, c^*, A`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.step.cottrell_step`
     - :math:`t, \tau, n, D, c^*, A, D_{\text{Red}}`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.step.cottrell_spherical`
     - :math:`t, r, n, D, c^*, A`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.step.cottrell_cylinder`
     - :math:`t, r_0, n, D, c^*, l, A`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.step.anson`
     - :math:`t, n, D, c^*, A, Q_{\text{dl}}, \Gamma, \tau`
     - :math:`Q(t)`
   * - :func:`~softpotato.analytical.step.sand`
     - :math:`t, I, n, D, c^*, A`
     - :math:`c(0, t)`
   * - :func:`~softpotato.analytical.step.sand_transition_time`
     - :math:`I, n, D, c^*, A`
     - :math:`\tau`
   * - :func:`~softpotato.analytical.step.sand_potential`
     - :math:`t, \tau, E_{1/2}, n, T`
     - :math:`E(t)`
   * - :func:`~softpotato.analytical.step.step_concentration_profile`
     - :math:`x, t, D, c^*`
     - :math:`c(x, t)`
   * - :func:`~softpotato.analytical.step.step_flux_profile`
     - :math:`x, t, D, c^*`
     - :math:`J(x, t)`

.. rubric:: Hydrodynamics & Convection Equations
   :class: submodule-heading

The :mod:`softpotato.analytical.hydrodynamics` module provides models for rotating disk
(RDE) and rotating ring-disk (RRDE) convective electrochemical systems.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.hydrodynamics.levich`
     - :math:`\omega, n, D, c^*, A, \nu`
     - :math:`I_{\text{L}}`
   * - :func:`~softpotato.analytical.hydrodynamics.levich_constant`
     - :math:`n, D, c^*, A, \nu`
     - :math:`B`
   * - :func:`~softpotato.analytical.hydrodynamics.koutecky_levich`
     - :math:`\omega, I_{\text{k}}, k, I_{\text{lim}}, n, D, c^*, A, \nu`
     - :math:`I`
   * - :func:`~softpotato.analytical.hydrodynamics.koutecky_levich_analysis`
     - :math:`\omega, I, A, c^*, n, \nu`
     - :math:`k, B`
   * - :func:`~softpotato.analytical.hydrodynamics.nernst_diffusion_layer`
     - :math:`\omega, D, \nu`
     - :math:`\delta`
   * - :func:`~softpotato.analytical.hydrodynamics.collection_efficiency`
     - :math:`r_1, r_2, r_3`
     - :math:`N`
   * - :func:`~softpotato.analytical.hydrodynamics.shielding_factor`
     - :math:`r_1, r_2, r_3, N`
     - :math:`S`
   * - :func:`~softpotato.analytical.hydrodynamics.ring_collection_current`
     - :math:`I_{\text{disk}}, N, n_{\text{ring}}, n_{\text{disk}}`
     - :math:`I_{\text{ring}}`
   * - :func:`~softpotato.analytical.hydrodynamics.ring_limiting_current`
     - :math:`\omega, r_1, r_2, r_3, n, D, c^*, \nu`
     - :math:`I_{\text{R,lim}}`
   * - :func:`~softpotato.analytical.hydrodynamics.rotating_ring_disk`
     - :math:`r_1, r_2, r_3, \omega, n, D, c^*, \nu`
     - :math:`N, S, I_{\text{D,lim}}, I_{\text{R,lim}}`
   * - :func:`~softpotato.analytical.hydrodynamics.rad_s_to_rpm`
     - :math:`\omega`
     - :math:`f_{\text{RPM}}`
   * - :func:`~softpotato.analytical.hydrodynamics.rpm_to_rad_s`
     - :math:`f_{\text{RPM}}`
     - :math:`\omega`

.. rubric:: Thermodynamics & Interfacial Kinetics
   :class: submodule-heading

The :mod:`softpotato.analytical.kinetics` module provides thermodynamic equilibrium potentials
and electrode kinetic models, including Nernst, Butler–Volmer, and Tafel formulations.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.kinetics.nernst`
     - :math:`E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T`
     - :math:`E`
   * - :func:`~softpotato.analytical.kinetics.nernst_potential`
     - :math:`E^\circ, c_{\text{Ox}}, c_{\text{Red}}, n, T`
     - :math:`E`
   * - :func:`~softpotato.analytical.kinetics.nernst_ratio`
     - :math:`E, E^\circ, n, T`
     - :math:`c_{\text{Ox}} / c_{\text{Red}}`
   * - :func:`~softpotato.analytical.kinetics.nernst_equilibrium_concentrations`
     - :math:`E, E^\circ, c_{\text{total}}, n, T`
     - :math:`c_{\text{Ox}}, c_{\text{Red}}`
   * - :func:`~softpotato.analytical.kinetics.butler_volmer`
     - :math:`\eta, I_0, \alpha, n, T`
     - :math:`I`
   * - :func:`~softpotato.analytical.kinetics.butler_volmer_current_density`
     - :math:`\eta, j_0, \alpha, n, T`
     - :math:`j`
   * - :func:`~softpotato.analytical.kinetics.butler_volmer_linear`
     - :math:`\eta, I_0, n, T`
     - :math:`I`
   * - :func:`~softpotato.analytical.kinetics.charge_transfer_resistance`
     - :math:`I_0, n, T`
     - :math:`R_{\text{ct}}`
   * - :func:`~softpotato.analytical.kinetics.exchange_current`
     - :math:`k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, A, \alpha, n`
     - :math:`I_0`
   * - :func:`~softpotato.analytical.kinetics.exchange_current_density`
     - :math:`k_0, c_{\text{Ox}}^*, c_{\text{Red}}^*, \alpha, n`
     - :math:`j_0`
   * - :func:`~softpotato.analytical.kinetics.tafel`
     - :math:`\eta, I_0, \alpha, n, T`
     - :math:`I`
   * - :func:`~softpotato.analytical.kinetics.tafel_overpotential`
     - :math:`I, I_0, \alpha, n, T`
     - :math:`\eta`
   * - :func:`~softpotato.analytical.kinetics.tafel_slope`
     - :math:`\alpha, n, T`
     - :math:`b`
   * - :func:`~softpotato.analytical.kinetics.tafel_analysis`
     - :math:`\eta, I, A, c^*, n, T`
     - :math:`I_0, \alpha, b`

.. rubric:: Voltammetry & Kinetic Diagnostics
   :class: submodule-heading

The :mod:`softpotato.analytical.voltammetry` module provides peak current expressions and
reversibility diagnostics for cyclic and linear sweep voltammetry.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik`
     - :math:`v, c^*, D, A, n, T`
     - :math:`I_{\text{p}}`
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik_irreversible`
     - :math:`v, \alpha, n_\alpha, c^*, D, A, n, T`
     - :math:`I_{\text{p}}`
   * - :func:`~softpotato.analytical.voltammetry.randles_sevcik_quasi`
     - :math:`v, k_0, \Lambda, c^*, D, A, \alpha, n, T`
     - :math:`I_{\text{p}}`
   * - :func:`~softpotato.analytical.voltammetry.peak_potential_irreversible`
     - :math:`v, k_0, E^{\circ\prime}, D, \alpha, n_\alpha, T`
     - :math:`E_{\text{p}}`
   * - :func:`~softpotato.analytical.voltammetry.nicholson_delta_ep`
     - :math:`\psi, n`
     - :math:`\Delta E_{\text{p}}`
   * - :func:`~softpotato.analytical.voltammetry.nicholson_psi`
     - :math:`\Delta E_{\text{p}}, k_0, v, D, n, T`
     - :math:`\psi`
   * - :func:`~softpotato.analytical.voltammetry.nicholson_rate_constant`
     - :math:`\Delta E_{\text{p}}, v, D, n, T`
     - :math:`k_0`
   * - :func:`~softpotato.analytical.voltammetry.matsuda_ayabe_lambda`
     - :math:`k_0, v, D, n, T`
     - :math:`\Lambda`
   * - :func:`~softpotato.analytical.voltammetry.matsuda_ayabe`
     - :math:`k_0, v, D, n, \alpha, T`
     - :math:`\Lambda, \text{zone}`

.. rubric:: Microelectrode Geometries
   :class: submodule-heading

The :mod:`softpotato.analytical.microelectrodes` module provides steady-state and transient
diffusion-limited current expressions for ultramicroelectrode (UME) geometries.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.microelectrodes.microdisc_limiting_current`
     - :math:`a, n, D, c^*`
     - :math:`I_{\text{ss}}`
   * - :func:`~softpotato.analytical.microelectrodes.microdisc_transient`
     - :math:`t, a, n, D, c^*`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.microelectrodes.mahon_oldham_transient`
     - :math:`t, a, n, D, c^*`
     - :math:`I(t)`
   * - :func:`~softpotato.analytical.microelectrodes.microsphere_limiting_current`
     - :math:`r, n, D, c^*`
     - :math:`I_{\text{ss}}`
   * - :func:`~softpotato.analytical.microelectrodes.microhemisphere_limiting_current`
     - :math:`r, n, D, c^*`
     - :math:`I_{\text{ss}}`
   * - :func:`~softpotato.analytical.microelectrodes.microband_limiting_current`
     - :math:`t, w, l, n, D, c^*`
     - :math:`I(t)`

.. rubric:: Scanning Electrochemical Microscopy
   :class: submodule-heading

The :mod:`softpotato.analytical.secm` module provides normalized approach curve models
and feedback current expressions for scanning electrochemical microscopy.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.secm.secm_limiting_current_infinite`
     - :math:`a, n, D, c^*`
     - :math:`I_{\text{T},\infty}`
   * - :func:`~softpotato.analytical.secm.secm_approach_curve`
     - :math:`L, R_{\text{g}}, \kappa`
     - :math:`I_{\text{T}}^{\text{norm}}`
   * - :func:`~softpotato.analytical.secm.secm_approach_positive_feedback`
     - :math:`L, R_{\text{g}}`
     - :math:`I_{\text{T}}^{\text{norm}}`
   * - :func:`~softpotato.analytical.secm.secm_approach_negative_feedback`
     - :math:`L, R_{\text{g}}`
     - :math:`I_{\text{T}}^{\text{norm}}`
   * - :func:`~softpotato.analytical.secm.secm_tip_current`
     - :math:`L, R_{\text{g}}, \kappa, a, n, D, c^*`
     - :math:`I_{\text{T}}`

.. rubric:: Parameter Fitting & Regression
   :class: submodule-heading

The :mod:`softpotato.analytical.fitting` module provides non-linear regression and parameter
extraction routines for experimental electrochemical data.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.fitting.fit_linear`
     - :math:`x, y`
     - :math:`m, c, R^2`
   * - :func:`~softpotato.analytical.fitting.fit_curve`
     - :math:`f(x, p), x, y, p_0, \sigma`
     - :math:`p, \sigma_p, R^2`
   * - :func:`~softpotato.analytical.fitting.fit_cottrell`
     - :math:`t, I, n, A, c^*`
     - :math:`D, I_{\text{dl}}`
   * - :func:`~softpotato.analytical.fitting.fit_levich`
     - :math:`\omega, I, n, A, c^*, \nu`
     - :math:`D, B`
   * - :func:`~softpotato.analytical.fitting.fit_koutecky_levich`
     - :math:`\omega, I, n, A, c^*, \nu`
     - :math:`I_{\text{k}}, k_0, D`
   * - :func:`~softpotato.analytical.fitting.fit_tafel`
     - :math:`\eta, I, n, T`
     - :math:`I_0, \alpha, b`
   * - :func:`~softpotato.analytical.fitting.fit_randles_sevcik`
     - :math:`v, I_{\text{p}}, n, A, c^*, T`
     - :math:`D`
   * - :func:`~softpotato.analytical.fitting.fit_microdisc_transient`
     - :math:`t, I, r, c^*, n`
     - :math:`D`
   * - :func:`~softpotato.analytical.fitting.fit_secm_approach`
     - :math:`d, I_{\text{norm}}, a`
     - :math:`R_{\text{g}}, d_{\text{offset}}`

.. rubric:: Simulation Benchmarking Suite
   :class: submodule-heading

The :mod:`softpotato.analytical.benchmark` module provides validation harnesses, exact
analytical benchmarks, and spatial/temporal order of convergence testing utilities.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Function
     - Inputs
     - Output
   * - :func:`~softpotato.analytical.benchmark.compute_error_metrics`
     - :math:`u_{\text{num}}, u_{\text{exact}}`
     - :math:`e_{\max}, \text{RMSE}, L_2`
   * - :func:`~softpotato.analytical.benchmark.estimate_convergence_order`
     - :math:`\Delta x, e`
     - :math:`p, R^2`
   * - :func:`~softpotato.analytical.benchmark.list_benchmark_cases`
     - :math:`\varnothing`
     - :math:`\text{cases}`
   * - :func:`~softpotato.analytical.benchmark.run_benchmark`
     - :math:`\text{case}, \text{solver}`
     - :math:`\text{result}`
   * - :func:`~softpotato.analytical.benchmark.run_cottrell_benchmark`
     - :math:`\text{solver}, D, c^*, A`
     - :math:`\text{result}`
   * - :func:`~softpotato.analytical.benchmark.run_fourier_decay_benchmark`
     - :math:`\text{solver}, D, L, t_{\text{end}}`
     - :math:`\text{result}`
   * - :func:`~softpotato.analytical.benchmark.run_mass_conservation_benchmark`
     - :math:`\text{solver}, D, L, t`
     - :math:`\text{result}`
   * - :func:`~softpotato.analytical.benchmark.run_benchmark_suite`
     - :math:`\text{solvers}, \text{cases}`
     - :math:`\text{suite}`
   * - :func:`~softpotato.analytical.benchmark.verify_spatial_convergence`
     - :math:`\text{solver}, \Delta x_i, D, L`
     - :math:`p_x`
   * - :func:`~softpotato.analytical.benchmark.verify_temporal_convergence`
     - :math:`\text{solver}, \Delta t_j, D, L`
     - :math:`p_t`


.. rubric:: Solver Module
   :class: module-heading

The :mod:`softpotato.solver` subpackage provides numerical partial differential equation (PDE)
solvers and boundary condition primitives for 1D multi-species diffusion and reaction-diffusion systems.

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Class / Function
     - Inputs
     - Output
   * - :func:`~softpotato.solver.get_solver`
     - :math:`\text{name}`
     - :math:`\text{solver}`
   * - :func:`~softpotato.solver.list_solvers`
     - :math:`\varnothing`
     - :math:`\text{names}`
   * - :func:`~softpotato.solver.register_solver`
     - :math:`\text{name}, \text{cls}`
     - :math:`\varnothing`
   * - :class:`~softpotato.solver.DirichletBC`
     - :math:`c_{\text{bc}}`
     - :math:`\text{BC}`
   * - :class:`~softpotato.solver.NeumannBC`
     - :math:`J_{\text{bc}}`
     - :math:`\text{BC}`
   * - :class:`~softpotato.solver.DiffusionProblem`
     - :math:`x, D_i, \text{BC}_i, c_{i,0}, R_i`
     - :math:`\text{Problem}`
   * - :class:`~softpotato.solver.ExplicitFiniteDifference`
     - :math:`\Delta t`
     - :math:`\text{solver}`
   * - :class:`~softpotato.solver.ImplicitFiniteDifference`
     - :math:`\Delta t`
     - :math:`\text{solver}`
   * - :class:`~softpotato.solver.CrankNicolson`
     - :math:`\Delta t`
     - :math:`\text{solver}`
   * - :class:`~softpotato.solver.ScipyIVPSolver`
     - :math:`\text{method}, \text{rtol}, \text{atol}`
     - :math:`\text{solver}`
   * - :meth:`~softpotato.solver.BaseSolver.solve`
     - :math:`\text{Problem}, (t_0, t_1)`
     - :math:`c_i(x, t)`
