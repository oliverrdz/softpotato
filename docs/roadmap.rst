Roadmap & Pending Implementations
=================================

This page tracks planned features, development milestones, and pending implementations across Soft Potato.

Pending Implementation Items
----------------------------

The following items are currently stubs in the codebase awaiting full implementation in upcoming releases:

.. todolist::

Release Milestones
------------------

* **v3.0.0 (Current MVP)**:

  - **Core Mechanism Framework**: Species, electrochemical reactions, homogeneous chemical reactions, and mechanism orchestration with strict CGS unit enforcement (:mod:`softpotato.core`).
  - **Analytical Solutions**: Closed-form, vectorized benchmark solutions for Cottrell, Anson, Randles-Sevcik, Saito microdisc, Levich, and Koutecký-Levich (:mod:`softpotato.analytical`).
  - **End-to-End Simulation MVP**:

    * **Excitation Technique**: Triangular potential waveforms for Cyclic Voltammetry (:class:`softpotato.techniques.CyclicVoltammetry`).
    * **Interfacial Kinetics**: Butler-Volmer electron transfer model (:class:`softpotato.kinetics.ButlerVolmer`) with potential-dependent forward and backward rate constants.
    * **Geometry & Mass Transport**: 1D semi-infinite planar diffusion at a macroelectrode (:class:`softpotato.geometry.PlanarElectrode`) on uniform spatial grids (:class:`softpotato.geometry.UniformGrid`).
    * **Numerical Solver**: Vectorized Explicit Finite Difference (EFD) solver (:mod:`softpotato.simulate.efd`), high-level dispatcher (:class:`softpotato.simulate.Solver`), and automatic CFL stability sub-stepping.
    * **Validation**: Numerical simulation benchmarked against the analytical Randles-Sevcik peak current within 3%.

  - **Documentation & Tutorials**: Interactive Jupyter tutorials for mechanism formulation, analytical benchmarks, and end-to-end CV simulation.

* **v3.1.0 (Upcoming - Techniques & Extended Kinetics)**:

  - **Additional Techniques**: Waveform generators for Chronoamperometry (CA) and Linear Sweep Voltammetry (LSV) (:mod:`softpotato.techniques`).
  - **Extended Kinetics Models**:

    * Nernstian equilibrium boundary conditions (infinitely fast reversible electron transfer).
    * Tafel kinetics model (high-overpotential irreversible limit).

  - **Coupled Reactions**: Numerical simulation of coupled homogeneous chemical reactions (EC, CE, ECE, and DISP mechanisms).

* **v3.2.0 (Planned - Advanced Solvers)**:

  - **SciPy Solver Wrapper**: Method of Lines (MOL) spatial discretization coupled to :func:`scipy.integrate.solve_ivp` (BDF, Radau, RK45).
  - **Implicit Finite Difference (IFD)**: Crank-Nicolson implicit solver (:mod:`softpotato.simulate.ifd`) for unconditionally stable time stepping.
  - **Adaptive Stepping**: Dynamic temporal error estimation and adaptive time-step control.

* **v3.3.0 (Planned - Extended Geometries & Transport)**:

  - **Spherical & Microelectrode Geometries**: Spherical electrodes (:class:`softpotato.geometry.SphericalElectrode`), cylindrical electrodes, and inlaid microdiscs.
  - **Thin-Layer Cells**: Confined finite diffusion geometry with reflective/impermeable boundary conditions.
  - **Non-Uniform Grids**: Exponentially expanding spatial grids (:class:`softpotato.geometry.ExpandingGrid`) to capture steep interfacial concentration gradients.
  - **Convective Transport**: Hydrodynamic rotating disk electrode (RDE) simulation.
