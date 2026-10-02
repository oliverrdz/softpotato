API Reference
=============

Welcome to the **Soft Potato** API Reference. The toolkit is organized into modular
subpackages and modules for physical constants, closed-form analytical electrochemistry,
and numerical PDE solvers.

.. automodule:: softpotato
   :no-members:
   :undoc-members:

Overview of Modules
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Scope & Description
   * - :doc:`constants`
     - Physical and electrochemical constants defined according to CODATA 2018 / 2019 SI conventions.
   * - :doc:`analytical/step`
     - Exact solutions for potentiostatic and galvanostatic step experiments (Cottrell, Anson, Sand).
   * - :doc:`analytical/hydrodynamics`
     - Convective electrochemical models for Rotating Disk (RDE) and Rotating Ring-Disk (RRDE) electrodes.
   * - :doc:`analytical/kinetics`
     - Thermodynamic and interfacial kinetics models (Nernst, Butler–Volmer, Tafel).
   * - :doc:`analytical/voltammetry`
     - Diagnostic equations and peak current relationships for cyclic and linear sweep voltammetry.
   * - :doc:`analytical/microelectrodes`
     - Steady-state and transient current expressions for ultramicroelectrode (UME) geometries.
   * - :doc:`analytical/secm`
     - Normalized approach curve models and feedback expressions for scanning electrochemical microscopy.
   * - :doc:`analytical/fitting`
     - Robust non-linear regression and parameter estimation utilities for electrochemical data.
   * - :doc:`analytical/benchmark`
     - Automated numerical solver validation suite, analytical benchmarks, and convergence testing.
   * - :doc:`solver`
     - Numerical PDE solvers for 1D multi-species chemical diffusion and reaction-diffusion problems.

.. toctree::
   :maxdepth: 1
   :caption: Physical Constants

   constants

.. toctree::
   :maxdepth: 1
   :caption: Analytical Electrochemistry

   analytical/step
   analytical/hydrodynamics
   analytical/kinetics
   analytical/voltammetry
   analytical/microelectrodes
   analytical/secm
   analytical/fitting
   analytical/benchmark

.. toctree::
   :maxdepth: 1
   :caption: Numerical Solvers

   solver
