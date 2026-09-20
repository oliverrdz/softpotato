.. Soft Potato documentation master file

Welcome to Soft Potato's Documentation!
=======================================

**Soft Potato** is a next-generation open-source electrochemical simulator and toolkit designed for electrochemists, materials scientists, and electrochemical engineers.

Key Features
------------

* **Electrochemical Mechanisms**: Model diverse multi-step electron-transfer and chemical reaction schemes (E, EC, ECE, catalytic mechanisms, and disproportionation).
* **Mass Transport & Simulation**: Numerical PDE solvers (Explicit & Implicit Finite Difference) for electrochemical diffusion and kinetic reaction layers.
* **Electrode Geometry**: Planar, spherical, cylindrical, and micro-disc electrode geometries with tailored spatial discretization.
* **Electrochemical Techniques**: Cyclic voltammetry (CV), chronoamperometry, and potential step simulations.
* **Analytical Solutions**: Built-in benchmark equations (Cottrell, Randles-Sevcik, etc.) for simulation validation.

Implementation Status
---------------------

The table below outlines the implementation readiness of each module in Soft Potato:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Module / Component
     - Status
     - Target
     - Notes
   * - :mod:`softpotato.core`
     - Ready
     - v3.0
     - Species, reactions, and mechanisms with CGS unit enforcement.
   * - :mod:`softpotato.analytical`
     - Ready
     - v3.0
     - Standard electrochemical Cottrell & voltammetry equations.
   * - :mod:`softpotato.simulate`
     - In Development
     - v3.1
     - Finite difference PDE solvers and dispatchers.
   * - :mod:`softpotato.geometry`
     - In Development
     - v3.1
     - Spatial grids and electrode geometry models.
   * - :mod:`softpotato.techniques`
     - Planned
     - v3.2
     - Numerical simulation waveforms (CV, LSV, step).
   * - :mod:`softpotato.kinetics`
     - Planned
     - v3.2
     - Butler-Volmer & Marcus-Hush kinetics models.

For detailed item-by-item progress and active development tasks, see the :doc:`roadmap`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Tutorials:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development & Status:

   roadmap
   design_spec


.. toctree::
   :maxdepth: 1
   :caption: Project Information:

   license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

