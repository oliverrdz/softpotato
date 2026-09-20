Roadmap & Pending Implementations
=================================

This page tracks planned features, pending implementations, and development stubs across Soft Potato.

Pending Implementation Items
----------------------------

The following items are currently stubs in the codebase awaiting full implementation:

.. todolist::

Release Milestones
------------------

* **v3.0.0 (Current)**:
  - Core species and reaction mechanism framework with strict CGS units validation.
  - Analytical benchmark solutions for fundamental electrochemical techniques (In-Development).
  - Documentation and interactive tutorial workflows.

* **v3.1.0 (Upcoming)**:
  - High-level numerical PDE solver dispatcher (:mod:`softpotato.simulate.solver`).
  - Explicit finite difference (:mod:`softpotato.simulate.efd`) and Crank-Nicolson implicit finite difference (:mod:`softpotato.simulate.ifd`) backends.
  - Spatial grid discretization generators (:mod:`softpotato.geometry.grids`).
  - Standard electrode geometry classes (:mod:`softpotato.geometry.electrodes`).

* **v3.2.0 (Planned)**:
  - Electrochemical simulation techniques (:mod:`softpotato.techniques.voltammetry` and :mod:`softpotato.techniques.step`).
  - Micro-electrode geometries and hydrodynamic boundary layers.
  - Butler-Volmer and Marcus-Hush kinetics models (:mod:`softpotato.kinetics.models`).

