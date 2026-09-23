API Reference
=============

.. automodule:: softpotato
   :undoc-members:

Solver Subpackage (``softpotato.solver``)
-----------------------------------------

The ``softpotato.solver`` subpackage provides numerical solvers for 1D multi-species
chemical diffusion and reaction-diffusion problems.

Solvers and Supported Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Solver Class & Aliases
     - Supported Configuration Options
     - Description & Defaults
   * - :class:`~softpotato.solver.ExplicitFiniteDifference`
       (``'explicit'``, ``'ftcs'``)
     - ``dt`` (float, optional)
     - Explicit Forward-Time Central-Space. If ``dt=None``, automatically defaults to
       :math:`0.45 \times \text{CFL limit}` (:math:`\Delta t \le \Delta x^2 / 2D_{\max}`).
   * - :class:`~softpotato.solver.ImplicitFiniteDifference`
       (``'implicit'``, ``'btcs'``)
     - ``dt`` (float, optional)
     - Backward-Time Central-Space implicit scheme. If ``dt=None``, defaults to
       100 uniform steps: :math:`(t_{\text{end}} - t_{\text{start}}) / 100`.
   * - :class:`~softpotato.solver.CrankNicolson`
       (``'crank_nicolson'``, ``'crank-nicolson'``)
     - ``dt`` (float, optional)
     - Second-order in time and space Crank-Nicolson implicit scheme. If ``dt=None``,
       defaults to 100 uniform steps: :math:`(t_{\text{end}} - t_{\text{start}}) / 100`.
   * - :class:`~softpotato.solver.ScipyIVPSolver`
       (``'scipy_ivp'``, ``'solve_ivp'``)
     - ``method`` (str, default ``'Radau'``), ``rtol`` (float, default ``1e-6``), ``atol`` (float, default ``1e-8``)
     - Method of Lines ODE integration using ``scipy.integrate.solve_ivp``. Supports stiff
       (``'Radau'``, ``'BDF'``) and non-stiff (``'RK45'``) integrators.

Module Contents
~~~~~~~~~~~~~~~

.. automodule:: softpotato.solver
   :members:
   :undoc-members:
   :show-inheritance:
