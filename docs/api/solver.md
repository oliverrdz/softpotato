# Solver Module (`softpotato.solver`)

The `softpotato.solver` subpackage provides numerical solvers for 1D multi-species chemical diffusion and reaction-diffusion problems.

## Solvers and Supported Options

| Solver Class & Aliases | Supported Configuration Options | Description & Defaults |
| :--- | :--- | :--- |
| [`ExplicitFiniteDifference`][softpotato.solver.ExplicitFiniteDifference] (`'explicit'`, `'ftcs'`) | `dt` (float, optional) | Explicit Forward-Time Central-Space. If `dt=None`, automatically defaults to $0.45 \times \text{CFL limit}$ ($\Delta t \le \Delta x^2 / 2D_{\max}$). |
| [`ImplicitFiniteDifference`][softpotato.solver.ImplicitFiniteDifference] (`'implicit'`, `'btcs'`) | `dt` (float, optional) | Backward-Time Central-Space implicit scheme. If `dt=None`, defaults to 100 uniform steps: $(t_{\text{end}} - t_{\text{start}}) / 100$. |
| [`CrankNicolson`][softpotato.solver.CrankNicolson] (`'crank_nicolson'`, `'crank-nicolson'`) | `dt` (float, optional) | Second-order in time and space Crank–Nicolson implicit scheme. If `dt=None`, defaults to 100 uniform steps: $(t_{\text{end}} - t_{\text{start}}) / 100$. |
| [`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver] (`'scipy_ivp'`, `'solve_ivp'`) | `method` (str, default `'Radau'`), `rtol` (float, default `1e-6`), `atol` (float, default `1e-8`) | Method of Lines ODE integration using `scipy.integrate.solve_ivp`. Supports stiff (`'Radau'`, `'BDF'`) and non-stiff (`'RK45'`) integrators. |

## Module Contents

::: softpotato.solver
