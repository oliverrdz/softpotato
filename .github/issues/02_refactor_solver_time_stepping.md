# Issue 2: Refactor Solver Module to Decouple Spatial Operators from Time Stepping

**Title**: `refactor(solver): decouple spatial discretization from solvers and delegate operators to Geometry`  
**Labels**: `refactor`, `architecture`, `module:solver`, `milestone-2`

---

## 1. Context & Motivation

Currently, the four diffusion solvers (`ScipyIVPSolver`, `ExplicitFiniteDifference`, `CrankNicolson`, and `ImplicitFiniteDifference`) interleave time integration with 1D Cartesian finite-difference spatial stencils:
- Derivative calculations (`d2c[1:-1] = (c[2:] - 2*c[1:-1] + c[:-2]) / dx²`) are duplicated in multiple files.
- Surface flux post-processing in `BaseSolver._post_process` hardcodes Cartesian gradients $\partial c / \partial x$.
- Changing coordinate systems currently requires modifying every solver.

By delegating spatial derivative evaluations and matrix assembly to the new `Geometry` module (Issue 1), the solver module becomes strictly responsible for **time marching and linear algebra**, adhering to the Single Responsibility Principle.

---

## 2. Proposed Changes

### 1. `DiffusionProblem` (`src/softpotato/solver/base.py`)
- Add `geometry: Geometry | str = "planar"` to `DiffusionProblem` with default `"planar"`:
  ```python
  @dataclass
  class DiffusionProblem:
      grid: np.ndarray
      diffusivity: dict[str, float]
      boundary_conditions: dict[str, tuple[BoundaryCondition, BoundaryCondition]]
      initial_conditions: dict[str, float | np.ndarray | Callable[[np.ndarray], np.ndarray]]
      reactions: Callable[...] | None = None
      species: list[str] = field(default_factory=list)
      geometry: Geometry | str = "planar"  # Backwards-compatible default
  ```
- Automatically resolve string aliases (`"planar"`, `"spherical"`, `"cylindrical"`) to corresponding `Geometry` objects via `get_geometry()`. If `geometry="planar"`, instantiate `PlanarGeometry(area=1.0)`.
- Existing user code omitting `geometry` continues running with zero behavioral difference.
- Validate that the problem grid satisfies the geometry requirements (e.g., $r[0] = r_0 > 0$ for spherical/cylindrical geometries).

### 2. `ScipyIVPSolver` (`src/softpotato/solver/scipy_ivp.py`)
- Inside `rhs(t, y)`:
  - Remove inline finite-difference evaluations.
  - Call `problem.geometry.laplacian(c_full, grid, bc_left, bc_right, t=t)` to obtain $\nabla^2 c$.
  - Evaluate reaction terms and assemble $\frac{dy}{dt} = D \nabla^2 c + R$.

### 3. `ExplicitFiniteDifference` (`src/softpotato/solver/explicit.py`)
- Inside `_step_explicit`:
  - Delegate interior and boundary derivative stencils to `problem.geometry.laplacian(...)`.
  - Update state: $c^{n+1} = c^n + \Delta t (D \nabla^2 c + R)$.

### 4. `CrankNicolson` & `ImplicitFiniteDifference` (`crank_nicolson.py`, `implicit.py`)
- Replace the uniform-grid scalar matrix diagonals ($1 + 2r$, $-r$) with variable-coefficient banded matrices assembled by `problem.geometry.laplacian_matrix(grid)`:
  $$\mathbf{A} = \mathbf{I} - \theta \Delta t D \mathbf{L}, \quad \mathbf{B} = \mathbf{I} + (1 - \theta) \Delta t D \mathbf{L}$$
  where $\theta = 0.5$ for Crank-Nicolson and $\theta = 1.0$ for implicit Euler (BTCS).
- Solve the tridiagonal banded system using `scipy.linalg.solve_banded`.

### 5. Post-Processing & Faradaic Current (`src/softpotato/solver/base.py`)
- In `BaseSolver._post_process`:
  - Delegate interfacial flux calculation to `problem.geometry.surface_flux(c, grid, D, side="left")`.
  - Compute total Faradaic current using `problem.geometry.current(flux, n_electrons)` and store in `SolverResult.currents`.

---

## 3. Tasks & Deliverables

- [ ] Update `DiffusionProblem` in `src/softpotato/solver/base.py` to accept `geometry`.
- [ ] Refactor `ScipyIVPSolver` to evaluate spatial derivatives via `problem.geometry`.
- [ ] Refactor `ExplicitFiniteDifference` to advance states using `problem.geometry.laplacian`.
- [ ] Refactor `CrankNicolson` and `ImplicitFiniteDifference` to construct banded matrices from `problem.geometry.laplacian_matrix`.
- [ ] Update `BaseSolver._post_process` to use `problem.geometry.surface_flux` and compute `result.currents`.
- [ ] Verify backwards compatibility: all existing tests in `tests/test_solvers.py` must pass unmodified.
- [ ] Add spherical benchmark tests in `tests/test_solvers.py`:
  - Verify numerical spherical step response against `softpotato.analytical.step.cottrell_spherical`.
  - Verify long-time numerical current against `softpotato.analytical.microelectrodes.microsphere_limiting_current`.
- [ ] Update solver documentation and examples in `docs/api/solver.md` (compatible with MkDocs and `mkdocstrings`).

---

## 4. Acceptance Criteria & Backwards Compatibility

- **Strict Backwards Compatibility**:
  - `DiffusionProblem` defaults to `geometry="planar"`. All existing scripts, tests, and Jupyter tutorials (e.g. `docs/examples/cottrell.ipynb`, `docs/examples/cyclic_voltammetry.ipynb`) run with identical results without any code modifications.
  - Solving with `geometry="planar"` produces bitwise or numerical-precision identical results to the previous solver implementation.
- All four solvers (`scipy_ivp`, `explicit`, `crank_nicolson`, `implicit`) contain zero geometry-specific finite-difference stencils.
- Solving with `geometry="spherical"` accurately reproduces the analytical spherical Cottrell transient ($< 0.5\%$ relative error on refined grids).
- All unit, integration, and benchmark test suites pass cleanly across Python 3.10–3.13.
- MkDocs documentation builds cleanly without warnings (`mkdocs build --strict`).

