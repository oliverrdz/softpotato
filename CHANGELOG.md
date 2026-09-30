# Changelog

All notable changes to the Soft Potato project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added
- **Microelectrode & Ultramicroelectrode (UME) Models (`softpotato.analytical.microelectrodes`)**:
  - Introduced dedicated microelectrode and ultramicroelectrode analytical submodule (`softpotato.analytical.microelectrodes`).
  - Added Saito steady-state diffusion-limited current equation (`microdisc_limiting_current`) for circular inlaid microdiscs ($I_{\text{lim}} = 4 n F D c^* a$).
  - Added Shoup and Szabo full-time chronoamperometric transient approximation (`microdisc_transient`) for inlaid microdiscs spanning planar Cottrell decay to Saito steady state with $<0.6\%$ relative error.
  - Added Mahon and Oldham analytical piecewise chronoamperometric transient model (`mahon_oldham_transient`) for inlaid microdiscs with $<0.02\%$ relative error across intermediate times.
  - Added hemispherical microelectrode steady-state diffusion-limited current (`microhemisphere_limiting_current`) on an insulator ($I_{\text{lim}} = 2 \pi n F D c^* r$).
  - Added spherical microelectrode steady-state diffusion-limited current (`microsphere_limiting_current`) in bulk solution ($I_{\text{lim}} = 4 \pi n F D c^* r$).
  - Added quasi-steady-state microband electrode diffusion-limited current (`microband_limiting_current`) with logarithmic time decay ($I_{\text{lim}}(t) = 2 \pi n F D c^* l / \ln(0.64 D t / w^2)$).
  - Re-exported all microelectrode analytical functions at `softpotato.analytical` and top-level `softpotato`.
  - Added comprehensive unit test suite in `tests/test_analytical_microelectrodes.py` with 20 tests and complete doctests.
  - Documented `softpotato.analytical.microelectrodes` in Sphinx API reference (`docs/api.rst`).
- **Voltammetry & Kinetic Diagnostics (`softpotato.analytical.voltammetry`)**:
  - Introduced voltammetric peak current and kinetic diagnostic submodule (`softpotato.analytical.voltammetry`).
  - Added classical Randles–Ševčík peak current equation for reversible planar diffusion (`randles_sevcik`) supporting cathodic and anodic sweeps with vectorization across scan rates.
  - Added irreversible Randles–Ševčík peak current equation (`randles_sevcik_irreversible`) supporting variable transfer coefficient $\alpha$ and stoichiometric electron count $n_\alpha$.
  - Added quasi-reversible peak current approximation (`randles_sevcik_quasi`) transitioning smoothly between reversible and irreversible limits via Matsuda–Ayabe shape factor $K(\Lambda, \alpha)$.
  - Added Nicholson kinetic diagnostic method (`nicholson_psi`) supporting forward calculation of peak separation $\Delta E_p$, inverse estimation of dimensionless rate parameter $\Psi$ (Swaddle, Lavagnini, and tabulated root-finding), and direct extraction of standard heterogeneous rate constant $k^0$.
  - Added Matsuda–Ayabe reversibility parameter calculation (`matsuda_ayabe_lambda`) and zone classification (`matsuda_ayabe`) returning structured dataclass (`MatsudaAyabeResult`).
  - Added irreversible peak potential shift diagnostics with scan rate (`peak_potential_irreversible`).
  - Re-exported all voltammetry functions and dataclasses at `softpotato.analytical` and top-level `softpotato`.
  - Added comprehensive test suite in `tests/test_analytical_voltammetry.py` with 23 tests and complete docstring examples.
  - Documented `softpotato.analytical.voltammetry` in Sphinx API reference (`docs/api.rst`).
- **Thermodynamics & Interfacial Kinetics (`softpotato.analytical.kinetics`)**:
  - Introduced thermodynamic and heterogeneous interfacial kinetics submodule (`softpotato.analytical.kinetics`).
  - Added Nernst equilibrium electrode potential equation (`nernst` / `nernst_potential`) supporting single-ratio ($c_{\text{Ox}} / c_{\text{Red}}$) or separate concentration arguments, with asymptotic $\pm\infty$ limit handling.
  - Added equilibrium concentration ratio inverter (`nernst_ratio`) and numerically stable species concentration distributor (`nernst_equilibrium_concentrations`) leveraging `scipy.special.expit`.
  - Added exchange current density (`exchange_current_density`) and total exchange current (`exchange_current`) scaling with electrode area.
  - Added charge-transfer activation resistance calculation (`charge_transfer_resistance`).
  - Added Butler–Volmer equation (`butler_volmer` and `butler_volmer_current_density`) supporting activation control, mass-transport/surface concentration corrections ($c_i(0) / c_i^*$), and IUPAC vs polarographic sign conventions.
  - Added low-overpotential linear Butler–Volmer approximation (`butler_volmer_linear`).
  - Added theoretical Tafel slope equation (`tafel_slope`) for anodic and signed cathodic branches.
  - Added high-overpotential Tafel current approximation (`tafel`) with automatic branch detection (`branch='auto'`) and overpotential inversion (`tafel_overpotential`).
  - Added diagnostic linear Tafel regression analysis (`tafel_analysis`) returning structured dataclass (`TafelResult`) with extracted $I_0$, Tafel slope $b$ (in V/dec and mV/dec), apparent transfer coefficient $\alpha$, $R^2$, regression standard errors, and standard rate constant $k^0$.
  - Re-exported all kinetics functions and dataclasses at `softpotato.analytical` and top-level `softpotato`.
  - Added comprehensive unit test suite in `tests/test_analytical_kinetics.py` with 27 tests and verified doctests.
  - Documented `softpotato.analytical.kinetics` in Sphinx API reference (`docs/api.rst`).
- **Hydrodynamics & Convection Equations (`softpotato.analytical.hydrodynamics`)**:
  - Introduced hydrodynamic convective-diffusion equations submodule (`softpotato.analytical.hydrodynamics`).
  - Added Levich equation (`levich`) for convective mass-transport limiting current at a Rotating Disk Electrode (RDE) with dual angular velocity (`omega` in rad/s) and rotational frequency (`rpm`) input support.
  - Added theoretical Levich slope parameter utility (`levich_constant`) and Nernst stagnant diffusion layer thickness calculation (`nernst_diffusion_layer`).
  - Added Koutecký–Levich equation (`koutecky_levich`) for mixed kinetic and mass-transfer limitation, accepting explicit kinetic current ($I_K$), heterogeneous rate constant ($k$), and rotation speeds.
  - Added diagnostic Koutecký–Levich linear regression analysis (`koutecky_levich_analysis`) extracting kinetic current ($I_K$), rate constant ($k$), Levich constant ($B$), estimated diffusivity ($D$), and $R^2$.
  - Added exact closed-form Albery–Bruckenstein theoretical collection efficiency (`collection_efficiency`) for Rotating Ring-Disk Electrodes (RRDE) evaluated analytically via hydrodynamic function $F(\theta)$.
  - Added unshielded and shielded ring limiting currents (`ring_limiting_current`), ring collection currents for disk-generated products (`ring_collection_current`), and geometric shielding factors (`shielding_factor`).
  - Added unified RRDE solver (`rotating_ring_disk`) returning a structured dataclass (`RRDEResult`) with complete geometric and current parameters.
  - Added rotational speed conversion utilities (`rpm_to_rad_s` and `rad_s_to_rpm`).
  - Re-exported all hydrodynamic functions and dataclasses at `softpotato.analytical` and top-level `softpotato`.
  - Added comprehensive test suite in `tests/test_analytical_hydrodynamics.py` with 100% statement coverage, benchmarked against commercial Pine RRDE geometries and tutorial numerical simulation data.
  - Documented `softpotato.analytical.hydrodynamics` in Sphinx API reference (`docs/api.rst`).
- **Transient Diffusion & Potential Step Equations (`softpotato.analytical.step`)**:
  - Introduced dedicated analytical equations subpackage (`softpotato.analytical.step`).
  - Added planar Cottrell equation (`cottrell`) for semi-infinite linear diffusion chronoamperometry.
  - Added spherical Cottrell equation (`cottrell_spherical`) for spherical electrodes, hanging mercury drops, and hemispherical microelectrodes with long-time steady-state convergence.
  - Added double potential step chronoamperometry reversal transient (`cottrell_step`) with diagnostic current ratio verification ($-I(2\tau)/I(\tau) = 1 - 1/\sqrt{2} \approx 0.292893$) and unequal diffusion coefficient ($D_{\text{Red}} \neq D_{\text{Ox}}$) support.
  - Added Anson equation (`anson`) for single- and double-step chronocoulometry cumulative charge transients, supporting double-layer charging ($Q_{\text{dl}}$) and reactant surface excess ($\Gamma$).
  - Added Sand equation for chronopotentiometry transition time (`sand_transition_time`), time-dependent surface concentration depletion (`sand`), and reversible chronopotentiometric wave potential (`sand_potential`).
  - Added cylindrical microelectrode chronoamperometry (`cottrell_cylinder`) implementing Aoki's full-range approximation and Oldham's rational Padé approximant across all dimensionless times $\theta$.
  - Added exact spatial concentration distribution (`step_concentration_profile`) and diffusion flux profile (`step_flux_profile`) for analytical solver verification.
  - Re-exported all step equations and the `analytical` subpackage at the top level of `softpotato`.
  - Added comprehensive test suite in `tests/test_analytical_step.py` cross-validating analytical solutions against `ScipyIVPSolver` and Sphinx documentation in `docs/api.rst`.
- **Centralized Physical & Electrochemical Constants (`softpotato.constants`)**:

  - Introduced standard electrochemical constants: Faraday constant (`FARADAY`), molar gas constant (`GAS_CONSTANT`), and standard thermodynamic temperature (`STANDARD_TEMPERATURE`).
  - Added shorthand aliases: `F`, `R`, and `T_STD`.
  - Added foundational physical constants from CODATA / 2019 SI via `scipy.constants`: `AVOGADRO`, `BOLTZMANN`, `ELEMENTARY_CHARGE`, and `VACUUM_PERMITTIVITY`.
  - Re-exported all constants and aliases at the top level of `softpotato`.
  - Added unit test suite in `tests/test_constants.py` and Sphinx API documentation in `docs/api.rst`.

## [3.0.0] - 2026-09-27

### Stable Baseline Release: 1D Diffusion Solvers, SciPy Method of Lines & Benchmark Suite
Soft Potato 3.0.0 is the official stable baseline release following the ground-up architecture reboot. This release finalizes the numerical solver module (`softpotato.solver`), integrating the stiff `scipy.integrate.solve_ivp` Method of Lines integration alongside classical tridiagonal finite difference schemes, analytical validation benchmarks, and a publication-quality suite of interactive electrochemical tutorials.

### Added
- **Interactive Tutorials & Validation Notebooks** (`examples/`):
  - **Cottrell Chronoamperometry with `solve_ivp`** (`examples/cottrell_solve_ivp_tutorial.ipynb`):
    - Simulates the Cottrell potential-step experiment using Soft Potato's adaptive `solve_ivp` solver and benchmarks against the analytical Cottrell equation.
  - **Cyclic Voltammetry with Nernst Kinetics** (`examples/cyclic_voltammetry_solve_ivp_tutorial.ipynb`):
    - Simulates reversible cyclic voltammetry using `solve_ivp`, visualizes spatio-temporal concentration profiles, validates peak currents against the Randles–Ševčík equation across scan rates, and evaluates diagnostic reversibility criteria.
  - **Cyclic Voltammetry with Butler–Volmer Kinetics** (`examples/cyclic_voltammetry_butler_volmer_tutorial.ipynb`):
    - Simulates cyclic voltammetry under Butler–Volmer quasi-reversible kinetics with unequal diffusion coefficients ($D_{\text{Red}} \neq D_{\text{Ox}}$) using second-order ghost-node flux discretization and benchmarks against Nicholson theory.
  - **Rotating Disk Electrode (RDE) Voltammetry with Levich Analysis** (`examples/rde_cyclic_voltammetry_solve_ivp_tutorial.ipynb`):
    - Simulates RDE cyclic voltammetry with Nernst kinetics using the Nernst stagnant diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Demonstrates the transition from transient peaks to steady-state waves across scan rates and validates limiting currents against the Levich equation.
  - **RDE Voltammetry with Butler–Volmer Kinetics & Koutecký–Levich Analysis** (`examples/rde_butler_volmer_koutecky_levich_tutorial.ipynb`):
    - Simulates RDE cyclic voltammetry for an oxidation under IUPAC convention with Butler–Volmer kinetics using the Nernst diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Demonstrates kinetic shifts via normalized voltammograms ($I/I_L$), validates limiting currents against the Levich equation, and extracts $k_0$ and $\alpha$ using Koutecký–Levich and Tafel analysis.
- **Project Roadmap** (`ROADMAP.md` and `docs/roadmap.md`):
  - Formulated and published strategic milestone roadmap for the 3.x series defining future backwards-compatible feature additions: Milestone 1: Grids (`3.1.0`), Milestone 2: Reactions (`3.2.0`), and Milestone 3: Techniques (`3.3.0`).
- **Core Solver Enhancements (`softpotato.solver`)**:
  - Implemented runtime option overrides via `**kwargs` in `BaseSolver.solve()`, allowing parameters such as `dt`, `method`, `rtol`, and `atol` passed at call time to take precedence over constructor defaults.
  - Added comprehensive parameter docstrings with type specifications across all solver implementations.
  - Added dedicated solver reference table in `docs/api.rst` summarizing available methods and default options.
- **CI/CD Quality & Automation Enhancements (`.github/workflows/ci.yml`)**:
  - Added dedicated GitHub Actions jobs for code formatting and lint verification (`ruff`).
  - Added static type checking job (`mypy`).
  - Added automated execution and testing of all Jupyter example notebooks via `nbmake` (`pytest -v --nbmake examples/`).
  - Added test coverage reporting with `pytest-cov` and docstring testing with `--doctest-modules`.
  - Added automated package build verification and distribution metadata checks (`python -m build && twine check`).
- **Sphinx Documentation Styling**:
  - Added custom stylesheet `docs/_static/custom.css` for consistent presentation.
  - Configured `IPython.sphinxext.ipython_console_highlighting` in `docs/conf.py` for reliable syntax highlighting of Jupyter notebook cells on Read the Docs.

### Changed
- Promoted package version to `3.0.0` and PyPI classifier to `Development Status :: 5 - Production/Stable` in `pyproject.toml`.
- Stabilized the `softpotato.solver` module as the official v3.0.0 baseline.
- **Streamlined README.md**:
  - Replaced verbose quickstart with minimal 1D diffusion example using `scipy_ivp`.
  - Added structured **Tutorials & Examples** table indexing all five interactive notebooks.
  - Removed pre-release rewrite warning banner.

### Fixed
- Fixed Python 3.10 `TypeError: ABCMeta.__new__() got multiple values for argument 'name'` during `BaseSolver` subclass registration via `_BaseSolverMeta` metaclass.
- Fixed LaTeX/MathJax math delimiters rendering across Jupyter notebook tutorials and documentation pages on Read the Docs.
- Fixed Ruff formatting and lint violations across tutorials and source files.

### Removed
- Removed legacy preliminary notebook `examples/1d_diffusion_comparison.ipynb`.
- Removed redundant `.github/workflows/docs.yml` workflow file in favor of direct Read the Docs integration.
- Removed temporary repository scaffolding script `scaffold.sh`.

## [3.0.0a1] - 2026-09-23

### Complete Architecture Reboot & 1D Numerical Diffusion Solvers
Ground-up redesign and modern reimplementation of Soft Potato (`v3.0.0a1`), introducing a clean repository foundation, modernized packaging, Sphinx documentation with Read the Docs and GitHub Actions CI workflows, a unified numerical solver framework for multi-species 1D diffusion and reaction-diffusion systems (`softpotato.solver`), a comprehensive test suite with analytical validation, and interactive benchmark tutorials.

### Added
- **1D Numerical Diffusion Solvers Framework (`softpotato.solver`)**:
  - Implemented `DiffusionProblem` formulation for 1D multi-species chemical diffusion and reaction-diffusion systems:
    $$\frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(\mathbf{c}, x, t)$$
    - Monotonically increasing 1D spatial grid validation with uniform step size detection (`dx` property).
    - Multi-species support with strict validation for positive diffusion coefficients ($D_i > 0$).
    - Flexible initial conditions supporting constant scalar concentrations, 1D NumPy profile arrays, or spatial callables $f(x)$.
    - Pluggable homogeneous reaction kinetics term $R_i(\mathbf{c}, x, t)$ returning reaction rates per species.
  - Implemented `BoundaryCondition` class and factory constructors:
    - `DirichletBC`: Fixed concentration boundary conditions ($c = \text{value}$), supporting static float values or time-dependent callables $f(t)$.
    - `NeumannBC`: Concentration gradient boundary conditions ($\partial c / \partial x = \text{value}$), supporting static float values or time-dependent callables $g(t)$.
  - Implemented `SolverResult` container:
    - Standardized simulation output holding time array $t$ ($N_t$), grid coordinates $x$ ($N_x$), and per-species 2D concentration arrays ($N_t \times N_x$).
    - Dict-like shorthand indexing (`result["A"]`) and species listing (`result.species`).
    - Automated surface flux computation at the left boundary $x=0$: $J_i(t) = -D_i \left.\frac{\partial c_i}{\partial x}\right|_{x=0}$ using second-order forward finite differences.
    - Status reporting with `success`, `message`, and `raw_output`.
  - Implemented `BaseSolver` abstract base class:
    - Dynamic solver registry with auto-registration on subclassing (`BaseSolver, name="..."`) and `@register_solver(name)` decorator.
    - Integrated `solve()` pipeline orchestrating pre-run input validation (`validate`), numerical integration dispatch (`_run_solver`), and post-processing flux calculation (`_post_process`).
    - Solver factory function `get_solver(name, **kwargs)` and discovery function `list_solvers()`.
  - Implemented four numerical diffusion solvers:
    - **Method of Lines (`ScipyIVPSolver`, aliases: `"scipy_ivp"`, `"solve_ivp"`)**:
      - Spatial semi-discretization into an ODE system integrated via `scipy.integrate.solve_ivp`.
      - Default stiff integrator using the implicit 5th-order Runge-Kutta `Radau` method, with support for `BDF`, `RK45`, `RK23`, and `LSODA`.
      - Direct coupling of homogeneous reaction kinetics $R_i(\mathbf{c}, x, t)$ and dynamic Dirichlet/Neumann boundary conditions.
    - **Crank-Nicolson (`CrankNicolson`, aliases: `"crank_nicolson"`, `"crank-nicolson"`)**:
      - Unconditionally stable second-order in time and space implicit finite difference method ($\mathcal{O}(\Delta t^2, \Delta x^2)$).
      - Efficient tridiagonal system solution via `scipy.linalg.solve_banded` ($O(N)$ Thomas algorithm).
      - Handles both Dirichlet and second-order central difference Neumann boundary conditions, plus homogeneous reaction term coupling.
    - **Implicit Finite Difference (`ImplicitFiniteDifference`, aliases: `"implicit"`, `"btcs"`)**:
      - Backward-Time Central-Space (BTCS) unconditionally stable implicit scheme ($\mathcal{O}(\Delta t, \Delta x^2)$).
      - Tridiagonal matrix solver via `scipy.linalg.solve_banded` supporting Dirichlet and Neumann boundary conditions.
    - **Explicit Finite Difference (`ExplicitFiniteDifference`, aliases: `"explicit"`, `"ftcs"`)**:
      - Forward-Time Central-Space (FTCS) explicit scheme ($\mathcal{O}(\Delta t, \Delta x^2)$).
      - Strict Courant-Friedrichs-Lewy (CFL) stability enforcement ($\Delta t \le \frac{\Delta x^2}{2 D_{\max}}$) with configurable auto-substepping (`auto_substep=True`) or explicit error raising.
    - Explicit `__init__` signatures, typed parameters, and NumPy docstrings across all solver classes (`dt` for FTCS/BTCS/Crank-Nicolson; `method`, `rtol`, `atol` for SciPy IVP) ensuring full option visibility in Sphinx autodoc and IDE autocompletion.
- **Top-Level Package API (`softpotato`)**:
  - Re-exported core solver classes (`BaseSolver`, `BoundaryCondition`, `DiffusionProblem`, `DirichletBC`, `NeumannBC`, `SolverResult`, `get_solver`, `list_solvers`) in `softpotato/__init__.py`.
  - Exposed module version `__version__ = "3.0.0a1"`.
- **Testing Suite (`tests/`)**:
  - Added 15 comprehensive unit tests in `tests/test_solvers.py`:
    - Registry and alias lookup validation (`test_solver_registry`).
    - Custom solver extensibility via subclassing and `@register_solver` (`test_user_defined_solver_subclass`, `test_user_defined_solver_decorator`).
    - Analytical benchmark test against the exact Fourier sine decay solution ($c(x, t) = \sin(\pi x / L) e^{-D (\pi/L)^2 t}$) verifying $< 5\times 10^{-3}$ maximum error across all 4 solvers (`test_analytical_dirichlet_benchmark`).
    - Total mass conservation test under zero-flux Neumann boundary conditions ($< 0.1\%$ relative variation) across all 4 solvers (`test_neumann_mass_conservation`).
    - CFL stability validation and error checking for the explicit solver (`test_explicit_cfl_validation`).
    - Multi-species diffusion simulation with differential diffusion coefficients (`test_multi_species_diffusion`).
    - First-order homogeneous reaction-diffusion kinetics ($R_A = -k c_A$) verified against analytical exponential decay (`test_reaction_diffusion`).
    - Input validation for non-monotonic grids and inverted time spans (`test_input_validation`).
  - Added baseline import and version smoke test `tests/test_basic.py`.
- **Documentation & Sphinx Framework (`docs/`)**:
  - Configured Sphinx documentation with Read the Docs theme (`sphinx_rtd_theme`), `autodoc`, `napoleon`, `viewcode`, `mathjax`, `nbsphinx`, and `myst_parser`.
  - Added native Markdown support via `myst_parser` with `dollarmath` extension for MathJax rendering of inline and display LaTeX math equations.
  - Symlinked root `CHANGELOG.md` to `docs/changelog.md` and added `Development & Releases` toctree section in `docs/index.rst` to publish the changelog directly on Read the Docs.
  - Added a dedicated "Solvers and Supported Options" quick-reference table in `docs/api.rst` summarizing all solvers, registered aliases, configuration parameters, and default values.
  - Added main landing page (`docs/index.rst`), installation guide (`docs/installation.rst`), and API reference (`docs/api.rst`).
  - Configured Read the Docs build specification (`.readthedocs.yaml`) targeting Ubuntu 24.04 and Python 3.11.
  - Added local documentation build automation scripts (`docs/Makefile`, `docs/make.bat`).
  - Added GitHub Actions workflow (`.github/workflows/docs.yml`) for automated documentation builds and ReadTheDocs deployment.
- **Interactive Tutorials & Benchmarking (`examples/`)**:
  - Added interactive tutorial notebook `examples/1d_diffusion_comparison.ipynb` (and mirrored in `docs/examples/1d_diffusion_comparison.ipynb`):
    - Comparative study of FTCS, BTCS, Crank-Nicolson, and SciPy Method of Lines against the exact analytical solution.
    - Demonstrations of numerical stability breakdown when exceeding the explicit CFL limit.
    - Step-size convergence scaling analysis and solver execution time profiling.
- **Project Infrastructure & Tooling**:
  - Configured standard `pyproject.toml` with PEP 621 package metadata, runtime dependencies (`numpy`, `scipy`, `matplotlib`), test dependencies (`pytest`), docs dependencies (`sphinx`, `sphinx-rtd-theme`, `sphinx-autodoc-typehints`, `nbsphinx`, `ipykernel`, `myst-parser`), and dev configurations for `ruff`, `black`, and `mypy`.
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) matrix-testing across Python 3.10, 3.11, 3.12, and 3.13 on Ubuntu.
  - Configured root `.gitignore` covering Python cache files, virtual environments, build artifacts, and Jupyter checkpoint files.
  - Added open-source BSD 3-Clause license (`LICENSE`).
  - Added automated scaffolding script `scaffold.sh`.
  - Added clear development notice banner in `README.md` highlighting the active `v3.0.0a1` architectural rewrite.

### Changed
- Refactored `README.md`:
  - Added active rewrite warning banner for `v3.0.0a1`.
  - Added comprehensive quickstart guide demonstrating 1D multi-species diffusion with `scipy_ivp`.
  - Added overview table of built-in numerical solvers (Method of Lines, Crank-Nicolson, Implicit, Explicit).
  - Added walkthrough on implementing and registering custom solvers with `BaseSolver`.
  - Linked interactive tutorial notebooks and Read the Docs documentation.
- Enhanced solver option discovery and type inspection across the entire `softpotato.solver` subpackage:
  - Added explicit typed `__init__` constructor methods for `ExplicitFiniteDifference`, `ImplicitFiniteDifference`, `CrankNicolson`, and `ScipyIVPSolver` forwarding `**options` to `BaseSolver`.
  - Added NumPy-formatted `Parameters` docstrings across all solvers, `BaseSolver.__init__`, `BaseSolver.solve`, and `get_solver`.
  - Updated runtime option resolution in solver engines so `kwargs` passed to `solve()` dynamically override instance options.
- Overhauled GitHub Actions CI workflow (`.github/workflows/ci.yml`) into a parallel multi-job pipeline:
  - Dedicated `lint-and-format` job running `ruff check .` and `ruff format --check .`.
  - Dedicated `type-check` job executing static type analysis with `mypy src`.
  - Expanded `test` matrix across Python 3.10, 3.11, 3.12, and 3.13 running unit tests, embedded docstring doctests (`pytest --doctest-modules src tests/`), statement test coverage (`--cov=softpotato`), and full Jupyter notebook execution (`pytest --nbmake examples/`).
  - Added `package-check` job validating sdist and wheel build generation (`python -m build`) and PyPI metadata/README rendering (`twine check dist/*`).
- Codebase formatting and type annotations:
  - Applied `ruff format` across all Python source modules, tests, markdown code snippets, and Jupyter notebook cells.
  - Alphabetized `__all__` exports across `softpotato` and `softpotato.solver`.
  - Migrated `Callable` typing import in `softpotato.solver.base` to `collections.abc.Callable` (PEP 585 / UP035).
  - Added typed scalar casting with `cast(Any, ic)` in `DiffusionProblem.get_initial_profile` for strict `mypy` compliance.
  - Removed unused imports in `tests/test_solvers.py` (`BoundaryCondition`) and `examples/1d_diffusion_comparison.ipynb` (`NeumannBC`).
- Fixed undefined analytical reference `c_exact_bench` in benchmark timing cell of `examples/1d_diffusion_comparison.ipynb`.
- Updated `docs/conf.py` and `docs/index.rst` to integrate `nbsphinx` and `myst_parser`, configuring `source_suffix` for both `.rst` and `.md`.
- Updated `pyproject.toml` with `[tool.mypy]` configuration and added `pytest-cov>=4.1.0`, `nbmake>=1.5.0`, and `mypy>=1.5.0` to optional dependencies.

### Fixed
- Fixed Python 3.10 class creation failure (`TypeError: ABCMeta.__new__() got multiple values for argument 'name'`) across all solvers by introducing `_BaseSolverMeta` on `BaseSolver` to safely extract `name` and `solver_name` class keyword arguments prior to delegating to `ABCMeta.__new__`.

### Removed
- Removed `.github/workflows/docs.yml` (obsolete standalone workflow referencing non-existent `readthedocs/upload-action@v1`; documentation builds and deployments are managed natively via Read the Docs GitHub webhook integration configured in `.readthedocs.yaml`).
- Removed legacy pre-3.0 codebase (monolithic `softpotato.core`, `softpotato.analytical`, `softpotato.geometry`, `softpotato.simulate`, `softpotato.techniques`, and `softpotato.kinetics`) to establish a modular, maintainable, and rigorously tested foundation.

## [3.0.0rc1] - 2026-09-20 [Yanked / Pre-Rewrite]

### Release Candidate 1
First release candidate (`rc1`) for **Soft Potato 3.0.0**, delivering an end-to-end simulation MVP for Cyclic Voltammetry alongside the core mechanisms, analytical solutions, and revised 4-stage development roadmap.

### Added
- **Core Module (`softpotato.core`)**:
  - Implemented the `Species` class with strict CGS unit enforcement:
    - Diffusion coefficient ($D$) in $\text{cm}^2/\text{s}$ ($D \ge 0$).
    - Bulk concentration ($c_{\text{bulk}}$) in $\text{mol}/\text{cm}^3$ ($c_{\text{bulk}} \ge 0$, $1\text{ mM} = 10^{-6}\text{ mol}/\text{cm}^3$).
    - Spatial concentration profile ($c_{\text{profile}}$) in $\text{mol}/\text{cm}^3$.
    - Ionic charge / valence number ($z$).
  - Added strict parameter validation for non-negative values, non-empty names, and integer charge.
  - Implemented profile memory management (`initialize_profile` to allocate 1D NumPy arrays and `reset` to restore initial bulk conditions).
  - Added `__hash__` method to `Species` allowing instances to be used in sets and as dictionary keys (e.g., in stoichiometry mappings).
  - Implemented `ElectrochemicalReaction` representing interfacial electron transfer ($O + n e^- \rightleftharpoons R$) with strict validation for `n_electrons`, `E0`, `kinetics`, and stoichiometric coefficients.
  - Implemented `ChemicalReaction` representing homogeneous bulk reactions ($\sum \nu_r R \rightleftharpoons \sum \nu_p P$) with support for kinetics and stoichiometry.
  - Implemented `Mechanism` container orchestrating reactions and species, featuring:
    - Automatic species discovery and order preservation from reactions.
    - Categorization into `e_reactions` (`electrochemical_reactions`) and `c_reactions` (`chemical_reactions`).
    - Indexing by species name (`mech["O"]` / `mech.get_species`) and reaction index (`mech[0]`).
    - Delegated profile initialization and reset across all species.
  - Implemented centralized physical constants module (`softpotato.core.constants`):
    - `FARADAY`: Faraday constant ($\approx 96485.33212\text{ C/mol}$).
    - `GAS_CONSTANT`: Molar gas constant $R$ ($\approx 8.3144626\text{ J/(mol}\cdot\text{K)}$).
    - `STANDARD_TEMPERATURE`: Standard temperature ($298.15\text{ K}$).
- **Analytical Solutions (`softpotato.analytical`)**:
  - Implemented closed-form, vectorized analytical solutions across modular subpackages:
    - **Potential Step & Chronocoulometry (`softpotato.analytical.techniques.step`)**:
      - `cottrell`: Transient current decay for planar diffusion.
      - `anson`: Cumulative charge transient for chronocoulometry including double-layer ($Q_{dl}$) and adsorption ($Q_{ads}$) charges in Coulombs.
      - `spherical_cottrell`: Transient current at a spherical electrode with area derived from radius $r_0$.
    - **Voltammetry (`softpotato.analytical.techniques.voltammetry`)**:
      - `randles_sevcik`: Peak current for reversible electron transfer.
      - `randles_sevcik_irreversible`: Peak current for totally irreversible electron transfer.
      - `peak_potential_irreversible`: Peak potential shift for totally irreversible electron transfer.
    - **Microelectrode Geometries (`softpotato.analytical.geometry.microelectrodes`)**:
      - `steady_state_microdisc`: Saito equation for steady-state limiting current at an inlaid microdisc with radius $a$.
      - `steady_state_microhemisphere`: Steady-state limiting current at a microhemisphere electrode.
    - **Hydrodynamics (`softpotato.analytical.geometry.hydrodynamics`)**:
      - `levich`: Mass-transport limiting current at a rotating disk electrode (RDE).
      - `koutecky_levich`: Net steady-state current combining kinetic and mass-transport limitations.
    - **Kinetics & Reversibility (`softpotato.analytical.kinetics.reversibility`)**:
      - `nicholson_psi`: Nicholson kinetic reversibility parameter $\Psi$.
    - **Reaction Mechanisms (`softpotato.analytical.kinetics.mechanisms`)**:
      - `catalytic_current`: Steady-state catalytic current for an $EC'$ mechanism (with alias `catalytic_current_ec_prime`).
  - Added top-level facade imports in `softpotato.analytical` for direct access to all equations.
- **Simulation MVP (`softpotato.simulate`, `softpotato.geometry`, `softpotato.techniques`, `softpotato.kinetics`)**:
  - Implemented the end-to-end numerical simulation pipeline for Cyclic Voltammetry:
    - `Grid` ABC and `UniformGrid` representing uniform 1D spatial discretization with strict CGS coordinates.
    - `Electrode` ABC and `PlanarElectrode` representing planar macroelectrodes with 1D spatial Laplacian differential operators.
    - `Technique` ABC and `CyclicVoltammetry` generating exact triangular potential waveforms and sampling parameters ($dt$, duration).
    - `KineticsModel` ABC and `ButlerVolmer` implementing potential-dependent forward/backward heterogeneous rate constants and surface reduction flux.
    - Vectorized Explicit Finite Difference (EFD) engine (`explicit_diffuse_step`) and surface flux boundary condition solver (`update_surface_concentrations`).
    - High-level `Solver` orchestrator supporting automated CFL stability sub-stepping (`auto_substep=True`) with informative `UserWarning`.
    - `SimulationResult` dataclass storing time, potential, current, concentration profiles, and a built-in `.plot()` method.
  - Added interactive tutorial notebook `examples/cv_simulation.ipynb` demonstrating end-to-end CV simulation, Butler-Volmer kinetics theory, and Randles-Sevcik benchmarking.
  - Added integration benchmark test `tests/test_cv_benchmark.py` validating simulated reversible peak current against analytical `randles_sevcik` (< 3% error).
- **Module Scaffolding**:
  - Established subpackage hierarchy:
    - `softpotato.geometry`: Grids (`grids.py`) and electrode geometries (`electrodes.py`).
    - `softpotato.simulate`: PDE solvers (`solver.py`), explicit finite difference (`efd.py`), and implicit finite difference (`ifd.py`).
    - `softpotato.techniques`: Electrochemical waveforms including voltammetry (`voltammetry.py`) and potential step (`step.py`).
    - `softpotato.kinetics`: Pluggable kinetic models (in development).
- **Testing & Quality Infrastructure**:
  - Added comprehensive test suite in `tests/test_core.py` covering `Species`, `ElectrochemicalReaction`, `ChemicalReaction`, and `Mechanism` (initialization, validations, memory allocations, property setters, representations, indexing, and stoichiometry matrix).
  - Configured `pyproject.toml` with package metadata, runtime dependencies (`numpy`, `scipy`, `matplotlib`), test dependencies (`pytest`), and tool configurations for `black`, `ruff`, and `mypy`.
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) matrix testing across Python 3.10, 3.11, and 3.12.
  - Added root `.gitignore` covering Python cache directories, virtual environments, build artifacts, and editor settings.
- **Licensing & Project Metadata**:
  - Adopted the open-source **BSD 3-Clause License** (`LICENSE`).
  - Added license metadata and PyPI classifier (`License :: OSI Approved :: BSD License`) in `pyproject.toml`.
- **Documentation & Sphinx Framework (`docs/`)**:
  - Configured Sphinx documentation with Read the Docs integration (`.readthedocs.yaml`) using the `sphinx_rtd_theme`.
  - Added Sphinx configuration (`docs/conf.py`) integrating `autodoc`, `napoleon`, `mathjax`, `viewcode`, `todo`, `sphinx-math-dollar`, and `nbsphinx`.
  - Added main landing page (`docs/index.rst`), roadmap document (`docs/roadmap.rst`), and API reference structure (`docs/api/index.rst`).
  - Added Sphinx `.. warning::` and `.. todo::` directives across all placeholder/scaffolding modules (`geometry`, `simulate`, `techniques`, `kinetics`) to clearly signal their development status in generated documentation.
  - Added `docs/Makefile` and `docs/make.bat` build scripts for local documentation generation.
  - Added dedicated license page (`docs/license.rst`) embedding the root `LICENSE` file into the Sphinx documentation table of contents.
  - Added automated `docs` build validation job in GitHub Actions workflow (`.github/workflows/ci.yml`).
- **Interactive Tutorials & Examples (`examples/`)**:
  - Added comprehensive tutorial notebook `examples/mechanisms.ipynb` covering mechanism definitions in Soft Potato 3.0:
    - Single-electron transfers ($E$).
    - Chemical equilibria ($C$).
    - Coupled EC and multi-step ECE mechanisms.
    - Disproportionation pathways (ECE vs. DISP competition with multi-reactant/product stoichiometry, and bimolecular radical disproportionation $2R \rightleftharpoons O + Z$).
    - Solver integration features (profile initialization `initialize_profiles`, reset `reset`, dictionary indexing, and stoichiometry matrix inspection).
  - Linked interactive tutorial notebook into Sphinx documentation via `docs/examples/mechanisms.nblink`.
- **Dependencies (`pyproject.toml`)**:
  - Added optional `docs` dependency group (`sphinx`, `sphinx-rtd-theme`, `sphinx-autodoc-typehints`, `nbsphinx`, `nbsphinx-link`, `ipython`, `ipykernel`).

### Changed
- Configured Black `target-version = ["py310"]` in `pyproject.toml` to ensure consistent AST checks across the Python 3.10–3.12 CI matrix.
- Structured package exports in `softpotato/__init__.py` using explicit `__all__` declarations.
- Cleaned up docstrings in `src/softpotato/core/reactions.py` by removing redundant `Attributes:` sections to optimize Sphinx autodoc rendering.
- Formatted `src/softpotato/core/reactions.py` and `tests/test_core.py` to adhere strictly to Ruff formatting and lint rules.

### Removed
- Removed legacy `src/softpotato/core/README.md` in favor of Sphinx documentation.

