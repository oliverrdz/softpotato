# Concepts

Electrochemical theory and numerical modelling fundamentals behind Soft Potato.

These conceptual guides bridge **laboratory intuition** (potentials, currents, diffusion layers, and reaction mechanisms) with **numerical simulation** (meshes, finite-difference operators, time integrators, and analytical benchmarks)—without burying you in boilerplate code.

## Electrochemistry

<div class="grid cards" markdown>

-   :material-flash: **[Electrode Kinetics](electrochemistry/kinetics.md)**

    ---

    Nernst equilibrium vs. Butler–Volmer and Tafel kinetics ($E^{0\prime}$, $k^0$, $\alpha$), reversibility regimes ($\Lambda$, $\Psi$), and IUPAC sign conventions.

-   :material-waves: **[Mass Transport](electrochemistry/transport.md)**

    ---

    Planar, spherical, cylindrical, and microelectrode diffusion, plus forced convection–diffusion at a Rotating Disk Electrode (RDE).

-   :material-molecule: **[Reaction Mechanisms](electrochemistry/mechanisms.md)**

    ---

    Coupled heterogeneous electron transfer and homogeneous solution chemistry: `E`, `EC`, `EE`, and catalytic `EC'` mechanisms.

</div>

## Numerical Modelling

<div class="grid cards" markdown>

-   :material-grid: **[Spatial Grids & Meshes](numerics/grids.md)**

    ---

    Discretizing the diffusion layer $\delta = 6\sqrt{D t_{\max}}$ with `Uniform1DMesh`, `ExponentialMesh`, and `GeometricExpandingMesh`.

-   :material-cog-sync: **[Time-Stepping & Solvers](numerics/solvers.md)**

    ---

    Explicit (FTCS), Implicit (BTCS), Crank–Nicolson, and Method-of-Lines (`ScipyIVPSolver`), CFL stability, and piecewise `event_times` integration.

-   :material-check-decagram: **[Benchmarking & Validation](numerics/benchmarking.md)**

    ---

    Verifying spatial/temporal convergence orders and validating solvers against closed-form analytical solutions via `softpotato.analytical.benchmark`.

</div>

## Related Documentation

- **[User Guide](../user-guide/index.md)**: Task-oriented guides for `simulate()`, waveforms, diagnostics, plotting, and curve fitting.
- **[Tutorials](../tutorials/index.md)**: Interactive Jupyter notebooks ([Chronoamperometry Quickstart](../tutorials/start/ca_quickstart.ipynb) · [Cyclic Voltammetry Quickstart](../tutorials/start/cv_quickstart.ipynb)).
- **[API Reference](../api/index.md)**: Full class and function signatures across all Soft Potato modules.
