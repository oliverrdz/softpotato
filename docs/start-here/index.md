# Start Here

<div class="sp-section-banner" markdown>
<div markdown>
**New to Soft Potato?** You do not need to write differential equations or build numerical meshes to get started. Use the decision table below to jump straight to the right API and documentation section for your goal.
</div>
<img src="../assets/images/chip-lab.png" alt="Chip in the Laboratory" />
</div>

## Which Tool Should I Use?

Soft Potato is organized in layers—from one-line closed-form equations up to custom multi-species PDE solvers. Pick the row that matches what you want to do:

| If you want to… | Use | Typical inputs | Where to go next |
| :--- | :--- | :--- | :--- |
| **Compute an exact theoretical baseline** (Cottrell, Randles–Ševčík, Levich, Butler–Volmer, microelectrodes, SECM) | `softpotato.analytical` | Physical constants ($D$, $c^*$, $n$, $A$, $v$, $\omega$) | [User Guide](../user-guide/index.md) · [API Reference](../api/index.md) |
| **Extract $D$, $k^0$, $\alpha$, or $RG$ from experimental data** with confidence intervals and $R^2$ | `softpotato.analytical.fitting` | Experimental arrays (`t`, `i`, `E`, `v`, `omega`) | [User Guide](../user-guide/index.md) · [API Reference](../api/index.md) |
| **Simulate a standard electrochemical experiment** (CA, LSV, CV, SCV, DPV, SWV, CP, RDE) with automatic plotting and diagnostics | `softpotato.techniques.simulate()` | Waveform object (e.g. `CyclicVoltammetry(...)`) + physical parameters | [User Guide](../user-guide/index.md) · [Tutorials](../tutorials/index.md) |
| **Build a custom reaction–diffusion model** with bespoke grids, geometries, boundary conditions, or PDE solvers | `softpotato.solver.DiffusionProblem()` | Spatial grid (`grid`), `diffusivity`, `boundary_conditions`, `initial_conditions`, `reactions` / `mechanism` | [Concepts](../concepts/index.md) · [Tutorials](../tutorials/index.md) |

!!! chip "Chip's Tip"
    If you are unsure where to start, begin with **`softpotato.analytical`** for quick reference curves or **`softpotato.techniques.simulate()`** for full voltammetric and transient simulations. Drop down to **`DiffusionProblem()`** only when you need custom meshes, non-standard boundary conditions, or custom solver settings.

---

## Try Soft Potato Right Now

=== "Zero-Install (Google Colab)"

    Run Soft Potato directly in your browser with nothing to install on your computer:

    [:octicons-play-24: Launch Soft Potato v3 Starter in Google Colab](https://colab.research.google.com/drive/1w9NfLV2j8ZZjjcVhAiPcogVnKb65wqE2){ .md-button .md-button--primary target="_blank" rel="noopener" }

=== "Local Install (`pip`)"

    Install the latest stable release from [PyPI](https://pypi.org/project/softpotato/):

    ```bash
    pip install softpotato
    ```

    Verify your installation in Python:

    ```python
    import softpotato as sp

    print(sp.__version__)
    ```

---

## Where to Go Next

<div class="grid cards" markdown>

-   :material-book-open-variant: **User Guide**

    ---

    Task-oriented guides for running simulations, plotting results, and running diagnostics.

    [:octicons-arrow-right-24: Open User Guide](../user-guide/index.md)

-   :material-atom: **Concepts**

    ---

    Electrochemical theory and numerical modelling fundamentals behind Soft Potato.

    [:octicons-arrow-right-24: Open Concepts](../concepts/index.md)

-   :material-notebook-outline: **Tutorials**

    ---

    Interactive Jupyter notebooks organized by technique, mechanism, and module.

    [:octicons-arrow-right-24: Open Tutorials](../tutorials/index.md)

-   :material-code-braces: **API Reference**

    ---

    Complete reference for `analytical`, `techniques`, `reactions`, `grid`, `geometry`, and `solver`.

    [:octicons-arrow-right-24: Open API Reference](../api/index.md)

</div>
