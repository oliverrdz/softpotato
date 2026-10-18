# Tutorials & Learning Guides

<div class="sp-section-banner" markdown>
<div markdown>
Welcome to the **Soft Potato Learning Hub**! Explore hands-on tutorials covering **electrochemistry simulations**, scientific **Python programming**, and the **Linux** command line for students, researchers, and computational scientists. This section is in active development and will be updated continuously.
</div>
<img src="../assets/images/chip-coding.png" alt="Chip — Code & Mesh Analyst" />
</div>

---

## Electrochemistry Simulations

Interactive simulations, numerical solvers, and analytical benchmarking with Soft Potato:

- [**Cottrell Chronoamperometry**](electrochemistry/cottrell.ipynb) — 1D potential-step diffusion simulation, concentration profiles, Cottrell plots, and diffusivity extraction.
- [**Cyclic Voltammetry**](electrochemistry/cyclic_voltammetry.ipynb) — Reversible electron transfer ($O + e^- \rightleftharpoons R$), triangular sweep generation, Randles–Ševčík diagnostics, and peak separation analysis.

---

## Python

Foundational scientific programming tutorials designed for researchers and engineers:

- [**Installing Python & First Simulation**](python/installing-python.md) — Installing official Python across operating systems, setting up an isolated virtual environment, comparing IDEs, and running an analytical chronoamperometry test in IDLE to verify your setup.
- [**Python Fundamentals: Variables & Functions**](python/variables-and-functions.md) — Variables, physical constants, typing, scientific functions, and calculating electrochemical parameters like thermal voltage.
- [**Numerical Computing with NumPy**](python/numpy-arrays.md) — Utilizing `np.ndarray` for vectorized math, evaluating equations without slow loops, and handling multidimensional concentration grids.
- [**Data Visualization with Matplotlib**](python/matplotlib-plotting.md) — Generating publication-quality figures, transient curves, cyclic voltammograms, custom styling, multi-panel subplots, and vector graphics export.

---

## Linux

Essential command-line skills for running simulations, managing data, and remote computing:

- [**Navigating the Filesystem**](linux/filesystem-navigation.md) — The Linux directory hierarchy, essential navigation commands (`pwd`, `ls`, `cd`, `mkdir`), and tab autocompletion.
- [**File Management & Data Inspection**](linux/file-management.md) — Creating, moving, copying files safely, fast data inspection with `head`, `tail`, `cat`, and text search with `grep`.
- [**Running Scripts & Redirection**](linux/running-scripts.md) — Running scientific Python scripts from the terminal, standard I/O redirection (`>`, `2>&1`), and managing background jobs.
- [**Remote Computing with SSH & Rsync**](linux/ssh-remote-access.md) — Connecting securely to remote lab workstations with SSH keys, persistent sessions with `tmux`, and syncing datasets with `rsync`.
