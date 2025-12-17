# SoftPotato v3.0 Documentation

Welcome to the documentation for **SoftPotato v3.0**, a modular electrochemistry simulation toolkit focused on **deterministic, validated, and testable numerical experiments**.

This documentation reflects the project state at **Milestones M0 and M1**, and provides a roadmap for upcoming scientific capabilities.

---

## What is SoftPotato?

SoftPotato is a ground-up rewrite of the SoftPotato electrochemistry framework with the following goals:

- Clear, minimal scientific APIs
- Deterministic numerical behavior
- Strict validation and error handling
- Modular design aligned with electrochemistry literature
- Milestone-driven development (each milestone is usable and testable)

SoftPotato is intended for **scientists**, not just developers. You should be able to understand what the code does without reading the source.

---

## Current project state

- Version: v3.0 (pre-release)
- Current milestone: **M1 – Time Grids & Potential Waveforms**
- License: MIT
- Language: Python (NumPy-based)

---

## Milestone M0 – Project Skeleton

**Status: Complete**

M0 establishes the **non-scientific foundation** of SoftPotato.

### What M0 provides

- A valid, installable Python package
- Versioning and package metadata
- Repository structure (`src/`, `tests/`, `docs/`)
- CI-ready tooling for linting, typing, and testing
- No scientific functionality

### What M0 does not provide

- No electrochemical models
- No waveforms
- No time grids
- No numerical simulations

M0 exists to ensure that everything built later has a solid, maintainable base.

---

## Milestone M1 – Time Grids & Potential Waveforms

**Status: Complete**

M1 introduces the **first scientific API** in SoftPotato: deterministic definitions of electrochemical experiments.

The goal of M1 is to define **time and potential in a single, validated representation** that later solvers (diffusion, kinetics, mechanisms) can consume.

---

### Scientific outputs (M1)

All waveforms return:

- A NumPy array of shape `(n, 2)`
- Column 0: potential `E`
- Column 1: time `t`
- `t` is always strictly increasing

---

### Available waveforms

#### Linear Sweep Voltammetry (LSV)

### lsv(E_start, E_end, dE, scan_rate)

- Potential resolution defined by `dE`
- Time step derived as:

dt_step = dE / scan_rate

- Exact endpoint enforcement
- Supports increasing and decreasing sweeps

---

#### Cyclic Voltammetry (CV)

### cv(E_start, E_vertex, scan_rate, dE, E_end=None, cycles=1)

- Deterministic forward and reverse scans
- No duplicated vertex points
- No duplicated samples between cycles
- Time strictly increasing across cycles

---

#### Potential Step

### step(E_before, E_after, dt, t_end)

- Uniform time grid with explicit `dt`
- Instantaneous step at `t = 0`
- No snapping to `t_end`

---

## Design guarantees (M1)

- Resolution-driven APIs (`dE` or `dt`, not array length)
- Internally derived time grids
- Strict input validation
- Deterministic output
- No hidden smoothing or adaptive behavior

---

## Documentation structure

- `index.md` – Project overview and roadmap (this page)
- `waveforms.md` – Detailed waveform behavior and examples
- `ROADMAP.md` – Full milestone plan
- `CHANGELOG.md` – Versioned changes

---

## Roadmap summary

SoftPotato v3.0 is developed in **incremental, reviewable milestones**.

### Completed

- **M0** – Project skeleton
- **M1** – Time grids & potential waveforms

---

### Planned

- **M2 – Diffusion**
  - 1D diffusion solvers
  - Finite-difference discretization
  - Boundary conditions tied to waveform output

- **M3 – Electrode kinetics**
  - Butler–Volmer and related rate laws
  - Coupling between surface concentration and current

- **M4 – Mechanism composition**
  - E, EC, CE, ECE mechanisms
  - Modular reaction graphs

- **M5 – Validation**
  - Comparison against analytical solutions
  - Regression tests against literature benchmarks

Each milestone builds **only** on completed, validated behavior from previous milestones.

---

## Getting started

### Development install

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Build and preview docs

```bash
mkdocs serve
```

---

## Philosophy

SoftPotato is designed around the idea that:

- Numerical experiments should be reproducible
- APIs should reflect physical assumptions
- Validation should happen at the boundary, not downstream
- Scientific code deserves the same rigor as production software

If you are looking for a transparent, extensible foundation for electrochemical simulations, you are in the right place.

