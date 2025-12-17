# SoftPotato v3.0

SoftPotato is a **modular electrochemistry simulation toolkit** designed for scientists who want transparent, deterministic, and testable numerical experiments.

Version 3.0 is a ground-up rewrite focused on **clean scientific APIs**, strict validation, and a milestone-driven roadmap.

This README documents the project state at **Milestone M0 and Milestone M1**.

---

## Project status

- Version: v3.0 (pre-release)
- Current milestone: **M1 – Time Grids & Potential Waveforms**
- License: MIT
- Intended audience: electrochemists, physical chemists, scientific Python users

---

## Installation

### Development install

```bash
pip install -e ".[dev]
```

This installs SoftPotato in editable mode together with linting, typing, and test dependencies.

---

## Milestone overview

### M0 – Project Skeleton (Completed)

M0 establishes the **non-scientific foundation** of the project.

#### Goals

- Create a valid, installable Python package
- Establish versioning, licensing, and repository structure
- Set up CI-compatible tooling (lint, type-check, tests)
- Provide no scientific behavior yet

#### What exists in M0

- Importable `softpotato` package
- Exposed version string
- Project metadata in `pyproject.toml`
- Empty but structured `src/`, `tests/`, and `docs/`
- ROADMAP and CHANGELOG scaffolding

#### What does NOT exist in M0

- No electrochemical models
- No solvers
- No waveforms
- No time grids
- No numerical results

M0 is purely structural.

---

### M1 – Time Grids & Potential Waveforms (Completed)

M1 introduces the **first scientific API**: deterministic experiment definitions.

The goal of M1 is to provide a **single, validated representation of time and potential** that later solvers can consume.

---

## Scientific API (as of M1)

All scientific outputs are NumPy arrays of shape `(n, 2)`:

- Column 0: potential `E`
- Column 1: time `t`

Time is **always strictly increasing**.

---

### Potential waveforms

#### Linear Sweep Voltammetry (LSV)

```python
lsv(E_start, E_end, dE, scan_rate)
```

- Users specify potential resolution via `dE`
- Time step is derived internally:

dt_step = dE / scan_rate

- Exact endpoint enforcement
- Supports increasing and decreasing sweeps

---

#### Cyclic Voltammetry (CV)

```python
cv(E_start, E_vertex, scan_rate, dE, E_end=None, cycles=1)
```

- Forward and reverse scans constructed deterministically
- No duplicated vertex points
- No duplicated samples between cycles
- Time strictly increasing across cycles

---

#### Potential Step

```python
step(E_before, E_after, dt, t_end)
```

- Uniform time grid with explicit `dt`
- Instantaneous step at `t = 0`
- No snapping to `t_end`
- Final time sample satisfies:

t[-1] ≤ t_end and t[-1] > t_end − dt

---

## Example usage

### Linear sweep

```python
import softpotato as sp
w = sp.lsv(0.0, 1.0, dE=0.25, scan_rate=0.5)
print(w)
```

---

### Cyclic voltammetry

```python
import softpotato as sp
w = sp.cv(0.0, 1.0, scan_rate=0.5, dE=0.5)
print(w)
```

---

### Potential step

```python
import softpotato as sp
w = sp.step(0.0, 1.0, dt=0.01, t_end=0.04)
print(w)
```

---

## Validation guarantees (M1)

All waveform constructors enforce:

- Finite numeric inputs
- Positive `dE`, `scan_rate`, `dt`, and `t_end`
- Integer `cycles ≥ 1`
- Deterministic output
- Strictly increasing time

Invalid inputs raise clear `TypeError` or `ValueError`.

---

## What is intentionally out of scope (≤ M1)

- Diffusion solvers
- Butler–Volmer kinetics
- Double-layer capacitance
- iR drop
- Adaptive or non-uniform grids
- Mechanism definitions (E, EC, CE, etc.)

These are planned for later milestones.

---

## Roadmap snapshot

- M0: Project skeleton ✔
- M1: Time grids & potential waveforms ✔
- M2: 1D diffusion solvers
- M3: Electrode kinetics
- M4: Mechanism composition
- M5: Validation against analytical solutions

See `ROADMAP.md` for full details.

---

## Documentation

- `docs/waveforms.md` – Detailed waveform behavior and examples
- `ROADMAP.md` – Milestone plan and scope
- `CHANGELOG.md` – Versioned changes

---

## Development workflow

### Lint

```bash
ruff check .
ruff format .
```

### Type-check

```bash
mypy .
```

### Tests

```bash
pytest
```

---

## Design philosophy

SoftPotato prioritizes:

- Determinism over convenience
- Explicit validation over silent assumptions
- Electrochemistry-first terminology
- Testability at every milestone

The project is designed so each milestone produces **scientifically meaningful, reviewable behavior**.

---

## License

MIT License. See `LICENSE` for details.

