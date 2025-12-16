# SoftPotato v3.0

**SoftPotato** is a Python library for building electrochemical simulations.  
Version 3.0 is a ground-up rewrite with a strict milestone-driven roadmap and strong guarantees around API stability, testing, and documentation.

This repository currently contains **M0** and **M1** only.

---

## Project Status

| Milestone | Status | Description |
|---------|--------|------------|
| M0 | ✅ Complete | Project skeleton, packaging, CI, versioning |
| M1 | ✅ Complete | Time grids and potential waveform generators |
| M2+ | ⏳ Planned | Transport, kinetics, mechanisms, solvers |

Current release stage: **pre-alpha**  
No scientific simulation is performed yet.

---

## Supported Python Versions

* Python **3.10+**

---

## Installation

### Development / local install

```bash
pip install -e ".[dev]"
```
### User install (once published)
```bash
pip install softpotato
```
---

## Versioning

SoftPotato follows **semantic versioning**.

### Runtime version check

```python
import softpotato
print(softpotato.__version__)
```

---

## What Exists Today

### M0 – Project Skeleton

M0 establishes the foundations required for all future work.

#### Included in M0
- Installable Python package (pyproject.toml)
- Version metadata wired to distribution
- Continuous integration (tests + lint)
- Test harness and smoke tests
- Documentation skeleton
- Roadmap and architecture documents

#### Explicitly excluded in M0
- No electrochemistry
- No numerics
- No solvers
- No plotting
- No CLI

---

### M1 – Time Grids & Potential Waveforms

M1 introduces the **first scientific building blocks**, without solving any equations.

#### Time Grids

Time grids are validated, immutable representations of experiment time.

##### Public API
- uniform_time_grid(...)

##### Characteristics
- Monotonic increasing time
- Explicit time step handling
- Validation enforced at construction

---

#### Potential Waveforms

Waveforms generate **potential vs time** arrays on a supplied time grid.

##### Public API
- lsv(...) — linear sweep voltammetry
- cv(...) — cyclic voltammetry

##### Characteristics
- Stateless
- Deterministic
- Fully defined by parameters + time grid

---

## Public API (as of M1)

Only the following symbols are considered user-facing and stable:

- softpotato.__version__
- softpotato.uniform_time_grid
- softpotato.lsv
- softpotato.cv

Anything else is internal and may change without notice.

---

## Minimal Working Example

### Example usage

Import SoftPotato, create a time grid, and generate a waveform:
```python
import softpotato as sp  
tg = sp.uniform_time_grid(dt=0.01, n=5)  
waveform = sp.lsv(0.0, 1.0, time_grid=tg)  
print(waveform.shape)  
print(waveform[:2])
```
### Expected behavior
- waveform is a NumPy array
- Shape is (n, 2)
- Column 0: time
- Column 1: potential

---

## Running Tests

### Run the test suite
```bash
pytest
```
All tests must pass before any release tag is created.

---

## Documentation

Documentation lives in the docs/ directory and is built with **MkDocs**.

### Local documentation preview
```bash
mkdocs serve
```
---

## Project Structure

### Repository layout
```
src/softpotato/  
  __init__.py  
  timegrid.py  
  waveforms.py  
  validation.py  

tests/  
  test_m0_skeleton.py  
  test_m1_timegrid.py  
  test_m1_waveforms.py  

docs/  
README.md  
ROADMAP.md  
ARCHITECTURE.md  
CHANGELOG.md  
```
---

## Roadmap Discipline

SoftPotato follows a **milestone-gated development model**:

- No milestone advances without:
  - Passing tests
  - Updated documentation
  - Explicit public API declaration
- No features are backfilled into earlier milestones
- Each release maps directly to a roadmap milestone

See ROADMAP.md for details.

---

## License

MIT License  
See LICENSE for details.

