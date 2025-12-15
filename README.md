# SoftPotato v3.0

**SoftPotato** is an open-source electrochemical simulation framework written in Python.

This repository currently hosts **SoftPotato v3.0 – M0**, which is a **project skeleton release**.
No scientific simulation functionality is implemented yet.

> ⚠️ This is an **alpha-stage structural milestone**, intended to validate packaging, testing,
> documentation structure, and development workflow.

---

## What Is SoftPotato?

SoftPotato aims to become a modular, extensible platform for simulating
electrochemical experiments (e.g. cyclic voltammetry, chronoamperometry)
using physically motivated numerical models.

In **M0**, SoftPotato does *not* perform simulations.
Only the project structure is established.

---

## Current Status (M0: Project Skeleton)

**Available today**
- Python package installs successfully
- Version metadata is exposed
- Test suite runs and passes
- Documentation, roadmap, and architecture files exist

**Not available yet**
- No electrochemical mechanisms
- No diffusion solvers
- No kinetics models
- No waveform generators
- No plotting
- No CLI or GUI

This is intentional.

---

## Quickstart

### Installation (development mode)
```bash
pip install -e .
```

### Verify installation
```bash
python -c "import softpotato; print(softpotato.__version__)"
```
Expected output (example):
```bash
v3.0.0-alpha0
```
### Run tests
```bash
pytest
```
All tests should pass.

---

## Developer Setup

### Requirements

- Python 3.10 or newer
- pip
- virtualenv or equivalent (recommended)

### Clone the repository
```bash
git clone https://github.com/<your-org>/softpotato.git  
cd softpotato
```
### Create and activate a virtual environment
```bash
python -m venv .venv  
source .venv/bin/activate
```
### Install in editable mode with dev dependencies
```bash
pip install -e ".[dev]"
```
### Run the test suite
```bash
pytest
```
---

## Public API (Current)

There is **no stable scientific API** in M0.

The only supported public interaction is:

import softpotato

No classes, solvers, functions, or CLI entry points are exposed yet.

---

## Documentation Structure

- `README.md` — project overview and quickstart
- `ROADMAP.md` — milestone-based development plan
- `ARCHITECTURE.md` — high-level system design
- `docs/` — future user and developer documentation

---

## Roadmap

Development follows **explicit milestones**.

You are currently looking at:

**M0 – Project Skeleton**

Upcoming milestones will introduce:
- Electrochemical mechanisms
- Numerical solvers
- Scientific APIs
- Visualization tools

See `ROADMAP.md` for details.

---

## Common Pitfalls

- **Expecting simulation features**  
  None exist yet. M0 is structural only.

- **Assuming API stability**  
  The API will change rapidly during alpha milestones.

- **Using SoftPotato for research now**  
  Do not use M0 for scientific work.

---

## Versioning and Stability

SoftPotato follows semantic versioning with milestone-based pre-releases.

- `v3.0.0-alpha0` — structural skeleton
- Future alpha versions may break APIs without notice

---

## License

See `LICENSE` for details.

