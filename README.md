# 🥔 SoftPotato v3.0

**SoftPotato** is an open-source **electrochemistry simulator and toolbox** focused on
**mass transport and electrode kinetics**, designed for researchers, educators,
and developers working with **voltammetry and diffusion–reaction systems**.

Version **v3.0** is a **ground-up rewrite** centered on a **robust 1D planar finite-difference engine**
with **explicit electrochemical mechanisms** (E / EC / CE / EE / general E–C networks).

> _A software potentiostat for theory._

---

## 🚧 Project Status

⚠️ **Alpha (v3.0.0)**  
The numerical core is under active development. APIs may change until v3.0 stabilizes.

Current state:
- ✔️ Packaging, linting, typing, and test scaffolding
- ✔️ CI-ready project layout
- 🚧 Numerical solvers and mechanisms in progress
- ❌ Not yet feature-complete vs SoftPotato v2.x

See the **Roadmap** section for detailed milestones and acceptance criteria.

---

## ✨ Key Features (v3 Design Goals)

- **1D Planar diffusion** (Fick’s laws, finite differences)
- **Implicit time integration**
  - Crank–Nicolson (default)
  - Backward Euler
- **Electrochemical mechanisms**
  - E (Nernst, Butler–Volmer)
  - EC / CE
  - EE and arbitrary E–C networks
- **Waveform-driven simulations**
  - Cyclic voltammetry (CV)
  - Linear sweep voltammetry (LSV)
  - Potential steps
- **Multi-species support**
- **Physically explicit boundary conditions**
- **Scientifically testable outputs**
  - mass balance
  - limiting cases
  - convergence checks

Planned (post v3.0):
- Additional geometries (spherical, thin-layer)
- IR drop and double-layer capacitance
- Numba / JAX acceleration paths
- GUI / wizard-based experiment builders

---

## 📦 Installation

### From source (recommended)

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e .
```

## Development installation
```bash
pip install -e ".[dev,docs]"
```
Python >= 3.10 is required.

---
# Development Workflow
```bash
# Run tests
pytest

# Lint & formatting
ruff check .
black .

# Static typing
mypy src/softpotato
```
Linting and typing are intentionally strict avoid silent numerical bugs.

---

# Project Philosophy

SoftPotato v3 prioritizes physical transparency and numerical robustness.

Core principles:
* Explicit mass transport
* Explicit reaction steps
* Explicit assumptions
* Stable, implicit solvers
* Reproducible simulations

If a model cannot be explained on a whiteboard, it does not belong in the core.

# Roadmap (Soft Potato v3.0)
Development is milestone-driven with clear acceptance criteria:

* SP3-M0 — Repository scaffolding & CI
* SP3-M1 — Potential waveforms
* SP3-M2 — Planar 1D grid & diffusion operator
* SP3-M3 — Implicit diffusion engine
* SP3-M4 — Reversible E (Nernst)
* SP3-M5 — Butler–Volmer kinetics
* SP3-M6 — Bulk chemical steps (EC / CE)
* SP3-M7 — Mechanism compiler
* SP3-M8 — Multi-electron systems (EE)
* SP3-M9 — General E–C reaction networks

See [ROADMAP.md](https://github.com/oliverrdz/softpotato/blob/v3.0/ROADMAP.md)
 for full details, tests, and example plots.

 ---

 # Documentation
 Documentation will be built using MkDocs once the public API stabilizes.

Planned sections:
* Governing equations & assumptions
* Numerical methods
* Mechanism builder API
* Validation and benchmark cases

---

# Contributing
Contributions are welcome once core APIs stabilize.

Guidelines:
Open an issue before major changes
Follow numerical and physical conventions
Include tests and justification
Keep APIs explicit and readable
CI enforces linting, formatting, and typing.

---

#  License

MIT License
© Oliver Rodriguez

---

# Name Origin

SoftPotato is a nod to:
* software potentiostats
* potato batteries
* and solving hard electrochemical problems the soft way

---

# Links

* Homepage: https://softpotato.xyz
* Repository: https://github.com/oliverrdz/softpotato
* Issues: https://github.com/oliverrdz/softpotato/issues
