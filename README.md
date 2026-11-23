# Soft Potato

[![PyPI](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1aHJVZaPPRcIK2uy6OgobL8FOKzcMjjLx?usp=drive_link)
[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--4bs3w-orange.svg)](https://doi.org/10.26434/chemrxiv-2022-4bs3w)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

**Full documentation**: [https://softpotato.xyz/](https://softpotato.xyz/) | **Try Live**: [Google Colab Starter](https://drive.google.com/drive/folders/1aHJVZaPPRcIK2uy6OgobL8FOKzcMjjLx?usp=drive_link) | **Roadmap**: [ROADMAP.md](ROADMAP.md) | **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Installation

### Zero-Install Browser Trial (Google Colab)

Run Soft Potato directly in your browser with no local Python setup required:
- [**Launch Soft Potato v3 Starter in Google Colab**](https://drive.google.com/drive/folders/1aHJVZaPPRcIK2uy6OgobL8FOKzcMjjLx?usp=drive_link)

### Stable Release (v3.4.2)

Soft Potato v3.4.2 is available on [PyPI](https://pypi.org/project/softpotato/) and can be installed via pip:

```bash
pip install softpotato
```

### Development Version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/oliverrdz/softpotato.git
```

Or clone the repository and install in development mode:

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[dev,docs]"
```

## Development Roadmap & Milestones

Soft Potato follows a staged milestone development roadmap for the 3.x series, maintaining strict backwards compatibility across releases:

| Milestone | Target | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Current Baseline** | `v3.0.0` | 1D multi-species diffusion solvers (`scipy_ivp`, Crank–Nicolson, BTCS, FTCS) | Released |
| **Milestone 1: Analytical Equations** | `v3.1.0` | Vectorized analytical models (`step`, `voltammetry`, `hydrodynamics`, `microelectrodes`, `secm`, `kinetics`), automated solver benchmarking (`benchmark`), and parameter fitting (`fitting`) | Released |
| **Milestone 2: Grids** | `v3.2.0` | Non-uniform, geometric, and exponentially expanding spatial discretization meshes (`grid`) and planar/spherical/cylindrical coordinate geometries (`geometry`) | Released |
| **Milestone 3: Reactions** | `v3.3.0` | Homogeneous chemical mechanisms (EC, EC', catalytic) and heterogeneous charge-transfer kinetics (`reactions`) | Released |
| **Milestone 4: Techniques** | `v3.4.0` | Standard electrochemical experimental waveform generators (CA, LSV, CV, SCV, DPV, SWV, CP, RDE), high-level simulation pipelines, declarative plotting, and diagnostics (`techniques`) | Released |
| **Milestone 5: Documentation Overhaul** | `v3.4.1` | Clean-slate [softpotato.xyz](https://softpotato.xyz/) restructure organized by expertise level (Experimentalist, Python User, Modeller) — [Milestone #6](https://github.com/oliverrdz/softpotato/milestone/6) | Released |

For detailed architectural plans and deliverable specifications, see the full [Roadmap](ROADMAP.md).

## Citation

If you use Soft Potato in your research or project, please cite the ChemRxiv preprint:

> Rodríguez, O. (2022). Soft Potato: an open source electrochemical simulator and toolkit. *ChemRxiv*. [doi:10.26434/chemrxiv-2022-4bs3w](https://doi.org/10.26434/chemrxiv-2022-4bs3w)

```bibtex
@article{rodriguez2022softpotato,
  author    = {Rodr{\'i}guez, Oliver},
  title     = {Soft Potato: an open source electrochemical simulator and toolkit},
  journal   = {ChemRxiv},
  year      = {2022},
  doi       = {10.26434/chemrxiv-2022-4bs3w},
  url       = {https://chemrxiv.org/doi/full/10.26434/chemrxiv-2022-4bs3w}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

