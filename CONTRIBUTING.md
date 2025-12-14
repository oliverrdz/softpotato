# Contributing to SoftPotato

Thank you for your interest in contributing to **SoftPotato** — an open-source
electrochemistry simulation and numerical modeling toolkit.

Contributions of all kinds are welcome: bug fixes, numerical improvements,
documentation, examples, and tests.

---

## Code of Conduct

Please be respectful and constructive.  
This project follows the spirit of the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Contributor License Agreement (CLA)

By submitting a contribution (e.g. a GitHub pull request or commit), you agree to
the terms of the **Contributor License Agreement**:

📄 [`CLA.md`](CLA.md)

In short:
- You keep copyright to your contributions
- You grant the maintainer the right to use and relicense them as part of SoftPotato

If you do not agree with the CLA, please do not submit contributions.

---

## How to Contribute

### 1. Fork and branch

1. Fork the repository on GitHub
2. Create a feature branch from `main`:

```bash
git checkout -b feature/my-change
```
---

### 2. Set up a development environment
SoftPotato uses a standard PEP 517/518 setup.
```bash
pip install -e .[dev]
```
This installs:
* pytest
* ruff
* black
* mypy
* pre-commit (optional but recommended)

---

### 3. Coding standards
Python style
* Python ≥ 3.10
* Follow PEP 8
* Format code with Black
* Lint with Ruff
* Type-check with mypy (where applicable)

Before submitting:
```bash
ruff check .
black .
pytest
```
CI will enforce these checks.

---

### 4. Numerical & scientific guidelines

Because SoftPotato is a numerical simulation library, please ensure:

* Units and physical meaning are documented
* Sign conventions (flux, current, charge) are explicit
* New solvers or operators include:
    * stability considerations
    * references where appropriate
* Avoid silent failures (NaNs, negative concentrations, divergence)

If in doubt, add a test.

---

### 5. Tests

All new features and bug fixes must include tests.

* Tests use pytest
* Place tests under tests/
* Prefer small, deterministic tests over long simulations
* Numerical comparisons should use tolerances (np.allclose, etc.)

Pull requests without tests may be rejected unless the change is trivial.

---

### 6. Documentation

If your change affects:

* public APIs
* numerical behavior
* assumptions or limitations

please update:
* docstrings
* README or docs (if applicable)

Clear documentation is as important as correct code.

---

## Commit messages
Please use clear, descriptive commit messages:
```
Add Crank–Nicolson integrator for planar diffusion
Fix sign error in Butler–Volmer flux
Docs: clarify EC mechanism assumptions
```

---

## Pull Request Checklist

Before submitting a PR, please ensure:

* [ ] Code builds and tests pass locally
* [ ] New functionality includes tests
* [ ] Linting and formatting are clean
* [ ] Documentation is updated if needed
* [ ] You agree to the CLA (CLA.md)

---

## Reporting Issues

If you find a bug or want to propose a feature:

* Open a GitHub Issue
* Include:
    * expected vs actual behavior
    * minimal reproducible example
    * plots or equations if relevant

---

## Questions & Discussion

For design questions or larger changes, please open an issue before
starting major work. This helps align contributions with the project roadmap.

---

Thank you for helping improve Soft Potato


