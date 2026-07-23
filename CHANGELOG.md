# Changelog

All notable changes to the **Soft Potato** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project scaffold using `src/` package layout and PEP 517/621 `pyproject.toml` standard targeting Python `>=3.11`.
- Core Abstract Base Classes (`BaseMesh`, `BaseModel`, `BaseBoundaryCondition`, `BaseDiscretizer`, `BaseSolver`) enforcing strict physics-numerics decoupling in `src/softpotato/core/abcs.py`.
- Top-level package imports and metadata exposed in `src/softpotato/__init__.py`.
- Architectural test suite enforcing abstract contract behavior in `tests/test_architecture.py`.
- AI memory and design guidelines in `GEMINI.txt`.
