## Summary

<!-- Briefly describe the purpose of this PR. What new modules, classes, features, or fixes are introduced? -->

## Changes

- 
- 

## Motivation & Context

<!-- Why is this change required? What problem does it solve? If related to an issue, link it here (e.g., Closes #12). -->

---

## Pre-Merge Checklist

### Code & Units
- [ ] Strictly adheres to **CGS units** ($D$ in $\text{cm}^2/\text{s}$, $c_{\text{bulk}}$ in $\text{mol}/\text{cm}^3$, area in $\text{cm}^2$, radius in $\text{cm}$, potential in $\text{V}$, time in $\text{s}$)
- [ ] Vectorized implementations for closed-form / benchmark equations (NumPy arrays supported)
- [ ] Public classes and functions exposed in corresponding `__init__.py` and `__all__`

### Quality & Tests
- [ ] Unit tests added or updated in `tests/`
- [ ] Local tests pass (`pytest`)
- [ ] Static type checking passes (`mypy src/`)
- [ ] Formatting and linting pass (`black --check src/ tests/` and `ruff check src/ tests/`)

### Documentation
- [ ] Docstrings follow Google/NumPy format with clear parameter types and CGS units
- [ ] Sphinx documentation builds locally without warnings (`cd docs && make html`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

