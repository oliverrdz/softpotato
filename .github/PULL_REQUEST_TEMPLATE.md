## Summary and Motivation

<!-- Briefly describe what this pull request changes and why. Link any related issue below. -->

Closes #

---

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New electrochemical mechanism, technique, or solver (non-breaking feature)
- [ ] Analytical equation addition or update (`softpotato.analytical`)
- [ ] Documentation, tutorial, or example update (`softpotato.xyz`)
- [ ] Refactor or performance improvement (no public API changes)
- [ ] Breaking change (requires prior discussion in an issue)

---

## Generative AI Disclosure

<!-- Per CONTRIBUTING.md, disclose if generative AI materially contributed to this PR. -->

- **AI Tools Used:** <!-- e.g., None, Antigravity, GitHub Copilot, Gemini -->
- **Scope of AI Assistance:** <!-- e.g., Boilerplate unit tests, docstring drafting, initial PDE discretization draft -->
- **Human Verification:** I confirm that I have personally read, understood, and verified every line of code, mathematical expression, and test in this pull request.

---

## Scientific and Numerical Verification

<!-- Explain how you verified the physics, units, sign conventions, and numerical accuracy. -->

- **Analytical / Literature Benchmark:** <!-- e.g., Verified CV peak current against Randles-Ševčík equation within 0.5% error -->
- **Edge Cases Tested:** <!-- e.g., Extreme scan rates, fast heterogeneous kinetics, zero bulk concentration -->

---

## Pre-Submission Checklist

- [ ] My changes follow Soft Potato's decoupled architecture (keeping physical chemistry and kinetics separate from grids and solvers).
- [ ] I have added or updated `pytest` unit tests covering my changes, and `pytest` passes locally.
- [ ] I have run `ruff check .` and `ruff format --check .` locally with zero warnings or formatting diffs.
- [ ] Any new or updated figures use color-blind friendly palettes, distinct line styles/markers, and `variable / unit` axis labels.
- [ ] I have updated `CHANGELOG.md` under `[Unreleased]` following Keep a Changelog conventions (for user-facing changes).
- [ ] My commit messages are clear and include an `Assisted-by:` trailer if generative AI materially contributed.