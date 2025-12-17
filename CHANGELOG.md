## v3.0.0-alpha2 — 2025-12-17

### Changed (breaking)
- Simplified M1 waveform generation to be **resolution-driven** rather than requiring a user-provided time grid.
- Updated public waveform APIs:
  - `lsv(E_start, E_end, dE, scan_rate)`
  - `cv(E_start, E_vertex, scan_rate, dE, E_end=None, cycles=1)`
  - `step(E_before, E_after, dt, t_end)`

### Behavior
- Scan waveforms (`lsv`, `cv`) now deterministically derive the time array from `dE` and `scan_rate`, with exact endpoint enforcement.
- Step waveforms now construct a uniform time grid using `dt` up to `t_end`.
- All waveform generators return `np.ndarray (n, 2)` with columns `[E, t]` and strictly increasing time.

### Documentation
- Updated waveform documentation and examples to reflect the new resolution-based API.
- Clarified parameter meanings, units, and deterministic sampling rules.

### Tests
- Updated M1 waveform tests to validate the new public signatures and derived time behavior.

### Notes
- This release introduces **breaking changes** relative to `v3.0.0-alpha1`.
- Users migrating from `alpha1` should replace time-grid-based inputs with `dE` (scan waveforms) or `dt` (step waveforms).


## [3.0.0-alpha1] — 2025-12-16

### Added
- Time grid utilities providing a validated, reusable time base for simulated electrochemical experiments.
- Potential waveform generators (M1) built on top of time grids (e.g. linear sweep–style waveforms).
- Input validation layer to enforce argument consistency and array shapes early.

### Documentation
- Expanded MkDocs documentation including quickstart, API references for time grids and waveforms, and runnable examples.
- Cross-linked tests and documentation for traceability of acceptance criteria.

### Quality
- Test coverage for M0 project skeleton and M1 time grids and waveforms.
- CI and documentation builds expected to pass on the release tag.

### Notes
- This is an **alpha** release. APIs may change as diffusion, kinetics, and reaction mechanism milestones are implemented.

---

## [3.0.0-alpha0] — 2025-12-15

**Status:** Alpha (M0 – project skeleton)

This is the first pre-release of SoftPotato v3.0.  
It validates repository structure, packaging, versioning, tests, and documentation workflows.
There is **no stable scientific or numerical API** in this release.

### Added
- Initial SoftPotato v3.0 repository skeleton using a `src/` layout.
- Packaging and versioning scaffold with runtime `softpotato.__version__`.
- Pytest-based M0 smoke tests (import and version exposure).
- Documentation baseline:
  - `README.md`
  - `ARCHITECTURE.md`
  - `ROADMAP.md`
  - MkDocs configuration and `docs/` skeleton.
- Project governance files:
  - `LICENSE`
  - `CLA.md`
  - `CONTRIBUTING.md`

### Changed
- Tooling and CI foundations for editable installs and test execution.

### Notes
- This release is intended **only** to establish a clean, testable, and documented starting point.
- All electrochemical models, solvers, mechanisms, and user-facing APIs are deferred to later alpha milestones.

