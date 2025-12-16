# Documentation ↔ Tests Traceability

This document maps **user-facing documentation claims** to the **tests that enforce them**.

Its purpose is to ensure:
- Docs do not promise untested behavior
- Tests clearly justify documented guarantees
- Future contributors know where to update docs when tests change

---

## Milestone Coverage

- **M0** — Project skeleton, packaging, versioning
- **M1** — Time grids and potential waveforms

No later milestones are covered here.

---

## M0: Project Skeleton

### Package imports successfully

**Docs**
- README.md — Quickstart
- docs/index.md — “What Exists Today”

**Tests**
- tests/test_m0_skeleton.py
  - test_imports_and_exposes_version_string

---

### Version string is accessible

**Docs**
- README.md — Minimal sanity check
- docs/quickstart.md

**Guarantee**
- `softpotato.__version__` exists

**Tests**
- tests/test_m0_skeleton.py
  - test_imports_and_exposes_version_string
  - test_runtime_version_matches_distribution_metadata

---

### No scientific API exists

**Docs**
- README.md — “Current Status (M0)”
- docs/api_reference_m1.md — M0 section
- docs/common_pitfalls.md — “Expecting simulations”

**Tests**
- Implicitly enforced by absence of symbols
- No solver, waveform, or mechanism tests exist in M0

---

## M1: Time Grids

### Uniform time grid creation

**Docs**
- docs/quickstart.md — Step 1
- docs/tutorial_timegrid_to_waveform.md — Time grid concept
- docs/api_reference_m1.md — uniform_time_grid

**Tests**
- tests/test_m1_timegrid.py
  - test_uniform_time_grid_basic
  - test_uniform_time_grid_properties

---

### Time grid validation rules

**Documented guarantees**
- Strictly increasing
- Finite values
- One-dimensional

**Docs**
- docs/tutorial_timegrid_to_waveform.md
- docs/common_pitfalls.md

**Tests**
- tests/test_m1_timegrid.py
  - test_time_validation_strictly_increasing
  - test_time_validation_finite
  - test_time_validation_shape

---

## M1: Waveforms

### Canonical waveform format

**Documented guarantee**
- Shape `(n, 2)`
- Columns `[E, t]`

**Docs**
- docs/waveforms.md
- docs/quickstart.md
- docs/tutorial_timegrid_to_waveform.md

**Tests**
- tests/test_m1_waveforms.py
  - test_waveform_shape
  - test_waveform_time_column

---

### Linear sweep waveform (LSV)

**Docs**
- docs/quickstart.md
- docs/tutorial_timegrid_to_waveform.md
- docs/api_reference_m1.md

**Tests**
- tests/test_m1_waveforms.py
  - test_lsv_linear_ramp
  - test_lsv_time_alignment

---

### Cyclic voltammetry waveform (CV)

**Docs**
- docs/tutorial_timegrid_to_waveform.md
- docs/common_pitfalls.md — CV vertex alignment

**Tests**
- tests/test_m1_waveforms.py
  - test_cv_vertex_turnaround
  - test_cv_monotonic_segments

---

### Step waveform behavior

**Documented guarantee**
- t < t_step → E_before
- t ≥ t_step → E_after

**Docs**
- docs/tutorial_timegrid_to_waveform.md
- docs/common_pitfalls.md

**Tests**
- tests/test_m1_waveforms.py
  - test_step_boundary_behavior

---

### Waveform validation

**Documented guarantees**
- Shape `(n, 2)`
- Finite values
- Strictly increasing time

**Docs**
- docs/tutorial_timegrid_to_waveform.md
- docs/api_reference_m1.md

**Tests**
- tests/test_m1_waveforms.py
  - test_validate_waveform_shape
  - test_validate_waveform_time_monotonic

---

## Non-Guarantees (Explicitly Untested)

The following are **documented as out of scope** and have **no tests**:

- Diffusion
- Kinetics
- Currents
- Physical correctness
- Units enforcement beyond array semantics

This is intentional.

---

## Maintenance Rule

When changing:
- Tests → update referenced docs
- Docs → ensure a test exists or downgrade wording

Docs and tests must evolve together.

