# SoftPotato v3.0 — Transport & Mechanisms Roadmap (1D Planar FD Engine)

This roadmap defines the delivery plan for the **SoftPotato v3.0** core numerical engine
supporting **1D planar diffusion (Fick)** with selectable **grids**, **solvers**, **potential waveforms**,
and user-selectable **mechanisms** (E / EC / CE / EE / arbitrary E–C networks).

Naming alignment (SoftPotato v3.0):
- **Waveforms**: potential programs used by the SoftPotato “wizard” / experiment builders
- **Species**: `Species` objects (D, bulk concentration, charge, etc.)
- **Geometry**: v1 is **Planar 1D**; future geometries can follow the same interface
- **Mechanisms**: composed from **E steps** (electrode) and **C steps** (chemical bulk)
- **Engine**: FD operator + time integrators (CN/BE) + BC handlers

---

## SP3-M0 — Repo scaffolding & CI
**Goal:** installable SoftPotato v3 module with CI gates.

### Tasks
- [x] Create `pyproject.toml` for `softpotato` / `softpotato.core` (PEP 517/518)
- [x] Establish `src/softpotato/` package layout for the engine module
- [x] Add `pytest` harness + smoke tests
- [x] Configure linting (ruff / black)
- [x] GitHub Actions: tests + lint on PR/push
- [ ] Initial docs skeleton (`README.md`, `docs/`)

### Acceptance criteria
- `pip install -e .` succeeds
- `pytest` passes on clean checkout
- CI green on main

---

## SP3-M1 — Waveforms (Potential Programs)
**Goal:** waveform generators return `np.ndarray (n,2)` with columns `[E, t]`.

### Tasks
- [ ] Linear sweep (LSV)
- [ ] Cyclic voltammetry (CV, multi-cycle)
- [ ] Potential step
- [ ] Validation utilities (shape, monotonic time)
- [ ] From-arrays constructor

### Acceptance criteria
**Tests**
- shape `(n,2)`; time strictly increasing; scan rate matches; CV turning points correct

**Plots**
- example scripts produce `E(t)` for CV and LSV without errors

---

## SP3-M2 — Geometry: Planar1D grid & diffusion operator
**Goal:** accurate spatial discretization for Planar 1D.

### Tasks
- [ ] `PlanarGrid1D` (uniform initially) + node conventions
- [ ] Laplacian operator (tri-diagonal bands)
- [ ] Stability diagnostics (explicit dt limit estimator)

### Acceptance criteria
**Tests**
- grid monotonicity; Laplacian of `x^2` ≈ constant; operator shapes correct

**Plots**
- grid spacing sanity plot

---

## SP3-M3 — Engine: time integrators (diffusion only)
**Goal:** stable diffusion solver foundation.

### Tasks
- [ ] Thomas solver (tri-diagonal)
- [ ] Backward Euler integrator
- [ ] Crank–Nicolson integrator (default)
- [ ] Dirichlet boundary support (x=0 and x=L)
- [ ] `SimulationResult` container + `simulate_planar_1d()` skeleton

### Acceptance criteria
**Tests**
- gaussian diffusion trend; CN/BE stable for large dt; no negative concentrations

**Plots**
- diffusion relaxation example `c(x,t)`

---

## SP3-M4 — Mechanism: E (reversible/Nernst)
**Goal:** single electrode step `O + e ⇌ R` at x=0 with Nernst constraint.

### Tasks
- [ ] multi-species state array
- [ ] reversible boundary (Nernst / algebraic constraint)
- [ ] flux → current conversion (sign conventions documented)
- [ ] far-field bulk boundary at x=L

### Acceptance criteria
**Tests**
- surface ratio matches Nernst; flux sign/charge balance consistent; correct limiting behavior

**Plots**
- reversible CV: `E(t)`, `i(t)`, and `i(E)`

---

## SP3-M5 — Mechanism: E (Butler–Volmer)
**Goal:** quasi-reversible electrode kinetics.

### Tasks
- [ ] Butler–Volmer flux law boundary
- [ ] safeguards against negative concentrations / divergence
- [ ] per-step current output

### Acceptance criteria
**Tests**
- BV → Nernst as `k0 → ∞`; smaller `k0` reduces current; no NaNs

**Plots**
- peak separation vs `k0` example

---

## SP3-M6 — Mechanism: C steps (bulk kinetics) + operator splitting
**Goal:** enable EC and CE (diffusion–reaction).

### Tasks
- [ ] mass-action C-step kinetics (start: first-order; then reversible)
- [ ] reaction term assembly `R(C)` for all species
- [ ] operator splitting: diffusion step + reaction step per node
- [ ] mass-balance diagnostics for closed systems

### Acceptance criteria
**Tests**
- first-order decay matches analytic; closed-system mass conserved; dt convergence

**Plots**
- EC chrono + CV dependence on k

---

## SP3-M7 — Mechanism framework + compiler (E / EC / CE)
**Goal:** user selects mechanisms via a SoftPotato v3 mechanism builder.

### Tasks
- [ ] `EStep` and `CStep` objects (explicit species mapping)
- [ ] `Mechanism` builder API
- [ ] mechanism compiler wiring species/reactions/boundaries/observables
- [ ] optional shorthand parser (`"E"`, `"EC"`, `"CE"`) only when mapping provided

### Acceptance criteria
**Tests**
- species registry stable; invalid mechanisms raise clear errors; compiled EC ≈ manual EC

**Plots**
- one example per mechanism using only `Mechanism()` + waveform + engine

---

## SP3-M8 — Multi-E (EE) + surface solver
**Goal:** multiple electrode steps with robust surface nonlinear solve.

### Tasks
- [ ] multi-E boundary handling
- [ ] surface Newton solver with damping/fallback
- [ ] per-step currents + total current

### Acceptance criteria
**Tests**
- additivity for independent couples; sequential redox yields two waves; solver converges robustly

**Plots**
- two-wave CV + per-step current breakdown

---

## SP3-M9 — General E–C networks + robustness
**Goal:** arbitrary E/C networks (ECE, EC′, branching).

### Tasks
- [ ] stoichiometric matrix representation
- [ ] branched & reversible C steps
- [ ] network diagnostics (conservation, negativity, convergence)
- [ ] optional adaptive dt or fully coupled implicit mode for stiff systems

### Acceptance criteria
**Tests**
- conserved quantities remain constant; stiff systems stable; parser/compiler consistency

**Plots**
- catalytic EC′ signature + branching selectivity demo

---

## Definition of Done (per PR)
- Tests added/updated
- Examples updated if public API changes
- Docs updated
- CI passes

