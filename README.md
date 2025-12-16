# SoftPotato v3.0

SoftPotato is a scientific Python library for building electrochemical experiments as **validated time grids and potential waveforms**.

**M1 introduces the first stable, user-facing scientific API**: time grids and potential waveform generators.

This milestone provides **data generators only**.
No physics, no solvers, no plotting.

---

## Installation

### Install (user)
```bash
pip install softpotato
```
### Install (development)
```bash
pip install -e ".[dev]"
```
---

## What exists in M1

You can now:

- Create validated **uniform time grids**
- Generate **potential waveforms**
  - Linear sweep voltammetry (LSV)
  - Cyclic voltammetry (CV)
  - Step waveforms
- Validate time arrays and waveform arrays
- Work with a **canonical waveform format**: `(n, 2)` → `[E, t]`

---

## What does NOT exist yet

The following are explicitly **out of scope** for M1:

- Diffusion models
- Electrochemical mechanisms (E, EC, CE, …)
- Current calculation
- Experiment orchestration objects
- Adaptive or nonuniform time grids
- Plotting helpers
- CLI or GUI

Docs do not reference these concepts.

---

## Minimal example

### Create a time grid and waveform
```bash
python -c "import softpotato as sp; tg=sp.uniform_time_grid(0.1,n=5); w=sp.lsv(0.0,1.0,scan_rate=1.0,dt=0.1); print(w)"
```
Expected behavior:

- Output is a NumPy array of shape `(n, 2)`
- Column 0: potential `E` (volts)
- Column 1: time `t` (seconds)
- Time is strictly increasing and finite

---

## Public API (M1)

Stable, user-facing symbols:

- softpotato.TimeGrid
- softpotato.uniform_time_grid(...)
- softpotato.lsv(...)
- softpotato.cv(...)
- softpotato.step(...)
- softpotato.waveform_from_arrays(E, t)
- softpotato.validate_time(t, ...)
- softpotato.validate_waveform(w)

