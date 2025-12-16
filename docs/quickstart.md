# Quickstart: Time Grid → Waveform (M1)

This quickstart demonstrates the **only supported scientific workflow in M1**:

1. Create a time grid
2. Generate a potential waveform
3. Inspect the resulting array

Waveforms are **data generators only**, not simulations.

---

## Step 1 — Create a Time Grid

A **time grid** defines when an experiment is sampled.

### Create a uniform time grid
```python
import softpotato as sp
tg = sp.uniform_time_grid(dt=0.1, n=11)
print(tg)
```
Properties:
- Uniform spacing
- Starts at t = 0
- Strictly increasing
- Validated on creation

---

## Step 2 — Generate a Waveform

Waveforms are NumPy arrays with shape `(n, 2)`.

### Generate a linear sweep waveform
```python
import softpotato as sp
w = sp.lsv(0.0, 1.0, scan_rate=1.0, dt=0.1)
print(w[:3])
```

Interpretation:
- Column 0 → potential E (V)
- Column 1 → time t (s)
- Scan rate is in V/s

---

## Step 3 — Inspect the Result

### Inspect shape and contents
```python
import softpotato as sp
w = sp.lsv(0.0, 1.0, 1.0, 0.1)
print(w.shape)
```

Canonical format:
- `(n, 2)`
- `[E, t]`

---

## Important Note

Waveforms do **not**:
- Solve diffusion equations
- Compute currents
- Apply kinetics
- Represent a physical experiment

---

## How This Is Tested

Every step in this quickstart is enforced by automated tests.

See:
- docs/traceability.md
- tests/test_m1_timegrid.py
- tests/test_m1_waveforms.py

