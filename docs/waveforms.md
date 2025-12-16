# Waveforms (M1)

SoftPotato waveforms are **potential programs**, not simulations.

They use a canonical NumPy array format:

- Shape `(n, 2)`
- Columns `[E, t]`
- Units: volts (V), seconds (s)

---

## Available Generators (M1)

- uniform_time_grid(...)
- lsv(...)
- cv(...)
- step(...)
- waveform_from_arrays(E, t)

---

## Canonical Format

All waveforms must follow the same representation.

### Example structure
```python
E = w[:, 0]  
t = w[:, 1]
```

Any other format is invalid.

---

## Scope Limitation

Waveforms do **not**:
- Model diffusion
- Compute currents
- Enforce electrochemical kinetics

