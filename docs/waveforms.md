# Waveforms (Potential Programs) — M1

SoftPotato waveforms are **potential programs**, not simulations.

A waveform is a NumPy array with:

- Shape: `(n, 2)`
- Column order: `[E, t]`
- Units: potential `E` in volts (V), time `t` in seconds (s)

Waveforms do **not**:
- Solve diffusion equations
- Compute currents
- Apply kinetics

---

## Canonical waveform format

Given a waveform `w`:

- `w[:, 0]` is potential `E` in volts (V)
- `w[:, 1]` is time `t` in seconds (s)

Time must be strictly increasing.

---

## lsv(E_start, E_end, scan_rate, dt)

Generate a **linear sweep voltammetry (LSV)** potential program: a linear ramp from `E_start` to `E_end`.

### Parameters

- `E_start` (float, V)  
  Start potential in volts.

- `E_end` (float, V)  
  End potential in volts.

- `scan_rate` (float, V/s)  
  Scan rate in volts per second. Must be positive.

- `dt` (float, s)  
  Time step in seconds. Must be positive.

### Returns

- `w` (numpy.ndarray, shape `(n, 2)`)  
  Waveform array with columns `[E, t]`.

### Example

```python
import softpotato as sp
w = sp.lsv(E_start=0.0, E_end=1.0, scan_rate=0.5, dt=0.1)
print(w.shape)
print(w[:5])
```

---

## cv(E_start, E_vertex, E_end, scan_rate, dt)

Generate a **cyclic voltammetry (CV)** potential program: a linear ramp from `E_start` to `E_vertex`, then a reversal to `E_end`.

### Parameters

- `E_start` (float, V)  
  Start potential in volts.

- `E_vertex` (float, V)  
  Vertex (turning point) potential in volts.

- `E_end` (float, V)  
  End potential in volts.

- `scan_rate` (float, V/s)  
  Scan rate in volts per second. Must be positive.

- `dt` (float, s)  
  Time step in seconds. Must be positive.

### Returns

- `w` (numpy.ndarray, shape `(n, 2)`)  
  Waveform array with columns `[E, t]`.

### Notes

- The **exact vertex index** depends on how `dt` aligns with the vertex time.
- Small changes in `dt` can shift the turning index.

### Example

```python
import softpotato as sp
w = sp.cv(E_start=0.0, E_vertex=1.0, E_end=0.0, scan_rate=1.0, dt=0.1)
print(w.shape)
print(w[:5]); print(w[-5:])
```

---

## step(E_before, E_after, t_step, dt, n)

Generate a **step potential** program: potential is `E_before` up to (but not including) `t_step`, then `E_after` from `t_step` onward.

### Parameters

- `E_before` (float, V)  
  Potential before the step, in volts.

- `E_after` (float, V)  
  Potential after the step, in volts.

- `t_step` (float, s)  
  Step time in seconds. Defines the boundary:
  - for `t < t_step` → `E_before`
  - for `t >= t_step` → `E_after`

- `dt` (float, s)  
  Time step in seconds. Must be positive.

- `n` (int, dimensionless)  
  Number of points in the waveform (length of the time grid). Must be an integer ≥ 1.

### Returns

- `w` (numpy.ndarray, shape `(n, 2)`)  
  Waveform array with columns `[E, t]`.

### Example

```python
import softpotato as sp
w = sp.step(E_before=0.0, E_after=1.0, t_step=0.5, dt=0.1, n=11)
print(w.shape)
print(w)
```

---

## waveform_from_arrays(E, t)

Create a waveform directly from explicit arrays.

### Parameters

- `E` (array-like of float, V)  
  Potential samples in volts. Must be finite and one-dimensional.

- `t` (array-like of float, s)  
  Time samples in seconds. Must be finite, one-dimensional, and strictly increasing.

### Returns

- `w` (numpy.ndarray, shape `(n, 2)`)  
  Waveform array with columns `[E, t]`.

### Example

```python
import numpy as np
import softpotato as sp
t = np.array([0.0, 0.5, 1.0])
E = np.array([0.0, 0.2, 0.4])
w = sp.waveform_from_arrays(E=E, t=t)
print(w.shape)
print(w)
```

