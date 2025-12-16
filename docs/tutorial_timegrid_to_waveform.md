# Tutorial: Time Grid → Potential Waveform (M1)

This tutorial explains **concepts only**.
No numerical electrochemistry is performed.

---

## Concept: Time Grid

A **time grid** defines when quantities are sampled.

Requirements enforced by validation:
- One-dimensional
- Finite values
- Strictly increasing

### Validate a time array
```python
import softpotato as sp
sp.validate_time([0.0, 0.1, 0.2])
```

---

## Concept: Potential Waveform

A **potential waveform** defines what potential is applied at each time.

Representation:
- NumPy array
- Shape `(n, 2)`
- Columns `[E, t]`

This is the **canonical waveform format** used throughout SoftPotato.

---

## Linear Sweep Example

### Generate an LSV waveform
```python
import softpotato as sp
w = sp.lsv(0.0, 1.0, scan_rate=1.0, dt=0.1); print(w[:3])
```
Meaning:
- Start at 0.0 V
- End at 1.0 V
- Ramp at 1 V/s
- Sample every 0.1 s

---

## Cyclic Voltammetry Example

### Generate a CV waveform
```python 
import softpotato as sp
w = sp.cv(0.0, 1.0, 0.0, scan_rate=1.0, dt=0.1)
```

Notes:
- Direction reverses at the vertex
- Vertex index depends on `dt`

---

## Step Waveform Example

### Generate a step waveform
```python
import softpotato as sp
w = sp.step(0.0, 1.0, t_step=0.5, dt=0.1, n=11)
```

Behavior:
- t < t_step → E_before
- t ≥ t_step → E_after

---

## Validation

### Validate a waveform
```python
import softpotato as sp
sp.validate_waveform(w)
```

Validation checks:
- Shape `(n, 2)`
- Finite values
- Strictly increasing time column

---

## Test Coverage

All behaviors described here are validated in:

- tests/test_m1_timegrid.py
- tests/test_m1_waveforms.py

If the tutorial and tests disagree, the tests are authoritative.

