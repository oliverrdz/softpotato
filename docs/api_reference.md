# API Reference (M0 + M1)

* This is a **reference stub**.
* Behavior is described in tutorials and tests.
* This reference only documents behavior that is enforced by tests.
* See docs/traceability.md for exact test mappings.

---

## M0 — Public Surface

### Package import
```python
import softpotato
```

No scientific API is available in M0.

---

## M1 — Time Grids

```python
uniform_time_grid(dt, n)
```
Create a validated, uniform time grid.

Parameters:
- dt — time step (s)
- n — number of points

Returns:
- TimeGrid

---

## M1 — Waveform Generators

```python
lsv(E_start, E_end, scan_rate, dt)
```
Generate a linear sweep waveform.

---

```python
cv(E_start, E_vertex, E_end, scan_rate, dt)
```
Generate a cyclic voltammetry waveform.

---

```python
step(E_before, E_after, t_step, dt, n)
```
Generate a step waveform.

---

```python
waveform_from_arrays(E, t)
```
Create a waveform from explicit arrays.

---

## Validation Helpers

```python
validate_time(t)
```
Validate a time array.

---

```python
validate_waveform(w)
```
Validate a waveform array.

