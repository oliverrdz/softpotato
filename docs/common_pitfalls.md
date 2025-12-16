# Common Pitfalls (M0 + M1)

---

## Expecting simulations

M1 only generates **time and potential data**.

There is:
- No diffusion
- No kinetics
- No current calculation

---

## CV vertex alignment

In cyclic voltammetry:
- The vertex index depends on `dt`
- Small changes in `dt` can shift the turning point
- This is expected and validated behavior

---

## Step boundary behavior

Step waveforms follow strict rules:
- t < t_step → E_before
- t ≥ t_step → E_after

This is intentional.

---

## Mixing formats

Only one waveform format is supported:
- Shape `(n, 2)`
- Columns `[E, t]`

Other formats are invalid.

---

## Assuming API stability

All M1 APIs are **alpha** and may change.
Do not rely on them for production or research yet.

---

## Why These Pitfalls Exist

Each pitfall corresponds to explicit test coverage.

See docs/traceability.md for details.

