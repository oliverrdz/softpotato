# Recipe: First Simulation (Cyclic Voltammetry)

Run a complete Butler–Volmer cyclic voltammetry simulation and plot the resulting voltammogram in a single `simulate()` + `plot_result()` call. Soft Potato automatically constructs a physics-informed spatial mesh and solves the reaction–diffusion PDE for you.

Edit only the physical parameters (`E0`, `D`, `c_bulk`, `k0`, `alpha`) and waveform parameters (`E_start`, `E_vertex1`, `E_vertex2`, `v`) at the top of the script.

## Full Runnable Script

```python
--8<-- "docs/scripts/recipe_first_cv.py"
```

## What `simulate()` Returns

`result` is a `TechniqueResult` object containing:

- `result.time` — time array ($\text{s}$)
- `result.potential` — applied potential ($\text{V}$)
- `result.current` — simulated current density ($\text{A}/\text{m}^2$ for planar 1D)
- `result["Ox"]` and `result["Red"]` — spatial concentration profiles

Pass `result` directly to `plot_result(result)` for an instant publication-ready figure.
