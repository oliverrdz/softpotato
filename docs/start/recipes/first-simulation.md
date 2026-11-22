# Recipe: First Simulation (Cyclic Voltammetry)

Run a complete Butler–Volmer cyclic voltammetry simulation and plot the resulting voltammogram in a single `simulate()` + `plot_result()` call. Soft Potato automatically constructs a physics-informed spatial mesh and solves the reaction–diffusion PDE for you.

Edit only the physical parameters (`E0`, `D`, `c_bulk`, `k0`, `alpha`, `area`) and waveform parameters (`E_start`, `E_vertex1`, `E_vertex2`, `v`) at the top of the script. Passing `PlanarGeometry(area)` to `simulate()` scales the Faradaic current to the electrode area in Amperes ($\text{A}$).

## Full Runnable Script

```python
--8 < --"docs/scripts/recipe_first_cv.py"
```

!!! note "Species Ordering (`[ox, red]`)"
    When passing a list of species directly to `simulate()`, the list **must** be ordered `[ox, red]` according to the reduction reaction $\text{Ox} + e^- \rightarrow \text{Red}$ (oxidized species first, reduced species second).

## What `simulate()` Returns

`result` is a `TechniqueResult` object containing:

- `result.time` — time array ($\text{s}$)
- `result.potential` — applied potential ($\text{V}$)
- `result.current` — simulated Faradaic current ($\text{A}$)
- `result["Ox"]` and `result["Red"]` — spatial concentration profiles

Pass `result` directly to `plot_result(result)` for an instant publication-ready figure.
