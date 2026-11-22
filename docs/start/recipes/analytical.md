# Recipe: Analytical Baselines

Compute exact theoretical baselines for **Cottrell** (chronoamperometry), **Levich** (RDE), **Randles–Ševčík** (CV peak currents), and **SECM approach curves**—with zero PDEs or meshes.

Edit only the physical parameters (`n`, `D`, `c_bulk`, `area`, `nu`, `T`, `RG`, `kappa`) at the top of the script.

## Full Runnable Script

```python
--8<-- "docs/scripts/recipe_analytical.py"
```

## Run Locally

Save the code above as `recipe_analytical.py` and run:

```bash
python recipe_analytical.py
```

!!! chip "Units Reminder"
    In `softpotato.analytical`, keep length units consistent across `D`, `area`, `nu`, and `c_bulk`: if `D` is in $\text{cm}^2/\text{s}$ and `area` is in $\text{cm}^2$, pass `c_bulk` in $\text{mol}/\text{cm}^3$ ($1\text{ mM} = 10^{-6}\text{ mol}/\text{cm}^3$) so the returned current is in Amperes ($\text{A}$).
