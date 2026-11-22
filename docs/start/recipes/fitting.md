# Recipe: Fitting Experimental Data

Extract physical and geometric parameters—with **$R^2$** and **95% confidence intervals**—from chronoamperometry, cyclic voltammetry, and SECM measurements using `softpotato.analytical.fitting`:

- `fit_cottrell()` → extracts diffusion coefficient $D$
- `fit_randles_sevcik()` → extracts diffusion coefficient $D$ from $I_p$ vs. $v$
- `fit_secm_approach()` → extracts tip insulator ratio $RG$ and zero-distance offset $d_{\text{offset}}$

## Full Runnable Script

```python
--8<-- "docs/scripts/recipe_fitting.py"
```

## Using Your Own Lab Data

Replace the synthetic `t_exp`, `i_exp_ca`, `v_exp`, `ip_exp`, `d_exp`, and `it_exp` arrays with columns loaded from your potentiostat CSV export (for example with `numpy.loadtxt("data.csv", delimiter=",", skiprows=1, unpack=True)`).
