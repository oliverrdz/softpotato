# Curve Fitting & Parameter Estimation (`softpotato.analytical.fitting`)

The `softpotato.analytical.fitting` submodule wraps SciPy's least-squares optimizers with physical parameter bounds, Student's $t$ confidence intervals, covariance/correlation matrices, and goodness-of-fit diagnostics ($R^2$, adjusted $R^2$, RMSE, and reduced $\chi_\nu^2$).

## Specialized & General-Purpose Fitters

| Function | Model / Equation | Primary Extracted Parameters |
| :--- | :--- | :--- |
| `fit_cottrell` | Planar Cottrell CA transient | Diffusion coefficient $D$, background $I_{\text{bg}}$ |
| `fit_randles_sevcik` | CV peak current $I_p$ vs. scan rate $v$ | Diffusion coefficient $D$, slope, intercept |
| `fit_secm_approach` | SECM feedback approach curve | Insulator ratio $RG$, zero-distance offset $d_{\text{offset}}$, optional $\kappa$ |
| `fit_levich` | RDE limiting current $I_L$ vs. $\omega$ | Diffusion coefficient $D$, Levich constant $B$ |
| `fit_koutecky_levich` | RDE mixed-control $I$ vs. $\omega$ | Kinetic current $I_K$, $k_{\text{rate}}$, $D$, Levich $B$ |
| `fit_tafel` | High-overpotential $\eta$ vs. $\log_{10}\|I\|$ | Exchange current $I_0$, Tafel slope $b$, $\alpha$, optional $k^0$ |
| `fit_butler_volmer` | Full-range $I$ vs. $\eta$ | $I_0$, $\alpha$, optional $k^0$ |
| `fit_microdisc_transient` | Shoup–Szabo / Mahon–Oldham UME transient | $D$ or microdisc radius $a$ |
| `fit_linear` | Weighted or ordinary $y = m x + b$ | Slope $m$, intercept $b$ |
| `fit_curve` | Any user-defined callable `f(x, *params)` | Custom `param_names` with bounds & uncertainty |

## The `FitResult` Container

Every fitter returns a `FitResult` (or specialized subclass such as `CottrellFitResult`, `RandlesSevcikFitResult`, or `SECMApproachFitResult`) exposing:

- `params`, `stderr`, `confidence_intervals`: dictionaries keyed by parameter name
- `covariance`, `correlation`: parameter covariance and normalized correlation matrices
- `r_squared`, `r_squared_adj`, `rmse`, `chi2_reduced`: goodness-of-fit metrics
- `y_fit`, `residuals`, `success`, `message`, `nfev`: model predictions and optimizer status

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_analytical_fitting.py"
```
