# Mass Transport

Once an electron transfers at the electrode surface, reactant is depleted and product accumulates. How quickly fresh reactant arrives—and product leaves—is governed by **mass transport**. In Soft Potato, mass transport is modelled in 1D across multiple electrode geometries (`PlanarGeometry`, `SphericalGeometry`, `CylindricalGeometry`) and hydrodynamic convection (`RotatingDiskElectrode`).

## Fick's Laws of Diffusion

In an unstirred (quiescent) solution with excess supporting electrolyte (so migration is negligible), mass transport is purely diffusive.

- **Fick's First Law** relates the molar flux $J$ ($\text{mol}\,\text{m}^{-2}\,\text{s}^{-1}$) to the concentration gradient:

  $$J = -D \frac{\partial c}{\partial x}$$

  At the electrode surface ($x = 0$), the Faradaic current is directly proportional to this interfacial gradient:

  $$I(t) = n F A \, D \left.\frac{\partial c_{\text{Ox}}}{\partial x}\right|_{x=0}$$

- **Fick's Second Law** describes how concentration evolves over time in a general 1D radial/planar coordinate $r$:

  $$\frac{\partial c}{\partial t} = D \nabla^2 c = D \left( \frac{\partial^2 c}{\partial r^2} + \frac{p}{r} \frac{\partial c}{\partial r} \right)$$

  where $p$ is the geometric dimensionality index.

---

## Electrode Geometries

Soft Potato unifies planar, cylindrical, and spherical diffusion through the geometry parameter $p$:

| Geometry | Soft Potato Class | Index $p$ | Coordinate $r$ | Physical Regime & Current Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Planar** | `PlanarGeometry(area=...)` | $0$ | $x \ge 0$ | Linear semi-infinite diffusion. Diffusion layer grows as $\delta \sim \sqrt{D t}$; step current decays as $t^{-1/2}$ (Cottrell) and never reaches a non-zero steady state. |
| **Cylindrical** | `CylindricalGeometry(r0=..., length=...)` | $1$ | $r \ge r_0$ | Radial diffusion to a wire/fiber microelectrode. Quasi-steady state at long times ($t \gg r_0^2 / D$) with logarithmic decay. |
| **Spherical / Hemispherical** | `SphericalGeometry(r0=..., hemisphere=...)` | $2$ | $r \ge r_0$ | Convergent radial diffusion. Reaches a true non-zero **steady-state** current at long times because spherical shells expand faster than the depletion zone. |

### Planar vs. Spherical & Microelectrode Diffusion

At a **macroelectrode** ($r_0 \gg \sqrt{D t}$), edge effects are negligible and diffusion is effectively **planar** ($p = 0$). Under a potential step to the mass-transport limit, the current follows the **Cottrell equation**:

$$I(t) = \frac{n F A D^{1/2} c^*}{\sqrt{\pi t}}$$

At a **microelectrode** (ultramicroelectrode, UME, typically $r_0 \le 25\,\mu\text{m}$), the diffusion layer quickly exceeds the electrode radius ($\sqrt{D t} \gg r_0$). Radial (convergent) diffusion dominates, producing a steady-state limiting current:

- **Hemisphere** ($A = 2\pi r_0^2$): $I_{\text{ss}} = 2 \pi n F D c^* r_0$
- **Full sphere** ($A = 4\pi r_0^2$): $I_{\text{ss}} = 4 \pi n F D c^* r_0$
- **Microdisc** (inlaid disk of radius $a$, analytical model in `softpotato.analytical.microelectrodes`): $I_{\text{ss}} = 4 n F D c^* a$

In cyclic voltammetry, this transition from planar to convergent diffusion transforms a **peak-shaped** voltammogram (at fast scan rates or large electrodes) into a **sigmoidal** steady-state wave (at slow scan rates or microelectrodes).

---

## Rotating Disk Electrode (RDE) Convection–Diffusion

When a disk electrode rotates at angular velocity $\omega$ ($\text{rad/s}$), centrifugal force flings solution outward radially and draws fresh bulk solution axially toward the disk. Near the surface, the **von Kármán** axial velocity is:

$$v_x(x) = -0.51023\,\omega^{3/2} \nu^{-1/2} x^2$$

where $\nu$ is the electrolyte kinematic viscosity ($\approx 10^{-6}\text{ m}^2/\text{s}$ for water at $298\text{ K}$).

Soft Potato's `RotatingDiskElectrode` supports two hydrodynamic descriptions:

1. **Nernst Diffusion Layer (`model="nernst_layer"`, default)**:
   Approximates forced convection by truncating pure diffusion at a fixed **Nernst diffusion layer thickness** $\delta_N$, where bulk concentration is held constant ($c(\delta_N, t) = c^*$):

   $$\delta_N = 1.610 \, D^{1/3} \nu^{1/6} \omega^{-1/2}$$

2. **Full Convective–Diffusion (`model="convective_diffusion"`)**:
   Solves the full 1D axial advection–diffusion PDE on an extended domain $x \in [0, k \delta_N]$ (default $k = 3.0$):

   $$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - v_x(x) \frac{\partial c}{\partial x}$$

Both models recover the **Levich equation** for the mass-transport-limited plateau current:

$$I_L = 0.620 \, n F A D^{2/3} \nu^{-1/6} c^* \omega^{1/2}$$

When interfacial kinetics are also finite, the total current obeys the **Koutecký–Levich** relation:

$$\frac{1}{I} = \frac{1}{I_K} + \frac{1}{I_L} = \frac{1}{I_K} + \frac{1}{B \, \omega^{1/2}}$$

allowing kinetic current $I_K$ and diffusivity $D$ to be separated cleanly by varying $\omega$.

---

## Where to Go Next

- **User Guide**: [Step Techniques (CA & CP)](../../user-guide/techniques/step.md) · [Rotating Disk Electrode (RDE)](../../user-guide/techniques/rde.md) · [Analytical Models](../../user-guide/analytical/models.md)
- **Tutorials**: [Chronoamperometry Quickstart](../../tutorials/start/ca_quickstart.ipynb)
- **Numerics**: [Spatial Grids & Meshes](../numerics/grids.md) (how Soft Potato resolves $\delta$ and $\delta_N$)
