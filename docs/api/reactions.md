# Reactions Module (`softpotato.reactions`)

The `softpotato.reactions` subpackage provides modular abstractions for multi-species chemical metadata, homogeneous (solution-phase) reaction kinetics with automated analytical Jacobians, heterogeneous interfacial charge-transfer kinetics, and unified electrochemical mechanism containers that plug directly into [`DiffusionProblem`][softpotato.solver.DiffusionProblem] and [`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver].

---

## Theoretical Background

### 1. Coupled Reaction–Diffusion Transport & Analytical Jacobians

In a 1D multi-species electrochemical system with coupled solution-phase chemistry, the concentration $c_i(x, t)$ of each chemical species $i$ evolves according to the reaction–diffusion partial differential equation:

$$\frac{\partial c_i}{\partial t} = D_i \nabla^2 c_i + R_i(\mathbf{c}, x, t), \qquad R_i(\mathbf{c}) = \sum_j \nu_{i,j}\, r_j(\mathbf{c})$$

where $D_i > 0$ is the diffusion coefficient of species $i$, $\nu_{i,j}$ is its net stoichiometric coefficient in reaction $j$ (negative for consumed reactants, positive for produced products), and $r_j(\mathbf{c})$ is the volumetric reaction rate.

Because homogeneous chemical rate constants $k$ can span many orders of magnitude ($10^{-2}$ to $10^{6}\,\text{s}^{-1}$), reaction–diffusion ODE systems produced by Method-of-Lines discretization are frequently **stiff**. Every [`HomogeneousReaction`][softpotato.reactions.HomogeneousReaction] subclass provides exact analytical partial derivatives $\partial r_j / \partial c_k$, allowing [`ReactionNetwork`][softpotato.reactions.ReactionNetwork] to assemble the exact local and global sparse kinetic Jacobian:

$$J_{ik} = \frac{\partial R_i}{\partial c_k} = \sum_j \nu_{i,j} \frac{\partial r_j}{\partial c_k}$$

[`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver] automatically combines the sparse spatial diffusion Jacobian with the sparse kinetic Jacobian for implicit stiff integrators (`Radau` and `BDF`), dramatically accelerating convergence and maintaining machine-precision mass conservation.

### 2. Characteristic Reaction Layer Thickness & Expanding Grids

For a coupled catalytic (EC') or follow-up (EC) mechanism with pseudo-first-order rate constant $k'$, homogeneous chemical conversion confines concentration gradients within a **reaction layer** of characteristic thickness:

$$\mu = \sqrt{\frac{D}{k'}}$$

When $k'$ is large, $\mu \ll \delta = 6\sqrt{D t_{\max}}$ and the interfacial concentration profile becomes exponentially steep near $x = 0$:

$$c_{\text{Ox}}(x) = c^* \left(1 - e^{-x/\mu}\right), \qquad I_{\text{cat}} = -n F A c^* \sqrt{D k'}$$

Pairing [`GeometricExpandingMesh`][softpotato.grid.GeometricExpandingMesh] or [`ExponentialMesh`][softpotato.grid.ExponentialMesh] from [`softpotato.grid`][softpotato.grid] with [`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver] (`method="Radau"`) concentrates spatial nodes inside the thin reaction layer $\mu$ while smoothly coarsening into the bulk, achieving $< 1\%$ current accuracy with fewer than 100 grid nodes.

### 3. Heterogeneous Interfacial Charge-Transfer Kinetics

For a single-step interfacial redox reaction at the electrode surface ($x = 0$):

$$\text{Ox} + n e^- \underset{k_b(E)}{\overset{k_f(E)}{\rightleftharpoons}} \text{Red}$$

- **Butler–Volmer kinetics** ([`ButlerVolmer`][softpotato.reactions.ButlerVolmer]) relates forward (reduction) and backward (oxidation) rate constants to the overpotential $\eta = E(t) - E^{0\prime}$:

  $$k_f(E) = k^0 \exp\!\left(-\frac{\alpha n F}{R T}(E - E^{0\prime})\right), \qquad k_b(E) = k^0 \exp\!\left(\frac{(1 - \alpha) n F}{R T}(E - E^{0\prime})\right)$$

  yielding the net interfacial reduction flux $J = k_f(E) c_{\text{Ox}}(0, t) - k_b(E) c_{\text{Red}}(0, t)$ and IUPAC Faradaic current density (anodic positive):

  $$j = -n F J = n F \left[k_b(E) c_{\text{Red}}(0, t) - k_f(E) c_{\text{Ox}}(0, t)\right]$$

- **Tafel kinetics** ([`TafelKinetics`][softpotato.reactions.TafelKinetics] / [`Tafel`][softpotato.reactions.TafelKinetics]) isolates the dominant unidirectional branch at high overpotential ($|\eta| \gg RT/nF$), retaining only $k_f(E)$ on the cathodic branch or $k_b(E)$ on the anodic branch.

- **Nernstian equilibrium** ([`NernstianEquilibrium`][softpotato.reactions.NernstianEquilibrium]) models infinitely fast (reversible) electron transfer by enforcing the thermodynamic surface concentration ratio directly as time-dependent Dirichlet boundary conditions:

  $$\frac{c_{\text{Ox}}(0, t)}{c_{\text{Red}}(0, t)} = \exp\!\left(\frac{n F}{R T}\left(E(t) - E^{0\prime}\right)\right)$$

---

## API Overview

### Chemical Species & Collections

| Class | Key Parameters / Properties | Description |
| :--- | :--- | :--- |
| [`Species`][softpotato.reactions.Species] | `name`, `diffusivity`, `bulk_concentration`, `charge`, `standard_potential`, `n_electrons` | Immutable dataclass encapsulating a chemical species and its physicochemical transport/thermodynamic metadata. |
| [`SpeciesCollection`][softpotato.reactions.SpeciesCollection] | `names`, `diffusivities`, `bulk_concentrations`, `n_electrons_map` | Ordered mapping container managing a group of `Species` objects with dict-like lookup and `DiffusionProblem` export helpers. |

### Homogeneous Reaction Kinetics

| Class | Mechanism / Rate Law | Key Parameters | Description |
| :--- | :--- | :--- | :--- |
| [`HomogeneousReaction`][softpotato.reactions.HomogeneousReaction] | Abstract | `stoichiometry`, `rate()`, `rate_derivatives()` | Abstract base class for custom solution-phase reactions with analytical or finite-difference Jacobians. |
| [`MassActionReaction`][softpotato.reactions.MassActionReaction] | $\sum a_i A_i \rightleftharpoons \sum b_j B_j$ | `reactants`, `products`, `k_f`, `k_b` | General elementary mass-action reaction supporting arbitrary integer stoichiometric orders and reversibility. |
| [`FirstOrderReaction`][softpotato.reactions.FirstOrderReaction] | $A \xrightarrow{k} B$ | `reactant`, `product`, `k` | Irreversible unimolecular reaction ($r = k c_A$), e.g. EC follow-up decay. |
| [`ReversibleReaction`][softpotato.reactions.ReversibleReaction] | $A \underset{k_b}{\overset{k_f}{\rightleftharpoons}} B$ | `reactant`, `product`, `k_f`, `k_b` | Reversible first-order equilibrium ($r = k_f c_A - k_b c_B$) with equilibrium constant $K_{\text{eq}} = k_f / k_b$. |
| [`SecondOrderReaction`][softpotato.reactions.SecondOrderReaction] | $A + B \xrightarrow{k} C$ | `reactant_1`, `reactant_2`, `product`, `k` | Irreversible bimolecular reaction ($r = k c_A c_B$). |
| [`DimerizationReaction`][softpotato.reactions.DimerizationReaction] | $2 A \xrightarrow{k} A_2$ | `monomer`, `dimer`, `k` | Irreversible homodimerization ($r = k c_A^2$). |
| [`DisproportionationReaction`][softpotato.reactions.DisproportionationReaction] | $2 B \xrightarrow{k} A + C$ | `reactant`, `product_1`, `product_2`, `k` | Irreversible disproportionation ($r = k c_B^2$). |
| [`CatalyticReaction`][softpotato.reactions.CatalyticReaction] | $B (+ Z) \xrightarrow{k} A (+ P)$ | `reactant`, `regenerated`, `k`, `substrate`, `product` | Pseudo-first-order ($r = k c_B$) or bimolecular ($r = k c_B c_Z$) EC' catalytic regeneration step. |
| [`CustomReaction`][softpotato.reactions.CustomReaction] | User callable $r(\mathbf{c}, x, t)$ | `stoichiometry`, `rate`, `rate_derivatives` | Wraps arbitrary user-defined rate laws (e.g. Michaelis–Menten) with optional analytical or automatic finite-difference derivatives. |
| [`ReactionNetwork`][softpotato.reactions.ReactionNetwork] | $\mathbf{R}(\mathbf{c}) = \sum_j \boldsymbol{\nu}_j r_j(\mathbf{c})$ | `reactions`, `species` | Composes multiple homogeneous reactions and assembles vectorized rates and sparse block Jacobians for `DiffusionProblem`. |

### Heterogeneous Charge-Transfer Kinetics

| Class / Function | Key Parameters | Description |
| :--- | :--- | :--- |
| [`HeterogeneousKinetics`][softpotato.reactions.HeterogeneousKinetics] | `E0`, `n`, `T`, `R`, `F` | Abstract base class for finite-rate interfacial electron-transfer models providing `rate_constants`, `net_flux`, `current_density`, and `boundary_conditions`. |
| [`ButlerVolmer`][softpotato.reactions.ButlerVolmer] | `k0`, `alpha`, `E0`, `n`, `T` | Reversible and quasi-reversible Butler–Volmer charge-transfer kinetics. |
| [`TafelKinetics`][softpotato.reactions.TafelKinetics] (`Tafel`) | `k0`, `alpha`, `E0`, `n`, `branch`, `T` | High-overpotential / irreversible unidirectional Tafel kinetics (`'auto'`, `'cathodic'`, or `'anodic'`). |
| [`NernstianEquilibrium`][softpotato.reactions.NernstianEquilibrium] | `E0`, `n`, `T` | Reversible thermodynamic equilibrium helper computing surface concentration ratios and Dirichlet boundary conditions. |
| [`create_electrochemical_bcs`][softpotato.reactions.create_electrochemical_bcs] | `kinetics`, `potential`, `ox`, `red`, `electrode_side`, `bulk_bc` | Factory constructing matched `(left_bc, right_bc)` boundary condition tuples for a redox pair in `DiffusionProblem`. |

### Mechanism Containers

| Class | Key Parameters | Description |
| :--- | :--- | :--- |
| [`ElectrochemicalStep`][softpotato.reactions.ElectrochemicalStep] | `ox`, `red`, `kinetics`, `electrode_side`, `bulk_bc` | Declarative specification of a single interfacial redox step at an electrode boundary. |
| [`Mechanism`][softpotato.reactions.Mechanism] | `species`, `reactions`, `electrochemical_steps`, `name` | Complete chemical/electrochemical model independent of grid and solver, with `create_problem(...)` helper. |
| [`CustomMechanism`][softpotato.reactions.CustomMechanism] | `species`, `reactions`, `rates`, `local_jacobian_func`, `electrochemical_steps`, `boundary_conditions_func` | Declarative extension point combining built-in reactions, custom rate/Jacobian callables, and custom boundary conditions. |

---

## User Guide & Examples

### 1. Defining Species and a Catalytic (EC') Reaction Network

Define [`Species`][softpotato.reactions.Species] objects and combine homogeneous steps in a [`ReactionNetwork`][softpotato.reactions.ReactionNetwork], then pass them directly to [`DiffusionProblem`][softpotato.solver.DiffusionProblem]:

```python
import math
import numpy as np
from softpotato.constants import FARADAY
from softpotato.geometry import PlanarGeometry
from softpotato.grid import GeometricExpandingMesh
from softpotato.reactions import CatalyticReaction, ReactionNetwork, Species
from softpotato.solver import DiffusionProblem, DirichletBC, ScipyIVPSolver

D = 1e-9       # m^2/s
c_bulk = 1.0   # mol/m^3 (1 mM)
area = 1e-4    # m^2 (1 cm^2)
k_prime = 10.0 # s^-1 (pseudo-first-order catalytic rate constant)

# Reaction layer thickness mu = sqrt(D / k'); span 10 * mu with an expanding mesh
mu = math.sqrt(D / k_prime)
mesh = GeometricExpandingMesh(0.0, 10.0 * mu, n_points=80, gamma=1.06)

ox = Species("Ox", diffusivity=D, bulk_concentration=c_bulk, n_electrons=1)
red = Species("Red", diffusivity=D, bulk_concentration=0.0, n_electrons=-1)
network = ReactionNetwork([CatalyticReaction(red, ox, k=k_prime)])

problem = DiffusionProblem(
    grid=mesh,
    species=[ox, red],
    boundary_conditions={
        "Ox": (DirichletBC(0.0), DirichletBC(c_bulk)),
        "Red": (DirichletBC(c_bulk), DirichletBC(0.0)),
    },
    reactions=network,
    geometry=PlanarGeometry(area=area),
)

solver = ScipyIVPSolver(method="Radau", rtol=1e-7, atol=1e-10)
result = solver.solve(problem, t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 100))

i_cat_exact = -FARADAY * area * c_bulk * math.sqrt(D * k_prime)
print(f"Simulated steady-state current: {result.currents['Ox'][-1] * 1e3:.4f} mA")
print(f"Analytical catalytic limit:     {i_cat_exact * 1e3:.4f} mA")
```

### 2. Heterogeneous Kinetics & Mechanism Containers

Evaluate Butler–Volmer rate constants and construct boundary conditions via [`create_electrochemical_bcs`][softpotato.reactions.create_electrochemical_bcs] or package species, homogeneous steps, and interfacial kinetics together in a [`Mechanism`][softpotato.reactions.Mechanism]:

```python
from softpotato.grid import create_diffusion_mesh
from softpotato.reactions import (
    ButlerVolmer,
    ElectrochemicalStep,
    FirstOrderReaction,
    Mechanism,
    NernstianEquilibrium,
    Species,
)
from softpotato.solver import ScipyIVPSolver

# 1. Inspect Butler-Volmer kinetics
bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
kf, kb = bv.rate_constants(E=-0.1)
j0 = bv.exchange_current_density(c_ox=1.0, c_red=1.0)

# 2. Build a complete EC mechanism (reversible E step + follow-up chemical decay)
ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
prod = Species("P", diffusivity=1e-9, bulk_concentration=0.0)

ec_mech = Mechanism(
    species=[ox, red, prod],
    reactions=[FirstOrderReaction(red, prod, k=2.0)],
    electrochemical_steps=[
        ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=0.0, n=1))
    ],
    name="EC_Mechanism",
)

mesh = create_diffusion_mesh(D=1e-9, t_max=1.0, kind="geometric", n_points=60, gamma=1.08)
problem = ec_mech.create_problem(mesh, potential=-0.3)
result = ScipyIVPSolver(method="Radau").solve(problem, t_span=(0.0, 1.0))
```

### 3. Interactive Tutorials

For full worked examples with plots and analytical benchmarks, explore the interactive Jupyter notebooks:

- **Catalytic EC' Mechanism**: [`examples/catalytic_ec.ipynb`](../examples/catalytic_ec.ipynb) — chronoamperometry of an EC' catalytic cycle, steady-state concentration profiles, and validation against the Delahay–Stiehl and catalytic limiting current equations.
- **Heterogeneous Kinetics & Cyclic Voltammetry**: [`examples/heterogeneous_kinetics.ipynb`](../examples/heterogeneous_kinetics.ipynb) — Butler–Volmer cyclic voltammetry across reversible, quasi-reversible, and irreversible regimes, with $\Delta E_p$ vs. scan rate analysis against Randles–Ševčík and Nicholson theory.

---

## Module Contents

::: softpotato.reactions
