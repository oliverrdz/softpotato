# Reaction Mechanisms

Real electrochemical systems often involve more than a single electron transfer: electrogenerated intermediates can react in solution, undergo a second electron transfer, or regenerate the starting material. Soft Potato models these coupled systems using `Mechanism`, combining heterogeneous `ElectrochemicalStep` boundaries with homogeneous `ReactionNetwork` terms in the reaction–diffusion PDE:

$$\frac{\partial c_i}{\partial t} = D_i \nabla^2 c_i + R_i(\mathbf{c}, x, t)$$

where $R_i(\mathbf{c}, x, t)$ is the net volumetric production/consumption rate of species $i$ from all homogeneous reactions.

---

## Canonical Mechanisms

In electrochemical nomenclature, **E** denotes a heterogeneous electron transfer at the electrode surface, and **C** denotes a homogeneous chemical reaction in solution.

| Mechanism | Reaction Scheme | Typical Voltammetric Signature | Soft Potato Building Blocks |
| :--- | :--- | :--- | :--- |
| **`E`** | $\text{Ox} + n e^- \rightleftharpoons \text{Red}$ | Classic reversible or quasi-reversible CV peaks ($I_{pa}/I_{pc} = 1$ for equal $D$). | `ElectrochemicalStep(ox, red, kinetics)` |
| **`EC`** | $\text{Ox} + n e^- \rightleftharpoons \text{Red}$<br>$\text{Red} \xrightarrow{k_c} \text{P}$ | Reverse (anodic) peak shrinks or disappears as $k_c / v$ increases; forward peak shifts positively. | `ElectrochemicalStep` + `FirstOrderReaction(red, P, k=...)` (or `ReversibleReaction`) |
| **`EE`** | $\text{A} + n_1 e^- \rightleftharpoons \text{B}$ ($E_1^0$)<br>$\text{B} + n_2 e^- \rightleftharpoons \text{C}$ ($E_2^0$) | Two distinct waves if $E_1^0 - E_2^0 \gg \frac{RT}{F}$, merging into a single multi-electron wave when $E_2^0 \ge E_1^0$. | Two `ElectrochemicalStep`s in `Mechanism` (or disproportionation $2\text{B} \rightleftharpoons \text{A} + \text{C}$ via `DisproportionationReaction`) |
| **`EC'` (Catalytic)** | $\text{Ox} + n e^- \rightleftharpoons \text{Red}$<br>$\text{Red} + \text{Z} \xrightarrow{k_{\text{cat}}} \text{Ox} + \text{P}$ | Forward current is amplified into a **sigmoidal plateau** ($I_{\text{cat}} \propto \sqrt{k_{\text{cat}}}$) while the reverse peak vanishes. | `ElectrochemicalStep` + `CatalyticReaction` |

---

## How Mechanisms Map to the PDE

### 1. Heterogeneous Steps (`E`)

Each `ElectrochemicalStep` applies **only at the electrode boundary** ($x = 0$, or `electrode_side="right"`). It does not appear in $R_i(\mathbf{c}, x, t)$ inside the bulk solution; instead, it sets the surface flux or concentration ratio for the coupled `ox`/`red` pair.

- For an **`EE`** mechanism ($\text{A} \rightleftharpoons \text{B} \rightleftharpoons \text{C}$), the intermediate $\text{B}$ participates in two boundary conditions simultaneously: its net interfacial flux is the sum of the reduction flux from $\text{A} \to \text{B}$ and the oxidation/reduction flux from $\text{B} \to \text{C}$.

### 2. Homogeneous Steps (`C`)

Each `HomogeneousReaction` (such as `FirstOrderReaction`, `SecondOrderReaction`, `ReversibleReaction`, `CatalyticReaction`, `DimerizationReaction`, `DisproportionationReaction`, or `CustomReaction`) contributes to the volumetric source term $R_i(\mathbf{c}, x, t)$ at every interior grid node.

For example, in an **`EC`** mechanism where $\text{Red} \xrightarrow{k_c} \text{P}$:

$$\frac{\partial c_{\text{Ox}}}{\partial t} = D_{\text{Ox}} \nabla^2 c_{\text{Ox}}, \qquad \frac{\partial c_{\text{Red}}}{\partial t} = D_{\text{Red}} \nabla^2 c_{\text{Red}} - k_c \, c_{\text{Red}}, \qquad \frac{\partial c_{\text{P}}}{\partial t} = D_{\text{P}} \nabla^2 c_{\text{P}} + k_c \, c_{\text{Red}}$$

And in a pseudo-first-order **catalytic `EC'`** mechanism where $\text{Red}$ regenerates $\text{Ox}$ at rate $k_{\text{cat}} c_{\text{Red}}$:

$$\frac{\partial c_{\text{Ox}}}{\partial t} = D_{\text{Ox}} \nabla^2 c_{\text{Ox}} + k_{\text{cat}} \, c_{\text{Red}}, \qquad \frac{\partial c_{\text{Red}}}{\partial t} = D_{\text{Red}} \nabla^2 c_{\text{Red}} - k_{\text{cat}} \, c_{\text{Red}}$$

---

## Timescale Competition & Stiffness

Just like heterogeneous kinetics, homogeneous chemistry is governed by a **dimensionless kinetic parameter** comparing the chemical rate $k_c$ to the experimental timescale $t_{\exp}$ (in CV, $t_{\exp} \sim \frac{R T}{n F v}$):

$$\lambda = k_c \, t_{\exp} = \frac{k_c R T}{n F v}$$

- **Slow chemistry ($\lambda \ll 1$)**: $\text{Red}$ diffuses away before reacting; the response resembles a simple `E` mechanism.
- **Fast chemistry ($\lambda \gg 1$)**: $\text{Red}$ is consumed within a thin **reaction–diffusion layer** of thickness $\mu = \sqrt{D / k_c} \ll \delta$ near the electrode.

!!! warning "Numerical Stiffness & Grid Resolution"
    When $k_c$ is large ($\lambda \gg 1$), the reaction layer $\mu = \sqrt{D / k_c}$ can be orders of magnitude thinner than the diffusion layer $\delta = 6\sqrt{D t_{\max}}$. Use a stretched mesh (`GeometricExpandingMesh` or `ExponentialMesh`) to resolve $\mu$ near $x = 0$, and use an **implicit** or **stiff** solver (`"scipy_ivp"` with `"Radau"`/`"BDF"`, `"implicit"`, or `"crank_nicolson"`) rather than an explicit solver. `ReactionNetwork` and `Mechanism` automatically supply analytical Jacobians to `ScipyIVPSolver` for fast, stable stiff integration.

---

## Where to Go Next

- **User Guide**: [Techniques & Waveforms](../../user-guide/index.md) · [Voltammetry (LSV & CV)](../../user-guide/techniques/voltammetry.md)
- **Numerics**: [Spatial Grids & Meshes](../numerics/grids.md) · [Time-Stepping & Solvers](../numerics/solvers.md)
