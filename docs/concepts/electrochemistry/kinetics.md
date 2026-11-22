# Electrode Kinetics

Interfacial electron transfer governs how rapidly an oxidized species $\text{Ox}$ and a reduced species $\text{Red}$ exchange electrons at an electrode surface:

$$\text{Ox} + n e^- \underset{k_b(E)}{\overset{k_f(E)}{\rightleftharpoons}} \text{Red}$$

In Soft Potato, you choose between **thermodynamic equilibrium** (`NernstianEquilibrium`) and **finite-rate kinetics** (`ButlerVolmer`, `TafelKinetics`), depending on whether electron transfer is fast or rate-limiting relative to mass transport.

## Sign Convention (IUPAC)

Soft Potato follows the **IUPAC convention** by default (`anodic_positive=True`):

- **Anodic (oxidation) current is positive** ($I > 0$ when $E > E^{0\prime}$, net $\text{Red} \to \text{Ox}$).
- **Cathodic (reduction) current is negative** ($I < 0$ when $E < E^{0\prime}$, net $\text{Ox} \to \text{Red}$).

For an interfacial reduction flux $J = k_f(E)\,c_{\text{Ox}}(0, t) - k_b(E)\,c_{\text{Red}}(0, t)$, the Faradaic current density is:

$$j = -n F J = n F \left[ k_b(E)\,c_{\text{Red}}(0, t) - k_f(E)\,c_{\text{Ox}}(0, t) \right]$$

---

## Nernst Equilibrium vs. Butler–Volmer Kinetics

### Nernst Equilibrium (`NernstianEquilibrium`)

When the heterogeneous rate constant $k^0$ is much faster than the rate of mass transport, the electrode interface remains in local thermodynamic equilibrium at all times. The surface concentrations are locked to the applied potential $E(t)$ by the **Nernst equation**:

$$E(t) = E^{0\prime} + \frac{R T}{n F} \ln \frac{c_{\text{Ox}}(0, t)}{c_{\text{Red}}(0, t)}$$

Equivalently, defining the dimensionless potential $\theta = \frac{n F}{R T}(E - E^{0\prime})$, the surface ratio is:

$$\frac{c_{\text{Ox}}(0, t)}{c_{\text{Red}}(0, t)} = \exp(\theta)$$

In a simulation, `NernstianEquilibrium` enforces this ratio as a **Dirichlet-like coupled boundary condition** at the electrode surface ($x = 0$), assuming equal diffusivities or flux balance $D_{\text{Ox}} \partial_x c_{\text{Ox}} + D_{\text{Red}} \partial_x c_{\text{Red}} = 0$.

### Butler–Volmer Kinetics (`ButlerVolmer`)

When electron transfer has a finite barrier, surface concentrations deviate from Nernstian equilibrium and the net flux is governed by the **Butler–Volmer** rate constants:

$$k_f(E) = k^0 \exp\!\left[-\frac{\alpha n F}{R T}(E - E^{0\prime})\right], \qquad k_b(E) = k^0 \exp\!\left[\frac{(1 - \alpha) n F}{R T}(E - E^{0\prime})\right]$$

where:

- $E^{0\prime}$ ($\text{V}$) is the formal reduction potential.
- $k^0$ ($\text{m}/\text{s}$) is the standard heterogeneous rate constant (evaluated at $E = E^{0\prime}$).
- $\alpha \in (0, 1)$ is the cathodic charge-transfer coefficient (often $\approx 0.5$ for symmetric barriers).
- $\eta = E - E^{0\prime}$ is the overpotential (relative to $E^{0\prime}$, or relative to $E_{\text{eq}}$ when $c_{\text{Ox}}^* \neq c_{\text{Red}}^*$).

At equilibrium ($E = E_{\text{eq}}$, $j = 0$), both opposing branches equal the **exchange current density** $j_0$:

$$j_0 = n F k^0 \left(c_{\text{Ox}}^*\right)^{1-\alpha} \left(c_{\text{Red}}^*\right)^{\alpha}$$

In a simulation, `ButlerVolmer` translates $k_f(E)$ and $k_b(E)$ into **Robin / Neumann flux boundary conditions** at $x = 0$:

$$D_{\text{Ox}} \left.\frac{\partial c_{\text{Ox}}}{\partial x}\right|_{x=0} = k_f(E)\,c_{\text{Ox}}(0, t) - k_b(E)\,c_{\text{Red}}(0, t)$$

### Tafel Limit (`TafelKinetics`)

At large overpotentials ($|\eta| \gg \frac{R T}{n F} \approx 25.7\text{ mV}$ at $298\text{ K}$), the reverse reaction becomes negligible. The Butler–Volmer equation simplifies to the exponential **Tafel** relation:

$$\eta = a + b \log_{10} |j|, \qquad |b_{\text{cathodic}}| = \frac{2.303\,R T}{\alpha n F}, \qquad |b_{\text{anodic}}| = \frac{2.303\,R T}{(1 - \alpha) n F}$$

At $298\text{ K}$ with $n = 1$ and $\alpha = 0.5$, the classic Tafel slope is $\approx 118\text{ mV}/\text{decade}$.

---

## Reversibility Regimes

Electrochemical "reversibility" is not an intrinsic property of a molecule alone—it is a **competition between electron-transfer kinetics ($k^0$) and the experimental timescale** (such as scan rate $v$ in cyclic voltammetry, or rotation rate $\omega$ in RDE).

For cyclic voltammetry, Matsuda and Ayabe defined the dimensionless parameter $\Lambda$, and Nicholson defined $\Psi$:

$$\Lambda = \frac{k^0}{\sqrt{\frac{n F v D}{R T}}}, \qquad \Psi = \gamma^\alpha \frac{k^0}{\sqrt{\pi \frac{n F v D}{R T}}}, \qquad \gamma = \frac{D_{\text{Ox}}}{D_{\text{Red}}}$$

| Regime | Matsuda–Ayabe $\Lambda$ | CV Peak Separation $\Delta E_p$ ($298\text{ K}$) | Peak Current Scaling | Best Model in Soft Potato |
| :--- | :--- | :--- | :--- | :--- |
| **Reversible** | $\Lambda \gtrsim 15$ | $\Delta E_p \approx \frac{59\text{ mV}}{n}$ (independent of $v$) | $I_p \propto v^{1/2}$ (Randles–Ševčík) | `NernstianEquilibrium` or `ButlerVolmer` with large $k^0$ |
| **Quasi-reversible** | $10^{-2(1+\alpha)} \lesssim \Lambda \lesssim 15$ | $\Delta E_p > \frac{59\text{ mV}}{n}$ (grows with $v$) | Non-linear in $v^{1/2}$; shifted peaks | `ButlerVolmer` ($k^0, \alpha, E^{0\prime}$) |
| **Irreversible** | $\Lambda \lesssim 10^{-2(1+\alpha)}$ | Reverse peak absent; $E_p$ shifts by $\frac{29.6\text{ mV}}{\alpha n}$ per decade of $v$ | $I_p \propto v^{1/2}$ (with $\sqrt{\alpha}$ factor) | `ButlerVolmer` (small $k^0$) or `TafelKinetics` |

!!! tip "Laboratory Intuition: Slowing Down vs. Speeding Up"
    Increasing the scan rate $v$ shrinks the diffusion layer ($\delta \propto v^{-1/2}$) and increases the diffusive flux. A redox couple that looks **reversible** at $10\text{ mV/s}$ can become **quasi-reversible** at $10\text{ V/s}$ because mass transport outpaces $k^0$.

---

## Where to Go Next

- **User Guide**: [Voltammetry (LSV & CV)](../../user-guide/techniques/voltammetry.md) · [Analytical Models](../../user-guide/analytical/models.md) · [Diagnostics & Post-Processing](../../user-guide/postprocessing/diagnostics.md) (`analyze_nicholson`, `analyze_tafel`)
- **Tutorials**: [Cyclic Voltammetry Quickstart](../../tutorials/start/cv_quickstart.ipynb)
