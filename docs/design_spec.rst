API & UI Design Specifications
==============================

This document outlines the user interface (UI) and API design principles for **Soft Potato 3.0**. It serves as a design guide for developers implementing and extending the simulation engine and analytical modules.

Design Principles
-----------------

1. **Strict CGS Units**:
   Soft Potato enforces the CGS unit system across all modules to maintain numerical stability and consistency with classic electrochemical literature:
   
   * **Diffusion coefficient (:math:`D`)**: :math:`\text{cm}^2/\text{s}`
   * **Concentration (:math:`c`)**: :math:`\text{mol}/\text{cm}^3` (Note: :math:`1\text{ mM} = 10^{-6}\text{ mol}/\text{cm}^3`)
   * **Electrode area (:math:`A`)**: :math:`\text{cm}^2`
   * **Electrode radius (:math:`r`)**: :math:`\text{cm}`
   * **Potential (:math:`E`)**: :math:`\text{V}`
   * **Scan rate (:math:`\nu`)**: :math:`\text{V}/\text{s}`
   * **Time (:math:`t`)**: :math:`\text{s}`
   * **Current (:math:`i`)**: :math:`\text{A}`

2. **Vectorized Closed-Form Analytical Functions**:
   Analytical benchmark equations accept NumPy arrays directly, eliminating slow Python loops and allowing seamless parameter sweeping and plotting.

3. **Modular, Composable Architecture**:
   Simulation workflows decouple physical definitions into distinct domains:
   
   * :mod:`softpotato.core`: Species, reactions, and reaction mechanisms.
   * :mod:`softpotato.kinetics`: Electron transfer and chemical kinetic models (Butler-Volmer, Nernst, first/second order).
   * :mod:`softpotato.geometry`: Spatial discretization and electrode geometries (Planar, Spherical, Uniform/Expanding grids).
   * :mod:`softpotato.techniques`: Electrochemical excitation signals (Cyclic Voltammetry, Potential Step).
   * :mod:`softpotato.simulate`: PDE solvers and orchestrators (EFD, IFD).

---

API Design Examples
-------------------

Analytical Module
~~~~~~~~~~~~~~~~~

Direct, vectorized evaluation of classic electrochemical solutions (Randles-Sevcik, Cottrell, and Saito steady-state microdisc):

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Import analytical equations directly from the Soft Potato facade
   from softpotato.analytical import randles_sevcik, cottrell, steady_state_microdisc

   # --- Parameters (Strict CGS Units Enforced by Soft Potato) ---
   # Standard T = 298.15 K handled internally by the analytical module
   n_electrons = 1
   D_O = 1e-5            # Diffusion coefficient (cm^2/s)
   C_bulk = 1e-6         # Bulk concentration (mol/cm^3) -> 1 mM
   area = 0.0707         # Planar macroelectrode area (cm^2)

   # --- Independent Variable Arrays (Vectorized) ---
   v_array = np.linspace(0.01, 1.0, 200)      # Scan rate (V/s)
   t_array = np.linspace(0.001, 5.0, 500)     # Time (s)
   r_array = np.linspace(1e-4, 25e-4, 200)    # Microdisc radius (cm)

   # --- Compute Analytical Solutions ---
   i_p = randles_sevcik(n=n_electrons, area=area, D=D_O, c_bulk=C_bulk, scan_rate=v_array)
   i_t = cottrell(t=t_array, n=n_electrons, area=area, D=D_O, c_bulk=C_bulk)
   i_ss = steady_state_microdisc(n=n_electrons, a=r_array, D=D_O, c_bulk=C_bulk)


   # --- Visualization ---
   fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

   # 1. Randles-Sevcik: i_p vs. sqrt(v)
   axes[0].plot(np.sqrt(v_array), i_p * 1e6, color='#1f77b4', lw=2)
   axes[0].set_xlabel(r'$\nu^{1/2}$ / (V/s)$^{1/2}$')
   axes[0].set_ylabel(r'$i_p$ / $\mu$A')
   axes[0].set_title('Randles-Sevcik (Reversible CV)')
   axes[0].grid(True, linestyle='--', alpha=0.7)

   # 2. Cottrell: i vs. t
   axes[1].plot(t_array, i_t * 1e6, color='#d62728', lw=2)
   axes[1].set_xlabel('Time / s')
   axes[1].set_ylabel(r'$i$ / $\mu$A')
   axes[1].set_title('Cottrell (Planar Step)')
   axes[1].grid(True, linestyle='--', alpha=0.7)

   # 3. Microdisc: i_ss vs. r
   axes[2].plot(r_array * 1e4, i_ss * 1e9, color='#2ca02c', lw=2)
   axes[2].set_xlabel(r'Radius / $\mu$m')
   axes[2].set_ylabel(r'$i_{ss}$ / nA')
   axes[2].set_title('Saito (Steady-State Microdisc)')
   axes[2].grid(True, linestyle='--', alpha=0.7)

   plt.tight_layout()
   plt.show()

E Mechanism, Macroelectrode, Cyclic Voltammetry, Butler-Volmer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quasi-reversible single-electron transfer at a planar macroelectrode:

.. code-block:: python

   import softpotato as sp

   # 1. Species (D in cm^2/s, c_bulk in mol/cm^3)
   spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
   spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)

   # 2. Kinetics & Mechanism
   # Butler-Volmer kinetics for a quasi-reversible process
   bv = sp.kinetics.ButlerVolmer(k0=1e-3, alpha=0.5)
   rxn_E = sp.core.ElectrochemicalReaction(
       reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=bv
   )
   mechanism = sp.core.Mechanism([rxn_E])

   # 3. Grid & Geometry
   # Planar semi-infinite diffusion. Area in cm^2.
   grid = sp.geometry.UniformGrid(x_max=0.05, nodes=500)
   electrode = sp.geometry.PlanarElectrode(area=0.0707, grid=grid)

   # 4. Technique
   cv = sp.techniques.CyclicVoltammetry(
       E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
   )

   # 5. Simulate
   sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
   results = sim.run()

ErCi Mechanism, Macroelectrode, Cyclic Voltammetry, Expanding Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reversible electron transfer followed by an irreversible homogeneous chemical reaction, solved on an expanding spatial grid:

.. code-block:: python

   import softpotato as sp

   # 1. Species 
   spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
   spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)
   spec_Z = sp.core.Species(name="Z", D=1e-5, c_bulk=0.0) # Electroinactive product

   # 2. Kinetics & Mechanism
   # Er: Nernstian boundary condition (fast kinetics, governed by thermodynamics)
   nernst = sp.kinetics.Nernst()
   rxn_E = sp.core.ElectrochemicalReaction(
       reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=nernst
   )

   # Ci: Irreversible first-order chemical reaction (kf in s^-1, kb = 0)
   chem_irrev = sp.kinetics.FirstOrder(kf=10.0, kb=0.0)
   rxn_C = sp.core.ChemicalReaction(
       reactants=[spec_R], products=[spec_Z], kinetics=chem_irrev
   )
   mechanism = sp.core.Mechanism([rxn_E, rxn_C])

   # 3. Grid & Geometry
   # Expanding grid (gamma=1.05) to capture sharp concentration gradients near the electrode
   # critical for fast following chemical reactions.
   exp_grid = sp.geometry.ExpandingGrid(x_max=0.05, nodes=300, gamma=1.05)
   electrode = sp.geometry.PlanarElectrode(area=0.0707, grid=exp_grid)

   # 4. Technique & Simulate
   cv = sp.techniques.CyclicVoltammetry(
       E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
   )
   sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
   results = sim.run()

E Mechanism, Spherical Electrode, Cyclic Voltammetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-electron transfer at a spherical electrode (e.g., hanging mercury drop or ultramicroelectrode):

.. code-block:: python

   import softpotato as sp

   # 1. Species
   spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
   spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)

   # 2. Kinetics & Mechanism
   bv = sp.kinetics.ButlerVolmer(k0=1e-3, alpha=0.5)
   rxn_E = sp.core.ElectrochemicalReaction(
       reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=bv
   )
   mechanism = sp.core.Mechanism([rxn_E])

   # 3. Grid & Geometry
   # Spherical electrode (radius strictly in cm).
   grid = sp.geometry.UniformGrid(x_max=0.05, nodes=500)
   electrode = sp.geometry.SphericalElectrode(radius=0.01, grid=grid)

   # 4. Technique & Simulate
   cv = sp.techniques.CyclicVoltammetry(
       E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
   )
   sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
   results = sim.run()


