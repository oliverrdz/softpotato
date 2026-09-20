# species module example
```python
from softpotato.core.species import Species[span_0](start_span)[span_0](end_span)

# Define the primary electroactive species (Oxidant)
# D is in cm^2/s, c_bulk is in mol/cm^3 (1 mM = 1e-6 mol/cm^3)
species_O = Species(name="O", D=1.0e-5, c_bulk=1.0e-6, charge=1)[span_1](start_span)[span_1](end_span)

# Define the product species (Reductant)
# Bulk concentration is typically 0 for the product before the experiment begins
species_R = Species(name="R", D=1.0e-5, c_bulk=0.0, charge=0)[span_2](start_span)[span_2](end_span)

# Define a homogeneous chemical reactant (e.g., for an EC' catalytic mechanism)
# Often present in large excess (e.g., 10 mM)
species_Z = Species(name="Z", D=1.0e-5, c_bulk=1.0e-5, charge=-1)[span_3](start_span)[span_3](end_span)

# Example of grouping them for injection into a mechanism/reaction parser
mechanism_species = [species_O, species_R, species_Z]
```

# reaction module example
```python
from softpotato.core.species import Species[span_0](start_span)[span_0](end_span)
from softpotato.core.reactions import HeterogeneousReaction, HomogeneousReaction, Mechanism[span_1](start_span)[span_1](end_span)

# 1. Define Species
# Conventions: D in cm²/s, c_bulk in mol/cm³ (note: 1 mM = 1e-6 mol/cm³)
species_o = Species(name="O", D=1.0e-5, c_bulk=1.0e-6, charge=1)[span_2](start_span)[span_2](end_span)
species_r = Species(name="R", D=1.0e-5, c_bulk=0.0, charge=0)[span_3](start_span)[span_3](end_span)
species_z = Species(name="Z", D=1.0e-5, c_bulk=0.0, charge=0)[span_4](start_span)[span_4](end_span)

# 2. Define Heterogeneous Kinetics (Electrode Surface)
# E0 in V, k0 in cm/s, alpha is dimensionless, T in Kelvin
# Here we define a standard reversible (Nernstian) single-electron transfer: O + e- <=> R
e_step = HeterogeneousReaction(
    ox=species_o,
    red=species_r,
    n=1,
    E0=-0.2,       # Standard reduction potential
    k0=1e4,        # Fast standard rate constant ensures reversibility
    alpha=0.5,     # Symmetric charge transfer
    T=298.15
)[span_5](start_span)[span_5](end_span)

# 3. Define Homogeneous Kinetics (Bulk Solution)
# Here we define a first-order following chemical reaction (EC mechanism): R -> Z
# kf in s⁻¹ for first-order reactions
c_step = HomogeneousReaction(
    reactants=[species_r],
    products=[species_z],
    kf=15.0,       # Forward rate constant
    kb=0.0,        # Irreversible chemical step
    stoich_reactants=[1],
    stoich_products=[1]
)[span_6](start_span)[span_6](end_span)

# 4. Construct the Mechanism
# This container encapsulates the full state vector and boundary logic for the FDM solver
ec_mechanism = Mechanism(
    species=[species_o, species_r, species_z],
    e_reactions=[e_step],
    c_reactions=[c_step]
)[span_7](start_span)[span_7](end_span)
```