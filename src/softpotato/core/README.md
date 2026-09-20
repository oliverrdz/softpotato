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