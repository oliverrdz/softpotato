from softpotato.physics.species import Species

# 1. Define an Oxidized species with standard properties
# D = 1e-9 m²/s, initial concentration = 1.0 mol/m³
ox_species = Species(D=1e-9, c0=1.0)

# 2. Define a Reduced species
# It has the same diffusion coefficient but starts with zero concentration in the bulk
red_species = Species(D=1e-9, c0=0.0)

# 3. Define a Catalytic intermediate or secondary product
# It might have a different diffusion coefficient depending on its size/charge
p_species = Species(D=5e-10, c0=0.0)

# Create a dictionary mapping the logical identifiers to their physical representations
species_map = {"O": ox_species, "R": red_species, "P": p_species}
