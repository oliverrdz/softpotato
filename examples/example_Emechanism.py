from softpotato.mechanism.steps import ElectronTransferStep
import softpotato.mechanism.builder as mechanism_builder

# 1. Define a standard E step using default parameters
# By default: n_electrons=1, e_formal=0.0 V, k0=0.01 m/s, alpha=0.5
simple_e_step = ElectronTransferStep(
    ox_species="Ox", 
    red_species="Red"
)

# 2. Define a fully customized E step for a specific system
custom_e_step = ElectronTransferStep(
    ox_species="A",
    red_species="B",
    n_electrons=2,       # A two-electron transfer
    e_formal=0.25,       # Formal potential in Volts (V)
    k0=0.005,            # Heterogeneous standard rate constant in SI units (m/s)
    alpha=0.3            # Charge transfer coefficient favoring oxidation
)

# 3. Build the composite mechanism
# The builder merges the elementary steps into an executable reaction scheme
e_mechanism = mechanism_builder.build([custom_e_step])

# Display the parameters of our defined mechanism step
print(f"Oxidized Species: {custom_e_step.ox_species}")
print(f"Reduced Species: {custom_e_step.red_species}")
print(f"Formal Potential (E^0): {custom_e_step.e_formal} V")
print(f"Rate Constant (k_0): {custom_e_step.k0} m/s")