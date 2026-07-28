from softpotato.mechanism.steps import ElectronTransferStep
from softpotato.mechanism.builder import build

# ==========================================
# 1. Defining a Single E Mechanism
# ==========================================
# A standard E mechanism consists of one electron transfer step.
e_step = ElectronTransferStep(
    ox_species="O",
    red_species="R",
    n_electrons=1,
    e_formal=0.0,    # Volts
    k0=0.01,         # m/s
    alpha=0.5
)

# The builder aggregates the list of steps into a unified Mechanism object
e_mechanism = build([e_step])

print("--- E Mechanism ---")
print(f"Total Steps: {e_mechanism.n_steps}")
print(f"Participating Species: {e_mechanism.get_species()}\n")


# ==========================================
# 2. Defining a Sequential EE Mechanism
# ==========================================
# An EE mechanism involves two consecutive electron transfers,
# typically forming an intermediate species (e.g., O -> I -> R).

# First electron transfer (O to Intermediate)
ee_step_1 = ElectronTransferStep(
    ox_species="O",
    red_species="I",
    n_electrons=1,
    e_formal=0.20,   
    k0=0.05
)

# Second electron transfer (Intermediate to R)
ee_step_2 = ElectronTransferStep(
    ox_species="I",
    red_species="R",
    n_electrons=1,
    e_formal=-0.15,  
    k0=0.001
)

# The builder combines both steps into a single composite Mechanism
ee_mechanism = build([ee_step_1, ee_step_2])

print("--- EE Mechanism ---")
print(f"Total Steps: {ee_mechanism.n_steps}")
print(f"Participating Species: {ee_mechanism.get_species()}")