from softpotato.mechanism.builder import build
from softpotato.mechanism.steps import ChemicalStep, ElectronTransferStep

# ==========================================
# 1. E Mechanism (Single Electron Transfer)
# ==========================================
# Represents: Ox + e- <=> Red
e_step = ElectronTransferStep(
    ox_species="O",
    red_species="R",
    n_electrons=1,
    e_formal=0.0,  # Volts (V)
    k0=0.01,  # Rate constant in SI units (m/s)
)

e_mechanism = build([e_step])

print("--- E Mechanism ---")
print(f"Total Steps: {e_mechanism.n_steps}")
print(f"Participating Species: {e_mechanism.get_species()}\n")


# ==========================================
# 2. EE Mechanism (Sequential Electron Transfers)
# ==========================================
# Step 1: Ox + e- <=> Int (Intermediate)
# Step 2: Int + e- <=> Red
ee_step_1 = ElectronTransferStep(
    ox_species="O", red_species="Int", n_electrons=1, e_formal=0.20, k0=0.05
)

ee_step_2 = ElectronTransferStep(
    ox_species="Int", red_species="R", n_electrons=1, e_formal=-0.15, k0=0.001
)

ee_mechanism = build([ee_step_1, ee_step_2])

print("--- EE Mechanism ---")
print(f"Total Steps: {ee_mechanism.n_steps}")
print(f"Participating Species: {ee_mechanism.get_species()}\n")


# ==========================================
# 3. EC Mechanism (Electron Transfer followed by Chemical Reaction)
# ==========================================
# Step 1 (E): Ox + e- <=> Red
# Step 2 (C): Red <=> Product
ec_step_1 = ElectronTransferStep(
    ox_species="O", red_species="R", n_electrons=1, e_formal=0.10, k0=0.02
)

ec_step_2 = ChemicalStep(
    reactants=["R"],
    products=["P"],
    kf=10.0,  # Forward homogeneous rate constant (1/s)
    kb=0.1,  # Backward homogeneous rate constant (1/s)
)

ec_mechanism = build([ec_step_1, ec_step_2])

print("--- EC Mechanism ---")
print(f"Total Steps: {ec_mechanism.n_steps}")
print(f"Participating Species: {ec_mechanism.get_species()}")
