import matplotlib.pyplot as plt

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import (
    ElectrochemicalStep,
    Mechanism,
    NernstianEquilibrium,
    Species,
)
from softpotato.techniques import CyclicVoltammetry, simulate

# Constants
D = 1e-9  # m^2/s
cR_bulk = 1  # mol/m^3 (1 mM)
cO_bulk = 0  # mol/m^3
area = 1e-4  # m^2 (1 cm^2)
E0 = 0  # V
n_rxn = 1

# Species definition
ox = Species(
    "Ox",
    diffusivity=D,
    bulk_concentration=cO_bulk,
    standard_potential=E0,
    n_electrons=n_rxn,
)
red = Species(
    "Red",
    diffusivity=D,
    bulk_concentration=cR_bulk,
    standard_potential=E0,
    n_electrons=-n_rxn,
)

# Mechanism definition (E)
mech = Mechanism(
    species=[ox, red],
    electrochemical_steps=[
        ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=E0, n=n_rxn))
    ],
    name="E",
)

# Set geometry
geom = PlanarGeometry(area=area)

# Define CV
cv = CyclicVoltammetry(
    E_start=-0.4,
    E_vertex1=0.4,
    E_vertex2=-0.4,
    scan_rate=0.1,
    n_cycles=3,
)

# Run simulation
sim = simulate(
    technique=cv,
    mechanism=mech,
    geometry=geom,
)

# Generate plots
fig_summary, axes = sim.plot_summary(t=10)
plt.show()
