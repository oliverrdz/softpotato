# Soft Potato 3.0 planning

Ideal UI:

```python
import softpotato as sp

# For an E mechanism where O + e <-> R

# Define species:
spec_R = sp.Species("R", D=1e-5, c_bulk=1e-6)
spec_O = sp.Species("O", D=1e-5, c_bulk=1e-6)

# Define the parameters for the reaction:
parameters = {
    "kinetics": "Butler-Volmer",
    "k0 cm/s": 1e-3,
    "alpha": 0.5
}

# Set the mechanism:
mech_E = sp.Reaction([R, O], parameters)
sp.mechanism(rxn = [mech_E])


# Define geometry
grid = sp.Grid{
    "type": "uniform"
}
geometry = sp.geometry.macrodisc(area, grid=grid)


# Define technique
cv = sp.cyclic_voltammetry(E_initial=-0.5, E_vertex1=0.5, scan_rate=0.1, n_sweeps=2, dE=0.01)


# Simulate
sim = sp.Simulate(mechanism=mech_E, geometry=electrode, technique=cv, method="EFD")

# Extract results
t = sim.t
i = sim.i
x = sim.x
c_O = sim.c["O"]
c_R = sim.c["R"]
```

The following sub-modules would need to be implemented:
* Species. Ability to define any species, only physical parameters such as the diffusion coefficient or the concentration are added here.
* Reaction. This module would connect the species and define the mechanism with its kinetics. The user should be able to set any mechanism such as E, EC, CE, ECE, EE, etc. Here, the kinetics can also be selected: Butler-Volmer, Nernst, Tafel, etc.
* Geometry. This would have sub-modules, for example macrodisc, microdisc, sphere, microdisc, RDE, thin_layer, etc. The grid is also set here, uniform, expanding.
* Technique. This would return the time and potential arrays for potentiostatic simulations and the time and current arrays for galvanostatic ones.
* Simulate. This is the solver, it would recieve everything that has been defined before and the solver method to use: explicit finite differences (EFD), a wrapper to the scipy.solve_ivp or any other third party solver.