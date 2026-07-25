import matplotlib.pyplot as plt

from softpotato.discretizers import FDM1DDiscretizer
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import ButlerVolmerBC, TwoSpeciesModel
from softpotato.solvers import ODESolver
from softpotato.techniques import CyclicVoltammetry

# 1. Define the CV technique (scan rate = 0.1 V/s)
technique = CyclicVoltammetry(E_start=-0.5, E_vertex1=0.5, scan_rate=0.1, n_cycles=1)

# 2. Define chemical species transport parameters (D in m^2/s, concentrations in mol/m^3)
model = TwoSpeciesModel(D_R=1e-9, D_O=1e-9, C_R_bulk=1.0, C_O_bulk=0.0)

# 3. Generate 1D spatial domain (1 mm thickness, 201 grid points)
mesh = Uniform1DMesh(x_min=0.0, x_max=1e-3, num_nodes=201)

# 4. Apply Butler-Volmer kinetic surface boundary condition
bc = ButlerVolmerBC(
    technique=technique,
    E0=0.0,  # Formal potential (V)
    k0=1e-5,  # Standard rate constant (m/s)
    alpha=0.5,  # Charge transfer coefficient
    n=1,  # Transferred electrons
    A=1e-4,  # Electrode area (m^2)
)

# 5. Assemble discretizer and ODE solver engine
discretizer = FDM1DDiscretizer()
solver = ODESolver(mesh=mesh, model=model, discretizer=discretizer, bc=bc)

# 6. Execute time-stepping integration from initial state vector
y0 = model.get_initial_state_vector(mesh.x)
result = solver.solve(t_span=technique.t_span, y0=y0)

# 7. Plot and save voltammogram (current converted to µA)
plt.plot(result.potential, result.current * 1e6)
plt.xlabel("Potential E (V)")
plt.ylabel("Faradaic Current i (µA)")
plt.title("Cyclic Voltammetry with Butler-Volmer Kinetics")
plt.grid(True)
plt.show()
