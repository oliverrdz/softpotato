# Soft Potato 🥔⚡

**Soft Potato** is an open-source, highly modular Python library for electrochemical simulations. 

Its core architectural principle is **strict physics-numerics decoupling**: physical transport/kinetics models, spatial meshes, numerical discretization schemes, experimental excitation techniques, and time solvers are completely independent components that assemble seamlessly.

---

## 🏛️ Architecture Philosophy

Soft Potato decouples simulations into three fundamental pillars:

* **Physics & Thermodynamics (`softpotato.physics`):** Species properties, diffusion coefficients, formal potentials, and surface kinetics.
* **Experimental Techniques (`softpotato.techniques`):** Excitation signals E(t) such as Cyclic Voltammetry (CV), Chronoamperometry (CA), or Square Wave Voltammetry (SWV).
* **Numerical Mechanics (`softpotato.mesh`, `softpotato.discretizers`, `softpotato.solvers`):** Spatial grids, finite difference/volume matrix stencils, and ODE time integration engines.

---

## 📁 Repository & File Structure
```
softpotato/
├── GEMINI.md                      # AI Project memory & architecture rules
├── LICENSE                        # Open-source MIT License
├── pyproject.toml                 # Build configuration & dependencies (PEP 517/621)
├── README.md                      # Project documentation
├── examples/                      # Runnable simulation examples
│   └── 01_reversible_cv.py        # 1D Reversible Cyclic Voltammetry example
├── src/
│   └── softpotato/
│       ├── __init__.py            # Top-level ABC exports and version info
│       ├── core/
│       │   ├── __init__.py
│       │   └── abcs.py            # Base contracts (BaseMesh, BaseModel, BaseTechnique, etc.)
│       ├── mesh/
│       │   ├── __init__.py
│       │   └── uniform_1d.py      # Uniform 1D spatial grid generator
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── species.py         # Chemical species data model & TwoSpeciesModel
│       │   └── nernst_bc.py       # Reversible surface Nernstian kinetics
│       ├── discretizers/
│       │   ├── __init__.py
│       │   └── fdm_1d.py          # 2nd-order 1D Finite Difference discretizer
│       ├── techniques/
│       │   ├── __init__.py
│       │   └── cyclic_voltammetry.py # Multi-cycle CV potential waveform generator
│       └── solvers/
│           ├── __init__.py
│           └── ode_solver.py      # SciPy solve_ivp ODE integration engine
└── tests/                         # Pytest test suite & Randles-Ševčík validation suite
```
---

## 🚀 Quickstart: Running a Cyclic Voltammogram

Below is a complete Python workflow demonstrating how to assemble independent modules to simulate a 1D reversible Cyclic Voltammetry experiment:
```python
# 1. Import required components
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import TwoSpeciesModel, NernstianEquilibriumBC
from softpotato.techniques import CyclicVoltammetry
from softpotato.discretizers import FDM1DDiscretizer
from softpotato.solvers import ODESolver

# 2. Define Spatial Domain (Diffusion Layer, x_max = 1 mm, 201 nodes)
mesh = Uniform1DMesh(x_min=0.0, x_max=1e-3, num_nodes=201)

# 3. Define Transport Physics (Reversible R -> O + e-)
model = TwoSpeciesModel(D_R=1e-9, D_O=1e-9, C_R_bulk=1.0, C_O_bulk=0.0)

# 4. Define Experimental Technique (Multi-cycle CV from -0.2 V to 0.3 V at 100 mV/s)
technique = CyclicVoltammetry(
    E_start=-0.2,
    E_vertex1=0.3,
    scan_rate=0.1,
    n_cycles=2
)

# 5. Couple Technique to Nernstian Surface Boundary Conditions
bc = NernstianEquilibriumBC(
    technique=technique,
    E0=0.0,
    n=1,
    T=298.15,
    A=1e-4
)

# 6. Build Discretizer & Execute Time Integration
discretizer = FDM1DDiscretizer()
solver = ODESolver(mesh=mesh, model=model, discretizer=discretizer, bc=bc)

y0 = model.get_initial_state_vector(mesh.x)
result = solver.solve(t_span=technique.t_span, y0=y0)

# 7. Access Time-Series Results
time_array = result.t          # Time in seconds
potential_array = result.potential  # Applied potential E(t) in Volts
current_array = result.current      # Faradaic response current i(t) in Amperes
```
---

## 🧪 Testing & Validation

All numerical stencils and physical models are validated using pytest. Run `pytest` from the root directory to execute the test suite.

* **`tests/test_reversible_cv.py`:** Validates simulated CV peak current i_p against the analytical Randles-Ševčík equation within **1.5%** relative error.
* **`tests/test_techniques.py`:** Validates potential trajectory bounds and switching vertex timings for single- and multi-cycle sweeps.
* **`tests/test_fdm_1d.py`:** Verifies second-order spatial derivative convergence against analytical trigonometric functions.

---

## 📜 License

Soft Potato is distributed as open-source software under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software in both academic and commercial environments. See the `LICENSE` file in the repository root for the full legal license text.
