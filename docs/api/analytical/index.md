# Analytical Module (`softpotato.analytical`)

The `softpotato.analytical` module provides direct, ready-to-use analytical equations and diagnostic formulas for experimental electrochemistry. Instead of juggling unit conversions, manual spreadsheet calculations, or scratchpad arithmetic, this module gives you standard theoretical predictions in a single function call to compare directly against bench data.

Whether you are validating an active electrode area, extracting a diffusion coefficient, or checking whether an experimental voltammogram behaves reversibly, the module serves as your instant quantitative reference.

## Core Applications for Experimental Work

* Transient Chronoamperometry (Cottrell Analysis): Calculate theoretical current for planar and spherical geometries. 
* Voltammetric Peak Predictions (Randles–Ševčík): Compute expected peak currents ($i_p$) across sweep rates ($\nu$) for reversible, quasi-reversible, and irreversible systems. Quickly determine whether your experimental peak currents scale with $\nu^{1/2}$ and verify your observed peak potentials against classic theory.
* Convective and Steady-State Electrochemistry: Evaluate mass-transport limits using standard hydrodynamic equations (such as the Levich and Koutecký–Levich equations for Rotating Disk Electrodes) or steady-state limits for ultramicroelectrodes (UMEs) to isolate kinetic parameters from mass-transfer limits.
* Kinetic and Overpotential Benchmarks: Obtain standard Tafel approximations and exchange current relations to benchmark polarization curves and verify overpotential trends directly against Butler–Volmer theory.
* Ground-Truth Simulation Benchmarks: Provide an exact, closed-form baseline to verify numerical grid convergence and solver accuracy before setting up complex, multi-species mechanisms.

## Example: calculate the Cottrell current
To import a specific equation, follow the rule `softpotato.analytical.submodule`, replacing `submodule` with the corresponding submodule for that equation. For example, the code below shows how to import the Cottrell equation from `softpotato.analytical.step` and use it to calculate the current at 10 s.

```python
from softpotato.analytical.step import cottrell

A = 1  # cm2, area
n = 1  # number of electrons
c_bulk = 1e-6  # mol/cm3, bulk concentration, 1 mM
D = 1e-5  # cm2/s, diffusion coefficient
t = 10  # s, time to calculate the current

i_cottrell = cottrell(t, n=n, D=D, c_bulk=c_bulk, area=A)  # current at 10 s
print(i_cottrell)
```

## Submodules

- [Transient Step Techniques (`softpotato.analytical.step`)](step.md)
- [Hydrodynamics & Convection Equations (`softpotato.analytical.hydrodynamics`)](hydrodynamics.md)
- [Thermodynamics & Interfacial Kinetics (`softpotato.analytical.kinetics`)](kinetics.md)
- [Voltammetry & Kinetic Diagnostics (`softpotato.analytical.voltammetry`)](voltammetry.md)
- [Microelectrode Geometries (`softpotato.analytical.microelectrodes`)](microelectrodes.md)
- [Scanning Electrochemical Microscopy (`softpotato.analytical.secm`)](secm.md)
- [Parameter Fitting & Experimental Data Analysis (`softpotato.analytical.fitting`)](fitting.md)
- [Simulation Benchmarking Suite (`softpotato.analytical.benchmark`)](benchmark.md)