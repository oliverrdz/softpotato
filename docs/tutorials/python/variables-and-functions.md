# Python Fundamentals: Variables & Functions

Writing clean, maintainable scientific code requires mastering Python's core data types, type annotations, and structured function design.

In this tutorial, you will learn:
1. Core Python data types used in electrochemical calculations.
2. Managing physical constants and unit conversions.
3. Defining functions with modern type annotations and docstrings.
4. Structuring reusable calculation scripts.

---

## 1. Variables and Data Types

In Python, variables are dynamically typed, but assigning clear variable names with explicit physical units is good scientific practice:

```python
# Integer: number of electrons transferred
n_electrons = 1

# Float: continuous physical quantities
temperature = 298.15  # Temperature in Kelvin (25 °C)
electrode_area = 1.0e-4  # Area in m^2 (1 cm^2)
diffusivity = 1.0e-9  # Diffusion coefficient in m^2 / s
bulk_concentration = 1.0  # Concentration in mol / m^3 (1 mM)

# String: electrode and material descriptions
working_electrode = "Glassy Carbon (3 mm diameter)"
electrolyte = "0.1 M KCl with 1 mM Ferri/Ferrocyanide"

# Boolean: simulation control flags
is_reversible = True
include_migration = False
```

---

## 2. Lists, Tuples, and Dictionaries

To store and structure collections of experimental measurements:

```python
# Tuples: immutable ordered records (e.g. initial and vertex potentials)
potential_window = (-0.2, 0.6)  # (E_start, E_vertex) in Volts

# Lists: mutable sequences (e.g. scan rates to evaluate)
scan_rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]  # V / s

# Dictionaries: key-value configuration dictionaries for simulations
cell_config = {
    "temperature_k": 298.15,
    "electrode_area_m2": 0.0707e-4,  # 3 mm disc
    "diffusivity_ox": 0.76e-9,  # m^2 / s
    "bulk_ox_mol_m3": 2.0,  # mol / m^3
}

print(
    f"Simulating on {working_electrode} with D = {cell_config['diffusivity_ox']} m²/s"
)
```

---

## 3. Defining Scientific Functions

In scientific computing, functions should do one thing well, document their input parameters and return types, and provide unit conventions in docstrings.

```python
# Physical constants (CODATA values)
FARADAY_CONSTANT = 96485.332  # C / mol
GAS_CONSTANT = 8.3144626  # J / (mol * K)


def thermal_voltage(temp_k: float, n: int = 1) -> float:
    """Calculate the electrochemical thermal voltage (RT / nF) in Volts.

    Parameters
    ----------
    temp_k : float
        Absolute temperature in Kelvin (K).
    n : int, optional
        Number of electrons transferred per redox event, by default 1.

    Returns
    -------
    float
        Thermal voltage in Volts (V).
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be strictly positive in Kelvin.")
    if n <= 0:
        raise ValueError("Number of electrons n must be a positive integer.")

    return (GAS_CONSTANT * temp_k) / (n * FARADAY_CONSTANT)


# Compute thermal voltage at room temperature (25 °C)
v_t = thermal_voltage(temp_k=298.15, n=1)
print(f"Thermal voltage at 25 °C: {v_t * 1000:.2f} mV")
```

Output:
```text
Thermal voltage at 25 °C: 25.69 mV
```

---

## 4. Calculating Nernstian Equilibrium

Let's build a practical function that calculates the reversible Nernst potential for a redox pair $\text{Ox} + n e^- \rightleftharpoons \text{Red}$:

$$E_{\text{eq}} = E^{0'} + \frac{R T}{n F} \ln\left( \frac{c_{\text{Ox}}}{c_{\text{Red}}} \right)$$

```python
import math


def nernst_potential(
    e_formal: float,
    c_ox: float,
    c_red: float,
    temp_k: float = 298.15,
    n: int = 1,
) -> float:
    """Compute the equilibrium potential via the Nernst equation.

    Parameters
    ----------
    e_formal : float
        Formal reduction potential in Volts (V).
    c_ox : float
        Concentration of oxidized species (mol / m^3).
    c_red : float
        Concentration of reduced species (mol / m^3).
    temp_k : float, optional
        Temperature in Kelvin, defaults to 298.15 K.
    n : int, optional
        Electron stoichiometry, defaults to 1.

    Returns
    -------
    float
        Equilibrium potential in Volts (V).
    """
    if c_ox <= 0 or c_red <= 0:
        raise ValueError("Concentrations must be strictly positive.")

    rt_over_nf = thermal_voltage(temp_k=temp_k, n=n)
    return e_formal + rt_over_nf * math.log(c_ox / c_red)


e_eq = nernst_potential(e_formal=0.220, c_ox=1.0, c_red=0.1)
print(f"Equilibrium potential: {e_eq:.4f} V")
```

Output:
```text
Equilibrium potential: 0.2792 V
```

!!! chip "Chip's Tip: Use Type Annotations!"
    Adding type hints (`temp_k: float`, `-> float`) lets modern IDEs (like VS Code, PyCharm, or Antigravity) catch bugs before you even run your code. Soft Potato is fully typed and checked with `mypy --strict`!
