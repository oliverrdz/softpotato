# Python Basics for Scientists

Python is one of the most popular programming languages in electrochemistry, materials science, and data analysis due to its readable syntax and rich scientific ecosystem (`numpy`, `scipy`, `matplotlib`, and `softpotato`).

---

## 1. Setting Up a Virtual Environment

Before installing scientific packages, always create an isolated **virtual environment** so project dependencies never conflict with your system Python:

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate it (Linux / macOS)
source .venv/bin/activate

# Install scientific packages and Soft Potato
pip install numpy scipy matplotlib softpotato
```

---

## 2. Variables, Lists, and Functions

In Python, variables do not require explicit type declarations, and functions are defined using the `def` keyword:

```python
# Physical constants
faraday = 96485.332  # C / mol
temperature = 298.15  # K


def thermal_voltage(temp_k: float, n: int = 1) -> float:
    """Compute RT / (nF) in Volts."""
    gas_constant = 8.31446
    return (gas_constant * temp_k) / (n * faraday)


print(f"Thermal voltage at 25 °C: {thermal_voltage(temperature) * 1000:.2f} mV")
```

---

## 3. Vectorized Arrays with NumPy

Instead of writing slow `for` loops over thousands of data points, use **NumPy** arrays (`np.ndarray`) to evaluate mathematical equations across an entire array at once:

```python
import numpy as np

# Generate 100 time points from 0.01 s to 5.0 s
t = np.linspace(0.01, 5.0, 100)

# Vectorized Cottrell equation evaluation
n = 1
F = 96485.332
A = 1e-4  # 1 cm^2 in m^2
D = 1e-9  # m^2 / s
c_bulk = 1.0  # mol / m^3

current = (n * F * A * np.sqrt(D) * c_bulk) / np.sqrt(np.pi * t)
```

---

## 4. Plotting Data with Matplotlib

Use `matplotlib.pyplot` to visualize experimental or simulated curves:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(t, current * 1e6, label="Planar Oxidation")
plt.xlabel("Time / s")
plt.ylabel("Current / µA")
plt.title("Cottrell Chronoamperometry")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
