# Getting Started with Google Colab and Soft Potato

## What is a Google Colab Notebook?
Google Colaboratory (or Colab) is a hosted Jupyter notebook service that requires no setup to use and provides free access to computing resources. A **notebook** is an interactive document that combines live executable code, equations, visualizations, and explanatory text in a single place.

### Understanding Cell Types
Colab documents are built out of sequential blocks called **cells**. There are two primary cell types you will use:

- **Code Cells:** These allow you to write and execute Python code directly in your browser. When you run a code cell, the output is displayed immediately beneath it.
- **Markdown Cells:** These contain plain text formatted using Markdown syntax. They are used to document your workflow, add explanations, headings, equations, and links.

---

## Accessing Google Colab
To open a new notebook environment, visit the [Google Colab Website](https://colab.research.google.com/). You can sign in with your Google account and click **New Notebook** to start a blank project.

---

## Installing Soft Potato
Before using Soft Potato, you need to install the package into your Colab session environment. Create a new code cell at the top of your notebook and run the package installation command:
```bash
!pip install softpotato
```
Execute the cell by clicking the play icon on the left or pressing `Shift + Enter`.

---

## Running the Chronoamperometry Transient Script
Once the package is installed, create a new code cell, paste the simulation script provided below, and execute it to generate the chronoamperometry plot.

```python
import numpy as np
import matplotlib.pyplot as plt
from softpotato.analytical import cottrell

# Experimental parameters
n = 1
c_bulk = 1.0  # mol/m^3
D = 7.5e-10  # m^2/s
electrode_radius = 1.5e-3  # m
area = np.pi*(electrode_radius**2)

# Time grid
t = np.linspace(0.01, 5.0, 500)

# Calculate the current using the analytical module
I = cottrell(t=t, n=n, D=D, c_bulk=c_bulk, area=area)

# Plot
plt.figure(figsize=(7, 4.5))
plt.plot(t, I*1e6, label="Cottrell")
plt.xlabel(r"$t$ / s")
plt.ylabel(r"$I$ / $\mu$A")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
```

Congrats! You now know how to run Soft Potato code in Google Colab.