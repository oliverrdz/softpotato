# Numerical Computing with NumPy

**NumPy** is the foundation of scientific computing in Python. It provides the N-dimensional array object (`numpy.ndarray`) and fast, compiled operations that eliminate the need for slow Python `for` loops.

In this tutorial, you will learn:
1. Creating NumPy arrays for time, spatial coordinates, and potentials.
2. The concept and performance benefit of **vectorization**.
3. Benchmarking vectorized code vs pure Python loops.
4. Handling multidimensional arrays for spatio-temporal concentration profiles.

---

## 1. Creating Arrays

Scientific simulations require continuous coordinate grids. NumPy provides convenient constructors:

```python
import numpy as np

# 1. Linearly spaced points (start, stop, number_of_points)
t = np.linspace(0.01, 5.0, 100)  # 100 time points from 0.01 s to 5.0 s
x = np.linspace(0.0, 1e-4, 200)  # Spatial domain: 200 points up to 100 µm

# 2. Stepped sequences (start, stop, step_size)
potentials = np.arange(-0.5, 0.51, 0.05)  # From -0.5 V to +0.5 V with 50 mV steps

# 3. Zeros and ones (essential for initial conditions and buffers)
c_profile = np.zeros(200)  # Pre-allocated 1D array of zeros
c_bulk_array = np.full(200, 1.0)  # Array filled with 1.0 mol / m^3
```

---

## 2. Vectorization: Ditching the `for` Loop

Suppose you need to calculate the theoretical Cottrell diffusion current:

$$I(t) = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}}$$

### The Slow Way (Pure Python Loop)
```python
import math

current_loop = []
for ti in t:
    val = (1 * 96485.332 * 1e-4 * math.sqrt(1e-9) * 1.0) / math.sqrt(
        math.pi * ti
    )
    current_loop.append(val)
```

### The Fast Way (NumPy Vectorization)
Instead of operating on one scalar at a time, pass the entire array `t` directly into the expression. NumPy applies the operations in compiled C:

```python
# Constants
n = 1
F = 96485.332
A = 1e-4
D = 1e-9
c_bulk = 1.0

# Vectorized in one clean line:
current_vec = (n * F * A * np.sqrt(D) * c_bulk) / np.sqrt(np.pi * t)
```

---

## 3. Benchmarking the Speedup

Let's measure how much faster vectorized NumPy is compared to a Python `for` loop over 100,000 points:

```python
import time

t_large = np.linspace(0.001, 10.0, 100_000)

# Pure Python Loop
start = time.perf_counter()
res_loop = [
    (n * F * A * math.sqrt(D) * c_bulk) / math.sqrt(math.pi * ti)
    for ti in t_large
]
t_loop = time.perf_counter() - start

# Vectorized NumPy
start = time.perf_counter()
res_np = (n * F * A * np.sqrt(D) * c_bulk) / np.sqrt(np.pi * t_large)
t_np = time.perf_counter() - start

print(f"Python list comprehension: {t_loop * 1000:.2f} ms")
print(f"Vectorized NumPy:         {t_np * 1000:.2f} ms")
print(f"Speedup:                  {t_loop / t_np:.1f}x faster!")
```

Typical output:
```text
Python list comprehension: 32.10 ms
Vectorized NumPy:         0.48 ms
Speedup:                  66.9x faster!
```

---

## 4. Multi-Dimensional Arrays & Concentration Grids

When simulating electrochemical diffusion, the concentration $c(x, t)$ is a function of both space ($x$) and time ($t$). In NumPy, this is naturally stored as a 2D array:

```python
Nx = 100  # 100 spatial nodes
Nt = 50  # 50 time steps

# 2D array: rows = space, columns = time
c_grid = np.zeros((Nx, Nt))

# Setting initial condition: bulk concentration everywhere at t = 0
c_grid[:, 0] = c_bulk

# Slicing and indexing:
surface_over_time = c_grid[0, :]  # Surface concentration (row 0, all columns)
final_spatial_profile = c_grid[:, -1]  # Final time step (all rows, last column)

print("Grid shape (Nx, Nt):", c_grid.shape)
print("Total data points:", c_grid.size)
```

Output:
```text
Grid shape (Nx, Nt): (100, 50)
Total data points: 5000
```

---

## 5. Seamless Compatibility with Soft Potato

All analytical equations in Soft Potato's analytical module accept either scalars or NumPy arrays out of the box:

```python
from softpotato.analytical import cottrell

# Compute for all time points at once
i_analytical = cottrell(
    time=t,
    c_bulk=c_bulk,
    d=D,
    area=A,
    n=n,
)
print(f"Calculated {len(i_analytical)} current points instantly.")
```

!!! chip "Chip's Tip: Use In-Place Operations for Big Simulations"
    When updating huge grids inside time-stepping loops, use in-place operations like `c[:] = ...` or `np.multiply(..., out=...)` to avoid allocating new memory on every iteration. This keeps your simulations fast and memory-efficient!
