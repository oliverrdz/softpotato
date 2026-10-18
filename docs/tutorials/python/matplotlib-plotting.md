# Data Visualization with Matplotlib

Scientific visualization turns raw numerical simulation arrays into clear physical insights. **Matplotlib** is the standard plotting library in Python.

In this tutorial, you will learn:
1. The object-oriented `fig, ax = plt.subplots()` paradigm.
2. Styling electrochemical plots following IUPAC conventions.
3. Creating multi-panel figures with shared axes.
4. Exporting publication-ready vector figures (`.svg` and `.pdf`).

---

## 1. Object-Oriented Plotting Basics

While `plt.plot(...)` works for quick sketches, the **object-oriented API** (`fig, ax = plt.subplots()`) gives you full control over every element in your figure:

```python
import matplotlib.pyplot as plt
import numpy as np
from softpotato.analytical import cottrell

# Generate sample data
t = np.linspace(0.01, 5.0, 200)
i_transient = cottrell(
    time=t,
    c_bulk=1.0,
    d=1e-9,
    area=1e-4,
    n=1,
)

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

# Plot current in microamperes
ax.plot(
    t,
    i_transient * 1e6,
    color="#FD8031",
    linewidth=2,
    label="Planar Diffusion",
)

# Labels and title
ax.set_xlabel("Time $t$ / s", fontsize=11)
ax.set_ylabel("Current $I$ / µA", fontsize=11)
ax.set_title("Cottrell Chronoamperometry Transient", fontsize=12, pad=10)

# Grid and legend
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="none")

# Clean layout
fig.tight_layout()
plt.show()
```

---

## 2. Formatting Cyclic Voltammograms

Electrochemical voltammograms require specific conventions:

```python
# Simulated CV data
potential = np.linspace(-0.2, 0.6, 200)
# Example reversible shape approximation
current_anodic = 15.0 / (1.0 + np.exp(-(potential - 0.23) / 0.03))
current_cathodic = -15.0 / (1.0 + np.exp((potential - 0.17) / 0.03))
cv_current = (current_anodic + current_cathodic) * 1e-6

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(potential, cv_current * 1e6, color="#344050", lw=2, label="100 mV/s")

# Zero lines
ax.axhline(0, color="gray", lw=0.8, linestyle=":")
ax.axvline(0, color="gray", lw=0.8, linestyle=":")

# IUPAC Convention: Anodic > 0, Cathodic < 0
ax.set_xlabel("Potential $E$ vs Reference / V")
ax.set_ylabel("Current $I$ / µA")
ax.set_title("Reversible Cyclic Voltammogram")

ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

---

## 3. Multi-Panel Figures (Subplots)

Often you need to show concentration profiles alongside the current response:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel 1: Current vs Time
ax1.plot(t, i_transient * 1e6, color="#FD8031", lw=2)
ax1.set_xlabel("Time $t$ / s")
ax1.set_ylabel("Current $I$ / µA")
ax1.set_title("(a) Chronoamperometric Transient")
ax1.grid(True, alpha=0.3)

# Panel 2: Cottrell Plot (I vs 1 / sqrt(t))
inv_sqrt_t = 1.0 / np.sqrt(t)
ax2.plot(inv_sqrt_t, i_transient * 1e6, "o-", color="#83BEB2", markersize=3)
ax2.set_xlabel("Inverse Square Root Time $t^{-1/2}$ / s$^{-1/2}$")
ax2.set_ylabel("Current $I$ / µA")
ax2.set_title("(b) Linear Cottrell Diagnostic")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
```

---

## 4. Exporting Publication-Ready Figures

When preparing figures for research papers, thesis chapters, or presentations, never export low-resolution blurry PNGs. Use vector graphics or high-DPI raster images:

```python
# 1. Vector PDF (best for LaTeX / Overleaf documents)
fig.savefig("cottrell_analysis.pdf", bbox_inches="tight")

# 2. Vector SVG (best for web pages, presentations, and Inkscape editing)
fig.savefig("cottrell_analysis.svg", bbox_inches="tight")

# 3. High-resolution PNG (300 DPI for Word / PowerPoint)
fig.savefig("cottrell_analysis.png", dpi=300, bbox_inches="tight")
```

!!! chip "Chip's Tip: Use `bbox_inches='tight'`"
    Always pass `bbox_inches='tight'` when saving figures! This automatically removes excess whitespace and prevents axis labels from being accidentally clipped off outside the image canvas.
