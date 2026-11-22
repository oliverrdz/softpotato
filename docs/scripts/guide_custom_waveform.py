"""User Guide: Custom Waveforms & Composable Segments.

Demonstrates:
1. Chaining ConstantSegment, LinearRampSegment, SinusoidalSegment, and CallableSegment
2. Simulating a CustomWaveform with simulate()
3. Multi-panel visualization with plot_summary()
"""

import matplotlib.pyplot as plt
import numpy as np

from softpotato.geometry import PlanarGeometry
from softpotato.reactions import ButlerVolmer, Species
from softpotato.techniques import (
    CallableSegment,
    ConstantSegment,
    CustomWaveform,
    LinearRampSegment,
    SinusoidalSegment,
    plot_summary,
    simulate,
)

# ==============================================================================
# 1. Physical System (SI units: m^2/s, mol/m^3 = mM, m^2, m/s)
# ==============================================================================
E0 = 0.0  # Formal potential [V]
D = 1.0e-9  # Diffusion coefficient [m^2/s]
c_bulk = 1.0  # Bulk concentration of Ox [mol/m^3] (= 1.0 mM)
k0 = 1.0e-3  # Heterogeneous rate constant [m/s]
alpha = 0.5  # Transfer coefficient [-]
area = 7.07e-6  # Electrode area [m^2]

ox = Species(
    "Ox", diffusivity=D, bulk_concentration=c_bulk, standard_potential=E0, n_electrons=1
)
red = Species("Red", diffusivity=D, bulk_concentration=0.0)
kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0, n=1)
geometry = PlanarGeometry(area)


# ==============================================================================
# 2. Compose Custom Waveform from Primitive Segments
# ==============================================================================
# Example: Conditioning hold -> Linear ramp -> AC sine wave -> Exponential decay
def exp_relaxation(t_local: np.ndarray | float) -> np.ndarray | float:
    """Relax potential exponentially from -0.2 V back toward +0.2 V."""
    return 0.2 - 0.4 * np.exp(-2.0 * np.asarray(t_local))


waveform = CustomWaveform(
    segments=[
        ConstantSegment(value=0.3, duration=0.5),
        LinearRampSegment(start=0.3, end=-0.2, duration=2.5),
        SinusoidalSegment(offset=-0.2, amplitude=0.05, frequency=2.0, duration=1.5),
        CallableSegment(func=exp_relaxation, duration=1.5),
    ],
    mode="potentiostatic",
)

print(f"CustomWaveform total duration: {waveform.duration:.2f} s")
print(f"Segment boundaries: {waveform.segment_boundaries}")
print(f"Interior event times: {waveform.event_times}")

# ==============================================================================
# 3. Simulate & Summarize
# ==============================================================================
result = simulate(waveform, [ox, red], geometry=geometry, kinetics=kinetics)
print(
    f"Simulated {len(result.time)} points; "
    f"current range: [{result.current.min() * 1e6:.2f}, {result.current.max() * 1e6:.2f}] uA"
)

fig, axes = plot_summary(result)
fig.suptitle("Custom Multi-Segment Waveform Simulation", y=1.02)

plt.show()
