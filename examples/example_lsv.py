import matplotlib.pyplot as plt

from softpotato.technique.sweep import lsv

# 1. Generate the LSV potential waveform
# Sweeps from -0.5 V to +0.5 V at a scan rate of 0.1 V/s with 2 mV steps
waveform = lsv(
    E_ini=-0.5,  # Initial potential in Volts (V)
    E_final=0.5,  # Final potential in Volts (V)
    scan_rate=0.1,  # Potential scan rate in Volts/second (V/s)
    dE=0.002,  # Potential step resolution in Volts (V)
)

# 2. Extract time (t) and potential (E) arrays
time_array = waveform.t
potential_array = waveform.E

# 3. Plot the potential signal E(t) versus time t
plt.figure()
plt.plot(
    time_array, potential_array, color="#1f77b4", linewidth=2, label="LSV Waveform"
)

plt.title(
    "Linear Sweep Voltammetry (LSV) Applied Potential Profile",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Time, $t$ (s)", fontsize=16)
plt.ylabel("Applied Potential, $E$ (V)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()

# Display the plot
plt.show()
