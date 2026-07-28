import matplotlib.pyplot as plt

from softpotato.technique.chrono import ca

# 1. Generate the Chronoamperometry (CA) waveform
# Step from E_ini = 0.0 V to E_step = 0.5 V for 5 seconds with dt = 0.01 s
waveform = ca(
    E_step=0.5,  # Stepped potential in Volts (V)
    t_tot=5.0,  # Total duration in seconds (s)
    dt=0.01,  # Time step interval in seconds (s)
    E_ini=0.0,  # Initial potential in Volts (V)
)

# 2. Extract time (t) and potential (E) arrays
time_array = waveform.t
potential_array = waveform.E

# 3. Plot the potential signal E(t) versus time t
plt.figure()
plt.step(
    time_array,
    potential_array,
    where="post",
    color="#2ca02c",
    linewidth=2,
    label="CA Potential Step",
)

plt.title(
    "Chronoamperometry (CA) Applied Potential Profile", fontsize=12, fontweight="bold"
)
plt.xlabel("Time, $t$ (s)", fontsize=16)
plt.ylabel("Applied Potential, $E$ (V)", fontsize=16)
plt.ylim(-0.1, 0.6)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper right")
plt.tight_layout()

# Display the plot
plt.show()
