import matplotlib.pyplot as plt

from softpotato.physics.kinetics import kinetics

# Assuming local module paths based on the softpotato architecture
from softpotato.technique.sweep import lsv


def plot_bv_kinetics():
    # 1. Define physical parameters
    # The kinetics factory requires k0, alpha, E0, n, and T.
    params = {
        "k0": 1e-3,  # Standard rate constant (m/s)
        "alpha": 0.5,  # Charge transfer coefficient (dimensionless)
        "E0": 0.0,  # Formal potential (V)
        "n": 1.0,  # Number of electrons transferred
        "T": 298.15,  # Temperature (K)
    }

    # 2. Generate the LSV Waveform
    # Sweeping from E_ini = -0.5 V to E_final = 0.5 V at a scan rate of 0.1 V/s
    # strictly using the parameters defined in sweep.py
    waveform = lsv(E_ini=-0.5, E_final=0.5, scan_rate=0.1, dE=0.001)

    # 3. Instantiate the Kinetics Factory
    # Returns a callable that processes potential arrays natively
    bv_model = kinetics("BV", params)

    # 4. Evaluate Rate Constants over the Sweep
    # Extracting the potential array E(t) from the waveform dataclass
    kf, kb = bv_model(waveform.E)

    # 5. Visualization
    # Exponential relationships are best visualized on a semi-logarithmic scale
    plt.figure(figsize=(8, 5))

    plt.semilogy(
        waveform.E,
        kf,
        label="Forward Rate Constant ($k_f$)",
        color="#1f77b4",
        linewidth=2,
    )
    plt.semilogy(
        waveform.E,
        kb,
        label="Backward Rate Constant ($k_b$)",
        color="#d62728",
        linewidth=2,
    )

    # Mark the formal potential
    plt.axvline(
        params["E0"], color="gray", linestyle="--", label="Formal Potential ($E^0$)"
    )

    plt.title("Butler-Volmer Kinetics over an LSV Sweep")
    plt.xlabel("Applied Potential, $E$ (V)")
    plt.ylabel("Rate Constant, $k$ (m/s)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_bv_kinetics()
