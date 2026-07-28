import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class SimulationResult:
    """Container storing simulation time-series, spatial profiles, and export/plotting utilities."""

    t: np.ndarray
    E: np.ndarray
    i: np.ndarray
    x: np.ndarray
    concentrations: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts time-series parameters (t, E, i) into a Pandas DataFrame."""
        return pd.DataFrame(
            {
                "time": self.t,
                "potential": self.E,
                "current": self.i,
            }
        )

    def to_csv(self, filepath: str | Path) -> None:
        """Exports time-series output directly to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def to_json(self, filepath: str | Path) -> None:
        """Saves metadata, operational options, and grid parameters to JSON format."""
        data = {
            "metadata": self.metadata,
            "n_time_steps": len(self.t),
            "n_spatial_nodes": len(self.x),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def to_npz(self, filepath: str | Path) -> None:
        """Saves raw time, potential, current, spatial grid, and concentration matrices into a compressed NumPy file."""
        save_dict = {
            "t": self.t,
            "E": self.E,
            "i": self.i,
            "x": self.x,
        }
        for species_name, c_matrix in self.concentrations.items():
            save_dict[f"c_{species_name}"] = c_matrix
        np.savez(filepath, **save_dict)

    def plot_cv(self, ax: Any | None = None, **kwargs: Any) -> Any | None:
        """Plots current i vs. potential E (Cyclic Voltammogram or linear sweep)."""
        if plt is None:
            raise ImportError(
                "Matplotlib is required for plotting helpers. Install it via pip install matplotlib."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        ax.plot(self.E, self.i, **kwargs)
        ax.set_xlabel("Applied Potential / V")
        ax.set_ylabel("Current / A")
        ax.set_title("Voltammogram")
        ax.grid(True, linestyle="--", alpha=0.6)

        return ax

    def plot_profiles(
        self, time_index: int = -1, ax: Any | None = None, **kwargs: Any
    ) -> Any | None:
        """Plots spatial concentration profiles c(x) across species at a specific time step."""
        if plt is None:
            raise ImportError(
                "Matplotlib is required for plotting helpers. Install it via pip install matplotlib."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        for species_name, c_matrix in self.concentrations.items():
            if c_matrix.ndim == 2:
                profile = c_matrix[:, time_index]
            else:
                profile = c_matrix
            ax.plot(self.x, profile, label=f"Species {species_name}", **kwargs)

        ax.set_xlabel("Distance from Electrode ($x$) / m")
        ax.set_ylabel("Concentration / mol m⁻³")
        ax.set_title(f"Concentration Profiles at Time Index {time_index}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        return ax
