"""Voltammetric potential sweep waveform generators."""

from typing import Optional
import numpy as np
from softpotato.technique.base import Waveform


def lsv(
    E_ini: float,
    E_final: float,
    scan_rate: float,
    dE: Optional[float] = None,
    dt: Optional[float] = None,
) -> Waveform:
    """Generate a Linear Sweep Voltammetry (LSV) potential waveform E(t).

    Calculates a vectorized time array $t$ and corresponding potential profile
    $E(t)$ progressing linearly from an initial potential $E_{\\text{ini}}$ to a
    final potential $E_{\\text{final}}$ at a constant scan rate $\\nu$.

    Parameters
    ----------
    E_ini : float
        Initial potential in Volts (V).
    E_final : float
        Final potential in Volts (V).
    scan_rate : float
        Potential scan rate $\\nu$ in Volts per second (V/s). Must be strictly positive.
    dE : float, optional
        Potential increment step in Volts (V). If provided, step time is derived as
        $dt = \\frac{|dE|}{\\nu}$.
    dt : float, optional
        Time step interval in seconds (s). Cannot be specified simultaneously with `dE`.

    Returns
    -------
    Waveform
        Dataclass container holding 1D NumPy arrays for time `t` (s) and potential `E` (V).

    Raises
    ------
    ValueError
        If `scan_rate` <= 0, both `dE` and `dt` are provided, or step size <= 0.
    """
    if scan_rate <= 0:
        raise ValueError(f"Scan rate must be strictly positive, got {scan_rate}.")

    if dE is not None and dt is not None:
        raise ValueError("Cannot specify both 'dE' and 'dt'. Provide only one.")

    total_delta_E = abs(E_final - E_ini)
    total_time = total_delta_E / scan_rate

    # Handle zero potential window edge case
    if np.isclose(total_delta_E, 0.0):
        return Waveform(t=np.array([0.0]), E=np.array([float(E_ini)]))

    # Resolve step size
    if dE is not None:
        if dE <= 0:
            raise ValueError(f"Potential step dE must be positive, got {dE}.")
        step_dt = dE / scan_rate
    elif dt is not None:
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")
        step_dt = dt
    else:
        # Default sampling resolution: 1 mV equivalent step size
        step_dt = 0.001 / scan_rate

    n_steps = max(int(np.round(total_time / step_dt)), 1)

    # Pure vectorized generation of t and E
    t = np.linspace(0.0, total_time, n_steps + 1)
    E = np.linspace(E_ini, E_final, n_steps + 1)

    return Waveform(t=t, E=E)