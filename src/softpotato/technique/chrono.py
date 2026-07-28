"""Chronoamperometry potential step waveform generator."""

from typing import Optional
import numpy as np
from softpotato.technique.base import Waveform


def ca(
    E_step: float,
    t_tot: float,
    dt: float,
    E_ini: Optional[float] = None,
) -> Waveform:
    """Generate a Chronoamperometry (CA) potential step waveform E(t).

    Calculates a vectorized time array $t$ and corresponding potential profile
    $E(t)$ representing a potential jump to $E_{\\text{step}}$ over duration $t_{\\text{tot}}$.

    Parameters
    ----------
    E_step : float
        Target stepped potential in Volts (V).
    t_tot : float
        Total duration of the potential step experiment in seconds (s). Must be > 0.
    dt : float
        Time step interval in seconds (s). Must be > 0 and <= t_tot.
    E_ini : float, optional
        Initial potential in Volts (V) prior to the step at $t = 0$.
        If None, $E(0) = E_{\\text{step}}$.

    Returns
    -------
    Waveform
        Dataclass container holding 1D NumPy arrays for time `t` (s) and potential `E` (V).

    Raises
    ------
    ValueError
        If `t_tot` <= 0, `dt` <= 0, or `dt` > `t_tot`.
    """
    if t_tot <= 0:
        raise ValueError(f"Total duration t_tot must be positive, got {t_tot}.")
    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")
    if dt > t_tot:
        raise ValueError(f"Time step dt ({dt}) cannot be greater than total time t_tot ({t_tot}).")

    n_steps = max(int(np.round(t_tot / dt)), 1)
    t = np.linspace(0.0, t_tot, n_steps + 1)
    E = np.full_like(t, fill_value=float(E_step), dtype=float)

    if E_ini is not None:
        E[0] = float(E_ini)

    return Waveform(t=t, E=E)