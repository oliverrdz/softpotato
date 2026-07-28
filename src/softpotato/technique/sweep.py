"""Voltammetric potential sweep waveform generators."""

import numpy as np

from softpotato.technique.base import Waveform


def lsv(
    E_ini: float,
    E_final: float,
    scan_rate: float,
    dE: float | None = None,
    dt: float | None = None,
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


def cv(
    E_ini: float,
    E_v1: float,
    E_v2: float,
    scan_rate: float,
    dE: float,
    n_cycles: int = 1,
) -> Waveform:
    """
    Generates a Cyclic Voltammetry (CV) potential waveform.

    Parameters:
        - E_ini: Initial potential (V)
        - E_v1: First switching potential (V)
        - E_v2: Second switching potential (V)
        - scan_rate: Potential scan rate (V/s)
        - dE: Potential increment step size (V)
        - n_cycles: Number of potential cycles

    Returns:
        - Waveform object containing time vector t and potential vector E.
    """
    if scan_rate <= 0:
        raise ValueError("Scan rate must be strictly positive.")
    if dE <= 0:
        raise ValueError("Potential increment dE must be strictly positive.")
    if n_cycles < 1:
        raise ValueError("Cycle count must be at least 1.")

    segments = []

    # Segment 1: Initial potential to first switching potential E_v1
    n_pts_1 = int(np.round(abs(E_v1 - E_ini) / dE)) + 1
    if n_pts_1 > 1:
        seg1 = np.linspace(E_ini, E_v1, n_pts_1)
    else:
        seg1 = np.array([E_ini])
    segments.append(seg1)

    # Segment 2: From E_v1 to second switching potential E_v2
    n_pts_2 = int(np.round(abs(E_v2 - E_v1) / dE)) + 1
    if n_pts_2 > 1:
        seg2 = np.linspace(E_v1, E_v2, n_pts_2)[
            1:
        ]  # Exclude start to avoid duplication
    else:
        seg2 = np.array([])

    # Segment 3: From E_v2 back to E_v1 (completing one full cycle)
    n_pts_3 = int(np.round(abs(E_v1 - E_v2) / dE)) + 1
    if n_pts_3 > 1:
        seg3 = np.linspace(E_v2, E_v1, n_pts_3)[1:]
    else:
        seg3 = np.array([])

    # Construct cycles
    current_cycle_segment = (
        np.concatenate([seg2, seg3])
        if (len(seg2) > 0 or len(seg3) > 0)
        else np.array([])
    )

    potential_list = [segments[0]]
    for _ in range(n_cycles):
        if len(current_cycle_segment) > 0:
            potential_list.append(current_cycle_segment)

    E_array = np.concatenate(potential_list)

    # Calculate time vector based on scan rate and cumulative potential distance
    # dt = dE / scan_rate for uniform steps
    dt = dE / scan_rate
    t_array = np.arange(len(E_array)) * dt

    return Waveform(t=t_array, E=E_array)
