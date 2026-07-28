from collections.abc import Callable

import numpy as np

# Physical Constants
F = 96485.3321  # Faraday constant (C/mol)
R = 8.3144626  # Ideal gas constant (J/(K*mol))


def butler_volmer(
    e_array: np.ndarray, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the Butler-Volmer heterogeneous kinetic rate constants.

    Parameters
    ----------
    e_array : np.ndarray
        1D array of applied electrode potentials (V).
    params : Dict[str, float]
        Dictionary containing kinetic and thermodynamic parameters:
        - 'k0': Standard heterogeneous rate constant (m/s). Default is 1e-3.
        - 'alpha': Charge transfer coefficient (dimensionless). Default is 0.5.
        - 'E0': Formal potential (V). Default is 0.0.
        - 'n': Number of electrons transferred. Default is 1.0.
        - 'T': Temperature (K). Default is 298.15.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 1D arrays:
        - kf: Forward (reduction) rate constants.
        - kb: Backward (oxidation) rate constants.
    """
    # Extract parameters with physical defaults
    k0 = params.get("k0", 1e-3)
    alpha = params.get("alpha", 0.5)
    e0 = params.get("E0", 0.0)
    n = params.get("n", 1.0)
    temp = params.get("T", 298.15)

    # Thermodynamic grouping term
    f_term = (n * F) / (R * temp)

    # Overpotential
    theta = e_array - e0

    # Vectorized calculation of forward and backward rate constants
    kf = k0 * np.exp(-alpha * f_term * theta)
    kb = k0 * np.exp((1 - alpha) * f_term * theta)

    return kf, kb


import numpy as np


def tafel(
    e_array: np.ndarray, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates heterogeneous kinetic rate constants using the empirical Tafel equation.

    Parameters
    ----------
    e_array : np.ndarray
        1D array of applied electrode potentials (V).
    params : Dict[str, float]
        Dictionary containing kinetic parameters:
        - 'k0': Standard heterogeneous rate constant (m/s). Default is 1e-3.
        - 'E0': Formal potential (V). Default is 0.0.
        - 'bc': Cathodic Tafel slope (V/dec). Default is 0.120.
        - 'ba': Anodic Tafel slope (V/dec). Default is 0.120.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 1D arrays:
        - kf: Forward (reduction) rate constants.
        - kb: Backward (oxidation) rate constants.
    """
    # Extract parameters with standard physical defaults
    k0 = params.get("k0", 1e-3)
    e0 = params.get("E0", 0.0)
    bc = params.get("bc", 0.120)  # Default ~120 mV/dec
    ba = params.get("ba", 0.120)

    # Overpotential
    theta = e_array - e0

    # Vectorized calculation using base-10 exponentials
    kf = k0 * np.power(10.0, -theta / bc)
    kb = k0 * np.power(10.0, theta / ba)

    return kf, kb


def nernst(e_array: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """
    Evaluates the Nernstian equilibrium surface concentration ratio.

    Calculates the required ratio of oxidized to reduced species at the
    electrode interface (c_O / c_R) for a fully reversible electron transfer.

    Parameters
    ----------
    e_array : np.ndarray
        1D array of applied electrode potentials (V).
    params : Dict[str, float]
        Dictionary containing thermodynamic parameters:
        - 'E0': Formal potential (V). Default is 0.0.
        - 'n': Number of electrons transferred. Default is 1.0.
        - 'T': Temperature (K). Default is 298.15.

    Returns
    -------
    np.ndarray
        1D array of the surface concentration ratio theta = c_O(0,t) / c_R(0,t).
    """
    # Extract parameters with physical defaults
    e0 = params.get("E0", 0.0)
    n = params.get("n", 1.0)
    temp = params.get("T", 298.15)

    # Thermodynamic grouping term: nF / RT
    f_term = (n * F) / (R * temp)

    # Calculate theta = exp[ (nF/RT) * (E - E0) ]
    theta = np.exp(f_term * (e_array - e0))

    return theta


def kinetics(
    model: str, params: dict[str, float]
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Factory function to instantiate kinetic rate evaluators.

    Parameters
    ----------
    model : str
        String identifier for the kinetic model (e.g., "bv", "tafel").
    params : Dict[str, float]
        Dictionary of physical parameters required by the specified model.

    Returns
    -------
    Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
        A function that takes an array of potentials and returns (kf, kb).

    Raises
    ------
    ValueError
        If the specified model identifier is not supported.
    """
    model_id = model.strip().lower()

    if model_id == "bv":
        # Assuming butler_volmer is already defined above
        return lambda e_array: butler_volmer(e_array, params)
    elif model_id == "tafel":
        return lambda e_array: tafel(e_array, params)
    # Future models (e.g., "MARCUS") can be added here
    else:
        raise ValueError(f"Kinetic model '{model}' is not supported by the factory.")
