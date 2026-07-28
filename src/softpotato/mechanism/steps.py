from dataclasses import dataclass


@dataclass
class ElectronTransferStep:
    """
    Data representation of an elementary electron-transfer (E) step.

    Represents the general half-reaction:
    Ox + n e- <=> Red

    Attributes
    ----------
    ox_species : str
        The identifier for the oxidized species (e.g., 'O').
    red_species : str
        The identifier for the reduced species (e.g., 'R').
    n_electrons : int, optional
        Number of electrons transferred in the step. Default is 1.
    e_formal : float, optional
        The formal potential (E^0) of the reaction in Volts (V). Default is 0.0 V.
    k0 : float, optional
        The standard heterogeneous rate constant in SI units (m/s). Default is 0.01 m/s.
    alpha : float, optional
        The charge transfer coefficient (typically between 0.0 and 1.0). Default is 0.5.
    """

    ox_species: str
    red_species: str
    n_electrons: int = 1
    e_formal: float = 0.0
    k0: float = 0.01  # SI unit: m/s (equivalent to 1.0 cm/s)
    alpha: float = 0.5

    def __post_init__(self):
        """Validates physical parameters upon initialization."""
        if self.n_electrons < 1:
            raise ValueError("Number of electrons (n_electrons) must be at least 1.")
        if self.k0 < 0.0:
            raise ValueError("Standard rate constant (k0) must be non-negative.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                "Charge transfer coefficient (alpha) must be between 0.0 and 1.0."
            )


@dataclass
class ChemicalStep:
    """
    Data representation of an elementary homogeneous chemical (C) step.

    Represents the general reaction:
    Reactants <=> Products

    Attributes
    ----------
    reactants : List[str]
        A list of string identifiers for the reactant species (e.g., ['O', 'Z']).
    products : List[str]
        A list of string identifiers for the product species (e.g., ['P']).
    kf : float, optional
        The forward homogeneous rate constant in SI units (e.g., 1/s for first-order,
        m³/(mol·s) for second-order). Default is 1.0.
    kb : float, optional
        The backward homogeneous rate constant in SI units. Default is 0.0 (irreversible).
    """

    reactants: list[str]
    products: list[str]
    kf: float = 1.0
    kb: float = 0.0

    def __post_init__(self):
        """Validates physical parameters upon initialization."""
        if not self.reactants:
            raise ValueError(
                "A chemical step must have at least one reactant species defined."
            )
        if not self.products:
            raise ValueError(
                "A chemical step must have at least one product species defined."
            )
        if self.kf < 0.0:
            raise ValueError("Forward rate constant (kf) must be non-negative.")
        if self.kb < 0.0:
            raise ValueError("Backward rate constant (kb) must be non-negative.")
