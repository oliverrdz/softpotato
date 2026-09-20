import numpy as np
from typing import List, Optional
from softpotato.core.species import Species

class HeterogeneousReaction:
    """
    Defines an electrochemical electron transfer step at the electrode surface
    Follows the convention: Ox + n e- <=> Red
    """
    def __init__(
        self,
        ox: Species,
        red: Species,
        n: int = 1,
        E0: float = 0.0,
        k0: float = 1e4,
        alpha: float = 0.5,
        T: float = 298.15
    ) -> None:
        """
        Args:
            ox: The oxidized Species object
            red: The reduced Species object
            n: Number of electrons transferred (dimensionless)
            E0: Standard reduction potential in V
            k0: Standard heterogeneous rate constant in cm/s
                (Default is fast/reversible: 1e4 cm/s).
            alpha: Transfer coefficient for reduction
            T: Temperature in Kelvin
        """
        self.ox = ox
        self.red = red
        self.n = n
        self.E0 = E0
        self.k0 = k0
        self.alpha = alpha
        
        # Physical constants
        self.F = 96485.3321  # Faraday constant, C/mol
        self.R = 8.3144626   # Gas constant, J/(mol K)
        self.f = (self.F) / (self.R * T) # Dimensionless potential factor (1/V)

    def get_rate_constants(self, E_app: float) -> tuple[float, float]:
        """
        Calculates the potential-dependent forward (kf) and backward (kb) 
        heterogeneous rate constants using Butler-Volmer kinetics
        
        Args:
            E_app: Applied electrode potential in V
            
        Returns:
            Tuple of (kf, kb) in cm/s
        """
        # Butler-Volmer exponential terms
        theta = self.f * (E_app - self.E0)
        
        kf = self.k0 * np.exp(-self.alpha * self.n * theta)
        kb = self.k0 * np.exp((1.0 - self.alpha) * self.n * theta)
        
        return kf, kb


class HomogeneousReaction:
    """
    Defines a chemical reaction occurring in the bulk/diffusion layer
    Handles standard generic kinetics: aA + bB <=> cC + dD
    """
    def __init__(
        self,
        reactants: List[Species],
        products: List[Species],
        kf: float = 0.0,
        kb: float = 0.0,
        stoich_reactants: Optional[List[int]] = None,
        stoich_products: Optional[List[int]] = None
    ) -> None:
        """
        Args:
            reactants: List of reactant Species objects
            products: List of product Species objects
            kf: Forward rate constant. Units depend on order (e.g., 1/s or cm^3/(mol s)).
            kb: Backward rate constant. Units depend on order.
            stoich_reactants: Stoichiometric coefficients for reactants. Defaults to 1 for all.
            stoich_products: Stoichiometric coefficients for products. Defaults to 1 for all.
        """
        self.reactants = reactants
        self.products = products
        self.kf = kf
        self.kb = kb
        
        self.stoich_reactants = stoich_reactants if stoich_reactants else [1] * len(reactants)
        self.stoich_products = stoich_products if stoich_products else [1] * len(products)

    def compute_kinetic_rates(self) -> dict[str, np.ndarray]:
        """
        Computes the vectorized concentration change rates (dC/dt) for the FDM solver
        Relies directly on the `c_profile` attribute of the species objects.
        
        Returns:
            Dictionary mapping species names to their respective dC/dt numpy arrays (mol/cm^3/s)
        """
        # Calculate forward rate (vectorized across the spatial grid)
        rate_f = self.kf
        for species, stoich in zip(self.reactants, self.stoich_reactants):
            if species.c_profile is not None:
                rate_f = rate_f * (species.c_profile ** stoich)
                
        # Calculate backward rate
        rate_b = self.kb
        for species, stoich in zip(self.products, self.stoich_products):
            if species.c_profile is not None:
                rate_b = rate_b * (species.c_profile ** stoich)

        net_rate = rate_f - rate_b

        # Map the net rate to each species based on stoichiometry
        dc_dt = {}
        
        for species, stoich in zip(self.reactants, self.stoich_reactants):
            dc_dt[species.name] = -stoich * net_rate
            
        for species, stoich in zip(self.products, self.stoich_products):
            # Accumulate in case a species is both a reactant and product (e.g., auto-catalysis)
            if species.name in dc_dt:
                dc_dt[species.name] += stoich * net_rate
            else:
                dc_dt[species.name] = stoich * net_rate
                
        return dc_dt


class Mechanism:
    """
    Container class to pass species and reactions to the FDM solver seamlessly
    """
    def __init__(self, species: List[Species], e_reactions: List[HeterogeneousReaction], c_reactions: List[HomogeneousReaction] = None) -> None:
        self.species = {s.name: s for s in species}
        self.e_reactions = e_reactions
        self.c_reactions = c_reactions if c_reactions else []
