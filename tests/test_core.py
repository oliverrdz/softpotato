import pytest
import numpy as np
from softpotato.core.species import Species
from softpotato.core.reactions import HeterogeneousReaction, HomogeneousReaction, Mechanism

def test_species_initialization() -> None:
    """Verifies species parameters and profile allocation."""
    # D in cm^2/s, C* in mol/cm^3
    o = Species(name="O", D=1e-5, c_bulk=1e-6, charge=1)
    assert o.name == "O"
    assert o.D == 1e-5
    assert o.c_profile is None

    o.initialize_profile(n_nodes=100)
    assert o.c_profile is not None
    assert len(o.c_profile) == 100
    assert np.all(o.c_profile == 1e-6)

def test_heterogeneous_reaction_kinetics() -> None:
    """Tests Butler-Volmer rate constant calculations."""
    o = Species("O", 1e-5, 1e-6)
    r = Species("R", 1e-5, 0.0)
    
    # E0 = 0.0 V, k0 = 1.0 cm/s, alpha = 0.5, T = 298.15 K
    rxn = HeterogeneousReaction(ox=o, red=r, n=1, E0=0.0, k0=1.0, alpha=0.5, T=298.15)
    
    # At standard potential (E_app = E0 = 0.0 V), kf should equal kb should equal k0
    kf, kb = rxn.get_rate_constants(E_app=0.0)
    np.testing.assert_almost_equal(kf, 1.0, decimal=5)
    np.testing.assert_almost_equal(kb, 1.0, decimal=5)
    
    # Apply cathodic overpotential (E_app = -0.1 V)
    kf_cathodic, kb_cathodic = rxn.get_rate_constants(E_app=-0.1)
    assert kf_cathodic > kb_cathodic  # Reduction favored

def test_homogeneous_reaction_rates() -> None:
    """Tests vectorized net rate computations for chemical steps."""
    r = Species("R", 1e-5, 0.0)
    z = Species("Z", 1e-5, 0.0)
    
    # Initialize with mock concentration profiles across 5 spatial nodes
    r.c_profile = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    z.c_profile = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    
    # R <=> Z (kf = 2.0 s^-1, kb = 1.0 s^-1)
    rxn = HomogeneousReaction(reactants=[r], products=[z], kf=2.0, kb=1.0)
    rates = rxn.compute_kinetic_rates()
    
    # Net rate = kf * [R] - kb * [Z]
    expected_net_rate = 2.0 * r.c_profile - 1.0 * z.c_profile
    
    # R is consumed (-net_rate), Z is produced (+net_rate)
    np.testing.assert_allclose(rates["R"], -expected_net_rate)
    np.testing.assert_allclose(rates["Z"], expected_net_rate)

def test_mechanism_assembly() -> None:
    """Verifies the Mechanism container correctly maps species."""
    o = Species("O", 1e-5, 1e-6)
    r = Species("R", 1e-5, 0.0)
    
    e_step = HeterogeneousReaction(o, r)
    mech = Mechanism(species=[o, r], e_reactions=[e_step])
    
    assert "O" in mech.species
    assert "R" in mech.species
    assert len(mech.e_reactions) == 1
    assert len(mech.c_reactions) == 0
