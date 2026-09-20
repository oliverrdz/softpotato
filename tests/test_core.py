"""
Tests for core species module.
"""

import numpy as np
import pytest

from softpotato.core import (
    ChemicalReaction,
    ElectrochemicalReaction,
    Mechanism,
    Species,
)


def test_species_initialization_defaults() -> None:
    """Verifies default parameters when initializing Species."""
    spec = Species(name="O", D=1e-5)
    assert spec.name == "O"
    assert spec.D == 1e-5
    assert spec.c_bulk == 0.0
    assert spec.charge == 0
    assert spec.c_profile is None


def test_species_initialization_custom() -> None:
    """Verifies custom parameters for Species."""
    spec = Species(name="O", D=1.0e-5, c_bulk=1.0e-6, charge=1)
    assert spec.name == "O"
    assert spec.D == 1.0e-5
    assert spec.c_bulk == 1.0e-6
    assert spec.charge == 1
    assert spec.c_profile is None


def test_species_name_validation() -> None:
    """Verifies validation of species name."""
    with pytest.raises(ValueError, match="non-empty string"):
        Species(name="", D=1e-5)

    with pytest.raises(ValueError, match="non-empty string"):
        Species(name="   ", D=1e-5)

    with pytest.raises(ValueError, match="non-empty string"):
        # Type error or value error for non-string
        Species(name=123, D=1e-5)  # type: ignore[arg-type]


def test_species_diffusion_validation() -> None:
    """Verifies validation of diffusion coefficient D."""
    with pytest.raises(ValueError, match="non-negative"):
        Species(name="O", D=-1e-5)

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D="fast")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=True)  # type: ignore[arg-type]

    # D = 0 is valid (e.g. immobilized / adsorbed species)
    spec = Species(name="Immobilized", D=0.0)
    assert spec.D == 0.0


def test_species_c_bulk_validation() -> None:
    """Verifies validation of bulk concentration c_bulk."""
    with pytest.raises(ValueError, match="non-negative"):
        Species(name="O", D=1e-5, c_bulk=-1e-6)

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=1e-5, c_bulk="1mM")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        Species(name="O", D=1e-5, c_bulk=False)  # type: ignore[arg-type]


def test_species_charge_validation() -> None:
    """Verifies validation of ionic charge."""
    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        Species(name="O", D=1e-5, charge="1")  # type: ignore[arg-type]

    # Negative charges are valid (e.g., anions)
    spec = Species(name="Anion", D=1e-5, charge=-2)
    assert spec.charge == -2


def test_species_profile_initialization_and_reset() -> None:
    """Verifies concentration profile allocation and reset."""
    spec = Species(name="O", D=1e-5, c_bulk=1e-6)
    assert spec.c_profile is None

    spec.initialize_profile(n_nodes=100)
    assert spec.c_profile is not None
    assert spec.c_profile.shape == (100,)
    assert spec.c_profile.dtype == np.float64
    assert np.all(spec.c_profile == 1e-6)

    # Modify the profile
    spec.c_profile[0] = 0.0
    spec.c_profile[1:10] = 0.5e-6
    assert spec.c_profile[0] == 0.0

    # Reset back to bulk concentration
    spec.reset()
    assert np.all(spec.c_profile == 1e-6)

    # Invalid n_nodes
    with pytest.raises(ValueError, match="greater than 0"):
        spec.initialize_profile(n_nodes=0)

    with pytest.raises(ValueError, match="greater than 0"):
        spec.initialize_profile(n_nodes=-10)

    with pytest.raises(TypeError, match="integer"):
        spec.initialize_profile(n_nodes=50.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        spec.initialize_profile(n_nodes=True)  # type: ignore[arg-type]


def test_species_setters() -> None:
    """Verifies property setters and their validations."""
    spec = Species(name="O", D=1e-5, c_bulk=1e-6, charge=0)

    spec.name = "Oxidant"
    assert spec.name == "Oxidant"
    with pytest.raises(ValueError):
        spec.name = ""

    spec.D = 2e-5
    assert spec.D == 2e-5
    with pytest.raises(ValueError):
        spec.D = -1.0

    spec.c_bulk = 5e-6
    assert spec.c_bulk == 5e-6
    with pytest.raises(ValueError):
        spec.c_bulk = -2.0

    spec.charge = -1
    assert spec.charge == -1
    with pytest.raises(TypeError):
        spec.charge = 1.2  # type: ignore[assignment]

    # c_profile setter validation
    spec.c_profile = np.array([1.0, 2.0, 3.0])
    assert len(spec.c_profile) == 3

    with pytest.raises(TypeError, match="numpy ndarray"):
        spec.c_profile = [1.0, 2.0]  # type: ignore[assignment]

    with pytest.raises(ValueError, match="1D array"):
        spec.c_profile = np.zeros((3, 3))

    spec.c_profile = None
    assert spec.c_profile is None
    # Reset when profile is None should not raise error
    spec.reset()


def test_species_equality_and_representations() -> None:
    """Verifies __repr__, __str__, and __eq__."""
    spec1 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=1)
    spec2 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=1)
    spec3 = Species(name="R", D=1e-5, c_bulk=1e-6, charge=1)
    spec4 = Species(name="O", D=2e-5, c_bulk=1e-6, charge=1)
    spec5 = Species(name="O", D=1e-5, c_bulk=0.0, charge=1)
    spec6 = Species(name="O", D=1e-5, c_bulk=1e-6, charge=0)

    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1 != spec4
    assert spec1 != spec5
    assert spec1 != spec6
    assert spec1 != "O"

    assert "Species(name='O'" in repr(spec1)
    assert "cm²/s" in repr(spec1)
    assert "mol/cm³" in repr(spec1)

    assert "Species O" in str(spec1)
    assert "cm²/s" in str(spec1)
    assert "mol/cm³" in str(spec1)


# =============================================================================
# Tests for ElectrochemicalReaction
# =============================================================================


def test_electrochemical_reaction_defaults() -> None:
    """Verifies default parameters for ElectrochemicalReaction."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)

    rxn = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])

    assert rxn.reactants == (spec_o,)
    assert rxn.products == (spec_r,)
    assert rxn.n_electrons == 1
    assert rxn.E0 == 0.0
    assert rxn.kinetics is None
    assert rxn.stoich_reactants == (1,)
    assert rxn.stoich_products == (1,)
    assert rxn.species == (spec_o, spec_r)
    assert rxn.stoichiometry == {spec_o: -1, spec_r: 1}


def test_electrochemical_reaction_custom_and_single_species() -> None:
    """Verifies single species input and custom parameters."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)

    # Passing single Species instead of a list
    rxn = ElectrochemicalReaction(
        reactants=spec_o,
        products=spec_r,
        n_electrons=2,
        E0=-0.35,
        kinetics="dummy_kinetics",
        stoich_reactants=[2],
        stoich_products=[1],
    )

    assert rxn.reactants == (spec_o,)
    assert rxn.products == (spec_r,)
    assert rxn.n_electrons == 2
    assert rxn.E0 == -0.35
    assert rxn.kinetics == "dummy_kinetics"
    assert rxn.stoich_reactants == (2,)
    assert rxn.stoich_products == (1,)
    assert rxn.stoichiometry == {spec_o: -2, spec_r: 1}


def test_electrochemical_reaction_validation() -> None:
    """Verifies validation of arguments in ElectrochemicalReaction."""
    spec_o = Species(name="O", D=1e-5)
    spec_r = Species(name="R", D=1e-5)

    # Empty reactants / products
    with pytest.raises(ValueError, match="reactants must not be empty"):
        ElectrochemicalReaction(reactants=[], products=[spec_r])

    with pytest.raises(ValueError, match="products must not be empty"):
        ElectrochemicalReaction(reactants=[spec_o], products=[])

    # Invalid type in reactants / products
    with pytest.raises(TypeError, match="must be Species instances"):
        ElectrochemicalReaction(reactants=["O"], products=[spec_r])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="must be Species instances"):
        ElectrochemicalReaction(reactants=[spec_o], products=[123])  # type: ignore[list-item]

    # n_electrons validation
    with pytest.raises(ValueError, match=">= 1"):
        ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], n_electrons=0)

    with pytest.raises(TypeError, match="must be an integer"):
        ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], n_electrons=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be an integer"):
        ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], n_electrons=True)  # type: ignore[arg-type]

    # E0 validation
    with pytest.raises(TypeError, match="real number"):
        ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], E0="0.0")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], E0=True)  # type: ignore[arg-type]

    # Stoichiometry length mismatch
    with pytest.raises(ValueError, match="length"):
        ElectrochemicalReaction(
            reactants=[spec_o], products=[spec_r], stoich_reactants=[1, 2]
        )

    # Stoichiometry value < 1
    with pytest.raises(ValueError, match=">= 1"):
        ElectrochemicalReaction(
            reactants=[spec_o], products=[spec_r], stoich_reactants=[0]
        )


def test_electrochemical_reaction_representations_and_equality() -> None:
    """Verifies __repr__, __str__, and __eq__ for ElectrochemicalReaction."""
    spec_o = Species(name="O", D=1e-5)
    spec_r = Species(name="R", D=1e-5)

    rxn1 = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], E0=0.2)
    rxn2 = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], E0=0.2)
    rxn3 = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r], E0=-0.2)

    assert rxn1 == rxn2
    assert rxn1 != rxn3
    assert rxn1 != "not_a_reaction"

    assert "ElectrochemicalReaction" in repr(rxn1)
    assert "reactants=[O]" in repr(rxn1)
    assert "products=[R]" in repr(rxn1)

    assert "O + e⁻ ⇌ R" in str(rxn1)
    assert "0.200 V" in str(rxn1)


# =============================================================================
# Tests for ChemicalReaction
# =============================================================================


def test_chemical_reaction_defaults() -> None:
    """Verifies default parameters for ChemicalReaction."""
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    spec_z = Species(name="Z", D=1e-5, c_bulk=0.0)

    rxn = ChemicalReaction(reactants=[spec_r], products=[spec_z])

    assert rxn.reactants == (spec_r,)
    assert rxn.products == (spec_z,)
    assert rxn.kinetics is None
    assert rxn.stoich_reactants == (1,)
    assert rxn.stoich_products == (1,)
    assert rxn.species == (spec_r, spec_z)
    assert rxn.stoichiometry == {spec_r: -1, spec_z: 1}


def test_chemical_reaction_custom_and_single_species() -> None:
    """Verifies custom stoichiometry and kinetics in ChemicalReaction."""
    spec_r = Species(name="R", D=1e-5)
    spec_z = Species(name="Z", D=1e-5)

    rxn = ChemicalReaction(
        reactants=spec_r,
        products=spec_z,
        kinetics="first_order_kinetics",
        stoich_reactants=[2],
        stoich_products=[3],
    )

    assert rxn.reactants == (spec_r,)
    assert rxn.products == (spec_z,)
    assert rxn.kinetics == "first_order_kinetics"
    assert rxn.stoich_reactants == (2,)
    assert rxn.stoich_products == (3,)
    assert rxn.stoichiometry == {spec_r: -2, spec_z: 3}


def test_chemical_reaction_validation() -> None:
    """Verifies validation in ChemicalReaction."""
    spec_r = Species(name="R", D=1e-5)
    spec_z = Species(name="Z", D=1e-5)

    with pytest.raises(ValueError, match="reactants must not be empty"):
        ChemicalReaction(reactants=[], products=[spec_z])

    with pytest.raises(ValueError, match="products must not be empty"):
        ChemicalReaction(reactants=[spec_r], products=[])

    with pytest.raises(TypeError, match="must be Species instances"):
        ChemicalReaction(reactants=["R"], products=[spec_z])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="length"):
        ChemicalReaction(reactants=[spec_r], products=[spec_z], stoich_products=[1, 2])

    with pytest.raises(ValueError, match=">= 1"):
        ChemicalReaction(reactants=[spec_r], products=[spec_z], stoich_reactants=[-1])


def test_chemical_reaction_representations_and_equality() -> None:
    """Verifies __repr__, __str__, and __eq__ for ChemicalReaction."""
    spec_r = Species(name="R", D=1e-5)
    spec_z = Species(name="Z", D=1e-5)

    rxn1 = ChemicalReaction(reactants=[spec_r], products=[spec_z])
    rxn2 = ChemicalReaction(reactants=[spec_r], products=[spec_z])
    rxn3 = ChemicalReaction(reactants=[spec_r], products=[spec_z], kinetics="k")

    assert rxn1 == rxn2
    assert rxn1 != rxn3
    assert rxn1 != 42

    assert "ChemicalReaction" in repr(rxn1)
    assert "reactants=[R]" in repr(rxn1)
    assert "R ⇌ Z" in str(rxn1)


# =============================================================================
# Tests for Mechanism
# =============================================================================


def test_mechanism_initialization_single_and_multiple() -> None:
    """Verifies mechanism initialization with single and multiple reactions."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    spec_z = Species(name="Z", D=1e-5, c_bulk=0.0)

    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])
    rxn_c = ChemicalReaction(reactants=[spec_r], products=[spec_z])

    # Single reaction as list
    mech1 = Mechanism([rxn_e])
    assert len(mech1) == 1
    assert mech1.e_reactions == (rxn_e,)
    assert mech1.c_reactions == ()
    assert mech1.species == (spec_o, spec_r)

    # Multiple reactions as list (EC mechanism)
    mech2 = Mechanism([rxn_e, rxn_c])
    assert len(mech2) == 2
    assert mech2.e_reactions == (rxn_e,)
    assert mech2.c_reactions == (rxn_c,)
    assert mech2.species == (spec_o, spec_r, spec_z)
    assert mech2.species_names == ["O", "R", "Z"]

    # Positional arguments: Mechanism(rxn_e, rxn_c)
    mech3 = Mechanism(rxn_e, rxn_c)
    assert len(mech3) == 2
    assert mech3.species == (spec_o, spec_r, spec_z)


def test_mechanism_explicit_species() -> None:
    """Verifies explicit species specification in Mechanism."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    spec_inert = Species(name="Inert", D=1e-5, c_bulk=1e-4)

    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])

    # Explicit species including an inert spectator
    mech = Mechanism([rxn_e], species=[spec_o, spec_r, spec_inert])
    assert mech.species == (spec_o, spec_r, spec_inert)
    assert "Inert" in mech


def test_mechanism_validation() -> None:
    """Verifies error handling in Mechanism."""
    spec_o = Species(name="O", D=1e-5)
    spec_r = Species(name="R", D=1e-5)
    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])

    # Invalid reaction type
    with pytest.raises(
        TypeError, match="must be ElectrochemicalReaction or ChemicalReaction"
    ):
        Mechanism(["not_a_reaction"])  # type: ignore[list-item]

    # Reaction uses species not in explicit species list
    with pytest.raises(ValueError, match="not present in the provided species list"):
        Mechanism([rxn_e], species=[spec_o])

    # Duplicate species name in explicit species list
    spec_o_dup = Species(name="O", D=1e-5)
    with pytest.raises(ValueError, match="Duplicate species name"):
        Mechanism([], species=[spec_o, spec_o_dup])

    # Conflicting species definition in auto-discovery
    spec_o_conflict = Species(name="O", D=2e-5)  # Different D!
    rxn_e_conflict = ElectrochemicalReaction(
        reactants=[spec_o_conflict], products=[spec_r]
    )
    with pytest.raises(ValueError, match="Conflicting species definitions"):
        Mechanism([rxn_e, rxn_e_conflict])


def test_mechanism_accessors_and_indexing() -> None:
    """Verifies get_species, __getitem__, __contains__, and __iter__."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])

    mech = Mechanism([rxn_e])

    # Access by species name
    assert mech.get_species("O") is spec_o
    assert mech["O"] is spec_o
    assert mech["R"] is spec_r

    # Missing species
    with pytest.raises(KeyError, match="not found in mechanism"):
        mech.get_species("Z")

    with pytest.raises(KeyError, match="not found in mechanism"):
        _ = mech["Z"]

    # Access by reaction index
    assert mech[0] is rxn_e

    # Invalid key type
    with pytest.raises(TypeError, match="Key must be a species name"):
        _ = mech[1.5]  # type: ignore[index]

    # Membership checks
    assert "O" in mech
    assert "Unknown" not in mech
    assert spec_o in mech
    assert rxn_e in mech
    assert 123 not in mech

    # Iteration
    rxns = list(mech)
    assert rxns == [rxn_e]


def test_mechanism_profiles_and_reset() -> None:
    """Verifies initialize_profiles and reset delegation to Species."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])

    mech = Mechanism([rxn_e])

    assert spec_o.c_profile is None
    assert spec_r.c_profile is None

    mech.initialize_profiles(n_nodes=50)

    assert spec_o.c_profile is not None
    assert spec_r.c_profile is not None
    assert len(spec_o.c_profile) == 50
    assert np.all(spec_o.c_profile == 1e-6)
    assert np.all(spec_r.c_profile == 0.0)

    # Modify profiles and reset
    spec_o.c_profile[0] = 0.5e-6
    spec_r.c_profile[0] = 0.5e-6

    mech.reset()

    assert np.all(spec_o.c_profile == 1e-6)
    assert np.all(spec_r.c_profile == 0.0)


def test_mechanism_homogeneous_stoichiometry_matrix() -> None:
    """Verifies homogeneous stoichiometry matrix computation."""
    spec_o = Species(name="O", D=1e-5, c_bulk=1e-6)
    spec_r = Species(name="R", D=1e-5, c_bulk=0.0)
    spec_z = Species(name="Z", D=1e-5, c_bulk=0.0)

    rxn_e = ElectrochemicalReaction(reactants=[spec_o], products=[spec_r])
    # R -> 2 Z
    rxn_c = ChemicalReaction(reactants=[spec_r], products=[spec_z], stoich_products=[2])

    mech = Mechanism([rxn_e, rxn_c])

    # Species: O (idx 0), R (idx 1), Z (idx 2)
    # Chemical reaction 0: -1 R, +2 Z, 0 O
    matrix = mech.homogeneous_stoichiometry_matrix
    assert matrix.shape == (3, 1)
    assert matrix[0, 0] == 0.0  # O
    assert matrix[1, 0] == -1.0  # R
    assert matrix[2, 0] == 2.0  # Z

    # Mechanism with no chemical reactions
    mech_e_only = Mechanism([rxn_e])
    matrix_e_only = mech_e_only.homogeneous_stoichiometry_matrix
    assert matrix_e_only.shape == (2, 0)
