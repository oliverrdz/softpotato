"""
Electrochemical and chemical reaction definitions with strict CGS unit enforcement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .species import Species


def _normalize_species_sequence(
    species_input: Sequence[Species] | Species,
    arg_name: str,
) -> tuple[Species, ...]:
    """
    Validates and normalizes a species or sequence of species into a tuple of Species.

    Args:
        species_input: A single Species or a sequence of Species.
        arg_name: The parameter name for clear error messaging.

    Returns:
        tuple[Species, ...]: A tuple of validated Species instances.

    Raises:
        TypeError: If species_input or any of its elements are not Species.
        ValueError: If species_input is empty.
    """
    if isinstance(species_input, Species):
        return (species_input,)

    if isinstance(species_input, Sequence) and not isinstance(
        species_input, (str, bytes)
    ):
        if len(species_input) == 0:
            raise ValueError(f"{arg_name} must not be empty.")
        for idx, item in enumerate(species_input):
            if not isinstance(item, Species):
                raise TypeError(
                    f"All items in {arg_name} must be Species instances, "
                    f"got {type(item).__name__} at index {idx}."
                )
        return tuple(species_input)

    raise TypeError(
        f"{arg_name} must be a Species or a sequence of Species, "
        f"got {type(species_input).__name__}."
    )


def _normalize_stoichiometry(
    stoich_input: Sequence[int] | None,
    expected_length: int,
    arg_name: str,
) -> tuple[int, ...]:
    """
    Validates and normalizes stoichiometric coefficients.

    Args:
        stoich_input: A sequence of positive integers or None.
        expected_length: Expected number of stoichiometric coefficients.
        arg_name: Parameter name for error messages.

    Returns:
        tuple[int, ...]: Tuple of positive integer coefficients.

    Raises:
        TypeError: If coefficients are not integers.
        ValueError: If length doesn't match or any coefficient is < 1.
    """
    if stoich_input is None:
        return tuple(1 for _ in range(expected_length))

    if not isinstance(stoich_input, Sequence) or isinstance(stoich_input, (str, bytes)):
        raise TypeError(f"{arg_name} must be a sequence of integers.")

    if len(stoich_input) != expected_length:
        raise ValueError(
            f"{arg_name} length ({len(stoich_input)}) must match the number of "
            f"corresponding species ({expected_length})."
        )

    for idx, coeff in enumerate(stoich_input):
        if isinstance(coeff, bool) or not isinstance(coeff, int):
            raise TypeError(
                f"Stoichiometric coefficient at index {idx} in {arg_name} must be an integer, "
                f"got {type(coeff).__name__}."
            )
        if coeff < 1:
            raise ValueError(
                f"Stoichiometric coefficient at index {idx} in {arg_name} must be >= 1, "
                f"got {coeff}."
            )

    return tuple(stoich_input)


class ElectrochemicalReaction:
    """
    Represents an electrochemical reaction at an electrode interface.

    Standard reduction convention:
        Reactants + n e⁻ ⇌ Products

    Attributes:
        reactants: Tuple of Species acting as reactants in the reduction step.
        products: Tuple of Species acting as products in the reduction step.
        n_electrons: Number of electrons transferred (n >= 1).
        E0: Standard reduction potential in Volts vs. reference electrode.
        kinetics: Optional kinetics model (e.g. ButlerVolmer, Nernst).
        stoich_reactants: Stoichiometric coefficients of reactants.
        stoich_products: Stoichiometric coefficients of products.
    """

    def __init__(
        self,
        reactants: Sequence[Species] | Species,
        products: Sequence[Species] | Species,
        n_electrons: int = 1,
        E0: float = 0.0,
        kinetics: Any = None,
        stoich_reactants: Sequence[int] | None = None,
        stoich_products: Sequence[int] | None = None,
    ) -> None:
        """
        Initialize an electrochemical reaction.

        Args:
            reactants: Reactant Species in the reduction direction (e.g., [spec_O]).
            products: Product Species in the reduction direction (e.g., [spec_R]).
            n_electrons: Number of electrons transferred (must be an integer >= 1).
            E0: Standard reduction potential in Volts (defaults to 0.0 V).
            kinetics: Kinetics model (e.g. ButlerVolmer, Nernst). Defaults to None.
            stoich_reactants: Optional stoichiometric coefficients for reactants.
            stoich_products: Optional stoichiometric coefficients for products.

        Raises:
            TypeError: If reactants, products, n_electrons, E0, or stoichiometry are of invalid type.
            ValueError: If reactants or products are empty, or n_electrons < 1.
        """
        self._reactants = _normalize_species_sequence(reactants, "reactants")
        self._products = _normalize_species_sequence(products, "products")

        if isinstance(n_electrons, bool) or not isinstance(n_electrons, int):
            raise TypeError(
                f"n_electrons must be an integer, got {type(n_electrons).__name__}."
            )
        if n_electrons < 1:
            raise ValueError(f"n_electrons must be >= 1, got {n_electrons}.")
        self._n_electrons = n_electrons

        if isinstance(E0, bool) or not isinstance(E0, (int, float)):
            raise TypeError(f"E0 must be a real number, got {type(E0).__name__}.")
        self._E0 = float(E0)

        self._kinetics = kinetics

        self._stoich_reactants = _normalize_stoichiometry(
            stoich_reactants, len(self._reactants), "stoich_reactants"
        )
        self._stoich_products = _normalize_stoichiometry(
            stoich_products, len(self._products), "stoich_products"
        )

        # Net stoichiometry: negative for reactants, positive for products
        stoich: dict[Species, int] = {}
        for sp, coeff in zip(self._reactants, self._stoich_reactants):
            stoich[sp] = stoich.get(sp, 0) - coeff
        for sp, coeff in zip(self._products, self._stoich_products):
            stoich[sp] = stoich.get(sp, 0) + coeff
        self._stoichiometry = stoich

        # Collect unique species preserving order of appearance
        seen_names: dict[str, Species] = {}
        unique_species: list[Species] = []
        for sp in self._reactants + self._products:
            if sp.name not in seen_names:
                seen_names[sp.name] = sp
                unique_species.append(sp)
            elif seen_names[sp.name] != sp:
                raise ValueError(
                    f"Conflicting species definitions found for name '{sp.name}': "
                    f"{seen_names[sp.name]} vs {sp}."
                )
        self._species = tuple(unique_species)

    @property
    def reactants(self) -> tuple[Species, ...]:
        """Reactant species in reduction direction."""
        return self._reactants

    @property
    def products(self) -> tuple[Species, ...]:
        """Product species in reduction direction."""
        return self._products

    @property
    def n_electrons(self) -> int:
        """Number of electrons transferred."""
        return self._n_electrons

    @property
    def E0(self) -> float:
        """Standard reduction potential in Volts."""
        return self._E0

    @property
    def kinetics(self) -> Any:
        """Kinetics model for interfacial charge transfer."""
        return self._kinetics

    @property
    def stoich_reactants(self) -> tuple[int, ...]:
        """Stoichiometric coefficients for reactants."""
        return self._stoich_reactants

    @property
    def stoich_products(self) -> tuple[int, ...]:
        """Stoichiometric coefficients for products."""
        return self._stoich_products

    @property
    def species(self) -> tuple[Species, ...]:
        """All unique species involved in this reaction."""
        return self._species

    @property
    def stoichiometry(self) -> dict[Species, int]:
        """Net stoichiometric coefficients (reactants negative, products positive)."""
        return dict(self._stoichiometry)

    def __repr__(self) -> str:
        reactants_str = ", ".join(s.name for s in self._reactants)
        products_str = ", ".join(s.name for s in self._products)
        kinetics_str = f", kinetics={self.kinetics!r}" if self.kinetics else ""
        return (
            f"ElectrochemicalReaction(reactants=[{reactants_str}], "
            f"products=[{products_str}], n_electrons={self.n_electrons}, "
            f"E0={self.E0}{kinetics_str})"
        )

    def __str__(self) -> str:
        r_terms = [
            f"{c if c > 1 else ''}{s.name}"
            for s, c in zip(self._reactants, self._stoich_reactants)
        ]
        p_terms = [
            f"{c if c > 1 else ''}{s.name}"
            for s, c in zip(self._products, self._stoich_products)
        ]
        e_term = f"{self.n_electrons if self.n_electrons > 1 else ''}e⁻"
        r_side = " + ".join(r_terms + [e_term])
        p_side = " + ".join(p_terms)
        return f"{r_side} ⇌ {p_side} (E0 = {self.E0:.3f} V)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElectrochemicalReaction):
            return False
        return (
            self._reactants == other._reactants
            and self._products == other._products
            and self._n_electrons == other._n_electrons
            and self._E0 == other._E0
            and self._kinetics == other._kinetics
            and self._stoich_reactants == other._stoich_reactants
            and self._stoich_products == other._stoich_products
        )


class ChemicalReaction:
    """
    Represents a homogeneous chemical reaction occurring in the bulk solution.

    Reaction stoichiometry:
        ∑ ν_r R ⇌ ∑ ν_p P

    Attributes:
        reactants: Tuple of Species acting as reactants.
        products: Tuple of Species acting as products.
        kinetics: Optional kinetics model (e.g. FirstOrder).
        stoich_reactants: Stoichiometric coefficients of reactants.
        stoich_products: Stoichiometric coefficients of products.
    """

    def __init__(
        self,
        reactants: Sequence[Species] | Species,
        products: Sequence[Species] | Species,
        kinetics: Any = None,
        stoich_reactants: Sequence[int] | None = None,
        stoich_products: Sequence[int] | None = None,
    ) -> None:
        """
        Initialize a homogeneous chemical reaction.

        Args:
            reactants: Reactant Species (e.g., [spec_R]).
            products: Product Species (e.g., [spec_Z]).
            kinetics: Kinetics model (e.g. FirstOrder). Defaults to None.
            stoich_reactants: Optional stoichiometric coefficients for reactants.
            stoich_products: Optional stoichiometric coefficients for products.

        Raises:
            TypeError: If reactants, products, or stoichiometry are of invalid type.
            ValueError: If reactants or products are empty.
        """
        self._reactants = _normalize_species_sequence(reactants, "reactants")
        self._products = _normalize_species_sequence(products, "products")
        self._kinetics = kinetics

        self._stoich_reactants = _normalize_stoichiometry(
            stoich_reactants, len(self._reactants), "stoich_reactants"
        )
        self._stoich_products = _normalize_stoichiometry(
            stoich_products, len(self._products), "stoich_products"
        )

        # Net stoichiometry: negative for reactants, positive for products
        stoich: dict[Species, int] = {}
        for sp, coeff in zip(self._reactants, self._stoich_reactants):
            stoich[sp] = stoich.get(sp, 0) - coeff
        for sp, coeff in zip(self._products, self._stoich_products):
            stoich[sp] = stoich.get(sp, 0) + coeff
        self._stoichiometry = stoich

        # Collect unique species preserving order of appearance
        seen_names: dict[str, Species] = {}
        unique_species: list[Species] = []
        for sp in self._reactants + self._products:
            if sp.name not in seen_names:
                seen_names[sp.name] = sp
                unique_species.append(sp)
            elif seen_names[sp.name] != sp:
                raise ValueError(
                    f"Conflicting species definitions found for name '{sp.name}': "
                    f"{seen_names[sp.name]} vs {sp}."
                )
        self._species = tuple(unique_species)

    @property
    def reactants(self) -> tuple[Species, ...]:
        """Reactant species."""
        return self._reactants

    @property
    def products(self) -> tuple[Species, ...]:
        """Product species."""
        return self._products

    @property
    def kinetics(self) -> Any:
        """Kinetics model for the homogeneous reaction."""
        return self._kinetics

    @property
    def stoich_reactants(self) -> tuple[int, ...]:
        """Stoichiometric coefficients for reactants."""
        return self._stoich_reactants

    @property
    def stoich_products(self) -> tuple[int, ...]:
        """Stoichiometric coefficients for products."""
        return self._stoich_products

    @property
    def species(self) -> tuple[Species, ...]:
        """All unique species involved in this reaction."""
        return self._species

    @property
    def stoichiometry(self) -> dict[Species, int]:
        """Net stoichiometric coefficients (reactants negative, products positive)."""
        return dict(self._stoichiometry)

    def __repr__(self) -> str:
        reactants_str = ", ".join(s.name for s in self._reactants)
        products_str = ", ".join(s.name for s in self._products)
        kinetics_str = f", kinetics={self.kinetics!r}" if self.kinetics else ""
        return (
            f"ChemicalReaction(reactants=[{reactants_str}], "
            f"products=[{products_str}]{kinetics_str})"
        )

    def __str__(self) -> str:
        r_terms = [
            f"{c if c > 1 else ''}{s.name}"
            for s, c in zip(self._reactants, self._stoich_reactants)
        ]
        p_terms = [
            f"{c if c > 1 else ''}{s.name}"
            for s, c in zip(self._products, self._stoich_products)
        ]
        return f"{' + '.join(r_terms)} ⇌ {' + '.join(p_terms)}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChemicalReaction):
            return False
        return (
            self._reactants == other._reactants
            and self._products == other._products
            and self._kinetics == other._kinetics
            and self._stoich_reactants == other._stoich_reactants
            and self._stoich_products == other._stoich_products
        )


class Mechanism:
    """
    Container for electrochemical and chemical reaction mechanisms.

    Encapsulates the reactions and associated species, providing convenient
    accessors and lifecycle hooks (profile initialization, resets) for numerical solvers.
    """

    def __init__(
        self,
        reactions: (
            Sequence[ElectrochemicalReaction | ChemicalReaction]
            | ElectrochemicalReaction
            | ChemicalReaction
            | None
        ) = None,
        *extra_reactions: ElectrochemicalReaction | ChemicalReaction,
        species: Sequence[Species] | None = None,
    ) -> None:
        """
        Initialize a reaction mechanism.

        Args:
            reactions: A sequence of reactions or a single reaction.
            *extra_reactions: Additional reactions if reactions were passed as positional arguments.
            species: Optional sequence of Species. If omitted, species are automatically
                inferred in order of appearance from the provided reactions.

        Raises:
            TypeError: If any reaction or species is not of the correct type.
            ValueError: If duplicate species names with conflicting properties are found,
                or if a reaction references a species not in the explicit species list.
        """
        all_rxns: list[ElectrochemicalReaction | ChemicalReaction] = []
        if reactions is not None:
            if isinstance(reactions, (ElectrochemicalReaction, ChemicalReaction)):
                all_rxns.append(reactions)
            elif isinstance(reactions, Sequence) and not isinstance(
                reactions, (str, bytes)
            ):
                all_rxns.extend(reactions)
            else:
                raise TypeError(
                    f"reactions must be a sequence of Reaction objects or a single Reaction, "
                    f"got {type(reactions).__name__}."
                )

        all_rxns.extend(extra_reactions)

        for idx, rxn in enumerate(all_rxns):
            if not isinstance(rxn, (ElectrochemicalReaction, ChemicalReaction)):
                raise TypeError(
                    f"Mechanism reactions must be ElectrochemicalReaction or ChemicalReaction "
                    f"instances, got {type(rxn).__name__} at index {idx}."
                )

        self._reactions: tuple[ElectrochemicalReaction | ChemicalReaction, ...] = tuple(
            all_rxns
        )
        self._e_reactions: tuple[ElectrochemicalReaction, ...] = tuple(
            r for r in all_rxns if isinstance(r, ElectrochemicalReaction)
        )
        self._c_reactions: tuple[ChemicalReaction, ...] = tuple(
            r for r in all_rxns if isinstance(r, ChemicalReaction)
        )

        if species is None:
            # Auto-infer species preserving order of appearance
            discovered: list[Species] = []
            seen_names: dict[str, Species] = {}
            for rxn in self._reactions:
                for sp in rxn.species:
                    if sp.name not in seen_names:
                        seen_names[sp.name] = sp
                        discovered.append(sp)
                    elif seen_names[sp.name] != sp:
                        raise ValueError(
                            f"Conflicting species definitions found for name '{sp.name}': "
                            f"{seen_names[sp.name]} vs {sp}."
                        )
            self._species: tuple[Species, ...] = tuple(discovered)
        else:
            if not isinstance(species, Sequence) or isinstance(species, (str, bytes)):
                raise TypeError("species must be a sequence of Species objects.")
            validated_species: list[Species] = []
            seen_names = {}
            for idx, sp in enumerate(species):
                if not isinstance(sp, Species):
                    raise TypeError(
                        f"All items in species must be Species instances, "
                        f"got {type(sp).__name__} at index {idx}."
                    )
                if sp.name in seen_names:
                    raise ValueError(
                        f"Duplicate species name '{sp.name}' provided in species list."
                    )
                seen_names[sp.name] = sp
                validated_species.append(sp)

            # Ensure all species used in reactions are present in the provided species list
            for rxn in self._reactions:
                for sp in rxn.species:
                    if sp.name not in seen_names:
                        raise ValueError(
                            f"Species '{sp.name}' used in reaction {rxn} is not present "
                            f"in the provided species list."
                        )
                    if seen_names[sp.name] != sp:
                        raise ValueError(
                            f"Reaction references species '{sp.name}' with different properties "
                            f"than the species list: {seen_names[sp.name]} vs {sp}."
                        )

            self._species = tuple(validated_species)

        self._species_by_name: dict[str, Species] = {
            sp.name: sp for sp in self._species
        }

    @property
    def species(self) -> tuple[Species, ...]:
        """All chemical species in the mechanism."""
        return self._species

    @property
    def species_names(self) -> list[str]:
        """List of names of all species in the mechanism."""
        return [sp.name for sp in self._species]

    @property
    def reactions(self) -> tuple[ElectrochemicalReaction | ChemicalReaction, ...]:
        """All reactions in the mechanism."""
        return self._reactions

    @property
    def electrochemical_reactions(self) -> tuple[ElectrochemicalReaction, ...]:
        """Electrochemical reactions in the mechanism."""
        return self._e_reactions

    @property
    def e_reactions(self) -> tuple[ElectrochemicalReaction, ...]:
        """Alias for electrochemical_reactions."""
        return self._e_reactions

    @property
    def chemical_reactions(self) -> tuple[ChemicalReaction, ...]:
        """Chemical reactions in the mechanism."""
        return self._c_reactions

    @property
    def c_reactions(self) -> tuple[ChemicalReaction, ...]:
        """Alias for chemical_reactions."""
        return self._c_reactions

    @property
    def homogeneous_stoichiometry_matrix(self) -> np.ndarray:
        """
        Homogeneous stoichiometry matrix (ν_i,j) for bulk chemical reactions.

        Rows correspond to species in self.species, columns correspond to
        chemical reactions in self.chemical_reactions.

        Returns:
            np.ndarray of shape (n_species, n_chemical_reactions) with float64 values.
        """
        n_sp = len(self._species)
        n_cr = len(self._c_reactions)
        matrix = np.zeros((n_sp, n_cr), dtype=np.float64)

        for col_idx, rxn in enumerate(self._c_reactions):
            for row_idx, sp in enumerate(self._species):
                matrix[row_idx, col_idx] = rxn.stoichiometry.get(sp, 0)

        return matrix

    def get_species(self, name: str) -> Species:
        """
        Look up a species in the mechanism by name.

        Args:
            name: String identifier of the species.

        Returns:
            Species: The corresponding Species instance.

        Raises:
            KeyError: If species with the given name is not in the mechanism.
        """
        if name not in self._species_by_name:
            raise KeyError(
                f"Species '{name}' not found in mechanism. "
                f"Available species: {list(self._species_by_name.keys())}"
            )
        return self._species_by_name[name]

    def initialize_profiles(self, n_nodes: int) -> None:
        """
        Initializes the spatial concentration profile array on all species.

        Args:
            n_nodes: Number of spatial grid nodes (must be > 0).
        """
        for sp in self._species:
            sp.initialize_profile(n_nodes)

    def reset(self) -> None:
        """Resets all species concentration profiles back to their bulk values."""
        for sp in self._species:
            sp.reset()

    def __getitem__(
        self, key: str | int
    ) -> Species | ElectrochemicalReaction | ChemicalReaction:
        """
        Access species by name or reaction by index.

        Args:
            key: Species name (str) or reaction index (int).

        Returns:
            Species or Reaction.
        """
        if isinstance(key, str):
            return self.get_species(key)
        if isinstance(key, int):
            return self._reactions[key]
        raise TypeError(
            f"Key must be a species name (str) or reaction index (int), "
            f"got {type(key).__name__}."
        )

    def __contains__(
        self, item: str | Species | ElectrochemicalReaction | ChemicalReaction
    ) -> bool:
        """Checks if a species name, Species, or Reaction is part of the mechanism."""
        if isinstance(item, str):
            return item in self._species_by_name
        if isinstance(item, Species):
            return item in self._species
        if isinstance(item, (ElectrochemicalReaction, ChemicalReaction)):
            return item in self._reactions
        return False

    def __len__(self) -> int:
        """Total number of reactions in the mechanism."""
        return len(self._reactions)

    def __iter__(self):
        """Iterates over all reactions in the mechanism."""
        return iter(self._reactions)

    def __repr__(self) -> str:
        return (
            f"Mechanism(reactions={list(self._reactions)!r}, "
            f"species={list(self._species)!r})"
        )

    def __str__(self) -> str:
        e_str = "\n  ".join(str(r) for r in self._e_reactions) or "None"
        c_str = "\n  ".join(str(r) for r in self._c_reactions) or "None"
        sp_str = ", ".join(str(s) for s in self._species) or "None"
        return (
            f"Mechanism:\n"
            f"  Species: {sp_str}\n"
            f"  Electrochemical Reactions:\n  {e_str}\n"
            f"  Chemical Reactions:\n  {c_str}"
        )
