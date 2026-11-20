"""Unit tests for softpotato.reactions.species (Species and SpeciesCollection)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import softpotato as sp
from softpotato.reactions import Species, SpeciesCollection
from softpotato.solver import DiffusionProblem, DirichletBC, NeumannBC, get_solver


class TestSpeciesCreationAndDefaults:
    """Tests for valid Species instantiation and defaults."""

    def test_minimal_initialization_defaults(self) -> None:
        ox = Species(name="Ox", diffusivity=1e-9)
        assert ox.name == "Ox"
        assert ox.diffusivity == 1e-9
        assert isinstance(ox.diffusivity, float)
        assert ox.bulk_concentration == 0.0
        assert isinstance(ox.bulk_concentration, float)
        assert ox.charge == 0
        assert ox.standard_potential is None
        assert ox.n_electrons is None

    def test_full_initialization(self) -> None:
        ferri = Species(
            name="Fe(CN)6_3-",
            diffusivity=7.2e-10,
            bulk_concentration=5.0,
            charge=-3,
            standard_potential=0.36,
            n_electrons=1,
        )
        assert ferri.name == "Fe(CN)6_3-"
        assert ferri.diffusivity == 7.2e-10
        assert ferri.bulk_concentration == 5.0
        assert ferri.charge == -3
        assert ferri.standard_potential == 0.36
        assert ferri.n_electrons == 1

    def test_numeric_int_coercion_to_float(self) -> None:
        species = Species(
            name="A",
            diffusivity=1,
            bulk_concentration=2,
            standard_potential=0,
        )
        assert species.diffusivity == 1.0
        assert isinstance(species.diffusivity, float)
        assert species.bulk_concentration == 2.0
        assert isinstance(species.bulk_concentration, float)
        assert species.standard_potential == 0.0
        assert isinstance(species.standard_potential, float)

    def test_negative_and_zero_n_electrons_allowed(self) -> None:
        red = Species(name="Red", diffusivity=1e-9, n_electrons=-1)
        inert = Species(name="Inert", diffusivity=1e-9, n_electrons=0)
        assert red.n_electrons == -1
        assert inert.n_electrons == 0

    def test_top_level_and_subpackage_exports(self) -> None:
        assert sp.Species is Species
        assert sp.SpeciesCollection is SpeciesCollection
        assert sp.reactions.Species is Species
        assert sp.reactions.SpeciesCollection is SpeciesCollection


class TestSpeciesValidation:
    """Tests for parameter validation in Species.__post_init__."""

    @pytest.mark.parametrize("bad_name", ["", "   ", "\t\n", None, 123])
    def test_invalid_name_raises_value_error(self, bad_name: object) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Species(name=bad_name, diffusivity=1e-9)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_d", [0.0, -1e-9, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_non_positive_or_non_finite_diffusivity_raises_value_error(
        self, bad_d: float
    ) -> None:
        with pytest.raises(ValueError, match="Diffusivity"):
            Species(name="Ox", diffusivity=bad_d)

    @pytest.mark.parametrize("bad_d", [True, False, "1e-9", None])
    def test_non_numeric_diffusivity_raises_type_error(self, bad_d: object) -> None:
        with pytest.raises(TypeError, match="Diffusivity"):
            Species(name="Ox", diffusivity=bad_d)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_conc", [-1e-12, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_negative_or_non_finite_bulk_concentration_raises_value_error(
        self, bad_conc: float
    ) -> None:
        with pytest.raises(ValueError, match="Bulk concentration"):
            Species(name="Ox", diffusivity=1e-9, bulk_concentration=bad_conc)

    @pytest.mark.parametrize("bad_conc", [True, False, "1.0", None])
    def test_non_numeric_bulk_concentration_raises_type_error(
        self, bad_conc: object
    ) -> None:
        with pytest.raises(TypeError, match="Bulk concentration"):
            Species(name="Ox", diffusivity=1e-9, bulk_concentration=bad_conc)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_charge", [1.5, True, False, "1", None])
    def test_invalid_charge_raises_type_error(self, bad_charge: object) -> None:
        with pytest.raises(TypeError, match="Charge"):
            Species(name="Ox", diffusivity=1e-9, charge=bad_charge)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_e0", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_standard_potential_raises_value_error(
        self, bad_e0: float
    ) -> None:
        with pytest.raises(ValueError, match="Standard potential"):
            Species(name="Ox", diffusivity=1e-9, standard_potential=bad_e0)

    @pytest.mark.parametrize("bad_e0", [True, False, "0.2"])
    def test_non_numeric_standard_potential_raises_type_error(
        self, bad_e0: object
    ) -> None:
        with pytest.raises(TypeError, match="Standard potential"):
            Species(name="Ox", diffusivity=1e-9, standard_potential=bad_e0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_n", [1.5, True, False, "1"])
    def test_invalid_n_electrons_raises_type_error(self, bad_n: object) -> None:
        with pytest.raises(TypeError, match="n_electrons"):
            Species(name="Ox", diffusivity=1e-9, n_electrons=bad_n)  # type: ignore[arg-type]


class TestSpeciesImmutabilityAndHashability:
    """Tests for frozen dataclass immutability, equality, and hashability."""

    def test_immutability(self) -> None:
        ox = Species(name="Ox", diffusivity=1e-9, bulk_concentration=1.0)
        with pytest.raises(FrozenInstanceError):
            ox.diffusivity = 2e-9  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            ox.name = "Red"  # type: ignore[misc]

    def test_hashability_and_equality(self) -> None:
        ox1 = Species(
            name="Ox",
            diffusivity=1e-9,
            bulk_concentration=1.0,
            charge=1,
            standard_potential=0.25,
            n_electrons=1,
        )
        ox2 = Species(
            name="Ox",
            diffusivity=1e-9,
            bulk_concentration=1.0,
            charge=1,
            standard_potential=0.25,
            n_electrons=1,
        )
        red = Species(
            name="Red",
            diffusivity=1e-9,
            bulk_concentration=0.0,
            charge=0,
            standard_potential=0.25,
            n_electrons=-1,
        )

        assert ox1 == ox2
        assert hash(ox1) == hash(ox2)
        assert ox1 != red

        species_set = {ox1, ox2, red}
        assert len(species_set) == 2
        assert ox1 in species_set
        assert red in species_set

        mapping = {ox1: "oxidized", red: "reduced"}
        assert mapping[ox2] == "oxidized"


class TestSpeciesCollection:
    """Tests for SpeciesCollection registry and DiffusionProblem export helpers."""

    @pytest.fixture
    def redox_pair(self) -> tuple[Species, Species]:
        ox = Species(
            name="Ox",
            diffusivity=1e-9,
            bulk_concentration=1.0,
            charge=-3,
            standard_potential=0.2,
            n_electrons=1,
        )
        red = Species(
            name="Red",
            diffusivity=1.2e-9,
            bulk_concentration=0.0,
            charge=-4,
            standard_potential=0.2,
            n_electrons=-1,
        )
        return ox, red

    def test_empty_collection(self) -> None:
        col = SpeciesCollection()
        assert len(col) == 0
        assert col.names == []
        assert col.diffusivities == {}
        assert col.initial_conditions == {}
        assert col.bulk_concentrations == {}
        assert col.charges == {}
        assert col.standard_potentials == {}
        assert col.n_electrons == {}

    def test_mapping_interface_and_order(
        self, redox_pair: tuple[Species, Species]
    ) -> None:
        ox, red = redox_pair
        col = SpeciesCollection(s for s in [ox, red])

        assert len(col) == 2
        assert list(col) == ["Ox", "Red"]
        assert col.names == ["Ox", "Red"]
        assert col["Ox"] is ox
        assert col["Red"] is red
        assert col.get("Ox") is ox
        assert col.get("Missing") is None
        assert "Ox" in col
        assert "Missing" not in col
        assert ox in col
        assert Species("Ox", diffusivity=2e-9) not in col
        assert list(col.keys()) == ["Ox", "Red"]
        assert list(col.values()) == [ox, red]
        assert list(col.items()) == [("Ox", ox), ("Red", red)]

        with pytest.raises(KeyError):
            _ = col["Unknown"]

    def test_add_and_remove(self, redox_pair: tuple[Species, Species]) -> None:
        ox, red = redox_pair
        col = SpeciesCollection([ox])
        col.add(red)
        assert col.names == ["Ox", "Red"]

        removed = col.remove("Ox")
        assert removed is ox
        assert col.names == ["Red"]

        with pytest.raises(KeyError):
            col.remove("Ox")

    def test_duplicate_name_rejected(self, redox_pair: tuple[Species, Species]) -> None:
        ox, _ = redox_pair
        ox_dup = Species(name="Ox", diffusivity=2e-9)

        with pytest.raises(ValueError, match="already present"):
            SpeciesCollection([ox, ox_dup])

        col = SpeciesCollection([ox])
        with pytest.raises(ValueError, match="already present"):
            col.add(ox_dup)

    def test_non_species_rejected(self) -> None:
        with pytest.raises(TypeError, match="Expected a Species instance"):
            SpeciesCollection(["Ox"])  # type: ignore[list-item]

        col = SpeciesCollection()
        with pytest.raises(TypeError, match="Expected a Species instance"):
            col.add({"name": "Ox"})  # type: ignore[arg-type]

    def test_equality_and_repr(self, redox_pair: tuple[Species, Species]) -> None:
        ox, red = redox_pair
        col1 = SpeciesCollection([ox, red])
        col2 = SpeciesCollection([ox, red])
        col3 = SpeciesCollection([ox])

        assert col1 == col2
        assert col1 != col3
        assert col1 != {"Ox": ox, "Red": red}
        assert "SpeciesCollection" in repr(col1)
        assert "Ox" in repr(col1)

    def test_export_properties(self, redox_pair: tuple[Species, Species]) -> None:
        ox, red = redox_pair
        product = Species(name="P", diffusivity=8e-10, bulk_concentration=0.0, charge=0)
        col = SpeciesCollection([ox, red, product])

        assert col.diffusivities == {"Ox": 1e-9, "Red": 1.2e-9, "P": 8e-10}
        assert col.initial_conditions == {"Ox": 1.0, "Red": 0.0, "P": 0.0}
        assert col.bulk_concentrations == {"Ox": 1.0, "Red": 0.0, "P": 0.0}
        assert col.charges == {"Ox": -3, "Red": -4, "P": 0}
        assert col.standard_potentials == {"Ox": 0.2, "Red": 0.2}
        assert col.n_electrons == {"Ox": 1, "Red": -1}

    def test_exported_dicts_are_fresh_copies(
        self, redox_pair: tuple[Species, Species]
    ) -> None:
        ox, red = redox_pair
        col = SpeciesCollection([ox, red])

        diffs = col.diffusivities
        diffs["Ox"] = 999.0
        assert col.diffusivities["Ox"] == 1e-9

        ics = col.initial_conditions
        ics["Ox"] = 42.0
        assert col.initial_conditions["Ox"] == 1.0

    def test_diffusion_problem_compatibility(self) -> None:
        ox = Species(
            name="Ox",
            diffusivity=1e-9,
            bulk_concentration=1.0,
            n_electrons=1,
        )
        red = Species(
            name="Red",
            diffusivity=1.2e-9,
            bulk_concentration=0.0,
            n_electrons=1,
        )
        collection = SpeciesCollection([ox, red])

        grid = np.linspace(0.0, 1e-3, 30)
        problem = DiffusionProblem(
            grid=grid,
            diffusivity=collection.diffusivities,
            boundary_conditions={
                "Ox": (
                    DirichletBC(0.0),
                    DirichletBC(collection["Ox"].bulk_concentration),
                ),
                "Red": (
                    NeumannBC(0.0),
                    DirichletBC(collection["Red"].bulk_concentration),
                ),
            },
            initial_conditions=collection.initial_conditions,
            species=collection.names,
            n_electrons=collection.n_electrons,
        )

        assert problem.species == ["Ox", "Red"]
        assert problem.diffusivity == {"Ox": 1e-9, "Red": 1.2e-9}
        assert problem.n_electrons == {"Ox": 1, "Red": 1}

        solver = get_solver("scipy_ivp")
        result = solver.solve(problem, t_span=(0.0, 1.0))
        assert result.success
        assert result["Ox"].shape == (101, 30)
        assert result["Red"].shape == (101, 30)
