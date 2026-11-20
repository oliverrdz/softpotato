"""Chemical species abstraction and multi-species metadata registry."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass


def _validate_finite_real(value: object, field_name: str, species_name: str) -> float:
    """Validate that a value is a finite real number (excluding bool) and return float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} for species '{species_name}' must be a real number, "
            f"got {type(value).__name__}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} for species '{species_name}' must be finite, got {value}."
        )
    return result


def _validate_int(value: object, field_name: str, species_name: str) -> int:
    """Validate that a value is an int (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} for species '{species_name}' must be an int, "
            f"got {type(value).__name__}."
        )
    return value


@dataclass(frozen=True)
class Species:
    r"""Encapsulation of a chemical species and its physicochemical properties.

    Parameters
    ----------
    name : str
        Unique chemical identifier (e.g., ``'Ox'``, ``'Red'``, ``'Fe(CN)6_3-'``).
    diffusivity : float
        Standard diffusion coefficient :math:`D_i > 0` in :math:`\text{m}^2/\text{s}`
        (or :math:`\text{cm}^2/\text{s}` depending on system units).
    bulk_concentration : float, default 0.0
        Bulk / resting concentration :math:`c_i^* \ge 0` (:math:`\text{mol/m}^3` or M).
    charge : int, default 0
        Formal ionic charge :math:`z_i` (e.g., ``-3`` for ferricyanide).
    standard_potential : float or None, default None
        Standard formal reduction potential :math:`E^{0\prime}` in Volts vs.
        reference electrode.
    n_electrons : int or None, default None
        Number of electrons transferred in the primary redox reaction.

    Examples
    --------
    >>> from softpotato.reactions import Species
    >>> ox = Species(
    ...     name="Ox",
    ...     diffusivity=1e-9,
    ...     bulk_concentration=1.0,
    ...     charge=-3,
    ...     standard_potential=0.2,
    ...     n_electrons=1,
    ... )
    >>> ox.name
    'Ox'
    >>> ox.diffusivity
    1e-09
    """

    name: str
    diffusivity: float
    bulk_concentration: float = 0.0
    charge: int = 0
    standard_potential: float | None = None
    n_electrons: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Species name must be a non-empty string.")

        diffusivity = _validate_finite_real(self.diffusivity, "Diffusivity", self.name)
        if diffusivity <= 0.0:
            raise ValueError(
                f"Diffusivity for species '{self.name}' must be positive, "
                f"got {self.diffusivity}."
            )
        object.__setattr__(self, "diffusivity", diffusivity)

        bulk_conc = _validate_finite_real(
            self.bulk_concentration, "Bulk concentration", self.name
        )
        if bulk_conc < 0.0:
            raise ValueError(
                f"Bulk concentration for species '{self.name}' must be non-negative, "
                f"got {self.bulk_concentration}."
            )
        object.__setattr__(self, "bulk_concentration", bulk_conc)

        _validate_int(self.charge, "Charge", self.name)

        if self.standard_potential is not None:
            e0 = _validate_finite_real(
                self.standard_potential, "Standard potential", self.name
            )
            object.__setattr__(self, "standard_potential", e0)

        if self.n_electrons is not None:
            _validate_int(self.n_electrons, "n_electrons", self.name)


class SpeciesCollection(Mapping[str, Species]):
    """Ordered container and registry managing a group of :class:`Species` objects.

    Supports dict-like lookup by species name (``collection["Ox"]``) and provides
    convenience property exports compatible with
    :class:`~softpotato.solver.DiffusionProblem`.

    Parameters
    ----------
    species : Iterable[Species], optional
        Initial collection of :class:`Species` instances. Duplicate species names
        raise ``ValueError``.

    Examples
    --------
    >>> from softpotato.reactions import Species, SpeciesCollection
    >>> ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
    >>> red = Species("Red", diffusivity=1.2e-9, bulk_concentration=0.0, n_electrons=-1)
    >>> col = SpeciesCollection([ox, red])
    >>> col["Ox"].diffusivity
    1e-09
    >>> col.diffusivities
    {'Ox': 1e-09, 'Red': 1.2e-09}
    >>> col.initial_conditions
    {'Ox': 1.0, 'Red': 0.0}
    >>> col.n_electrons
    {'Ox': 1, 'Red': -1}
    """

    def __init__(self, species: Iterable[Species] = ()) -> None:
        self._species: dict[str, Species] = {}
        for item in species:
            self.add(item)

    def add(self, species: Species) -> None:
        """Register a :class:`Species` instance in the collection.

        Parameters
        ----------
        species : Species
            Species object to add.

        Raises
        ------
        TypeError
            If ``species`` is not a :class:`Species` instance.
        ValueError
            If a species with the same name is already registered.
        """
        if not isinstance(species, Species):
            raise TypeError(
                f"Expected a Species instance, got {type(species).__name__}."
            )
        if species.name in self._species:
            raise ValueError(
                f"Species '{species.name}' is already present in the collection."
            )
        self._species[species.name] = species

    def remove(self, name: str) -> Species:
        """Remove and return a species by name.

        Parameters
        ----------
        name : str
            Name of the species to remove.

        Returns
        -------
        Species
            The removed species instance.

        Raises
        ------
        KeyError
            If ``name`` is not found in the collection.
        """
        return self._species.pop(name)

    def __getitem__(self, name: str) -> Species:
        return self._species[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._species)

    def __len__(self) -> int:
        return len(self._species)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Species):
            return self._species.get(item.name) == item
        return item in self._species

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SpeciesCollection):
            return self._species == other._species
        return NotImplemented

    def __repr__(self) -> str:
        return f"SpeciesCollection({list(self._species.values())!r})"

    @property
    def names(self) -> list[str]:
        """Ordered list of registered species names."""
        return list(self._species.keys())

    @property
    def diffusivities(self) -> dict[str, float]:
        """Mapping from species name to diffusion coefficient :math:`D_i`."""
        return {name: sp.diffusivity for name, sp in self._species.items()}

    @property
    def bulk_concentrations(self) -> dict[str, float]:
        """Mapping from species name to bulk concentration :math:`c_i^*`."""
        return {name: sp.bulk_concentration for name, sp in self._species.items()}

    @property
    def initial_conditions(self) -> dict[str, float]:
        """Mapping from species name to initial (bulk) concentration for ``DiffusionProblem``."""
        return self.bulk_concentrations

    @property
    def charges(self) -> dict[str, int]:
        """Mapping from species name to formal ionic charge :math:`z_i`."""
        return {name: sp.charge for name, sp in self._species.items()}

    @property
    def standard_potentials(self) -> dict[str, float]:
        """Mapping from species name to standard potential :math:`E^{0\\prime}` for species defining it."""
        return {
            name: sp.standard_potential
            for name, sp in self._species.items()
            if sp.standard_potential is not None
        }

    @property
    def n_electrons(self) -> dict[str, int]:
        """Mapping from species name to electron transfer count for species defining ``n_electrons``."""
        return {
            name: sp.n_electrons
            for name, sp in self._species.items()
            if sp.n_electrons is not None
        }
