"""Unit tests for softpotato.reactions.homogeneous."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

import softpotato as sp
from softpotato.reactions import (
    CatalyticReaction,
    CustomReaction,
    DimerizationReaction,
    DisproportionationReaction,
    FirstOrderReaction,
    HomogeneousReaction,
    MassActionReaction,
    ReactionNetwork,
    ReversibleReaction,
    SecondOrderReaction,
    Species,
)
from softpotato.solver import DiffusionProblem, NeumannBC, get_solver

RNG = np.random.default_rng(1234)
N = 7


def _random_c(names: list[str], n: int = N) -> dict[str, np.ndarray]:
    return {name: RNG.uniform(0.1, 2.0, n) for name in names}


def _numerical_local_jacobian(
    network: ReactionNetwork, c: dict[str, np.ndarray], node: int
) -> np.ndarray:
    """Finite-difference dR_i/dc_k at a single grid node via approx_fprime."""
    names = network.species

    def f(y: np.ndarray) -> np.ndarray:
        c_local = {name: np.array([y[i]]) for i, name in enumerate(names)}
        rates = network(c_local, None, 0.0)
        return np.array([rates[name][0] for name in names])

    y0 = np.array([c[name][node] for name in names])
    return np.atleast_2d(approx_fprime(y0, f, 1e-7))


ALL_REACTIONS = [
    FirstOrderReaction("A", "B", k=1.7),
    ReversibleReaction("A", "B", k_f=3.0, k_b=0.4),
    SecondOrderReaction("A", "B", "C", k=2.5),
    CatalyticReaction("B", "A", k=4.0, substrate="Z", product="Y"),
    CatalyticReaction("B", "A", k=4.0),
    DimerizationReaction("B", "B2", k=0.8),
    DisproportionationReaction("B", "A", "C", k=1.3),
    MassActionReaction({"A": 2, "B": 1}, {"C": 1, "D": 3}, k_f=0.7, k_b=0.2),
    MassActionReaction({"A": 1, "K": 1}, {"B": 1, "K": 1}, k_f=1.1),
]


# ---------------------------------------------------------------------------
# Rate vectors
# ---------------------------------------------------------------------------
class TestRateVectors:
    def test_first_order(self) -> None:
        rxn = FirstOrderReaction("A", "B", k=2.0)
        c = _random_c(["A", "B"])
        rates = rxn.rates(c, None, 0.0)
        np.testing.assert_allclose(rates["A"], -2.0 * c["A"])
        np.testing.assert_allclose(rates["B"], 2.0 * c["A"])
        assert rxn.k == 2.0

    def test_first_order_untracked_product(self) -> None:
        rxn = FirstOrderReaction("A", None, k=2.0)
        rates = rxn.rates({"A": np.ones(3)}, None, 0.0)
        assert set(rates) == {"A"}
        np.testing.assert_allclose(rates["A"], -2.0)

    def test_reversible(self) -> None:
        rxn = ReversibleReaction("A", "B", k_f=3.0, k_b=0.5)
        c = _random_c(["A", "B"])
        rates = rxn.rates(c, None, 0.0)
        expected = -3.0 * c["A"] + 0.5 * c["B"]
        np.testing.assert_allclose(rates["A"], expected)
        np.testing.assert_allclose(rates["B"], -expected)
        assert rxn.equilibrium_constant == pytest.approx(6.0)
        assert rxn.is_reversible

    def test_reversible_from_equilibrium(self) -> None:
        rxn = ReversibleReaction.from_equilibrium("A", "B", K=4.0, k_f=8.0)
        assert rxn.k_b == pytest.approx(2.0)
        with pytest.raises(ValueError):
            ReversibleReaction.from_equilibrium("A", "B", K=0.0, k_f=1.0)

    def test_reversible_equilibrium_rate_vanishes(self) -> None:
        rxn = ReversibleReaction("A", "B", k_f=3.0, k_b=1.5)
        c = {"A": np.full(4, 1.0), "B": np.full(4, 2.0)}  # K = 2 = cB/cA
        np.testing.assert_allclose(rxn.rate(c, None, 0.0), 0.0, atol=1e-14)

    def test_second_order(self) -> None:
        rxn = SecondOrderReaction("A", "B", "C", k=2.0)
        c = _random_c(["A", "B", "C"])
        rates = rxn.rates(c, None, 0.0)
        r = 2.0 * c["A"] * c["B"]
        np.testing.assert_allclose(rates["A"], -r)
        np.testing.assert_allclose(rates["B"], -r)
        np.testing.assert_allclose(rates["C"], r)

    def test_second_order_requires_distinct_reactants(self) -> None:
        with pytest.raises(ValueError, match="DimerizationReaction"):
            SecondOrderReaction("A", "A", "C", k=1.0)

    def test_catalytic_explicit_substrate(self) -> None:
        rxn = CatalyticReaction("B", "A", k=5.0, substrate="Z", product="Y")
        c = _random_c(["A", "B", "Z", "Y"])
        rates = rxn.rates(c, None, 0.0)
        r = 5.0 * c["B"] * c["Z"]
        np.testing.assert_allclose(rates["B"], -r)
        np.testing.assert_allclose(rates["Z"], -r)
        np.testing.assert_allclose(rates["A"], r)
        np.testing.assert_allclose(rates["Y"], r)

    def test_catalytic_pseudo_first_order(self) -> None:
        rxn = CatalyticReaction("B", "A", k=5.0)
        c = _random_c(["A", "B"])
        rates = rxn.rates(c, None, 0.0)
        np.testing.assert_allclose(rates["B"], -5.0 * c["B"])
        np.testing.assert_allclose(rates["A"], 5.0 * c["B"])

    def test_dimerization(self) -> None:
        rxn = DimerizationReaction("B", "B2", k=0.5)
        c = _random_c(["B", "B2"])
        rates = rxn.rates(c, None, 0.0)
        np.testing.assert_allclose(rates["B"], -2.0 * 0.5 * c["B"] ** 2)
        np.testing.assert_allclose(rates["B2"], 0.5 * c["B"] ** 2)

    def test_disproportionation(self) -> None:
        rxn = DisproportionationReaction("B", "A", "C", k=0.5)
        c = _random_c(["A", "B", "C"])
        rates = rxn.rates(c, None, 0.0)
        r = 0.5 * c["B"] ** 2
        np.testing.assert_allclose(rates["B"], -2.0 * r)
        np.testing.assert_allclose(rates["A"], r)
        np.testing.assert_allclose(rates["C"], r)

    def test_disproportionation_same_products(self) -> None:
        rxn = DisproportionationReaction("B", "A", "A", k=1.0)
        assert rxn.stoichiometry == {"B": -2, "A": 2}

    def test_mass_action_catalyst_cancels(self) -> None:
        rxn = MassActionReaction({"A": 1, "K": 1}, {"B": 1, "K": 1}, k_f=2.0)
        assert rxn.stoichiometry == {"A": -1, "K": 0, "B": 1}
        c = _random_c(["A", "B", "K"])
        rates = rxn.rates(c, None, 0.0)
        assert "K" not in rates
        np.testing.assert_allclose(rates["B"], 2.0 * c["A"] * c["K"])

    def test_call_alias_and_species_objects(self) -> None:
        a = Species("A", diffusivity=1e-9, bulk_concentration=1.0)
        rxn = FirstOrderReaction(a, "B", k=1.0)
        c = _random_c(["A", "B"])
        assert rxn.species == ("A", "B")
        np.testing.assert_allclose(rxn(c, None, 0.0)["B"], c["A"])

    def test_scalar_concentrations(self) -> None:
        rxn = SecondOrderReaction("A", "B", "C", k=2.0)
        rates = rxn.rates({"A": 1.5, "B": 2.0, "C": 0.0}, None, 0.0)
        assert rates["C"] == pytest.approx(6.0)

    def test_missing_concentration_raises(self) -> None:
        rxn = SecondOrderReaction("A", "B", "C", k=2.0)
        with pytest.raises(KeyError, match="'B'"):
            rxn.rates({"A": np.ones(3)}, None, 0.0)

    def test_repr(self) -> None:
        assert "A + B -> C" in repr(SecondOrderReaction("A", "B", "C", k=1.0))
        assert "<=>" in repr(ReversibleReaction("A", "B", k_f=1.0, k_b=1.0))
        assert "2B -> B2" in repr(DimerizationReaction("B", "B2", k=1.0))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize("k", [-1.0, float("nan"), float("inf")])
    def test_invalid_rate_constant_values(self, k: float) -> None:
        with pytest.raises(ValueError):
            FirstOrderReaction("A", "B", k=k)

    @pytest.mark.parametrize("k", ["1.0", True, None])
    def test_invalid_rate_constant_types(self, k: object) -> None:
        with pytest.raises(TypeError):
            FirstOrderReaction("A", "B", k=k)  # type: ignore[arg-type]

    def test_invalid_species_name(self) -> None:
        with pytest.raises(ValueError):
            FirstOrderReaction("", "B", k=1.0)

    def test_invalid_coefficients(self) -> None:
        with pytest.raises(ValueError):
            MassActionReaction({"A": 0}, {"B": 1}, k_f=1.0)
        with pytest.raises(TypeError):
            MassActionReaction({"A": 1.5}, {"B": 1}, k_f=1.0)  # type: ignore[dict-item]
        with pytest.raises(ValueError):
            MassActionReaction({}, {"B": 1}, k_f=1.0)

    def test_reversible_requires_products(self) -> None:
        with pytest.raises(ValueError):
            MassActionReaction({"A": 1}, {}, k_f=1.0, k_b=1.0)

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            HomogeneousReaction()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------
class TestMassConservation:
    @pytest.mark.parametrize(
        "rxn",
        [
            FirstOrderReaction("A", "B", k=1.0),
            ReversibleReaction("A", "B", k_f=2.0, k_b=1.0),
            CatalyticReaction("B", "A", k=3.0, substrate="Z", product="Y"),
            DisproportionationReaction("B", "A", "C", k=1.0),
        ],
    )
    def test_net_stoichiometry_sums_to_zero(self, rxn: HomogeneousReaction) -> None:
        assert sum(rxn.stoichiometry.values()) == 0
        c = _random_c(list(rxn.species))
        total = sum(rxn.rates(c, None, 0.0).values())
        np.testing.assert_allclose(total, 0.0, atol=1e-12)

    def test_weighted_conservation_second_order_and_dimer(self) -> None:
        # A + B -> C conserves c_A + c_C and c_B + c_C
        rxn = SecondOrderReaction("A", "B", "C", k=2.0)
        rates = rxn.rates(_random_c(["A", "B", "C"]), None, 0.0)
        np.testing.assert_allclose(rates["A"] + rates["C"], 0.0, atol=1e-12)
        np.testing.assert_allclose(rates["B"] + rates["C"], 0.0, atol=1e-12)
        # 2B -> B2 conserves c_B + 2 c_B2
        dimer = DimerizationReaction("B", "B2", k=1.0)
        rates = dimer.rates(_random_c(["B", "B2"]), None, 0.0)
        np.testing.assert_allclose(rates["B"] + 2.0 * rates["B2"], 0.0, atol=1e-12)

    def test_network_conservation_and_stoichiometry_matrix(self) -> None:
        network = ReactionNetwork(
            [
                ReversibleReaction("A", "B", k_f=2.0, k_b=0.3),
                FirstOrderReaction("B", "C", k=1.0),
                DisproportionationReaction("C", "A", "B", k=0.5),
            ]
        )
        nu = network.stoichiometry_matrix
        assert nu.shape == (3, 3)
        np.testing.assert_allclose(nu.sum(axis=0), 0.0)
        rates = network(_random_c(network.species), None, 0.0)
        np.testing.assert_allclose(sum(rates.values()), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# ReactionNetwork
# ---------------------------------------------------------------------------
class TestReactionNetwork:
    def test_rate_summation(self) -> None:
        r1 = FirstOrderReaction("A", "B", k=1.0)
        r2 = SecondOrderReaction("A", "B", "C", k=2.0)
        network = ReactionNetwork([r1, r2])
        c = _random_c(["A", "B", "C"])
        rates = network.evaluate(c, None, 0.0)
        assert network.species == ["A", "B", "C"]
        np.testing.assert_allclose(
            rates["A"], r1.rates(c, None, 0.0)["A"] + r2.rates(c, None, 0.0)["A"]
        )
        np.testing.assert_allclose(rates["B"], c["A"] - 2.0 * c["A"] * c["B"])
        np.testing.assert_allclose(rates["C"], 2.0 * c["A"] * c["B"])

    def test_uninvolved_species_get_zero(self) -> None:
        network = ReactionNetwork(
            [MassActionReaction({"A": 1, "K": 1}, {"B": 1, "K": 1}, k_f=1.0)]
        )
        rates = network(_random_c(["A", "B", "K"]), None, 0.0)
        np.testing.assert_array_equal(rates["K"], 0.0)

    def test_container_interface(self) -> None:
        network = ReactionNetwork()
        assert len(network) == 0
        rxn = FirstOrderReaction("A", "B", k=1.0)
        network.add(rxn)
        assert len(network) == 1
        assert network[0] is rxn
        assert list(network) == [rxn]
        assert network.reactions == [rxn]
        assert "FirstOrderReaction" in repr(network)
        with pytest.raises(TypeError):
            network.add(lambda c, x, t: {})  # type: ignore[arg-type]

    def test_explicit_species_order(self) -> None:
        network = ReactionNetwork(
            [FirstOrderReaction("A", "B", k=1.0)], species=["B", "X", "A"]
        )
        assert network.species == ["B", "X", "A"]
        with pytest.raises(ValueError, match="not present"):
            network.add(FirstOrderReaction("A", "Q", k=1.0))
        with pytest.raises(ValueError, match="twice"):
            ReactionNetwork(species=["A", "A"])

    def test_empty_network(self) -> None:
        network = ReactionNetwork()
        assert network({"A": np.ones(3)}, None, 0.0) == {}
        assert network.stoichiometry_matrix.shape == (0, 0)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
class TestJacobian:
    @pytest.mark.parametrize("rxn", ALL_REACTIONS, ids=repr)
    def test_analytical_vs_numerical_single_reaction(
        self, rxn: HomogeneousReaction
    ) -> None:
        network = ReactionNetwork([rxn])
        c = _random_c(network.species)
        local = network.local_jacobian(c, None, 0.0)
        for node in (0, N // 2, N - 1):
            numerical = _numerical_local_jacobian(network, c, node)
            np.testing.assert_allclose(
                local[:, :, node], numerical, rtol=1e-5, atol=1e-6
            )

    def test_analytical_vs_numerical_network(self) -> None:
        network = ReactionNetwork(ALL_REACTIONS)
        c = _random_c(network.species)
        local = network.local_jacobian(c, None, 0.0)
        for node in range(N):
            numerical = _numerical_local_jacobian(network, c, node)
            np.testing.assert_allclose(
                local[:, :, node], numerical, rtol=1e-5, atol=1e-6
            )

    def test_full_jacobian_vs_flattened_finite_difference(self) -> None:
        network = ReactionNetwork(
            [
                ReversibleReaction("A", "B", k_f=2.0, k_b=0.5),
                DimerizationReaction("B", "C", k=1.5),
            ]
        )
        names = network.species
        c = _random_c(names)
        y0 = np.concatenate([c[s] for s in names])

        def f(y: np.ndarray) -> np.ndarray:
            c_y = {s: y[i * N : (i + 1) * N] for i, s in enumerate(names)}
            rates = network(c_y, None, 0.0)
            return np.concatenate([rates[s] for s in names])

        numerical = approx_fprime(y0, f, 1e-7)
        J_sparse = network.jacobian(c, None, 0.0)
        J_dense = network.jacobian(c, None, 0.0, sparse=False)
        assert J_sparse.shape == (3 * N, 3 * N)
        assert isinstance(J_dense, np.ndarray)
        np.testing.assert_allclose(J_sparse.toarray(), J_dense)
        np.testing.assert_allclose(J_dense, numerical, rtol=1e-5, atol=1e-6)
        # Block structure: each block is diagonal -> nnz <= n_blocks * N
        assert J_sparse.nnz <= 9 * N

    def test_jacobian_species_ordering(self) -> None:
        network = ReactionNetwork([FirstOrderReaction("A", "B", k=2.0)])
        c = _random_c(["A", "B"])
        local = network.local_jacobian(c, None, 0.0, species=["B", "A", "X"])
        np.testing.assert_allclose(local[0, 1], 2.0)  # dR_B/dc_A
        np.testing.assert_allclose(local[1, 1], -2.0)  # dR_A/dc_A
        np.testing.assert_allclose(local[2], 0.0)
        with pytest.raises(KeyError):
            network.local_jacobian(c, None, 0.0, species=["A"])

    def test_jacobian_requires_1d(self) -> None:
        network = ReactionNetwork([FirstOrderReaction("A", "B", k=2.0)])
        c = {"A": np.ones((2, 3)), "B": np.ones((2, 3))}
        with pytest.raises(ValueError, match="1D"):
            network.jacobian(c, None, 0.0)

    def test_jacobian_scalar_profiles(self) -> None:
        network = ReactionNetwork([FirstOrderReaction("A", "B", k=2.0)])
        J = network.jacobian({"A": 1.0, "B": 0.0}, None, 0.0, sparse=False)
        np.testing.assert_allclose(J, [[-2.0, 0.0], [2.0, 0.0]])

    def test_empty_jacobian(self) -> None:
        network = ReactionNetwork(
            [FirstOrderReaction("A", "B", k=0.0)], species=["A", "B"]
        )
        J = network.jacobian({"A": np.ones(3), "B": np.ones(3)}, None, 0.0)
        assert J.shape == (6, 6)
        assert J.nnz == 0

    def test_mass_action_jacobian_at_zero_concentration(self) -> None:
        rxn = DimerizationReaction("B", "B2", k=3.0)
        jac = rxn.jacobian({"B": np.zeros(3), "B2": np.zeros(3)}, None, 0.0)
        np.testing.assert_allclose(jac[("B", "B")], 0.0)
        assert np.all(np.isfinite(jac[("B", "B")]))


# ---------------------------------------------------------------------------
# User-defined reactions
# ---------------------------------------------------------------------------
def _mm_rate(c, x, t):  # type: ignore[no-untyped-def]
    return 2.0 * c["S"] / (0.5 + c["S"])


def _mm_drate(c, x, t):  # type: ignore[no-untyped-def]
    return 2.0 * 0.5 / (0.5 + c["S"]) ** 2


class TestCustomReaction:
    def test_rates(self) -> None:
        rxn = CustomReaction({"S": -1, "P": 1}, _mm_rate, name="MM")
        c = _random_c(["S", "P"])
        rates = rxn.rates(c, None, 0.0)
        np.testing.assert_allclose(rates["S"], -_mm_rate(c, None, 0.0))
        np.testing.assert_allclose(rates["P"], _mm_rate(c, None, 0.0))
        assert rxn.species == ("S", "P")
        assert "MM" in repr(rxn)

    def test_finite_difference_jacobian(self) -> None:
        rxn = CustomReaction({"S": -1, "P": 1}, _mm_rate)
        c = _random_c(["S", "P"])
        jac = rxn.jacobian(c, None, 0.0)
        np.testing.assert_allclose(jac[("P", "S")], _mm_drate(c, None, 0.0), rtol=1e-7)
        np.testing.assert_allclose(jac[("S", "S")], -_mm_drate(c, None, 0.0), rtol=1e-7)
        np.testing.assert_allclose(jac[("S", "P")], 0.0, atol=1e-10)

    def test_analytical_derivatives_used(self) -> None:
        calls = []

        def drate(c, x, t):  # type: ignore[no-untyped-def]
            calls.append(1)
            return _mm_drate(c, x, t)

        rxn = CustomReaction(
            {"S": -1, "P": 1},
            _mm_rate,
            rate_derivatives={"S": drate},
            dependencies=["S"],
        )
        c = _random_c(["S", "P"])
        jac = rxn.jacobian(c, None, 0.0)
        assert calls
        assert set(jac) == {("S", "S"), ("P", "S")}
        np.testing.assert_allclose(jac[("P", "S")], _mm_drate(c, None, 0.0))

    def test_extra_dependency_catalyst(self) -> None:
        # Enzyme E catalyses S -> P but is not consumed
        rxn = CustomReaction(
            {"S": -1, "P": 1},
            lambda c, x, t: 3.0 * c["E"] * c["S"],
            dependencies=["S", "E"],
        )
        assert rxn.species == ("S", "P", "E")
        network = ReactionNetwork([rxn])
        c = _random_c(network.species)
        local = network.local_jacobian(c, None, 0.0)
        for node in range(N):
            numerical = _numerical_local_jacobian(network, c, node)
            np.testing.assert_allclose(
                local[:, :, node], numerical, rtol=1e-5, atol=1e-6
            )

    def test_time_and_space_dependent_rate(self) -> None:
        x = np.linspace(0.0, 1.0, N)
        rxn = CustomReaction({"A": -1}, lambda c, x, t: (1.0 + x) * np.exp(-t) * c["A"])
        c = {"A": np.ones(N)}
        np.testing.assert_allclose(rxn.rates(c, x, 0.0)["A"], -(1.0 + x))
        np.testing.assert_allclose(
            rxn.jacobian(c, x, 1.0)[("A", "A")], -(1.0 + x) * np.exp(-1.0), rtol=1e-7
        )

    def test_constant_rate_broadcast(self) -> None:
        rxn = CustomReaction({"P": 1}, lambda c, x, t: 0.5, dependencies=[])
        rates = rxn.rates({"P": np.zeros(N)}, None, 0.0)
        np.testing.assert_allclose(rates["P"], np.full(N, 0.5))
        assert rxn.jacobian({"P": np.zeros(N)}, None, 0.0) == {}

    def test_fractional_stoichiometry(self) -> None:
        rxn = CustomReaction({"A": -0.5, "B": 1.0}, lambda c, x, t: c["A"])
        rates = rxn.rates({"A": np.ones(2), "B": np.zeros(2)}, None, 0.0)
        np.testing.assert_allclose(rates["A"], -0.5)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            CustomReaction({}, _mm_rate)
        with pytest.raises(ValueError):
            CustomReaction({"S": 0}, _mm_rate)
        with pytest.raises(TypeError):
            CustomReaction({"S": -1}, "not callable")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            CustomReaction({"S": "x"}, _mm_rate)  # type: ignore[dict-item]
        with pytest.raises(ValueError):
            CustomReaction({"S": float("nan")}, _mm_rate)
        with pytest.raises(TypeError):
            CustomReaction(
                {"S": -1},
                _mm_rate,
                rate_derivatives={"S": 1.0},  # type: ignore[dict-item]
            )
        with pytest.raises(ValueError, match="dependencies"):
            CustomReaction(
                {"S": -1}, _mm_rate, rate_derivatives={"S": _mm_drate}, dependencies=[]
            )


class TestSubclassing:
    class Autocatalytic(HomogeneousReaction):
        """A + B -> 2B with r = k cA cB, analytical derivatives."""

        def __init__(self, k: float) -> None:
            self.k = k

        @property
        def stoichiometry(self) -> dict[str, int]:
            return {"A": -1, "B": 1}

        def rate(self, c, x, t):  # type: ignore[no-untyped-def]
            return self.k * np.asarray(c["A"]) * np.asarray(c["B"])

    def test_subclass_default_fd_jacobian(self) -> None:
        rxn = self.Autocatalytic(k=2.0)
        network = ReactionNetwork([rxn])
        c = _random_c(["A", "B"])
        local = network.local_jacobian(c, None, 0.0)
        np.testing.assert_allclose(local[0, 0], -2.0 * c["B"], rtol=1e-7)
        np.testing.assert_allclose(local[0, 1], -2.0 * c["A"], rtol=1e-7)
        np.testing.assert_allclose(local[1, 0], 2.0 * c["B"], rtol=1e-7)


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------
class TestSolverIntegration:
    def _problem(self, reactions) -> DiffusionProblem:  # type: ignore[no-untyped-def]
        grid = np.linspace(0.0, 1.0, 21)
        return DiffusionProblem(
            grid=grid,
            diffusivity={"A": 1e-3, "B": 1e-3},
            boundary_conditions={
                "A": (NeumannBC(0.0), NeumannBC(0.0)),
                "B": (NeumannBC(0.0), NeumannBC(0.0)),
            },
            initial_conditions={"A": 1.0, "B": 0.0},
            reactions=reactions,
        )

    @pytest.mark.parametrize("solver_name", ["scipy_ivp", "crank_nicolson"])
    def test_first_order_decay_matches_exponential(self, solver_name: str) -> None:
        k = 1.5
        network = ReactionNetwork([FirstOrderReaction("A", "B", k=k)])
        t_eval = np.linspace(0.0, 1.0, 11)
        solver = get_solver(solver_name)
        kwargs = {"t_eval": t_eval}
        if solver_name != "scipy_ivp":
            kwargs["dt"] = 1e-3
        result = solver.solve(self._problem(network), t_span=(0.0, 1.0), **kwargs)
        assert result.success
        expected = np.exp(-k * result.t)[:, None]
        np.testing.assert_allclose(
            result["A"], np.broadcast_to(expected, result["A"].shape), rtol=5e-3
        )
        np.testing.assert_allclose(result["A"] + result["B"], 1.0, rtol=1e-6)

    def test_reversible_reaches_equilibrium(self) -> None:
        network = ReactionNetwork([ReversibleReaction("A", "B", k_f=4.0, k_b=1.0)])
        result = get_solver("scipy_ivp").solve(
            self._problem(network), t_span=(0.0, 5.0)
        )
        assert result.success
        np.testing.assert_allclose(result["A"][-1], 0.2, rtol=1e-4)
        np.testing.assert_allclose(result["B"][-1], 0.8, rtol=1e-4)

    def test_custom_reaction_in_solver(self) -> None:
        rxn = CustomReaction({"A": -1, "B": 1}, lambda c, x, t: 1.5 * c["A"])
        result = get_solver("scipy_ivp").solve(self._problem(rxn), t_span=(0.0, 1.0))
        assert result.success
        np.testing.assert_allclose(result["A"][-1], np.exp(-1.5), rtol=1e-4)


def test_top_level_exports() -> None:
    for name in [
        "HomogeneousReaction",
        "CustomReaction",
        "MassActionReaction",
        "FirstOrderReaction",
        "ReversibleReaction",
        "SecondOrderReaction",
        "CatalyticReaction",
        "DimerizationReaction",
        "DisproportionationReaction",
        "ReactionNetwork",
    ]:
        assert hasattr(sp, name)
        assert name in sp.__all__
        assert name in sp.reactions.__all__
