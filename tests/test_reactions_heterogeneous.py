"""Unit tests for softpotato.reactions.heterogeneous."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

import softpotato as sp
from softpotato.analytical import kinetics as ak
from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
from softpotato.reactions import (
    ButlerVolmer,
    HeterogeneousKinetics,
    NernstianEquilibrium,
    Species,
    SpeciesCollection,
    Tafel,
    TafelKinetics,
    create_electrochemical_bcs,
)
from softpotato.reactions import heterogeneous as het_mod
from softpotato.solver import DiffusionProblem, get_solver


class TestExportsAndStructure:
    """Verify public exports, aliases, and absence of MarcusHushChidsey."""

    def test_exports_and_aliases(self) -> None:
        assert sp.HeterogeneousKinetics is HeterogeneousKinetics
        assert sp.ButlerVolmer is ButlerVolmer
        assert sp.TafelKinetics is TafelKinetics
        assert sp.Tafel is TafelKinetics
        assert Tafel is TafelKinetics
        assert sp.NernstianEquilibrium is NernstianEquilibrium
        assert sp.create_electrochemical_bcs is create_electrochemical_bcs

        assert sp.reactions.ButlerVolmer is ButlerVolmer
        assert sp.reactions.TafelKinetics is TafelKinetics
        assert sp.reactions.NernstianEquilibrium is NernstianEquilibrium
        assert sp.reactions.create_electrochemical_bcs is create_electrochemical_bcs

    def test_marcus_hush_chidsey_not_included(self) -> None:
        assert not hasattr(het_mod, "MarcusHushChidsey")
        assert not hasattr(sp.reactions, "MarcusHushChidsey")
        assert not hasattr(sp, "MarcusHushChidsey")

    def test_abc_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            HeterogeneousKinetics()  # type: ignore[abstract]

    def test_custom_subclass_default_flux_and_current_density(self) -> None:
        class ConstantKinetics(HeterogeneousKinetics):
            def __init__(self, kf: float, kb: float) -> None:
                self.E0 = 0.0
                self.n = 2.0
                self.T = STANDARD_TEMPERATURE
                self.R = GAS_CONSTANT
                self.F = FARADAY
                self._kf = kf
                self._kb = kb

            def rate_constants(self, E: float | ArrayLike) -> tuple[float, float]:
                return self._kf, self._kb

        kin = ConstantKinetics(kf=2.0, kb=0.5)
        flux = kin.flux(0.0, c_ox=3.0, c_red=4.0)
        assert flux == pytest.approx(2.0 * 3.0 - 0.5 * 4.0)
        assert kin.current_density(
            0.0, 3.0, 4.0, anodic_positive=True
        ) == pytest.approx(-2.0 * FARADAY * flux)
        assert kin.current_density(
            0.0, 3.0, 4.0, anodic_positive=False
        ) == pytest.approx(2.0 * FARADAY * flux)


class TestButlerVolmer:
    """Tests for ButlerVolmer heterogeneous kinetics."""

    def test_zero_overpotential_equilibrium(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.35, E0=0.25, n=1)
        kf, kb = bv.rate_constants(0.25)
        assert isinstance(kf, float)
        assert isinstance(kb, float)
        assert kf == pytest.approx(1e-4)
        assert kb == pytest.approx(1e-4)
        assert bv.flux(0.25, c_ox=2.0, c_red=2.0) == pytest.approx(0.0, abs=1e-18)
        assert bv.current_density(0.25, c_ox=2.0, c_red=2.0) == pytest.approx(
            0.0, abs=1e-15
        )

    def test_detailed_balance_matches_nernst_ratio(self) -> None:
        bv = ButlerVolmer(k0=5e-3, alpha=0.4, E0=0.1, n=2, T=310.0)
        potentials = np.linspace(-0.2, 0.4, 25)
        kf, kb = bv.rate_constants(potentials)
        expected_ratio = ak.nernst_ratio(potentials, E0=0.1, n=2, T=310.0)
        np.testing.assert_allclose(kb / kf, expected_ratio, rtol=1e-12)

    def test_matches_analytical_butler_volmer_and_exchange_current(self) -> None:
        bv = ButlerVolmer(k0=2e-3, alpha=0.3, E0=0.15, n=2, T=305.0)
        assert bv.exchange_current_density(c_ox=1.5, c_red=0.5) == pytest.approx(
            ak.exchange_current_density(
                2e-3, c_ox=1.5, c_red=0.5, alpha=0.3, n=2, F=FARADAY
            )
        )

        E_vals = np.linspace(-0.1, 0.4, 20)
        j_bv = bv.current_density(E_vals, c_ox=1.2, c_red=0.8)
        j0_unit = ak.exchange_current_density(2e-3, alpha=0.3, n=2)
        j_expected = ak.butler_volmer_current_density(
            E_vals - 0.15,
            j0=j0_unit,
            alpha=0.3,
            n=2,
            T=305.0,
            c_ox_surf=1.2,
            c_red_surf=0.8,
        )
        np.testing.assert_allclose(j_bv, j_expected, rtol=1e-12)

        # Polarographic convention reverses sign
        j_polar = bv.current_density(E_vals, c_ox=1.2, c_red=0.8, anodic_positive=False)
        np.testing.assert_allclose(j_polar, -j_bv, rtol=1e-12)

    def test_scalar_E_with_array_concentrations(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
        c_ox = np.array([0.0, 0.5, 1.0])
        c_red = np.array([1.0, 0.5, 0.0])
        j_arr = bv.current_density(0.1, c_ox, c_red)
        flux_arr = bv.flux(0.1, c_ox, c_red)
        assert isinstance(j_arr, np.ndarray)
        assert isinstance(flux_arr, np.ndarray)
        assert j_arr.shape == (3,)
        np.testing.assert_allclose(j_arr, -FARADAY * flux_arr, rtol=1e-12)

    def test_high_overpotential_tafel_limiting_slopes(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.4, E0=0.0, n=1, T=STANDARD_TEMPERATURE)
        ln10 = np.log(10.0)

        # Cathodic high-overpotential slope: d(log10|j|)/dE = -alpha * n * F / (ln10 * R * T)
        e_cath = np.linspace(-0.4, -0.25, 50)
        j_cath = np.abs(bv.current_density(e_cath, c_ox=1.0, c_red=1.0))
        slope_cath = np.polyfit(e_cath, np.log10(j_cath), 1)[0]
        expected_cath = -(0.4 * 1 * FARADAY) / (
            ln10 * GAS_CONSTANT * STANDARD_TEMPERATURE
        )
        assert slope_cath == pytest.approx(expected_cath, rel=1e-3)
        assert bv.tafel_slope("cathodic", signed=True) == pytest.approx(
            1.0 / expected_cath, rel=1e-12
        )

        # Anodic high-overpotential slope: d(log10|j|)/dE = +(1 - alpha) * n * F / (ln10 * R * T)
        e_anod = np.linspace(0.25, 0.4, 50)
        j_anod = np.abs(bv.current_density(e_anod, c_ox=1.0, c_red=1.0))
        slope_anod = np.polyfit(e_anod, np.log10(j_anod), 1)[0]
        expected_anod = (
            (1.0 - 0.4) * 1 * FARADAY / (ln10 * GAS_CONSTANT * STANDARD_TEMPERATURE)
        )
        assert slope_anod == pytest.approx(expected_anod, rel=1e-3)
        assert bv.tafel_slope("anodic", signed=True) == pytest.approx(
            1.0 / expected_anod, rel=1e-12
        )

    def test_delegates_to_analytical_kinetics(self) -> None:
        bv = ButlerVolmer(k0=1e-3, alpha=0.5, E0=0.0, n=1)
        with patch.object(
            ak, "butler_volmer_current_density", wraps=ak.butler_volmer_current_density
        ) as spy_bv:
            bv.current_density(0.1, 1.0, 1.0)
            assert spy_bv.call_count == 1

        with patch.object(ak, "tafel", wraps=ak.tafel) as spy_tafel:
            bv.rate_constants(0.1)
            assert spy_tafel.call_count == 2

    def test_immutability_and_validation(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.5)
        with pytest.raises(FrozenInstanceError):
            bv.k0 = 1e-3  # type: ignore[misc]

        for bad_k0 in [0.0, -1e-4, float("nan"), float("inf")]:
            with pytest.raises(ValueError, match="k0"):
                ButlerVolmer(k0=bad_k0)

        for bad_alpha in [0.0, 1.0, -0.1, 1.2, float("nan")]:
            with pytest.raises(ValueError, match="alpha"):
                ButlerVolmer(k0=1e-4, alpha=bad_alpha)

        with pytest.raises(TypeError, match="k0"):
            ButlerVolmer(k0=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n"):
            ButlerVolmer(k0=1e-4, n=0)
        with pytest.raises(ValueError, match="T"):
            ButlerVolmer(k0=1e-4, T=-10.0)


class TestTafelKinetics:
    """Tests for TafelKinetics heterogeneous charge-transfer model."""

    def test_cathodic_branch(self) -> None:
        tk = TafelKinetics(k0=1e-4, alpha=0.4, E0=0.1, n=1, branch="cathodic")
        E = np.array([-0.3, -0.1, 0.1, 0.2])
        kf, kb = tk.rate_constants(E)
        np.testing.assert_array_equal(kb, np.zeros_like(E))
        expected_kf = ak.tafel(
            E - 0.1, i0=1e-4, alpha=0.4, branch="cathodic", n=1, anodic_positive=False
        )
        np.testing.assert_allclose(kf, expected_kf, rtol=1e-12)

        j = tk.current_density(E, c_ox=2.0, c_red=5.0)
        # c_red has no effect on purely cathodic Tafel branch
        np.testing.assert_allclose(j, -FARADAY * kf * 2.0, rtol=1e-12)

    def test_anodic_branch(self) -> None:
        tk = TafelKinetics(k0=1e-4, alpha=0.4, E0=0.1, n=2, branch="anodic")
        E = np.array([-0.1, 0.1, 0.3, 0.5])
        kf, kb = tk.rate_constants(E)
        np.testing.assert_array_equal(kf, np.zeros_like(E))
        expected_kb = ak.tafel(
            E - 0.1, i0=1e-4, alpha=0.4, branch="anodic", n=2, anodic_positive=True
        )
        np.testing.assert_allclose(kb, expected_kb, rtol=1e-12)

        j = tk.current_density(E, c_ox=5.0, c_red=2.0)
        np.testing.assert_allclose(j, 2 * FARADAY * kb * 2.0, rtol=1e-12)

    def test_auto_branch_matches_analytical_tafel(self) -> None:
        tk = TafelKinetics(k0=1e-3, alpha=0.5, E0=0.2, n=1, branch="auto")
        E = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
        j_tk = tk.current_density(E, c_ox=1.0, c_red=1.0)
        j0 = ak.exchange_current_density(1e-3, c_ox=1.0, c_red=1.0, alpha=0.5, n=1)
        j_expected = ak.tafel(E - 0.2, i0=j0, alpha=0.5, branch="auto", n=1)
        np.testing.assert_allclose(j_tk, j_expected, rtol=1e-12)

        # Scalar switching
        kf_neg, kb_neg = tk.rate_constants(0.0)
        assert kf_neg > 0.0 and kb_neg == 0.0
        kf_pos, kb_pos = tk.rate_constants(0.4)
        assert kf_pos == 0.0 and kb_pos > 0.0

    def test_butler_volmer_converges_to_tafel_at_high_overpotential(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.45, E0=0.0, n=1)
        tk = TafelKinetics(k0=1e-4, alpha=0.45, E0=0.0, n=1, branch="auto")

        e_high = np.array([-0.3, -0.25, 0.25, 0.3])
        j_bv = bv.current_density(e_high, c_ox=1.0, c_red=1.0)
        j_tk = tk.current_density(e_high, c_ox=1.0, c_red=1.0)
        np.testing.assert_allclose(j_bv, j_tk, rtol=1e-4)

    def test_tafel_slope_and_overpotential_round_trip(self) -> None:
        tk_cath = TafelKinetics(k0=1e-4, alpha=0.5, E0=0.0, n=1, branch="cathodic")
        assert tk_cath.tafel_slope(signed=True) == pytest.approx(
            ak.tafel_slope(alpha=0.5, n=1, branch="cathodic", signed=True)
        )

        eta_target = -0.2
        j_val = tk_cath.current_density(eta_target, c_ox=2.0, c_red=0.0)
        eta_recovered = tk_cath.overpotential_for_current(j_val, c_ox=2.0)
        assert eta_recovered == pytest.approx(eta_target, rel=1e-12)

        tk_auto = TafelKinetics(k0=1e-4, alpha=0.5, E0=0.0, n=1, branch="auto")
        with pytest.raises(ValueError, match="explicit branch"):
            tk_auto.tafel_slope()
        with pytest.raises(ValueError, match="explicit branch"):
            tk_auto.overpotential_for_current(1.0)

    def test_invalid_branch_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid branch"):
            TafelKinetics(k0=1e-4, branch="invalid")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="branch"):
            TafelKinetics(k0=1e-4, branch=123)  # type: ignore[arg-type]


class TestNernstianEquilibrium:
    """Tests for NernstianEquilibrium helper."""

    def test_delegates_to_analytical_nernst_functions(self) -> None:
        eq = NernstianEquilibrium(E0=0.2, n=1, T=STANDARD_TEMPERATURE)
        expected_e = ak.nernst(10.0, 1.0, E0=0.2, n=1)

        with patch.object(ak, "nernst", wraps=ak.nernst) as spy_nernst:
            assert eq.potential(10.0, 1.0) == pytest.approx(expected_e)
            assert spy_nernst.call_count == 1

        with patch.object(ak, "nernst_ratio", wraps=ak.nernst_ratio) as spy_ratio:
            assert eq.concentration_ratio(0.2) == pytest.approx(1.0)
            assert spy_ratio.call_count == 1

        with patch.object(
            ak,
            "nernst_equilibrium_concentrations",
            wraps=ak.nernst_equilibrium_concentrations,
        ) as spy_eq:
            c_ox, c_red = eq.surface_concentrations(0.2, c_total=2.0)
            assert c_ox == pytest.approx(1.0)
            assert c_red == pytest.approx(1.0)
            assert spy_eq.call_count == 1

    def test_unequal_diffusivities_surface_concentrations(self) -> None:
        eq = NernstianEquilibrium(E0=0.1, n=1)
        D_ox = 1e-9
        D_red = 4e-9
        c_ox_bulk = 1.5
        c_red_bulk = 0.5
        E_vals = np.linspace(-0.2, 0.4, 15)

        c_ox_s, c_red_s = eq.surface_concentrations(
            E_vals,
            D_ox=D_ox,
            D_red=D_red,
            c_ox_bulk=c_ox_bulk,
            c_red_bulk=c_red_bulk,
        )

        # Must satisfy Nernst ratio c_ox / c_red = exp(nF(E - E0)/RT)
        np.testing.assert_allclose(
            c_ox_s / c_red_s, eq.concentration_ratio(E_vals), rtol=1e-12
        )
        # Must satisfy semi-infinite flux balance:
        # sqrt(D_ox)*c_ox(0) + sqrt(D_red)*c_red(0) = sqrt(D_ox)*c_ox* + sqrt(D_red)*c_red*
        lhs = np.sqrt(D_ox) * c_ox_s + np.sqrt(D_red) * c_red_s
        rhs = np.sqrt(D_ox) * c_ox_bulk + np.sqrt(D_red) * c_red_bulk
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_extreme_overpotential_overflow_safety(self) -> None:
        eq = NernstianEquilibrium(E0=0.0, n=1)
        c_ox_pos, c_red_pos = eq.surface_concentrations(50.0, c_total=2.0)
        assert c_ox_pos == pytest.approx(2.0)
        assert c_red_pos == pytest.approx(0.0, abs=1e-15)

        c_ox_neg, c_red_neg = eq.surface_concentrations(-50.0, c_total=2.0)
        assert c_ox_neg == pytest.approx(0.0, abs=1e-15)
        assert c_red_neg == pytest.approx(2.0)


class TestCreateElectrochemicalBCs:
    """Tests for create_electrochemical_bcs helper and DiffusionProblem integration."""

    def test_nernstian_bcs_evaluation(self) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-9, bulk_concentration=0.0, n_electrons=-1)
        eq = NernstianEquilibrium(E0=0.0, n=1)

        bcs = create_electrochemical_bcs(
            eq,
            potential=lambda t: 0.1 * t - 0.1,
            ox=ox,
            red=red,
        )
        ox_left, ox_right = bcs["Ox"]
        red_left, red_right = bcs["Red"]

        assert ox_left.btype == "dirichlet"
        assert ox_right.btype == "dirichlet"
        assert red_left.btype == "dirichlet"
        assert red_right.btype == "dirichlet"

        # At t = 1.0, E = 0.0 = E0 => equal surface concentrations (0.5, 0.5)
        assert ox_left.evaluate(1.0) == pytest.approx(0.5)
        assert red_left.evaluate(1.0) == pytest.approx(0.5)
        assert ox_right.evaluate(1.0) == pytest.approx(1.0)
        assert red_right.evaluate(1.0) == pytest.approx(0.0)

        # At t = 2.0, E = +0.1 V => c_ox / c_red = nernst_ratio(0.1)
        c_ox_2 = ox_left.evaluate(2.0)
        c_red_2 = red_left.evaluate(2.0)
        assert c_ox_2 + c_red_2 == pytest.approx(1.0)
        assert c_ox_2 / c_red_2 == pytest.approx(eq.concentration_ratio(0.1))

    @pytest.mark.parametrize(
        "kinetics",
        [
            ButlerVolmer(k0=1e-4, alpha=0.4, E0=0.0, n=1),
            TafelKinetics(k0=1e-4, alpha=0.4, E0=0.0, n=1, branch="cathodic"),
            TafelKinetics(k0=1e-4, alpha=0.4, E0=0.0, n=1, branch="anodic"),
            TafelKinetics(k0=1e-4, alpha=0.4, E0=0.0, n=1, branch="auto"),
        ],
    )
    def test_flux_continuity_for_finite_kinetics(
        self, kinetics: HeterogeneousKinetics
    ) -> None:
        ox = Species("Ox", diffusivity=1e-9, bulk_concentration=1.0)
        red = Species("Red", diffusivity=2.5e-9, bulk_concentration=0.5)

        bcs = create_electrochemical_bcs(
            kinetics,
            potential=lambda t: -0.15 + 0.3 * t,
            ox=ox,
            red=red,
        )
        ox_left, _ = bcs["Ox"]
        red_left, _ = bcs["Red"]
        assert ox_left.btype == "neumann"
        assert red_left.btype == "neumann"

        for t in [0.0, 0.5, 1.0]:
            j_ox = ox.diffusivity * ox_left.evaluate(t)
            j_red = red.diffusivity * red_left.evaluate(t)
            assert j_ox == pytest.approx(-j_red, rel=1e-12, abs=1e-18)

    def test_right_electrode_side_and_neumann_bulk(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
        bcs = create_electrochemical_bcs(
            bv,
            potential=-0.1,
            ox="A",
            red="B",
            D_ox=1e-9,
            D_red=1e-9,
            c_ox_bulk=1.0,
            c_red_bulk=0.0,
            electrode_side="right",
            bulk_bc="neumann",
        )
        a_left, a_right = bcs["A"]
        b_left, b_right = bcs["B"]
        assert a_left.btype == "neumann" and a_left.evaluate(0.0) == 0.0
        assert b_left.btype == "neumann" and b_left.evaluate(0.0) == 0.0
        assert a_right.btype == "neumann"
        assert b_right.btype == "neumann"
        # At right boundary, consumption of A gives dc_A/dx < 0 and dc_B/dx > 0
        assert a_right.evaluate(0.0) < 0.0
        assert b_right.evaluate(0.0) == pytest.approx(-a_right.evaluate(0.0))

    def test_validation_errors(self) -> None:
        bv = ButlerVolmer(k0=1e-4, alpha=0.5)
        with pytest.raises(ValueError, match="D_ox and D_red"):
            create_electrochemical_bcs(bv, potential=0.0, ox="Ox", red="Red")
        with pytest.raises(ValueError, match="distinct names"):
            create_electrochemical_bcs(
                bv, potential=0.0, ox="A", red="A", D_ox=1e-9, D_red=1e-9
            )
        with pytest.raises(ValueError, match="electrode_side"):
            create_electrochemical_bcs(
                bv,
                potential=0.0,
                D_ox=1e-9,
                D_red=1e-9,
                electrode_side="both",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="bulk_bc"):
            create_electrochemical_bcs(
                bv,
                potential=0.0,
                D_ox=1e-9,
                D_red=1e-9,
                bulk_bc="robin",  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="kinetics"):
            create_electrochemical_bcs("not_kinetics", potential=0.0)  # type: ignore[arg-type]

    def test_end_to_end_nernstian_diffusion_simulation(self) -> None:
        ox = Species("Ox", diffusivity=1e-5, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-5, bulk_concentration=0.0, n_electrons=1)
        col = SpeciesCollection([ox, red])

        eq = NernstianEquilibrium(E0=0.0, n=1)
        bcs = create_electrochemical_bcs(eq, potential=-0.3, ox=ox, red=red)

        grid = np.linspace(0.0, 0.05, 51)
        problem = DiffusionProblem(
            grid=grid,
            diffusivity=col.diffusivities,
            boundary_conditions=bcs,
            initial_conditions={"Ox": 1.0, "Red": 0.0},
            species=col.names,
            n_electrons=col.n_electrons,
        )
        solver = get_solver("scipy_ivp", method="Radau")
        result = solver.solve(
            problem, t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 11)
        )

        assert result.success
        # Mass conservation c_Ox + c_Red = 1.0 everywhere (since D_Ox == D_Red)
        np.testing.assert_allclose(result["Ox"] + result["Red"], 1.0, atol=1e-4)
        # Surface concentration ratio matches Nernst equilibrium at t > 0
        c_ox_surf = result["Ox"][-1, 0]
        c_red_surf = result["Red"][-1, 0]
        assert c_ox_surf / c_red_surf == pytest.approx(
            eq.concentration_ratio(-0.3), rel=1e-6
        )

    def test_end_to_end_butler_volmer_neumann_simulation(self) -> None:
        ox = Species("Ox", diffusivity=1e-5, bulk_concentration=1.0, n_electrons=1)
        red = Species("Red", diffusivity=1e-5, bulk_concentration=0.0, n_electrons=1)
        col = SpeciesCollection([ox, red])

        bv = ButlerVolmer(k0=1e-4, alpha=0.5, E0=0.0, n=1)
        bcs = create_electrochemical_bcs(bv, potential=-0.05, ox=ox, red=red)

        grid = np.linspace(0.0, 0.05, 51)
        problem = DiffusionProblem(
            grid=grid,
            diffusivity=col.diffusivities,
            boundary_conditions=bcs,
            initial_conditions={"Ox": 1.0, "Red": 0.0},
            species=col.names,
            n_electrons=col.n_electrons,
        )
        solver = get_solver("scipy_ivp", method="Radau")
        result = solver.solve(
            problem, t_span=(0.0, 0.5), t_eval=np.linspace(0.0, 0.5, 6)
        )

        assert result.success
        # Under cathodic potential (-0.05 V), Ox is consumed and Red is produced at x=0
        assert result["Ox"][-1, 0] < 1.0
        assert result["Red"][-1, 0] > 0.0
        # Flux continuity at electrode surface: J_left_Ox == -J_left_Red
        np.testing.assert_allclose(
            result.fluxes["Ox"], -result.fluxes["Red"], rtol=1e-3, atol=1e-6
        )
