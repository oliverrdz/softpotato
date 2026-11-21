"""Unit and integration tests for the declarative electrochemical plotting suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.fitting import (
    fit_cottrell,
    fit_koutecky_levich,
    fit_levich,
    fit_randles_sevcik,
    fit_tafel,
)
from softpotato.analytical.hydrodynamics import koutecky_levich, levich
from softpotato.analytical.kinetics import butler_volmer
from softpotato.analytical.step import cottrell
from softpotato.analytical.voltammetry import randles_sevcik
from softpotato.geometry import SphericalGeometry
from softpotato.reactions import ButlerVolmer, NernstianEquilibrium, Species
from softpotato.reactions.mechanisms import ElectrochemicalStep, Mechanism
from softpotato.techniques import (
    Chronoamperometry,
    Chronopotentiometry,
    ConstantSegment,
    CustomWaveform,
    CyclicVoltammetry,
    DifferentialPulseVoltammetry,
    LinearSweepVoltammetry,
    PlotSpec,
    RDESweep,
    RotatingDiskElectrode,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
    analyze_cottrell,
    analyze_randles_sevcik,
    find_peaks,
    plot_cottrell,
    plot_koutecky_levich,
    plot_levich,
    plot_peaks,
    plot_randles_sevcik,
    plot_result,
    plot_summary,
    plot_sweep,
    plot_tafel,
    resolve_plot_kind,
    simulate,
    simulate_sweep,
)


def _reversible_mechanism(
    e0: float = 0.0, d: float = 1e-9, c_ox: float = 1.0
) -> Mechanism:
    ox = Species(
        "Ox",
        diffusivity=d,
        bulk_concentration=c_ox,
        n_electrons=1,
        standard_potential=e0,
    )
    red = Species(
        "Red",
        diffusivity=d,
        bulk_concentration=0.0,
        n_electrons=-1,
        standard_potential=e0,
    )
    return Mechanism(
        species=[ox, red],
        electrochemical_steps=[
            ElectrochemicalStep(ox, red, NernstianEquilibrium(E0=e0, n=1))
        ],
    )


def _bv_mechanism(e0: float = 0.0, k0: float = 1e-4) -> Mechanism:
    ox = Species(
        "Ox",
        diffusivity=1e-9,
        bulk_concentration=1.0,
        n_electrons=1,
        standard_potential=e0,
    )
    red = Species(
        "Red",
        diffusivity=1e-9,
        bulk_concentration=0.0,
        n_electrons=-1,
        standard_potential=e0,
    )
    return Mechanism(
        species=[ox, red],
        electrochemical_steps=[
            ElectrochemicalStep(ox, red, ButlerVolmer(k0=k0, E0=e0, alpha=0.5, n=1))
        ],
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ==============================================================================
# 1. Declarative PlotSpec, available_plots, default_plot & Aliases
# ==============================================================================


class TestDeclarativePlotSpecs:
    def test_cv_available_and_default_plots(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.4, E_vertex1=-0.4, E_vertex2=0.4, scan_rate=0.5
        )
        assert cv.default_plot == "voltammogram"
        assert cv.available_plots == ("voltammogram", "i_vs_t", "E_vs_t", "c_vs_x")
        spec = cv.plot_specs["voltammogram"]
        assert isinstance(spec, PlotSpec)
        assert "$E$" in spec.x_label and "V" in spec.x_label
        assert "$I$" in spec.y_label and "A" in spec.y_label

    def test_lsv_available_and_default_plots(self) -> None:
        lsv = LinearSweepVoltammetry(E_start=0.4, E_end=-0.4, scan_rate=0.5)
        assert lsv.default_plot == "voltammogram"
        assert lsv.available_plots == ("voltammogram", "i_vs_t", "E_vs_t", "c_vs_x")

    def test_ca_available_and_default_plots(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        assert ca.default_plot == "i_vs_t"
        assert ca.available_plots == ("i_vs_t", "Q_vs_t", "E_vs_t", "c_vs_x")
        assert ca.plot_specs["i_vs_t"].title == "Chronoamperogram"

    def test_cp_available_and_default_plots(self) -> None:
        cp = Chronopotentiometry(current=-1e-4, duration=1.0)
        assert cp.default_plot == "E_vs_t"
        assert cp.available_plots == ("E_vs_t", "i_vs_t", "c_vs_x")
        assert cp.plot_specs["E_vs_t"].title == "Chronopotentiogram"

    def test_pulse_techniques_available_plots(self) -> None:
        dpv = DifferentialPulseVoltammetry(
            E_start=0.3,
            E_end=-0.3,
            step_height=0.05,
            pulse_amplitude=-0.05,
            pulse_width=0.05,
            pulse_period=0.2,
        )
        swv = SquareWaveVoltammetry(
            E_start=0.3, E_end=-0.3, step_height=0.05, amplitude=0.025, frequency=5.0
        )
        scv = SampledCurrentVoltammetry(potentials=[0.2, 0.0, -0.2], duration=1.0)

        assert dpv.default_plot == "dpv"
        assert dpv.available_plots == ("dpv", "i_vs_t", "E_vs_t", "c_vs_x")
        assert swv.default_plot == "swv"
        assert swv.available_plots == ("swv", "i_vs_t", "E_vs_t", "c_vs_x")
        assert scv.default_plot == "scv"
        assert scv.available_plots == ("scv", "i_vs_t", "E_vs_t", "c_vs_x")

    def test_rde_delegates_plot_specs_to_waveform(self) -> None:
        rde_lsv = RotatingDiskElectrode(E_start=0.3, E_end=-0.3, scan_rate=0.1, rpm=900)
        assert rde_lsv.default_plot == "voltammogram"
        assert rde_lsv.plot_specs["voltammogram"].title.startswith("RDE ")

        rde_ca = RotatingDiskElectrode(
            Chronoamperometry(E_step=-0.5, duration=1.0), rpm=900
        )
        assert rde_ca.default_plot == "i_vs_t"
        assert "Q_vs_t" in rde_ca.available_plots

    def test_custom_waveform_defaults_by_mode(self) -> None:
        pot = CustomWaveform([ConstantSegment(0.0, 1.0)], mode="potentiostatic")
        galv = CustomWaveform([ConstantSegment(-1e-4, 1.0)], mode="galvanostatic")
        assert pot.default_plot == "i_vs_t"
        assert pot.available_plots == ("i_vs_t", "E_vs_t", "c_vs_x")
        assert galv.default_plot == "E_vs_t"
        assert galv.available_plots == ("E_vs_t", "i_vs_t", "c_vs_x")

    def test_verbose_aliases_resolve_to_concise_keys(self) -> None:
        assert resolve_plot_kind("current_vs_potential") == "voltammogram"
        assert resolve_plot_kind("i_vs_E") == "voltammogram"
        assert resolve_plot_kind("current_vs_time") == "i_vs_t"
        assert resolve_plot_kind("chronoamperogram") == "i_vs_t"
        assert resolve_plot_kind("potential_vs_time") == "E_vs_t"
        assert resolve_plot_kind("chronopotentiogram") == "E_vs_t"
        assert resolve_plot_kind("concentration_profile") == "c_vs_x"
        assert resolve_plot_kind("concentration_vs_distance") == "c_vs_x"
        assert resolve_plot_kind("charge_vs_time") == "Q_vs_t"
        assert resolve_plot_kind("differential_pulse") == "dpv"
        assert resolve_plot_kind("square_wave") == "swv"
        assert resolve_plot_kind("sampled_current") == "scv"

        with pytest.raises(TypeError, match="string"):
            resolve_plot_kind(123)  # type: ignore[arg-type]


# ==============================================================================
# 2. End-to-End TechniqueResult.plot() & Universal Plots
# ==============================================================================


class TestTechniquePlotting:
    def test_cv_default_universal_verbose_and_cycles(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.3, E_vertex1=-0.3, E_vertex2=0.3, scan_rate=1.0, n_cycles=2
        )
        res = simulate(cv, _reversible_mechanism(), mesh_kwargs={"n_points": 35})

        assert res.available_plots == ("voltammogram", "i_vs_t", "E_vs_t", "c_vs_x")
        assert res.default_plot == "voltammogram"

        # Default plot
        _, ax = res.plot()
        assert ax.get_xlabel() == "$E$ / V"
        assert ax.get_ylabel() == "$I$ / A"
        assert len(ax.lines) == 1

        # Verbose alias + per-cycle plotting + peaks overlay
        _, ax2 = res.plot("current_vs_potential", cycle="all", peaks=True)
        assert len(ax2.lines) == 2
        assert len(ax2.collections) >= 1  # peak scatter markers

        # Universal plots via both concise and verbose keys
        _, ax_it = res.plot("current_vs_time")
        assert ax_it.get_xlabel() == "$t$ / s" and ax_it.get_ylabel() == "$I$ / A"

        _, ax_et = res.plot("potential_vs_time")
        assert ax_et.get_xlabel() == "$t$ / s" and ax_et.get_ylabel() == "$E$ / V"

        _, ax_cx = res.plot("concentration_profile", t=0.5, species=["Ox"])
        assert ax_cx.get_xlabel() == "$x$ / m"
        assert len(ax_cx.lines) == 1
        assert ax_cx.lines[0].get_label() == "Ox"

    def test_ca_default_and_chronocoulogram(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        res = simulate(ca, _reversible_mechanism(), mesh_kwargs={"n_points": 35})

        _, ax = res.plot()
        assert ax.get_title() == "Chronoamperogram"
        assert ax.get_xlabel() == "$t$ / s"
        assert ax.get_ylabel() == "$I$ / A"

        _, ax_q = res.plot("charge_vs_time")
        assert ax_q.get_title() == "Chronocoulogram"
        assert ax_q.get_ylabel() == "$Q$ / C"

    def test_cp_default_and_transition_time_marker(self) -> None:
        cp = Chronopotentiometry(current=-5e-4, duration=1.0)
        res = simulate(cp, _reversible_mechanism(), mesh_kwargs={"n_points": 35})

        _, ax = res.plot()
        assert ax.get_title() == "Chronopotentiogram"
        assert ax.get_ylabel() == "$E$ / V"
        if res.transition_time is not None:
            # Main curve + vertical dashed tau line
            assert len(ax.lines) == 2

    def test_dpv_swv_scv_plots(self) -> None:
        mech = _reversible_mechanism()

        dpv = DifferentialPulseVoltammetry(
            E_start=0.2,
            E_end=-0.2,
            step_height=0.05,
            pulse_amplitude=-0.05,
            pulse_width=0.05,
            pulse_period=0.2,
        )
        res_dpv = simulate(dpv, mech, mesh_kwargs={"n_points": 30})
        _, ax_dpv = res_dpv.plot(show_components=True)
        assert len(ax_dpv.lines) == 3

        swv = SquareWaveVoltammetry(
            E_start=0.2, E_end=-0.2, step_height=0.05, amplitude=0.025, frequency=5.0
        )
        res_swv = simulate(swv, mech, mesh_kwargs={"n_points": 30})
        _, ax_swv = res_swv.plot()
        assert len(ax_swv.lines) == 3

        scv = SampledCurrentVoltammetry(
            potentials=[0.2, 0.0, -0.2], duration=0.5, sample_times=[0.2, 0.5]
        )
        res_scv = simulate(scv, mech, mesh_kwargs={"n_points": 30})
        _, ax_scv = res_scv.plot()
        assert len(ax_scv.lines) == 2
        _, ax_scv_it = res_scv.plot("i_vs_t")
        assert len(ax_scv_it.lines) == 3
        _, ax_scv_cx = res_scv.plot("c_vs_x", t=0.25, step=2)
        assert len(ax_scv_cx.lines) == 2

    def test_spherical_geometry_radial_distance_label(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=0.5)
        geom = SphericalGeometry(r0=1e-4)
        res = simulate(
            ca, _reversible_mechanism(), geometry=geom, mesh_kwargs={"n_points": 30}
        )
        _, ax = res.plot("c_vs_x", t=0.25)
        assert ax.get_xlabel() == "$r$ / m"

    def test_multi_time_concentration_profile_and_validation(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        res = simulate(ca, _reversible_mechanism(), mesh_kwargs={"n_points": 30})

        _, ax = res.plot("c_vs_x", t=[0.1, 0.5, 1.0], species="Ox")
        assert len(ax.lines) == 3

        with pytest.raises(ValueError, match="outside simulated range"):
            res.plot("c_vs_x", t=5.0)
        with pytest.raises(KeyError, match="not found"):
            res.plot("c_vs_x", species="Unknown")

    def test_existing_axes_reuse_and_invalid_kind(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        res = simulate(ca, _reversible_mechanism(), mesh_kwargs={"n_points": 30})

        fig, axes = plt.subplots(1, 2)
        returned_fig, returned_ax = res.plot("i_vs_t", ax=axes[0])
        assert returned_fig is fig
        assert returned_ax is axes[0]

        with pytest.raises(ValueError, match="Available plots"):
            res.plot("voltammogram")
        with pytest.raises(TypeError, match="TechniqueResult"):
            plot_result("not_a_result")  # type: ignore[arg-type]


# ==============================================================================
# 3. plot_summary & plot_sweep
# ==============================================================================


class TestSummaryAndSweepPlots:
    def test_plot_summary_layout_for_cv_and_ca(self) -> None:
        mech = _reversible_mechanism()

        cv = CyclicVoltammetry(
            E_start=0.3, E_vertex1=-0.3, E_vertex2=0.3, scan_rate=1.0
        )
        res_cv = simulate(cv, mech, mesh_kwargs={"n_points": 30})
        fig_cv, axes_cv = res_cv.plot_summary(t=0.3)
        assert axes_cv.shape == (2, 2)
        assert fig_cv is not None

        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        res_ca = simulate(ca, mech, mesh_kwargs={"n_points": 30})
        _, axes_ca = plot_summary(res_ca, t=0.5)
        assert axes_ca.shape == (3,)

    def test_plot_sweep_mapping_sequence_and_rde(self) -> None:
        mech = _reversible_mechanism()
        sweep_dict = simulate_sweep(
            lambda v: LinearSweepVoltammetry(E_start=0.3, E_end=-0.3, scan_rate=v),
            mech,
            values=[0.5, 1.0],
            mesh_kwargs={"n_points": 30},
        )
        _, ax_dict = plot_sweep(sweep_dict, legend_title="Scan rate (V/s)")
        assert len(ax_dict.lines) == 2

        _, ax_seq = plot_sweep(list(sweep_dict.values()))
        assert len(ax_seq.lines) == 2

        rde_sweep = RDESweep(E_start=0.3, E_end=-0.3, scan_rate=1.0, rpms=[400, 900])
        rde_res = rde_sweep.simulate(D=1e-9, n_points=30)
        _, ax_rde = plot_sweep(rde_res)
        assert len(ax_rde.lines) == 2


# ==============================================================================
# 4. Diagnostic Plotting Helpers
# ==============================================================================


class TestDiagnosticPlottingHelpers:
    def test_plot_peaks_helper(self) -> None:
        cv = CyclicVoltammetry(
            E_start=0.3, E_vertex1=-0.3, E_vertex2=0.3, scan_rate=1.0
        )
        res = simulate(cv, _reversible_mechanism(), mesh_kwargs={"n_points": 35})
        diag = find_peaks(res)
        _, ax = plot_peaks(res, diag)
        assert len(ax.lines) == 1
        assert len(ax.collections) >= 1

    def test_plot_randles_sevcik_helper(self) -> None:
        mech = _reversible_mechanism()
        sweep = simulate_sweep(
            lambda v: CyclicVoltammetry(
                E_start=0.3, E_vertex1=-0.3, E_vertex2=0.3, scan_rate=v
            ),
            mech,
            values=[0.2, 0.5, 1.0],
            mesh_kwargs={"n_points": 35},
        )
        fit = analyze_randles_sevcik(sweep, n=1, area=1.0, c_bulk=1.0)
        _, ax = plot_randles_sevcik(sweep, fit)
        assert len(ax.collections) == 1
        assert len(ax.lines) == 1

        # Raw arrays path
        v = np.array([0.05, 0.1, 0.2, 0.5])
        ip = randles_sevcik(v, n=1, D=1e-9, area=1.0, c_bulk=1.0)
        fit_raw = fit_randles_sevcik(v, ip, n=1, area=1.0, c_bulk=1.0)
        _, ax_raw = plot_randles_sevcik(v, fit_raw, peak_currents=ip)
        assert len(ax_raw.lines) == 1

    def test_plot_cottrell_helper(self) -> None:
        ca = Chronoamperometry(E_step=-0.5, duration=1.0)
        res = simulate(ca, _reversible_mechanism(), mesh_kwargs={"n_points": 40})
        fit = analyze_cottrell(res, n=1, area=1.0, c_bulk=1.0, t_min=0.1)
        _, ax = plot_cottrell(res, fit, t_min=0.1)
        assert len(ax.collections) == 1
        assert len(ax.lines) == 1

        # Raw arrays path
        t = np.linspace(0.1, 2.0, 20)
        i_vals = cottrell(t, n=1, D=1e-9, c_bulk=1.0, area=1.0)
        fit_raw = fit_cottrell(t, i_vals, n=1, area=1.0, c_bulk=1.0)
        _, ax_raw = plot_cottrell(t, fit_raw, current=i_vals)
        assert len(ax_raw.lines) == 1

    def test_plot_levich_and_koutecky_levich_helpers(self) -> None:
        omegas = np.array([40.0, 90.0, 160.0, 250.0])
        i_l = levich(omegas, n=1, D=1e-9, c_bulk=1.0, area=1.0, nu=1e-6)
        fit_l = fit_levich(omegas, i_l, n=1, area=1.0, c_bulk=1.0, nu=1e-6)
        _, ax_l = plot_levich(omegas, fit_l, currents=i_l)
        assert len(ax_l.collections) == 1
        assert len(ax_l.lines) == 1

        i_kl = koutecky_levich(
            omegas, i_k=1e-2, n=1, D=1e-9, c_bulk=1.0, area=1.0, nu=1e-6
        )
        fit_kl = fit_koutecky_levich(omegas, i_kl, n=1, area=1.0, c_bulk=1.0, nu=1e-6)
        _, ax_kl = plot_koutecky_levich(omegas, fit_kl, currents=i_kl)
        assert len(ax_kl.collections) == 1
        assert len(ax_kl.lines) == 1

    def test_plot_tafel_helper(self) -> None:
        eta = np.linspace(-0.25, -0.10, 25)
        i_vals = butler_volmer(eta, i0=1e-5, alpha=0.5, n=1)
        fit_t = fit_tafel(eta, i_vals, branch="cathodic")
        _, ax = plot_tafel(eta, fit_t, current=i_vals)
        assert len(ax.collections) == 1
        assert len(ax.lines) == 1

    def test_top_level_exports_exist(self) -> None:
        for name in (
            "PlotSpec",
            "PLOT_ALIASES",
            "resolve_plot_kind",
            "plot_result",
            "plot_summary",
            "plot_sweep",
            "plot_voltammogram",
            "plot_current_vs_time",
            "plot_potential_vs_time",
            "plot_concentration_profile",
            "plot_charge_vs_time",
            "plot_dpv",
            "plot_swv",
            "plot_scv",
            "plot_peaks",
            "plot_randles_sevcik",
            "plot_cottrell",
            "plot_levich",
            "plot_koutecky_levich",
            "plot_tafel",
        ):
            assert hasattr(sp, name), f"Missing top-level export: {name}"
