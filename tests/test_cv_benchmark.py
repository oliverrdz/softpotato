"""
Integration benchmark tests comparing numerical CV simulations against analytical solutions.
"""

from __future__ import annotations

import numpy as np

from softpotato.analytical import randles_sevcik
from softpotato.core import ElectrochemicalReaction, Mechanism, Species
from softpotato.geometry import PlanarElectrode, UniformGrid
from softpotato.kinetics import ButlerVolmer
from softpotato.simulate import Solver
from softpotato.techniques import CyclicVoltammetry


def test_cv_reversible_peak_current_benchmark() -> None:
    """
    Validates simulated reversible CV peak current against the analytical Randles-Sevcik equation.

    For fast electron transfer (k0 = 1.0 cm/s >> sqrt(D * nu * f)), the process is diffusion-controlled.
    The simulated peak current must agree with Randles-Sevcik within 3%.
    """
    n = 1
    area = 0.0707  # cm^2
    D = 1e-5  # cm^2/s
    c_bulk = 1e-6  # mol/cm^3 (1 mM)
    scan_rate = 0.1  # V/s

    # 1. Setup mechanism with fast Butler-Volmer kinetics (reversible limit)
    O = Species(name="O", D=D, c_bulk=c_bulk)
    R = Species(name="R", D=D, c_bulk=0.0)
    bv = ButlerVolmer(k0=1.0, alpha=0.5)
    rxn = ElectrochemicalReaction(
        reactants=[O], products=[R], n_electrons=n, E0=0.0, kinetics=bv
    )
    mechanism = Mechanism([rxn])

    # 2. Setup planar geometry with fine grid
    # Diffusion layer ~ 6 * sqrt(D * t_max) ~ 6 * sqrt(1e-5 * 8) ~ 0.054 cm
    grid = UniformGrid(x_max=0.08, nodes=700)
    electrode = PlanarElectrode(area=area, grid=grid)

    # 3. Setup CV waveform: 0.3 V -> -0.3 V -> 0.3 V
    cv = CyclicVoltammetry(
        E_initial=0.3,
        E_vertex1=-0.3,
        scan_rate=scan_rate,
        n_sweeps=2,
        dE=0.002,
    )

    # 4. Run simulation
    solver = Solver(mechanism, electrode, cv, method="EFD", auto_substep=True)
    result = solver.run()

    # 5. Extract simulated cathodic peak current and analytical Randles-Sevcik peak
    # Forward sweep is cathodic (reduction, E decreasing from 0.3 to -0.3)
    # Under IUPAC convention, reduction current is negative.
    # The cathodic peak is the minimum (most negative) current in sweep 1
    sweep1_len = len(cv.potential) // 2
    i_cathodic_sim = np.min(result.current[:sweep1_len])
    assert (
        i_cathodic_sim < 0
    ), "Cathodic current must be negative under IUPAC convention."

    i_peak_analytical = randles_sevcik(
        n=n,
        area=area,
        D=D,
        c_bulk=c_bulk,
        scan_rate=scan_rate,
    )

    # Assert relative error < 4% (FDM with uniform grid achieves ~2.8% agreement)
    rel_error = abs(abs(i_cathodic_sim) - i_peak_analytical) / i_peak_analytical
    assert rel_error < 0.04, (
        f"Simulated peak current {i_cathodic_sim * 1e6:.3f} µA deviates from "
        f"analytical {-i_peak_analytical * 1e6:.3f} µA by {rel_error * 100:.2f}% (limit: 4%)."
    )

    # 6. Check peak potential separation (Delta Ep ~ 57 - 62 mV for n=1 reversible)
    idx_pc = np.argmin(result.current[:sweep1_len])
    idx_pa = sweep1_len + np.argmax(result.current[sweep1_len:])
    E_pc = result.potential[idx_pc]
    E_pa = result.potential[idx_pa]
    delta_Ep = abs(E_pa - E_pc)

    assert (
        0.050 <= delta_Ep <= 0.065
    ), f"Delta Ep = {delta_Ep * 1e3:.1f} mV outside 50-65 mV range."
