import numpy as np

from softpotato.grid.mesh_generators import planar
from softpotato.mechanism.builder import build
from softpotato.mechanism.steps import ElectronTransferStep
from softpotato.physics.kinetics import kinetics
from softpotato.physics.species import Species
from softpotato.simulator import simulate
from softpotato.technique.sweep import lsv


def dummy_solver(rhs, t_span, y0, t_eval, **kwargs):
    """Mock numerical solver returning steady-state-like linear evaluation for unit testing."""

    class MockSolution:
        def __init__(self, t, y):
            self.t = t
            self.y = y

    n_steps = len(t_eval)
    # Return mock concentration matrix constant over time for validation
    y = np.outer(y0, np.ones(n_steps))
    return MockSolution(t=t_eval, y=y)


def test_simulate_pipeline_execution():
    # 1. Define physical species using strict SI units (m²/s, mol/m³)
    species_dict = {"O": Species(D=1e-9, c0=1.0), "R": Species(D=1e-9, c0=0.0)}

    # 2. Define Butler-Volmer kinetics
    kin = kinetics("BV", {"k0": 1e-3, "alpha": 0.5, "E0": 0.0})

    # 3. Build mechanism
    step = ElectronTransferStep(
        ox_species="O", red_species="R", e_formal=0.0, k0=1e-3, alpha=0.5
    )
    mech = build([step])

    # 4. Generate spatial grid and potential waveform
    grid = planar(x_max=1e-3, nx=10, grid_type="uniform")
    waveform = lsv(E_ini=0.5, E_final=-0.5, scan_rate=0.1, dE=0.01)

    # 5. Execute simulation via simulator orchestrator
    result = simulate(
        waveform=waveform,
        mechanism=mech,
        grid=grid,
        solver=dummy_solver,
        species=species_dict,
        kinetics=kin,
    )

    # Assertions
    assert isinstance(result.t, np.ndarray)
    assert isinstance(result.E, np.ndarray)
    assert isinstance(result.i, np.ndarray)
    assert isinstance(result.x, np.ndarray)

    assert len(result.t) == len(waveform.t)
    assert len(result.E) == len(waveform.t)
    assert len(result.i) == len(waveform.t)
    assert len(result.x) == 10

    assert "O" in result.concentrations
    assert "R" in result.concentrations
    assert result.concentrations["O"].shape == (10, len(waveform.t))
