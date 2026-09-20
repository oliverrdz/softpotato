"""
Unit tests for numerical PDE solvers and explicit finite difference methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from softpotato.core import ElectrochemicalReaction, Mechanism, Species
from softpotato.geometry import PlanarElectrode, UniformGrid
from softpotato.kinetics import ButlerVolmer
from softpotato.simulate import (
    SimulationResult,
    Solver,
    explicit_diffuse_step,
)
from softpotato.simulate.efd import update_surface_concentrations
from softpotato.techniques import CyclicVoltammetry


def test_explicit_diffuse_step_vectorized() -> None:
    """Tests explicit Euler diffusion step on a 1D concentration profile."""
    # A localized concentration peak in the interior
    c = np.zeros(21)
    c[10] = 1.0
    dx = 0.01
    dt = 0.001
    D = 1e-5
    # lambda = D * dt / dx^2 = 1e-5 * 1e-3 / 1e-4 = 1e-4 <= 0.5
    c_next = explicit_diffuse_step(c, D=D, dt=dt, dx=dx)

    assert len(c_next) == 21
    # Peak at node 10 should decrease, adjacent nodes should increase
    assert c_next[10] < 1.0
    assert c_next[9] > 0.0
    assert c_next[11] > 0.0
    # Mass conservation in interior
    assert np.isclose(np.sum(c_next), np.sum(c))


def test_update_surface_concentrations() -> None:
    """Tests boundary condition surface concentration update."""
    c_O_0, c_R_0, flux = update_surface_concentrations(
        c_O_interior=1e-6,
        c_R_interior=0.0,
        D_O=1e-5,
        D_R=1e-5,
        dx=1e-4,
        kf=0.1,
        kb=0.0,
    )
    assert c_O_0 >= 0.0
    assert c_R_0 >= 0.0
    assert flux > 0.0
    # Reduction consumes O, so c_O_0 < c_O_interior
    assert c_O_0 < 1e-6
    # Generates R at the surface, so c_R_0 > 0
    assert c_R_0 > 0.0


def test_solver_initialization_and_validation() -> None:
    """Tests Solver initialization and parameter validation."""
    O = Species(name="O", D=1e-5, c_bulk=1e-6)
    R = Species(name="R", D=1e-5, c_bulk=0.0)
    bv = ButlerVolmer(k0=1e-3, alpha=0.5)
    rxn = ElectrochemicalReaction(
        reactants=[O], products=[R], n_electrons=1, E0=0.0, kinetics=bv
    )
    mech = Mechanism([rxn])

    grid = UniformGrid(x_max=0.05, nodes=100)
    electrode = PlanarElectrode(area=0.0707, grid=grid)
    cv = CyclicVoltammetry(E_initial=0.2, E_vertex1=-0.2, scan_rate=0.1, dE=0.005)

    solver = Solver(mechanism=mech, electrode=electrode, technique=cv, method="EFD")
    assert solver.mechanism is mech
    assert solver.electrode is electrode
    assert solver.technique is cv
    assert solver.method == "EFD"

    # Unsupported method
    with pytest.raises(NotImplementedError, match="Solver method 'IFD'"):
        Solver(mechanism=mech, electrode=electrode, technique=cv, method="IFD")

    # Invalid input types
    with pytest.raises(TypeError, match="Mechanism instance"):
        Solver(mechanism=None, electrode=electrode, technique=cv)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Electrode instance"):
        Solver(mechanism=mech, electrode=None, technique=cv)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Technique instance"):
        Solver(mechanism=mech, electrode=electrode, technique=None)  # type: ignore[arg-type]


def test_solver_cfl_handling() -> None:
    """Tests CFL stability checking and automatic sub-stepping."""
    O = Species(name="O", D=1e-5, c_bulk=1e-6)
    R = Species(name="R", D=1e-5, c_bulk=0.0)
    bv = ButlerVolmer(k0=1e-3, alpha=0.5)
    rxn = ElectrochemicalReaction(
        reactants=[O], products=[R], n_electrons=1, E0=0.0, kinetics=bv
    )
    mech = Mechanism([rxn])

    # Choose parameters that cause lambda > 0.5:
    # dx = 0.01 / 100 = 1e-4 => dx^2 = 1e-8
    # dt = dE / scan_rate = 0.01 / 0.1 = 0.1 s
    # lambda = 1e-5 * 0.1 / 1e-8 = 100 >> 0.5
    grid = UniformGrid(x_max=0.01, nodes=101)
    electrode = PlanarElectrode(area=0.0707, grid=grid)
    cv = CyclicVoltammetry(E_initial=0.1, E_vertex1=-0.1, scan_rate=0.1, dE=0.01)

    # With auto_substep=False: should raise ValueError
    solver_strict = Solver(mech, electrode, cv, auto_substep=False)
    with pytest.raises(ValueError, match="CFL stability condition violated"):
        solver_strict.run()

    # With auto_substep=True: should emit UserWarning and complete
    solver_auto = Solver(mech, electrode, cv, auto_substep=True)
    with pytest.warns(UserWarning, match="CFL stability condition exceeded"):
        res = solver_auto.run()

    assert isinstance(res, SimulationResult)
    assert len(res.current) == len(cv.potential)


def test_simulation_result_properties_and_plot() -> None:
    """Tests SimulationResult data attributes and plotting."""
    O = Species(name="O", D=1e-5, c_bulk=1e-6)
    R = Species(name="R", D=1e-5, c_bulk=0.0)
    bv = ButlerVolmer(k0=1e-3, alpha=0.5)
    rxn = ElectrochemicalReaction(
        reactants=[O], products=[R], n_electrons=1, E0=0.0, kinetics=bv
    )
    mech = Mechanism([rxn])

    grid = UniformGrid(x_max=0.05, nodes=100)
    electrode = PlanarElectrode(area=0.0707, grid=grid)
    cv = CyclicVoltammetry(E_initial=0.2, E_vertex1=-0.2, scan_rate=0.5, dE=0.005)

    solver = Solver(mech, electrode, cv)
    res = solver.run()

    assert isinstance(res, SimulationResult)
    assert len(res.time) == len(cv.time)
    assert len(res.potential) == len(cv.potential)
    assert len(res.current) == len(cv.time)
    assert "O" in res.species_profiles
    assert "R" in res.species_profiles

    # Test plot method without opening display
    ax = res.plot(show=False)
    assert ax is not None
    assert ax.get_xlabel() == "Potential / V"
    assert ax.get_ylabel() == "Current / µA"
