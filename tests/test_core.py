"""
Test Suite: tests.test_core
Verifies the correctness of spatial diffusion operator construction and
PDE assembler functionality.
"""

import numpy as np

from softpotato.core.boundary_conditions import BoundaryHandler
from softpotato.core.pde_builder import PDEAssembler
from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism
from softpotato.mechanism.steps import ElectronTransferStep
from softpotato.physics.species import Species


def test_pde_assembler_initialization():
    """Test that the PDEAssembler correctly maps grid dimensions and mechanism species."""
    x_nodes = np.linspace(0.0, 1e-3, 50)
    dx_steps = np.diff(x_nodes)
    grid = Grid(x=x_nodes, dx=dx_steps)

    step = ElectronTransferStep(ox_species="O", red_species="R")
    mechanism = Mechanism(steps=[step])

    assembler = PDEAssembler(grid=grid, mechanism=mechanism)
    assert assembler.n_nodes == 50
    assert "O" in assembler.mechanism.get_species()
    assert "R" in assembler.mechanism.get_species()


def test_diffusion_matrix_shape_and_scaling():
    """Test that the diffusion matrix has correct dimensions and scales with D."""
    x_nodes = np.linspace(0.0, 1e-3, 20)
    dx_steps = np.full(19, 1e-4)
    grid = Grid(x=x_nodes, dx=dx_steps)

    mechanism = Mechanism(steps=[])
    assembler = PDEAssembler(grid=grid, mechanism=mechanism)

    D_val = 1e-9
    diff_matrix = assembler.build_diffusion_matrix(D_val)

    assert diff_matrix.shape == (20, 20)
    # Check interior node scaling
    expected_factor = D_val / (1e-4**2)
    assert np.isclose(diff_matrix[1, 0], expected_factor)
    assert np.isclose(diff_matrix[1, 1], -2.0 * expected_factor)


def test_boundary_handler_initialization():
    """Validates that BoundaryHandler initializes correctly with grid and species."""
    x = np.linspace(0.0, 1e-3, 100)
    dx = np.diff(x)
    grid = Grid(x=x, dx=dx)

    species_dict = {"O": Species(D=1e-9, c0=1.0), "R": Species(D=1e-9, c0=0.0)}

    handler = BoundaryHandler(grid=grid, species_dict=species_dict)
    assert handler.nx == 100
    assert "O" in handler.species_dict
    assert "R" in handler.species_dict


def test_surface_flux_zero_kinetics():
    """Validates default zero-flux behavior when no kinetic evaluator is supplied."""
    x = np.linspace(0.0, 1e-3, 50)
    dx = np.diff(x)
    grid = Grid(x=x, dx=dx)

    handler = BoundaryHandler(grid=grid, species_dict={})
    surface_concs = {"O": 1.0, "R": 0.0}

    fluxes = handler.compute_surface_flux(
        E_potential=0.0, surface_concentrations=surface_concs
    )
    assert fluxes["O"] == 0.0
    assert fluxes["R"] == 0.0
