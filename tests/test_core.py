"""
Test Suite: tests.test_core
Verifies the correctness of spatial diffusion operator construction and
PDE assembler functionality.
"""

import numpy as np

from softpotato.core.pde_builder import PDEAssembler
from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism
from softpotato.mechanism.steps import ElectronTransferStep


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
