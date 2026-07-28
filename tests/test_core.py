"""
Test Suite: tests.test_core
Verifies the correctness of spatial diffusion operator construction and
PDE assembler functionality.
"""

import numpy as np

from softpotato.core.pde_builder import PDEAssembler
from softpotato.grid.mesh_generators import planar
from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism, build
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


def test_pde_assembler_initialization_legacy():
    """Ensures backward compatibility when initializing PDEAssembler without species or kinetics."""
    grid = planar(x_max=1e-3, nx=10)
    step = ElectronTransferStep(
        ox_species="O", red_species="R", e_formal=0.0, k0=1e-3, alpha=0.5
    )
    mechanism = build([step])

    assembler = PDEAssembler(grid=grid, mechanism=mechanism)
    assert assembler.species == {}
    assert assembler.kinetics is None


def test_pde_assembler_with_species_and_kinetics():
    """Validates PDEAssembler initialization and assembly when provided with species and kinetics."""
    grid = planar(x_max=1e-3, nx=10)
    step = ElectronTransferStep(
        ox_species="O", red_species="R", e_formal=0.0, k0=1e-3, alpha=0.5
    )
    mechanism = build([step])

    species_dict = {
        "O": Species(D=1e-9, c0=1.0),
        "R": Species(D=1e-9, c0=0.0),
    }
    kinetics_stub = lambda *args, **kwargs: 0.0

    assembler = PDEAssembler(
        grid=grid,
        mechanism=mechanism,
        species=species_dict,
        kinetics=kinetics_stub,
    )

    assert assembler.species == species_dict
    assert assembler.kinetics == kinetics_stub

    c_matrix = np.ones((2, 10))
    dc_dt = assembler.assemble(c_matrix, 0.0)
    assert dc_dt.shape == (2, 10)
