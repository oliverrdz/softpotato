import numpy as np
import pytest
import scipy.sparse as sp

from softpotato.discretizers.fdm_1d import FDM1DDiscretizer
from softpotato.mesh.uniform_1d import Uniform1DMesh
from softpotato.physics.species import TwoSpeciesModel


def test_fdm_laplacian_matrix_shape_and_type():
    """Verify standard Laplacian matrix dimensions and sparse format."""
    mesh = Uniform1DMesh(x_min=0.0, x_max=1e-4, num_nodes=101)
    discretizer = FDM1DDiscretizer()

    L = discretizer.build_laplacian_matrix(mesh)

    assert isinstance(L, sp.csc_matrix)
    assert L.shape == (101, 101)


def test_fdm_laplacian_accuracy_on_sine_wave():
    """
    Test spatial accuracy of interior central differences against the analytical second derivative:
    f(x) = sin(pi * x), f''(x) = -pi^2 * sin(pi * x)
    """
    num_nodes = 501
    mesh = Uniform1DMesh(x_min=0.0, x_max=1.0, num_nodes=num_nodes)
    discretizer = FDM1DDiscretizer()

    L = discretizer.build_laplacian_matrix(mesh)

    x = mesh.x
    u = np.sin(np.pi * x)
    d2u_analytical = -(np.pi**2) * np.sin(np.pi * x)

    d2u_numerical = L @ u

    # Exclude boundary nodes (index 0 and -1) where BCs are handled separately
    interior = slice(1, -1)
    max_error = np.max(np.abs(d2u_numerical[interior] - d2u_analytical[interior]))

    # 2nd-order convergence check (O(dx^2))
    assert max_error < 1e-3


def test_fdm_system_matrix_assembly():
    """Verify multi-species block diagonal system matrix assembly."""
    mesh = Uniform1DMesh(x_min=0.0, x_max=1e-4, num_nodes=50)
    model = TwoSpeciesModel(D_R=1e-9, D_O=2e-9, C_R_bulk=1.0, C_O_bulk=0.0)
    discretizer = FDM1DDiscretizer()

    A = discretizer.build_system_matrix(mesh, model)

    assert isinstance(A, sp.csc_matrix)
    assert A.shape == (100, 100)  # 2 species * 50 spatial nodes

    # Check that diffusion coefficients scale the sub-blocks properly
    L_single = discretizer.build_laplacian_matrix(mesh)

    block_R = A[:50, :50]
    block_O = A[50:, 50:]

    np.testing.assert_allclose(block_R.toarray(), (1e-9 * L_single).toarray())
    np.testing.assert_allclose(block_O.toarray(), (2e-9 * L_single).toarray())


def test_fdm_minimum_nodes_validation():
    """Ensure discretizer raises error if grid has fewer than 3 nodes."""
    mesh = Uniform1DMesh(x_min=0.0, x_max=1.0, num_nodes=2)
    discretizer = FDM1DDiscretizer()

    with pytest.raises(ValueError, match="at least 3 mesh nodes"):
        discretizer.build_laplacian_matrix(mesh)
