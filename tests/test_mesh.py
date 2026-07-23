import numpy as np
import pytest
from pydantic import ValidationError

from softpotato.core.abcs import BaseMesh
from softpotato.mesh.uniform_1d import Uniform1DMesh


def test_uniform_1d_mesh_initialization():
    """Test standard mesh creation and calculated properties."""
    mesh = Uniform1DMesh(x_min=0.0, x_max=1.0, n_points=11)

    assert isinstance(mesh, BaseMesh)
    assert mesh.num_nodes() == 11
    assert mesh.L == 1.0
    assert mesh.dx == pytest.approx(0.1)

    nodes = mesh.get_nodes()
    assert isinstance(nodes, np.ndarray)
    assert nodes.shape == (11,)
    assert nodes[0] == 0.0
    assert nodes[-1] == 1.0
    np.testing.assert_allclose(nodes, np.linspace(0.0, 1.0, 11))


def test_uniform_1d_mesh_positional_args():
    """Test initialization using positional arguments."""
    mesh = Uniform1DMesh(0.0, 2.0, 21)
    assert mesh.L == 2.0
    assert mesh.dx == pytest.approx(0.1)
    assert mesh.num_nodes() == 21


def test_uniform_1d_mesh_negative_domain():
    """Test mesh over a negative-coordinate domain."""
    mesh = Uniform1DMesh(x_min=-1.0, x_max=1.0, n_points=5)
    assert mesh.L == 2.0
    assert mesh.dx == pytest.approx(0.5)
    np.testing.assert_allclose(mesh.get_nodes(), np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_uniform_1d_mesh_invalid_bounds():
    """Test that x_max <= x_min raises ValidationError."""
    with pytest.raises(ValidationError):
        Uniform1DMesh(x_min=1.0, x_max=0.0, n_points=10)

    with pytest.raises(ValidationError):
        Uniform1DMesh(x_min=1.0, x_max=1.0, n_points=10)


def test_uniform_1d_mesh_invalid_n_points():
    """Test that n_points < 2 raises ValidationError."""
    with pytest.raises(ValidationError):
        Uniform1DMesh(x_min=0.0, x_max=1.0, n_points=1)

    with pytest.raises(ValidationError):
        Uniform1DMesh(x_min=0.0, x_max=1.0, n_points=0)


def test_uniform_1d_mesh_invalid_types():
    """Test input type enforcement via Pydantic."""
    with pytest.raises(ValidationError):
        Uniform1DMesh(x_min="invalid_str", x_max=1.0, n_points=10)
