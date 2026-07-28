import numpy as np
import pytest

from softpotato.grid.mesh_generators import planar
from softpotato.grid.spatial import Grid


def test_planar_uniform():
    x_max = 1.0
    nx = 20
    grid = planar(x_max, nx, grid_type="uniform")

    assert isinstance(grid, Grid)
    assert len(grid.x) == nx
    assert len(grid.dx) == nx
    assert grid.x[0] == 0.0
    assert grid.x[-1] == x_max

    # For a uniform grid, all dx should be identical
    np.testing.assert_allclose(grid.dx, grid.dx[0])


def test_planar_expanding():
    x_max = 1.0
    nx = 20
    grid = planar(x_max, nx, grid_type="expanding")

    assert grid.x[0] == 0.0
    assert np.isclose(grid.x[-1], x_max)

    # For an expanding grid, dx should strictly increase
    assert np.all(np.diff(grid.dx[:-1]) > 0)


def test_planar_invalid_inputs():
    with pytest.raises(ValueError, match="must be greater than 0"):
        planar(-1.0, 50)

    with pytest.raises(ValueError, match="must be greater than 1"):
        planar(1.0, 1)

    with pytest.raises(ValueError, match="Unknown grid_type"):
        planar(1.0, 50, grid_type="logarithmic")
