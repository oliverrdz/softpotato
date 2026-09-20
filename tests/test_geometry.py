"""
Unit tests for the geometry and grid module.
"""

from __future__ import annotations

import numpy as np
import pytest

from softpotato.geometry import (
    Grid,
    PlanarElectrode,
    UniformGrid,
)


def test_uniform_grid_initialization() -> None:
    """Tests proper initialization and attributes of UniformGrid."""
    grid = UniformGrid(x_max=0.05, nodes=501)
    assert isinstance(grid, Grid)
    assert grid.nodes == 501
    assert grid.x_max == 0.05
    assert np.isclose(grid.dx, 0.05 / 500)
    assert len(grid.x) == 501
    assert np.isclose(grid.x[0], 0.0)
    assert np.isclose(grid.x[-1], 0.05)


def test_uniform_grid_validation() -> None:
    """Tests validation of input parameters for UniformGrid."""
    # Invalid x_max
    with pytest.raises(TypeError, match="real number"):
        UniformGrid(x_max="0.05", nodes=100)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        UniformGrid(x_max=True, nodes=100)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        UniformGrid(x_max=0.0, nodes=100)
    with pytest.raises(ValueError, match="positive"):
        UniformGrid(x_max=-0.05, nodes=100)

    # Invalid nodes
    with pytest.raises(TypeError, match="integer"):
        UniformGrid(x_max=0.05, nodes=100.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        UniformGrid(x_max=0.05, nodes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2"):
        UniformGrid(x_max=0.05, nodes=1)


def test_planar_electrode_initialization() -> None:
    """Tests PlanarElectrode initialization and properties."""
    grid = UniformGrid(x_max=0.05, nodes=100)
    electrode = PlanarElectrode(area=0.0707, grid=grid)

    assert electrode.area == 0.0707
    assert electrode.grid is grid
    assert "PlanarElectrode" in repr(electrode)


def test_planar_electrode_validation() -> None:
    """Tests validation of PlanarElectrode parameters."""
    grid = UniformGrid(x_max=0.05, nodes=100)

    # Invalid area
    with pytest.raises(TypeError, match="real number"):
        PlanarElectrode(area="0.0707", grid=grid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        PlanarElectrode(area=False, grid=grid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        PlanarElectrode(area=0.0, grid=grid)
    with pytest.raises(ValueError, match="positive"):
        PlanarElectrode(area=-1.0, grid=grid)

    # Invalid grid
    with pytest.raises(TypeError, match="Grid instance"):
        PlanarElectrode(area=0.0707, grid=None)  # type: ignore[arg-type]


def test_planar_electrode_laplacian() -> None:
    """Tests 1D planar Laplacian calculation."""
    grid = UniformGrid(x_max=1.0, nodes=11)
    electrode = PlanarElectrode(area=1.0, grid=grid)

    # For quadratic profile c(x) = x^2, d^2c/dx^2 = 2
    c = grid.x**2
    lap = electrode.laplacian(c)
    assert lap.shape == (9,)
    assert np.allclose(lap, 2.0, rtol=1e-10)
