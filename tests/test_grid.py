"""Unit tests for the spatial discretization and meshing module (softpotato.grid)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.grid import Mesh1D, Uniform1DMesh
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver


class DummyNonUniformMesh(Mesh1D):
    """Concrete non-uniform mesh subclass for testing base Mesh1D behaviors."""

    def __init__(self, nodes: np.ndarray) -> None:
        self._nodes = np.asarray(nodes, dtype=float)
        self._dx = np.diff(self._nodes)

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def dx(self) -> np.ndarray:
        return self._dx


class TestGridModuleExports:
    """Verify module exports and top-level package namespace."""

    def test_grid_exports(self) -> None:
        assert hasattr(sp, "grid")
        assert sp.Mesh1D is Mesh1D
        assert sp.Uniform1DMesh is Uniform1DMesh
        assert "Mesh1D" in sp.__all__
        assert "Uniform1DMesh" in sp.__all__
        assert "grid" in sp.__all__


class TestMesh1DBase:
    """Verify abstract Mesh1D base class properties, array protocol, and arithmetic."""

    def test_abstract_instantiation_fails(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Mesh1D()  # type: ignore[abstract]

    def test_base_properties_uniform(self) -> None:
        nodes = np.array([0.0, 0.5, 1.0])
        mesh = DummyNonUniformMesh(nodes)
        assert mesh.n_points == 3
        assert len(mesh) == 3
        assert mesh.bounds == (0.0, 1.0)
        assert mesh.x_min == 0.0
        assert mesh.x_max == 1.0
        assert mesh.is_uniform is True
        assert np.allclose(mesh.dx, np.array([0.5, 0.5]))
        assert "DummyNonUniformMesh" in repr(mesh)

    def test_base_properties_non_uniform(self) -> None:
        nodes = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        mesh = DummyNonUniformMesh(nodes)
        assert mesh.n_points == 5
        assert len(mesh) == 5
        assert mesh.bounds == (0.0, 1.0)
        assert mesh.is_uniform is False
        assert np.allclose(mesh.dx, np.array([0.1, 0.2, 0.4, 0.3]))

    def test_array_protocol_and_indexing(self) -> None:
        nodes = np.linspace(0.0, 1.0, 11)
        mesh = DummyNonUniformMesh(nodes)

        # np.asarray and np.array
        arr = np.asarray(mesh)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, nodes)

        arr_copy = np.array(mesh, copy=True)
        assert np.array_equal(arr_copy, nodes)

        # Indexing and slicing
        assert mesh[0] == 0.0
        assert mesh[-1] == 1.0
        assert np.array_equal(mesh[1:4], nodes[1:4])
        assert np.array_equal(mesh[[0, 2, 4]], nodes[[0, 2, 4]])

        # Iteration
        iterated = list(mesh)
        assert len(iterated) == 11
        assert np.allclose(iterated, nodes)

        # Boolean masking
        filtered = mesh[mesh > 0.5]
        assert np.allclose(filtered, nodes[nodes > 0.5])

    def test_math_operations(self) -> None:
        nodes = np.linspace(0.0, 1.0, 5)
        mesh = DummyNonUniformMesh(nodes)

        # Addition
        assert np.allclose(mesh + 2.0, nodes + 2.0)
        assert np.allclose(2.0 + mesh, 2.0 + nodes)
        assert np.allclose(mesh + mesh, nodes + nodes)

        # Subtraction
        assert np.allclose(mesh - 1.0, nodes - 1.0)
        assert np.allclose(1.0 - mesh, 1.0 - nodes)
        assert np.allclose(mesh - mesh, np.zeros_like(nodes))

        # Multiplication
        assert np.allclose(mesh * 3.0, nodes * 3.0)
        assert np.allclose(3.0 * mesh, 3.0 * nodes)
        assert np.allclose(mesh * mesh, nodes * nodes)

        # Division
        mesh_pos = DummyNonUniformMesh(nodes + 1.0)
        nodes_pos = nodes + 1.0
        assert np.allclose(mesh_pos / 2.0, nodes_pos / 2.0)
        assert np.allclose(2.0 / mesh_pos, 2.0 / nodes_pos)
        assert np.allclose(mesh_pos / mesh_pos, np.ones_like(nodes_pos))

        # Power
        assert np.allclose(mesh**2, nodes**2)

        # Unary operations
        assert np.allclose(-mesh, -nodes)
        assert np.allclose(+mesh, +nodes)
        assert np.allclose(abs(-mesh), nodes)

        # Comparisons
        assert np.array_equal(mesh < 0.5, nodes < 0.5)
        assert np.array_equal(mesh <= 0.5, nodes <= 0.5)
        assert np.array_equal(mesh > 0.5, nodes > 0.5)
        assert np.array_equal(mesh >= 0.5, nodes >= 0.5)

        # Ufuncs
        assert np.allclose(np.sin(mesh), np.sin(nodes))
        assert np.allclose(np.exp(mesh), np.exp(nodes))
        assert np.allclose(np.diff(mesh), np.diff(nodes))

    def test_equality(self) -> None:
        mesh1 = DummyNonUniformMesh(np.linspace(0.0, 1.0, 5))
        mesh2 = DummyNonUniformMesh(np.linspace(0.0, 1.0, 5))
        mesh3 = DummyNonUniformMesh(np.linspace(0.0, 1.0, 6))
        mesh4 = DummyNonUniformMesh(np.linspace(0.0, 2.0, 5))

        assert mesh1 == mesh2
        assert mesh1 != mesh3
        assert mesh1 != mesh4
        assert mesh1 == np.linspace(0.0, 1.0, 5)
        assert mesh1 != np.linspace(0.0, 1.0, 6)
        assert mesh1 != "not_a_mesh"


class TestUniform1DMesh:
    """Verify Uniform1DMesh construction, validations, properties, and interoperability."""

    def test_construct_with_n_points(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, n_points=11)
        assert mesh.n_points == 11
        assert len(mesh) == 11
        assert mesh.bounds == (0.0, 1.0)
        assert mesh.x_min == 0.0
        assert mesh.x_max == 1.0
        assert math.isclose(mesh.spacing, 0.1)
        assert math.isclose(mesh.step, 0.1)
        assert mesh.is_uniform is True
        assert len(mesh.dx) == 10
        assert np.allclose(mesh.dx, 0.1)
        expected_nodes = np.linspace(0.0, 1.0, 11)
        assert np.allclose(mesh.nodes, expected_nodes)
        assert mesh[0] == 0.0
        assert mesh[-1] == 1.0

    def test_construct_with_dx(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, dx=0.05)
        assert mesh.n_points == 21
        assert len(mesh) == 21
        assert mesh.bounds == (0.0, 1.0)
        assert math.isclose(mesh.spacing, 0.05)
        assert math.isclose(mesh.step, 0.05)
        assert mesh.is_uniform is True
        assert len(mesh.dx) == 20
        assert np.allclose(mesh.dx, 0.05)
        expected_nodes = np.linspace(0.0, 1.0, 21)
        assert np.allclose(mesh.nodes, expected_nodes)

    def test_construct_non_zero_offset(self) -> None:
        mesh = Uniform1DMesh(2.5, 7.5, n_points=51)
        assert mesh.bounds == (2.5, 7.5)
        assert mesh.x_min == 2.5
        assert mesh.x_max == 7.5
        assert math.isclose(mesh.spacing, 0.1)
        assert mesh[0] == 2.5
        assert mesh[-1] == 7.5

    def test_construct_negative_bounds(self) -> None:
        mesh = Uniform1DMesh(-2.0, 2.0, dx=0.5)
        assert mesh.bounds == (-2.0, 2.0)
        assert mesh.n_points == 9
        assert np.allclose(mesh.nodes, np.arange(-2.0, 2.5, 0.5))

    def test_minimum_n_points(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, n_points=2)
        assert mesh.n_points == 2
        assert len(mesh) == 2
        assert math.isclose(mesh.spacing, 1.0)
        assert len(mesh.dx) == 1
        assert mesh.dx[0] == 1.0
        assert np.allclose(mesh.nodes, [0.0, 1.0])

    def test_immutability(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, n_points=5)
        with pytest.raises(ValueError, match="read-only"):
            mesh.nodes[0] = 99.0
        with pytest.raises(ValueError, match="read-only"):
            mesh.dx[0] = 99.0

    def test_repr_and_str(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, n_points=5)
        repr_str = repr(mesh)
        assert "Uniform1DMesh" in repr_str
        assert "x_min=0.0" in repr_str
        assert "x_max=1.0" in repr_str
        assert "n_points=5" in repr_str

        str_str = str(mesh)
        assert "Uniform1DMesh" in str_str
        assert "n_points=5" in str_str

    def test_errors_bounds(self) -> None:
        # x_max == x_min
        with pytest.raises(ValueError, match="strictly greater than x_min"):
            Uniform1DMesh(1.0, 1.0, n_points=10)

        # x_max < x_min
        with pytest.raises(ValueError, match="strictly greater than x_min"):
            Uniform1DMesh(2.0, 1.0, n_points=10)

        # Non-finite bounds
        with pytest.raises(ValueError, match="must be finite"):
            Uniform1DMesh(float("nan"), 1.0, n_points=10)
        with pytest.raises(ValueError, match="must be finite"):
            Uniform1DMesh(0.0, float("inf"), n_points=10)

        # Non-numeric bounds
        with pytest.raises(TypeError, match="real numbers"):
            Uniform1DMesh("invalid", 1.0, n_points=10)  # type: ignore[arg-type]

    def test_errors_mutually_exclusive(self) -> None:
        # Neither specified
        with pytest.raises(ValueError, match="Must specify either 'n_points' or 'dx'"):
            Uniform1DMesh(0.0, 1.0)

        # Both specified
        with pytest.raises(ValueError, match="Cannot specify both 'n_points' and 'dx'"):
            Uniform1DMesh(0.0, 1.0, n_points=10, dx=0.1)

    def test_errors_n_points(self) -> None:
        # n_points < 2
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            Uniform1DMesh(0.0, 1.0, n_points=1)
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            Uniform1DMesh(0.0, 1.0, n_points=0)
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            Uniform1DMesh(0.0, 1.0, n_points=-5)

        # Non-integer n_points
        with pytest.raises(TypeError, match="n_points must be an integer"):
            Uniform1DMesh(0.0, 1.0, n_points=10.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_points must be an integer"):
            Uniform1DMesh(0.0, 1.0, n_points="10")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_points must be an integer"):
            Uniform1DMesh(0.0, 1.0, n_points=True)  # type: ignore[arg-type]

    def test_errors_dx(self) -> None:
        # dx <= 0
        with pytest.raises(ValueError, match="dx must be strictly positive"):
            Uniform1DMesh(0.0, 1.0, dx=0.0)
        with pytest.raises(ValueError, match="dx must be strictly positive"):
            Uniform1DMesh(0.0, 1.0, dx=-0.1)

        # dx > domain length
        with pytest.raises(
            ValueError, match="cannot be greater than the domain length"
        ):
            Uniform1DMesh(0.0, 1.0, dx=1.5)

        # Non-integer multiple
        with pytest.raises(ValueError, match="not an integer multiple of dx"):
            Uniform1DMesh(0.0, 1.0, dx=0.3)

        # Non-finite dx
        with pytest.raises(ValueError, match="dx must be finite"):
            Uniform1DMesh(0.0, 1.0, dx=float("nan"))
        with pytest.raises(ValueError, match="dx must be finite"):
            Uniform1DMesh(0.0, 1.0, dx=float("inf"))

        # Non-numeric dx
        with pytest.raises(TypeError, match="real number"):
            Uniform1DMesh(0.0, 1.0, dx="bad")  # type: ignore[arg-type]


class TestDiffusionProblemInteroperability:
    """Verify that Mesh1D seamlessly interoperates with DiffusionProblem and solvers."""

    def test_uniform_mesh_in_diffusion_problem(self) -> None:
        mesh = Uniform1DMesh(0.0, 1.0, n_points=31)

        # Direct mesh passing relies on array protocol at runtime
        # (DiffusionProblem full type integration is scheduled for issue #46)
        problem = DiffusionProblem(
            grid=mesh,  # type: ignore[arg-type]
            diffusivity={"A": 1e-4},
            boundary_conditions={"A": (DirichletBC(0.0), DirichletBC(1.0))},
            initial_conditions={"A": 1.0},
        )
        assert problem.n_points == 31
        assert problem.is_uniform is True
        assert math.isclose(problem.dx, mesh.spacing)

        solver = get_solver("scipy_ivp")
        res = solver.solve(problem, t_span=(0.0, 0.5))
        assert res.success is True
        assert res["A"].shape[1] == 31
