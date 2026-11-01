"""Unit tests for the spatial discretization and meshing module (softpotato.grid)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import softpotato as sp
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Mesh1D,
    Uniform1DMesh,
    create_diffusion_mesh,
    diffusion_layer_thickness,
)
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
        assert sp.GeometricExpandingMesh is GeometricExpandingMesh
        assert sp.ExponentialMesh is ExponentialMesh
        assert sp.diffusion_layer_thickness is diffusion_layer_thickness
        assert sp.create_diffusion_mesh is create_diffusion_mesh
        for sym in (
            "Mesh1D",
            "Uniform1DMesh",
            "GeometricExpandingMesh",
            "ExponentialMesh",
            "diffusion_layer_thickness",
            "create_diffusion_mesh",
            "grid",
        ):
            assert sym in sp.__all__


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


class TestGeometricExpandingMesh:
    """Verify GeometricExpandingMesh parameterizations, ratios, two-sided expansion, and validations."""

    def test_construct_n_points_and_gamma(self) -> None:
        mesh = GeometricExpandingMesh(0.0, 1.0, n_points=21, gamma=1.15)
        assert mesh.n_points == 21
        assert len(mesh) == 21
        assert mesh.bounds == (0.0, 1.0)
        assert mesh.x_min == 0.0
        assert mesh.x_max == 1.0
        assert math.isclose(mesh.gamma, 1.15)
        assert math.isclose(mesh.expansion_ratio, 1.15)
        assert mesh.two_sided is False
        assert mesh.is_uniform is False

        # Node monotonicity and interval ratios
        assert np.all(mesh.dx > 0.0)
        assert np.all(np.diff(mesh.nodes) > 0.0)
        ratios = mesh.dx[1:] / mesh.dx[:-1]
        assert np.allclose(ratios, 1.15, rtol=1e-12)
        assert math.isclose(float(np.sum(mesh.dx)), 1.0, rel_tol=1e-12)
        assert math.isclose(mesh.dx_min, mesh.dx[0])
        assert math.isclose(mesh.dx_0, mesh.dx[0])
        assert math.isclose(mesh.dx_max, mesh.dx[-1])
        assert mesh.dx_min < mesh.dx_max

    def test_construct_dx_min_and_gamma_exact_and_fractional(self) -> None:
        # Exact geometric sum: dx_0 = 0.01, gamma = 1.2, M = 15 intervals
        gamma = 1.2
        dx_0 = 0.01
        m_intervals = 15
        domain_len = dx_0 * (gamma**m_intervals - 1.0) / (gamma - 1.0)

        mesh = GeometricExpandingMesh(0.0, domain_len, dx_min=dx_0, gamma=gamma)
        assert mesh.n_points == m_intervals + 1
        assert mesh.bounds == (0.0, domain_len)
        assert math.isclose(mesh.dx_min, dx_0, rel_tol=1e-12)
        assert np.allclose(mesh.dx[1:] / mesh.dx[:-1], gamma, rtol=1e-12)

        # Arbitrary domain length: dx_min acts as upper bound on initial step
        mesh2 = GeometricExpandingMesh(0.0, 1.0, dx_min=0.02, gamma=1.1)
        assert mesh2.bounds == (0.0, 1.0)
        assert mesh2.dx_min <= 0.02 + 1e-12
        assert np.allclose(mesh2.dx[1:] / mesh2.dx[:-1], 1.1, rtol=1e-12)

    def test_construct_n_points_and_dx_min_roundtrip(self) -> None:
        ref = GeometricExpandingMesh(0.5, 3.5, n_points=30, gamma=1.08)
        solved = GeometricExpandingMesh(0.5, 3.5, n_points=30, dx_min=ref.dx_min)

        assert solved.n_points == 30
        assert solved.bounds == (0.5, 3.5)
        assert math.isclose(solved.gamma, 1.08, rel_tol=1e-10)
        assert math.isclose(solved.dx_min, ref.dx_min, rel_tol=1e-10)
        assert np.allclose(solved.nodes, ref.nodes, rtol=1e-10, atol=1e-12)
        assert np.allclose(solved.dx[1:] / solved.dx[:-1], solved.gamma, rtol=1e-10)

    def test_two_sided_odd_and_even_n_points(self) -> None:
        # Odd n_points (21 -> 20 intervals, 10 per half, midpoint node at 0.5)
        mesh_odd = GeometricExpandingMesh(
            0.0, 1.0, n_points=21, gamma=1.2, two_sided=True
        )
        assert mesh_odd.two_sided is True
        assert mesh_odd.n_points == 21
        assert mesh_odd.bounds == (0.0, 1.0)
        assert math.isclose(mesh_odd.nodes[10], 0.5, abs_tol=1e-15)
        assert np.allclose(mesh_odd.dx, mesh_odd.dx[::-1], rtol=1e-12)
        assert np.allclose(
            mesh_odd.nodes, 1.0 - mesh_odd.nodes[::-1], rtol=1e-12, atol=1e-12
        )
        assert math.isclose(mesh_odd.dx[0], mesh_odd.dx[-1], rel_tol=1e-12)
        assert math.isclose(mesh_odd.dx_min, mesh_odd.dx[0])
        assert math.isclose(mesh_odd.dx_max, mesh_odd.dx[9])
        # Left half ratios equal gamma
        left_ratios = mesh_odd.dx[1:10] / mesh_odd.dx[:9]
        assert np.allclose(left_ratios, 1.2, rtol=1e-12)
        # Right half ratios equal 1 / gamma
        right_ratios = mesh_odd.dx[11:] / mesh_odd.dx[10:-1]
        assert np.allclose(right_ratios, 1.0 / 1.2, rtol=1e-12)

        # Even n_points (20 -> 19 intervals, symmetric with central interval)
        mesh_even = GeometricExpandingMesh(
            0.0, 1.0, n_points=20, gamma=1.2, two_sided=True
        )
        assert mesh_even.n_points == 20
        assert np.allclose(mesh_even.dx, mesh_even.dx[::-1], rtol=1e-12)
        assert math.isclose(mesh_even.dx_max, mesh_even.dx[9])

    def test_two_sided_with_dx_min_and_gamma(self) -> None:
        mesh = GeometricExpandingMesh(0.0, 2.0, dx_min=0.01, gamma=1.25, two_sided=True)
        assert mesh.two_sided is True
        assert mesh.n_points % 2 == 1
        assert mesh.bounds == (0.0, 2.0)
        assert mesh.dx_min <= 0.01 + 1e-12
        assert np.allclose(mesh.dx, mesh.dx[::-1], rtol=1e-12)
        mid_idx = mesh.n_points // 2
        assert math.isclose(mesh.nodes[mid_idx], 1.0, abs_tol=1e-15)

    def test_two_sided_with_n_points_and_dx_min(self) -> None:
        ref = GeometricExpandingMesh(0.0, 1.0, n_points=21, gamma=1.15, two_sided=True)
        solved = GeometricExpandingMesh(
            0.0, 1.0, n_points=21, dx_min=ref.dx_min, two_sided=True
        )
        assert math.isclose(solved.gamma, 1.15, rel_tol=1e-10)
        assert np.allclose(solved.nodes, ref.nodes, rtol=1e-10, atol=1e-12)

    def test_immutability_and_repr(self) -> None:
        mesh = GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=1.1)
        with pytest.raises(ValueError, match="read-only"):
            mesh.nodes[0] = 5.0
        with pytest.raises(ValueError, match="read-only"):
            mesh.dx[0] = 5.0

        assert "GeometricExpandingMesh" in repr(mesh)
        assert "gamma=1.1" in repr(mesh)
        assert "GeometricExpandingMesh" in str(mesh)

    def test_validation_errors(self) -> None:
        # Invalid bounds
        with pytest.raises(ValueError, match="strictly greater than x_min"):
            GeometricExpandingMesh(1.0, 1.0, n_points=10, gamma=1.1)
        with pytest.raises(ValueError, match="strictly greater than x_min"):
            GeometricExpandingMesh(2.0, 1.0, n_points=10, gamma=1.1)
        with pytest.raises(ValueError, match="must be finite"):
            GeometricExpandingMesh(float("nan"), 1.0, n_points=10, gamma=1.1)
        with pytest.raises(TypeError, match="real numbers"):
            GeometricExpandingMesh(True, 1.0, n_points=10, gamma=1.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="real numbers"):
            GeometricExpandingMesh("0", 1.0, n_points=10, gamma=1.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="two_sided must be a boolean"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=1.1, two_sided=1)  # type: ignore[arg-type]

        # Parameter combination count != 2
        with pytest.raises(ValueError, match="Must specify exactly two"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10)
        with pytest.raises(ValueError, match="Must specify exactly two"):
            GeometricExpandingMesh(0.0, 1.0)
        with pytest.raises(ValueError, match="Must specify exactly two"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=1.1, dx_min=0.01)

        # Invalid gamma
        with pytest.raises(ValueError, match="gamma must be strictly greater than 1.0"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=1.0)
        with pytest.raises(ValueError, match="gamma must be strictly greater than 1.0"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=0.95)
        with pytest.raises(ValueError, match="gamma must be finite"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=float("inf"))
        with pytest.raises(TypeError, match="gamma must be a real number"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="gamma must be a real number"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10, gamma="bad")  # type: ignore[arg-type]

        # Invalid dx_min
        with pytest.raises(ValueError, match="dx_min must be strictly positive"):
            GeometricExpandingMesh(0.0, 1.0, dx_min=0.0, gamma=1.1)
        with pytest.raises(ValueError, match="dx_min must be strictly positive"):
            GeometricExpandingMesh(0.0, 1.0, dx_min=-0.01, gamma=1.1)
        with pytest.raises(ValueError, match="dx_min must be finite"):
            GeometricExpandingMesh(0.0, 1.0, dx_min=float("nan"), gamma=1.1)
        with pytest.raises(TypeError, match="dx_min must be a real number"):
            GeometricExpandingMesh(0.0, 1.0, dx_min=False, gamma=1.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dx_min must be a real number"):
            GeometricExpandingMesh(0.0, 1.0, dx_min="bad", gamma=1.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="strictly less than the domain length"):
            GeometricExpandingMesh(0.0, 1.0, dx_min=1.0, gamma=1.1)
        with pytest.raises(
            ValueError, match="strictly less than half the domain length"
        ):
            GeometricExpandingMesh(0.0, 1.0, dx_min=0.5, gamma=1.1, two_sided=True)
        with pytest.raises(ValueError, match="strictly less than the domain length"):
            GeometricExpandingMesh(0.0, 1.0, n_points=11, dx_min=0.1)

        # Invalid n_points
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            GeometricExpandingMesh(0.0, 1.0, n_points=1, gamma=1.1)
        with pytest.raises(ValueError, match="n_points must be >= 3"):
            GeometricExpandingMesh(0.0, 1.0, n_points=2, dx_min=0.1)
        with pytest.raises(ValueError, match="n_points must be >= 3"):
            GeometricExpandingMesh(0.0, 1.0, n_points=2, gamma=1.1, two_sided=True)
        with pytest.raises(ValueError, match="n_points must be >= 4"):
            GeometricExpandingMesh(0.0, 1.0, n_points=3, dx_min=0.1, two_sided=True)
        with pytest.raises(TypeError, match="n_points must be an integer"):
            GeometricExpandingMesh(0.0, 1.0, n_points=10.5, gamma=1.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_points must be an integer"):
            GeometricExpandingMesh(0.0, 1.0, n_points=True, gamma=1.1)  # type: ignore[arg-type]


class TestExponentialMesh:
    """Verify ExponentialMesh mappings, properties, two-sided clustering, and validations."""

    def test_exponential_mapping_one_sided(self) -> None:
        mesh = ExponentialMesh(0.0, 2.0, n_points=41, beta=3.5)
        assert mesh.n_points == 41
        assert len(mesh) == 41
        assert mesh.bounds == (0.0, 2.0)
        assert math.isclose(mesh.beta, 3.5)
        assert mesh.two_sided is False
        assert mesh.mapping == "exponential"
        assert mesh.is_uniform is False

        # Exact analytical formula check
        xi = np.linspace(0.0, 1.0, 41)
        expected = 2.0 * (np.exp(3.5 * xi) - 1.0) / (np.exp(3.5) - 1.0)
        assert np.allclose(mesh.xi, xi)
        assert np.allclose(mesh.nodes, expected, rtol=1e-12, atol=1e-12)
        assert np.all(mesh.dx > 0.0)

        # Consecutive interval ratio for exponential mapping is exp(beta / (N - 1))
        expected_ratio = math.exp(3.5 / 40.0)
        assert np.allclose(mesh.dx[1:] / mesh.dx[:-1], expected_ratio, rtol=1e-12)
        assert math.isclose(mesh.dx_min, mesh.dx[0])
        assert math.isclose(mesh.dx_max, mesh.dx[-1])

    def test_tanh_and_sinh_mappings(self) -> None:
        mesh_tanh = ExponentialMesh.from_tanh(0.0, 1.0, n_points=51, beta=2.5)
        assert mesh_tanh.mapping == "tanh"
        assert mesh_tanh.bounds == (0.0, 1.0)
        assert np.all(mesh_tanh.dx > 0.0)
        assert np.all(np.diff(mesh_tanh.dx) > 0.0)

        mesh_sinh = ExponentialMesh.from_sinh(0.0, 1.0, n_points=51, beta=3.0)
        assert mesh_sinh.mapping == "sinh"
        assert mesh_sinh.bounds == (0.0, 1.0)
        assert np.all(mesh_sinh.dx > 0.0)
        assert np.all(np.diff(mesh_sinh.dx) > 0.0)

    @pytest.mark.parametrize("mapping", ["exponential", "tanh", "sinh"])
    def test_two_sided_mappings(self, mapping: str) -> None:
        mesh = ExponentialMesh(
            -1.0, 1.0, n_points=31, beta=3.0, two_sided=True, mapping=mapping
        )
        assert mesh.two_sided is True
        assert mesh.bounds == (-1.0, 1.0)
        assert math.isclose(mesh.nodes[15], 0.0, abs_tol=1e-15)
        assert np.all(mesh.dx > 0.0)
        assert np.allclose(mesh.dx, mesh.dx[::-1], rtol=1e-12)
        assert np.allclose(mesh.nodes, -mesh.nodes[::-1], rtol=1e-12, atol=1e-12)
        assert math.isclose(mesh.dx_min, mesh.dx[0])
        assert math.isclose(mesh.dx_min, mesh.dx[-1])
        assert mesh.dx_max > mesh.dx_min

    def test_immutability_and_repr(self) -> None:
        mesh = ExponentialMesh(0.0, 1.0, n_points=15, beta=2.0)
        with pytest.raises(ValueError, match="read-only"):
            mesh.nodes[0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            mesh.dx[0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            mesh.xi[0] = 1.0

        assert "ExponentialMesh" in repr(mesh)
        assert "beta=2" in repr(mesh)
        assert "ExponentialMesh" in str(mesh)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="strictly greater than x_min"):
            ExponentialMesh(1.0, 0.0, n_points=10, beta=2.0)
        with pytest.raises(ValueError, match="must be finite"):
            ExponentialMesh(0.0, float("inf"), n_points=10, beta=2.0)
        with pytest.raises(TypeError, match="real numbers"):
            ExponentialMesh(False, 1.0, n_points=10, beta=2.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="real numbers"):
            ExponentialMesh("0", 1.0, n_points=10, beta=2.0)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="n_points must be >= 2"):
            ExponentialMesh(0.0, 1.0, n_points=1, beta=2.0)
        with pytest.raises(ValueError, match="n_points must be >= 3"):
            ExponentialMesh(0.0, 1.0, n_points=2, beta=2.0, two_sided=True)
        with pytest.raises(TypeError, match="n_points must be an integer"):
            ExponentialMesh(0.0, 1.0, n_points=5.5, beta=2.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="two_sided must be a boolean"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=2.0, two_sided="yes")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="beta must be strictly positive"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=0.0)
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=-1.5)
        with pytest.raises(ValueError, match="beta must be finite"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=float("nan"))
        with pytest.raises(ValueError, match="too large"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=600.0)
        with pytest.raises(TypeError, match="beta must be a real number"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="beta must be a real number"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta="bad")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unknown mapping"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=2.0, mapping="linear")
        with pytest.raises(TypeError, match="mapping must be a string"):
            ExponentialMesh(0.0, 1.0, n_points=10, beta=2.0, mapping=123)  # type: ignore[arg-type]


class TestDiffusionLayerFactories:
    """Verify diffusion_layer_thickness and automatic mesh generation factories."""

    def test_diffusion_layer_thickness(self) -> None:
        D = 1e-9
        t_max = 4.0
        delta = diffusion_layer_thickness(D, t_max)
        assert math.isclose(delta, 6.0 * math.sqrt(D * t_max), rel_tol=1e-12)

        delta_custom = diffusion_layer_thickness(D, t_max, factor=5.0)
        assert math.isclose(delta_custom, 5.0 * math.sqrt(D * t_max), rel_tol=1e-12)

        with pytest.raises(ValueError, match="D must be strictly positive"):
            diffusion_layer_thickness(0.0, 1.0)
        with pytest.raises(ValueError, match="t_max must be strictly positive"):
            diffusion_layer_thickness(1e-9, -1.0)
        with pytest.raises(ValueError, match="factor must be strictly positive"):
            diffusion_layer_thickness(1e-9, 1.0, factor=0.0)
        with pytest.raises(ValueError, match="must be finite"):
            diffusion_layer_thickness(float("inf"), 1.0)
        with pytest.raises(TypeError, match="must be a real number"):
            diffusion_layer_thickness(True, 1.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a real number"):
            diffusion_layer_thickness("1e-9", 1.0)  # type: ignore[arg-type]

    def test_from_diffusion_layer_methods(self) -> None:
        D = 1e-9
        t_max = 1.0
        expected_max = 6.0 * math.sqrt(D * t_max)

        # GeometricExpandingMesh defaults and custom options
        g_mesh = GeometricExpandingMesh.from_diffusion_layer(D, t_max)
        assert isinstance(g_mesh, GeometricExpandingMesh)
        assert g_mesh.n_points == 100
        assert math.isclose(g_mesh.gamma, 1.05)
        assert math.isclose(g_mesh.x_max, expected_max)

        g_mesh_n = GeometricExpandingMesh.from_diffusion_layer(D, t_max, n_points=50)
        assert g_mesh_n.n_points == 50
        assert math.isclose(g_mesh_n.gamma, 1.05)

        g_mesh_g = GeometricExpandingMesh.from_diffusion_layer(D, t_max, gamma=1.1)
        assert g_mesh_g.n_points == 100
        assert math.isclose(g_mesh_g.gamma, 1.1)

        # ExponentialMesh
        e_mesh = ExponentialMesh.from_diffusion_layer(
            D, t_max, x_min=1e-5, n_points=60, beta=4.0
        )
        assert isinstance(e_mesh, ExponentialMesh)
        assert e_mesh.n_points == 60
        assert math.isclose(e_mesh.beta, 4.0)
        assert math.isclose(e_mesh.x_min, 1e-5)
        assert math.isclose(e_mesh.x_max, 1e-5 + expected_max)

        # Uniform1DMesh
        u_mesh = Uniform1DMesh.from_diffusion_layer(D, t_max, n_points=80)
        assert isinstance(u_mesh, Uniform1DMesh)
        assert u_mesh.n_points == 80
        assert math.isclose(u_mesh.x_max, expected_max)

        u_mesh_default = Uniform1DMesh.from_diffusion_layer(D, t_max)
        assert u_mesh_default.n_points == 100

    def test_create_diffusion_mesh_factory(self) -> None:
        D = 1e-9
        t_max = 2.0
        expected_max = 6.0 * math.sqrt(D * t_max)

        m_geom = create_diffusion_mesh(
            D, t_max, kind="geometric", n_points=40, gamma=1.08
        )
        assert isinstance(m_geom, GeometricExpandingMesh)
        assert m_geom.n_points == 40
        assert math.isclose(m_geom.x_max, expected_max)

        m_exp = create_diffusion_mesh(
            D, t_max, kind="exponential", n_points=40, beta=3.5, mapping="tanh"
        )
        assert isinstance(m_exp, ExponentialMesh)
        assert m_exp.n_points == 40
        assert m_exp.mapping == "tanh"
        assert math.isclose(m_exp.x_max, expected_max)

        m_uni = create_diffusion_mesh(D, t_max, kind="uniform", n_points=40)
        assert isinstance(m_uni, Uniform1DMesh)
        assert m_uni.n_points == 40
        assert math.isclose(m_uni.x_max, expected_max)

        with pytest.raises(ValueError, match="Unknown mesh kind"):
            create_diffusion_mesh(D, t_max, kind="chebyshev")
        with pytest.raises(TypeError, match="kind must be a string"):
            create_diffusion_mesh(D, t_max, kind=123)  # type: ignore[arg-type]
