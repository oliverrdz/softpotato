"""Unit tests for the spatial coordinate geometry module (softpotato.geometry)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

import softpotato as sp
from softpotato.analytical.microelectrodes import (
    microhemisphere_limiting_current,
    microsphere_limiting_current,
)
from softpotato.constants import FARADAY
from softpotato.geometry import (
    CylindricalGeometry,
    Geometry,
    PlanarGeometry,
    SphericalGeometry,
    UnknownGeometryError,
    get_geometry,
    list_geometries,
    register_geometry,
)
from softpotato.solver import DirichletBC, NeumannBC


def _banded_matvec(l_banded: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Multiply a (3, N) banded tridiagonal matrix in (1, 1) format by vector c."""
    out = l_banded[1, :] * c
    out[:-1] += l_banded[0, 1:] * c[1:]
    out[1:] += l_banded[2, :-1] * c[:-1]
    return out


def test_geometry_exports_and_registry() -> None:
    """Verify public exports and geometry registry aliases."""
    assert hasattr(sp, "geometry")
    assert sp.Geometry is Geometry
    assert sp.PlanarGeometry is PlanarGeometry
    assert sp.SphericalGeometry is SphericalGeometry
    assert sp.CylindricalGeometry is CylindricalGeometry
    assert sp.get_geometry is get_geometry
    assert sp.list_geometries is list_geometries
    assert sp.register_geometry is register_geometry

    geoms = list_geometries()
    for expected in (
        "planar",
        "cartesian",
        "1d",
        "linear",
        "spherical",
        "sphere",
        "hemispherical",
        "hemisphere",
        "cylindrical",
        "cylinder",
        "wire",
    ):
        assert expected in geoms


def test_get_geometry_factory_and_aliases() -> None:
    """Verify get_geometry resolves strings, aliases, and existing Geometry instances."""
    default_geom = get_geometry()
    assert isinstance(default_geom, PlanarGeometry)
    assert default_geom.area == 1.0
    assert default_geom.name == "planar"
    assert default_geom.coordinate_name == "x"
    assert default_geom.dimension == 1
    assert default_geom.symmetry_factor == 0

    planar = get_geometry("Cartesian", area=2.5)
    assert isinstance(planar, PlanarGeometry)
    assert planar.area == 2.5
    assert planar.surface_area == 2.5
    assert get_geometry(planar) is planar

    sphere = get_geometry("spherical", r0=1e-4)
    assert isinstance(sphere, SphericalGeometry)
    assert not sphere.hemisphere
    assert math.isclose(sphere.area, 4.0 * math.pi * (1e-4**2))

    hemisphere = get_geometry("hemispherical", r0=1e-4)
    assert isinstance(hemisphere, SphericalGeometry)
    assert hemisphere.hemisphere
    assert math.isclose(hemisphere.area, 2.0 * math.pi * (1e-4**2))

    hemisphere_mode = SphericalGeometry(radius=2e-4, mode="hemisphere")
    assert hemisphere_mode.hemisphere
    assert math.isclose(hemisphere_mode.radius, 2e-4)
    assert math.isclose(hemisphere_mode.area, 2.0 * math.pi * (2e-4**2))

    cylinder = get_geometry("cylindrical", r0=5e-6, length=2e-2)
    assert isinstance(cylinder, CylindricalGeometry)
    assert math.isclose(cylinder.radius, 5e-6)
    assert math.isclose(cylinder.L, 2e-2)
    assert math.isclose(cylinder.area, 2.0 * math.pi * 5e-6 * 2e-2)

    with pytest.raises((KeyError, ValueError), match="Unknown geometry"):
        get_geometry("toroidal")

    with pytest.raises(UnknownGeometryError, match="Unknown geometry"):
        get_geometry("non_existent")

    with pytest.raises(TypeError, match="must be a string or Geometry"):
        get_geometry(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Cannot pass constructor keyword arguments"):
        get_geometry(planar, area=5.0)


def test_custom_geometry_registration_and_py310_metaclass() -> None:
    """Verify custom geometry registration via subclassing and decorator."""
    from softpotato.geometry.base import _GeometryMeta

    class SimulatedPy310ABCMeta(type):
        def __new__(mcls, name, bases, namespace, **kwargs):
            return super().__new__(mcls, name, bases, namespace, **kwargs)

    class SimulatedGeometryMeta(_GeometryMeta, SimulatedPy310ABCMeta):
        pass

    class SimulatedBaseGeometry(metaclass=SimulatedGeometryMeta):
        _registry: ClassVar[dict[str, type]] = {}

    class SubGeom(SimulatedBaseGeometry, name="simulated_geom"):
        pass

    assert "simulated_geom" in SimulatedBaseGeometry._registry

    @register_geometry("custom_test_geom")
    class CustomTestGeometry(PlanarGeometry):
        pass

    assert "custom_test_geom" in list_geometries()
    assert isinstance(get_geometry("custom_test_geom"), CustomTestGeometry)

    with pytest.raises(TypeError, match="must inherit from Geometry"):

        @register_geometry("invalid_geom_class")  # type: ignore[arg-type]
        class NotAGeometry:
            pass


@pytest.mark.parametrize("is_uniform", [True, False])
def test_laplacian_exact_on_quadratic_polynomials(is_uniform: bool) -> None:
    """Verify 2nd-order Laplacian operators are exact for quadratic polynomials c(r) = r²."""
    if is_uniform:
        x = np.linspace(0.0, 2.0, 41)
        r = np.linspace(0.5, 2.5, 41)
    else:
        s = np.linspace(0.0, 1.0, 41) ** 1.3
        x = 2.0 * s
        r = 0.5 + 2.0 * s

    # Planar: ∇²(3x² - 2x + 5) = 6
    planar = PlanarGeometry()
    c_planar = 3.0 * (x**2) - 2.0 * x + 5.0
    np.testing.assert_allclose(
        planar.laplacian(c_planar, x), 6.0, rtol=1e-11, atol=1e-11
    )

    # Cylindrical (p=1): ∇²(r²) = d²(r²)/dr² + (1/r)*d(r²)/dr = 2 + 2 = 4
    cylinder = CylindricalGeometry(r0=0.5)
    c_cyl = r**2
    np.testing.assert_allclose(
        cylinder.laplacian(c_cyl, r), 4.0, rtol=1e-11, atol=1e-11
    )

    # Spherical (p=2): ∇²(r²) = d²(r²)/dr² + (2/r)*d(r²)/dr = 2 + 4 = 6
    sphere = SphericalGeometry(r0=0.5)
    c_sph = r**2
    np.testing.assert_allclose(sphere.laplacian(c_sph, r), 6.0, rtol=1e-11, atol=1e-11)


def test_spherical_harmonic_laplacian_and_convergence() -> None:
    """Verify c(r) = 1/r has ∇²c = 0 and spherical Laplacian achieves 2nd-order O(Δr²) convergence."""
    sphere = SphericalGeometry(r0=1.0)

    # 1. Harmonic profile c(r) = 1/r has exact ∇²(1/r) = 0 at interior points
    r_harm = np.linspace(1.0, 3.0, 101)
    lap_harm = sphere.laplacian(1.0 / r_harm, r_harm)
    np.testing.assert_allclose(lap_harm[1:-1], 0.0, atol=1e-11)

    # 2. 2nd-order O(Δr²) convergence on smooth spherical eigenfunction c(r) = sin(r)/r
    # Exact spherical Laplacian: ∇²(sin(r)/r) = -sin(r)/r
    errors = []
    d_rs = []
    for n_pts in (26, 51, 101, 201, 401):
        r = np.linspace(1.0, 3.0, n_pts)
        dr = float(r[1] - r[0])
        c = np.sin(r) / r
        exact = -np.sin(r) / r
        lap = sphere.laplacian(c, r)
        errors.append(float(np.max(np.abs(lap[1:-1] - exact[1:-1]))))
        d_rs.append(dr)

    assert errors[-1] < 1e-4
    slope, _ = np.polyfit(np.log(d_rs), np.log(errors), 1)
    assert slope > 1.95


def test_cylindrical_harmonic_laplacian_and_convergence() -> None:
    """Verify c(r) = ln(r) has ∇²c -> 0 with 2nd-order O(Δr²) convergence in cylindrical coordinates."""
    cylinder = CylindricalGeometry(r0=1.0)
    errors = []
    d_rs = []

    for n_pts in (51, 101, 201, 401, 801):
        r = np.linspace(1.0, 3.0, n_pts)
        dr = float(r[1] - r[0])
        c = np.log(r)
        lap = cylinder.laplacian(c, r)
        errors.append(float(np.sqrt(np.mean(lap[1:-1] ** 2))))
        d_rs.append(dr)

    assert errors[-1] < 1e-5
    slope, _ = np.polyfit(np.log(d_rs), np.log(errors), 1)
    assert slope > 1.95


def test_planar_sinusoidal_laplacian_and_convergence() -> None:
    """Verify 2nd-order O(Δx²) spatial convergence on c(x) = sin(pi * x) in planar geometry."""
    planar = PlanarGeometry()
    errors = []
    d_xs = []

    for n_pts in (26, 51, 101, 201):
        x = np.linspace(0.0, 1.0, n_pts)
        dx = float(x[1] - x[0])
        c = np.sin(math.pi * x)
        exact = -(math.pi**2) * np.sin(math.pi * x)
        lap = planar.laplacian(c, x)
        errors.append(float(np.max(np.abs(lap - exact))))
        d_xs.append(dx)

    assert errors[-1] < 1e-3
    slope, _ = np.polyfit(np.log(d_xs), np.log(errors), 1)
    assert slope > 1.95


@pytest.mark.parametrize("geom_name", ["planar", "spherical", "cylindrical"])
@pytest.mark.parametrize("is_uniform", [True, False])
@pytest.mark.parametrize(
    "left_btype,right_btype",
    [
        ("dirichlet", "dirichlet"),
        ("neumann", "dirichlet"),
        ("dirichlet", "neumann"),
        ("neumann", "neumann"),
    ],
)
def test_laplacian_and_laplacian_matrix_consistency(
    geom_name: str,
    is_uniform: bool,
    left_btype: str,
    right_btype: str,
) -> None:
    """Verify laplacian_matrix matches laplacian across geometries, grids, and BCs."""
    geom = (
        get_geometry(geom_name, r0=0.5)
        if geom_name != "planar"
        else get_geometry("planar")
    )
    if is_uniform:
        grid = np.linspace(0.5, 2.5, 31)
    else:
        grid = 0.5 + 2.0 * (np.linspace(0.0, 1.0, 31) ** 1.4)

    c = np.sin(grid) + 0.25 * (grid**2)
    g_left = -0.35
    g_right = 0.65

    bc_left = DirichletBC(1.0) if left_btype == "dirichlet" else NeumannBC(g_left)
    bc_right = DirichletBC(0.0) if right_btype == "dirichlet" else NeumannBC(g_right)

    lap_vec = geom.laplacian(c, grid, bc_left=bc_left, bc_right=bc_right, t=0.5)
    l_banded, bc_scale = geom.laplacian_matrix(grid, bc_left=bc_left, bc_right=bc_right)

    bc_source = np.zeros_like(grid)
    if left_btype == "neumann":
        bc_source[0] = bc_scale[0] * g_left
    if right_btype == "neumann":
        bc_source[-1] = bc_scale[-1] * g_right

    lap_from_mat = _banded_matvec(l_banded, c) + bc_source
    np.testing.assert_allclose(lap_from_mat, lap_vec, rtol=1e-12, atol=1e-12)


def test_surface_flux_and_current_scaling() -> None:
    """Verify surface_flux and Faradaic current scaling across planar, spherical, and cylindrical geometries."""
    D = 1e-9  # m^2/s
    c_bulk = 1.0  # mol/m^3
    r0 = 1e-5  # 10 um

    # 1. Spherical steady-state profile: c(r) = c_bulk * (1 - r0 / r)
    # Exact analytical gradient at r = r0: dc/dr|_{r0} = c_bulk / r0
    # Normal flux into solution at r = r0: J = -D * c_bulk / r0
    r = r0 + (1e-3 - r0) * (
        np.linspace(0.0, 1.0, 501) ** 2.0
    )  # non-uniform refined near r0
    c_sph = c_bulk * (1.0 - r0 / r)

    sphere = SphericalGeometry(r0=r0, hemisphere=False)
    flux_sph = sphere.surface_flux(c_sph, r, diffusivity=D, side="left")
    assert isinstance(flux_sph, float)
    expected_flux = -D * c_bulk / r0
    assert math.isclose(flux_sph, expected_flux, rel_tol=1e-4)

    # Full sphere current matches microsphere_limiting_current (cathodic < 0)
    curr_sph = sphere.current(flux_sph, n_electrons=1)
    expected_i_sphere = -float(
        microsphere_limiting_current(r0, n=1, D=D, c_bulk=c_bulk)
    )
    assert math.isclose(curr_sph, expected_i_sphere, rel_tol=1e-4)

    # Hemisphere current is half of full sphere and matches microhemisphere_limiting_current
    hemisphere = SphericalGeometry(r0=r0, hemisphere=True)
    curr_hemi = hemisphere.current(flux_sph, n_electrons=1)
    expected_i_hemi = -float(
        microhemisphere_limiting_current(r0, n=1, D=D, c_bulk=c_bulk)
    )
    assert math.isclose(curr_hemi, expected_i_hemi, rel_tol=1e-4)
    assert math.isclose(curr_sph, 2.0 * curr_hemi, rel_tol=1e-12)

    # 2. Cylindrical current scaling: I = n * F * (2 * pi * r0 * L) * J
    length = 5e-3
    cylinder = CylindricalGeometry(r0=r0, length=length)
    curr_cyl = cylinder.current(flux_sph, n_electrons=2)
    expected_i_cyl = 2 * FARADAY * (2.0 * math.pi * r0 * length) * flux_sph
    assert math.isclose(curr_cyl, expected_i_cyl, rel_tol=1e-12)

    # 3. Planar 2D concentration array (time series) & right-boundary flux
    L_dom = 2.0
    x = np.linspace(0.0, L_dom, 41)
    # 2D time series of linear profiles: c(t, x) = scale(t) * (1 - x / L_dom)
    scales = np.array([1.0, 2.0, 3.0])
    c_2d = scales[:, None] * (1.0 - x[None, :] / L_dom)
    planar = PlanarGeometry(area=0.25)

    j_left = planar.surface_flux(c_2d, x, diffusivity=D, side="left")
    j_right = planar.surface_flux(c_2d, x, diffusivity=D, side="right")
    assert isinstance(j_left, np.ndarray) and j_left.shape == (3,)
    assert isinstance(j_right, np.ndarray) and j_right.shape == (3,)
    np.testing.assert_allclose(j_left, scales * (D / L_dom), rtol=1e-12)
    np.testing.assert_allclose(j_right, -scales * (D / L_dom), rtol=1e-12)

    i_left = planar.current(j_left, n_electrons=1)
    assert isinstance(i_left, np.ndarray) and i_left.shape == (3,)
    np.testing.assert_allclose(i_left, FARADAY * 0.25 * j_left, rtol=1e-12)

    # Test planar right current
    i_right = planar.current(j_right, n_electrons=1, side="right", grid=x)
    np.testing.assert_allclose(i_right, FARADAY * 0.25 * j_right, rtol=1e-12)

    # Test spherical right current scaling with outer radius
    r_grid = np.linspace(1e-4, 5e-4, 50)
    sphere = SphericalGeometry(r0=1e-4)
    i_sph_left = sphere.current(1.0, n_electrons=1, side="left")
    i_sph_right = sphere.current(1.0, n_electrons=1, side="right", grid=r_grid)
    np.testing.assert_allclose(i_sph_left, FARADAY * (4.0 * np.pi * (1e-4**2)) * 1.0)
    np.testing.assert_allclose(i_sph_right, FARADAY * (4.0 * np.pi * (5e-4**2)) * 1.0)

    # Test cylindrical right current scaling with outer radius
    cyl = CylindricalGeometry(r0=1e-4, length=0.01)
    i_cyl_left = cyl.current(1.0, n_electrons=1, side="left")
    i_cyl_right = cyl.current(1.0, n_electrons=1, side="right", grid=r_grid)
    np.testing.assert_allclose(i_cyl_left, FARADAY * (2.0 * np.pi * 1e-4 * 0.01) * 1.0)
    np.testing.assert_allclose(i_cyl_right, FARADAY * (2.0 * np.pi * 5e-4 * 0.01) * 1.0)


def test_geometry_parameter_and_grid_validation() -> None:
    """Verify parameter validation (r0 > 0, area > 0, valid grid bounds)."""
    # Invalid planar area
    with pytest.raises(ValueError, match="area must be positive"):
        PlanarGeometry(area=0.0)
    with pytest.raises(ValueError, match="area must be positive"):
        PlanarGeometry(area=-1.0)

    # Invalid spherical r0, mode, area
    with pytest.raises(ValueError, match="r0 must be positive"):
        SphericalGeometry(r0=0.0)
    with pytest.raises(ValueError, match="r0 must be positive"):
        SphericalGeometry(r0=-1e-5)
    with pytest.raises(ValueError, match="Cannot specify both 'r0' and 'radius'"):
        SphericalGeometry(r0=1e-5, radius=2e-5)
    with pytest.raises(ValueError, match="mode must be 'sphere' or 'hemisphere'"):
        SphericalGeometry(r0=1e-5, mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="area must be positive"):
        SphericalGeometry(r0=1e-5, area=-1.0)

    # Invalid cylindrical r0, length, area
    with pytest.raises(ValueError, match="r0 must be positive"):
        CylindricalGeometry(r0=0.0)
    with pytest.raises(ValueError, match="length must be positive"):
        CylindricalGeometry(r0=1e-5, length=0.0)
    with pytest.raises(ValueError, match="Cannot specify both 'r0' and 'radius'"):
        CylindricalGeometry(r0=1e-5, radius=2e-5)
    with pytest.raises(ValueError, match="Cannot specify both 'length' and 'L'"):
        CylindricalGeometry(r0=1e-5, length=1.0, L=2.0)
    with pytest.raises(ValueError, match="area must be positive"):
        CylindricalGeometry(r0=1e-5, area=0.0)

    # Radial geometries reject grids starting at r <= 0
    sphere = SphericalGeometry()
    cylinder = CylindricalGeometry()
    for radial_geom in (sphere, cylinder):
        with pytest.raises(ValueError, match="strictly positive radial coordinates"):
            radial_geom.validate_grid(np.linspace(0.0, 1.0, 10))
        with pytest.raises(ValueError, match="strictly positive radial coordinates"):
            radial_geom.laplacian(np.ones(10), np.linspace(-0.5, 1.0, 10))

    # General grid validation (1D, >= 3 points, strictly increasing)
    planar = PlanarGeometry()
    with pytest.raises(ValueError, match="at least 3 points"):
        planar.validate_grid(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="monotonically increasing"):
        planar.validate_grid(np.array([0.0, 0.5, 0.5, 1.0]))

    # Shape and parameter checks on operators
    valid_grid = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="must match grid shape"):
        planar.laplacian(np.ones(5), valid_grid)
    with pytest.raises(ValueError, match="Diffusivity must be positive"):
        planar.surface_flux(np.ones(11), valid_grid, diffusivity=0.0)
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        planar.surface_flux(np.ones(11), valid_grid, diffusivity=1.0, side="top")
    with pytest.raises(ValueError, match="incompatible with grid"):
        planar.surface_flux(np.ones(5), valid_grid, diffusivity=1.0)
    with pytest.raises(ValueError, match="n_electrons must be positive"):
        planar.current(1.0, n_electrons=0)
