"""Abstract base class and registry for spatial coordinate geometries."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np

from softpotato.constants import FARADAY


class UnknownGeometryError(KeyError, ValueError):
    """Exception raised when an unrecognized geometry identifier is requested."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class _GeometryMeta(ABCMeta):
    """Metaclass for Geometry ensuring compatibility across Python versions.

    Intercepts ``name`` or ``geometry_name`` keyword arguments during class
    creation before delegating to ``ABCMeta.__new__``, avoiding parameter name
    collisions in Python <= 3.10.
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        geom_name = kwargs.pop("name", None)
        if geom_name is None:
            geom_name = kwargs.pop("geometry_name", None)
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if geom_name is not None and hasattr(cls, "_registry"):
            cls._registry[geom_name.lower()] = cls
        return cls


class Geometry(ABC, metaclass=_GeometryMeta):
    r"""Abstract base class representing spatial domain coordinate geometry.

    Encapsulates coordinate metrics, finite-difference differential operators
    (:math:`\nabla^2`, :math:`\nabla`), normal interfacial flux evaluation, and
    active electrode surface area integration for 1D and curvilinear coordinate
    systems:

    .. math::

        \nabla^2 c = \frac{1}{r^p} \frac{\partial}{\partial r}
                     \left( r^p \frac{\partial c}{\partial r} \right)
                   = \frac{\partial^2 c}{\partial r^2} + \frac{p}{r} \frac{\partial c}{\partial r}

    where :math:`p = 0` corresponds to planar Cartesian coordinates (:math:`x`),
    :math:`p = 1` to cylindrical radial coordinates (:math:`r`), and :math:`p = 2`
    to spherical radial coordinates (:math:`r`).

    Attributes
    ----------
    name : str
        Canonical identifier of the coordinate geometry (e.g., ``'planar'``,
        ``'spherical'``, ``'cylindrical'``).
    dimension : int
        Number of active spatial dimensions discretized by the geometry.
    coordinate_name : str
        Symbol of the primary spatial coordinate (e.g., ``'x'`` or ``'r'``).
    p : int
        Radial symmetry factor (:math:`p = 0` planar, :math:`p = 1` cylindrical,
        :math:`p = 2` spherical).
    """

    _registry: ClassVar[dict[str, type[Geometry]]] = {}

    name: str = "geometry"
    dimension: int = 1
    coordinate_name: str = "x"
    p: int = 0

    def __init_subclass__(
        cls,
        name: str | None = None,
        geometry_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        resolved_name = name or geometry_name
        if resolved_name:
            cls._registry[resolved_name.lower()] = cls

    @property
    @abstractmethod
    def area(self) -> float:
        r"""Active electrode surface area :math:`A` in :math:`\text{m}^2` or :math:`\text{cm}^2`."""
        raise NotImplementedError

    @property
    def surface_area(self) -> float:
        r"""Active electrode surface area :math:`A` (alias for :attr:`area`)."""
        return self.area

    @property
    def symmetry_factor(self) -> int:
        r"""Radial coordinate symmetry exponent :math:`p \in \{0, 1, 2\}`."""
        return self.p

    def validate_grid(self, grid: np.ndarray) -> np.ndarray:
        """Validate that a spatial grid satisfies the coordinate domain constraints.

        Parameters
        ----------
        grid : np.ndarray
            1D array of spatial coordinates.

        Returns
        -------
        np.ndarray
            Validated 1D floating-point coordinate grid.

        Raises
        ------
        ValueError
            If ``grid`` is not 1D, has fewer than 3 points, is not strictly
            monotonically increasing, or violates radial domain bounds (:math:`r_0 > 0`).
        """
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 1 or len(arr) < 3:
            raise ValueError("grid must be a 1D array with at least 3 points.")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("grid points must be strictly monotonically increasing.")
        if self.p > 0 and arr[0] <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} requires strictly positive radial coordinates "
                f"(r0 = grid[0] > 0), got grid[0]={arr[0]}."
            )
        return arr

    @abstractmethod
    def laplacian(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
        t: float = 0.0,
    ) -> np.ndarray:
        r"""Evaluate the spatial Laplacian :math:`\nabla^2 c` at interior and boundary points.

        Parameters
        ----------
        c : np.ndarray
            1D concentration profile array of shape ``(N,)``.
        grid : np.ndarray
            1D spatial coordinate array of shape ``(N,)``.
        bc_left : BoundaryCondition, optional
            Boundary condition at the inner/left boundary (:math:`i = 0`).
        bc_right : BoundaryCondition, optional
            Boundary condition at the outer/right boundary (:math:`i = N - 1`).
        t : float, default 0.0
            Evaluation time in seconds for time-dependent boundary conditions.

        Returns
        -------
        np.ndarray
            1D array of :math:`\nabla^2 c` values of shape ``(N,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def laplacian_matrix(
        self,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return banded matrix representation of :math:`\nabla^2` for implicit solvers.

        Constructs the tridiagonal finite-difference operator :math:`\mathbf{L}` in
        ``scipy.linalg.solve_banded`` ``(1, 1)`` format of shape ``(3, N)``, together
        with a boundary gradient scaling vector :math:`\mathbf{b}_{\text{scale}}` of
        shape ``(N,)`` such that for Neumann boundary gradients :math:`g_{\text{left}}`,
        :math:`g_{\text{right}}`:

        .. math::

            \nabla^2 \mathbf{c} = \mathbf{L}\mathbf{c}
                + \mathbf{b}_{\text{scale}} \odot [g_{\text{left}}, 0, \dots, 0, g_{\text{right}}]^T

        Parameters
        ----------
        grid : np.ndarray
            1D spatial coordinate array of shape ``(N,)``.
        bc_left : BoundaryCondition, optional
            Boundary condition at the left boundary (:math:`i = 0`).
        bc_right : BoundaryCondition, optional
            Boundary condition at the right boundary (:math:`i = N - 1`).

        Returns
        -------
        L_banded : np.ndarray
            Banded tridiagonal matrix of shape ``(3, N)`` where row 0 holds the
            superdiagonal, row 1 holds the main diagonal, and row 2 holds the subdiagonal.
        bc_scale : np.ndarray
            1D boundary gradient multiplier array of shape ``(N,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def surface_flux(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        diffusivity: float,
        side: str = "left",
    ) -> float | np.ndarray:
        r"""Calculate normal interfacial diffusion flux :math:`J = -D (\partial c / \partial n)` at a boundary.

        Follows the electrode-normal convention where the unit normal :math:`\mathbf{n}`
        points from the boundary into the solution domain (:math:`+\hat{x}` or :math:`+\hat{r}`
        at the left boundary, and :math:`-\hat{x}` or :math:`-\hat{r}` at the right boundary).

        Parameters
        ----------
        c : np.ndarray
            Concentration profile of shape ``(N_x,)`` or time series of shape ``(N_t, N_x)``.
        grid : np.ndarray
            1D spatial coordinate array of shape ``(N_x,)``.
        diffusivity : float
            Species diffusion coefficient :math:`D > 0`.
        side : {'left', 'right'}, default 'left'
            Domain boundary at which to evaluate the normal flux.

        Returns
        -------
        float or np.ndarray
            Interfacial flux :math:`J` (scalar ``float`` if ``c`` is 1D, or 1D
            ``np.ndarray`` of shape ``(N_t,)`` if ``c`` is 2D).
        """
        raise NotImplementedError

    @abstractmethod
    def current(
        self,
        flux: float | np.ndarray,
        n_electrons: int = 1,
        side: Literal["left", "right"] = "left",
        grid: np.ndarray | None = None,
    ) -> float | np.ndarray:
        r"""Integrate normal flux over the active electrode surface to compute Faradaic current.

        Computes the total Faradaic current :math:`I = n F A J` according to IUPAC
        sign conventions (cathodic reduction :math:`I < 0` when reactant flux into
        solution is negative, :math:`J < 0`).

        Parameters
        ----------
        flux : float or np.ndarray
            Normal interfacial diffusion flux :math:`J` at the electrode surface.
        n_electrons : int, default 1
            Number of electrons transferred per reaction (:math:`n > 0`).
        side : {'left', 'right'}, default 'left'
            Domain boundary at which to compute the Faradaic current.
        grid : np.ndarray, optional
            Spatial grid coordinates used to determine surface area if coordinate-dependent.

        Returns
        -------
        float or np.ndarray
            Faradaic current :math:`I` in Amperes (:math:`\text{A}`).
        """
        raise NotImplementedError

    def _eval_1d_laplacian(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Evaluate 1D Cartesian or curvilinear radial Laplacian with 2nd-order stencils."""
        grid_arr = self.validate_grid(grid)
        c_arr = np.asarray(c, dtype=float)
        if c_arr.shape != grid_arr.shape:
            raise ValueError(
                f"Concentration array shape {c_arr.shape} must match grid shape {grid_arr.shape}."
            )

        n_pts = len(grid_arr)
        diffs = np.diff(grid_arr)
        is_uniform = bool(np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8))
        p = self.p

        lap = np.empty(n_pts, dtype=float)

        # 1. Interior points (i = 1, ..., N - 2)
        if is_uniform:
            dr = float(diffs[0])
            inv_dr2 = 1.0 / (dr**2)
            d2c = (c_arr[2:] - 2.0 * c_arr[1:-1] + c_arr[:-2]) * inv_dr2
            if p == 0:
                lap[1:-1] = d2c
            else:
                dc = (c_arr[2:] - c_arr[:-2]) / (2.0 * dr)
                lap[1:-1] = d2c + (p / grid_arr[1:-1]) * dc
        else:
            hm1 = diffs[:-1]
            hi = diffs[1:]
            d2c = (2.0 / (hm1 + hi)) * (
                (c_arr[2:] - c_arr[1:-1]) / hi - (c_arr[1:-1] - c_arr[:-2]) / hm1
            )
            if p == 0:
                lap[1:-1] = d2c
            else:
                dc = (
                    (hm1**2) * (c_arr[2:] - c_arr[1:-1])
                    + (hi**2) * (c_arr[1:-1] - c_arr[:-2])
                ) / (hm1 * hi * (hm1 + hi))
                lap[1:-1] = d2c + (p / grid_arr[1:-1]) * dc

        # 2. Left boundary (i = 0)
        if bc_left is not None:
            btype_left = getattr(bc_left, "btype", None)
            if btype_left == "dirichlet":
                lap[0] = 0.0
            elif btype_left == "neumann":
                g_left = float(bc_left.evaluate(t))
                h0 = float(diffs[0])
                if is_uniform:
                    d2c_0 = (2.0 * c_arr[1] - 2.0 * c_arr[0] - 2.0 * h0 * g_left) / (
                        h0**2
                    )
                else:
                    d2c_0 = (2.0 / h0) * ((c_arr[1] - c_arr[0]) / h0 - g_left)
                lap[0] = d2c_0 if p == 0 else d2c_0 + (p / grid_arr[0]) * g_left
            else:
                raise ValueError(
                    f"Unsupported left boundary condition type: {btype_left!r}."
                )
        else:
            # Pure operator evaluation at i = 0 using one-sided 2nd-order stencils
            h0, h1 = float(diffs[0]), float(diffs[1])
            dc_0 = (
                -(2.0 * h0 + h1) / (h0 * (h0 + h1)) * c_arr[0]
                + (h0 + h1) / (h0 * h1) * c_arr[1]
                - h0 / (h1 * (h0 + h1)) * c_arr[2]
            )
            if n_pts >= 4:
                d1 = grid_arr[1] - grid_arr[0]
                d2 = grid_arr[2] - grid_arr[0]
                d3 = grid_arr[3] - grid_arr[0]
                d2c_0 = -2.0 * (
                    -(d1 + d2 + d3) / (d1 * d2 * d3) * c_arr[0]
                    + (d2 + d3) / (d1 * (d1 - d2) * (d1 - d3)) * c_arr[1]
                    + (d1 + d3) / (d2 * (d2 - d1) * (d2 - d3)) * c_arr[2]
                    + (d1 + d2) / (d3 * (d3 - d1) * (d3 - d2)) * c_arr[3]
                )
            else:
                d2c_0 = (2.0 / (h0 + h1)) * (
                    (c_arr[2] - c_arr[1]) / h1 - (c_arr[1] - c_arr[0]) / h0
                )
            lap[0] = d2c_0 if p == 0 else d2c_0 + (p / grid_arr[0]) * dc_0

        # 3. Right boundary (i = N - 1)
        if bc_right is not None:
            btype_right = getattr(bc_right, "btype", None)
            if btype_right == "dirichlet":
                lap[-1] = 0.0
            elif btype_right == "neumann":
                g_right = float(bc_right.evaluate(t))
                h_end = float(diffs[-1])
                if is_uniform:
                    d2c_end = (
                        2.0 * c_arr[-2] - 2.0 * c_arr[-1] + 2.0 * h_end * g_right
                    ) / (h_end**2)
                else:
                    d2c_end = (2.0 / h_end) * (
                        g_right - (c_arr[-1] - c_arr[-2]) / h_end
                    )
                lap[-1] = d2c_end if p == 0 else d2c_end + (p / grid_arr[-1]) * g_right
            else:
                raise ValueError(
                    f"Unsupported right boundary condition type: {btype_right!r}."
                )
        else:
            # Pure operator evaluation at i = N - 1 using one-sided 2nd-order stencils
            h_end, h_end2 = float(diffs[-1]), float(diffs[-2])
            dc_end = (
                (2.0 * h_end + h_end2) / (h_end * (h_end + h_end2)) * c_arr[-1]
                - (h_end + h_end2) / (h_end * h_end2) * c_arr[-2]
                + h_end / (h_end2 * (h_end + h_end2)) * c_arr[-3]
            )
            if n_pts >= 4:
                d1 = grid_arr[-2] - grid_arr[-1]
                d2 = grid_arr[-3] - grid_arr[-1]
                d3 = grid_arr[-4] - grid_arr[-1]
                d2c_end = -2.0 * (
                    -(d1 + d2 + d3) / (d1 * d2 * d3) * c_arr[-1]
                    + (d2 + d3) / (d1 * (d1 - d2) * (d1 - d3)) * c_arr[-2]
                    + (d1 + d3) / (d2 * (d2 - d1) * (d2 - d3)) * c_arr[-3]
                    + (d1 + d2) / (d3 * (d3 - d1) * (d3 - d2)) * c_arr[-4]
                )
            else:
                d2c_end = (2.0 / (h_end2 + h_end)) * (
                    (c_arr[-1] - c_arr[-2]) / h_end - (c_arr[-2] - c_arr[-3]) / h_end2
                )
            lap[-1] = d2c_end if p == 0 else d2c_end + (p / grid_arr[-1]) * dc_end

        return lap

    def _build_1d_laplacian_matrix(
        self,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble banded tridiagonal Laplacian matrix and boundary source scaling."""
        grid_arr = self.validate_grid(grid)
        n_pts = len(grid_arr)
        diffs = np.diff(grid_arr)
        is_uniform = bool(np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8))
        p = self.p

        l_banded = np.zeros((3, n_pts), dtype=float)
        bc_scale = np.zeros(n_pts, dtype=float)

        # Interior rows (i = 1, ..., N - 2)
        if is_uniform:
            dr = float(diffs[0])
            inv_dr2 = 1.0 / (dr**2)
            l_banded[1, 1:-1] = -2.0 * inv_dr2
            if p == 0:
                l_banded[0, 2:] = inv_dr2
                l_banded[2, :-2] = inv_dr2
            else:
                radial_term = p / (2.0 * grid_arr[1:-1] * dr)
                l_banded[0, 2:] = inv_dr2 + radial_term
                l_banded[2, :-2] = inv_dr2 - radial_term
        else:
            hm1 = diffs[:-1]
            hi = diffs[1:]
            denom_sum = hm1 + hi
            w2_lower = 2.0 / (hm1 * denom_sum)
            w2_diag = -2.0 / (hm1 * hi)
            w2_upper = 2.0 / (hi * denom_sum)

            if p == 0:
                l_banded[2, :-2] = w2_lower
                l_banded[1, 1:-1] = w2_diag
                l_banded[0, 2:] = w2_upper
            else:
                p_over_r = p / grid_arr[1:-1]
                w1_lower = -hi / (hm1 * denom_sum)
                w1_diag = (hi - hm1) / (hm1 * hi)
                w1_upper = hm1 / (hi * denom_sum)

                l_banded[2, :-2] = w2_lower + p_over_r * w1_lower
                l_banded[1, 1:-1] = w2_diag + p_over_r * w1_diag
                l_banded[0, 2:] = w2_upper + p_over_r * w1_upper

        # Left boundary (i = 0)
        btype_left = (
            getattr(bc_left, "btype", "neumann") if bc_left is not None else "neumann"
        )
        if btype_left == "dirichlet":
            l_banded[1, 0] = 0.0
            l_banded[0, 1] = 0.0
            bc_scale[0] = 0.0
        elif btype_left == "neumann":
            h0 = float(diffs[0])
            l_banded[1, 0] = -2.0 / (h0**2)
            l_banded[0, 1] = 2.0 / (h0**2)
            bc_scale[0] = -(2.0 / h0 - (p / grid_arr[0] if p > 0 else 0.0))
        else:
            raise ValueError(
                f"Unsupported left boundary condition type: {btype_left!r}."
            )

        # Right boundary (i = N - 1)
        btype_right = (
            getattr(bc_right, "btype", "neumann") if bc_right is not None else "neumann"
        )
        if btype_right == "dirichlet":
            l_banded[1, -1] = 0.0
            l_banded[2, -2] = 0.0
            bc_scale[-1] = 0.0
        elif btype_right == "neumann":
            hm1_end = float(diffs[-1])
            l_banded[1, -1] = -2.0 / (hm1_end**2)
            l_banded[2, -2] = 2.0 / (hm1_end**2)
            bc_scale[-1] = 2.0 / hm1_end + (p / grid_arr[-1] if p > 0 else 0.0)
        else:
            raise ValueError(
                f"Unsupported right boundary condition type: {btype_right!r}."
            )

        return l_banded, bc_scale

    def _eval_1d_surface_flux(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        diffusivity: float,
        side: str = "left",
    ) -> float | np.ndarray:
        """Compute 2nd-order one-sided interfacial diffusion flux at left or right boundary."""
        if diffusivity <= 0.0:
            raise ValueError(
                f"Diffusivity must be positive (diffusivity > 0), got {diffusivity}."
            )
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")

        grid_arr = self.validate_grid(grid)
        c_arr = np.asarray(c, dtype=float)
        if c_arr.ndim not in (1, 2) or c_arr.shape[-1] != len(grid_arr):
            raise ValueError(
                f"Concentration array shape {c_arr.shape} is incompatible with grid of length {len(grid_arr)}."
            )

        diffs = np.diff(grid_arr)
        is_uniform = bool(np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8))

        if side == "left":
            if is_uniform:
                dx = float(diffs[0])
                grad = (-3.0 * c_arr[..., 0] + 4.0 * c_arr[..., 1] - c_arr[..., 2]) / (
                    2.0 * dx
                )
            else:
                h0, h1 = float(diffs[0]), float(diffs[1])
                grad = (
                    -(2.0 * h0 + h1) / (h0 * (h0 + h1)) * c_arr[..., 0]
                    + (h0 + h1) / (h0 * h1) * c_arr[..., 1]
                    - h0 / (h1 * (h0 + h1)) * c_arr[..., 2]
                )
            flux = -diffusivity * grad
        else:
            if is_uniform:
                dx = float(diffs[0])
                grad = (
                    3.0 * c_arr[..., -1] - 4.0 * c_arr[..., -2] + c_arr[..., -3]
                ) / (2.0 * dx)
            else:
                hm1, hm2 = float(diffs[-1]), float(diffs[-2])
                grad = (
                    (2.0 * hm1 + hm2) / (hm1 * (hm1 + hm2)) * c_arr[..., -1]
                    - (hm1 + hm2) / (hm1 * hm2) * c_arr[..., -2]
                    + hm1 / (hm2 * (hm1 + hm2)) * c_arr[..., -3]
                )
            flux = diffusivity * grad

        if c_arr.ndim == 1:
            return float(flux)
        return np.asarray(flux, dtype=float)

    def _eval_1d_current(
        self,
        flux: float | np.ndarray,
        n_electrons: int = 1,
        area: float | None = None,
    ) -> float | np.ndarray:
        """Compute Faradaic current I = n * F * A * J."""
        if n_electrons <= 0:
            raise ValueError(
                f"Number of electrons n_electrons must be positive (n_electrons > 0), got {n_electrons}."
            )
        active_area = self.area if area is None else float(area)
        scale = float(n_electrons) * FARADAY * active_area
        if np.isscalar(flux):
            return float(scale * float(flux))  # type: ignore[arg-type]
        flux_arr = np.asarray(flux, dtype=float)
        if flux_arr.ndim == 0:
            return float(scale * float(flux_arr.item()))
        return scale * flux_arr


def register_geometry(name: str) -> Callable[[type[Geometry]], type[Geometry]]:
    """Decorator to register a custom geometry class under a given alias.

    Parameters
    ----------
    name : str
        Name identifier to associate with the decorated geometry class.

    Returns
    -------
    Callable[[type[Geometry]], type[Geometry]]
        Class decorator registering the subclass in ``Geometry._registry``.

    Examples
    --------
    >>> from softpotato.geometry import Geometry, register_geometry, get_geometry
    >>> @register_geometry("custom_planar_demo")
    ... class CustomPlanar(Geometry):
    ...     @property
    ...     def area(self) -> float:
    ...         return 2.0
    ...     def laplacian(self, c, grid, bc_left=None, bc_right=None, t=0.0):
    ...         return self._eval_1d_laplacian(c, grid, bc_left, bc_right, t)
    ...     def laplacian_matrix(self, grid, bc_left=None, bc_right=None):
    ...         return self._build_1d_laplacian_matrix(grid, bc_left, bc_right)
    ...     def surface_flux(self, c, grid, diffusivity, side="left"):
    ...         return self._eval_1d_surface_flux(c, grid, diffusivity, side)
    ...     def current(self, flux, n_electrons=1):
    ...         return self._eval_1d_current(flux, n_electrons)
    >>> geom = get_geometry("custom_planar_demo")
    >>> geom.area
    2.0
    """

    def decorator(cls: type[Geometry]) -> type[Geometry]:
        if not issubclass(cls, Geometry):
            raise TypeError(f"Class {cls.__name__} must inherit from Geometry.")
        Geometry._registry[name.lower()] = cls
        return cls

    return decorator


def get_geometry(
    name_or_instance: str | Geometry = "planar",
    **kwargs: Any,
) -> Geometry:
    """Instantiate or resolve a spatial coordinate geometry by name or instance.

    Parameters
    ----------
    name_or_instance : str or Geometry, default 'planar'
        Registered geometry identifier (case-insensitive) such as ``'planar'``,
        ``'spherical'``, ``'hemispherical'``, or ``'cylindrical'``, or an
        already-instantiated :class:`~softpotato.geometry.Geometry` object.
    **kwargs : Any
        Configuration parameters forwarded to the geometry constructor:

        - For ``'planar'`` / ``'cartesian'``:
          - ``area`` (float, default 1.0): Electrode surface area :math:`A`.
        - For ``'spherical'`` / ``'sphere'`` / ``'hemispherical'`` / ``'hemisphere'``:
          - ``r0`` or ``radius`` (float, default 1.0): Electrode radius :math:`r_0 > 0`.
          - ``hemisphere`` (bool, default False): If True, uses hemispherical surface
            area :math:`A = 2\\pi r_0^2` instead of full sphere :math:`A = 4\\pi r_0^2`.
          - ``area`` (float, optional): Explicit electrode surface area override.
        - For ``'cylindrical'`` / ``'cylinder'``:
          - ``r0`` or ``radius`` (float, default 1.0): Cylinder radius :math:`r_0 > 0`.
          - ``length`` or ``L`` (float, default 1.0): Cylinder length :math:`L > 0`.
          - ``area`` (float, optional): Explicit electrode surface area override.

    Returns
    -------
    Geometry
        Resolved :class:`~softpotato.geometry.Geometry` instance.

    Raises
    ------
    UnknownGeometryError
        If ``name_or_instance`` is an unrecognized string identifier (inherits from
        both :exc:`KeyError` and :exc:`ValueError`).
    TypeError
        If ``name_or_instance`` is neither a string nor a :class:`~softpotato.geometry.Geometry`.
    ValueError
        If keyword arguments are supplied when ``name_or_instance`` is already a
        :class:`~softpotato.geometry.Geometry` instance.

    Examples
    --------
    >>> from softpotato.geometry import get_geometry
    >>> planar = get_geometry("planar", area=0.05)
    >>> planar.area
    0.05
    >>> sphere = get_geometry("spherical", r0=1e-4)
    >>> sphere.coordinate_name
    'r'
    >>> get_geometry(planar) is planar
    True
    """
    if isinstance(name_or_instance, Geometry):
        if kwargs:
            raise ValueError(
                "Cannot pass constructor keyword arguments when name_or_instance is already a Geometry instance."
            )
        return name_or_instance

    if not isinstance(name_or_instance, str):
        raise TypeError(
            f"name_or_instance must be a string or Geometry instance, got {type(name_or_instance).__name__}."
        )

    normalized = name_or_instance.strip().lower()
    if normalized in ("hemisphere", "hemispherical"):
        kwargs.setdefault("hemisphere", True)
        normalized = "spherical"

    if normalized not in Geometry._registry:
        available = ", ".join(sorted(Geometry._registry.keys()))
        raise UnknownGeometryError(
            f"Unknown geometry '{name_or_instance}'. Available geometries: {available}"
        )
    return Geometry._registry[normalized](**kwargs)


def list_geometries() -> list[str]:
    """Return a sorted list of all registered geometry names and aliases.

    Returns
    -------
    list[str]
        Alphabetically sorted list of registered geometry identifiers.

    Examples
    --------
    >>> from softpotato.geometry import list_geometries
    >>> geoms = list_geometries()
    >>> "planar" in geoms and "spherical" in geoms and "cylindrical" in geoms
    True
    """
    return sorted(Geometry._registry.keys())
