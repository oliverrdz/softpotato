"""Planar 1D Cartesian coordinate geometry implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Geometry


class PlanarGeometry(Geometry, name="planar"):
    r"""1D planar Cartesian coordinate geometry (:math:`p = 0`).

    Models semi-infinite or finite linear diffusion perpendicular to a flat macroelectrode
    along spatial coordinate :math:`x \in [x_0, x_{\max}]`:

    .. math::

        \nabla^2 c = \frac{\partial^2 c}{\partial x^2}

    The active electrode surface area is constant (:math:`A`), yielding Faradaic current:

    .. math::

        I = n F A J, \quad J_{\text{left}} = -D \left.\frac{\partial c}{\partial x}\right|_{x_0}

    Parameters
    ----------
    area : float, default 1.0
        Active electrode surface area :math:`A` in :math:`\text{m}^2` or :math:`\text{cm}^2`
        (:math:`A > 0`).

    Attributes
    ----------
    name : str
        Geometry identifier (``'planar'``).
    dimension : int
        Spatial dimension (``1``).
    coordinate_name : str
        Primary coordinate symbol (``'x'``).
    p : int
        Symmetry factor (``0``).

    Raises
    ------
    ValueError
        If ``area <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.geometry import PlanarGeometry
    >>> geom = PlanarGeometry(area=0.5)
    >>> x = np.linspace(0.0, 1.0, 21)
    >>> c = x**2  # d²(x²)/dx² = 2.0 everywhere
    >>> lap = geom.laplacian(c, x)
    >>> bool(np.allclose(lap, 2.0))
    True
    >>> flux = geom.surface_flux(x, x, diffusivity=1e-5, side="left")
    >>> round(flux, 8)
    -1e-05
    """

    name: str = "planar"
    dimension: int = 1
    coordinate_name: str = "x"
    p: int = 0

    def __init__(self, area: float = 1.0) -> None:
        if area <= 0.0:
            raise ValueError(f"Electrode area must be positive (area > 0), got {area}.")
        self._area = float(area)

    @property
    def area(self) -> float:
        r"""Active electrode surface area :math:`A`."""
        return self._area

    def laplacian(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
        t: float = 0.0,
    ) -> np.ndarray:
        r"""Evaluate the 1D Cartesian Laplacian :math:`\nabla^2 c = \partial^2 c / \partial x^2`.

        Parameters
        ----------
        c : np.ndarray
            1D concentration array of shape ``(N,)``.
        grid : np.ndarray
            1D spatial grid :math:`x` of shape ``(N,)``.
        bc_left : BoundaryCondition, optional
            Left boundary condition at :math:`x_0`.
        bc_right : BoundaryCondition, optional
            Right boundary condition at :math:`x_{N-1}`.
        t : float, default 0.0
            Time in seconds for evaluating time-dependent boundary conditions.

        Returns
        -------
        np.ndarray
            1D array of :math:`\partial^2 c / \partial x^2` values of shape ``(N,)``.
        """
        return self._eval_1d_laplacian(c, grid, bc_left=bc_left, bc_right=bc_right, t=t)

    def laplacian_matrix(
        self,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Assemble banded matrix representation of :math:`\partial^2 / \partial x^2`.

        Parameters
        ----------
        grid : np.ndarray
            1D spatial grid :math:`x` of shape ``(N,)``.
        bc_left : BoundaryCondition, optional
            Left boundary condition at :math:`x_0`.
        bc_right : BoundaryCondition, optional
            Right boundary condition at :math:`x_{N-1}`.

        Returns
        -------
        L_banded : np.ndarray
            Banded tridiagonal operator matrix of shape ``(3, N)``.
        bc_scale : np.ndarray
            Boundary gradient scaling vector of shape ``(N,)``.
        """
        return self._build_1d_laplacian_matrix(grid, bc_left=bc_left, bc_right=bc_right)

    def surface_flux(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        diffusivity: float,
        side: str = "left",
    ) -> float | np.ndarray:
        r"""Calculate normal interfacial diffusion flux :math:`J = -D (\partial c / \partial n)`.

        Parameters
        ----------
        c : np.ndarray
            Concentration array of shape ``(N_x,)`` or ``(N_t, N_x)``.
        grid : np.ndarray
            1D spatial grid :math:`x` of shape ``(N_x,)``.
        diffusivity : float
            Species diffusion coefficient :math:`D > 0`.
        side : {'left', 'right'}, default 'left'
            Boundary side at which to evaluate the normal flux.

        Returns
        -------
        float or np.ndarray
            Normal interfacial diffusion flux :math:`J`.
        """
        return self._eval_1d_surface_flux(c, grid, diffusivity=diffusivity, side=side)

    def current(
        self,
        flux: float | np.ndarray,
        n_electrons: int = 1,
    ) -> float | np.ndarray:
        r"""Compute Faradaic current :math:`I = n F A J`.

        Parameters
        ----------
        flux : float or np.ndarray
            Interfacial diffusion flux :math:`J`.
        n_electrons : int, default 1
            Number of electrons transferred per reaction (:math:`n > 0`).

        Returns
        -------
        float or np.ndarray
            Faradaic current :math:`I` in Amperes (:math:`\text{A}`).
        """
        return self._eval_1d_current(flux, n_electrons=n_electrons)


# Register common aliases for planar geometry
Geometry._registry["cartesian"] = PlanarGeometry
Geometry._registry["1d"] = PlanarGeometry
Geometry._registry["linear"] = PlanarGeometry

