"""Cylindrical 1D radial coordinate geometry implementation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Geometry


class CylindricalGeometry(Geometry, name="cylindrical"):
    r"""1D cylindrical radial coordinate geometry (:math:`p = 1`).

    Models axisymmetric radial diffusion to or from a cylindrical wire or carbon
    fiber microelectrode of radius :math:`r_0 > 0` and length :math:`L > 0` across
    radial domain :math:`r \in [r_0, r_{\max}]`:

    .. math::

        \nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{1}{r} \frac{\partial c}{\partial r}
                   = \frac{1}{r} \frac{\partial}{\partial r} \left( r \frac{\partial c}{\partial r} \right)

    Neglecting end effects (:math:`L \gg r_0`), the active lateral electrode surface area
    at :math:`r = r_0` is:

    .. math::

        A = 2 \pi r_0 L, \quad I = n F A J

    Parameters
    ----------
    r0 : float, optional
        Cylinder radius :math:`r_0 > 0` in meters or cm. Defaults to ``1.0`` until
        bound to a radial coordinate grid.
    length : float, default 1.0
        Active cylinder length :math:`L > 0` in meters or cm.
    radius : float, optional
        Keyword alias for ``r0``.
    L : float, optional
        Keyword alias for ``length``.
    area : float, optional
        Explicit surface area override :math:`A > 0`. If None, computed analytically
        as :math:`A = 2 \pi r_0 L`.

    Attributes
    ----------
    name : str
        Geometry identifier (``'cylindrical'``).
    dimension : int
        Spatial dimension (``1``).
    coordinate_name : str
        Primary coordinate symbol (``'r'``).
    p : int
        Symmetry factor (``1``).
    r0 : float
        Cylinder radius :math:`r_0 > 0`.
    length : float
        Cylinder length :math:`L > 0`.

    Raises
    ------
    ValueError
        If ``r0 <= 0``, ``length <= 0``, or ``area <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.geometry import CylindricalGeometry
    >>> geom = CylindricalGeometry(r0=5e-6, length=1e-2)
    >>> r = np.linspace(1.0, 5.0, 201)
    >>> c = np.log(r)  # Harmonic profile in cylindrical coordinates: ∇²(ln r) = 0
    >>> lap = geom.laplacian(c, r)
    >>> bool(np.max(np.abs(lap[1:-1])) < 1e-4)
    True
    """

    name: str = "cylindrical"
    dimension: int = 1
    coordinate_name: str = "r"
    p: int = 1

    def __init__(
        self,
        r0: float | None = None,
        length: float | None = None,
        *,
        radius: float | None = None,
        L: float | None = None,
        area: float | None = None,
    ) -> None:
        if r0 is not None and radius is not None and r0 != radius:
            raise ValueError(
                "Cannot specify both 'r0' and 'radius' with different values."
            )
        if length is not None and L is not None and length != L:
            raise ValueError(
                "Cannot specify both 'length' and 'L' with different values."
            )

        resolved_r0 = radius if radius is not None else r0
        if resolved_r0 is not None:
            if resolved_r0 <= 0.0:
                raise ValueError(
                    f"Cylinder radius r0 must be positive (r0 > 0), got {resolved_r0}."
                )
            self.r0 = float(resolved_r0)
            self._r0_explicit = True
        else:
            self.r0 = 1.0
            self._r0_explicit = False

        resolved_length = (
            1.0 if (length is None and L is None) else (L if L is not None else length)
        )
        assert resolved_length is not None
        if resolved_length <= 0.0:
            raise ValueError(
                f"Cylinder length must be positive (length > 0), got {resolved_length}."
            )
        self.length = float(resolved_length)

        if area is not None:
            if area <= 0.0:
                raise ValueError(
                    f"Electrode area must be positive (area > 0), got {area}."
                )
            self._explicit_area: float | None = float(area)
        else:
            self._explicit_area = None

    @property
    def radius(self) -> float:
        r"""Cylinder radius :math:`r_0` (alias for :attr:`r0`)."""
        return self.r0

    @property
    def L(self) -> float:
        r"""Cylinder length :math:`L` (alias for :attr:`length`)."""
        return self.length

    @property
    def area(self) -> float:
        r"""Active lateral cylinder surface area :math:`A = 2\pi r_0 L`."""
        if self._explicit_area is not None:
            return self._explicit_area
        return 2.0 * math.pi * self.r0 * self.length

    def validate_grid(self, grid: np.ndarray) -> np.ndarray:
        """Validate radial grid bounds (:math:`r_0 > 0`) and infer ``r0`` if not explicitly set."""
        arr = super().validate_grid(grid)
        if not self._r0_explicit:
            self.r0 = float(arr[0])
        return arr

    def laplacian(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
        t: float = 0.0,
    ) -> np.ndarray:
        r"""Evaluate the cylindrical Laplacian :math:`\nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{1}{r}\frac{\partial c}{\partial r}`.

        Parameters
        ----------
        c : np.ndarray
            1D concentration array of shape ``(N,)``.
        grid : np.ndarray
            1D radial coordinate grid :math:`r \in [r_0, r_{\max}]` of shape ``(N,)`` with :math:`r_0 > 0`.
        bc_left : BoundaryCondition, optional
            Inner boundary condition at :math:`r = r_0`.
        bc_right : BoundaryCondition, optional
            Outer boundary condition at :math:`r = r_{\max}`.
        t : float, default 0.0
            Time in seconds for evaluating time-dependent boundary conditions.

        Returns
        -------
        np.ndarray
            1D array of :math:`\nabla^2 c` values of shape ``(N,)``.
        """
        return self._eval_1d_laplacian(c, grid, bc_left=bc_left, bc_right=bc_right, t=t)

    def laplacian_matrix(
        self,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Assemble banded matrix representation of the cylindrical Laplacian :math:`\nabla^2`.

        Parameters
        ----------
        grid : np.ndarray
            1D radial grid :math:`r` of shape ``(N,)`` with :math:`r_0 > 0`.
        bc_left : BoundaryCondition, optional
            Inner boundary condition at :math:`r = r_0`.
        bc_right : BoundaryCondition, optional
            Outer boundary condition at :math:`r = r_{\max}`.

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
            Concentration array of shape ``(N_r,)`` or ``(N_t, N_r)``.
        grid : np.ndarray
            1D radial grid :math:`r` of shape ``(N_r,)`` with :math:`r_0 > 0`.
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
        r"""Integrate normal flux over the cylindrical electrode surface :math:`I = n F A J`.

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


# Register common aliases for cylindrical geometry
Geometry._registry["cylinder"] = CylindricalGeometry
Geometry._registry["wire"] = CylindricalGeometry
