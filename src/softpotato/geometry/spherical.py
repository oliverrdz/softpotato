"""Spherical 1D radial coordinate geometry implementation."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .base import Geometry


class SphericalGeometry(Geometry, name="spherical"):
    r"""1D spherical radial coordinate geometry (:math:`p = 2`).

    Models radial diffusion to or from a spherical microelectrode, hanging mercury
    drop electrode (HMDE), or hemispherical microelectrode on an insulating plane
    across radial domain :math:`r \in [r_0, r_{\max}]` with :math:`r_0 > 0`:

    .. math::

        \nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{2}{r} \frac{\partial c}{\partial r}
                   = \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial c}{\partial r} \right)

    The active electrode surface area at :math:`r = r_0` is:

    - **Full sphere** (``hemisphere=False``): :math:`A = 4 \pi r_0^2`
    - **Hemisphere** (``hemisphere=True``): :math:`A = 2 \pi r_0^2`

    Parameters
    ----------
    r0 : float, optional
        Electrode radius :math:`r_0 > 0` in meters or cm. Defaults to ``1.0`` until
        bound to a radial coordinate grid.
    hemisphere : bool, default False
        If True, computes surface area for a hemisphere (:math:`A = 2 \pi r_0^2`)
        instead of a full sphere (:math:`A = 4 \pi r_0^2`).
    radius : float, optional
        Keyword alias for ``r0``.
    mode : {'sphere', 'hemisphere'}, optional
        Alternative specification of full sphere vs. hemisphere geometry.
    area : float, optional
        Explicit surface area override :math:`A > 0`. If None, computed analytically
        from :math:`r_0` and ``hemisphere``.

    Attributes
    ----------
    name : str
        Geometry identifier (``'spherical'``).
    dimension : int
        Spatial dimension (``1``).
    coordinate_name : str
        Primary coordinate symbol (``'r'``).
    p : int
        Symmetry factor (``2``).
    r0 : float
        Electrode radius :math:`r_0 > 0`.
    hemisphere : bool
        Whether the electrode is a hemisphere on an insulating substrate.

    Raises
    ------
    ValueError
        If ``r0 <= 0``, ``area <= 0``, or ``mode`` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from softpotato.geometry import SphericalGeometry
    >>> geom = SphericalGeometry(r0=1e-4)
    >>> r = np.linspace(1.0, 5.0, 201)
    >>> c = 1.0 / r  # Harmonic profile in spherical coordinates: ∇²(1/r) = 0
    >>> lap = geom.laplacian(c, r)
    >>> bool(np.max(np.abs(lap[1:-1])) < 1e-4)
    True
    """

    name: str = "spherical"
    dimension: int = 1
    coordinate_name: str = "r"
    p: int = 2

    def __init__(
        self,
        r0: float | None = None,
        hemisphere: bool = False,
        *,
        radius: float | None = None,
        mode: Literal["sphere", "hemisphere"] | None = None,
        area: float | None = None,
    ) -> None:
        if r0 is not None and radius is not None and r0 != radius:
            raise ValueError(
                "Cannot specify both 'r0' and 'radius' with different values."
            )

        resolved_r0 = radius if radius is not None else r0
        if resolved_r0 is not None:
            if resolved_r0 <= 0.0:
                raise ValueError(
                    f"Electrode radius r0 must be positive (r0 > 0), got {resolved_r0}."
                )
            self.r0 = float(resolved_r0)
            self._r0_explicit = True
        else:
            self.r0 = 1.0
            self._r0_explicit = False

        if mode is not None:
            if mode not in ("sphere", "hemisphere"):
                raise ValueError(
                    f"mode must be 'sphere' or 'hemisphere', got {mode!r}."
                )
            self.hemisphere = (mode == "hemisphere") or bool(hemisphere)
        else:
            self.hemisphere = bool(hemisphere)

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
        r"""Electrode radius :math:`r_0` (alias for :attr:`r0`)."""
        return self.r0

    @property
    def area(self) -> float:
        r"""Active electrode surface area :math:`A = 4\pi r_0^2` (or :math:`2\pi r_0^2` for hemisphere)."""
        if self._explicit_area is not None:
            return self._explicit_area
        factor = 2.0 * math.pi if self.hemisphere else 4.0 * math.pi
        return factor * (self.r0**2)

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
        r"""Evaluate the spherical Laplacian :math:`\nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{2}{r}\frac{\partial c}{\partial r}`.

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
        r"""Assemble banded matrix representation of the spherical Laplacian :math:`\nabla^2`.

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
        r"""Integrate normal flux over the spherical or hemispherical surface :math:`I = n F A J`.

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


# Register common aliases for spherical geometry
Geometry._registry["sphere"] = SphericalGeometry
Geometry._registry["hemispherical"] = SphericalGeometry
Geometry._registry["hemisphere"] = SphericalGeometry
