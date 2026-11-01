"""Uniform 1D spatial discretization mesh implementation."""

from __future__ import annotations

import math

import numpy as np

from .base import Mesh1D, diffusion_layer_thickness


class Uniform1DMesh(Mesh1D):
    r"""Uniform 1D spatial discretization mesh with equidistant nodes.

    Generates spatial coordinate nodes :math:`x_j` with constant spacing:

    .. math::

        x_j = x_{\min} + j \cdot \Delta x, \quad j = 0, 1, \dots, N-1

    where :math:`\Delta x = (x_{\max} - x_{\min}) / (N - 1)`.

    Can be constructed either by specifying the total number of points :math:`N`
    (``n_points``) or the desired uniform spacing :math:`\Delta x` (``dx``).

    Parameters
    ----------
    x_min : float
        Lower boundary of the spatial domain (:math:`x_{\min}`).
    x_max : float
        Upper boundary of the spatial domain (:math:`x_{\max}`).
        Must be strictly greater than ``x_min``.
    n_points : int | None, optional
        Total number of grid nodes :math:`N`. Must be :math:`\ge 2`.
        Exactly one of ``n_points`` or ``dx`` must be specified.
    dx : float | None, optional
        Target uniform node spacing :math:`\Delta x`. Must be strictly positive
        and evenly divide the domain length (:math:`x_{\max} - x_{\min}`).
        Exactly one of ``n_points`` or ``dx`` must be specified.

    Attributes
    ----------
    nodes : np.ndarray
        1D array of spatial node positions of length :math:`N`.
    dx : np.ndarray
        Array of consecutive node intervals of length :math:`N-1`, where
        every element equals :math:`\Delta x`.
    spacing : float
        Constant scalar grid spacing :math:`\Delta x`.
    step : float
        Alias for ``spacing``.
    n_points : int
        Total number of spatial nodes :math:`N`.
    bounds : tuple[float, float]
        Spatial domain boundaries :math:`(x_{\min}, x_{\max})`.
    is_uniform : bool
        Always ``True`` for uniform meshes.

    Raises
    ------
    ValueError
        If ``x_max <= x_min``.
        If neither ``n_points`` nor ``dx`` is specified, or if both are specified.
        If ``n_points < 2``.
        If ``dx <= 0`` or ``dx > (x_max - x_min)``.
        If domain length is not an integer multiple of ``dx``.
    TypeError
        If ``n_points`` is not an integer, or if inputs cannot be converted to float.

    Examples
    --------
    Constructing with point count:

    >>> from softpotato.grid import Uniform1DMesh
    >>> mesh = Uniform1DMesh(0.0, 1.0, n_points=5)
    >>> mesh.nodes
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> mesh.spacing
    0.25
    >>> mesh.is_uniform
    True

    Constructing with spacing step:

    >>> mesh = Uniform1DMesh(0.0, 1.0, dx=0.2)
    >>> mesh.n_points
    6
    >>> mesh.bounds
    (0.0, 1.0)
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int | None = None,
        dx: float | None = None,
    ) -> None:
        try:
            x_min_f = float(x_min)
            x_max_f = float(x_max)
        except (TypeError, ValueError) as exc:
            raise TypeError("x_min and x_max must be real numbers.") from exc

        if not (math.isfinite(x_min_f) and math.isfinite(x_max_f)):
            raise ValueError(f"x_min and x_max must be finite, got ({x_min}, {x_max}).")

        if x_max_f <= x_min_f:
            raise ValueError(
                f"x_max ({x_max_f}) must be strictly greater than x_min ({x_min_f})."
            )

        if n_points is None and dx is None:
            raise ValueError("Must specify either 'n_points' or 'dx'.")
        if n_points is not None and dx is not None:
            raise ValueError("Cannot specify both 'n_points' and 'dx'.")

        domain_len = x_max_f - x_min_f

        if n_points is not None:
            if isinstance(n_points, bool) or not isinstance(
                n_points, (int, np.integer)
            ):
                raise TypeError(
                    f"n_points must be an integer, got {type(n_points).__name__}."
                )
            if n_points < 2:
                raise ValueError(f"n_points must be >= 2, got {n_points}.")
            n_pts = int(n_points)
            spacing = domain_len / (n_pts - 1)
        else:
            assert dx is not None
            try:
                dx_f = float(dx)
            except (TypeError, ValueError) as exc:
                raise TypeError("dx must be a real number.") from exc

            if not math.isfinite(dx_f):
                raise ValueError(f"dx must be finite, got {dx}.")
            if dx_f <= 0.0:
                raise ValueError(f"dx must be strictly positive, got {dx_f}.")
            if dx_f > domain_len:
                raise ValueError(
                    f"dx ({dx_f}) cannot be greater than the domain length ({domain_len})."
                )

            n_intervals_float = domain_len / dx_f
            n_intervals = round(n_intervals_float)
            if n_intervals < 1 or not np.isclose(
                n_intervals * dx_f, domain_len, rtol=1e-5, atol=1e-8
            ):
                raise ValueError(
                    f"Domain length ({domain_len}) is not an integer multiple of dx ({dx_f})."
                )
            n_pts = int(n_intervals) + 1
            spacing = dx_f

        nodes = np.linspace(x_min_f, x_max_f, n_pts, dtype=float)
        nodes.flags.writeable = False

        self._nodes = nodes
        self._spacing = float(spacing)
        self._dx = np.diff(self._nodes)
        self._dx.flags.writeable = False

    @classmethod
    def from_diffusion_layer(
        cls,
        D: float,
        t_max: float,
        *,
        x_min: float = 0.0,
        factor: float = 6.0,
        n_points: int | None = None,
        dx: float | None = None,
    ) -> Uniform1DMesh:
        r"""Create a uniform mesh sized to the diffusion layer thickness.

        Sets :math:`x_{\max} = x_{\min} + \text{factor} \cdot \sqrt{D \, t_{\max}}`,
        where ``factor`` defaults to :math:`6.0` (:math:`\delta \approx 6 \sqrt{D t_{\max}}`).

        Parameters
        ----------
        D : float
            Diffusion coefficient :math:`D > 0` (:math:`\text{m}^2/\text{s}`).
        t_max : float
            Maximum characteristic simulation time :math:`t_{\max} > 0` (:math:`\text{s}`).
        x_min : float, default 0.0
            Electrode surface coordinate :math:`x_{\min}`.
        factor : float, default 6.0
            Dimensionless multiplier for diffusion layer thickness
            :math:`\delta = \text{factor} \cdot \sqrt{D t_{\max}}`.
        n_points : int | None, optional
            Total number of grid nodes :math:`N`. Defaults to ``100`` if neither
            ``n_points`` nor ``dx`` is specified.
        dx : float | None, optional
            Uniform grid spacing :math:`\Delta x`.

        Returns
        -------
        Uniform1DMesh
            Configured uniform mesh spanning :math:`[x_{\min}, x_{\min} + \delta]`.
        """
        delta = diffusion_layer_thickness(D, t_max, factor=factor)
        if isinstance(x_min, bool):
            raise TypeError("x_min must be a real number.")
        try:
            x_min_f = float(x_min)
        except (TypeError, ValueError) as exc:
            raise TypeError("x_min must be a real number.") from exc

        if n_points is None and dx is None:
            n_points = 100

        return cls(
            x_min=x_min_f,
            x_max=x_min_f + delta,
            n_points=n_points,
            dx=dx,
        )

    @property
    def nodes(self) -> np.ndarray:
        """1D array of spatial node positions."""
        return self._nodes

    @property
    def dx(self) -> np.ndarray:
        """Interval spacings between consecutive nodes."""
        return self._dx

    @property
    def spacing(self) -> float:
        """Constant grid spacing Delta x."""
        return self._spacing

    @property
    def step(self) -> float:
        """Alias for constant grid spacing Delta x."""
        return self._spacing

    @property
    def is_uniform(self) -> bool:
        """Whether the grid spacing is uniform within numerical tolerance."""
        return True

    def __repr__(self) -> str:
        return (
            f"Uniform1DMesh(x_min={self.bounds[0]}, x_max={self.bounds[1]}, "
            f"n_points={self.n_points}, dx={self.spacing:.4g})"
        )

    def __str__(self) -> str:
        return (
            f"Uniform1DMesh(bounds=({self.bounds[0]}, {self.bounds[1]}), "
            f"n_points={self.n_points}, dx={self.spacing:.4g})"
        )
