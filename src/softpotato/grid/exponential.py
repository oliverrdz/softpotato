"""Exponential and conformal 1D spatial discretization mesh implementation."""

from __future__ import annotations

import math

import numpy as np

from .base import Mesh1D, diffusion_layer_thickness

_VALID_MAPPINGS = ("exponential", "tanh", "sinh")


class ExponentialMesh(Mesh1D):
    r"""Exponential and conformal-mapping 1D spatial discretization mesh.

    Generates a smooth, non-uniform 1D spatial grid across :math:`[x_{\min}, x_{\max}]`
    parameterized by a stretching factor :math:`\beta > 0` applied to uniform
    computational coordinates :math:`\xi \in [0, 1]`.

    For the default ``"exponential"`` mapping (one-sided):

    .. math::

        x(\xi) = x_{\min} + (x_{\max} - x_{\min}) \frac{\exp(\beta \xi) - 1}{\exp(\beta) - 1},
        \quad \xi \in [0, 1]

    Also supports conformal / boundary-layer coordinate transformations:

    - ``"tanh"`` (hyperbolic tangent coordinate clustering):

      .. math::

          x(\xi) = x_{\min} + (x_{\max} - x_{\min})
          \left(1 - \frac{\tanh(\beta (1 - \xi))}{\tanh(\beta)}\right)

    - ``"sinh"`` (hyperbolic sine conformal stretching):

      .. math::

          x(\xi) = x_{\min} + (x_{\max} - x_{\min}) \frac{\sinh(\beta \xi)}{\sinh(\beta)}

    When ``two_sided=True``, nodes are symmetrically clustered near both
    boundaries (:math:`x_{\min}` and :math:`x_{\max}`) while coarsening toward
    the domain center.

    Parameters
    ----------
    x_min : float
        Lower boundary of the spatial domain (:math:`x_{\min}`).
    x_max : float
        Upper boundary of the spatial domain (:math:`x_{\max}`).
        Must be strictly greater than ``x_min``.
    n_points : int
        Total number of grid nodes :math:`N`. Must be :math:`\ge 2` (or :math:`\ge 3`
        when ``two_sided=True``).
    beta : float, default 3.0
        Dimensionless stretching parameter :math:`\beta > 0`. Larger values
        concentrate more nodes near the boundary.
    two_sided : bool, default False
        If ``True``, clusters nodes symmetrically at both :math:`x_{\min}` and
        :math:`x_{\max}`.
    mapping : str, default "exponential"
        Coordinate transformation function: ``"exponential"``, ``"tanh"``, or ``"sinh"``.

    Attributes
    ----------
    nodes : np.ndarray
        1D array of spatial node positions of length :math:`N`.
    dx : np.ndarray
        Interval spacings between consecutive nodes of length :math:`N-1`.
    xi : np.ndarray
        Uniform computational coordinates :math:`\xi \in [0, 1]` of length :math:`N`.
    beta : float
        Stretching parameter :math:`\beta > 0`.
    two_sided : bool
        Whether the mesh is symmetrically clustered at both boundaries.
    mapping : str
        Name of the coordinate mapping transformation.
    dx_min : float
        Minimum interval spacing across the mesh.
    dx_max : float
        Maximum interval spacing across the mesh.
    n_points : int
        Total number of spatial nodes :math:`N`.
    bounds : tuple[float, float]
        Spatial domain boundaries :math:`(x_{\min}, x_{\max})`.
    is_uniform : bool
        Always ``False`` for exponential meshes.

    Raises
    ------
    ValueError
        If ``x_max <= x_min``, ``n_points`` is too small, ``beta <= 0``,
        ``beta > 500``, or ``mapping`` is unknown.
    TypeError
        If inputs have invalid types.

    Examples
    --------
    >>> from softpotato.grid import ExponentialMesh
    >>> mesh = ExponentialMesh(0.0, 1.0, n_points=51, beta=3.0)
    >>> mesh.bounds
    (0.0, 1.0)
    >>> mesh.is_uniform
    False
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int,
        beta: float = 3.0,
        two_sided: bool = False,
        mapping: str = "exponential",
    ) -> None:
        if isinstance(x_min, (bool, str, bytes)) or isinstance(
            x_max, (bool, str, bytes)
        ):
            raise TypeError("x_min and x_max must be real numbers.")
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

        if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
            raise TypeError(
                f"n_points must be an integer, got {type(n_points).__name__}."
            )
        n_pts = int(n_points)

        if not isinstance(two_sided, bool):
            raise TypeError(
                f"two_sided must be a boolean, got {type(two_sided).__name__}."
            )

        min_pts = 3 if two_sided else 2
        if n_pts < min_pts:
            raise ValueError(f"n_points must be >= {min_pts}, got {n_pts}.")

        if isinstance(beta, (bool, str, bytes)):
            raise TypeError("beta must be a real number.")
        try:
            beta_f = float(beta)
        except (TypeError, ValueError) as exc:
            raise TypeError("beta must be a real number.") from exc

        if not math.isfinite(beta_f):
            raise ValueError(f"beta must be finite, got {beta}.")
        if beta_f <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta_f}.")
        if beta_f > 500.0:
            raise ValueError(
                f"beta ({beta_f}) is too large (must be <= 500.0 to avoid overflow)."
            )

        if not isinstance(mapping, str):
            raise TypeError(f"mapping must be a string, got {type(mapping).__name__}.")
        mapping_norm = mapping.strip().lower()
        if mapping_norm not in _VALID_MAPPINGS:
            raise ValueError(
                f"Unknown mapping '{mapping}'. Must be one of {_VALID_MAPPINGS}."
            )

        xi = np.linspace(0.0, 1.0, n_pts, dtype=float)
        eta = self._compute_normalized_coords(xi, beta_f, two_sided, mapping_norm)

        domain_len = x_max_f - x_min_f
        nodes = x_min_f + domain_len * eta
        nodes[0] = x_min_f
        nodes[-1] = x_max_f
        if two_sided and n_pts % 2 == 1:
            nodes[n_pts // 2] = 0.5 * (x_min_f + x_max_f)

        dx_arr = np.diff(nodes)
        xi.flags.writeable = False
        nodes.flags.writeable = False
        dx_arr.flags.writeable = False

        self._xi = xi
        self._nodes = nodes
        self._dx = dx_arr
        self._beta = beta_f
        self._two_sided = two_sided
        self._mapping = mapping_norm

    @staticmethod
    def _compute_normalized_coords(
        xi: np.ndarray,
        beta: float,
        two_sided: bool,
        mapping: str,
    ) -> np.ndarray:
        """Compute normalized coordinates eta in [0, 1] from uniform xi in [0, 1]."""
        if mapping == "exponential":
            denom = math.expm1(beta)
            if not two_sided:
                return np.expm1(beta * xi) / denom
            eta = np.empty_like(xi)
            left = xi <= 0.5
            right = ~left
            eta[left] = 0.5 * np.expm1(2.0 * beta * xi[left]) / denom
            eta[right] = 1.0 - 0.5 * np.expm1(2.0 * beta * (1.0 - xi[right])) / denom
            return eta

        if mapping == "tanh":
            tanh_b = math.tanh(beta)
            if not two_sided:
                return 1.0 - np.tanh(beta * (1.0 - xi)) / tanh_b
            return 0.5 * (1.0 + np.tanh(beta * (2.0 * xi - 1.0)) / tanh_b)

        # mapping == "sinh"
        sinh_b = math.sinh(beta)
        if not two_sided:
            return np.sinh(beta * xi) / sinh_b
        eta = np.empty_like(xi)
        left = xi <= 0.5
        right = ~left
        eta[left] = 0.5 * np.sinh(2.0 * beta * xi[left]) / sinh_b
        eta[right] = 1.0 - 0.5 * np.sinh(2.0 * beta * (1.0 - xi[right])) / sinh_b
        return eta

    @classmethod
    def from_tanh(
        cls,
        x_min: float,
        x_max: float,
        n_points: int,
        beta: float = 2.5,
        two_sided: bool = False,
    ) -> ExponentialMesh:
        """Construct a mesh using hyperbolic tangent (``tanh``) coordinate stretching."""
        return cls(
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            beta=beta,
            two_sided=two_sided,
            mapping="tanh",
        )

    @classmethod
    def from_sinh(
        cls,
        x_min: float,
        x_max: float,
        n_points: int,
        beta: float = 3.0,
        two_sided: bool = False,
    ) -> ExponentialMesh:
        """Construct a mesh using hyperbolic sine (``sinh``) conformal stretching."""
        return cls(
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            beta=beta,
            two_sided=two_sided,
            mapping="sinh",
        )

    @classmethod
    def from_diffusion_layer(
        cls,
        D: float,
        t_max: float,
        *,
        x_min: float = 0.0,
        factor: float = 6.0,
        n_points: int = 100,
        beta: float = 3.0,
        two_sided: bool = False,
        mapping: str = "exponential",
    ) -> ExponentialMesh:
        r"""Create an exponential mesh sized to the diffusion layer thickness.

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
        n_points : int, default 100
            Total number of grid nodes :math:`N`.
        beta : float, default 3.0
            Dimensionless stretching parameter :math:`\beta > 0`.
        two_sided : bool, default False
            Whether to cluster nodes symmetrically at both boundaries.
        mapping : str, default "exponential"
            Coordinate transformation function: ``"exponential"``, ``"tanh"``, or ``"sinh"``.

        Returns
        -------
        ExponentialMesh
            Configured exponential mesh spanning :math:`[x_{\min}, x_{\min} + \delta]`.
        """
        delta = diffusion_layer_thickness(D, t_max, factor=factor)
        if isinstance(x_min, bool):
            raise TypeError("x_min must be a real number.")
        try:
            x_min_f = float(x_min)
        except (TypeError, ValueError) as exc:
            raise TypeError("x_min must be a real number.") from exc

        return cls(
            x_min=x_min_f,
            x_max=x_min_f + delta,
            n_points=n_points,
            beta=beta,
            two_sided=two_sided,
            mapping=mapping,
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
    def xi(self) -> np.ndarray:
        """Uniform computational coordinate array xi in [0, 1]."""
        return self._xi

    @property
    def beta(self) -> float:
        """Stretching parameter beta > 0."""
        return self._beta

    @property
    def two_sided(self) -> bool:
        """Whether the mesh is symmetrically clustered at both boundaries."""
        return self._two_sided

    @property
    def mapping(self) -> str:
        """Coordinate mapping function ('exponential', 'tanh', or 'sinh')."""
        return self._mapping

    @property
    def dx_min(self) -> float:
        """Minimum interval spacing across the mesh."""
        return float(np.min(self._dx))

    @property
    def dx_max(self) -> float:
        """Maximum interval spacing across the mesh."""
        return float(np.max(self._dx))

    @property
    def is_uniform(self) -> bool:
        """Whether the grid spacing is uniform within numerical tolerance."""
        return False

    def __repr__(self) -> str:
        return (
            f"ExponentialMesh(x_min={self.bounds[0]}, x_max={self.bounds[1]}, "
            f"n_points={self.n_points}, beta={self.beta:.4g}, "
            f"two_sided={self.two_sided}, mapping='{self.mapping}')"
        )

    def __str__(self) -> str:
        return (
            f"ExponentialMesh(bounds=({self.bounds[0]}, {self.bounds[1]}), "
            f"n_points={self.n_points}, beta={self.beta:.4g}, "
            f"two_sided={self.two_sided}, mapping='{self.mapping}')"
        )
