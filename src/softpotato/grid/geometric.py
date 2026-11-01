"""Geometrically expanding 1D spatial discretization mesh implementation."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .base import Mesh1D, diffusion_layer_thickness


class GeometricExpandingMesh(Mesh1D):
    r"""Geometrically expanding 1D spatial discretization mesh.

    Generates a non-uniform spatial grid where consecutive interval steps
    scale by a constant expansion ratio :math:`\gamma > 1`:

    .. math::

        \Delta x_j = \Delta x_0 \cdot \gamma^j \quad (j = 0, \dots, N-2)

    .. math::

        x_{j+1} = x_j + \Delta x_j

    Supports three construction parameterizations (specify any two of
    ``n_points``, ``gamma``, and ``dx_min``):

    1. ``(x_min, x_max, n_points, gamma)``: Computes initial step :math:`\Delta x_0`
       so that the geometric series sums to :math:`x_{\max} - x_{\min}`.
    2. ``(x_min, x_max, dx_min, gamma)``: Automatically determines the required
       number of nodes :math:`N` so the initial step is at most ``dx_min``.
    3. ``(x_min, x_max, n_points, dx_min)``: Numerically solves for the expansion
       ratio :math:`\gamma > 1` that satisfies the domain length and ``dx_min``.

    Also supports two-sided symmetric expansion (``two_sided=True``), providing
    fine spatial resolution :math:`\Delta x_0` at both boundaries (:math:`x_{\min}`
    and :math:`x_{\max}`) while coarsening toward the domain midpoint.

    Parameters
    ----------
    x_min : float
        Lower boundary of the spatial domain (:math:`x_{\min}`).
    x_max : float
        Upper boundary of the spatial domain (:math:`x_{\max}`).
        Must be strictly greater than ``x_min``.
    n_points : int | None, optional
        Total number of grid nodes :math:`N`.
    gamma : float | None, optional
        Geometric expansion ratio :math:`\gamma > 1`.
    dx_min : float | None, optional
        Minimum (boundary) interval spacing :math:`\Delta x_0 > 0`.
    two_sided : bool, default False
        If ``True``, generates a symmetric mesh with fine resolution at both
        :math:`x_{\min}` and :math:`x_{\max}`, expanding toward the center.

    Attributes
    ----------
    nodes : np.ndarray
        1D array of spatial node positions of length :math:`N`.
    dx : np.ndarray
        Interval spacings between consecutive nodes of length :math:`N-1`.
    gamma : float
        Geometric expansion ratio :math:`\gamma > 1`.
    expansion_ratio : float
        Alias for ``gamma``.
    dx_min : float
        Initial (minimum) boundary step size :math:`\Delta x_0`.
    dx_0 : float
        Alias for ``dx_min``.
    dx_max : float
        Maximum interval spacing across the mesh.
    two_sided : bool
        Whether the mesh expands symmetrically from both boundaries.
    n_points : int
        Total number of spatial nodes :math:`N`.
    bounds : tuple[float, float]
        Spatial domain boundaries :math:`(x_{\min}, x_{\max})`.
    is_uniform : bool
        Always ``False`` for geometrically expanding meshes.

    Raises
    ------
    ValueError
        If ``x_max <= x_min`` or bounds are non-finite.
        If not exactly two of ``n_points``, ``gamma``, and ``dx_min`` are provided.
        If ``gamma <= 1.0``, ``dx_min <= 0.0``, or ``n_points`` is below the minimum
        required node count.
        If ``dx_min * (n_points - 1) >= (x_max - x_min)`` when solving for ``gamma``.
    TypeError
        If parameters have invalid types.

    Examples
    --------
    Constructing from ``n_points`` and ``gamma``:

    >>> from softpotato.grid import GeometricExpandingMesh
    >>> mesh = GeometricExpandingMesh(0.0, 1.0, n_points=21, gamma=1.1)
    >>> mesh.n_points
    21
    >>> mesh.bounds
    (0.0, 1.0)
    >>> mesh.is_uniform
    False
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int | None = None,
        gamma: float | None = None,
        dx_min: float | None = None,
        two_sided: bool = False,
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

        if not isinstance(two_sided, bool):
            raise TypeError(
                f"two_sided must be a boolean, got {type(two_sided).__name__}."
            )

        provided_count = sum(p is not None for p in (n_points, gamma, dx_min))
        if provided_count != 2:
            raise ValueError(
                "Must specify exactly two of 'n_points', 'gamma', and 'dx_min'."
            )

        domain_len = x_max_f - x_min_f

        # Validate n_points if provided
        n_pts: int | None = None
        if n_points is not None:
            if isinstance(n_points, bool) or not isinstance(
                n_points, (int, np.integer)
            ):
                raise TypeError(
                    f"n_points must be an integer, got {type(n_points).__name__}."
                )
            n_pts = int(n_points)
            if two_sided:
                min_pts = 4 if dx_min is not None else 3
                if n_pts < min_pts:
                    raise ValueError(
                        f"n_points must be >= {min_pts} for a two-sided expanding mesh, "
                        f"got {n_pts}."
                    )
            else:
                min_pts = 3 if dx_min is not None else 2
                if n_pts < min_pts:
                    raise ValueError(f"n_points must be >= {min_pts}, got {n_pts}.")

        # Validate gamma if provided
        gamma_f: float | None = None
        if gamma is not None:
            if isinstance(gamma, (bool, str, bytes)):
                raise TypeError("gamma must be a real number.")
            try:
                gamma_f = float(gamma)
            except (TypeError, ValueError) as exc:
                raise TypeError("gamma must be a real number.") from exc
            if not math.isfinite(gamma_f):
                raise ValueError(f"gamma must be finite, got {gamma}.")
            if gamma_f <= 1.0:
                raise ValueError(
                    f"gamma must be strictly greater than 1.0, got {gamma_f}."
                )

        # Validate dx_min if provided
        dx_min_f: float | None = None
        if dx_min is not None:
            if isinstance(dx_min, (bool, str, bytes)):
                raise TypeError("dx_min must be a real number.")
            try:
                dx_min_f = float(dx_min)
            except (TypeError, ValueError) as exc:
                raise TypeError("dx_min must be a real number.") from exc
            if not math.isfinite(dx_min_f):
                raise ValueError(f"dx_min must be finite, got {dx_min}.")
            if dx_min_f <= 0.0:
                raise ValueError(f"dx_min must be strictly positive, got {dx_min_f}.")

        # Case 1: (n_points, gamma) -> compute dx_0
        if n_pts is not None and gamma_f is not None:
            resolved_n = n_pts
            resolved_gamma = gamma_f
            exponents = self._interval_exponents(resolved_n - 1, two_sided)
            weights = resolved_gamma**exponents
            resolved_dx0 = float(domain_len / np.sum(weights))

        # Case 2: (dx_min, gamma) -> determine required n_points
        elif dx_min_f is not None and gamma_f is not None:
            resolved_gamma = gamma_f
            if two_sided:
                half_len = 0.5 * domain_len
                if dx_min_f >= half_len:
                    raise ValueError(
                        f"dx_min ({dx_min_f}) must be strictly less than half the "
                        f"domain length ({half_len}) for a two-sided mesh."
                    )
                k_exact = math.log(
                    1.0 + half_len * (resolved_gamma - 1.0) / dx_min_f
                ) / math.log(resolved_gamma)
                k_half = max(2, math.ceil(k_exact - 1e-12))
                resolved_n = 2 * k_half + 1
            else:
                if dx_min_f >= domain_len:
                    raise ValueError(
                        f"dx_min ({dx_min_f}) must be strictly less than the "
                        f"domain length ({domain_len})."
                    )
                m_exact = math.log(
                    1.0 + domain_len * (resolved_gamma - 1.0) / dx_min_f
                ) / math.log(resolved_gamma)
                m_intervals = max(2, math.ceil(m_exact - 1e-12))
                resolved_n = m_intervals + 1

            exponents = self._interval_exponents(resolved_n - 1, two_sided)
            weights = resolved_gamma**exponents
            resolved_dx0 = float(domain_len / np.sum(weights))

        # Case 3: (n_points, dx_min) -> solve for gamma > 1
        else:
            assert n_pts is not None and dx_min_f is not None
            resolved_n = n_pts
            resolved_dx0 = dx_min_f
            m_intervals = resolved_n - 1

            if resolved_dx0 * m_intervals >= domain_len:
                raise ValueError(
                    f"dx_min * (n_points - 1) ({resolved_dx0 * m_intervals}) must be "
                    f"strictly less than the domain length ({domain_len}) for an "
                    f"expanding mesh (gamma > 1)."
                )

            resolved_gamma = self._solve_gamma(
                domain_len=domain_len,
                dx_min=resolved_dx0,
                m_intervals=m_intervals,
                two_sided=two_sided,
            )
            exponents = self._interval_exponents(m_intervals, two_sided)
            weights = resolved_gamma**exponents
            # Re-normalize dx0 for exact boundary landing
            resolved_dx0 = float(domain_len / np.sum(weights))

        dx_arr = resolved_dx0 * weights
        nodes = np.empty(resolved_n, dtype=float)
        nodes[0] = x_min_f
        nodes[1:] = x_min_f + np.cumsum(dx_arr)
        nodes[-1] = x_max_f
        if two_sided and resolved_n % 2 == 1:
            nodes[resolved_n // 2] = 0.5 * (x_min_f + x_max_f)

        dx_final = np.diff(nodes)
        nodes.flags.writeable = False
        dx_final.flags.writeable = False

        self._nodes = nodes
        self._dx = dx_final
        self._gamma = float(resolved_gamma)
        self._dx_min = float(dx_final[0])
        self._two_sided = two_sided

    @staticmethod
    def _interval_exponents(m_intervals: int, two_sided: bool) -> np.ndarray:
        """Compute geometric power exponents for each of the M intervals."""
        idx = np.arange(m_intervals, dtype=float)
        if two_sided:
            return np.minimum(idx, (m_intervals - 1) - idx)
        return idx

    @classmethod
    def _solve_gamma(
        cls,
        domain_len: float,
        dx_min: float,
        m_intervals: int,
        two_sided: bool,
    ) -> float:
        """Solve for expansion ratio gamma > 1 given domain_len, dx_min, and M."""
        target_ratio = domain_len / dx_min  # > m_intervals
        exponents = cls._interval_exponents(m_intervals, two_sided)
        max_exp = float(np.max(exponents))

        # Work in log space to avoid overflow for large interval counts:
        # log(sum(gamma^exponents)) == log(target_ratio)
        log_target = math.log(target_ratio)

        def objective(g: float) -> float:
            log_g = math.log(g)
            log_terms = exponents * log_g
            max_log = float(np.max(log_terms))
            log_sum = max_log + math.log(float(np.sum(np.exp(log_terms - max_log))))
            return log_sum - log_target

        # Bracket:
        # Since sum(gamma^exponents) < m_intervals * gamma^max_exp,
        # we have gamma > (target_ratio / m_intervals)^(1 / max_exp).
        # And since sum(gamma^exponents) > gamma^max_exp,
        # we have gamma < target_ratio^(1 / max_exp).
        g_low = max(1.0 + 1e-15, (target_ratio / m_intervals) ** (1.0 / max_exp))
        g_high = max(g_low * 1.01, target_ratio ** (1.0 / max_exp))

        while objective(g_low) > 0.0 and g_low > 1.0 + 1e-15:
            g_low = 1.0 + 0.5 * (g_low - 1.0)
        while objective(g_high) < 0.0:
            g_high *= 2.0

        return float(brentq(objective, g_low, g_high, xtol=1e-15, rtol=1e-15))

    @classmethod
    def from_diffusion_layer(
        cls,
        D: float,
        t_max: float,
        *,
        x_min: float = 0.0,
        factor: float = 6.0,
        n_points: int | None = None,
        gamma: float | None = None,
        dx_min: float | None = None,
        two_sided: bool = False,
    ) -> GeometricExpandingMesh:
        r"""Create a geometric expanding mesh sized to the diffusion layer thickness.

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
            Total number of grid nodes :math:`N`. Defaults to ``100`` if fewer than
            two of ``(n_points, gamma, dx_min)`` are specified.
        gamma : float | None, optional
            Geometric expansion ratio :math:`\gamma > 1`. Defaults to ``1.05`` if
            fewer than two of ``(n_points, gamma, dx_min)`` are specified.
        dx_min : float | None, optional
            Initial boundary step size :math:`\Delta x_0 > 0`.
        two_sided : bool, default False
            Whether to expand symmetrically from both boundaries.

        Returns
        -------
        GeometricExpandingMesh
            Configured geometric mesh spanning :math:`[x_{\min}, x_{\min} + \delta]`.
        """
        delta = diffusion_layer_thickness(D, t_max, factor=factor)
        if isinstance(x_min, bool):
            raise TypeError("x_min must be a real number.")
        try:
            x_min_f = float(x_min)
        except (TypeError, ValueError) as exc:
            raise TypeError("x_min must be a real number.") from exc

        provided = sum(p is not None for p in (n_points, gamma, dx_min))
        if provided == 0:
            n_points = 100
            gamma = 1.05
        elif provided == 1:
            if gamma is None:
                gamma = 1.05
            else:
                n_points = 100

        return cls(
            x_min=x_min_f,
            x_max=x_min_f + delta,
            n_points=n_points,
            gamma=gamma,
            dx_min=dx_min,
            two_sided=two_sided,
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
    def gamma(self) -> float:
        """Geometric expansion ratio gamma > 1."""
        return self._gamma

    @property
    def expansion_ratio(self) -> float:
        """Alias for geometric expansion ratio gamma."""
        return self._gamma

    @property
    def dx_min(self) -> float:
        """Minimum (boundary) interval spacing Delta x_0."""
        return self._dx_min

    @property
    def dx_0(self) -> float:
        """Alias for initial boundary step size Delta x_0."""
        return self._dx_min

    @property
    def dx_max(self) -> float:
        """Maximum interval spacing across the mesh."""
        return float(np.max(self._dx))

    @property
    def two_sided(self) -> bool:
        """Whether the mesh expands symmetrically from both boundaries."""
        return self._two_sided

    @property
    def is_uniform(self) -> bool:
        """Whether the grid spacing is uniform within numerical tolerance."""
        return False

    def __repr__(self) -> str:
        return (
            f"GeometricExpandingMesh(x_min={self.bounds[0]}, x_max={self.bounds[1]}, "
            f"n_points={self.n_points}, gamma={self.gamma:.4g}, "
            f"dx_min={self.dx_min:.4g}, two_sided={self.two_sided})"
        )

    def __str__(self) -> str:
        return (
            f"GeometricExpandingMesh(bounds=({self.bounds[0]}, {self.bounds[1]}), "
            f"n_points={self.n_points}, gamma={self.gamma:.4g}, "
            f"dx_min={self.dx_min:.4g}, two_sided={self.two_sided})"
        )
