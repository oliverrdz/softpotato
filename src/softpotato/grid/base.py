"""Abstract base class and protocols for 1D spatial discretization meshes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np


def diffusion_layer_thickness(
    D: float,
    t_max: float,
    factor: float = 6.0,
) -> float:
    r"""Estimate the semi-infinite diffusion layer thickness :math:`\delta`.

    Computes the characteristic domain extent required to simulate semi-infinite
    diffusion over duration :math:`t_{\max}` without outer boundary interference:

    .. math::

        \delta = \text{factor} \cdot \sqrt{D \, t_{\max}}

    At the default ``factor=6.0``, :math:`\operatorname{erfc}(6 / 2) \approx 2.2 \times 10^{-5}`,
    ensuring concentration perturbations at :math:`x = \delta` remain negligible.

    Parameters
    ----------
    D : float
        Species diffusion coefficient :math:`D > 0` (:math:`\text{m}^2/\text{s}`).
    t_max : float
        Maximum characteristic simulation time :math:`t_{\max} > 0` (:math:`\text{s}`).
    factor : float, default 6.0
        Dimensionless multiplier :math:`> 0` applied to :math:`\sqrt{D t_{\max}}`.

    Returns
    -------
    float
        Estimated diffusion layer thickness :math:`\delta > 0`.

    Raises
    ------
    ValueError
        If ``D <= 0``, ``t_max <= 0``, ``factor <= 0``, or any input is non-finite.
    TypeError
        If inputs are not real numbers.
    """
    for name, val in (("D", D), ("t_max", t_max), ("factor", factor)):
        if isinstance(val, (bool, str, bytes)):
            raise TypeError(f"{name} must be a real number.")
        try:
            val_f = float(val)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number.") from exc
        if not math.isfinite(val_f):
            raise ValueError(f"{name} must be finite, got {val}.")
        if val_f <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {val_f}.")

    return float(factor) * math.sqrt(float(D) * float(t_max))


class Mesh1D(ABC):
    r"""Abstract base class representing a 1D spatial discretization mesh.

    Encapsulates 1D coordinate points :math:`x_j`, cell intervals
    :math:`\Delta x_j = x_{j+1} - x_j`, spatial domain boundaries,
    and grid uniformity diagnostics.

    Provides full interoperability with the NumPy array protocol
    (slicing, indexing, mathematical operators, and universal functions).

    Attributes
    ----------
    nodes : np.ndarray
        1D array of spatial node positions.
    dx : np.ndarray
        Interval spacings between consecutive nodes: :math:`\Delta x_j = x_{j+1} - x_j`.
    n_points : int
        Total number of spatial grid nodes :math:`N`.
    bounds : tuple[float, float]
        Spatial domain boundaries :math:`(x_{\min}, x_{\max})`.
    is_uniform : bool
        Whether the grid spacing is uniform within numerical tolerance.
    """

    @property
    @abstractmethod
    def nodes(self) -> np.ndarray:
        """1D array of spatial node positions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dx(self) -> np.ndarray:
        """Interval spacings between consecutive nodes: dx[j] = nodes[j+1] - nodes[j]."""
        raise NotImplementedError

    @property
    def n_points(self) -> int:
        """Total number of spatial nodes N."""
        return len(self.nodes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape of the spatial nodes `(N,)`."""
        return self.nodes.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions (`1`)."""
        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        """Spatial domain boundaries (x_min, x_max)."""
        return float(self.nodes[0]), float(self.nodes[-1])

    @property
    def x_min(self) -> float:
        """Spatial domain minimum boundary coordinate."""
        return self.bounds[0]

    @property
    def x_max(self) -> float:
        """Spatial domain maximum boundary coordinate."""
        return self.bounds[1]

    @property
    def dx_min(self) -> float:
        """Minimum interval spacing across the mesh."""
        return float(np.min(self.dx))

    @property
    def dx_max(self) -> float:
        """Maximum interval spacing across the mesh."""
        return float(np.max(self.dx))

    @property
    def is_uniform(self) -> bool:
        """Whether the grid spacing is uniform within numerical tolerance."""
        if len(self.dx) <= 1:
            return True
        return bool(np.allclose(self.dx, self.dx[0], rtol=1e-5, atol=0.0))

    def __array__(
        self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """NumPy array protocol support: np.asarray(mesh) yields mesh.nodes."""
        if copy:
            return np.array(self.nodes, dtype=dtype, copy=True)
        return np.asarray(self.nodes, dtype=dtype)

    def __len__(self) -> int:
        """Return the number of spatial nodes."""
        return self.n_points

    def __getitem__(self, item: Any) -> Any:
        """Index or slice the spatial nodes array."""
        return self.nodes[item]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the spatial node coordinates."""
        return iter(self.nodes)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Delegate NumPy ufunc evaluations to the underlying nodes array."""
        converted_inputs = tuple(
            np.asarray(x) if isinstance(x, Mesh1D) else x for x in inputs
        )
        return getattr(ufunc, method)(*converted_inputs, **kwargs)

    def __add__(self, other: Any) -> np.ndarray:
        return self.nodes + (other.nodes if isinstance(other, Mesh1D) else other)

    def __radd__(self, other: Any) -> np.ndarray:
        return (other.nodes if isinstance(other, Mesh1D) else other) + self.nodes

    def __sub__(self, other: Any) -> np.ndarray:
        return self.nodes - (other.nodes if isinstance(other, Mesh1D) else other)

    def __rsub__(self, other: Any) -> np.ndarray:
        return (other.nodes if isinstance(other, Mesh1D) else other) - self.nodes

    def __mul__(self, other: Any) -> np.ndarray:
        return self.nodes * (other.nodes if isinstance(other, Mesh1D) else other)

    def __rmul__(self, other: Any) -> np.ndarray:
        return (other.nodes if isinstance(other, Mesh1D) else other) * self.nodes

    def __truediv__(self, other: Any) -> np.ndarray:
        return self.nodes / (other.nodes if isinstance(other, Mesh1D) else other)

    def __rtruediv__(self, other: Any) -> np.ndarray:
        return (other.nodes if isinstance(other, Mesh1D) else other) / self.nodes

    def __pow__(self, other: Any) -> np.ndarray:
        return self.nodes ** (other.nodes if isinstance(other, Mesh1D) else other)

    def __neg__(self) -> np.ndarray:
        return -self.nodes

    def __pos__(self) -> np.ndarray:
        return +self.nodes

    def __abs__(self) -> np.ndarray:
        return np.abs(self.nodes)

    def __lt__(self, other: Any) -> np.ndarray:
        return self.nodes < (other.nodes if isinstance(other, Mesh1D) else other)

    def __le__(self, other: Any) -> np.ndarray:
        return self.nodes <= (other.nodes if isinstance(other, Mesh1D) else other)

    def __gt__(self, other: Any) -> np.ndarray:
        return self.nodes > (other.nodes if isinstance(other, Mesh1D) else other)

    def __ge__(self, other: Any) -> np.ndarray:
        return self.nodes >= (other.nodes if isinstance(other, Mesh1D) else other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mesh1D):
            return bool(
                self.n_points == other.n_points
                and np.allclose(self.nodes, other.nodes, rtol=1e-12, atol=1e-12)
            )
        if isinstance(other, (np.ndarray, list, tuple)):
            arr = np.asarray(other)
            if arr.shape != self.nodes.shape:
                return False
            return bool(np.allclose(self.nodes, arr, rtol=1e-12, atol=1e-12))
        return False

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"n_points={self.n_points} "
            f"bounds=({self.bounds[0]:.4g}, {self.bounds[1]:.4g})>"
        )
