"""Spatial discretization meshes and grid generators for numerical simulation."""

from __future__ import annotations

from .base import Mesh1D, diffusion_layer_thickness
from .exponential import ExponentialMesh
from .geometric import GeometricExpandingMesh
from .uniform import Uniform1DMesh


def create_diffusion_mesh(
    D: float,
    t_max: float,
    *,
    kind: str = "geometric",
    x_min: float = 0.0,
    factor: float = 6.0,
    n_points: int | None = None,
    gamma: float | None = None,
    dx_min: float | None = None,
    beta: float = 3.0,
    dx: float | None = None,
    two_sided: bool = False,
    mapping: str = "exponential",
) -> Mesh1D:
    r"""Create a 1D spatial mesh sized to the diffusion layer thickness :math:`\delta`.

    Automatically sets the outer domain boundary to
    :math:`x_{\max} = x_{\min} + \text{factor} \cdot \sqrt{D \, t_{\max}}`
    (with default :math:`\delta = 6 \sqrt{D t_{\max}}`) and constructs a
    ``GeometricExpandingMesh``, ``ExponentialMesh``, or ``Uniform1DMesh``.

    Parameters
    ----------
    D : float
        Species diffusion coefficient :math:`D > 0` (:math:`\text{m}^2/\text{s}`).
    t_max : float
        Maximum characteristic simulation time :math:`t_{\max} > 0` (:math:`\text{s}`).
    kind : str, default "geometric"
        Mesh generator type: ``"geometric"``, ``"exponential"``, or ``"uniform"``.
    x_min : float, default 0.0
        Lower boundary coordinate :math:`x_{\min}`.
    factor : float, default 6.0
        Dimensionless multiplier for diffusion layer thickness
        :math:`\delta = \text{factor} \cdot \sqrt{D t_{\max}}`.
    n_points : int | None, optional
        Total number of spatial nodes :math:`N`.
    gamma : float | None, optional
        Expansion ratio :math:`\gamma > 1` (for ``kind="geometric"``).
    dx_min : float | None, optional
        Minimum boundary step size :math:`\Delta x_0 > 0` (for ``kind="geometric"``).
    beta : float, default 3.0
        Stretching parameter :math:`\beta > 0` (for ``kind="exponential"``).
    dx : float | None, optional
        Uniform step size :math:`\Delta x > 0` (for ``kind="uniform"``).
    two_sided : bool, default False
        Whether to cluster nodes symmetrically at both boundaries
        (for ``kind="geometric"`` or ``kind="exponential"``).
    mapping : str, default "exponential"
        Coordinate transformation (``"exponential"``, ``"tanh"``, or ``"sinh"``)
        when ``kind="exponential"``.

    Returns
    -------
    Mesh1D
        Constructed 1D mesh instance spanning :math:`[x_{\min}, x_{\min} + \delta]`.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or mesh parameters are invalid.
    TypeError
        If ``kind`` is not a string or parameters have invalid types.
    """
    if not isinstance(kind, str):
        raise TypeError(f"kind must be a string, got {type(kind).__name__}.")
    kind_norm = kind.strip().lower()

    if kind_norm == "geometric":
        return GeometricExpandingMesh.from_diffusion_layer(
            D=D,
            t_max=t_max,
            x_min=x_min,
            factor=factor,
            n_points=n_points,
            gamma=gamma,
            dx_min=dx_min,
            two_sided=two_sided,
        )
    if kind_norm == "exponential":
        return ExponentialMesh.from_diffusion_layer(
            D=D,
            t_max=t_max,
            x_min=x_min,
            factor=factor,
            n_points=100 if n_points is None else n_points,
            beta=beta,
            two_sided=two_sided,
            mapping=mapping,
        )
    if kind_norm == "uniform":
        return Uniform1DMesh.from_diffusion_layer(
            D=D,
            t_max=t_max,
            x_min=x_min,
            factor=factor,
            n_points=n_points,
            dx=dx,
        )

    raise ValueError(
        f"Unknown mesh kind '{kind}'. Must be one of ('geometric', 'exponential', 'uniform')."
    )


__all__ = [
    "ExponentialMesh",
    "GeometricExpandingMesh",
    "Mesh1D",
    "Uniform1DMesh",
    "create_diffusion_mesh",
    "diffusion_layer_thickness",
]
