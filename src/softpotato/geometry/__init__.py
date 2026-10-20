r"""Spatial coordinate geometries and differential operators for electrochemical simulation.

This subpackage abstracts coordinate system metrics, spatial finite-difference
operators (:math:`\nabla^2`, :math:`\nabla`), normal interfacial flux evaluation,
and active electrode surface integration:

.. math::

    \nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{p}{r} \frac{\partial c}{\partial r}

Available built-in geometries include:

- :class:`~softpotato.geometry.PlanarGeometry` (``"planar"``, :math:`p = 0`):
  1D Cartesian linear diffusion perpendicular to a planar macroelectrode.
- :class:`~softpotato.geometry.SphericalGeometry` (``"spherical"``, ``"hemispherical"``, :math:`p = 2`):
  1D radial diffusion to a full spherical or hemispherical electrode.
- :class:`~softpotato.geometry.CylindricalGeometry` (``"cylindrical"``, :math:`p = 1`):
  1D axisymmetric radial diffusion to a cylindrical wire or fiber microelectrode.

Examples
--------
Evaluating spatial operators across planar, spherical, and cylindrical geometries:

>>> import numpy as np
>>> from softpotato.geometry import (
...     PlanarGeometry,
...     SphericalGeometry,
...     CylindricalGeometry,
...     get_geometry,
... )
>>>
>>> planar = get_geometry("planar", area=1.0)
>>> x = np.linspace(0.0, 1.0, 51)
>>> bool(np.allclose(planar.laplacian(x**2, x), 2.0))
True
>>>
>>> sphere = SphericalGeometry(r0=1e-4)
>>> r = np.linspace(1.0, 5.0, 201)
>>> bool(np.max(np.abs(sphere.laplacian(1.0 / r, r)[1:-1])) < 1e-4)
True
>>>
>>> cylinder = CylindricalGeometry(r0=5e-6, length=1e-2)
>>> bool(np.max(np.abs(cylinder.laplacian(np.log(r), r)[1:-1])) < 1e-4)
True
"""

from .base import (
    Geometry,
    UnknownGeometryError,
    get_geometry,
    list_geometries,
    register_geometry,
)
from .cylindrical import CylindricalGeometry
from .planar import PlanarGeometry
from .spherical import SphericalGeometry

__all__ = [
    "CylindricalGeometry",
    "Geometry",
    "PlanarGeometry",
    "SphericalGeometry",
    "UnknownGeometryError",
    "get_geometry",
    "list_geometries",
    "register_geometry",
]
