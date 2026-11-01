"""Spatial discretization meshes and grid generators for numerical simulation."""

from __future__ import annotations

from .base import Mesh1D
from .uniform import Uniform1DMesh

__all__ = [
    "Mesh1D",
    "Uniform1DMesh",
]

