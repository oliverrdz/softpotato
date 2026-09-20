"""
Soft Potato 3.0
Next-generation open-source electrochemical simulator and toolkit.
"""

__version__ = "3.0.0.dev1"
__author__ = "Soft Potato Developers"

# Expose primary submodules for easy access
from . import (
    analytical as analytical,
)
from . import (
    core as core,
)
from . import (
    geometry as geometry,
)
from . import (
    kinetics as kinetics,
)
from . import (
    simulate as simulate,
)
from . import (
    techniques as techniques,
)

__all__ = [
    "analytical",
    "core",
    "geometry",
    "kinetics",
    "simulate",
    "techniques",
]
