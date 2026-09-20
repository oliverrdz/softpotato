"""
Analytical closed-form solutions for electrochemical benchmarking.
"""

from . import geometry, kinetics, techniques
from .geometry import (
    koutecky_levich,
    levich,
    steady_state_microdisc,
    steady_state_microhemisphere,
)
from .kinetics import (
    catalytic_current,
    nicholson_psi,
)
from .techniques import (
    anson,
    cottrell,
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
    spherical_cottrell,
)

__all__ = [
    "anson",
    "catalytic_current",
    "cottrell",
    "geometry",
    "kinetics",
    "koutecky_levich",
    "levich",
    "nicholson_psi",
    "peak_potential_irreversible",
    "randles_sevcik",
    "randles_sevcik_irreversible",
    "spherical_cottrell",
    "steady_state_microdisc",
    "steady_state_microhemisphere",
    "techniques",
]
