"""
Analytical solutions for electrochemical techniques.
"""

from .step import anson, cottrell, spherical_cottrell
from .voltammetry import (
    peak_potential_irreversible,
    randles_sevcik,
    randles_sevcik_irreversible,
)

__all__ = [
    "anson",
    "cottrell",
    "peak_potential_irreversible",
    "randles_sevcik",
    "randles_sevcik_irreversible",
    "spherical_cottrell",
]
