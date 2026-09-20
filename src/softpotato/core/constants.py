"""
Physical and chemical constants with strict CGS/SI standard definitions.
"""

from scipy import constants as _const

# Faraday constant (C/mol)
FARADAY: float = float(_const.physical_constants["Faraday constant"][0])

# Molar gas constant R (J/(mol*K))
GAS_CONSTANT: float = float(_const.R)

# Standard thermodynamic temperature in Kelvin (25 °C)
STANDARD_TEMPERATURE: float = 298.15

__all__ = [
    "FARADAY",
    "GAS_CONSTANT",
    "STANDARD_TEMPERATURE",
]
