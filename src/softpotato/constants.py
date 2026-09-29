"""Physical and chemical constants with standardized SI definitions.

This module provides fundamental physical and electrochemical constants sourced
from CODATA (via :mod:`scipy.constants`) and standardized IUPAC conventions.

Constants
---------
FARADAY : float
    Faraday constant :math:`F` in Coulombs per mole (:math:`\\text{C}\\cdot\\text{mol}^{-1}`).
    Represents the magnitude of electric charge per mole of electrons
    (:math:`F = e \\cdot N_A \\approx 96485.332\\,\\text{C}\\cdot\\text{mol}^{-1}`).
GAS_CONSTANT : float
    Molar gas constant :math:`R` in Joules per mole per Kelvin
    (:math:`\\text{J}\\cdot\\text{mol}^{-1}\\cdot\\text{K}^{-1}`).
    Represents the constant of proportionality in the ideal gas law
    (:math:`R = k_B \\cdot N_A \\approx 8.31446\\,\\text{J}\\cdot\\text{mol}^{-1}\\cdot\\text{K}^{-1}`).
STANDARD_TEMPERATURE : float
    Standard thermodynamic temperature :math:`T^\\circ` in Kelvin (:math:`298.15\\,\\text{K}`,
    corresponding to :math:`25\\,^\\circ\\text{C}`).

Aliases
-------
F : float
    Shorthand alias for :data:`FARADAY`.
R : float
    Shorthand alias for :data:`GAS_CONSTANT`.
T_STD : float
    Shorthand alias for :data:`STANDARD_TEMPERATURE`.

Related Physical Constants
--------------------------
AVOGADRO : float
    Avogadro constant :math:`N_A` in reciprocal moles (:math:`\\text{mol}^{-1}`).
    Number of constituent particles per mole
    (:math:`6.02214076 \\times 10^{23}\\,\\text{mol}^{-1}`).
BOLTZMANN : float
    Boltzmann constant :math:`k_B` in Joules per Kelvin (:math:`\\text{J}\\cdot\\text{K}^{-1}`).
    Relates average kinetic energy of particles to temperature
    (:math:`1.380649 \\times 10^{-23}\\,\\text{J}\\cdot\\text{K}^{-1}`).
ELEMENTARY_CHARGE : float
    Elementary charge :math:`e` in Coulombs (:math:`\\text{C}`).
    Electric charge carried by a single proton (:math:`1.602176634 \\times 10^{-19}\\,\\text{C}`).
VACUUM_PERMITTIVITY : float
    Vacuum permittivity (electric constant) :math:`\\varepsilon_0` in Farads per meter
    (:math:`\\text{F}\\cdot\\text{m}^{-1}`). Capability of a vacuum to permit electric field lines
    (:math:`\\approx 8.854188 \\times 10^{-12}\\,\\text{F}\\cdot\\text{m}^{-1}`).

Examples
--------
>>> from softpotato.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE
>>> round(FARADAY, 2)
96485.33
>>> round(GAS_CONSTANT, 4)
8.3145
>>> STANDARD_TEMPERATURE
298.15
>>> from softpotato.constants import F, R, T_STD
>>> F == FARADAY and R == GAS_CONSTANT and T_STD == STANDARD_TEMPERATURE
True
"""

from scipy import constants as _const

# Primary electrochemical constants
FARADAY: float = float(_const.physical_constants["Faraday constant"][0])
GAS_CONSTANT: float = float(_const.R)
STANDARD_TEMPERATURE: float = 298.15

# Shorthand aliases
F: float = FARADAY
R: float = GAS_CONSTANT
T_STD: float = STANDARD_TEMPERATURE

# Foundational physical constants
AVOGADRO: float = float(_const.N_A)
BOLTZMANN: float = float(_const.k)
ELEMENTARY_CHARGE: float = float(_const.e)
VACUUM_PERMITTIVITY: float = float(_const.epsilon_0)

__all__ = [
    "AVOGADRO",
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "FARADAY",
    "GAS_CONSTANT",
    "STANDARD_TEMPERATURE",
    "T_STD",
    "VACUUM_PERMITTIVITY",
    "F",
    "R",
]
