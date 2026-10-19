r"""Physical and chemical constants with standardized SI definitions.

This module provides fundamental physical and electrochemical constants sourced
from CODATA (via :mod:`scipy.constants`) and standardized IUPAC conventions.

Constants
---------

| Constant | Symbol | Value | Units | Description |
| :--- | :---: | :--- | :--- | :--- |
| :data:`~softpotato.constants.FARADAY` | :math:`F` | :math:`\approx 96485.332` | :math:`\text{C}\cdot\text{mol}^{-1}` | Faraday constant (:math:`F = e \cdot N_A`); electric charge per mole of electrons |
| :data:`~softpotato.constants.GAS_CONSTANT` | :math:`R` | :math:`\approx 8.31446` | :math:`\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}` | Molar gas constant (:math:`R = k_B \cdot N_A`); ideal gas law proportionality constant |
| :data:`~softpotato.constants.STANDARD_TEMPERATURE` | :math:`T^\circ` | :math:`298.15` | :math:`\text{K}` | Standard thermodynamic temperature (corresponding to :math:`25\,^\circ\text{C}`) |

Aliases
-------

| Alias | Target | Description |
| :--- | :--- | :--- |
| :data:`~softpotato.constants.F` | :data:`~softpotato.constants.FARADAY` | Shorthand alias for Faraday constant (:math:`F`) |
| :data:`~softpotato.constants.R` | :data:`~softpotato.constants.GAS_CONSTANT` | Shorthand alias for molar gas constant (:math:`R`) |
| :data:`~softpotato.constants.T_STD` | :data:`~softpotato.constants.STANDARD_TEMPERATURE` | Shorthand alias for standard thermodynamic temperature (:math:`T^\circ`) |

Related Physical Constants
--------------------------

| Constant | Symbol | Value | Units | Description |
| :--- | :---: | :--- | :--- | :--- |
| :data:`~softpotato.constants.AVOGADRO` | :math:`N_A` | :math:`6.02214076 \times 10^{23}` | :math:`\text{mol}^{-1}` | Avogadro constant; constituent particles per mole |
| :data:`~softpotato.constants.BOLTZMANN` | :math:`k_B` | :math:`1.380649 \times 10^{-23}` | :math:`\text{J}\cdot\text{K}^{-1}` | Boltzmann constant; relates kinetic energy to temperature |
| :data:`~softpotato.constants.ELEMENTARY_CHARGE` | :math:`e` | :math:`1.602176634 \times 10^{-19}` | :math:`\text{C}` | Elementary charge; electric charge of a single proton |
| :data:`~softpotato.constants.VACUUM_PERMITTIVITY` | :math:`\varepsilon_0` | :math:`\approx 8.854188 \times 10^{-12}` | :math:`\text{F}\cdot\text{m}^{-1}` | Vacuum permittivity; capability to permit electric fields |


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
