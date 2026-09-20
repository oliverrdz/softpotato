"""
Electrochemical and chemical kinetics models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from softpotato.core.constants import FARADAY, GAS_CONSTANT, STANDARD_TEMPERATURE


class KineticsModel(ABC):
    """
    Abstract base class for heterogeneous electron transfer kinetics models.
    """

    @abstractmethod
    def calculate_rates(
        self,
        E: float | np.ndarray,
        E0: float,
        n_electrons: int = 1,
        T: float = STANDARD_TEMPERATURE,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Calculate forward and backward heterogeneous rate constants.

        Parameters
        ----------
        E : float or np.ndarray
            Electrode potential in Volts.
        E0 : float
            Standard reduction potential in Volts.
        n_electrons : int, optional
            Number of electrons transferred, by default 1.
        T : float, optional
            Temperature in Kelvin, by default 298.15 K.

        Returns
        -------
        tuple[float or np.ndarray, float or np.ndarray]
            (k_f, k_b) forward and backward rate constants in cm/s.
        """

    @abstractmethod
    def calculate_flux(
        self,
        E: float,
        E0: float,
        n_electrons: int,
        c_O: float,
        c_R: float,
        T: float = STANDARD_TEMPERATURE,
    ) -> float:
        """
        Calculate the net reduction flux at the electrode surface: :math:`J = k_f c_O - k_b c_R`.

        Parameters
        ----------
        E : float
            Electrode potential in Volts.
        E0 : float
            Standard reduction potential in Volts.
        n_electrons : int
            Number of electrons transferred.
        c_O : float
            Surface concentration of oxidized species in mol/cm³.
        c_R : float
            Surface concentration of reduced species in mol/cm³.
        T : float, optional
            Temperature in Kelvin, by default 298.15 K.

        Returns
        -------
        float
            Net molar flux in mol/(cm²·s).
        """


class ButlerVolmer(KineticsModel):
    """
    Butler-Volmer model for heterogeneous electron transfer kinetics.

    Mathematical expressions:

    .. math::

        k_f(E) = k_0 \\exp\\left(-\\alpha \\frac{n F}{R T} (E - E^0)\\right)

    .. math::

        k_b(E) = k_0 \\exp\\left((1 - \\alpha) \\frac{n F}{R T} (E - E^0)\\right)

    .. math::

        J(E) = k_f(E) c_O(0, t) - k_b(E) c_R(0, t)
    """

    def __init__(self, k0: float, alpha: float = 0.5) -> None:
        """
        Initialize Butler-Volmer kinetics.

        Parameters
        ----------
        k0 : float
            Standard heterogeneous rate constant in cm/s (:math:`k_0 \\ge 0`).
        alpha : float, optional
            Charge transfer coefficient (:math:`0 \\le \\alpha \\le 1`), by default 0.5.

        Raises
        ------
        TypeError
            If parameters are not real numbers.
        ValueError
            If `k0 < 0` or `alpha < 0` or `alpha > 1`.
        """
        if isinstance(k0, bool) or not isinstance(k0, (int, float)):
            raise TypeError(f"k0 must be a real number, got {type(k0).__name__}.")
        if k0 < 0:
            raise ValueError(f"k0 must be non-negative, got {k0}.")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")

        self._k0 = float(k0)
        self._alpha = float(alpha)

    @property
    def k0(self) -> float:
        """Standard heterogeneous rate constant in cm/s."""
        return self._k0

    @property
    def alpha(self) -> float:
        """Charge transfer coefficient."""
        return self._alpha

    def calculate_rates(
        self,
        E: float | np.ndarray,
        E0: float,
        n_electrons: int = 1,
        T: float = STANDARD_TEMPERATURE,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Calculate forward and backward heterogeneous rate constants (cm/s).
        """
        f = FARADAY / (GAS_CONSTANT * T)
        eta = np.asarray(E, dtype=np.float64) - E0
        is_scalar = eta.ndim == 0

        # Clipping overpotential exponent to prevent numerical overflow at extreme potentials
        arg_f = np.clip(-self._alpha * n_electrons * f * eta, -300.0, 300.0)
        arg_b = np.clip((1.0 - self._alpha) * n_electrons * f * eta, -300.0, 300.0)

        kf = self._k0 * np.exp(arg_f)
        kb = self._k0 * np.exp(arg_b)

        if is_scalar:
            return float(kf), float(kb)
        return kf, kb

    def calculate_flux(
        self,
        E: float,
        E0: float,
        n_electrons: int,
        c_O: float,
        c_R: float,
        T: float = STANDARD_TEMPERATURE,
    ) -> float:
        """
        Calculate net molar reduction flux :math:`J = k_f c_O - k_b c_R` in mol/(cm²·s).
        """
        kf, kb = self.calculate_rates(E=E, E0=E0, n_electrons=n_electrons, T=T)
        return float(kf * c_O - kb * c_R)

    def __repr__(self) -> str:
        return f"ButlerVolmer(k0={self._k0}, alpha={self._alpha})"
