"""Structured result container for high-level electrochemical technique simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

from softpotato.solver.base import SolverResult

from .base import BaseTechnique
from .chronopotentiometry import Chronopotentiometry
from .pulse import (
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
)
from .voltammetry import CyclicVoltammetry


def _compute_cumulative_charge(time: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compute cumulative charge Q(t) = integral_0^t I(tau) dtau via trapezoidal rule."""
    if time.size < 2:
        return np.zeros_like(current, dtype=float)
    return np.asarray(
        cumulative_trapezoid(current, time, axis=-1, initial=0.0),
        dtype=float,
    )


def _slice_solver_result(
    solver_result: SolverResult,
    mask: np.ndarray,
) -> SolverResult:
    """Return a time-sliced copy of a SolverResult along boolean index mask."""
    return SolverResult(
        t=solver_result.t[mask].copy(),
        x=solver_result.x.copy(),
        concentrations={
            sp: arr[mask].copy() for sp, arr in solver_result.concentrations.items()
        },
        fluxes={sp: arr[mask].copy() for sp, arr in solver_result.fluxes.items()},
        right_fluxes={
            sp: arr[mask].copy() for sp, arr in solver_result.right_fluxes.items()
        },
        currents={sp: arr[mask].copy() for sp, arr in solver_result.currents.items()},
        left_currents={
            sp: arr[mask].copy() for sp, arr in solver_result.left_currents.items()
        },
        right_currents={
            sp: arr[mask].copy() for sp, arr in solver_result.right_currents.items()
        },
        success=solver_result.success,
        message=solver_result.message,
        raw_output=solver_result.raw_output,
        mesh=solver_result.mesh,
    )


@dataclass
class TechniqueResult:
    r"""Structured container for the output of :func:`~softpotato.techniques.simulate`.

    Wraps the underlying :class:`~softpotato.solver.SolverResult` with
    electrochemical observables (``time``, ``potential``, ``current``, ``charge``),
    dict-like species concentration profile access (``result["Ox"]``), and
    technique-specific convenience views (CV ``cycles``, SWV ``E_step`` /
    ``i_forward`` / ``i_reverse`` / ``i_net``, DPV ``E_base`` / ``i_pre`` /
    ``i_pulse`` / ``i_diff``, and CP ``transition_time``).

    Attributes
    ----------
    technique : BaseTechnique
        The executed electrochemical technique specification.
    solver_result : SolverResult
        Underlying full PDE solution containing species concentration profiles,
        interfacial fluxes, per-species currents, spatial grid, and solver status.
    time : np.ndarray
        1D array of evaluation timestamps :math:`t` in seconds.
    potential : np.ndarray
        Electrode potential :math:`E(t)` in Volts (applied waveform for
        potentiostatic techniques; recovered surface potential for galvanostatic
        chronopotentiometry when kinetics/redox metadata are available).
    current : np.ndarray
        Net Faradaic current :math:`I(t)` in Amperes under IUPAC polarity
        conventions (anodic positive, cathodic negative).
    charge : np.ndarray
        Cumulative charge :math:`Q(t) = \int_0^t I(\tau)\,d\tau` in Coulombs
        computed via trapezoidal integration. Automatically computed if omitted.
    metadata : dict[str, Any]
        Auxiliary execution metadata (solver name, geometry, segment count,
        technique-specific quantities such as ``transition_time``).
    """

    technique: BaseTechnique
    solver_result: SolverResult
    time: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    charge: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _dpv_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _swv_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cycles_cache: tuple[TechniqueResult, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.potential = np.asarray(self.potential, dtype=float)
        self.current = np.asarray(self.current, dtype=float)
        if self.charge is None:
            self.charge = _compute_cumulative_charge(self.time, self.current)
        else:
            self.charge = np.asarray(self.charge, dtype=float)

    # ------------------------------------------------------------------
    # Delegation & dict-like species concentration access
    # ------------------------------------------------------------------

    @property
    def species(self) -> list[str]:
        """List of chemical species present in the solution."""
        return self.solver_result.species

    def __getitem__(self, species_name: str) -> np.ndarray:
        """Shorthand to access 2D concentration profile ``(N_t, N_x)`` for a species."""
        return self.solver_result[species_name]

    def __contains__(self, species_name: object) -> bool:
        return species_name in self.solver_result.concentrations

    @property
    def concentrations(self) -> dict[str, np.ndarray]:
        """Mapping from species name to 2D concentration profile ``(N_t, N_x)``."""
        return self.solver_result.concentrations

    @property
    def fluxes(self) -> dict[str, np.ndarray]:
        """Mapping from species name to left-boundary interfacial flux ``(N_t,)``."""
        return self.solver_result.fluxes

    @property
    def currents(self) -> dict[str, np.ndarray]:
        """Mapping from species name to per-species Faradaic current ``(N_t,)``."""
        return self.solver_result.currents

    @property
    def grid(self) -> np.ndarray:
        """1D spatial coordinate grid ``x`` from the underlying solver result."""
        return self.solver_result.x

    @property
    def x(self) -> np.ndarray:
        """Alias for :attr:`grid`."""
        return self.solver_result.x

    @property
    def t(self) -> np.ndarray:
        """Alias for :attr:`time`."""
        return self.time

    @property
    def E(self) -> np.ndarray:
        """Alias for :attr:`potential`."""
        return self.potential

    @property
    def i(self) -> np.ndarray:
        """Alias for :attr:`current`."""
        return self.current

    @property
    def Q(self) -> np.ndarray:
        """Alias for :attr:`charge`."""
        assert self.charge is not None
        return self.charge

    @property
    def success(self) -> bool:
        """Whether the underlying numerical integration succeeded."""
        return self.solver_result.success

    @property
    def message(self) -> str:
        """Status message from the underlying numerical solver."""
        return self.solver_result.message

    # ------------------------------------------------------------------
    # Cyclic Voltammetry convenience views
    # ------------------------------------------------------------------

    @property
    def cycles(self) -> tuple[TechniqueResult, ...]:
        """Per-cycle :class:`TechniqueResult` slices for :class:`CyclicVoltammetry`.

        Returns
        -------
        tuple[TechniqueResult, ...]
            One sliced :class:`TechniqueResult` per voltammetric cycle in
            :attr:`CyclicVoltammetry.cycle_time_ranges`.

        Raises
        ------
        TypeError
            If :attr:`technique` is not a :class:`CyclicVoltammetry` instance.
        """
        if not isinstance(self.technique, CyclicVoltammetry):
            raise TypeError(
                "cycles is only available for CyclicVoltammetry results, "
                f"got {type(self.technique).__name__}."
            )
        if self._cycles_cache is None:
            tol = 1e-12 * max(1.0, self.technique.duration)
            cycle_views: list[TechniqueResult] = []
            for idx, (t_start, t_end) in enumerate(self.technique.cycle_time_ranges):
                mask = (self.time >= t_start - tol) & (self.time <= t_end + tol)
                t_cyc = self.time[mask].copy()
                e_cyc = self.potential[mask].copy()
                i_cyc = self.current[mask].copy()
                sr_cyc = _slice_solver_result(self.solver_result, mask)
                cycle_views.append(
                    TechniqueResult(
                        technique=self.technique,
                        solver_result=sr_cyc,
                        time=t_cyc,
                        potential=e_cyc,
                        current=i_cyc,
                        metadata={
                            **self.metadata,
                            "cycle_index": idx,
                            "cycle_time_range": (t_start, t_end),
                        },
                    )
                )
            self._cycles_cache = tuple(cycle_views)
        return self._cycles_cache

    # ------------------------------------------------------------------
    # Square-Wave Voltammetry convenience views
    # ------------------------------------------------------------------

    def _get_swv_sampled(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(self.technique, SquareWaveVoltammetry):
            raise TypeError(
                "SWV sampled views are only available for SquareWaveVoltammetry results, "
                f"got {type(self.technique).__name__}."
            )
        if self._swv_cache is None:
            self._swv_cache = self.technique.sample_currents(self.time, self.current)
        return self._swv_cache

    @property
    def E_step(self) -> np.ndarray:
        """Staircase step potentials for :class:`SquareWaveVoltammetry`."""
        return self._get_swv_sampled()[0]

    @property
    def i_forward(self) -> np.ndarray:
        """Forward half-cycle sampled currents for :class:`SquareWaveVoltammetry`."""
        return self._get_swv_sampled()[1]

    @property
    def i_reverse(self) -> np.ndarray:
        """Reverse half-cycle sampled currents for :class:`SquareWaveVoltammetry`."""
        return self._get_swv_sampled()[2]

    @property
    def i_net(self) -> np.ndarray:
        """Net square-wave current ``i_forward - i_reverse`` for :class:`SquareWaveVoltammetry`."""
        return self._get_swv_sampled()[3]

    # ------------------------------------------------------------------
    # Differential Pulse Voltammetry convenience views
    # ------------------------------------------------------------------

    def _get_dpv_sampled(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(self.technique, DifferentialPulseVoltammetry):
            raise TypeError(
                "DPV sampled views are only available for DifferentialPulseVoltammetry results, "
                f"got {type(self.technique).__name__}."
            )
        if self._dpv_cache is None:
            self._dpv_cache = self.technique.sample_currents(self.time, self.current)
        return self._dpv_cache

    @property
    def E_base(self) -> np.ndarray:
        """Staircase base potentials for :class:`DifferentialPulseVoltammetry`."""
        return self._get_dpv_sampled()[0]

    @property
    def i_pre(self) -> np.ndarray:
        """Pre-pulse sampled currents for :class:`DifferentialPulseVoltammetry`."""
        return self._get_dpv_sampled()[1]

    @property
    def i_pulse(self) -> np.ndarray:
        """End-of-pulse sampled currents for :class:`DifferentialPulseVoltammetry`."""
        return self._get_dpv_sampled()[2]

    @property
    def i_diff(self) -> np.ndarray:
        """Differential current ``i_pulse - i_pre`` for :class:`DifferentialPulseVoltammetry`."""
        return self._get_dpv_sampled()[3]

    # ------------------------------------------------------------------
    # Chronopotentiometry convenience views
    # ------------------------------------------------------------------

    @property
    def transition_time(self) -> float | None:
        r"""Detected chronopotentiometric transition time :math:`\tau` in seconds, or ``None``.

        Raises
        ------
        TypeError
            If :attr:`technique` is not a :class:`Chronopotentiometry` instance.
        """
        if not isinstance(self.technique, Chronopotentiometry):
            raise TypeError(
                "transition_time is only available for Chronopotentiometry results, "
                f"got {type(self.technique).__name__}."
            )
        if "transition_time" in self.metadata:
            val = self.metadata["transition_time"]
            return None if val is None else float(val)
        return None

    # ------------------------------------------------------------------
    # Sampled-Current Voltammetry convenience views
    # ------------------------------------------------------------------

    def sample_currents(
        self,
        sample_times: float
        | list[float]
        | tuple[float, ...]
        | np.ndarray
        | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample voltammogram(s) at ``sample_times`` for :class:`SampledCurrentVoltammetry`."""
        if not isinstance(self.technique, SampledCurrentVoltammetry):
            raise TypeError(
                "sample_currents() is only available for SampledCurrentVoltammetry results, "
                f"got {type(self.technique).__name__}."
            )
        return self.technique.sample_currents(
            self.time, self.current, sample_times=sample_times
        )
