"""
High-level simulation orchestrator and numerical PDE solver dispatcher.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from softpotato.core.constants import FARADAY
from softpotato.core.reactions import Mechanism
from softpotato.geometry.electrodes import Electrode
from softpotato.geometry.grids import Grid
from softpotato.techniques.voltammetry import Technique

from .efd import explicit_diffuse_step, update_surface_concentrations


@dataclass
class SimulationResult:
    """
    Container for electrochemical simulation output data.

    Follows the IUPAC sign convention:
    - Cathodic (reduction) current is negative (:math:`i < 0`).
    - Anodic (oxidation) current is positive (:math:`i > 0`).
    """

    time: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    species_profiles: dict[str, np.ndarray]
    technique: Technique
    mechanism: Mechanism
    electrode: Electrode

    def plot(self, show: bool = True, ax: Any = None) -> Any:
        """
        Convenience method to plot the simulated voltammogram (:math:`i` vs. :math:`E`).

        Parameters
        ----------
        show : bool, optional
            Whether to call `matplotlib.pyplot.show()`, by default True.
        ax : matplotlib.axes.Axes or None, optional
            Target axes for plotting. If None, creates a new figure and axes.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the plotted curve.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))

        ax.plot(self.potential, self.current * 1e6, lw=2, color="#1f77b4")
        ax.set_xlabel("Potential / V")
        ax.set_ylabel("Current / µA")
        ax.set_title("Cyclic Voltammogram")
        ax.grid(True, linestyle="--", alpha=0.7)

        if show:
            plt.show()

        return ax


class Solver:
    """
    High-level simulation orchestrator for electrochemical mass transport and kinetics.
    """

    def __init__(
        self,
        mechanism: Mechanism,
        electrode: Electrode,
        technique: Technique,
        method: str = "EFD",
        auto_substep: bool = True,
    ) -> None:
        """
        Initialize the simulation solver.

        Parameters
        ----------
        mechanism : Mechanism
            Reaction mechanism containing species, reactions, and kinetics.
        electrode : Electrode
            Electrode geometry with associated spatial discretization grid.
        technique : Technique
            Electrochemical excitation waveform.
        method : str, optional
            Numerical solver method, by default "EFD" (Explicit Finite Difference).
        auto_substep : bool, optional
            If True, automatically subdivides time steps when the CFL stability criterion
            (:math:`\\lambda = \\frac{D \\Delta t}{\\Delta x^2} \\le 0.5`) is exceeded,
            emitting a UserWarning. If False, raises ValueError upon CFL violation.

        Raises
        ------
        TypeError
            If input objects are of invalid types.
        ValueError
            If mechanism contains no electrochemical reactions or if method is unknown.
        NotImplementedError
            If an unsupported solver method is requested.
        """
        if not isinstance(mechanism, Mechanism):
            raise TypeError(
                f"mechanism must be a Mechanism instance, got {type(mechanism).__name__}."
            )
        if not isinstance(electrode, Electrode):
            raise TypeError(
                f"electrode must be an Electrode instance, got {type(electrode).__name__}."
            )
        if not isinstance(technique, Technique):
            raise TypeError(
                f"technique must be a Technique instance, got {type(technique).__name__}."
            )

        if len(mechanism.electrochemical_reactions) == 0:
            raise ValueError(
                "Mechanism must contain at least one ElectrochemicalReaction."
            )

        if method.upper() != "EFD":
            raise NotImplementedError(
                f"Solver method '{method}' is not implemented in v3.0.0. "
                "Only 'EFD' (Explicit Finite Difference) is currently supported."
            )

        self._mechanism = mechanism
        self._electrode = electrode
        self._technique = technique
        self._method = method.upper()
        self._auto_substep = bool(auto_substep)

    @property
    def mechanism(self) -> Mechanism:
        """Reaction mechanism."""
        return self._mechanism

    @property
    def electrode(self) -> Electrode:
        """Electrode geometry."""
        return self._electrode

    @property
    def technique(self) -> Technique:
        """Excitation technique."""
        return self._technique

    @property
    def method(self) -> str:
        """Numerical method."""
        return self._method

    def run(self) -> SimulationResult:
        """
        Execute the electrochemical simulation.

        Returns
        -------
        SimulationResult
            Container with simulated time, potential, current, and concentration profiles.
        """
        rxn = self._mechanism.electrochemical_reactions[0]
        if rxn.kinetics is None:
            raise ValueError(
                f"Electrochemical reaction {rxn} has no kinetics model assigned."
            )

        spec_O = rxn.reactants[0]
        spec_R = rxn.products[0]
        n_electrons = rxn.n_electrons
        E0 = rxn.E0
        kinetics = rxn.kinetics

        grid: Grid = self._electrode.grid
        dx = grid.dx
        if isinstance(dx, np.ndarray):
            raise NotImplementedError(
                "Non-uniform grid simulation is planned for v3.3.0."
            )

        nodes = grid.nodes
        x_max = grid.x_max
        area = self._electrode.area
        dt = self._technique.dt
        t_arr = self._technique.time
        E_arr = self._technique.potential
        duration = self._technique.duration

        # Validate semi-infinite diffusion boundary condition
        D_max = max(spec_O.D, spec_R.D)
        diffusion_layer_est = 6.0 * np.sqrt(D_max * max(duration, 1e-6))
        if x_max < diffusion_layer_est:
            warnings.warn(
                f"x_max ({x_max:.4e} cm) may be too small for semi-infinite diffusion. "
                f"Estimated diffusion layer thickness is ~{diffusion_layer_est:.4e} cm.",
                UserWarning,
                stacklevel=2,
            )

        # Check Courant-Friedrichs-Lewy (CFL) stability criterion: lambda = D * dt / dx^2 <= 0.5
        lambda_max = D_max * dt / (dx**2)
        m_sub = 1
        if lambda_max > 0.5:
            if self._auto_substep:
                m_sub = int(np.ceil(lambda_max / 0.45))
                warnings.warn(
                    f"CFL stability condition exceeded (lambda = {lambda_max:.3f} > 0.5). "
                    f"Automatically sub-stepping with {m_sub} internal steps per technique point.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    f"CFL stability condition violated: lambda = {lambda_max:.3f} > 0.5. "
                    "Refine grid, decrease time step, or set auto_substep=True."
                )

        dt_sub = dt / m_sub

        # Initialize concentration profiles
        self._mechanism.initialize_profiles(nodes)
        c_O = spec_O.c_profile
        c_R = spec_R.c_profile
        assert c_O is not None and c_R is not None

        n_pts = len(t_arr)
        current = np.zeros(n_pts, dtype=np.float64)

        # Time-stepping simulation loop
        for k in range(n_pts):
            E_curr = E_arr[k]
            if k == 0:
                kf, kb = kinetics.calculate_rates(E_curr, E0, n_electrons)
                c_O_0, c_R_0, flux = update_surface_concentrations(
                    c_O[1], c_R[1], spec_O.D, spec_R.D, dx, kf, kb
                )
                c_O[0] = c_O_0
                c_R[0] = c_R_0
                # IUPAC convention: reduction current is negative
                current[0] = -n_electrons * FARADAY * area * flux
            else:
                E_prev = E_arr[k - 1]
                for m in range(1, m_sub + 1):
                    E_sub = E_prev + (m / m_sub) * (E_curr - E_prev)
                    kf, kb = kinetics.calculate_rates(E_sub, E0, n_electrons)

                    # 1. Update surface boundary condition from adjacent node
                    c_O_0, c_R_0, _ = update_surface_concentrations(
                        c_O[1], c_R[1], spec_O.D, spec_R.D, dx, kf, kb
                    )
                    c_O[0] = c_O_0
                    c_R[0] = c_R_0

                    # 2. Vectorized diffusion step across interior nodes
                    c_O = explicit_diffuse_step(c_O, spec_O.D, dt_sub, dx)
                    c_R = explicit_diffuse_step(c_R, spec_R.D, dt_sub, dx)

                    # 3. Enforce bulk boundary condition
                    c_O[-1] = spec_O.c_bulk
                    c_R[-1] = spec_R.c_bulk

                # Compute current at end of time step k
                kf, kb = kinetics.calculate_rates(E_curr, E0, n_electrons)
                c_O_0, c_R_0, flux = update_surface_concentrations(
                    c_O[1], c_R[1], spec_O.D, spec_R.D, dx, kf, kb
                )
                c_O[0] = c_O_0
                c_R[0] = c_R_0
                # IUPAC convention: reduction current is negative
                current[k] = -n_electrons * FARADAY * area * flux

        # Update species profiles
        spec_O.c_profile = c_O
        spec_R.c_profile = c_R

        profiles = {
            sp.name: sp.c_profile.copy()
            for sp in self._mechanism.species
            if sp.c_profile is not None
        }

        result = SimulationResult(
            time=t_arr.copy(),
            potential=E_arr.copy(),
            current=current,
            species_profiles=profiles,
            technique=self._technique,
            mechanism=self._mechanism,
            electrode=self._electrode,
        )

        # Reset mechanism profiles for subsequent runs
        self._mechanism.reset()

        return result


def solve_1d(*args: Any, **kwargs: Any) -> Any:
    """
    Convenience function for 1D electrochemical simulation.
    """
    raise NotImplementedError("Use sp.simulate.Solver for 1D simulations.")


def solve_homogeneous_1d(*args: Any, **kwargs: Any) -> Any:
    """
    Solve 1D homogeneous chemical reactions coupled with mass transport.

    .. warning::
        Planned for v3.1.0.
    """
    raise NotImplementedError("Homogeneous reaction solvers are planned for v3.1.0.")
