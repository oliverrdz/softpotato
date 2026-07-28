from collections.abc import Callable
from typing import Any

import numpy as np

from softpotato.core.boundary_conditions import BoundaryHandler
from softpotato.core.pde_builder import PDEAssembler
from softpotato.grid.spatial import Grid
from softpotato.mechanism.builder import Mechanism
from softpotato.physics.kinetics import kinetics
from softpotato.physics.species import Species
from softpotato.results import SimulationResult
from softpotato.technique.base import Waveform


def simulate(
    waveform: Waveform,
    mechanism: Mechanism,
    grid: Grid,
    solver: Callable,
    species: dict[str, Species],
    kinetics: kinetics,
    params: dict[str, Any] | None = None,
) -> SimulationResult:
    """
    Orchestrates the electrochemical simulation pipeline by assembling PDEs,
    applying boundary conditions, and integrating over time using a numerical solver.

    Parameters:
    - waveform: Waveform object containing time vector t and potential vector E.
    - mechanism: Mechanism object containing reaction steps and species.
    - grid: Grid object containing spatial nodes x and intervals dx.
    - solver: Callable wrapper around ODE/PDE time integrator (e.g. solve_ivp).
    - species: Dictionary mapping species identifiers to Species dataclasses.
    - kinetics: Kinetics evaluator object or function.
    - params: Optional dictionary of solver tolerances or additional backend parameters.

    Returns:
    - SimulationResult object containing time series (t, E, i) and spatial concentration profiles.
    """
    if params is None:
        params = {}

    # 1. Initialize Assembler and Boundary Handler
    pde_assembler = PDEAssembler(
        grid=grid, mechanism=mechanism, species=species, kinetics=kinetics
    )
    boundary_handler = BoundaryHandler(
        grid=grid, mechanism=mechanism, species=species, kinetics=kinetics
    )

    species_keys = list(species.keys())
    n_nodes = len(grid.x)
    n_species = len(species_keys)

    # 2. Define the right-hand side (RHS) for the ODE solver as a function of (t, c)
    def rhs(t: float, c_flat: np.ndarray) -> np.ndarray:
        # Interpolate applied potential E(t) at current integration time t
        E_t = float(np.interp(t, waveform.t, waveform.E))

        # Reshape flat concentration vector back to spatial matrix per species: (n_species, n_nodes)
        c_matrix = c_flat.reshape((n_species, n_nodes))

        # Compute spatial diffusion and homogeneous reaction derivatives
        dc_dt = pde_assembler.assemble(c_matrix, t)

        # Apply electrode surface flux and bulk boundary conditions
        dc_dt = boundary_handler.apply(dc_dt, c_matrix, E_t)

        return dc_dt.flatten()

    # 3. Set initial concentration profile across grid (uniform bulk concentration c0)
    c0_list = []
    for key in species_keys:
        sp_obj = species[key]
        c0_list.append(np.full(n_nodes, sp_obj.c0))
    c_init = np.array(c0_list).flatten()

    # 4. Run time integration using the solver backend over the waveform time span
    t_span = (float(waveform.t[0]), float(waveform.t[-1]))
    t_eval = waveform.t

    sol = solver(rhs, t_span, c_init, t_eval=t_eval, **params)

    # 5. Process solution into SimulationResult container
    t_out = sol.t
    n_steps = len(t_out)
    c_sol = sol.y.reshape((n_species, n_nodes, n_steps))

    # Reconstruct applied potential vector matching output time steps
    E_out = np.interp(t_out, waveform.t, waveform.E)

    # Calculate electrode current response i(t) from surface concentration gradient at x=0
    # i = n * F * A * D * (dc/dx)_{x=0}
    i_out = np.zeros_like(t_out)
    if n_species > 0 and n_nodes > 1:
        primary_sp = species[species_keys[0]]
        D = primary_sp.D
        dx0 = grid.dx[0]
        # Concentration gradient at surface: (c[1] - c[0]) / dx0
        surface_gradient = (c_sol[0, 1, :] - c_sol[0, 0, :]) / dx0
        F = 96485.33212  # Faraday constant in C/mol
        i_out = F * D * surface_gradient

    # Package concentration profiles into a dictionary mapping species names to 2D arrays c(x, t)
    concentration_profiles = {key: c_sol[idx] for idx, key in enumerate(species_keys)}

    return SimulationResult(
        t=t_out, E=E_out, i=i_out, x=grid.x, concentrations=concentration_profiles
    )
