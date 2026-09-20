"""
Explicit finite difference methods for 1D electrochemical diffusion.
"""

from __future__ import annotations

import numpy as np


def explicit_diffuse_step(
    c: np.ndarray,
    D: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    Advance concentration profile by one explicit Euler time step for 1D planar diffusion.

    Uses fully vectorized NumPy slice operations:
    :math:`c_i^{m+1} = c_i^m + \\lambda (c_{i+1}^m - 2 c_i^m + c_{i-1}^m)`
    where :math:`\\lambda = \\frac{D \\Delta t}{\\Delta x^2}`.

    Parameters
    ----------
    c : np.ndarray
        1D array of concentration values across grid nodes.
    D : float
        Diffusion coefficient in cm²/s.
    dt : float
        Time step in seconds.
    dx : float
        Spatial node spacing in cm.

    Returns
    -------
    np.ndarray
        Updated concentration profile array.
    """
    lambda_param = D * dt / (dx**2)
    c_new = c.copy()
    # Vectorized interior diffusion update
    c_new[1:-1] = c[1:-1] + lambda_param * (c[2:] - 2.0 * c[1:-1] + c[:-2])
    return c_new


def update_surface_concentrations(
    c_O_interior: float,
    c_R_interior: float,
    D_O: float,
    D_R: float,
    dx: float,
    kf: float,
    kb: float,
) -> tuple[float, float, float]:
    """
    Compute surface concentrations :math:`(c_{O,0}, c_{R,0})` and net flux :math:`J`
    for an :math:`E` mechanism (:math:`O + n e^- \\rightleftharpoons R`) using
    the finite difference flux boundary condition:

    .. math::

        \\frac{D_O}{\\Delta x} (c_{O,1} - c_{O,0}) = J = k_f c_{O,0} - k_b c_{R,0}

    .. math::

        \\frac{D_R}{\\Delta x} (c_{R,1} - c_{R,0}) = -J = -(k_f c_{O,0} - k_b c_{R,0})

    Parameters
    ----------
    c_O_interior : float
        Concentration of O at node 1 (adjacent to surface) in mol/cm³.
    c_R_interior : float
        Concentration of R at node 1 (adjacent to surface) in mol/cm³.
    D_O : float
        Diffusion coefficient of O in cm²/s.
    D_R : float
        Diffusion coefficient of R in cm²/s.
    dx : float
        Spatial grid spacing in cm.
    kf : float
        Forward heterogeneous rate constant in cm/s.
    kb : float
        Backward heterogeneous rate constant in cm/s.

    Returns
    -------
    tuple[float, float, float]
        (c_O_surface, c_R_surface, flux) where flux :math:`J = k_f c_{O,0} - k_b c_{R,0}`
        in mol/(cm²·s).
    """
    h_O = D_O / dx
    h_R = D_R / dx

    det = h_O * h_R + h_O * kb + h_R * kf
    if det == 0.0:
        return c_O_interior, c_R_interior, 0.0

    c_O_0 = ((h_R + kb) * (h_O * c_O_interior) + kb * (h_R * c_R_interior)) / det
    c_R_0 = (kf * (h_O * c_O_interior) + (h_O + kf) * (h_R * c_R_interior)) / det

    c_O_0 = max(float(c_O_0), 0.0)
    c_R_0 = max(float(c_R_0), 0.0)

    flux = kf * c_O_0 - kb * c_R_0
    return c_O_0, c_R_0, float(flux)
