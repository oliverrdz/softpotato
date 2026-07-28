import numpy as np

from softpotato.grid.spatial import Grid


def planar(
    x_max: float, nx: int, grid_type: str = "expanding", gamma: float = 3.0
) -> Grid:
    """
    Generates a 1D planar spatial mesh.

    Parameters
    ----------
    x_max : float
        Maximum spatial distance (bulk boundary), must be > 0.
    nx : int
        Number of spatial nodes, must be > 1.
    grid_type : str, optional
        Type of mesh distribution: 'uniform' or 'expanding' (default).
    gamma : float, optional
        Expansion factor for non-uniform meshes (default is 3.0).

    Returns
    -------
    Grid
        Object containing node coordinates (x) and intervals (dx).
    """
    if x_max <= 0:
        raise ValueError("Maximum spatial distance x_max must be greater than 0.")
    if nx <= 1:
        raise ValueError("Number of nodes nx must be greater than 1.")

    if grid_type == "uniform":
        x = np.linspace(0, x_max, nx)
    elif grid_type in ("expanding", "exponential"):
        # Normalized indices from 0 to 1
        indices = np.linspace(0, 1, nx)
        # Exponential expansion clustering points near x=0
        x = x_max * (np.exp(gamma * indices) - 1) / (np.exp(gamma) - 1)
    else:
        raise ValueError(
            f"Unknown grid_type: '{grid_type}'. Use 'uniform' or 'expanding'."
        )

    # Calculate intervals between adjacent nodes
    dx = np.diff(x)

    # Duplicate the final dx to maintain consistent array lengths
    # (nx points should have nx spacing evaluations for the downstream PDE builder)
    dx = np.append(dx, dx[-1])

    return Grid(x=x, dx=dx)
