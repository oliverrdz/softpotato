from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """
    Dataclass representing 1D spatial node coordinates and their intervals.
    """

    x: np.ndarray
    dx: np.ndarray
