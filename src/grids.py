from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """
    Cartesian computational grid for 2D optical field simulations.

    Parameters
    ----------
    x, y:
        One-dimensional Cartesian coordinates.
    X, Y:
        Two-dimensional Cartesian coordinate meshes.
    r:
        Radial coordinate, r = sqrt(X^2 + Y^2).
    phi:
        Azimuthal coordinate, phi = arctan2(Y, X).
    dx:
        Spatial sampling interval.
    """

    x: RealArray
    y: RealArray
    X: RealArray
    Y: RealArray
    r: RealArray
    phi: RealArray
    dx: float


def create_grid(
    n: int,
    window_size: float,
) -> Grid:
    """
    Create a centered square Cartesian computational grid.

    Parameters
    ----------
    n:
        Number of samples along each spatial dimension.

    window_size:
        Physical size of the computational window [m].

    Returns
    -------
    Grid
        A Grid object containing Cartesian and polar coordinates.

    Notes
    -----
    The spatial sampling interval is

        dx = window_size / n.

    The coordinates are centered around the origin.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    dx = window_size / n

    coordinates = (
        np.arange(n) - n / 2
    ) * dx

    x = coordinates.copy()
    y = coordinates.copy()

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)

    return Grid(
        x=x,
        y=y,
        X=X,
        Y=Y,
        r=r,
        phi=phi,
        dx=dx,
    )
