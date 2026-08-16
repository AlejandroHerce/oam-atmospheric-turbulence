"""
Phase structure-function utilities.

This module contains numerical methods used to evaluate the phase
structure function of atmospheric turbulence realizations.
"""

import numpy as np
from numpy.typing import NDArray

from scipy.integrate import quad
from scipy.special import gamma, j0, kv

from src.phase_screens import (
    modified_von_karman_psd,
)

RealArray = NDArray[np.float64]


# ============================================================
# Numerical phase structure function
# ============================================================

def structure_function_xy(
    phase: RealArray,
    delta: float,
    max_shift: int,
) -> tuple[RealArray, RealArray]:
    """
    Calculate the phase structure function along the x and y
    directions and average both estimates.

    Linear autocorrelation is evaluated using zero-padded FFTs,
    avoiding artificial periodic correlations across the numerical
    boundaries.

    Parameters
    ----------
    phase:
        Two-dimensional phase screen [rad].

    delta:
        Spatial sampling interval [m].

    max_shift:
        Maximum pixel displacement considered. Structure-function
        values are evaluated for shifts 1, ..., max_shift - 1.

    Returns
    -------
    rho:
        Spatial separations [m].

    structure:
        Phase structure function D_phi(rho) [rad^2].
    """

    if phase.ndim != 2:
        raise ValueError(
            "phase must be a two-dimensional array."
        )

    ny, nx = phase.shape

    if nx != ny:
        raise ValueError(
            "phase must be square."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    if max_shift <= 1:
        raise ValueError(
            "max_shift must be greater than one."
        )

    if max_shift > nx:
        raise ValueError(
            "max_shift cannot exceed the grid size."
        )

    n = nx

    shifts = np.arange(
        1,
        max_shift,
        dtype=int,
    )

    # --------------------------------------------------------
    # x direction
    # --------------------------------------------------------

    spectrum_x = np.fft.rfft(
        phase,
        n=2 * n,
        axis=1,
    )

    correlation_x = np.fft.irfft(
        np.sum(
            np.abs(spectrum_x) ** 2,
            axis=0,
        ),
        n=2 * n,
    )[:n]

    column_energy = np.sum(
        phase**2,
        axis=0,
    )

    cumulative_x = np.concatenate(
        (
            [0.0],
            np.cumsum(column_energy),
        )
    )

    total_x = cumulative_x[-1]

    first_energy_x = cumulative_x[
        n - shifts
    ]

    second_energy_x = (
        total_x
        - cumulative_x[shifts]
    )

    number_pairs_x = (
        ny
        * (n - shifts)
    )

    structure_x = (
        first_energy_x
        + second_energy_x
        - 2.0 * correlation_x[shifts]
    ) / number_pairs_x

    # --------------------------------------------------------
    # y direction
    # --------------------------------------------------------

    spectrum_y = np.fft.rfft(
        phase,
        n=2 * n,
        axis=0,
    )

    correlation_y = np.fft.irfft(
        np.sum(
            np.abs(spectrum_y) ** 2,
            axis=1,
        ),
        n=2 * n,
    )[:n]

    row_energy = np.sum(
        phase**2,
        axis=1,
    )

    cumulative_y = np.concatenate(
        (
            [0.0],
            np.cumsum(row_energy),
        )
    )

    total_y = cumulative_y[-1]

    first_energy_y = cumulative_y[
        n - shifts
    ]

    second_energy_y = (
        total_y
        - cumulative_y[shifts]
    )

    number_pairs_y = (
        nx
        * (n - shifts)
    )

    structure_y = (
        first_energy_y
        + second_energy_y
        - 2.0 * correlation_y[shifts]
    ) / number_pairs_y

    # --------------------------------------------------------
    # Directional average
    # --------------------------------------------------------

    rho = (
        shifts
        * delta
    )

    structure = (
        0.5
        * (
            structure_x
            + structure_y
        )
    )

    # Remove only tiny negative values caused by floating-point
    # roundoff.
    structure = np.maximum(
        structure,
        0.0,
    )

    return (
        rho.astype(np.float64),
        structure.astype(np.float64),
    )


# ============================================================
# Kolmogorov theoretical structure function
# ============================================================

def kolmogorov_structure_function(
    rho: RealArray,
    r0: float,
) -> RealArray:
    """
    Calculate the theoretical Kolmogorov phase structure function.

        D_phi(rho) = 6.88 (rho/r0)^(5/3)
    """

    if r0 <= 0:
        raise ValueError(
            "r0 must be positive."
        )

    if np.any(rho < 0):
        raise ValueError(
            "rho must not contain negative values."
        )

    return (
        6.88
        * (rho / r0) ** (5.0 / 3.0)
    )

# ============================================================
# von Kármán theoretical structure function
# ============================================================

def von_karman_structure_function(
    rho: RealArray,
    r0: float,
    outer_scale: float,
) -> RealArray:
    """
    Calculate the theoretical von Kármán phase structure function.

    The expression is consistent with the phase PSD

        Phi_phi(kappa)
        =
        0.49 r0^(-5/3)
        (kappa^2 + kappa0^2)^(-11/6),

    where

        kappa0 = 2*pi/L0.
    """

    if r0 <= 0:
        raise ValueError(
            "r0 must be positive."
        )

    if outer_scale <= 0:
        raise ValueError(
            "outer_scale must be positive."
        )

    rho = np.asarray(
        rho,
        dtype=np.float64,
    )

    if np.any(rho < 0.0):
        raise ValueError(
            "rho must not contain negative values."
        )

    kappa0 = (
        2.0
        * np.pi
        / outer_scale
    )

    c_vk = (
        4.0
        * np.pi
        * 0.49
        * (3.0 / 5.0)
        * (2.0 * np.pi) ** (-5.0 / 3.0)
    )

    x = (
        kappa0
        * rho
    )

    result = np.zeros_like(
        x,
        dtype=np.float64,
    )

    positive = (
        x > 0.0
    )

    xp = (
        x[positive]
    )

    bessel_term = (
        2.0 ** (1.0 / 6.0)
        / gamma(5.0 / 6.0)
        * xp ** (5.0 / 6.0)
        * kv(
            5.0 / 6.0,
            xp,
        )
    )

    result[positive] = (
        c_vk
        * (
            outer_scale
            / r0
        ) ** (5.0 / 3.0)
        * (
            1.0
            - bessel_term
        )
    )

    return result


# ============================================================
# Modified von Kármán theoretical structure function
# ============================================================

def modified_von_karman_structure_function(
    rho: RealArray,
    r0: float,
    outer_scale: float,
    inner_scale: float,
) -> RealArray:
    """
    Calculate the modified von Kármán phase structure function
    by numerical integration of the phase PSD.

        D_phi(rho)
        =
        4*pi integral[
            Phi_phi(kappa)
            (1 - J0(kappa*rho))
            kappa d kappa
        ].
    """

    if r0 <= 0:
        raise ValueError(
            "r0 must be positive."
        )

    if outer_scale <= 0:
        raise ValueError(
            "outer_scale must be positive."
        )

    if inner_scale <= 0:
        raise ValueError(
            "inner_scale must be positive."
        )

    rho = np.asarray(
        rho,
        dtype=np.float64,
    )

    if np.any(rho < 0.0):
        raise ValueError(
            "rho must not contain negative values."
        )

    kappa_m = (
        5.92
        / inner_scale
    )

    kappa_max = (
        8.0
        * kappa_m
    )

    structure = np.zeros_like(
        rho,
        dtype=np.float64,
    )

    for index, separation in enumerate(
        rho
    ):
        def integrand(
            kappa: float,
        ) -> float:
            psd = modified_von_karman_psd(
                kappa=np.asarray(kappa),
                r0=r0,
                outer_scale=outer_scale,
                inner_scale=inner_scale,
            )

            return float(
                4.0
                * np.pi
                * psd
                * (
                    1.0
                    - j0(
                        kappa
                        * separation
                    )
                )
                * kappa
            )

        structure[index], _ = quad(
            integrand,
            0.0,
            kappa_max,
            epsabs=1e-8,
            epsrel=1e-7,
            limit=500,
        )

    return structure
