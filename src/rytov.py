"""
Rytov-theory utilities for weak atmospheric turbulence.

This module contains the analytical expressions used to validate
the split-step propagation model in the weak-turbulence regime.
"""

import numpy as np

from scipy.special import hyp2f1



# ============================================================
# Plane-wave Rytov variance
# ============================================================

def plane_wave_rytov_variance(
    cn2: float,
    wavelength: float,
    distance: float,
) -> float:
    """
    Calculate the plane-wave Rytov variance:

        sigma_R^2
        =
        1.23 Cn^2 k^(7/6) z^(11/6).

    Parameters
    ----------
    cn2:
        Refractive-index structure parameter [m^(-2/3)].

    wavelength:
        Optical wavelength [m].

    distance:
        Propagation distance [m].
    """

    if cn2 < 0.0:
        raise ValueError(
            "cn2 must be non-negative."
        )

    if wavelength <= 0.0:
        raise ValueError(
            "wavelength must be positive."
        )

    if distance < 0.0:
        raise ValueError(
            "distance must be non-negative."
        )

    if distance == 0.0 or cn2 == 0.0:
        return 0.0

    wave_number = (
        2.0
        * np.pi
        / wavelength
    )

    return float(
        1.23
        * cn2
        * wave_number ** (7.0 / 6.0)
        * distance ** (11.0 / 6.0)
    )


# ============================================================
# Gaussian-beam dimensionless parameters
# ============================================================

def gaussian_beam_rytov_parameters(
    wavelength: float,
    waist_radius: float,
    distance: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Calculate the output-plane Gaussian-beam parameters used in
    the Rytov scintillation expression.

    The beam waist is assumed to be located at the input plane,
    so that

        Theta_0 = 1

    and

        Lambda_0 = 2 z / (k w0^2).

    Returns
    -------
    theta:
        Output-plane Theta parameter.

    lambda_parameter:
        Output-plane Lambda parameter.

    theta_bar:
        1 - Theta.
    """

    if wavelength <= 0.0:
        raise ValueError(
            "wavelength must be positive."
        )

    if waist_radius <= 0.0:
        raise ValueError(
            "waist_radius must be positive."
        )

    if distance < 0.0:
        raise ValueError(
            "distance must be non-negative."
        )

    if distance == 0.0:
        return (
            1.0,
            0.0,
            0.0,
        )

    wave_number = (
        2.0
        * np.pi
        / wavelength
    )

    lambda_0 = (
        2.0
        * distance
        / (
            wave_number
            * waist_radius**2
        )
    )

    denominator = (
        1.0
        + lambda_0**2
    )

    theta = (
        1.0
        / denominator
    )

    lambda_parameter = (
        lambda_0
        / denominator
    )

    theta_bar = (
        1.0
        - theta
    )

    return (
        float(theta),
        float(lambda_parameter),
        float(theta_bar),
    )


# ============================================================
# On-axis Gaussian-beam scintillation index
# ============================================================

def gaussian_on_axis_scintillation_index(
    cn2: float,
    wavelength: float,
    waist_radius: float,
    distance: float,
) -> float:
    """
    Calculate the weak-turbulence Rytov prediction for the
    on-axis scintillation index of a Gaussian beam:

        sigma_I^2(0,z)
        =
        3.86 sigma_R^2 Re[
            i^(5/6)
            2F1(
                -5/6,
                11/6;
                17/6;
                theta_bar + i lambda
            )
        ]
        -
        2.64 sigma_R^2 lambda^(5/6).

    The beam waist is assumed to be located at z = 0.
    """

    if distance == 0.0:
        return 0.0

    sigma_r_squared = (
        plane_wave_rytov_variance(
            cn2=cn2,
            wavelength=wavelength,
            distance=distance,
        )
    )

    (
        theta,
        lambda_parameter,
        theta_bar,
    ) = gaussian_beam_rytov_parameters(
        wavelength=wavelength,
        waist_radius=waist_radius,
        distance=distance,
    )

    del theta

    hypergeometric_argument = (
        theta_bar
        + 1j * lambda_parameter
    )

    hypergeometric_term = hyp2f1(
        -5.0 / 6.0,
        11.0 / 6.0,
        17.0 / 6.0,
        hypergeometric_argument,
    )

    complex_prefactor = (
        1j ** (5.0 / 6.0)
    )

    first_term = (
        3.86
        * sigma_r_squared
        * np.real(
            complex_prefactor
            * hypergeometric_term
        )
    )

    second_term = (
        2.64
        * sigma_r_squared
        * lambda_parameter ** (5.0 / 6.0)
    )

    scintillation_index = (
        first_term
        - second_term
    )

    return float(
        scintillation_index
    )


# ============================================================
# Rytov curve
# ============================================================

def gaussian_on_axis_scintillation_curve(
    distances: np.ndarray,
    cn2: float,
    wavelength: float,
    waist_radius: float,
) -> np.ndarray:
    """
    Evaluate the on-axis Gaussian-beam scintillation index at
    multiple propagation distances.
    """

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    if distances.ndim != 1:
        raise ValueError(
            "distances must be one-dimensional."
        )

    if np.any(
        distances < 0.0
    ):
        raise ValueError(
            "distances must not contain negative values."
        )

    return np.array(
        [
            gaussian_on_axis_scintillation_index(
                cn2=cn2,
                wavelength=wavelength,
                waist_radius=waist_radius,
                distance=float(distance),
            )
            for distance in distances
        ],
        dtype=np.float64,
    )

