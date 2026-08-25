"""
Analytical and semi-analytical models for OAM propagation.
"""

import numpy as np


def paterson_lg_oam_spectrum(
    ell_values: np.ndarray,
    transmitted_ell: int,
    w0: float,
    wavelength: float,
    propagation_distance: float,
    r0_total: float,
    radial_samples: int = 512,
    azimuthal_samples: int = 4096,
    radial_extent_factor: float = 6.0,
) -> np.ndarray:
    """
    Ensemble-averaged OAM spectrum predicted by the
    weak-turbulence rotational-coherence model of
    Paterson (2005) for an LG_p=0 beam.

    Parameters
    ----------
    ell_values:
        OAM charges at which the theoretical spectrum is
        evaluated.

    transmitted_ell:
        Transmitted OAM charge ell_0.

    w0:
        Gaussian waist parameter at z=0 [m].

    wavelength:
        Optical wavelength [m].

    propagation_distance:
        Propagation distance [m].

    r0_total:
        Fried parameter associated with the complete
        turbulent path [m].

    radial_samples:
        Number of radial quadrature samples.

    azimuthal_samples:
        Number of samples used for the Fourier decomposition
        of the rotational coherence function.

    radial_extent_factor:
        Maximum integration radius in units of w(z).

    Returns
    -------
    np.ndarray
        Normalized theoretical OAM spectrum evaluated at
        ``ell_values``.
    """

    ell_values = np.asarray(
        ell_values,
        dtype=np.int64,
    )

    if ell_values.ndim != 1:
        raise ValueError(
            "ell_values must be one-dimensional."
        )

    if w0 <= 0.0:
        raise ValueError(
            "w0 must be positive."
        )

    if wavelength <= 0.0:
        raise ValueError(
            "wavelength must be positive."
        )

    if propagation_distance < 0.0:
        raise ValueError(
            "propagation_distance must be non-negative."
        )

    if r0_total <= 0.0:
        raise ValueError(
            "r0_total must be positive."
        )

    if radial_samples <= 1:
        raise ValueError(
            "radial_samples must be greater than one."
        )

    if azimuthal_samples <= 1:
        raise ValueError(
            "azimuthal_samples must be greater than one."
        )

    maximum_delta_ell = int(
        np.max(
            np.abs(
                ell_values
                - transmitted_ell
            )
        )
    )

    if maximum_delta_ell >= (
        azimuthal_samples // 2
    ):
        raise ValueError(
            "azimuthal_samples is insufficient for the "
            "requested OAM interval."
        )

    # --------------------------------------------------------
    # Unperturbed Gaussian-beam radius at observation plane
    # --------------------------------------------------------

    rayleigh_range = (
        np.pi
        * w0**2
        / wavelength
    )

    beam_radius = (
        w0
        * np.sqrt(
            1.0
            + (
                propagation_distance
                / rayleigh_range
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Radial intensity profile of LG_0^ell
    # --------------------------------------------------------

    maximum_radius = (
        radial_extent_factor
        * beam_radius
    )

    radial_coordinate = np.linspace(
        0.0,
        maximum_radius,
        radial_samples,
        dtype=np.float64,
    )

    abs_ell = abs(
        transmitted_ell
    )

    radial_intensity = (
        (
            np.sqrt(2.0)
            * radial_coordinate
            / beam_radius
        ) ** (
            2
            * abs_ell
        )
        * np.exp(
            -2.0
            * radial_coordinate**2
            / beam_radius**2
        )
    )

    # Normalize with the same radial measure that appears
    # in the OAM probability integral.
    radial_norm = np.trapezoid(
        radial_intensity
        * radial_coordinate,
        radial_coordinate,
    )

    if (
        not np.isfinite(radial_norm)
        or radial_norm <= 0.0
    ):
        raise RuntimeError(
            "Invalid radial normalization."
        )

    radial_intensity /= (
        radial_norm
    )

    # --------------------------------------------------------
    # Rotational coherence function
    #
    # D_phi(rho) = 6.88 (rho/r0)^(5/3)
    #
    # C_phi = exp[-D_phi/2]
    #
    # rho = 2 r |sin(theta/2)|
    # --------------------------------------------------------

    theta = (
        2.0
        * np.pi
        * np.arange(
            azimuthal_samples,
            dtype=np.float64,
        )
        / azimuthal_samples
    )

    sin_half_theta = np.abs(
        np.sin(
            theta
            / 2.0
        )
    )

    separation = (
        2.0
        * radial_coordinate[
            :,
            np.newaxis
        ]
        * sin_half_theta[
            np.newaxis,
            :
        ]
    )

    rotational_coherence = np.exp(
        -3.44
        * (
            separation
            / r0_total
        ) ** (
            5.0
            / 3.0
        )
    )

    # --------------------------------------------------------
    # Fourier coefficients of rotational coherence
    # --------------------------------------------------------

    coherence_coefficients = (
        np.fft.fft(
            rotational_coherence,
            axis=1,
        )
        / azimuthal_samples
    )

    delta_ell = (
        ell_values
        - transmitted_ell
    )

    coefficient_indices = (
        delta_ell
        % azimuthal_samples
    )

    selected_coefficients = np.real(
        coherence_coefficients[
            :,
            coefficient_indices,
        ]
    )

    # --------------------------------------------------------
    # Radial integration
    # --------------------------------------------------------

    integrand = (
        radial_intensity[
            :,
            np.newaxis
        ]
        * selected_coefficients
        * radial_coordinate[
            :,
            np.newaxis
        ]
    )

    modal_probability = np.trapezoid(
        integrand,
        radial_coordinate,
        axis=0,
    )

    # Remove tiny negative roundoff errors.
    modal_probability = np.maximum(
        modal_probability,
        0.0,
    )

    total_probability = float(
        np.sum(
            modal_probability
        )
    )

    if (
        not np.isfinite(
            total_probability
        )
        or total_probability <= 0.0
    ):
        raise RuntimeError(
            "Invalid theoretical OAM spectrum."
        )

    # Normalize over the same ell interval used by the
    # numerical spectrum.
    modal_probability /= (
        total_probability
    )

    return np.asarray(
        modal_probability,
        dtype=np.float64,
    )
