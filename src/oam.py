"""
Orbital angular momentum (OAM) analysis utilities.

This module contains the numerical tools used to compute the azimuthal
OAM spectrum of a complex optical field sampled on a Cartesian grid.

The implementation reproduces the Cartesian-to-polar interpolation and
azimuthal Fourier decomposition used in the Chapter 2 OAM-conservation
validation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from src.grids import Grid


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


# ============================================================
# Cartesian-to-polar interpolation
# ============================================================

def interpolate_to_polar_grid(
    field: ComplexArray,
    grid: Grid,
    radial_samples: int = 256,
    azimuthal_samples: int = 720,
    maximum_radius: float | None = None,
) -> tuple[RealArray, RealArray, ComplexArray]:
    """
    Interpolate a Cartesian optical field U(x, y) onto a polar grid U(r, phi).

    Parameters
    ----------
    field:
        Complex optical field sampled on ``grid``.

    grid:
        Cartesian computational grid associated with ``field``.

    radial_samples:
        Number of radial samples used in the polar representation.

    azimuthal_samples:
        Number of uniformly spaced azimuthal samples over [0, 2*pi).

    maximum_radius:
        Maximum radius of the polar grid [m]. If ``None``, an inscribed
        circle occupying 98% of the available Cartesian half-width is used.

    Returns
    -------
    r:
        One-dimensional radial coordinate [m].

    phi:
        One-dimensional azimuthal coordinate [rad].

    polar_field:
        Complex field sampled on the polar grid, with shape
        ``(radial_samples, azimuthal_samples)``.

    Notes
    -----
    The polar domain is restricted to an inscribed circle so that the
    interpolation does not require values outside the Cartesian domain.
    """

    if field.shape != grid.X.shape:
        raise ValueError(
            "The field shape must match the computational grid shape."
        )

    if radial_samples <= 1:
        raise ValueError("radial_samples must be greater than 1.")

    if azimuthal_samples <= 1:
        raise ValueError("azimuthal_samples must be greater than 1.")

    if maximum_radius is None:
        maximum_radius = 0.98 * min(
            abs(grid.x[0]),
            abs(grid.x[-1]),
            abs(grid.y[0]),
            abs(grid.y[-1]),
        )

    if maximum_radius <= 0:
        raise ValueError("maximum_radius must be positive.")

    r = np.linspace(
        0.0,
        maximum_radius,
        radial_samples,
    )

    phi = np.linspace(
        0.0,
        2.0 * np.pi,
        azimuthal_samples,
        endpoint=False,
    )

    R, PHI = np.meshgrid(
        r,
        phi,
        indexing="ij",
    )

    x_polar = R * np.cos(PHI)
    y_polar = R * np.sin(PHI)

    real_interpolator = RegularGridInterpolator(
        (grid.y, grid.x),
        np.real(field),
        bounds_error=False,
        fill_value=0.0,
    )

    imag_interpolator = RegularGridInterpolator(
        (grid.y, grid.x),
        np.imag(field),
        bounds_error=False,
        fill_value=0.0,
    )

    interpolation_points = np.column_stack(
        (
            y_polar.ravel(),
            x_polar.ravel(),
        )
    )

    polar_field = (
        real_interpolator(interpolation_points)
        + 1j * imag_interpolator(interpolation_points)
    ).reshape(R.shape)

    return (
        r,
        phi,
        polar_field.astype(np.complex128),
    )


# ============================================================
# OAM spectrum
# ============================================================

def calculate_oam_spectrum(
    field: ComplexArray,
    grid: Grid,
    ell_min: int = -8,
    ell_max: int = 8,
    radial_samples: int = 256,
    azimuthal_samples: int = 720,
    maximum_radius: float | None = None,
) -> tuple[IntegerArray, RealArray]:
    """
    Calculate the normalized azimuthal OAM spectrum of an optical field.

    The field is decomposed as

        U(r, phi) = sum_l c_l(r) exp(i l phi),

    with

        c_l(r) = (1 / 2*pi) integral U(r, phi) exp(-i l phi) dphi.

    The power associated with each azimuthal component is

        P_l = 2*pi integral |c_l(r)|^2 r dr.

    The returned modal powers are normalized over the requested interval
    ``ell_min <= ell <= ell_max`` in order to reproduce the Chapter 2
    validation procedure.

    Parameters
    ----------
    field:
        Complex optical field sampled on ``grid``.

    grid:
        Cartesian computational grid associated with ``field``.

    ell_min, ell_max:
        Minimum and maximum topological charges included in the spectrum.

    radial_samples, azimuthal_samples:
        Resolution of the intermediate polar grid.

    maximum_radius:
        Optional maximum polar radius [m].

    Returns
    -------
    ell_values:
        Integer OAM charges included in the calculation.

    modal_power:
        Normalized modal-power distribution over ``ell_values``.
    """

    if ell_min > ell_max:
        raise ValueError("ell_min must be less than or equal to ell_max.")

    r, _, polar_field = interpolate_to_polar_grid(
        field=field,
        grid=grid,
        radial_samples=radial_samples,
        azimuthal_samples=azimuthal_samples,
        maximum_radius=maximum_radius,
    )

    # Fourier transform along the azimuthal coordinate. Dividing by N_phi
    # provides the discrete approximation to the 1/(2*pi) Fourier-series
    # normalization used in the continuous definition of c_l(r).
    coefficients = (
        np.fft.fft(polar_field, axis=1)
        / azimuthal_samples
    )

    coefficients = np.fft.fftshift(
        coefficients,
        axes=1,
    )

    available_ell = np.fft.fftshift(
        np.fft.fftfreq(
            azimuthal_samples,
            d=1.0 / azimuthal_samples,
        )
    ).astype(int)

    ell_values = np.arange(
        ell_min,
        ell_max + 1,
        dtype=np.int64,
    )

    modal_power = np.zeros(
        ell_values.size,
        dtype=np.float64,
    )

    for index, ell in enumerate(ell_values):
        position = np.where(available_ell == ell)[0]

        if position.size != 1:
            raise RuntimeError(
                f"The azimuthal component ell={ell} is not available."
            )

        radial_coefficient = coefficients[:, position[0]]

        integrand = (
            np.abs(radial_coefficient) ** 2
            * r
        )

        modal_power[index] = (
            2.0
            * np.pi
            * np.trapezoid(integrand, r)
        )

    total_power = float(np.sum(modal_power))

    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError(
            "The calculated OAM spectrum has invalid or zero power."
        )

    modal_power /= total_power

    return ell_values, modal_power


# ============================================================
# Derived quantities used in Chapter 2 validation
# ============================================================

def calculate_mean_oam(
    ell_values: IntegerArray,
    modal_power: RealArray,
) -> float:
    """
    Calculate the dimensionless expectation value of the OAM charge.

        <ell> = sum_l l P_l.

    The corresponding physical angular momentum per photon is

        <L_z> = hbar <ell>.
    """

    _validate_spectrum_arrays(
        ell_values,
        modal_power,
    )

    return float(
        np.sum(ell_values * modal_power)
    )


def modal_power_at_charge(
    ell_values: IntegerArray,
    modal_power: RealArray,
    charge: int,
) -> float:
    """
    Return the normalized modal power associated with a selected charge.
    """

    _validate_spectrum_arrays(
        ell_values,
        modal_power,
    )

    position = np.where(
        ell_values == charge
    )[0]

    if position.size != 1:
        raise ValueError(
            f"Charge ell={charge} is outside the analyzed OAM range."
        )

    return float(
        modal_power[position[0]]
    )


def spectral_l1_distance(
    spectrum_a: RealArray,
    spectrum_b: RealArray,
) -> float:
    """
    Calculate the L1 distance between two OAM spectra.

        D_L1 = sum_l |P_l - Q_l|.

    This is the same spectral-difference metric used in the Chapter 2
    conservation test.
    """

    if spectrum_a.shape != spectrum_b.shape:
        raise ValueError(
            "The two spectra must have the same shape."
        )

    return float(
        np.sum(
            np.abs(
                spectrum_a - spectrum_b
            )
        )
    )


# ============================================================
# Internal validation
# ============================================================

def _validate_spectrum_arrays(
    ell_values: IntegerArray,
    modal_power: RealArray,
) -> None:
    """Validate compatible one-dimensional OAM-spectrum arrays."""

    if ell_values.ndim != 1 or modal_power.ndim != 1:
        raise ValueError(
            "ell_values and modal_power must be one-dimensional arrays."
        )

    if ell_values.size != modal_power.size:
        raise ValueError(
            "ell_values and modal_power must have the same length."
        )

def calculate_rms_oam_spread(
    ell_values: NDArray[np.int64],
    modal_power: RealArray,
    transmitted_charge: int,
) -> float:
    """
    Calculate the RMS OAM spread with respect to the transmitted mode.

        sigma_Delta_ell =
        sqrt[
            sum_l (l - l0)^2 P_l
        ]

    Unlike the conventional standard deviation, the reference is
    the transmitted OAM charge l0 rather than the mean OAM.
    """

    if ell_values.shape != modal_power.shape:
        raise ValueError(
            "ell_values and modal_power must have the same shape."
        )

    if np.any(modal_power < 0.0):
        raise ValueError(
            "modal_power must not contain negative values."
        )

    total_power = float(
        np.sum(modal_power)
    )

    if total_power <= 0.0:
        raise ValueError(
            "modal_power must contain positive total power."
        )

    normalized_power = (
        modal_power
        / total_power
    )

    return float(
        np.sqrt(
            np.sum(
                (
                    ell_values
                    - transmitted_charge
                ) ** 2
                * normalized_power
            )
        )
    )
