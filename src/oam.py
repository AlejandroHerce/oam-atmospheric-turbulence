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

from pathlib import Path

import matplotlib.pyplot as plt

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

        c_l(r) = (1 / 2*pi)
                 integral U(r, phi) exp(-i l phi) dphi.

    The power associated with each azimuthal component is

        W_l = 2*pi integral |c_l(r)|^2 r dr.

    The returned modal powers are normalized over the requested
    interval ``ell_min <= ell <= ell_max``.

    The radial integrations for all requested OAM components are
    performed simultaneously using vectorized NumPy operations.
    """

    if ell_min > ell_max:
        raise ValueError(
            "ell_min must be less than or equal to ell_max."
        )

    if radial_samples <= 1:
        raise ValueError(
            "radial_samples must be greater than 1."
        )

    if azimuthal_samples <= 1:
        raise ValueError(
            "azimuthal_samples must be greater than 1."
        )

    # --------------------------------------------------------
    # Cartesian -> polar interpolation
    # --------------------------------------------------------

    (
        r,
        _,
        polar_field,
    ) = interpolate_to_polar_grid(
        field=field,
        grid=grid,
        radial_samples=radial_samples,
        azimuthal_samples=azimuthal_samples,
        maximum_radius=maximum_radius,
    )

    # --------------------------------------------------------
    # Azimuthal Fourier decomposition
    # --------------------------------------------------------

    coefficients = (
        np.fft.fft(
            polar_field,
            axis=1,
        )
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
    ).astype(np.int64)

    # --------------------------------------------------------
    # Validate requested OAM interval
    # --------------------------------------------------------

    available_min = int(
        available_ell[0]
    )

    available_max = int(
        available_ell[-1]
    )

    if (
        ell_min < available_min
        or ell_max > available_max
    ):
        raise ValueError(
            "Requested OAM interval "
            f"[{ell_min}, {ell_max}] "
            "lies outside the interval supported by the "
            f"azimuthal sampling "
            f"[{available_min}, {available_max}]."
        )

    # --------------------------------------------------------
    # Select requested OAM components
    # --------------------------------------------------------

    selection_mask = (
        (available_ell >= ell_min)
        & (available_ell <= ell_max)
    )

    ell_values = (
        available_ell[
            selection_mask
        ]
    )

    selected_coefficients = (
        coefficients[
            :,
            selection_mask,
        ]
    )

    # --------------------------------------------------------
    # Vectorized radial integration
    # --------------------------------------------------------

    integrand = (
        np.abs(
            selected_coefficients
        ) ** 2
        * r[:, np.newaxis]
    )

    modal_power = (
        2.0
        * np.pi
        * np.trapezoid(
            integrand,
            r,
            axis=0,
        )
    )

    modal_power = np.asarray(
        modal_power,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Normalize selected spectrum
    # --------------------------------------------------------

    total_power = float(
        np.sum(
            modal_power
        )
    )

    if (
        not np.isfinite(
            total_power
        )
        or total_power <= 0.0
    ):
        raise ValueError(
            "The calculated OAM spectrum has "
            "non-positive or non-finite total power."
        )

    modal_power = (
        modal_power
        / total_power
    )

    return (
        ell_values,
        modal_power,
    )


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

def plot_oam_spectrum(
    ell_values: np.ndarray,
    modal_power: np.ndarray,
    *,
    reference_power: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    transmitted_charge: int | None = None,
    ell_plot_min: int | None = None,
    ell_plot_max: int | None = None,
    title: str | None = None,
    numerical_label: str = "Split-Step",
    reference_label: str = "Referencia teórica",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot an already calculated OAM spectrum.

    This function is intended only for diagnostic and visualization
    purposes. It does not recalculate or renormalize the spectrum.

    Parameters
    ----------
    ell_values:
        OAM indices.

    modal_power:
        Normalized numerical OAM spectrum.

    reference_power:
        Optional reference spectrum evaluated on the same OAM indices.

    lower, upper:
        Optional lower and upper confidence bounds for the numerical
        spectrum.

    transmitted_charge:
        Optional transmitted OAM charge. A vertical marker is drawn
        at this value.

    ell_plot_min, ell_plot_max:
        Optional visual limits for the OAM interval. These affect only
        the displayed region and do not modify the underlying spectrum.

    title:
        Optional figure title.

    numerical_label:
        Label for the numerical spectrum.

    reference_label:
        Label for the reference spectrum.

    output_path:
        Optional path where the figure is saved.

    show:
        If True, display the figure interactively.
    """

    ell_values = np.asarray(
        ell_values,
        dtype=np.int64,
    )

    modal_power = np.asarray(
        modal_power,
        dtype=np.float64,
    )

    if ell_values.ndim != 1:
        raise ValueError(
            "ell_values must be one-dimensional."
        )

    if modal_power.shape != ell_values.shape:
        raise ValueError(
            "modal_power must have the same shape as ell_values."
        )

    if reference_power is not None:
        reference_power = np.asarray(
            reference_power,
            dtype=np.float64,
        )

        if reference_power.shape != ell_values.shape:
            raise ValueError(
                "reference_power must have the same shape as ell_values."
            )

    if (
        lower is None
        and upper is not None
    ) or (
        lower is not None
        and upper is None
    ):
        raise ValueError(
            "lower and upper must either both be provided or both be None."
        )

    if lower is not None:
        lower = np.asarray(
            lower,
            dtype=np.float64,
        )

        upper = np.asarray(
            upper,
            dtype=np.float64,
        )

        if (
            lower.shape != ell_values.shape
            or upper.shape != ell_values.shape
        ):
            raise ValueError(
                "Confidence bounds must have the same shape as ell_values."
            )

    # --------------------------------------------------------
    # Visual interval
    # --------------------------------------------------------

    if ell_plot_min is None:
        ell_plot_min = int(
            ell_values[0]
        )

    if ell_plot_max is None:
        ell_plot_max = int(
            ell_values[-1]
        )

    if ell_plot_min > ell_plot_max:
        raise ValueError(
            "ell_plot_min must not exceed ell_plot_max."
        )

    mask = (
        (ell_values >= ell_plot_min)
        & (ell_values <= ell_plot_max)
    )

    if not np.any(
        mask
    ):
        raise ValueError(
            "Requested plotting interval does not overlap ell_values."
        )

    ell_plot = (
        ell_values[
            mask
        ]
    )

    numerical_plot = (
        modal_power[
            mask
        ]
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(8.0, 4.8)
    )

    axis.plot(
        ell_plot,
        numerical_plot,
        marker="o",
        linewidth=1.5,
        label=numerical_label,
    )

    if lower is not None:

        axis.fill_between(
            ell_plot,
            lower[
                mask
            ],
            upper[
                mask
            ],
            alpha=0.2,
            label="IC 95 %",
        )

    if reference_power is not None:

        axis.plot(
            ell_plot,
            reference_power[
                mask
            ],
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
        )

    if transmitted_charge is not None:

        axis.axvline(
            transmitted_charge,
            linestyle=":",
            linewidth=1.2,
            label=(
                rf"Modo transmitido "
                rf"$\ell_0={transmitted_charge}$"
            ),
        )

    axis.set_xlabel(
        r"Índice OAM $\ell$"
    )

    axis.set_ylabel(
        r"Potencia modal normalizada $P_\ell$"
    )

    if title is not None:
        axis.set_title(
            title
        )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close(
        figure
    )
