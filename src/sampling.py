"""
Spatial and spectral sampling diagnostics.

This module contains reusable numerical diagnostics used to evaluate
whether a sampled two-dimensional field is adequately represented by
the computational grid.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


RealArray = NDArray[np.float64]


# ============================================================
# Sampling result
# ============================================================

@dataclass(frozen=True)
class SamplingDiagnostics:
    """
    Numerical sampling diagnostics for a two-dimensional phase field.
    """

    rms_neighbor_difference: float
    maximum_neighbor_difference: float
    percentile_neighbor_difference: float
    fraction_above_pi: float

    maximum_gradient: float
    percentile_gradient: float
    maximum_gradient_phase_change: float
    percentile_gradient_phase_change: float

    nyquist_power_fraction: float


# ============================================================
# Phase-screen sampling diagnostics
# ============================================================

def analyze_phase_sampling(
    phase: RealArray,
    delta: float,
    spectral_guard_fraction: float = 0.8,
    percentile: float = 99.9,
) -> tuple[SamplingDiagnostics, int, int]:
    """
    Evaluate spatial and spectral sampling of a phase field.

    Parameters
    ----------
    phase:
        Two-dimensional phase field [rad].

    delta:
        Spatial sampling interval [m].

    spectral_guard_fraction:
        Fraction of the Cartesian Nyquist frequency above which
        spectral power is considered close to the sampling limit.

    percentile:
        Percentile used as a robust high-value spatial metric.

    Returns
    -------
    diagnostics:
        Sampling metrics.

    number_above_pi:
        Number of neighboring-pixel phase differences larger than pi.

    number_of_neighbor_pairs:
        Total number of neighboring-pixel differences evaluated.
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

    if not 0.0 < spectral_guard_fraction <= 1.0:
        raise ValueError(
            "spectral_guard_fraction must be in (0, 1]."
        )

    if not 0.0 < percentile <= 100.0:
        raise ValueError(
            "percentile must be in (0, 100]."
        )

    # --------------------------------------------------------
    # Neighboring-pixel phase differences
    # --------------------------------------------------------

    delta_phase_x = np.diff(
        phase,
        axis=1,
    )

    delta_phase_y = np.diff(
        phase,
        axis=0,
    )

    signed_differences = np.concatenate(
        (
            delta_phase_x.ravel(),
            delta_phase_y.ravel(),
        )
    )

    absolute_differences = np.abs(
        signed_differences
    )

    number_above_pi = int(
        np.count_nonzero(
            absolute_differences > np.pi
        )
    )

    number_of_neighbor_pairs = int(
        absolute_differences.size
    )

    rms_neighbor_difference = float(
        np.sqrt(
            np.mean(
                signed_differences**2
            )
        )
    )

    maximum_neighbor_difference = float(
        np.max(
            absolute_differences
        )
    )

    percentile_neighbor_difference = float(
        np.percentile(
            absolute_differences,
            percentile,
        )
    )

    fraction_above_pi = (
        number_above_pi
        / number_of_neighbor_pairs
    )

    # --------------------------------------------------------
    # Spatial phase gradients
    # --------------------------------------------------------

    gradient_y, gradient_x = np.gradient(
        phase,
        delta,
        delta,
    )

    gradient_magnitude = np.hypot(
        gradient_x,
        gradient_y,
    )

    gradient_phase_change = (
        delta
        * gradient_magnitude
    )

    maximum_gradient = float(
        np.max(
            gradient_magnitude
        )
    )

    percentile_gradient = float(
        np.percentile(
            gradient_magnitude,
            percentile,
        )
    )

    maximum_gradient_phase_change = float(
        np.max(
            gradient_phase_change
        )
    )

    percentile_gradient_phase_change = float(
        np.percentile(
            gradient_phase_change,
            percentile,
        )
    )

    # --------------------------------------------------------
    # Spectral power near the Nyquist limit
    # --------------------------------------------------------

    phase_without_piston = (
        phase
        - np.mean(phase)
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            phase_without_piston
        )
    )

    spectral_power = (
        np.abs(spectrum) ** 2
    )

    k = (
        2.0
        * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(
                nx,
                d=delta,
            )
        )
    )

    kx, ky = np.meshgrid(
        k,
        k,
        indexing="xy",
    )

    kappa = np.hypot(
        kx,
        ky,
    )

    kappa_nyquist = (
        np.pi
        / delta
    )

    guard_limit = (
        spectral_guard_fraction
        * kappa_nyquist
    )

    valid_region = (
        kappa > 0.0
    )

    high_frequency_region = (
        valid_region
        & (kappa >= guard_limit)
    )

    total_spectral_power = float(
        np.sum(
            spectral_power[
                valid_region
            ]
        )
    )

    high_frequency_power = float(
        np.sum(
            spectral_power[
                high_frequency_region
            ]
        )
    )

    if total_spectral_power > 0.0:
        nyquist_power_fraction = (
            high_frequency_power
            / total_spectral_power
        )
    else:
        nyquist_power_fraction = 0.0

    diagnostics = SamplingDiagnostics(
        rms_neighbor_difference=rms_neighbor_difference,
        maximum_neighbor_difference=maximum_neighbor_difference,
        percentile_neighbor_difference=percentile_neighbor_difference,
        fraction_above_pi=fraction_above_pi,
        maximum_gradient=maximum_gradient,
        percentile_gradient=percentile_gradient,
        maximum_gradient_phase_change=maximum_gradient_phase_change,
        percentile_gradient_phase_change=percentile_gradient_phase_change,
        nyquist_power_fraction=nyquist_power_fraction,
    )

    return (
        diagnostics,
        number_above_pi,
        number_of_neighbor_pairs,
    )


# ============================================================
# Relative discrete PSD
# ============================================================

def calculate_relative_psd(
    field: RealArray,
    remove_mean: bool = True,
) -> RealArray:
    """
    Calculate the relative discrete spatial power spectrum.

    Absolute PSD normalization is not required when only spectral
    power fractions or normalized radial profiles are needed.
    """

    if field.ndim != 2:
        raise ValueError(
            "field must be a two-dimensional array."
        )

    working_field = field.astype(
        np.float64,
        copy=True,
    )

    if remove_mean:
        working_field -= np.mean(
            working_field
        )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            working_field
        )
    )

    return (
        np.abs(spectrum) ** 2
    ).astype(np.float64)


# ============================================================
# Radial spectral profile
# ============================================================

def radial_spectral_profile(
    spectral_power: RealArray,
    delta: float,
) -> tuple[RealArray, RealArray]:
    """
    Calculate the azimuthally averaged radial spectral profile.

    Parameters
    ----------
    spectral_power:
        Two-dimensional centered spectral-power distribution.

    delta:
        Spatial sampling interval [m].

    Returns
    -------
    kappa:
        Radial spatial angular-frequency axis [rad/m].

    radial_average:
        Azimuthally averaged spectral power.
    """

    if spectral_power.ndim != 2:
        raise ValueError(
            "spectral_power must be two-dimensional."
        )

    ny, nx = spectral_power.shape

    if nx != ny:
        raise ValueError(
            "spectral_power must be square."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    n = nx

    k = (
        2.0
        * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(
                n,
                d=delta,
            )
        )
    )

    kx, ky = np.meshgrid(
        k,
        k,
        indexing="xy",
    )

    kappa = np.hypot(
        kx,
        ky,
    )

    dk = float(
        abs(k[1] - k[0])
    )

    bins = np.floor(
        kappa / dk
    ).astype(int)

    number_of_bins = (
        int(bins.max())
        + 1
    )

    radial_sum = np.bincount(
        bins.ravel(),
        weights=spectral_power.ravel(),
        minlength=number_of_bins,
    )

    radial_count = np.bincount(
        bins.ravel(),
        minlength=number_of_bins,
    )

    radial_average = np.divide(
        radial_sum,
        radial_count,
        out=np.zeros_like(
            radial_sum,
            dtype=np.float64,
        ),
        where=radial_count > 0,
    )

    kappa_axis = (
        np.arange(
            number_of_bins,
            dtype=np.float64,
        )
        * dk
    )

    return (
        kappa_axis,
        radial_average,
    )
