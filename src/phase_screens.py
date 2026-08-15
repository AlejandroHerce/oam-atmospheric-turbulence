"""
Phase-screen generation utilities.

This module contains the numerical core used to generate atmospheric
turbulence phase screens from a prescribed phase power spectral density.
"""

import numpy as np
from numpy.typing import NDArray


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ============================================================
# Hermitian Gaussian noise
# ============================================================

def _hermitian_complex_noise(
    n: int,
    rng: np.random.Generator,
) -> ComplexArray:
    """
    Generate complex Gaussian noise with Hermitian symmetry.

    The Hermitian symmetry guarantees that the inverse Fourier
    transform is real-valued.
    """

    noise = np.zeros(
        (n, n),
        dtype=np.complex128,
    )

    filled = np.zeros(
        (n, n),
        dtype=bool,
    )

    for iy in range(n):
        for ix in range(n):
            if filled[iy, ix]:
                continue

            jy = (-iy) % n
            jx = (-ix) % n

            if (iy == jy) and (ix == jx):
                noise[iy, ix] = rng.normal()
                filled[iy, ix] = True

            else:
                z = (
                    rng.normal()
                    + 1j * rng.normal()
                ) / np.sqrt(2.0)

                noise[iy, ix] = z
                noise[jy, jx] = np.conj(z)

                filled[iy, ix] = True
                filled[jy, jx] = True

    return noise


# ============================================================
# Kolmogorov PSD
# ============================================================

def kolmogorov_psd(
    kappa: RealArray,
    r0: float,
) -> RealArray:
    """
    Calculate the Kolmogorov phase power spectral density.

    Parameters
    ----------
    kappa:
        Spatial angular-frequency magnitude [rad/m].

    r0:
        Fried parameter [m].

    Returns
    -------
    RealArray
        Kolmogorov phase PSD.

    Notes
    -----
    The implementation uses

        Phi_theta(kappa)
        = 0.49 r0^(-5/3) kappa^(-11/3).

    The zero-frequency component is explicitly set to zero.
    """

    if r0 <= 0:
        raise ValueError(
            "r0 must be positive."
        )

    psd = np.zeros_like(
        kappa,
        dtype=np.float64,
    )

    nonzero = kappa > 0.0

    psd[nonzero] = (
        0.49
        * r0 ** (-5.0 / 3.0)
        * kappa[nonzero] ** (-11.0 / 3.0)
    )

    return psd


# ============================================================
# Generic phase-screen generator
# ============================================================

def generate_phase_screen_from_psd(
    psd: RealArray,
    delta: float,
    rng: np.random.Generator,
    remove_piston: bool = True,
) -> RealArray:
    """
    Generate a real-valued phase screen from a prescribed PSD.

    Parameters
    ----------
    psd:
        Two-dimensional phase PSD sampled on the FFT spatial-frequency
        grid.

    delta:
        Spatial sampling interval [m].

    rng:
        NumPy random-number generator.

    remove_piston:
        If True, subtract the mean phase from the resulting screen.

    Returns
    -------
    RealArray
        Real-valued atmospheric phase screen [rad].

    Notes
    -----
    The spectral sampling interval is

        Delta kappa = 2*pi / (N*delta),

    and the Fourier coefficients are generated according to

        c(kappa) =
            g(kappa) sqrt(Phi_theta(kappa) Delta kappa^2).

    The inverse FFT normalization follows the convention used in
    the original Chapter 3 implementation.
    """

    if psd.ndim != 2:
        raise ValueError(
            "psd must be a two-dimensional array."
        )

    n_y, n_x = psd.shape

    if n_x != n_y:
        raise ValueError(
            "psd must be square."
        )

    if n_x <= 1:
        raise ValueError(
            "psd must contain more than one sample per dimension."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    if np.any(psd < 0):
        raise ValueError(
            "psd must not contain negative values."
        )

    if not np.all(np.isfinite(psd)):
        raise ValueError(
            "psd must contain only finite values."
        )

    n = n_x

    dk = (
        2.0
        * np.pi
        / (n * delta)
    )

    noise = _hermitian_complex_noise(
        n=n,
        rng=rng,
    )

    coefficients = (
        noise
        * np.sqrt(
            psd
            * dk**2
        )
    )

    phase = (
        np.fft.ifft2(
            coefficients
        ).real
        * n**2
    )

    if remove_piston:
        phase -= np.mean(phase)

    return phase


# ============================================================
# Kolmogorov phase screen
# ============================================================

def kolmogorov_phase_screen(
    n: int,
    delta: float,
    r0: float,
    rng: np.random.Generator,
    remove_piston: bool = True,
) -> RealArray:
    """
    Generate a Kolmogorov turbulence phase screen.

    Parameters
    ----------
    n:
        Number of samples along each spatial dimension.

    delta:
        Spatial sampling interval [m].

    r0:
        Fried parameter [m].

    rng:
        NumPy random-number generator.

    remove_piston:
        If True, subtract the mean phase.

    Returns
    -------
    RealArray
        Kolmogorov phase screen [rad].
    """

    if n <= 1:
        raise ValueError(
            "n must be greater than one."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    if r0 <= 0:
        raise ValueError(
            "r0 must be positive."
        )

    k = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            n,
            d=delta,
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

    psd = kolmogorov_psd(
        kappa=kappa,
        r0=r0,
    )

    return generate_phase_screen_from_psd(
        psd=psd,
        delta=delta,
        rng=rng,
        remove_piston=remove_piston,
    )
