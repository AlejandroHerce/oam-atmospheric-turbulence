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
# Spatial-frequency grid
# ============================================================

def spatial_frequency_grid(
    n: int,
    delta: float,
) -> tuple[RealArray, RealArray, RealArray]:
    """
    Construct the FFT spatial angular-frequency grid.

    Parameters
    ----------
    n:
        Number of samples along each spatial dimension.

    delta:
        Spatial sampling interval [m].

    Returns
    -------
    kx, ky, kappa:
        Cartesian and radial spatial angular frequencies [rad/m].
    """
    if n <= 1:
        raise ValueError(
            "n must be greater than one."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
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

    return kx, ky, kappa

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
# von Kármán PSD
# ============================================================

def von_karman_psd(
    kappa: RealArray,
    r0: float,
    outer_scale: float,
) -> RealArray:
    """
    Calculate the von Kármán phase power spectral density.

    Parameters
    ----------
    kappa:
        Spatial angular-frequency magnitude [rad/m].

    r0:
        Fried parameter [m].

    outer_scale:
        Turbulence outer scale L0 [m].

    Returns
    -------
    RealArray
        von Kármán phase PSD.

    Notes
    -----
    The implementation uses

        Phi_theta(kappa)
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

    kappa0 = (
        2.0
        * np.pi
        / outer_scale
    )

    return (
        0.49
        * r0 ** (-5.0 / 3.0)
        * (
            kappa**2
            + kappa0**2
        ) ** (-11.0 / 6.0)
    )


# ============================================================
# Modified von Kármán PSD
# ============================================================

def modified_von_karman_psd(
    kappa: RealArray,
    r0: float,
    outer_scale: float,
    inner_scale: float,
) -> RealArray:
    """
    Calculate the modified von Kármán phase power spectral density.

    Parameters
    ----------
    kappa:
        Spatial angular-frequency magnitude [rad/m].

    r0:
        Fried parameter [m].

    outer_scale:
        Turbulence outer scale L0 [m].

    inner_scale:
        Turbulence inner scale l0 [m].

    Returns
    -------
    RealArray
        Modified von Kármán phase PSD.

    Notes
    -----
    The implementation uses

        Phi_theta(kappa)
        =
        0.49 r0^(-5/3)
        exp[-(kappa/kappa_m)^2]
        (kappa^2 + kappa0^2)^(-11/6),

    where

        kappa0 = 2*pi/L0

    and

        kappa_m = 5.92/l0.
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

    kappa0 = (
        2.0
        * np.pi
        / outer_scale
    )

    kappa_m = (
        5.92
        / inner_scale
    )

    return (
        0.49
        * r0 ** (-5.0 / 3.0)
        * np.exp(
            -(kappa / kappa_m) ** 2
        )
        * (
            kappa**2
            + kappa0**2
        ) ** (-11.0 / 6.0)
    )

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

    # Construct the FFT spatial-frequency grid.
    # Only the radial magnitude kappa is required because
    # the Kolmogorov PSD is isotropic.
    _kx, _ky, kappa = spatial_frequency_grid(
        n=n,
        delta=delta,
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

# ============================================================
# von Kármán phase screen
# ============================================================

def von_karman_phase_screen(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    rng: np.random.Generator,
    remove_piston: bool = True,
) -> RealArray:
    """
    Generate a von Kármán turbulence phase screen.
    """

    if n <= 1:
        raise ValueError(
            "n must be greater than one."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    _kx, _ky, kappa = spatial_frequency_grid(
        n=n,
        delta=delta,
    )

    psd = von_karman_psd(
        kappa=kappa,
        r0=r0,
        outer_scale=outer_scale,
    )

    return generate_phase_screen_from_psd(
        psd=psd,
        delta=delta,
        rng=rng,
        remove_piston=remove_piston,
    )


# ============================================================
# Modified von Kármán phase screen
# ============================================================

def modified_von_karman_phase_screen(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    inner_scale: float,
    rng: np.random.Generator,
    remove_piston: bool = True,
) -> RealArray:
    """
    Generate a modified von Kármán turbulence phase screen.
    """

    if n <= 1:
        raise ValueError(
            "n must be greater than one."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    _kx, _ky, kappa = spatial_frequency_grid(
        n=n,
        delta=delta,
    )

    psd = modified_von_karman_psd(
        kappa=kappa,
        r0=r0,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )

    return generate_phase_screen_from_psd(
        psd=psd,
        delta=delta,
        rng=rng,
        remove_piston=remove_piston,
    )


# ============================================================
# Single subharmonic level
# ============================================================

def generate_subharmonic_level(
    n: int,
    delta: float,
    level: int,
    psd_function,
    psd_kwargs: dict,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate one low-spatial-frequency subharmonic level.

    Each level samples the 3x3 subharmonic grid at

        Delta kappa_b = Delta kappa / 3^b,

    where

        Delta kappa = 2*pi/L
        L = n*delta.

    Parameters
    ----------
    n:
        Number of samples along each spatial dimension.

    delta:
        Spatial sampling interval [m].

    level:
        Subharmonic level b. Must satisfy b >= 1.

    psd_function:
        Function used to evaluate the phase PSD.

    psd_kwargs:
        Keyword arguments passed to psd_function.

    rng:
        NumPy random-number generator.

    remove_piston:
        If True, subtract the mean phase of this level.

    Returns
    -------
    RealArray
        Phase contribution associated with the selected
        subharmonic level [rad].
    """

    if n <= 1:
        raise ValueError(
            "n must be greater than one."
        )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    if level < 1:
        raise ValueError(
            "level must be greater than or equal to one."
        )

    window_size = (
        n * delta
    )

    fundamental_dk = (
        2.0
        * np.pi
        / window_size
    )

    dk_level = (
        fundamental_dk
        / (3.0 ** level)
    )

    coordinates = (
        np.arange(n)
        * delta
    )

    x, y = np.meshgrid(
        coordinates,
        coordinates,
        indexing="xy",
    )

    phase_level = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    independent_modes = (
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    )

    for mode_x, mode_y in independent_modes:
        kx = (
            mode_x
            * dk_level
        )

        ky = (
            mode_y
            * dk_level
        )

        kappa = np.hypot(
            kx,
            ky,
        )

        psd_value = psd_function(
            np.asarray(kappa),
            **psd_kwargs,
        )

        psd_value = float(
            psd_value
        )

        gaussian_coefficient = (
            rng.normal()
            + 1j * rng.normal()
        ) / np.sqrt(2.0)

        amplitude = (
            gaussian_coefficient
            * np.sqrt(psd_value)
            * dk_level
        )

        phase_argument = (
            kx * x
            + ky * y
        )

        phase_level += (
            2.0
            * np.real(
                amplitude
                * np.exp(
                    1j * phase_argument
                )
            )
        )

    if remove_piston:
        phase_level -= np.mean(
            phase_level
        )

    return phase_level


def kolmogorov_subharmonic_level(
    n: int,
    delta: float,
    r0: float,
    level: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate one Kolmogorov subharmonic level.
    """

    return generate_subharmonic_level(
        n=n,
        delta=delta,
        level=level,
        psd_function=kolmogorov_psd,
        psd_kwargs={
            "r0": r0,
        },
        rng=rng,
        remove_piston=remove_piston,
    )

def von_karman_subharmonic_level(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    level: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate one von Kármán subharmonic level.
    """

    return generate_subharmonic_level(
        n=n,
        delta=delta,
        level=level,
        psd_function=von_karman_psd,
        psd_kwargs={
            "r0": r0,
            "outer_scale": outer_scale,
        },
        rng=rng,
        remove_piston=remove_piston,
    )


def modified_von_karman_subharmonic_level(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    inner_scale: float,
    level: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate one modified von Kármán subharmonic level.
    """

    return generate_subharmonic_level(
        n=n,
        delta=delta,
        level=level,
        psd_function=modified_von_karman_psd,
        psd_kwargs={
            "r0": r0,
            "outer_scale": outer_scale,
            "inner_scale": inner_scale,
        },
        rng=rng,
        remove_piston=remove_piston,
    )


def generate_subharmonic_component(
    n: int,
    delta: float,
    n_subharmonics: int,
    psd_function,
    psd_kwargs: dict,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate the low-spatial-frequency subharmonic contribution
    associated with a prescribed phase PSD.
    """

    if n <= 1:
        raise ValueError("n must be greater than one.")

    if delta <= 0:
        raise ValueError("delta must be positive.")

    if n_subharmonics < 0:
        raise ValueError(
            "n_subharmonics must be non-negative."
        )

    window_size = n * delta

    fundamental_dk = (
        2.0
        * np.pi
        / window_size
    )

    coordinates = np.arange(n) * delta

    x, y = np.meshgrid(
        coordinates,
        coordinates,
        indexing="xy",
    )

    phi_sub = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    # Independent representatives of the 3x3 subharmonic grid.
    # Their conjugate partners are included through the real
    # reconstruction below.
    independent_modes = (
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    )

    for level in range(
        1,
        n_subharmonics + 1,
    ):
        dk_level = (
            fundamental_dk
            / (3.0 ** level)
        )

        for mode_x, mode_y in independent_modes:
            kx = mode_x * dk_level
            ky = mode_y * dk_level

            kappa = np.hypot(
                kx,
                ky,
            )

            psd_value = psd_function(
                np.asarray(kappa),
                **psd_kwargs,
            )

            psd_value = float(psd_value)

            gaussian_coefficient = (
                rng.normal()
                + 1j * rng.normal()
            ) / np.sqrt(2.0)

            amplitude = (
                gaussian_coefficient
                * np.sqrt(psd_value)
                * dk_level
            )

            phase_argument = (
                kx * x
                + ky * y
            )

            phi_sub += (
                2.0
                * np.real(
                    amplitude
                    * np.exp(
                        1j * phase_argument
                    )
                )
            )

    if remove_piston:
        phi_sub -= np.mean(phi_sub)

    return phi_sub


def kolmogorov_subharmonics(
    n: int,
    delta: float,
    r0: float,
    n_subharmonics: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate the Kolmogorov subharmonic contribution.
    """

    return generate_subharmonic_component(
        n=n,
        delta=delta,
        n_subharmonics=n_subharmonics,
        psd_function=kolmogorov_psd,
        psd_kwargs={
            "r0": r0,
        },
        rng=rng,
        remove_piston=remove_piston,
    )


# ============================================================
# Kolmogorov phase screen with subharmonics
# ============================================================

def kolmogorov_phase_screen_with_subharmonics(
    n: int,
    delta: float,
    r0: float,
    n_subharmonics: int,
    rng: np.random.Generator,
    remove_piston: bool = True,
) -> RealArray:
    """
    Generate a Kolmogorov phase screen including low-frequency
    subharmonic compensation.

    The high-frequency FFT component and the subharmonic
    component are generated sequentially using the same random
    number generator, preserving the convention used in the
    original Chapter 3 implementation.

    Parameters
    ----------
    n:
        Number of samples along each spatial dimension.

    delta:
        Spatial sampling interval [m].

    r0:
        Fried parameter [m].

    n_subharmonics:
        Number of subharmonic levels.

    rng:
        NumPy random-number generator.

    remove_piston:
        If True, remove the mean phase from the final screen.

    Returns
    -------
    RealArray
        Kolmogorov phase screen including subharmonic correction [rad].
    """

    if n_subharmonics < 0:
        raise ValueError(
            "n_subharmonics must be non-negative."
        )

    # FFT-based component.
    phase_fft = kolmogorov_phase_screen(
        n=n,
        delta=delta,
        r0=r0,
        rng=rng,
        remove_piston=False,
    )

    # Low-frequency subharmonic contribution.
    phase_subharmonics = kolmogorov_subharmonics(
        n=n,
        delta=delta,
        r0=r0,
        n_subharmonics=n_subharmonics,
        rng=rng,
        remove_piston=False,
    )

    phase = (
        phase_fft
        + phase_subharmonics
    )

    if remove_piston:
        phase -= np.mean(phase)

    return phase

def von_karman_subharmonics(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    n_subharmonics: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate the von Kármán subharmonic contribution.
    """

    return generate_subharmonic_component(
        n=n,
        delta=delta,
        n_subharmonics=n_subharmonics,
        psd_function=von_karman_psd,
        psd_kwargs={
            "r0": r0,
            "outer_scale": outer_scale,
        },
        rng=rng,
        remove_piston=remove_piston,
    )


def modified_von_karman_subharmonics(
    n: int,
    delta: float,
    r0: float,
    outer_scale: float,
    inner_scale: float,
    n_subharmonics: int,
    rng: np.random.Generator,
    remove_piston: bool = False,
) -> RealArray:
    """
    Generate the modified von Kármán subharmonic contribution.
    """

    return generate_subharmonic_component(
        n=n,
        delta=delta,
        n_subharmonics=n_subharmonics,
        psd_function=modified_von_karman_psd,
        psd_kwargs={
            "r0": r0,
            "outer_scale": outer_scale,
            "inner_scale": inner_scale,
        },
        rng=rng,
        remove_piston=remove_piston,
    )
