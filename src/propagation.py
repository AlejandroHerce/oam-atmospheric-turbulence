import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


def angular_spectrum_propagation(
    field: ComplexArray,
    wavelength: float,
    distance: float,
    dx: float,
) -> ComplexArray:
    """
    Propagate a 2D optical field using the Angular Spectrum Method (ASM).

    The propagation is performed according to

        U(x, y, z)
            = F^{-1}{
                F[U(x, y, 0)] H(fx, fy; z)
              },

    where

        H(fx, fy; z) = exp(i kz z)

    and

        kz = sqrt(k^2 - kx^2 - ky^2).

    Evanescent components are represented using a complex-valued kz.

    Parameters
    ----------
    field:
        Complex input field U(x, y, 0).

    wavelength:
        Optical wavelength [m].

    distance:
        Propagation distance [m].

    dx:
        Spatial sampling interval [m].

    Returns
    -------
    ComplexArray
        Propagated complex field U(x, y, z).
    """

    if field.ndim != 2:
        raise ValueError(
            "The input field must be a 2D array."
        )

    ny, nx = field.shape

    if nx != ny:
        raise ValueError(
            "The computational grid must be square."
        )

    if wavelength <= 0:
        raise ValueError(
            "Wavelength must be positive."
        )

    if dx <= 0:
        raise ValueError(
            "Spatial sampling interval must be positive."
        )

    # Spatial frequencies associated with the FFT ordering.
    fx = np.fft.fftfreq(
        nx,
        d=dx,
    )

    fy = np.fft.fftfreq(
        ny,
        d=dx,
    )

    FX, FY = np.meshgrid(
        fx,
        fy,
        indexing="xy",
    )

    # Wave number.
    k = 2.0 * np.pi / wavelength

    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY

    # Longitudinal wave-number component.
    kz_squared = (
        k**2
        - kx**2
        - ky**2
    )

    kz = np.sqrt(
        kz_squared.astype(
            np.complex128
        )
    )

    # Transfer function.
    transfer_function = np.exp(
        1j * kz * distance
    )

    # Forward propagation in Fourier space.
    input_spectrum = np.fft.fft2(
        field
    )

    propagated_spectrum = (
        input_spectrum
        * transfer_function
    )

    # Return to real space.
    propagated_field = np.fft.ifft2(
        propagated_spectrum
    )

    return propagated_field.astype(
        np.complex128
    )
