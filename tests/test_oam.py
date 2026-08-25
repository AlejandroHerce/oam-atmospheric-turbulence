import numpy as np

from experiments.chapter_4.ensemble_size_convergence import (
    create_input_beam,
)

from configs.chapter_2 import (
    L_WINDOW,
    N_GRID,
)


def test_create_input_beam():
    grid, field = create_input_beam()

    assert grid.x.shape == (N_GRID,)
    assert grid.y.shape == (N_GRID,)

    assert field.shape == (
        N_GRID,
        N_GRID,
    )

    assert np.all(
        np.isfinite(field)
    )

    assert np.iscomplexobj(field)

    assert np.max(
        np.abs(field)
    ) > 0.0

    dx = L_WINDOW / N_GRID

    assert np.isclose(
        grid.dx,
        dx,
    )

def calculate_oam_spectrum_reference(
    field,
    grid,
    ell_min,
    ell_max,
    radial_samples=256,
    azimuthal_samples=720,
    maximum_radius=None,
):
    """
    Reference implementation using an explicit loop over ell.

    Used only to verify the vectorized implementation.
    """

    from src.oam import (
        interpolate_to_polar_grid,
    )

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

    for index, ell in enumerate(
        ell_values
    ):

        position = np.where(
            available_ell == ell
        )[0]

        radial_coefficient = (
            coefficients[
                :,
                position[0],
            ]
        )

        integrand = (
            np.abs(
                radial_coefficient
            ) ** 2
            * r
        )

        modal_power[index] = (
            2.0
            * np.pi
            * np.trapezoid(
                integrand,
                r,
            )
        )

    modal_power /= np.sum(
        modal_power
    )

    return (
        ell_values,
        modal_power,
    )


def test_vectorized_oam_spectrum_matches_reference():
    """
    The vectorized OAM implementation must reproduce the
    original loop-based calculation.
    """

    from configs.chapter_2 import (
        L_WINDOW,
        N_GRID,
        W0_LG,
    )

    from src.beams import (
        laguerre_gaussian_beam,
    )

    from src.grids import (
        create_grid,
    )

    from src.oam import (
        calculate_oam_spectrum,
    )

    grid = create_grid(
        n=N_GRID,
        window_size=L_WINDOW,
    )

    field = laguerre_gaussian_beam(
        grid=grid,
        w0=W0_LG,
        charge=3,
    )

    ell_min = -30
    ell_max = 30

    (
        ell_reference,
        power_reference,
    ) = calculate_oam_spectrum_reference(
        field=field,
        grid=grid,
        ell_min=ell_min,
        ell_max=ell_max,
    )

    (
        ell_vectorized,
        power_vectorized,
    ) = calculate_oam_spectrum(
        field=field,
        grid=grid,
        ell_min=ell_min,
        ell_max=ell_max,
    )

    np.testing.assert_array_equal(
        ell_vectorized,
        ell_reference,
    )

    np.testing.assert_allclose(
        power_vectorized,
        power_reference,
        rtol=1e-13,
        atol=1e-15,
    )
