import numpy as np

from src.structure_functions import (
    kolmogorov_structure_function,
    modified_von_karman_structure_function,
    structure_function_xy,
    von_karman_structure_function,
)

def direct_structure_function_xy(
    phase: np.ndarray,
    delta: float,
    max_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct reference implementation of the phase structure function.

    This intentionally uses the explicit displacement loop because
    its mathematical meaning is transparent. It is used only as a
    reference for testing the FFT implementation.
    """

    rho = (
        np.arange(
            1,
            max_shift,
        )
        * delta
    )

    structure = np.zeros_like(
        rho,
        dtype=np.float64,
    )

    for index, shift in enumerate(
        range(1, max_shift)
    ):
        difference_x = (
            phase[:, shift:]
            - phase[:, :-shift]
        )

        difference_y = (
            phase[shift:, :]
            - phase[:-shift, :]
        )

        structure[index] = (
            0.5
            * (
                np.mean(
                    difference_x**2
                )
                + np.mean(
                    difference_y**2
                )
            )
        )

    return (
        rho,
        structure,
    )


def test_fft_structure_function_matches_direct_method():
    """
    The FFT implementation must reproduce the direct definition.
    """

    rng = np.random.default_rng(
        12345
    )

    n = 64
    delta = 0.4 / n
    max_shift = 25

    phase = rng.normal(
        size=(n, n)
    )

    (
        rho_direct,
        structure_direct,
    ) = direct_structure_function_xy(
        phase=phase,
        delta=delta,
        max_shift=max_shift,
    )

    (
        rho_fft,
        structure_fft,
    ) = structure_function_xy(
        phase=phase,
        delta=delta,
        max_shift=max_shift,
    )

    assert np.array_equal(
        rho_fft,
        rho_direct,
    )

    assert np.allclose(
        structure_fft,
        structure_direct,
        rtol=1e-12,
        atol=1e-12,
    )


def test_constant_phase_has_zero_structure_function():
    """
    A spatially constant phase has zero structure function.
    """

    phase = np.full(
        (64, 64),
        3.5,
    )

    _, structure = structure_function_xy(
        phase=phase,
        delta=0.4 / 64,
        max_shift=20,
    )

    assert np.allclose(
        structure,
        0.0,
        atol=1e-12,
    )


def test_kolmogorov_theoretical_structure_function():
    """
    Basic properties of the Kolmogorov analytical expression.
    """

    rho = np.array(
        [
            0.0,
            0.001,
            0.01,
            0.1,
        ]
    )

    structure = (
        kolmogorov_structure_function(
            rho=rho,
            r0=6.7e-3,
        )
    )

    assert structure[0] == 0.0

    assert np.all(
        structure >= 0.0
    )

    assert np.all(
        np.diff(structure) >= 0.0
    )

def test_von_karman_structure_function():
    """
    Basic properties of the theoretical von Kármán structure function.
    """

    rho = np.linspace(
        0.0,
        0.1,
        50,
    )

    structure = von_karman_structure_function(
        rho=rho,
        r0=6.7e-3,
        outer_scale=10.0,
    )

    assert structure.shape == rho.shape
    assert np.all(np.isfinite(structure))
    assert np.all(structure >= 0.0)

    assert structure[0] == 0.0

    assert np.all(
        np.diff(structure) >= 0.0
    )


def test_modified_von_karman_structure_function():
    """
    Basic properties of the theoretical modified von Kármán
    structure function.
    """

    rho = np.linspace(
        0.0,
        0.1,
        25,
    )

    structure = modified_von_karman_structure_function(
        rho=rho,
        r0=6.7e-3,
        outer_scale=10.0,
        inner_scale=5.0e-3,
    )

    assert structure.shape == rho.shape
    assert np.all(np.isfinite(structure))
    assert np.all(structure >= 0.0)

    assert np.isclose(
        structure[0],
        0.0,
        atol=1e-12,
    )

    assert np.all(
        np.diff(structure) >= 0.0
    )


def test_modified_von_karman_is_below_von_karman():
    """
    The inner-scale cutoff suppresses high-frequency phase
    fluctuations relative to the standard von Kármán model.
    """

    rho = np.linspace(
        0.001,
        0.1,
        25,
    )

    von_karman = von_karman_structure_function(
        rho=rho,
        r0=6.7e-3,
        outer_scale=10.0,
    )

    modified = modified_von_karman_structure_function(
        rho=rho,
        r0=6.7e-3,
        outer_scale=10.0,
        inner_scale=5.0e-3,
    )

    assert np.all(
        modified <= von_karman
    )
