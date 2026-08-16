import numpy as np

from src.sampling import (
    analyze_phase_sampling,
    calculate_relative_psd,
    radial_spectral_profile,
)


def test_constant_phase_has_zero_sampling_content():
    """
    A constant phase field has no gradients, neighboring differences,
    or non-zero-frequency spectral content.
    """

    n = 64
    delta = 0.4 / n

    phase = np.ones(
        (n, n),
        dtype=np.float64,
    )

    (
        diagnostics,
        number_above_pi,
        number_of_pairs,
    ) = analyze_phase_sampling(
        phase=phase,
        delta=delta,
    )

    assert diagnostics.rms_neighbor_difference == 0.0
    assert diagnostics.maximum_neighbor_difference == 0.0
    assert diagnostics.fraction_above_pi == 0.0

    assert diagnostics.maximum_gradient == 0.0
    assert diagnostics.maximum_gradient_phase_change == 0.0

    assert diagnostics.nyquist_power_fraction == 0.0

    assert number_above_pi == 0
    assert number_of_pairs > 0


def test_relative_psd_is_nonnegative():
    """
    Spectral power must be real and non-negative.
    """

    rng = np.random.default_rng(
        12345
    )

    field = rng.normal(
        size=(64, 64)
    )

    psd = calculate_relative_psd(
        field
    )

    assert psd.shape == field.shape
    assert np.isrealobj(psd)
    assert np.all(
        np.isfinite(psd)
    )
    assert np.all(
        psd >= 0.0
    )


def test_radial_spectral_profile():
    """
    The radial profile must contain finite and non-negative values.
    """

    rng = np.random.default_rng(
        12345
    )

    field = rng.normal(
        size=(64, 64)
    )

    psd = calculate_relative_psd(
        field
    )

    kappa, profile = (
        radial_spectral_profile(
            spectral_power=psd,
            delta=0.4 / 64,
        )
    )

    assert kappa.ndim == 1
    assert profile.ndim == 1

    assert kappa.shape == profile.shape

    assert np.all(
        np.isfinite(kappa)
    )

    assert np.all(
        np.isfinite(profile)
    )

    assert np.all(
        kappa >= 0.0
    )

    assert np.all(
        profile >= 0.0
    )

    assert kappa[0] == 0.0
