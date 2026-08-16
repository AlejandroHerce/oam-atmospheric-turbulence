import numpy as np

from src.phase_screens import (
    kolmogorov_phase_screen,
    kolmogorov_psd,
    modified_von_karman_phase_screen,
    modified_von_karman_psd,
    spatial_frequency_grid,
    von_karman_phase_screen,
    von_karman_psd,
    kolmogorov_subharmonics,
    von_karman_subharmonics,
    kolmogorov_phase_screen_with_subharmonics,
    kolmogorov_subharmonic_level,
    modified_von_karman_subharmonics,
    von_karman_subharmonic_level,
    modified_von_karman_subharmonic_level,
)


def test_spatial_frequency_grid():
    """
    Check the basic properties of the FFT spatial-frequency grid.
    """
    n = 512
    window_size = 0.4
    delta = window_size / n

    kx, ky, kappa = spatial_frequency_grid(
        n=n,
        delta=delta,
    )

    # All arrays must have the expected shape.
    assert kx.shape == (n, n)
    assert ky.shape == (n, n)
    assert kappa.shape == (n, n)

    # Radial spatial frequency cannot be negative.
    assert np.all(kappa >= 0.0)

    # The zero-frequency component must be at [0, 0]
    # when using the unshifted FFT convention.
    assert kappa[0, 0] == 0.0

    # By definition:
    # kappa = sqrt(kx^2 + ky^2)
    assert np.allclose(
        kappa,
        np.hypot(kx, ky),
    )


def test_kolmogorov_phase_screen():
    """
    Check basic numerical properties of a Kolmogorov phase screen.
    """
    n = 128
    window_size = 0.4
    delta = window_size / n
    r0 = 6.7e-3

    rng = np.random.default_rng(12345)

    phase = kolmogorov_phase_screen(
        n=n,
        delta=delta,
        r0=r0,
        rng=rng,
        remove_piston=True,
    )

    # Correct dimensions.
    assert phase.shape == (n, n)

    # Phase must be real-valued.
    assert np.isrealobj(phase)

    # All values must be finite.
    assert np.all(np.isfinite(phase))

    # Removing piston should produce approximately zero mean.
    assert np.isclose(
        np.mean(phase),
        0.0,
        atol=1e-12,
    )

    # A turbulence screen should not be spatially constant.
    assert np.std(phase) > 0.0

def test_von_karman_psd():
    """
    Check basic physical and numerical properties of the
    von Kármán PSD.
    """
    kappa = np.linspace(
        0.0,
        1000.0,
        1000,
    )

    r0 = 6.7e-3
    outer_scale = 10.0

    psd = von_karman_psd(
        kappa=kappa,
        r0=r0,
        outer_scale=outer_scale,
    )

    assert psd.shape == kappa.shape
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0.0)

    # Unlike Kolmogorov, the von Kármán spectrum remains
    # finite at zero spatial frequency.
    assert psd[0] > 0.0

    # The spectrum must decrease toward high frequencies.
    assert psd[-1] < psd[0]

def test_modified_von_karman_psd():
    """
    Check basic physical and numerical properties of the
    modified von Kármán PSD.
    """
    kappa = np.linspace(
        0.0,
        5000.0,
        2000,
    )

    r0 = 6.7e-3
    outer_scale = 10.0
    inner_scale = 5.0e-3

    von_karman = von_karman_psd(
        kappa=kappa,
        r0=r0,
        outer_scale=outer_scale,
    )

    modified = modified_von_karman_psd(
        kappa=kappa,
        r0=r0,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )

    assert modified.shape == kappa.shape
    assert np.all(np.isfinite(modified))
    assert np.all(modified >= 0.0)

    # The exponential inner-scale cutoff can only suppress
    # power relative to the standard von Kármán spectrum.
    assert np.all(
        modified <= von_karman
    )

    # At zero frequency the exponential factor is one,
    # so both spectra must coincide.
    assert np.isclose(
        modified[0],
        von_karman[0],
    )

    # At sufficiently high spatial frequencies, the
    # modified spectrum must be more strongly suppressed.
    assert modified[-1] < von_karman[-1]


def test_von_karman_phase_screen():
    """
    Check basic properties of a von Kármán phase screen.
    """
    n = 128
    window_size = 0.4
    delta = window_size / n

    rng = np.random.default_rng(12345)

    phase = von_karman_phase_screen(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        rng=rng,
    )

    assert phase.shape == (n, n)
    assert np.isrealobj(phase)
    assert np.all(np.isfinite(phase))

    assert np.isclose(
        np.mean(phase),
        0.0,
        atol=1e-12,
    )

    assert np.std(phase) > 0.0

def test_modified_von_karman_phase_screen():
    """
    Check basic properties of a modified von Kármán phase screen.
    """
    n = 128
    window_size = 0.4
    delta = window_size / n

    rng = np.random.default_rng(12345)

    phase = modified_von_karman_phase_screen(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        inner_scale=5.0e-3,
        rng=rng,
    )

    assert phase.shape == (n, n)
    assert np.isrealobj(phase)
    assert np.all(np.isfinite(phase))

    assert np.isclose(
        np.mean(phase),
        0.0,
        atol=1e-12,
    )

    assert np.std(phase) > 0.0


def test_phase_screen_reproducibility():
    """
    The same random seed must reproduce the same phase screen.
    """
    n = 64
    delta = 0.4 / n

    rng_1 = np.random.default_rng(2026)
    rng_2 = np.random.default_rng(2026)

    phase_1 = von_karman_phase_screen(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        rng=rng_1,
    )

    phase_2 = von_karman_phase_screen(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        rng=rng_2,
    )

    assert np.array_equal(
        phase_1,
        phase_2,
    )

def test_kolmogorov_subharmonics():
    """
    Check basic properties of the Kolmogorov subharmonic component.
    """
    n = 64
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    subharmonics = kolmogorov_subharmonics(
        n=n,
        delta=delta,
        r0=6.7e-3,
        n_subharmonics=3,
        rng=rng,
        remove_piston=True,
    )

    assert subharmonics.shape == (n, n)
    assert np.isrealobj(subharmonics)
    assert np.all(np.isfinite(subharmonics))

    assert np.std(subharmonics) > 0.0

    assert np.isclose(
        np.mean(subharmonics),
        0.0,
        atol=1e-12,
    )

def test_zero_subharmonics_returns_zero():
    """
    Zero subharmonic levels must produce a zero contribution.
    """
    n = 32
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    subharmonics = kolmogorov_subharmonics(
        n=n,
        delta=delta,
        r0=6.7e-3,
        n_subharmonics=0,
        rng=rng,
    )

    assert np.allclose(
        subharmonics,
        0.0,
    )

def test_von_karman_subharmonics():
    n = 64
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    subharmonics = von_karman_subharmonics(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        n_subharmonics=3,
        rng=rng,
        remove_piston=True,
    )

    assert subharmonics.shape == (n, n)
    assert np.isrealobj(subharmonics)
    assert np.all(np.isfinite(subharmonics))
    assert np.std(subharmonics) > 0.0
    assert np.isclose(
        np.mean(subharmonics),
        0.0,
        atol=1e-12,
    )

def test_modified_von_karman_subharmonics():
    n = 64
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    subharmonics = modified_von_karman_subharmonics(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        inner_scale=5.0e-3,
        n_subharmonics=3,
        rng=rng,
        remove_piston=True,
    )

    assert subharmonics.shape == (n, n)
    assert np.isrealobj(subharmonics)
    assert np.all(np.isfinite(subharmonics))
    assert np.std(subharmonics) > 0.0
    assert np.isclose(
        np.mean(subharmonics),
        0.0,
        atol=1e-12,
    )

def test_kolmogorov_full_screen_matches_manual_construction():
    """
    The combined Kolmogorov screen must reproduce the manual
    FFT + subharmonic construction when the same random seed
    and RNG consumption order are used.
    """
    n = 64
    delta = 0.4 / n
    r0 = 6.7e-3
    n_subharmonics = 3
    seed = 12345

    # --------------------------------------------------------
    # Manual construction
    # --------------------------------------------------------

    rng_manual = np.random.default_rng(seed)

    phase_fft = kolmogorov_phase_screen(
        n=n,
        delta=delta,
        r0=r0,
        rng=rng_manual,
        remove_piston=False,
    )

    phase_sub = kolmogorov_subharmonics(
        n=n,
        delta=delta,
        r0=r0,
        n_subharmonics=n_subharmonics,
        rng=rng_manual,
        remove_piston=False,
    )

    phase_manual = (
        phase_fft
        + phase_sub
    )

    phase_manual -= np.mean(
        phase_manual
    )

    # --------------------------------------------------------
    # Combined implementation
    # --------------------------------------------------------

    rng_combined = np.random.default_rng(seed)

    phase_combined = (
        kolmogorov_phase_screen_with_subharmonics(
            n=n,
            delta=delta,
            r0=r0,
            n_subharmonics=n_subharmonics,
            rng=rng_combined,
            remove_piston=True,
        )
    )

    assert np.allclose(
        phase_combined,
        phase_manual,
        rtol=0.0,
        atol=1e-12,
    )

def test_zero_subharmonics_matches_fft_screen():
    """
    A full screen with zero subharmonic levels must reduce to
    the standard FFT phase screen.
    """
    n = 64
    delta = 0.4 / n
    r0 = 6.7e-3
    seed = 2026

    rng_fft = np.random.default_rng(seed)

    phase_fft = kolmogorov_phase_screen(
        n=n,
        delta=delta,
        r0=r0,
        rng=rng_fft,
        remove_piston=True,
    )

    rng_full = np.random.default_rng(seed)

    phase_full = (
        kolmogorov_phase_screen_with_subharmonics(
            n=n,
            delta=delta,
            r0=r0,
            n_subharmonics=0,
            rng=rng_full,
            remove_piston=True,
        )
    )

    assert np.allclose(
        phase_full,
        phase_fft,
        rtol=0.0,
        atol=1e-12,
    )

def test_accumulated_subharmonic_levels_match_combined_generator():
    """
    Sequentially generated subharmonic levels must reproduce
    the combined subharmonic generator when the same random
    seed and RNG order are used.
    """

    n = 64
    delta = 0.4 / n
    r0 = 6.7e-3
    maximum_level = 5
    seed = 12345

    # --------------------------------------------------------
    # Combined generator
    # --------------------------------------------------------

    rng_combined = np.random.default_rng(
        seed
    )

    combined = kolmogorov_subharmonics(
        n=n,
        delta=delta,
        r0=r0,
        n_subharmonics=maximum_level,
        rng=rng_combined,
        remove_piston=False,
    )

    # --------------------------------------------------------
    # Sequential levels
    # --------------------------------------------------------

    rng_sequential = np.random.default_rng(
        seed
    )

    sequential = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    for level in range(
        1,
        maximum_level + 1,
    ):
        sequential += (
            kolmogorov_subharmonic_level(
                n=n,
                delta=delta,
                r0=r0,
                level=level,
                rng=rng_sequential,
                remove_piston=False,
            )
        )

    assert np.allclose(
        sequential,
        combined,
        rtol=0.0,
        atol=1e-12,
    )

def test_von_karman_subharmonic_level():
    n = 64
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    phase = von_karman_subharmonic_level(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        level=3,
        rng=rng,
        remove_piston=True,
    )

    assert phase.shape == (n, n)
    assert np.isrealobj(phase)
    assert np.all(np.isfinite(phase))
    assert np.std(phase) > 0.0
    assert np.isclose(
        np.mean(phase),
        0.0,
        atol=1e-12,
    )

def test_modified_von_karman_subharmonic_level():
    n = 64
    delta = 0.4 / n

    rng = np.random.default_rng(12345)

    phase = modified_von_karman_subharmonic_level(
        n=n,
        delta=delta,
        r0=6.7e-3,
        outer_scale=10.0,
        inner_scale=5.0e-3,
        level=3,
        rng=rng,
        remove_piston=True,
    )

    assert phase.shape == (n, n)
    assert np.isrealobj(phase)
    assert np.all(np.isfinite(phase))
    assert np.std(phase) > 0.0
    assert np.isclose(
        np.mean(phase),
        0.0,
        atol=1e-12,
    )
