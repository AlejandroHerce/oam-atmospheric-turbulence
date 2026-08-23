import numpy as np

from experiments.chapter_4.ensemble_size_convergence import (
    create_input_beam,
)

from experiments.chapter_4.weak_turbulence_rytov_validation import (
    calculate_numerical_scintillation,
    create_gaussian_input,
    observation_distances,
)

from experiments.chapter_4.split_step_screen_convergence import (
    calculate_convergence_metrics,
    calculate_scintillation_index,
    generate_screen_seeds,
    segment_fried_parameter,
    calculate_paired_bootstrap_difference_ci,
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

def test_create_gaussian_input():
    grid, field = create_gaussian_input()

    assert field.shape == (
        N_GRID,
        N_GRID,
    )

    assert np.all(
        np.isfinite(field)
    )

    assert np.iscomplexobj(field)

    assert np.isclose(
        grid.dx,
        L_WINDOW / N_GRID,
    )


def test_rytov_observation_distances():
    distances = (
        observation_distances()
    )

    assert distances.size == 17

    assert np.isclose(
        distances[0],
        0.0,
    )

    assert np.isclose(
        distances[-1],
        1000.0,
    )

    assert np.allclose(
        np.diff(distances),
        62.5,
    )


def test_numerical_scintillation_zero_for_identical_samples():
    intensity_samples = np.ones(
        (10, 17),
        dtype=float,
    )

    (
        mean_intensity,
        mean_squared_intensity,
        scintillation,
    ) = calculate_numerical_scintillation(
        intensity_samples
    )

    assert np.allclose(
        mean_intensity,
        1.0,
    )

    assert np.allclose(
        mean_squared_intensity,
        1.0,
    )

    assert np.allclose(
        scintillation,
        0.0,
    )

def test_segment_fried_parameter():
    total_r0 = 0.0067

    r0_16 = segment_fried_parameter(
        total_r0=total_r0,
        number_of_screens=16,
    )

    expected = (
        total_r0
        * 16 ** (3.0 / 5.0)
    )

    assert np.isclose(
        r0_16,
        expected,
    )


def test_segment_fried_parameter_increases_with_screen_number():
    total_r0 = 0.0067

    r0_1 = segment_fried_parameter(
        total_r0,
        1,
    )

    r0_16 = segment_fried_parameter(
        total_r0,
        16,
    )

    assert r0_16 > r0_1


def test_scintillation_zero_for_constant_intensity():
    samples = np.ones(
        100,
        dtype=float,
    )

    scintillation = (
        calculate_scintillation_index(
            samples
        )
    )

    assert np.isclose(
        scintillation,
        0.0,
    )


def test_convergence_metrics_zero_at_reference():
    screen_numbers = (
        1,
        2,
        4,
        8,
    )

    values = np.array(
        [
            0.10,
            0.12,
            0.13,
            0.14,
        ]
    )

    (
        reference_error,
        incremental_change,
    ) = calculate_convergence_metrics(
        screen_numbers,
        values,
    )

    assert np.isclose(
        reference_error[-1],
        0.0,
    )

    assert np.isnan(
        incremental_change[0]
    )

    assert np.all(
        np.isfinite(
            incremental_change[1:]
        )
    )

def test_screen_seed_hierarchy_is_nested():
    realization_seed = 123456

    seeds_8 = generate_screen_seeds(
        realization_seed=realization_seed,
        number_of_screens=8,
    )

    seeds_64 = generate_screen_seeds(
        realization_seed=realization_seed,
        number_of_screens=64,
    )

    assert seeds_8 == seeds_64[:8]


def test_screen_seed_hierarchy_is_reproducible():
    realization_seed = 123456

    first = generate_screen_seeds(
        realization_seed=realization_seed,
        number_of_screens=16,
    )

    second = generate_screen_seeds(
        realization_seed=realization_seed,
        number_of_screens=16,
    )

    assert first == second

def test_paired_bootstrap_difference_identical_samples():
    samples = np.linspace(
        1.0,
        2.0,
        100,
    )

    (
        difference,
        lower,
        upper,
    ) = calculate_paired_bootstrap_difference_ci(
        previous_samples=samples,
        current_samples=samples.copy(),
        number_of_bootstrap_samples=500,
        confidence_level=0.95,
        seed=12345,
    )

    assert np.isclose(
        difference,
        0.0,
    )

    assert np.isclose(
        lower,
        0.0,
    )

    assert np.isclose(
        upper,
        0.0,
    )
