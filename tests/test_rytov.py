import numpy as np

from src.rytov import (
    gaussian_beam_rytov_parameters,
    gaussian_on_axis_scintillation_curve,
    gaussian_on_axis_scintillation_index,
    plane_wave_rytov_variance,
)


WAVELENGTH = 632.8e-9
WAIST_RADIUS = 0.025
CN2_WEAK = 1.0e-15


def test_plane_wave_rytov_variance_at_zero_distance():
    result = plane_wave_rytov_variance(
        cn2=CN2_WEAK,
        wavelength=WAVELENGTH,
        distance=0.0,
    )

    assert result == 0.0


def test_plane_wave_rytov_variance_final_distance():
    result = plane_wave_rytov_variance(
        cn2=CN2_WEAK,
        wavelength=WAVELENGTH,
        distance=1000.0,
    )

    assert np.isclose(
        result,
        5.66e-2,
        rtol=2.0e-2,
    )


def test_gaussian_beam_parameters_at_zero_distance():
    theta, lambda_parameter, theta_bar = (
        gaussian_beam_rytov_parameters(
            wavelength=WAVELENGTH,
            waist_radius=WAIST_RADIUS,
            distance=0.0,
        )
    )

    assert theta == 1.0
    assert lambda_parameter == 0.0
    assert theta_bar == 0.0


def test_gaussian_beam_parameters_are_finite():
    theta, lambda_parameter, theta_bar = (
        gaussian_beam_rytov_parameters(
            wavelength=WAVELENGTH,
            waist_radius=WAIST_RADIUS,
            distance=1000.0,
        )
    )

    assert np.isfinite(theta)
    assert np.isfinite(lambda_parameter)
    assert np.isfinite(theta_bar)

    assert 0.0 < theta <= 1.0
    assert lambda_parameter >= 0.0

    assert np.isclose(
        theta_bar,
        1.0 - theta,
    )


def test_on_axis_scintillation_is_zero_at_origin():
    result = gaussian_on_axis_scintillation_index(
        cn2=CN2_WEAK,
        wavelength=WAVELENGTH,
        waist_radius=WAIST_RADIUS,
        distance=0.0,
    )

    assert result == 0.0


def test_on_axis_scintillation_is_positive():
    result = gaussian_on_axis_scintillation_index(
        cn2=CN2_WEAK,
        wavelength=WAVELENGTH,
        waist_radius=WAIST_RADIUS,
        distance=1000.0,
    )

    assert np.isfinite(result)
    assert result > 0.0


def test_scintillation_curve_shape():
    distances = np.linspace(
        0.0,
        1000.0,
        17,
    )

    curve = gaussian_on_axis_scintillation_curve(
        distances=distances,
        cn2=CN2_WEAK,
        wavelength=WAVELENGTH,
        waist_radius=WAIST_RADIUS,
    )

    assert curve.shape == distances.shape
    assert np.all(
        np.isfinite(curve)
    )

    assert curve[0] == 0.0
    assert np.all(
        curve >= 0.0
    )
