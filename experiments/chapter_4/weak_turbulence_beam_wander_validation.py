"""
Weak-turbulence beam-wander validation.

A collimated Gaussian beam is propagated through weak
Kolmogorov turbulence using the Chapter 4 split-step
configuration.

At every observation plane the intensity centroid is calculated,

    x_c = integral x I dxdy / integral I dxdy
    y_c = integral y I dxdy / integral I dxdy

and the ensemble beam wander is estimated as

    sigma_BW = sqrt(<x_c^2 + y_c^2>).

The numerical result is compared with the weak-turbulence
prediction

    <r_c^2> = 2.42 Cn2 z^3 w0^(-1/3).
"""

import argparse
import csv
import os

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from configs.chapter_2 import (
    W0_GAUSSIAN,
)

from configs.chapter_4 import (
    BEAM_WANDER_BOOTSTRAP_CONFIDENCE_LEVEL,
    BEAM_WANDER_BOOTSTRAP_SAMPLES,
    BEAM_WANDER_BOOTSTRAP_SEED,
    BEAM_WANDER_NUMBER_OF_REALIZATIONS,
    BEAM_WANDER_R0_SCREEN,
    BEAM_WANDER_SUBHARMONIC_LEVEL,
    DX,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    TOTAL_PROPAGATION_DISTANCE,
    WAVELENGTH,
    WEAK_CN2,
)

from src.beams import (
    gaussian_beam,
)

from src.grids import (
    create_grid,
)

from src.phase_screens import (
    kolmogorov_phase_screen_with_subharmonics,
)

from src.propagation import (
    angular_spectrum_propagation,
)


# ============================================================
# Execution configuration
# ============================================================

DEFAULT_NUMBER_OF_WORKERS = min(
    12,
    os.cpu_count() or 1,
)

MASTER_SEED = 20260824

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/"
    "weak_turbulence_beam_wander_validation"
)


# ============================================================
# Observation planes
# ============================================================

def observation_distances() -> np.ndarray:
    """
    Return the receiver planes associated with the split-step
    segments.
    """

    return np.linspace(
        0.0,
        TOTAL_PROPAGATION_DISTANCE,
        NUMBER_OF_PHASE_SCREENS + 1,
        dtype=np.float64,
    )


# ============================================================
# Input beam
# ============================================================

def create_input_beam():
    """
    Create the Gaussian input field.
    """

    grid = create_grid(
        n=N_GRID,
        window_size=L_WINDOW,
    )

    if not np.isclose(
        grid.dx,
        DX,
    ):
        raise RuntimeError(
            "Grid spacing does not match Chapter 4 configuration."
        )

    field = gaussian_beam(
        grid=grid,
        w0=W0_GAUSSIAN,
    )

    return (
        grid,
        field,
    )


# ============================================================
# Intensity centroid
# ============================================================

def calculate_intensity_centroid(
    field,
    grid,
) -> tuple[float, float]:
    """
    Calculate the transverse intensity centroid.
    """

    intensity = (
        np.abs(field) ** 2
    )

    total_intensity = float(
        np.sum(intensity)
    )

    if (
        not np.isfinite(total_intensity)
        or total_intensity <= 0.0
    ):
        raise ValueError(
            "Invalid field intensity."
        )

    x_centroid = float(
        np.sum(
            grid.x * intensity
        )
        / total_intensity
    )

    y_centroid = float(
        np.sum(
            grid.y * intensity
        )
        / total_intensity
    )

    return (
        x_centroid,
        y_centroid,
    )


# ============================================================
# One atmospheric realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
) -> np.ndarray:
    """
    Propagate one independent atmospheric realization.

    Returns
    -------
    radial_squared:
        x_c^2 + y_c^2 at every observation plane.
    """

    rng = np.random.default_rng(
        realization_seed
    )

    grid, field = (
        create_input_beam()
    )

    radial_squared = np.zeros(
        NUMBER_OF_PHASE_SCREENS + 1,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Input plane
    # --------------------------------------------------------

    (
        x_centroid,
        y_centroid,
    ) = calculate_intensity_centroid(
        field=field,
        grid=grid,
    )

    radial_squared[0] = (
        x_centroid**2
        + y_centroid**2
    )

    # --------------------------------------------------------
    # Split-step propagation
    # --------------------------------------------------------

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        # Half step to phase screen
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=BEAM_WANDER_R0_SCREEN,
                n_subharmonics=(
                    BEAM_WANDER_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        # Half step to observation plane
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        (
            x_centroid,
            y_centroid,
        ) = calculate_intensity_centroid(
            field=field,
            grid=grid,
        )

        radial_squared[
            screen_index + 1
        ] = (
            x_centroid**2
            + y_centroid**2
        )

    return radial_squared


# ============================================================
# Seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
) -> list[int]:
    """
    Generate deterministic independent realization seeds.
    """

    seed_sequence = np.random.SeedSequence(
        MASTER_SEED
    )

    children = seed_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in children
    ]


# ============================================================
# Ensemble
# ============================================================

def run_ensemble(
    number_of_realizations: int,
    number_of_workers: int,
) -> np.ndarray:
    """
    Run all atmospheric realizations.
    """

    seeds = generate_realization_seeds(
        number_of_realizations
    )

    samples = np.zeros(
        (
            number_of_realizations,
            NUMBER_OF_PHASE_SCREENS + 1,
        ),
        dtype=np.float64,
    )

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            simulate_one_realization,
            seeds,
            chunksize=1,
        )

        for index, result in enumerate(
            results
        ):

            samples[index] = result

            completed = (
                index + 1
            )

            if (
                completed == 1
                or completed % 25 == 0
                or completed
                == number_of_realizations
            ):
                print(
                    "Realizaciones completadas: "
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return samples


# ============================================================
# Numerical centroid wander
# ============================================================

def calculate_beam_wander(
    radial_squared_samples: np.ndarray,
) -> np.ndarray:
    """
    Calculate the RMS radial centroid displacement

        sigma_c(z)
        =
        sqrt(<x_c^2 + y_c^2>).
    """

    return np.sqrt(
        np.mean(
            radial_squared_samples,
            axis=0,
        )
    )


# ============================================================
# Conventional beam-wander reference
# ============================================================

def theoretical_beam_wander(
    z: np.ndarray,
) -> np.ndarray:
    """
    Conventional weak-turbulence beam-wander reference

        <r_c^2>
        =
        2.42 Cn2 z^3 w0^(-1/3).

    This curve is retained only as a secondary reference in the
    numerical table. It is not used for the validation error.
    """

    variance = (
        2.42
        * WEAK_CN2
        * z**3
        * W0_GAUSSIAN ** (-1.0 / 3.0)
    )

    return np.sqrt(
        variance
    )


# ============================================================
# Centroid-wander theoretical prediction
# ============================================================

def theoretical_centroid_wander(
    z: np.ndarray,
) -> np.ndarray:
    """
    RMS radial centroid displacement for a collimated Gaussian
    beam in homogeneous Kolmogorov turbulence.

    The centroid variance is related to the conventional
    beam-wander variance through

        <rho_c^2>
        =
        0.56 <r_c^2>,

    with

        <r_c^2>
        =
        (7.25 / 3)
        Cn2 z^3 W0^(-1/3).

    Therefore

        sigma_centroid
        =
        sqrt[
            0.56 (7.25/3)
            Cn2 z^3 W0^(-1/3)
        ].
    """

    variance = (
        0.56
        * (
            7.25
            / 3.0
        )
        * WEAK_CN2
        * z**3
        * W0_GAUSSIAN ** (-1.0 / 3.0)
    )

    return np.sqrt(
        variance
    )


# ============================================================
# Bootstrap confidence intervals
# ============================================================

def bootstrap_beam_wander(
    radial_squared_samples: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Bootstrap confidence interval of the RMS centroid wander.
    """

    rng = np.random.default_rng(
        BEAM_WANDER_BOOTSTRAP_SEED
    )

    number_of_realizations = (
        radial_squared_samples.shape[0]
    )

    number_of_planes = (
        radial_squared_samples.shape[1]
    )

    bootstrap_values = np.zeros(
        (
            BEAM_WANDER_BOOTSTRAP_SAMPLES,
            number_of_planes,
        ),
        dtype=np.float64,
    )

    for bootstrap_index in range(
        BEAM_WANDER_BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            number_of_realizations,
            size=number_of_realizations,
        )

        bootstrap_values[
            bootstrap_index
        ] = np.sqrt(
            np.mean(
                radial_squared_samples[
                    indices
                ],
                axis=0,
            )
        )

    alpha = (
        1.0
        - BEAM_WANDER_BOOTSTRAP_CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_values,
        alpha / 2.0,
        axis=0,
    )

    upper = np.quantile(
        bootstrap_values,
        1.0 - alpha / 2.0,
        axis=0,
    )

    return (
        lower,
        upper,
    )


# ============================================================
# Relative errors
# ============================================================

def pointwise_relative_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> np.ndarray:
    """
    Relative error [%] at every non-zero theoretical value.
    """

    error = np.full_like(
        numerical,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        theoretical > 0.0
    )

    error[valid] = (
        100.0
        * np.abs(
            numerical[valid]
            - theoretical[valid]
        )
        / theoretical[valid]
    )

    return error


def relative_l2_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Global relative L2 error [%].
    """

    denominator = np.linalg.norm(
        theoretical
    )

    if denominator <= 0.0:
        raise ValueError(
            "Theoretical norm must be positive."
        )

    return float(
        100.0
        * np.linalg.norm(
            numerical
            - theoretical
        )
        / denominator
    )


# ============================================================
# Save raw data
# ============================================================

def save_raw_samples(
    samples: np.ndarray,
) -> None:
    """
    Save x_c^2 + y_c^2 for every realization and observation
    plane.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        OUTPUT_DIRECTORY
        / "beam_wander_radial_squared_samples.csv",
        samples,
        delimiter=",",
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    z: np.ndarray,
    numerical: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    centroid_theory: np.ndarray,
    beam_wander_reference: np.ndarray,
    relative_error: np.ndarray,
    theory_inside_ci: np.ndarray,
) -> None:
    """
    Save plane-by-plane centroid-wander validation.
    """

    filename = (
        OUTPUT_DIRECTORY
        / "beam_wander_validation_summary.csv"
    )

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "z_m",
                "centroid_wander_numerical_m",
                "ci95_lower_m",
                "ci95_upper_m",
                "centroid_theory_m",
                "beam_wander_reference_2p42_m",
                "relative_error_centroid_percent",
                "centroid_theory_inside_ci95",
            ]
        )

        for index in range(
            z.size
        ):

            writer.writerow(
                [
                    z[index],
                    numerical[index],
                    lower[index],
                    upper[index],
                    centroid_theory[index],
                    beam_wander_reference[index],
                    relative_error[index],
                    bool(
                        theory_inside_ci[
                            index
                        ]
                    ),
                ]
            )


# ============================================================
# Plot
# ============================================================

def plot_validation(
    z: np.ndarray,
    numerical: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    centroid_theory: np.ndarray,
) -> None:
    """
    Plot numerical centroid wander and its corresponding
    theoretical prediction.

    The conventional 2.42 beam-wander reference is deliberately
    omitted from the figure because it does not correspond to
    the centroid observable used for validation.
    """

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        z,
        numerical,
        marker="o",
        linewidth=1.5,
        label="Split-Step",
    )

    axis.fill_between(
        z,
        lower,
        upper,
        alpha=0.2,
        label="IC bootstrap 95 %",
    )

    axis.plot(
        z,
        centroid_theory,
        linestyle="--",
        linewidth=1.8,
        label="Teoría del centroide",
    )

    axis.set_xlabel(
        r"$z$ [m]"
    )

    axis.set_ylabel(
        r"$\sigma_{\mathrm{c}}$ [m]"
    )

    axis.set_title(
        "Validación del desplazamiento RMS del centroide"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "beam_wander_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Print summary
# ============================================================

def print_summary(
    z: np.ndarray,
    numerical: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    centroid_theory: np.ndarray,
    beam_wander_reference: np.ndarray,
    relative_error: np.ndarray,
    theory_inside_ci: np.ndarray,
) -> None:
    """
    Print the centroid-wander validation table.
    """

    print()
    print(
        "Validación Split-Step vs centroid wander teórico"
    )

    print(
        "=" * 105
    )

    print(
        f"{'z [m]':>9} "
        f"{'num [m]':>13} "
        f"{'IC95 inf':>13} "
        f"{'IC95 sup':>13} "
        f"{'centroide [m]':>15} "
        f"{'ref. 2.42 [m]':>15} "
        f"{'err [%]':>10} "
        f"{'IC?':>5}"
    )

    print(
        "-" * 105
    )

    for index in range(
        z.size
    ):

        if np.isnan(
            relative_error[index]
        ):
            error_text = "-"
        else:
            error_text = (
                f"{relative_error[index]:.4f}"
            )

        if z[index] <= 0.0:
            interval_text = "-"
        else:
            interval_text = (
                "sí"
                if theory_inside_ci[index]
                else "no"
            )

        print(
            f"{z[index]:9.2f} "
            f"{numerical[index]:13.6e} "
            f"{lower[index]:13.6e} "
            f"{upper[index]:13.6e} "
            f"{centroid_theory[index]:15.6e} "
            f"{beam_wander_reference[index]:15.6e} "
            f"{error_text:>10} "
            f"{interval_text:>5}"
        )


# ============================================================
# Command line
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--realizations",
        type=int,
        default=(
            BEAM_WANDER_NUMBER_OF_REALIZATIONS
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=(
            DEFAULT_NUMBER_OF_WORKERS
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    arguments = parse_arguments()

    if arguments.realizations <= 0:
        raise ValueError(
            "realizations must be positive."
        )

    if arguments.workers <= 0:
        raise ValueError(
            "workers must be positive."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Validación del desplazamiento del centroide"
    )

    print(
        "=========================================="
    )

    print(
        f"Haz gaussiano: "
        f"w0 = {W0_GAUSSIAN:.6f} m"
    )

    print(
        f"Cn2 = "
        f"{WEAK_CN2:.3e} m^(-2/3)"
    )

    print(
        f"r0 por pantalla = "
        f"{BEAM_WANDER_R0_SCREEN:.6f} m"
    )

    print(
        f"Pantallas = "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Subarmónicos = "
        f"{BEAM_WANDER_SUBHARMONIC_LEVEL}"
    )

    print(
        f"Realizaciones = "
        f"{arguments.realizations}"
    )

    print(
        f"Workers = "
        f"{arguments.workers}"
    )

    print()

    # --------------------------------------------------------
    # Monte Carlo propagation
    # --------------------------------------------------------

    samples = run_ensemble(
        number_of_realizations=(
            arguments.realizations
        ),
        number_of_workers=(
            arguments.workers
        ),
    )

    save_raw_samples(
        samples
    )

    # --------------------------------------------------------
    # Observation planes
    # --------------------------------------------------------

    z = observation_distances()

    # --------------------------------------------------------
    # Numerical centroid wander
    # --------------------------------------------------------

    numerical = calculate_beam_wander(
        samples
    )

    # --------------------------------------------------------
    # Theoretical references
    # --------------------------------------------------------

    centroid_theory = (
        theoretical_centroid_wander(
            z
        )
    )

    beam_wander_reference = (
        theoretical_beam_wander(
            z
        )
    )

    # --------------------------------------------------------
    # Bootstrap confidence interval
    # --------------------------------------------------------

    (
        lower,
        upper,
    ) = bootstrap_beam_wander(
        samples
    )

    # --------------------------------------------------------
    # Pointwise and global errors
    # --------------------------------------------------------

    relative_error = (
        pointwise_relative_error(
            numerical=numerical,
            theoretical=centroid_theory,
        )
    )

    l2_error = (
        relative_l2_error(
            numerical=numerical,
            theoretical=centroid_theory,
        )
    )

    # --------------------------------------------------------
    # Theory inside bootstrap confidence interval
    # --------------------------------------------------------

    theory_inside_ci = (
        (centroid_theory >= lower)
        & (centroid_theory <= upper)
    )

    valid_planes = (
        z > 0.0
    )

    number_inside_ci = int(
        np.sum(
            theory_inside_ci[
                valid_planes
            ]
        )
    )

    number_of_planes = int(
        np.sum(
            valid_planes
        )
    )

    fraction_inside_ci = (
        number_inside_ci
        / number_of_planes
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print_summary(
        z=z,
        numerical=numerical,
        lower=lower,
        upper=upper,
        centroid_theory=centroid_theory,
        beam_wander_reference=(
            beam_wander_reference
        ),
        relative_error=relative_error,
        theory_inside_ci=theory_inside_ci,
    )

    print()

    print(
        f"Error relativo global L2: "
        f"{l2_error:.4f} %"
    )

    print()

    print(
        "Concordancia con IC95%:"
    )

    print(
        f"Teoría centroidal dentro del IC95% en "
        f"{number_inside_ci}/"
        f"{number_of_planes} planos "
        f"({100.0 * fraction_inside_ci:.1f} %)"
    )

    # --------------------------------------------------------
    # Final plane
    # --------------------------------------------------------

    final_index = -1

    print()

    print(
        "Plano final:"
    )

    print(
        "  Split-Step       = "
        f"{numerical[final_index]:.8e} m"
    )

    print(
        "  IC95%            = "
        f"[{lower[final_index]:.8e}, "
        f"{upper[final_index]:.8e}] m"
    )

    print(
        "  Teoría centroide = "
        f"{centroid_theory[final_index]:.8e} m"
    )

    print(
        "  Referencia 2.42  = "
        f"{beam_wander_reference[final_index]:.8e} m"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_summary(
        z=z,
        numerical=numerical,
        lower=lower,
        upper=upper,
        centroid_theory=centroid_theory,
        beam_wander_reference=(
            beam_wander_reference
        ),
        relative_error=relative_error,
        theory_inside_ci=theory_inside_ci,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_validation(
        z=z,
        numerical=numerical,
        lower=lower,
        upper=upper,
        centroid_theory=centroid_theory,
    )

    print()

    print(
        "Resultados guardados en:"
    )

    print(
        OUTPUT_DIRECTORY.resolve()
    )


if __name__ == "__main__":
    main()
