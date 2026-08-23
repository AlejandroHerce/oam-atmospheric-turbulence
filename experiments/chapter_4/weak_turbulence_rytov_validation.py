"""
Weak-turbulence validation of the split-step propagation model
against the analytical Rytov prediction for a Gaussian beam.

A Gaussian beam is propagated over 1000 m through homogeneous
weak Kolmogorov turbulence. The path is discretized into 16 equal
segments, with one phase screen located at the center of each
segment.

For every independent atmospheric realization, the on-axis
irradiance is recorded at

    z = 0, 62.5, 125, ..., 1000 m.

The ensemble moments are then used to calculate the numerical
on-axis scintillation index and compare it with the weak-turbulence
Rytov prediction.
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
    DX,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    RYTOV_NUMBER_OF_REALIZATIONS,
    RYTOV_R0_SCREEN,
    RYTOV_SUBHARMONIC_LEVEL,
    SCREEN_SPACING,
    TOTAL_PROPAGATION_DISTANCE,
    WEAK_CN2,
    WEAK_R0_TOTAL,
    WAVELENGTH,
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

from src.rytov import (
    gaussian_on_axis_scintillation_curve,
    plane_wave_rytov_variance,
)


# ============================================================
# Execution configuration
# ============================================================

DEFAULT_NUMBER_OF_WORKERS = min(
    8,
    os.cpu_count() or 1,
)

DEFAULT_SEED = 20260819

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/weak_turbulence_rytov_validation"
)


# ============================================================
# Grid and input beam
# ============================================================

def create_gaussian_input():
    """
    Create the Gaussian beam used in the Rytov validation.
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
# Observation planes
# ============================================================

def observation_distances() -> np.ndarray:
    """
    Return observation distances

        z = 0, dz, 2 dz, ..., L.
    """

    return (
        np.arange(
            NUMBER_OF_PHASE_SCREENS + 1,
            dtype=np.float64,
        )
        * SCREEN_SPACING
    )


# ============================================================
# One realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
) -> np.ndarray:
    """
    Propagate one independent atmospheric realization.

    For each segment:

        propagate dz/2
        -> apply phase screen
        -> propagate dz/2
        -> record on-axis irradiance.

    Returns
    -------
    intensities:
        On-axis irradiance at all observation planes.
        Shape = (N_phase_screens + 1,).
    """

    rng = np.random.default_rng(
        realization_seed
    )

    grid, field = (
        create_gaussian_input()
    )

    number_of_planes = (
        NUMBER_OF_PHASE_SCREENS
        + 1
    )

    intensities = np.zeros(
        number_of_planes,
        dtype=np.float64,
    )

    # Because the centered grid contains x = y = 0 at N//2.
    center = (
        N_GRID // 2
    )

    # --------------------------------------------------------
    # z = 0
    # --------------------------------------------------------

    intensities[0] = float(
        np.abs(
            field[
                center,
                center,
            ]
        ) ** 2
    )

    # --------------------------------------------------------
    # Split-step propagation
    # --------------------------------------------------------

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        # First half of the segment.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        # Turbulence at the segment center.
        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=RYTOV_R0_SCREEN,
                n_subharmonics=(
                    RYTOV_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        # Second half of the segment.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        # Observation at the segment boundary.
        intensities[
            screen_index + 1
        ] = float(
            np.abs(
                field[
                    center,
                    center,
                ]
            ) ** 2
        )

    return intensities


# ============================================================
# Reproducible seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
) -> list[int]:
    """
    Generate independent reproducible realization seeds.
    """

    if number_of_realizations <= 0:
        raise ValueError(
            "number_of_realizations must be positive."
        )

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in child_sequences
    ]


# ============================================================
# Parallel ensemble
# ============================================================

def run_ensemble(
    number_of_realizations: int,
    number_of_workers: int,
) -> np.ndarray:
    """
    Execute the atmospheric ensemble in parallel.

    Returns
    -------
    intensity_samples:
        Array with shape

            (N_realizations, N_observation_planes).
    """

    realization_seeds = (
        generate_realization_seeds(
            number_of_realizations
        )
    )

    intensity_samples = np.zeros(
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
            realization_seeds,
            chunksize=1,
        )

        for index, intensities in enumerate(
            results
        ):
            intensity_samples[index] = (
                intensities
            )

            completed = (
                index + 1
            )

            if (
                completed == 1
                or completed % 25 == 0
                or completed == number_of_realizations
            ):
                print(
                    f"Realizaciones completadas: "
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return intensity_samples


# ============================================================
# Numerical scintillation
# ============================================================

def calculate_numerical_scintillation(
    intensity_samples: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate ensemble irradiance moments and scintillation index.

        sigma_I^2 =
        <I^2> / <I>^2 - 1.
    """

    if intensity_samples.ndim != 2:
        raise ValueError(
            "intensity_samples must be two-dimensional."
        )

    mean_intensity = np.mean(
        intensity_samples,
        axis=0,
    )

    mean_squared_intensity = np.mean(
        intensity_samples**2,
        axis=0,
    )

    scintillation = (
        mean_squared_intensity
        / mean_intensity**2
        - 1.0
    )

    # At z=0 all realizations are identical.
    # Remove numerical round-off around zero.
    scintillation[0] = 0.0

    return (
        mean_intensity,
        mean_squared_intensity,
        scintillation,
    )


# ============================================================
# Comparison metrics
# ============================================================

def calculate_relative_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> np.ndarray:
    """
    Calculate pointwise relative error in percent.

    The z=0 value is undefined because the theoretical
    scintillation index is zero and is returned as NaN.
    """

    relative_error = np.full(
        numerical.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        theoretical > 0.0
    )

    relative_error[valid] = (
        100.0
        * np.abs(
            numerical[valid]
            - theoretical[valid]
        )
        / theoretical[valid]
    )

    return relative_error


def calculate_relative_l2_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Calculate global relative L2 error between the numerical
    and theoretical scintillation curves.
    """

    valid = (
        theoretical > 0.0
    )

    numerator = np.linalg.norm(
        numerical[valid]
        - theoretical[valid]
    )

    denominator = np.linalg.norm(
        theoretical[valid]
    )

    if denominator <= 0.0:
        raise ValueError(
            "The theoretical curve has zero norm."
        )

    return float(
        100.0
        * numerator
        / denominator
    )


# ============================================================
# Save raw samples
# ============================================================

def save_intensity_samples(
    distances: np.ndarray,
    intensity_samples: np.ndarray,
) -> None:
    """
    Save on-axis irradiance samples for every realization.
    """

    filename = (
        OUTPUT_DIRECTORY
        / "on_axis_intensity_realizations.csv"
    )

    header = [
        "realization"
    ] + [
        f"I_z_{distance:.4f}_m"
        for distance in distances
    ]

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            header
        )

        for index, row in enumerate(
            intensity_samples,
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    *row,
                ]
            )


# ============================================================
# Save comparison table
# ============================================================

def save_comparison_results(
    distances: np.ndarray,
    mean_intensity: np.ndarray,
    mean_squared_intensity: np.ndarray,
    numerical_scintillation: np.ndarray,
    rytov_scintillation: np.ndarray,
    relative_error: np.ndarray,
) -> None:
    """
    Save numerical and theoretical results.
    """

    filename = (
        OUTPUT_DIRECTORY
        / "rytov_comparison.csv"
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
                "distance_m",
                "mean_intensity",
                "mean_squared_intensity",
                "numerical_scintillation",
                "rytov_scintillation",
                "relative_error_percent",
            ]
        )

        for values in zip(
            distances,
            mean_intensity,
            mean_squared_intensity,
            numerical_scintillation,
            rytov_scintillation,
            relative_error,
        ):
            writer.writerow(
                values
            )


# ============================================================
# Terminal output
# ============================================================

def print_comparison_table(
    distances: np.ndarray,
    numerical_scintillation: np.ndarray,
    rytov_scintillation: np.ndarray,
    relative_error: np.ndarray,
) -> None:
    """
    Print numerical-versus-Rytov comparison.
    """

    print()
    print(
        "Validación Split-Step vs Rytov"
    )
    print(
        "=============================="
    )

    header = (
        f"{'z [m]':>10}"
        f"{'sigma_I^2 num':>18}"
        f"{'sigma_I^2 Rytov':>20}"
        f"{'error [%]':>14}"
    )

    print(
        header
    )
    print(
        "-" * len(header)
    )

    for (
        distance,
        numerical,
        theoretical,
        error,
    ) in zip(
        distances,
        numerical_scintillation,
        rytov_scintillation,
        relative_error,
    ):

        error_text = (
            "-"
            if not np.isfinite(error)
            else f"{error:.4f}"
        )

        print(
            f"{distance:>10.2f}"
            f"{numerical:>18.8e}"
            f"{theoretical:>20.8e}"
            f"{error_text:>14}"
        )


# ============================================================
# Plots
# ============================================================

def plot_scintillation_comparison(
    distances: np.ndarray,
    numerical_scintillation: np.ndarray,
    rytov_scintillation: np.ndarray,
) -> None:
    """
    Plot Split-Step and Rytov scintillation curves.
    """

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        distances,
        numerical_scintillation,
        marker="o",
        linewidth=1.7,
        label="Split-Step",
    )

    axis.plot(
        distances,
        rytov_scintillation,
        linestyle="--",
        linewidth=2.0,
        label="Teoría de Rytov",
    )

    axis.set_xlabel(
        "Distancia de propagación z [m]"
    )

    axis.set_ylabel(
        r"Índice de centelleo "
        r"$\sigma_I^2(0,z)$"
    )

    axis.set_title(
        "Validación en turbulencia débil"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "split_step_vs_rytov.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_relative_error(
    distances: np.ndarray,
    relative_error: np.ndarray,
) -> None:
    """
    Plot pointwise relative error.
    """

    valid = np.isfinite(
        relative_error
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        distances[valid],
        relative_error[valid],
        marker="o",
        linewidth=1.7,
    )

    axis.set_xlabel(
        "Distancia de propagación z [m]"
    )

    axis.set_ylabel(
        "Error relativo [%]"
    )

    axis.set_title(
        "Error de la simulación respecto a Rytov"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "rytov_relative_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Command line
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--realizations",
        type=int,
        default=RYTOV_NUMBER_OF_REALIZATIONS,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUMBER_OF_WORKERS,
    )

    return parser.parse_args()


# ============================================================
# Execution
# ============================================================

def run(
    number_of_realizations: int,
    number_of_workers: int,
) -> None:
    """
    Execute the weak-turbulence validation.
    """

    if number_of_realizations <= 0:
        raise ValueError(
            "number_of_realizations must be positive."
        )

    if number_of_workers <= 0:
        raise ValueError(
            "number_of_workers must be positive."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    distances = (
        observation_distances()
    )

    final_rytov_variance = (
        plane_wave_rytov_variance(
            cn2=WEAK_CN2,
            wavelength=WAVELENGTH,
            distance=(
                TOTAL_PROPAGATION_DISTANCE
            ),
        )
    )

    print(
        "Validación de Rytov en turbulencia débil"
    )
    print(
        "========================================"
    )

    print(
        f"Distancia total: "
        f"{TOTAL_PROPAGATION_DISTANCE:.1f} m"
    )

    print(
        f"Pantallas de fase: "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Separación: "
        f"{SCREEN_SPACING:.2f} m"
    )

    print(
        f"Cn^2: "
        f"{WEAK_CN2:.3e} m^(-2/3)"
    )

    print(
        f"r0 total: "
        f"{WEAK_R0_TOTAL:.6f} m"
    )

    print(
        f"r0 por pantalla: "
        f"{RYTOV_R0_SCREEN:.6f} m"
    )

    print(
        f"Subarmónicos: "
        f"{RYTOV_SUBHARMONIC_LEVEL}"
    )

    print(
        f"Varianza Rytov final: "
        f"{final_rytov_variance:.8e}"
    )

    print(
        f"Realizaciones: "
        f"{number_of_realizations}"
    )

    print(
        f"Workers: "
        f"{number_of_workers}"
    )

    print()

    # --------------------------------------------------------
    # Numerical ensemble
    # --------------------------------------------------------

    intensity_samples = (
        run_ensemble(
            number_of_realizations=(
                number_of_realizations
            ),
            number_of_workers=(
                number_of_workers
            ),
        )
    )

    (
        mean_intensity,
        mean_squared_intensity,
        numerical_scintillation,
    ) = calculate_numerical_scintillation(
        intensity_samples
    )

    # --------------------------------------------------------
    # Analytical Rytov curve
    # --------------------------------------------------------

    rytov_scintillation = (
        gaussian_on_axis_scintillation_curve(
            distances=distances,
            cn2=WEAK_CN2,
            wavelength=WAVELENGTH,
            waist_radius=W0_GAUSSIAN,
        )
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    relative_error = (
        calculate_relative_error(
            numerical=(
                numerical_scintillation
            ),
            theoretical=(
                rytov_scintillation
            ),
        )
    )

    l2_error = (
        calculate_relative_l2_error(
            numerical=(
                numerical_scintillation
            ),
            theoretical=(
                rytov_scintillation
            ),
        )
    )

    print_comparison_table(
        distances=distances,
        numerical_scintillation=(
            numerical_scintillation
        ),
        rytov_scintillation=(
            rytov_scintillation
        ),
        relative_error=(
            relative_error
        ),
    )

    print()
    print(
        f"Error relativo global L2: "
        f"{l2_error:.4f} %"
    )

    print()
    print(
        "Plano final:"
    )
    print(
        f"  Split-Step = "
        f"{numerical_scintillation[-1]:.8e}"
    )
    print(
        f"  Rytov      = "
        f"{rytov_scintillation[-1]:.8e}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_intensity_samples(
        distances=distances,
        intensity_samples=(
            intensity_samples
        ),
    )

    save_comparison_results(
        distances=distances,
        mean_intensity=mean_intensity,
        mean_squared_intensity=(
            mean_squared_intensity
        ),
        numerical_scintillation=(
            numerical_scintillation
        ),
        rytov_scintillation=(
            rytov_scintillation
        ),
        relative_error=(
            relative_error
        ),
    )

    plot_scintillation_comparison(
        distances=distances,
        numerical_scintillation=(
            numerical_scintillation
        ),
        rytov_scintillation=(
            rytov_scintillation
        ),
    )

    plot_relative_error(
        distances=distances,
        relative_error=relative_error,
    )

    print(
        "\nResultados guardados en:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    arguments = parse_arguments()

    run(
        number_of_realizations=(
            arguments.realizations
        ),
        number_of_workers=(
            arguments.workers
        ),
    )
