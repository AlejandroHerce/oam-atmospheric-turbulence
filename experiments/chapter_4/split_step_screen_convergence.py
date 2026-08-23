"""
Longitudinal split-step convergence in moderate and strong turbulence.

The final on-axis scintillation index is evaluated for

    Ns = 1, 2, 4, 8, 16, 32

phase screens while keeping the integrated turbulence strength
of the complete 1000 m path constant.

For each Ns, the Fried parameter represented by one screen is

    r0_screen = r0_total * Ns^(3/5).

The experiment is performed independently for moderate and
strong Kolmogorov turbulence.
"""

import argparse
import csv
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_2 import (
    W0_GAUSSIAN,
)

from configs.chapter_4 import (
    DX,
    L_WINDOW,
    MODERATE_R0_TOTAL,
    N_GRID,
    SCREEN_CONVERGENCE_BOOTSTRAP_CONFIDENCE_LEVEL,
    SCREEN_CONVERGENCE_BOOTSTRAP_SAMPLES,
    SCREEN_CONVERGENCE_BOOTSTRAP_SEED,
    SCREEN_CONVERGENCE_LEVELS,
    SCREEN_CONVERGENCE_NUMBER_OF_REALIZATIONS,
    SCREEN_CONVERGENCE_SEED,
    SCREEN_CONVERGENCE_SUBHARMONIC_LEVEL,
    STRONG_R0_TOTAL,
    TOTAL_PROPAGATION_DISTANCE,
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


# ============================================================
# Configuration
# ============================================================

DEFAULT_NUMBER_OF_WORKERS = min(
    8,
    os.cpu_count() or 1,
)

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/split_step_screen_convergence"
)


# ============================================================
# Input beam
# ============================================================

def create_gaussian_input():
    """
    Create the Gaussian input beam.
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
            "Grid spacing does not match "
            "Chapter 4 configuration."
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
# Fried parameter for one segment
# ============================================================

def segment_fried_parameter(
    total_r0: float,
    number_of_screens: int,
) -> float:
    """
    Convert the Fried parameter of the complete homogeneous
    propagation path into the Fried parameter represented by
    each equal longitudinal segment.
    """

    if total_r0 <= 0.0:
        raise ValueError(
            "total_r0 must be positive."
        )

    if number_of_screens <= 0:
        raise ValueError(
            "number_of_screens must be positive."
        )

    return float(
        total_r0
        * number_of_screens ** (3.0 / 5.0)
    )


# ============================================================
# Reproducible realization seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
) -> list[int]:
    """
    Generate one reproducible seed per atmospheric realization.

    The same realization-seed sequence is reused for every Ns.
    """

    seed_sequence = np.random.SeedSequence(
        SCREEN_CONVERGENCE_SEED
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

def generate_screen_seeds(
    realization_seed: int,
    number_of_screens: int,
) -> list[int]:
    """
    Generate deterministic screen seeds for one atmospheric
    realization.

    Each atmospheric realization has one master seed. From this
    seed, an ordered sequence of independent longitudinal screen
    seeds is generated.

    Therefore, for the same realization, refinements reuse the
    beginning of the same deterministic random hierarchy:

        Ns = 1  -> s0
        Ns = 2  -> s0, s1
        Ns = 4  -> s0, s1, s2, s3
        ...

    The phase screens are not identical between refinements
    because their Fried parameter changes with Ns. The purpose
    of this construction is reproducibility and common random
    numbers, not an exact pointwise nesting of atmospheric
    realizations.
    """

    if number_of_screens <= 0:
        raise ValueError(
            "number_of_screens must be positive."
        )

    realization_sequence = np.random.SeedSequence(
        realization_seed
    )

    screen_sequences = realization_sequence.spawn(
        number_of_screens
    )

    return [
        int(
            sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for sequence in screen_sequences
    ]

# ============================================================
# One atmospheric realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
    number_of_screens: int,
    total_r0: float,
) -> float:
    """
    Propagate one Gaussian beam realization and return the
    final on-axis irradiance.

    Screens are placed at the centers of equal longitudinal
    segments:

        dz/2 -> screen -> dz/2.

    A deterministic hierarchy of screen seeds is derived from
    the realization seed. This implements common random numbers
    across longitudinal refinements while preserving the correct
    Fried parameter for each value of Ns.
    """

    grid, field = (
        create_gaussian_input()
    )

    screen_spacing = (
        TOTAL_PROPAGATION_DISTANCE
        / number_of_screens
    )

    half_screen_spacing = (
        screen_spacing / 2.0
    )

    r0_screen = (
        segment_fried_parameter(
            total_r0=total_r0,
            number_of_screens=number_of_screens,
        )
    )

    screen_seeds = (
        generate_screen_seeds(
            realization_seed=realization_seed,
            number_of_screens=number_of_screens,
        )
    )

    for screen_index in range(
        number_of_screens
    ):

        # First half of the longitudinal segment.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=half_screen_spacing,
            dx=DX,
        )

        # Independent deterministic random stream for this
        # longitudinal screen.
        screen_rng = np.random.default_rng(
            screen_seeds[
                screen_index
            ]
        )

        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                n_subharmonics=(
                    SCREEN_CONVERGENCE_SUBHARMONIC_LEVEL
                ),
                rng=screen_rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        # Second half of the longitudinal segment.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=half_screen_spacing,
            dx=DX,
        )

    center = (
        N_GRID // 2
    )

    return float(
        np.abs(
            field[
                center,
                center,
            ]
        ) ** 2
    )


# ============================================================
# Ensemble for one Ns
# ============================================================

def run_ensemble_for_screen_number(
    number_of_screens: int,
    total_r0: float,
    realization_seeds: list[int],
    number_of_workers: int,
) -> np.ndarray:
    """
    Calculate the final on-axis irradiance for one longitudinal
    discretization.
    """

    worker = partial(
        simulate_one_realization,
        number_of_screens=number_of_screens,
        total_r0=total_r0,
    )

    number_of_realizations = (
        len(realization_seeds)
    )

    intensity_samples = np.zeros(
        number_of_realizations,
        dtype=np.float64,
    )

    r0_screen = (
        segment_fried_parameter(
            total_r0=total_r0,
            number_of_screens=number_of_screens,
        )
    )

    print()
    print(
        f"Ns = {number_of_screens}"
    )

    print(
        f"Delta z = "
        f"{TOTAL_PROPAGATION_DISTANCE / number_of_screens:.4f} m"
    )

    print(
        f"r0 por pantalla = "
        f"{r0_screen:.6f} m"
    )

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            worker,
            realization_seeds,
            chunksize=1,
        )

        for index, intensity in enumerate(
            results
        ):

            intensity_samples[index] = (
                intensity
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
                    f"Ns={number_of_screens}: "
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return intensity_samples


# ============================================================
# Scintillation estimator
# ============================================================

def calculate_scintillation_index(
    intensity_samples: np.ndarray,
) -> float:
    """
    Calculate

        sigma_I^2 = <I^2>/<I>^2 - 1.
    """

    mean_intensity = float(
        np.mean(
            intensity_samples
        )
    )

    mean_squared_intensity = float(
        np.mean(
            intensity_samples**2
        )
    )

    return float(
        mean_squared_intensity
        / mean_intensity**2
        - 1.0
    )


# ============================================================
# Bootstrap
# ============================================================

def calculate_bootstrap_ci(
    intensity_samples: np.ndarray,
    number_of_bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[
    float,
    float,
]:
    """
    Percentile bootstrap confidence interval for the
    scintillation index.
    """

    rng = np.random.default_rng(
        seed
    )

    number_of_realizations = (
        intensity_samples.size
    )

    bootstrap_values = np.empty(
        number_of_bootstrap_samples,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        number_of_bootstrap_samples
    ):

        indices = rng.integers(
            low=0,
            high=number_of_realizations,
            size=number_of_realizations,
        )

        sample = intensity_samples[
            indices
        ]

        bootstrap_values[
            bootstrap_index
        ] = (
            calculate_scintillation_index(
                sample
            )
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower = float(
        np.quantile(
            bootstrap_values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


# ============================================================
# Convergence metrics
# ============================================================

def calculate_convergence_metrics(
    screen_numbers: tuple[int, ...],
    scintillation_values: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate:

        reference error with respect to maximum Ns

    and

        incremental change relative to previous refinement.
    """

    reference_value = (
        scintillation_values[-1]
    )

    reference_error = (
        100.0
        * np.abs(
            scintillation_values
            - reference_value
        )
        / reference_value
    )

    incremental_change = np.full(
        scintillation_values.shape,
        np.nan,
        dtype=np.float64,
    )

    for index in range(
        1,
        len(screen_numbers),
    ):

        current_value = (
            scintillation_values[index]
        )

        previous_value = (
            scintillation_values[
                index - 1
            ]
        )

        incremental_change[index] = (
            100.0
            * abs(
                current_value
                - previous_value
            )
            / current_value
        )

    return (
        reference_error,
        incremental_change,
    )


# ============================================================
# One turbulence regime
# ============================================================

def run_regime(
    regime_name: str,
    total_r0: float,
    number_of_realizations: int,
    number_of_workers: int,
) -> dict:
    """
    Run the complete screen-number convergence experiment for
    one turbulence regime.
    """

    screen_numbers = tuple(
        SCREEN_CONVERGENCE_LEVELS
    )

    realization_seeds = (
        generate_realization_seeds(
            number_of_realizations
        )
    )

    scintillation_values = np.zeros(
        len(screen_numbers),
        dtype=np.float64,
    )

    ci_lower = np.zeros_like(
        scintillation_values
    )

    ci_upper = np.zeros_like(
        scintillation_values
    )

    raw_samples = {}

    print()
    print(
        f"Régimen: {regime_name}"
    )

    print(
        "=" * (
            len(regime_name) + 9
        )
    )

    for index, number_of_screens in enumerate(
        screen_numbers
    ):

        intensity_samples = (
            run_ensemble_for_screen_number(
                number_of_screens=(
                    number_of_screens
                ),
                total_r0=total_r0,
                realization_seeds=(
                    realization_seeds
                ),
                number_of_workers=(
                    number_of_workers
                ),
            )
        )

        raw_samples[
            number_of_screens
        ] = intensity_samples

        scintillation_values[index] = (
            calculate_scintillation_index(
                intensity_samples
            )
        )

        (
            ci_lower[index],
            ci_upper[index],
        ) = calculate_bootstrap_ci(
            intensity_samples=(
                intensity_samples
            ),
            number_of_bootstrap_samples=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SAMPLES
            ),
            confidence_level=(
                SCREEN_CONVERGENCE_BOOTSTRAP_CONFIDENCE_LEVEL
            ),
            seed=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SEED
                + number_of_screens
            ),
        )

    (
        reference_error,
        incremental_change,
    ) = calculate_convergence_metrics(
        screen_numbers=screen_numbers,
        scintillation_values=(
            scintillation_values
        ),
    )

    return {
        "regime":
            regime_name,
        "total_r0":
            total_r0,
        "screen_numbers":
            screen_numbers,
        "scintillation":
            scintillation_values,
        "ci_lower":
            ci_lower,
        "ci_upper":
            ci_upper,
        "reference_error":
            reference_error,
        "incremental_change":
            incremental_change,
        "raw_samples":
            raw_samples,
    }


def load_raw_samples(
    regime_name: str,
    number_of_screens: int,
) -> np.ndarray:
    """
    Load previously saved on-axis irradiance samples for one
    turbulence regime and one value of Ns.
    """

    filename = (
        OUTPUT_DIRECTORY
        / regime_name
        / f"intensity_Ns_{number_of_screens}.csv"
    )

    if not filename.exists():
        raise FileNotFoundError(
            f"Saved samples not found: {filename}"
        )

    samples = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )

    samples = np.atleast_1d(
        samples
    )

    if samples.ndim != 1:
        raise ValueError(
            f"Unexpected data shape in {filename}."
        )

    if not np.all(
        np.isfinite(samples)
    ):
        raise ValueError(
            f"Non-finite values found in {filename}."
        )

    return samples

# ============================================================
# Save raw samples
# ============================================================

def save_raw_samples(
    result: dict,
) -> None:
    """
    Save final on-axis irradiance samples for each Ns.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    regime_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        number_of_screens,
        samples,
    ) in result["raw_samples"].items():

        filename = (
            regime_directory
            / f"intensity_Ns_{number_of_screens}.csv"
        )

        np.savetxt(
            filename,
            samples,
            delimiter=",",
            header="on_axis_intensity",
            comments="",
        )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    result: dict,
) -> None:
    """
    Save convergence statistics.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    regime_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        regime_directory
        / "screen_convergence_summary.csv"
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
                "Ns",
                "delta_z_m",
                "r0_screen_m",
                "scintillation",
                "ci95_lower",
                "ci95_upper",
                "reference_error_percent",
                "incremental_change_percent",
            ]
        )

        for index, number_of_screens in enumerate(
            result["screen_numbers"]
        ):

            writer.writerow(
                [
                    number_of_screens,
                    (
                        TOTAL_PROPAGATION_DISTANCE
                        / number_of_screens
                    ),
                    segment_fried_parameter(
                        total_r0=(
                            result["total_r0"]
                        ),
                        number_of_screens=(
                            number_of_screens
                        ),
                    ),
                    result["scintillation"][
                        index
                    ],
                    result["ci_lower"][
                        index
                    ],
                    result["ci_upper"][
                        index
                    ],
                    result["reference_error"][
                        index
                    ],
                    result[
                        "incremental_change"
                    ][index],
                ]
            )


# ============================================================
# Terminal table
# ============================================================

def print_summary(
    result: dict,
) -> None:
    """
    Print screen-convergence results.
    """

    print()
    print(
        f"Convergencia longitudinal: "
        f"{result['regime']}"
    )

    print(
        "=" * 54
    )

    header = (
        f"{'Ns':>5}"
        f"{'sigma_I^2':>14}"
        f"{'IC95 inf':>14}"
        f"{'IC95 sup':>14}"
        f"{'err ref [%]':>14}"
        f"{'Delta Ns [%]':>15}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for index, number_of_screens in enumerate(
        result["screen_numbers"]
    ):

        incremental = (
            result[
                "incremental_change"
            ][index]
        )

        incremental_text = (
            "-"
            if not np.isfinite(
                incremental
            )
            else f"{incremental:.4f}"
        )

        print(
            f"{number_of_screens:5d}"
            f"{result['scintillation'][index]:14.6e}"
            f"{result['ci_lower'][index]:14.6e}"
            f"{result['ci_upper'][index]:14.6e}"
            f"{result['reference_error'][index]:14.4f}"
            f"{incremental_text:>15}"
        )


# ============================================================
# Plot scintillation convergence
# ============================================================

def plot_scintillation_convergence(
    result: dict,
) -> None:
    """
    Plot scintillation versus number of phase screens with
    bootstrap confidence intervals.
    """

    screen_numbers = np.asarray(
        result["screen_numbers"]
    )

    scintillation = (
        result["scintillation"]
    )

    lower_error = (
        scintillation
        - result["ci_lower"]
    )

    upper_error = (
        result["ci_upper"]
        - scintillation
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.errorbar(
        screen_numbers,
        scintillation,
        yerr=np.vstack(
            (
                lower_error,
                upper_error,
            )
        ),
        marker="o",
        capsize=4,
        linewidth=1.6,
    )

    axis.axhline(
        scintillation[-1],
        linestyle="--",
        linewidth=1.4,
        label=(
            rf"Referencia $N_s="
            rf"{screen_numbers[-1]}$"
        ),
    )

    axis.set_xlabel(
        "Número de pantallas de fase $N_s$"
    )

    axis.set_ylabel(
        r"Índice de centelleo "
        r"$\sigma_I^2(0,L)$"
    )

    axis.set_title(
        f"Convergencia longitudinal: "
        f"turbulencia {result['regime']}"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    figure.savefig(
        regime_directory
        / "scintillation_vs_phase_screens.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Plot convergence errors
# ============================================================

def plot_convergence_metrics(
    result: dict,
) -> None:
    """
    Plot reference error and incremental change.
    """

    screen_numbers = np.asarray(
        result["screen_numbers"]
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        screen_numbers,
        result["reference_error"],
        marker="o",
        linewidth=1.6,
        label="Error respecto a la referencia",
    )

    axis.plot(
        screen_numbers[1:],
        result["incremental_change"][1:],
        marker="s",
        linewidth=1.6,
        label="Cambio entre refinamientos",
    )

    axis.set_xlabel(
        "Número de pantallas de fase $N_s$"
    )

    axis.set_ylabel(
        "Cambio relativo [%]"
    )

    axis.set_title(
        f"Indicadores de convergencia: "
        f"turbulencia {result['regime']}"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    figure.savefig(
        regime_directory
        / "screen_convergence_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--realizations",
        type=int,
        default=(
            SCREEN_CONVERGENCE_NUMBER_OF_REALIZATIONS
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUMBER_OF_WORKERS,
    )

    parser.add_argument(
        "--regime",
        choices=(
            "moderate",
            "strong",
            "both",
        ),
        default="both",
    )

    parser.add_argument(
        "--extend-to",
        type=int,
        default=None,
        help=(
            "Extend previously saved ensembles to the requested "
            "number of realizations without recomputing existing samples."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Execute the longitudinal convergence experiment.

    The experiment can either start a new ensemble or extend
    previously saved ensembles without recomputing existing
    atmospheric realizations.
    """

    arguments = (
        parse_arguments()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = []

    if arguments.regime in (
        "moderate",
        "both",
    ):
        configurations.append(
            (
                "moderada",
                MODERATE_R0_TOTAL,
            )
        )

    if arguments.regime in (
        "strong",
        "both",
    ):
        configurations.append(
            (
                "fuerte",
                STRONG_R0_TOTAL,
            )
        )

    print(
        "Convergencia longitudinal del método Split-Step"
    )

    print(
        "=============================================="
    )

    if arguments.extend_to is None:

        print(
            f"Realizaciones: "
            f"{arguments.realizations}"
        )

    else:

        print(
            f"Extender ensamble hasta: "
            f"{arguments.extend_to}"
        )

    print(
        f"Workers: "
        f"{arguments.workers}"
    )

    print(
        f"Ns: "
        f"{SCREEN_CONVERGENCE_LEVELS}"
    )

    print(
        f"Subarmónicos: "
        f"{SCREEN_CONVERGENCE_SUBHARMONIC_LEVEL}"
    )

    for (
        regime_name,
        total_r0,
    ) in configurations:

        # ----------------------------------------------------
        # Run a new ensemble or extend an existing one
        # ----------------------------------------------------

        if arguments.extend_to is not None:

            result = extend_regime(
                regime_name=regime_name,
                total_r0=total_r0,
                target_size=(
                    arguments.extend_to
                ),
                number_of_workers=(
                    arguments.workers
                ),
            )

        else:

            result = run_regime(
                regime_name=regime_name,
                total_r0=total_r0,
                number_of_realizations=(
                    arguments.realizations
                ),
                number_of_workers=(
                    arguments.workers
                ),
            )

        # ----------------------------------------------------
        # Consecutive-refinement bootstrap statistics
        # ----------------------------------------------------

        (
            difference,
            difference_lower,
            difference_upper,
            difference_contains_zero,
        ) = calculate_refinement_difference_statistics(
            result
        )

        result["refinement_difference"] = (
            difference
        )

        result["refinement_difference_lower"] = (
            difference_lower
        )

        result["refinement_difference_upper"] = (
            difference_upper
        )

        result[
            "refinement_difference_contains_zero"
        ] = (
            difference_contains_zero
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        save_raw_samples(
            result
        )

        save_summary(
            result
        )

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print_summary(
            result
        )

        print_refinement_difference_summary(
            result
        )

        # ----------------------------------------------------
        # Figures
        # ----------------------------------------------------

        plot_scintillation_convergence(
            result
        )

        plot_convergence_metrics(
            result
        )   

def extend_regime(
    regime_name: str,
    total_r0: float,
    target_size: int,
    number_of_workers: int,
) -> dict:
    """
    Extend all saved Ns ensembles for one turbulence regime
    without recomputing existing realizations.
    """

    screen_numbers = tuple(
        SCREEN_CONVERGENCE_LEVELS
    )

    # --------------------------------------------------------
    # Determine current ensemble size
    # --------------------------------------------------------

    existing_samples = {}

    current_sizes = []

    for number_of_screens in screen_numbers:

        samples = load_raw_samples(
            regime_name=regime_name,
            number_of_screens=number_of_screens,
        )

        existing_samples[
            number_of_screens
        ] = samples

        current_sizes.append(
            samples.size
        )

    if len(set(current_sizes)) != 1:
        raise RuntimeError(
            "Saved Ns ensembles do not all have the same size."
        )

    current_size = (
        current_sizes[0]
    )

    if target_size <= current_size:
        raise ValueError(
            f"target_size={target_size} must be greater than "
            f"the existing size {current_size}."
        )

    additional_number = (
        target_size
        - current_size
    )

    print()
    print(
        f"Extensión del régimen: {regime_name}"
    )
    print(
        "=" * (
            len(regime_name) + 22
        )
    )
    print(
        f"Ensamble existente: {current_size}"
    )
    print(
        f"Nuevas realizaciones: {additional_number}"
    )
    print(
        f"Ensamble final: {target_size}"
    )

    # --------------------------------------------------------
    # Deterministic complete seed sequence
    # --------------------------------------------------------

    all_realization_seeds = (
        generate_realization_seeds(
            target_size
        )
    )

    new_realization_seeds = (
        all_realization_seeds[
            current_size:
            target_size
        ]
    )

    # --------------------------------------------------------
    # Extend every Ns
    # --------------------------------------------------------

    raw_samples = {}

    for number_of_screens in screen_numbers:

        new_samples = (
            run_ensemble_for_screen_number(
                number_of_screens=(
                    number_of_screens
                ),
                total_r0=total_r0,
                realization_seeds=(
                    new_realization_seeds
                ),
                number_of_workers=(
                    number_of_workers
                ),
            )
        )

        combined = np.concatenate(
            (
                existing_samples[
                    number_of_screens
                ],
                new_samples,
            )
        )

        raw_samples[
            number_of_screens
        ] = combined

    # --------------------------------------------------------
    # Recompute statistics using the full extended ensemble
    # --------------------------------------------------------

    scintillation_values = np.zeros(
        len(screen_numbers),
        dtype=np.float64,
    )

    ci_lower = np.zeros_like(
        scintillation_values
    )

    ci_upper = np.zeros_like(
        scintillation_values
    )

    for index, number_of_screens in enumerate(
        screen_numbers
    ):

        samples = raw_samples[
            number_of_screens
        ]

        scintillation_values[index] = (
            calculate_scintillation_index(
                samples
            )
        )

        (
            ci_lower[index],
            ci_upper[index],
        ) = calculate_bootstrap_ci(
            intensity_samples=samples,
            number_of_bootstrap_samples=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SAMPLES
            ),
            confidence_level=(
                SCREEN_CONVERGENCE_BOOTSTRAP_CONFIDENCE_LEVEL
            ),
            seed=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SEED
                + number_of_screens
            ),
        )

    (
        reference_error,
        incremental_change,
    ) = calculate_convergence_metrics(
        screen_numbers=screen_numbers,
        scintillation_values=(
            scintillation_values
        ),
    )

    return {
        "regime":
            regime_name,
        "total_r0":
            total_r0,
        "screen_numbers":
            screen_numbers,
        "scintillation":
            scintillation_values,
        "ci_lower":
            ci_lower,
        "ci_upper":
            ci_upper,
        "reference_error":
            reference_error,
        "incremental_change":
            incremental_change,
        "raw_samples":
            raw_samples,
    }


def calculate_paired_bootstrap_difference_ci(
    previous_samples: np.ndarray,
    current_samples: np.ndarray,
    number_of_bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Estimate the difference in scintillation between two
    consecutive longitudinal refinements:

        D_Ns =
            sigma_I^2(Ns)
            - sigma_I^2(Ns/2)

    together with a paired percentile-bootstrap confidence
    interval.

    The same realization indices are resampled in both
    ensembles, preserving the common-random-number pairing.
    """

    if (
        previous_samples.shape
        != current_samples.shape
    ):
        raise ValueError(
            "Paired ensembles must have identical shapes."
        )

    number_of_realizations = (
        current_samples.size
    )

    previous_scintillation = (
        calculate_scintillation_index(
            previous_samples
        )
    )

    current_scintillation = (
        calculate_scintillation_index(
            current_samples
        )
    )

    observed_difference = (
        current_scintillation
        - previous_scintillation
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_differences = np.empty(
        number_of_bootstrap_samples,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        number_of_bootstrap_samples
    ):

        indices = rng.integers(
            low=0,
            high=number_of_realizations,
            size=number_of_realizations,
        )

        previous_bootstrap = (
            previous_samples[
                indices
            ]
        )

        current_bootstrap = (
            current_samples[
                indices
            ]
        )

        bootstrap_differences[
            bootstrap_index
        ] = (
            calculate_scintillation_index(
                current_bootstrap
            )
            - calculate_scintillation_index(
                previous_bootstrap
            )
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower = float(
        np.quantile(
            bootstrap_differences,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_differences,
            1.0 - alpha / 2.0,
        )
    )

    return (
        float(observed_difference),
        lower,
        upper,
    )


def calculate_refinement_difference_statistics(
    result: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate paired-bootstrap statistics for consecutive
    longitudinal refinements.
    """

    screen_numbers = (
        result["screen_numbers"]
    )

    number_of_levels = (
        len(screen_numbers)
    )

    differences = np.full(
        number_of_levels,
        np.nan,
        dtype=np.float64,
    )

    lower = np.full_like(
        differences,
        np.nan,
    )

    upper = np.full_like(
        differences,
        np.nan,
    )

    contains_zero = np.zeros(
        number_of_levels,
        dtype=bool,
    )

    for index in range(
        1,
        number_of_levels,
    ):

        previous_ns = (
            screen_numbers[
                index - 1
            ]
        )

        current_ns = (
            screen_numbers[
                index
            ]
        )

        (
            differences[index],
            lower[index],
            upper[index],
        ) = calculate_paired_bootstrap_difference_ci(
            previous_samples=(
                result["raw_samples"][
                    previous_ns
                ]
            ),
            current_samples=(
                result["raw_samples"][
                    current_ns
                ]
            ),
            number_of_bootstrap_samples=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SAMPLES
            ),
            confidence_level=(
                SCREEN_CONVERGENCE_BOOTSTRAP_CONFIDENCE_LEVEL
            ),
            seed=(
                SCREEN_CONVERGENCE_BOOTSTRAP_SEED
                + 10_000
                + current_ns
            ),
        )

        contains_zero[index] = (
            lower[index] <= 0.0
            <= upper[index]
        )

    return (
        differences,
        lower,
        upper,
        contains_zero,
    )

def print_refinement_difference_summary(
    result: dict,
) -> None:
    """
    Print paired-bootstrap differences between consecutive
    longitudinal refinements.

    For each transition Ns/2 -> Ns, the reported quantity is

        D_Ns = sigma_I^2(Ns) - sigma_I^2(Ns/2).

    The confidence interval is obtained by paired bootstrap
    resampling of the common realization indices.
    """

    print()

    print(
        "Diferencia entre refinamientos consecutivos"
    )

    print(
        "==========================================="
    )

    header = (
        f"{'Refinamiento':>16}"
        f"{'D_Ns':>14}"
        f"{'IC95 inf':>14}"
        f"{'IC95 sup':>14}"
        f"{'Incluye 0':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    screen_numbers = (
        result["screen_numbers"]
    )

    for index in range(
        1,
        len(screen_numbers),
    ):

        previous_ns = (
            screen_numbers[
                index - 1
            ]
        )

        current_ns = (
            screen_numbers[
                index
            ]
        )

        contains_zero = (
            result[
                "refinement_difference_contains_zero"
            ][index]
        )

        zero_text = (
            "sí"
            if contains_zero
            else "no"
        )

        print(
            f"{previous_ns:>6d}"
            f" -> {current_ns:<6d}"
            f"{result['refinement_difference'][index]:14.6e}"
            f"{result['refinement_difference_lower'][index]:14.6e}"
            f"{result['refinement_difference_upper'][index]:14.6e}"
            f"{zero_text:>12}"
        )

if __name__ == "__main__":
    main()
