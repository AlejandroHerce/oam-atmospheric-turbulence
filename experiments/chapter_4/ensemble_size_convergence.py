"""
Statistical convergence of the turbulent-propagation ensemble.

A LG_0^3 beam is propagated through strong Kolmogorov turbulence
over a total distance of 1000 m using 16 phase screens located at
the centers of equal propagation segments.

Each independent atmospheric realization returns two OAM metrics:

    - transmitted-mode retention P_l0
    - RMS OAM spread sigma_Delta_l

A single maximum ensemble is generated. Smaller ensemble sizes are
constructed as cumulative subsets of this same realization sequence.
"""

import argparse
import csv
import os

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_2 import (
    W0_LG,
)

from configs.chapter_4 import (
    DX,
    ENSEMBLE_BEAM_CHARGE,
    ENSEMBLE_CHECKPOINTS,
    ENSEMBLE_REFERENCE_SIZE,
    ENSEMBLE_R0_SCREEN,
    ENSEMBLE_SEED,
    ENSEMBLE_SUBHARMONIC_LEVEL,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    OAM_ELL_MAX,
    OAM_ELL_MIN,
    SCREEN_SPACING,
    STRONG_R0_TOTAL,
    TOTAL_PROPAGATION_DISTANCE,
    WAVELENGTH,
)

from src.beams import (
    laguerre_gaussian_beam,
)

from src.grids import (
    create_grid,
)

from src.oam import (
    calculate_oam_spectrum,
    calculate_rms_oam_spread,
    modal_power_at_charge,
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
    8,
    os.cpu_count() or 1,
)

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/ensemble_size_convergence"
)


# ============================================================
# Input beam
# ============================================================

def create_input_beam():
    """
    Create the LG_0^3 input field used for the
    ensemble-convergence experiment.
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

    field = laguerre_gaussian_beam(
        grid=grid,
        w0=W0_LG,
        charge=ENSEMBLE_BEAM_CHARGE,
    )

    return (
        grid,
        field,
    )


# ============================================================
# One atmospheric realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
) -> tuple[float, float]:
    """
    Propagate one independent atmospheric realization.

    Phase screens are located at the centers of equal propagation
    segments:

        dz/2 -> screen -> dz -> ... -> screen -> dz/2.

    Returns
    -------
    retention:
        Fraction of OAM power remaining in the transmitted mode.

    rms_spread:
        RMS OAM spread relative to the transmitted charge.
    """

    rng = np.random.default_rng(
        realization_seed
    )

    grid, field = create_input_beam()

    # --------------------------------------------------------
    # Propagation to the first screen
    # --------------------------------------------------------

    field = angular_spectrum_propagation(
        field=field,
        wavelength=WAVELENGTH,
        distance=(
            HALF_SCREEN_SPACING
        ),
        dx=DX,
    )

    # --------------------------------------------------------
    # Turbulent path
    # --------------------------------------------------------

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):
        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=ENSEMBLE_R0_SCREEN,
                n_subharmonics=(
                    ENSEMBLE_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        if (
            screen_index
            < NUMBER_OF_PHASE_SCREENS - 1
        ):
            field = angular_spectrum_propagation(
                field=field,
                wavelength=WAVELENGTH,
                distance=(
                    SCREEN_SPACING
                ),
                dx=DX,
            )

    # --------------------------------------------------------
    # Propagation from the final screen to the receiver
    # --------------------------------------------------------

    field = angular_spectrum_propagation(
        field=field,
        wavelength=WAVELENGTH,
        distance=(
            HALF_SCREEN_SPACING
        ),
        dx=DX,
    )

    # --------------------------------------------------------
    # OAM analysis
    # --------------------------------------------------------

    (
        ell_values,
        modal_power,
    ) = calculate_oam_spectrum(
        field=field,
        grid=grid,
        ell_min=OAM_ELL_MIN,
        ell_max=OAM_ELL_MAX,
    )

    retention = modal_power_at_charge(
        ell_values=ell_values,
        modal_power=modal_power,
        charge=ENSEMBLE_BEAM_CHARGE,
    )

    rms_spread = calculate_rms_oam_spread(
        ell_values=ell_values,
        modal_power=modal_power,
        transmitted_charge=(
            ENSEMBLE_BEAM_CHARGE
        ),
    )

    return (
        retention,
        rms_spread,
    )


def generate_realization_seeds(
    total_number_of_realizations: int,
) -> list[int]:
    """
    Generate the deterministic realization-seed sequence used by
    the ensemble experiment.
    """

    if total_number_of_realizations <= 0:
        raise ValueError(
            "total_number_of_realizations must be positive."
        )

    seed_sequence = np.random.SeedSequence(
        ENSEMBLE_SEED
    )

    child_sequences = seed_sequence.spawn(
        total_number_of_realizations
    )

    realization_seeds = [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in child_sequences
    ]

    return realization_seeds


# ============================================================
# Maximum ensemble
# ============================================================

def run_maximum_ensemble(
    number_of_realizations: int,
    number_of_workers: int,
    start_index: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Execute a deterministic subset of atmospheric realizations.

    start_index is zero-based within the complete ensemble seed sequence.
    """

    total_required = (
        start_index
        + number_of_realizations
    )

    realization_seeds = generate_realization_seeds(
        total_number_of_realizations=total_required,
    )[
        start_index:
        total_required
    ]

    retention = np.zeros(
        number_of_realizations,
        dtype=np.float64,
    )

    rms_spread = np.zeros(
        number_of_realizations,
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

        for index, (
            current_retention,
            current_spread,
        ) in enumerate(results):

            retention[index] = current_retention
            rms_spread[index] = current_spread

            completed = index + 1

            if (
                completed == 1
                or completed % 25 == 0
                or completed == number_of_realizations
            ):
                print(
                    f"Realizaciones adicionales completadas: "
                    f"{completed}/{number_of_realizations}"
                )

    return (
        retention,
        rms_spread,
    )


# ============================================================
# Cumulative statistics
# ============================================================

def calculate_metric_statistics(
    values: np.ndarray,
    checkpoints: tuple[int, ...],
) -> list[dict[str, float | int]]:
    """
    Calculate cumulative mean, standard deviation, SEM,
    relative SEM, and error relative to the maximum ensemble.
    """

    reference_mean = float(
        np.mean(values)
    )

    results = []

    for ensemble_size in checkpoints:
        subset = (
            values[:ensemble_size]
        )

        mean = float(
            np.mean(subset)
        )

        standard_deviation = float(
            np.std(
                subset,
                ddof=1,
            )
        ) if ensemble_size > 1 else 0.0

        sem = (
            standard_deviation
            / np.sqrt(
                ensemble_size
            )
        )

        relative_sem = (
            100.0
            * sem
            / abs(mean)
        ) if mean != 0.0 else np.nan

        reference_error = (
            100.0
            * abs(
                mean
                - reference_mean
            )
            / abs(
                reference_mean
            )
        ) if reference_mean != 0.0 else np.nan

        results.append(
            {
                "N": ensemble_size,
                "mean": mean,
                "std": standard_deviation,
                "sem": sem,
                "relative_sem_percent":
                    relative_sem,
                "reference_error_percent":
                    reference_error,
            }
        )

    return results


# ============================================================
# Output
# ============================================================

def print_convergence_table(
    retention_results,
    spread_results,
) -> None:
    """
    Print the convergence table.
    """

    print()
    print(
        "Convergencia estadística del ensamble"
    )
    print(
        "====================================="
    )

    header = (
        f"{'Nens':>6}"
        f"{'<P_l0>':>12}"
        f"{'err P [%]':>12}"
        f"{'SEMrel P [%]':>14}"
        f"{'<sigma>':>12}"
        f"{'err sig [%]':>14}"
        f"{'SEMrel sig [%]':>16}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for retention_row, spread_row in zip(
        retention_results,
        spread_results,
    ):
        print(
            f"{int(retention_row['N']):>6d}"
            f"{float(retention_row['mean']):>12.6f}"
            f"{float(retention_row['reference_error_percent']):>12.4f}"
            f"{float(retention_row['relative_sem_percent']):>14.4f}"
            f"{float(spread_row['mean']):>12.6f}"
            f"{float(spread_row['reference_error_percent']):>14.4f}"
            f"{float(spread_row['relative_sem_percent']):>16.4f}"
        )


def save_raw_results(
    retention: np.ndarray,
    rms_spread: np.ndarray,
) -> None:
    """
    Save per-realization metrics.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        OUTPUT_DIRECTORY
        / "ensemble_realizations.csv"
    )

    with filename.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "realization",
                "mode_retention",
                "rms_oam_spread",
            ]
        )

        for index, (
            retention_value,
            spread_value,
        ) in enumerate(
            zip(
                retention,
                rms_spread,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    retention_value,
                    spread_value,
                ]
            )


def save_convergence_results(
    retention_results,
    spread_results,
) -> None:
    """
    Save cumulative statistics.
    """

    filename = (
        OUTPUT_DIRECTORY
        / "ensemble_convergence.csv"
    )

    with filename.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "ensemble_size",
                "retention_mean",
                "retention_std",
                "retention_sem",
                "retention_relative_sem_percent",
                "retention_reference_error_percent",
                "spread_mean",
                "spread_std",
                "spread_sem",
                "spread_relative_sem_percent",
                "spread_reference_error_percent",
            ]
        )

        for retention_row, spread_row in zip(
            retention_results,
            spread_results,
        ):
            writer.writerow(
                [
                    retention_row["N"],
                    retention_row["mean"],
                    retention_row["std"],
                    retention_row["sem"],
                    retention_row[
                        "relative_sem_percent"
                    ],
                    retention_row[
                        "reference_error_percent"
                    ],
                    spread_row["mean"],
                    spread_row["std"],
                    spread_row["sem"],
                    spread_row[
                        "relative_sem_percent"
                    ],
                    spread_row[
                        "reference_error_percent"
                    ],
                ]
            )


# ============================================================
# Plots
# ============================================================

def plot_cumulative_means(
    retention: np.ndarray,
    rms_spread: np.ndarray,
) -> None:
    """
    Plot cumulative means using linear scales.
    """

    numbers = np.arange(
        1,
        retention.size + 1,
    )

    retention_mean = (
        np.cumsum(retention)
        / numbers
    )

    spread_mean = (
        np.cumsum(rms_spread)
        / numbers
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        numbers,
        retention_mean,
        linewidth=1.5,
    )

    axis.set_xlabel(
        "Número de realizaciones"
    )

    axis.set_ylabel(
        r"Retención media "
        r"$\overline{P}_{\ell_0}$"
    )

    axis.set_title(
        "Convergencia de la retención del modo transmitido"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "cumulative_mode_retention.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        numbers,
        spread_mean,
        linewidth=1.5,
    )

    axis.set_xlabel(
        "Número de realizaciones"
    )

    axis.set_ylabel(
        r"Dispersión RMS media "
        r"$\overline{\sigma}_{\Delta\ell}$"
    )

    axis.set_title(
        "Convergencia de la dispersión del espectro OAM"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "cumulative_oam_spread.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_checkpoint_statistics(
    retention_results,
    spread_results,
) -> None:
    """
    Plot reference error and relative SEM.
    """

    ensemble_sizes = np.array(
        [
            row["N"]
            for row in retention_results
        ]
    )

    # --------------------------------------------------------
    # Reference error
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        ensemble_sizes,
        [
            row["reference_error_percent"]
            for row in retention_results
        ],
        marker="o",
        label=r"$P_{\ell_0}$",
    )

    axis.plot(
        ensemble_sizes,
        [
            row["reference_error_percent"]
            for row in spread_results
        ],
        marker="s",
        label=r"$\sigma_{\Delta\ell}$",
    )

    axis.set_xlabel(
        "Tamaño del ensamble"
    )

    axis.set_ylabel(
        "Error respecto al ensamble máximo [%]"
    )

    axis.set_title(
        "Convergencia respecto al ensamble de referencia"
    )

    axis.set_xticks(
        ensemble_sizes
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "reference_error_vs_ensemble_size.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # Relative SEM
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        ensemble_sizes,
        [
            row["relative_sem_percent"]
            for row in retention_results
        ],
        marker="o",
        label=r"$P_{\ell_0}$",
    )

    axis.plot(
        ensemble_sizes,
        [
            row["relative_sem_percent"]
            for row in spread_results
        ],
        marker="s",
        label=r"$\sigma_{\Delta\ell}$",
    )

    axis.set_xlabel(
        "Tamaño del ensamble"
    )

    axis.set_ylabel(
        "SEM relativo [%]"
    )

    axis.set_title(
        "Incertidumbre estadística de las métricas"
    )

    axis.set_xticks(
        ensemble_sizes
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "relative_sem_vs_ensemble_size.png",
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--realizations",
        type=int,
        default=ENSEMBLE_REFERENCE_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUMBER_OF_WORKERS,
    )

    parser.add_argument(
        "--extend-to",
        type=int,
        default=None,
        help=(
            "Extend the saved ensemble to the requested total "
            "number of realizations without recomputing existing ones."
        ),
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Load the saved ensemble_realizations.csv and "
            "recompute convergence statistics and figures "
            "without running new simulations."
        ),
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
    Execute the ensemble convergence experiment.
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

    print(
        "Convergencia estadística del ensamble"
    )
    print(
        "====================================="
    )

    print(
        f"Haz: LG_0^{ENSEMBLE_BEAM_CHARGE}"
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
        f"Separación entre centros: "
        f"{SCREEN_SPACING:.2f} m"
    )

    print(
        f"r0 total: "
        f"{STRONG_R0_TOTAL:.6f} m"
    )

    print(
        f"r0 por pantalla: "
        f"{ENSEMBLE_R0_SCREEN:.6f} m"
    )

    print(
        f"Subarmónicos: "
        f"{ENSEMBLE_SUBHARMONIC_LEVEL}"
    )

    print(
        f"Rango OAM: "
        f"[{OAM_ELL_MIN}, {OAM_ELL_MAX}]"
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

    (
        retention,
        rms_spread,
    ) = run_maximum_ensemble(
        number_of_realizations=(
            number_of_realizations
        ),
        number_of_workers=(
            number_of_workers
        ),
    )

    save_raw_results(
        retention=retention,
        rms_spread=rms_spread,
    )

    checkpoints = tuple(
        value
        for value in ENSEMBLE_CHECKPOINTS
        if value <= number_of_realizations
    )

    if (
        number_of_realizations
        not in checkpoints
    ):
        checkpoints = (
            *checkpoints,
            number_of_realizations,
        )

    retention_results = (
        calculate_metric_statistics(
            values=retention,
            checkpoints=checkpoints,
        )
    )

    spread_results = (
        calculate_metric_statistics(
            values=rms_spread,
            checkpoints=checkpoints,
        )
    )

    print_convergence_table(
        retention_results=(
            retention_results
        ),
        spread_results=(
            spread_results
        ),
    )

    save_convergence_results(
        retention_results=(
            retention_results
        ),
        spread_results=(
            spread_results
        ),
    )

    plot_cumulative_means(
        retention=retention,
        rms_spread=rms_spread,
    )

    plot_checkpoint_statistics(
        retention_results=(
            retention_results
        ),
        spread_results=(
            spread_results
        ),
    )

    print(
        "\nResultados guardados en:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


def generate_realization_seeds(
    total_number_of_realizations: int,
) -> list[int]:
    """
    Generate the deterministic realization-seed sequence used by
    the ensemble experiment.
    """

    seed_sequence = np.random.SeedSequence(
        ENSEMBLE_SEED
    )

    child_sequences = seed_sequence.spawn(
        total_number_of_realizations
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

def load_raw_results(
    filename: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load previously computed per-realization metrics.
    """

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
    )

    retention = np.asarray(
        data["mode_retention"],
        dtype=np.float64,
    )

    rms_spread = np.asarray(
        data["rms_oam_spread"],
        dtype=np.float64,
    )

    return (
        retention,
        rms_spread,
    )


def analyze_saved_ensemble() -> None:
    """
    Recompute convergence statistics and figures from the
    saved per-realization results without running propagation.
    """

    raw_file = (
        OUTPUT_DIRECTORY
        / "ensemble_realizations.csv"
    )

    if not raw_file.exists():
        raise FileNotFoundError(
            "No saved ensemble_realizations.csv was found."
        )

    retention, rms_spread = (
        load_raw_results(
            raw_file
        )
    )

    total_realizations = (
        retention.size
    )

    print()
    print(
        "Análisis del ensamble guardado"
    )
    print(
        "=============================="
    )
    print(
        f"Realizaciones disponibles: "
        f"{total_realizations}"
    )

    base_checkpoints = (
        25,
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        750,
        1000,
        1250,
        1500,
    )

    checkpoints = tuple(
        checkpoint
        for checkpoint in base_checkpoints
        if checkpoint < total_realizations
    )

    checkpoints = (
        *checkpoints,
        total_realizations,
    )

    retention_results = (
        calculate_metric_statistics(
            values=retention,
            checkpoints=checkpoints,
        )
    )

    spread_results = (
        calculate_metric_statistics(
            values=rms_spread,
            checkpoints=checkpoints,
        )
    )

    print_convergence_table(
        retention_results,
        spread_results,
    )

    save_convergence_results(
        retention_results,
        spread_results,
    )

    plot_cumulative_means(
        retention,
        rms_spread,
    )

    plot_checkpoint_statistics(
        retention_results,
        spread_results,
    )

    print()
    print(
        "Tabla y figuras actualizadas."
    )

def extend_existing_ensemble(
    target_size: int,
    number_of_workers: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Extend an existing saved ensemble without recomputing
    previously completed realizations.
    """

    raw_file = (
        OUTPUT_DIRECTORY
        / "ensemble_realizations.csv"
    )

    if not raw_file.exists():
        raise FileNotFoundError(
            "No previous ensemble_realizations.csv was found."
        )

    old_retention, old_spread = (
        load_raw_results(
            raw_file
        )
    )

    current_size = (
        old_retention.size
    )

    if target_size <= current_size:
        raise ValueError(
            f"target_size={target_size} must exceed "
            f"the existing ensemble size {current_size}."
        )

    additional_number = (
        target_size
        - current_size
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

    new_retention, new_spread = (
        run_maximum_ensemble(
            number_of_realizations=(
                additional_number
            ),
            number_of_workers=(
                number_of_workers
            ),
            start_index=current_size,
        )
    )

    retention = np.concatenate(
        (
            old_retention,
            new_retention,
        )
    )

    rms_spread = np.concatenate(
        (
            old_spread,
            new_spread,
        )
    )

    return (
        retention,
        rms_spread,
    )


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.analyze_only:
        analyze_saved_ensemble()

    elif arguments.extend_to is not None:
        retention, rms_spread = (
            extend_existing_ensemble(
                target_size=arguments.extend_to,
                number_of_workers=arguments.workers,
            )
        )

        save_raw_results(
            retention=retention,
            rms_spread=rms_spread,
        )

        total_realizations = (
            retention.size
        )

        base_checkpoints = (
            25,
            50,
            100,
            150,
            200,
            300,
            400,
            500,
            750,
            1000,
            1250,
            1500,
        )

        checkpoints = tuple(
            checkpoint
            for checkpoint in base_checkpoints
            if checkpoint < total_realizations
        )

        checkpoints = (
            *checkpoints,
            total_realizations,
        )

        retention_results = (
            calculate_metric_statistics(
                values=retention,
                checkpoints=checkpoints,
            )
        )

        spread_results = (
            calculate_metric_statistics(
                values=rms_spread,
                checkpoints=checkpoints,
            )
        )

        print_convergence_table(
            retention_results,
            spread_results,
        )

        save_convergence_results(
            retention_results,
            spread_results,
        )

        plot_cumulative_means(
            retention,
            rms_spread,
        )

        plot_checkpoint_statistics(
            retention_results,
            spread_results,
        )

    else:
        run(
            number_of_realizations=(
                arguments.realizations
            ),
            number_of_workers=(
                arguments.workers
            ),
        )
