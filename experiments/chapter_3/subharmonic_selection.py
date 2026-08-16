"""
Selection of the number of atmospheric-turbulence subharmonic levels.

For each independent phase-screen realization, one FFT-based phase
screen is generated and subharmonic levels are then added cumulatively:

    b = 0, 1, ..., b_max.

For every level, the ensemble-averaged phase structure function is
compared with the theoretical structure function corresponding to the
selected turbulence PSD.

Supported models
----------------
    kolmogorov
    von_karman
    modified_von_karman

Metrics
-------
    - Mean absolute percentage error (MAPE)
    - Relative L2 error
    - Incremental change Delta_b
    - MAPE reduction between consecutive levels

Independent realizations are evaluated in parallel.
"""

import argparse
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_3 import (
    DEFAULT_SEED,
    DX,
    INNER_SCALE,
    N_GRID,
    OUTER_SCALE,
    R0,
)

from src.phase_screens import (
    kolmogorov_phase_screen,
    kolmogorov_subharmonic_level,
    modified_von_karman_phase_screen,
    modified_von_karman_subharmonic_level,
    von_karman_phase_screen,
    von_karman_subharmonic_level,
)

from src.structure_functions import (
    kolmogorov_structure_function,
    modified_von_karman_structure_function,
    structure_function_xy,
    von_karman_structure_function,
)


# ============================================================
# Common experiment configuration
# ============================================================

NUMBER_OF_SCREENS = 1000

RHO_MAX = 0.10  # [m]

NUMBER_OF_WORKERS = min(
    8,
    os.cpu_count() or 1,
)

CHUNKSIZE = 2


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIG = {
    "kolmogorov": {
        "label": "Kolmogorov",
        "maximum_subharmonic_level": 12,
    },

    "von_karman": {
        "label": "von Kármán",
        "maximum_subharmonic_level": 7,
    },

    "modified_von_karman": {
        "label": "von Kármán modificado",
        "maximum_subharmonic_level": 7,
    },
}


# ============================================================
# Error metrics
# ============================================================

def calculate_mape(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Calculate the mean absolute percentage error.
    """

    valid = (
        np.isfinite(numerical)
        & np.isfinite(theoretical)
        & (theoretical > 0.0)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid samples are available for MAPE calculation."
        )

    return float(
        100.0
        * np.mean(
            np.abs(
                numerical[valid]
                - theoretical[valid]
            )
            / theoretical[valid]
        )
    )


def calculate_relative_l2_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Calculate the relative L2 error in percent.
    """

    valid = (
        np.isfinite(numerical)
        & np.isfinite(theoretical)
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
            "The theoretical norm must be positive."
        )

    return float(
        100.0
        * numerator
        / denominator
    )


def calculate_incremental_change(
    current_structure: np.ndarray,
    previous_structure: np.ndarray,
) -> float:
    """
    Calculate the relative change between consecutive levels:

        Delta_b =
        100 ||D^(b)-D^(b-1)||_2 / ||D^(b)||_2.
    """

    valid = (
        np.isfinite(current_structure)
        & np.isfinite(previous_structure)
    )

    numerator = np.linalg.norm(
        current_structure[valid]
        - previous_structure[valid]
    )

    denominator = np.linalg.norm(
        current_structure[valid]
    )

    if denominator <= 0.0:
        raise ValueError(
            "The current structure-function norm is zero."
        )

    return float(
        100.0
        * numerator
        / denominator
    )


# ============================================================
# Model-dependent numerical generation
# ============================================================

def generate_fft_phase_screen(
    model: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one FFT-based phase screen for the selected model.
    """

    if model == "kolmogorov":
        return kolmogorov_phase_screen(
            n=N_GRID,
            delta=DX,
            r0=R0,
            rng=rng,
            remove_piston=True,
        )

    if model == "von_karman":
        return von_karman_phase_screen(
            n=N_GRID,
            delta=DX,
            r0=R0,
            outer_scale=OUTER_SCALE,
            rng=rng,
            remove_piston=True,
        )

    if model == "modified_von_karman":
        return modified_von_karman_phase_screen(
            n=N_GRID,
            delta=DX,
            r0=R0,
            outer_scale=OUTER_SCALE,
            inner_scale=INNER_SCALE,
            rng=rng,
            remove_piston=True,
        )

    raise ValueError(
        f"Unknown turbulence model: {model}"
    )


def generate_subharmonic_level_for_model(
    model: str,
    level: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one subharmonic level for the selected model.
    """

    if model == "kolmogorov":
        return kolmogorov_subharmonic_level(
            n=N_GRID,
            delta=DX,
            r0=R0,
            level=level,
            rng=rng,
            remove_piston=False,
        )

    if model == "von_karman":
        return von_karman_subharmonic_level(
            n=N_GRID,
            delta=DX,
            r0=R0,
            outer_scale=OUTER_SCALE,
            level=level,
            rng=rng,
            remove_piston=False,
        )

    if model == "modified_von_karman":
        return modified_von_karman_subharmonic_level(
            n=N_GRID,
            delta=DX,
            r0=R0,
            outer_scale=OUTER_SCALE,
            inner_scale=INNER_SCALE,
            level=level,
            rng=rng,
            remove_piston=False,
        )

    raise ValueError(
        f"Unknown turbulence model: {model}"
    )


# ============================================================
# Theoretical structure function
# ============================================================

def calculate_theoretical_structure_function(
    model: str,
    rho: np.ndarray,
) -> np.ndarray:
    """
    Calculate the theoretical phase structure function
    corresponding to the selected turbulence model.
    """

    if model == "kolmogorov":
        return kolmogorov_structure_function(
            rho=rho,
            r0=R0,
        )

    if model == "von_karman":
        return von_karman_structure_function(
            rho=rho,
            r0=R0,
            outer_scale=OUTER_SCALE,
        )

    if model == "modified_von_karman":
        return modified_von_karman_structure_function(
            rho=rho,
            r0=R0,
            outer_scale=OUTER_SCALE,
            inner_scale=INNER_SCALE,
        )

    raise ValueError(
        f"Unknown turbulence model: {model}"
    )


# ============================================================
# One realization
# ============================================================

def analyze_one_realization(
    realization_seed: int,
    model: str,
    max_shift: int,
    maximum_subharmonic_level: int,
) -> np.ndarray:
    """
    Generate one FFT phase screen and progressively add
    subharmonic levels from b=1 to b_max.
    """

    rng = np.random.default_rng(
        realization_seed
    )

    phase_fft = generate_fft_phase_screen(
        model=model,
        rng=rng,
    )

    structures = np.zeros(
        (
            maximum_subharmonic_level + 1,
            max_shift - 1,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # b = 0
    # --------------------------------------------------------

    _, structures[0] = structure_function_xy(
        phase=phase_fft,
        delta=DX,
        max_shift=max_shift,
    )

    phase_cumulative = (
        phase_fft.copy()
    )

    # --------------------------------------------------------
    # b = 1, ..., b_max
    # --------------------------------------------------------

    for level in range(
        1,
        maximum_subharmonic_level + 1,
    ):
        phase_level = (
            generate_subharmonic_level_for_model(
                model=model,
                level=level,
                rng=rng,
            )
        )

        phase_cumulative += (
            phase_level
        )

        phase_cumulative -= np.mean(
            phase_cumulative
        )

        _, structures[level] = (
            structure_function_xy(
                phase=phase_cumulative,
                delta=DX,
                max_shift=max_shift,
            )
        )

    return structures


# ============================================================
# Ensemble
# ============================================================

def ensemble_subharmonic_convergence(
    model: str,
    maximum_subharmonic_level: int,
    number_of_screens: int,
    number_of_workers: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate ensemble-averaged structure functions for all
    subharmonic levels.
    """

    maximum_shift_pixels = int(
        np.floor(
            RHO_MAX / DX
        )
    )

    max_shift = (
        maximum_shift_pixels + 1
    )

    actual_rho_max = (
        maximum_shift_pixels
        * DX
    )

    model_label = (
        MODEL_CONFIG[model]["label"]
    )

    print()
    print("Configuración de la prueba")
    print("==========================")
    print(f"Modelo                  = {model_label}")
    print(f"N                       = {N_GRID}")
    print(f"Delta x                 = {DX * 1e3:.6f} mm")
    print(f"r0                      = {R0 * 1e3:.4f} mm")

    if model in {
        "von_karman",
        "modified_von_karman",
    }:
        print(
            f"L0                      = "
            f"{OUTER_SCALE:.4f} m"
        )

    if model == "modified_von_karman":
        print(
            f"l0                      = "
            f"{INNER_SCALE * 1e3:.4f} mm"
        )

    print(f"rho_max solicitado      = {RHO_MAX:.6f} m")
    print(f"rho_max implementado    = {actual_rho_max:.6f} m")
    print(f"Máximo desplazamiento   = {maximum_shift_pixels} píxeles")
    print(f"Pantallas del ensamble  = {number_of_screens}")
    print(f"b máximo                = {maximum_subharmonic_level}")
    print(f"Procesos                 = {number_of_workers}")
    print()

    # --------------------------------------------------------
    # Independent reproducible seeds
    # --------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        number_of_screens
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

    worker = partial(
        analyze_one_realization,
        model=model,
        max_shift=max_shift,
        maximum_subharmonic_level=(
            maximum_subharmonic_level
        ),
    )

    accumulated = np.zeros(
        (
            maximum_subharmonic_level + 1,
            max_shift - 1,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Parallel ensemble
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            worker,
            realization_seeds,
            chunksize=CHUNKSIZE,
        )

        for index, structures in enumerate(
            results,
            start=1,
        ):
            accumulated += (
                structures
            )

            if (
                index == 1
                or index % 25 == 0
                or index == number_of_screens
            ):
                print(
                    f"Realizaciones completadas: "
                    f"{index}/{number_of_screens}"
                )

    mean_structures = (
        accumulated
        / number_of_screens
    )

    rho = (
        np.arange(
            1,
            max_shift,
        )
        * DX
    )

    return (
        rho,
        mean_structures,
    )


# ============================================================
# Convergence metrics
# ============================================================

def calculate_convergence_metrics(
    model: str,
    rho: np.ndarray,
    mean_structures: np.ndarray,
) -> list[dict[str, float | int]]:
    """
    Calculate MAPE, relative L2 error and incremental change
    for every subharmonic level.
    """

    theoretical = (
        calculate_theoretical_structure_function(
            model=model,
            rho=rho,
        )
    )

    results = []

    previous_mape = None

    for level in range(
        mean_structures.shape[0]
    ):
        numerical = (
            mean_structures[level]
        )

        mape = calculate_mape(
            numerical=numerical,
            theoretical=theoretical,
        )

        l2_error = (
            calculate_relative_l2_error(
                numerical=numerical,
                theoretical=theoretical,
            )
        )

        if level == 0:
            incremental_change = np.nan
            mape_improvement = np.nan

        else:
            incremental_change = (
                calculate_incremental_change(
                    current_structure=(
                        mean_structures[level]
                    ),
                    previous_structure=(
                        mean_structures[level - 1]
                    ),
                )
            )

            mape_improvement = (
                previous_mape
                - mape
            )

        results.append(
            {
                "b": level,
                "MAPE [%]": mape,
                "Error L2 [%]": l2_error,
                "Delta_b [%]":
                    incremental_change,
                "Reducción MAPE [p.p.]":
                    mape_improvement,
            }
        )

        previous_mape = mape

    return results


# ============================================================
# Terminal table
# ============================================================

def print_convergence_table(
    results: list[dict[str, float | int]],
) -> None:
    """
    Print convergence metrics.
    """

    print()
    print(
        "Convergencia con el número de subarmónicos"
    )
    print(
        "==========================================="
    )

    header = (
        f"{'b':>4}"
        f"{'MAPE [%]':>14}"
        f"{'Error L2 [%]':>16}"
        f"{'Delta_b [%]':>16}"
        f"{'Red. MAPE [p.p.]':>20}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for row in results:
        delta_b = float(
            row["Delta_b [%]"]
        )

        improvement = float(
            row["Reducción MAPE [p.p.]"]
        )

        delta_text = (
            "-"
            if np.isnan(delta_b)
            else f"{delta_b:.6f}"
        )

        improvement_text = (
            "-"
            if np.isnan(improvement)
            else f"{improvement:.6f}"
        )

        print(
            f"{int(row['b']):>4d}"
            f"{float(row['MAPE [%]']):>14.6f}"
            f"{float(row['Error L2 [%]']):>16.6f}"
            f"{delta_text:>16}"
            f"{improvement_text:>20}"
        )


# ============================================================
# CSV output
# ============================================================

def save_results_csv(
    results: list[dict[str, float | int]],
    output_directory: Path,
) -> None:
    """
    Save convergence metrics to CSV.
    """

    output_file = (
        output_directory
        / "subharmonic_convergence.csv"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "b,"
            "MAPE_percent,"
            "L2_error_percent,"
            "Delta_b_percent,"
            "MAPE_improvement_percentage_points\n"
        )

        for row in results:
            file.write(
                f"{int(row['b'])},"
                f"{float(row['MAPE [%]']):.8f},"
                f"{float(row['Error L2 [%]']):.8f},"
                f"{float(row['Delta_b [%]']):.8f},"
                f"{float(row['Reducción MAPE [p.p.]']):.8f}\n"
            )


# ============================================================
# Structure-function plot
# ============================================================

def plot_structure_functions(
    model: str,
    rho: np.ndarray,
    mean_structures: np.ndarray,
    output_directory: Path,
) -> None:
    """
    Plot representative subharmonic levels.
    """

    theoretical = (
        calculate_theoretical_structure_function(
            model=model,
            rho=rho,
        )
    )

    model_label = (
        MODEL_CONFIG[model]["label"]
    )

    maximum_level = (
        mean_structures.shape[0] - 1
    )

    figure, axis = plt.subplots(
        figsize=(7.4, 5.3)
    )

    axis.plot(
        rho,
        theoretical,
        linestyle="--",
        linewidth=2.3,
        label=f"Teoría {model_label}",
    )

    candidate_levels = (
        0,
        1,
        3,
        5,
        7,
        9,
        12,
    )

    for level in candidate_levels:
        if level > maximum_level:
            continue

        axis.plot(
            rho,
            mean_structures[level],
            linewidth=1.6,
            label=rf"$b={level}$",
        )

    axis.set_xlabel(
        r"Separación espacial $\rho$ [m]"
    )

    axis.set_ylabel(
        r"Función de estructura "
        r"$D_\phi(\rho)$ [rad$^2$]"
    )

    axis.set_title(
        f"Efecto de los niveles subarmónicos: "
        f"{model_label}"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend(
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "structure_functions_by_subharmonic_level.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Error plot
# ============================================================

def plot_convergence_metrics(
    results: list[dict[str, float | int]],
    model: str,
    output_directory: Path,
) -> None:
    """
    Plot theoretical-error metrics versus subharmonic level.
    """

    model_label = (
        MODEL_CONFIG[model]["label"]
    )

    levels = np.array(
        [
            int(row["b"])
            for row in results
        ]
    )

    mape = np.array(
        [
            float(row["MAPE [%]"])
            for row in results
        ]
    )

    l2_error = np.array(
        [
            float(row["Error L2 [%]"])
            for row in results
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 4.8)
    )

    axis.plot(
        levels,
        mape,
        marker="o",
        linewidth=1.8,
        label="MAPE",
    )

    axis.plot(
        levels,
        l2_error,
        marker="s",
        linewidth=1.8,
        label=r"Error relativo $L_2$",
    )

    axis.set_xlabel(
        "Número de niveles subarmónicos b"
    )

    axis.set_ylabel(
        "Error [%]"
    )

    axis.set_title(
        f"Error respecto a la teoría: {model_label}"
    )

    axis.set_xticks(
        levels
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "error_vs_subharmonic_level.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Incremental plot
# ============================================================

def plot_incremental_change(
    results: list[dict[str, float | int]],
    model: str,
    output_directory: Path,
) -> None:
    """
    Plot the incremental convergence criterion.
    """

    model_label = (
        MODEL_CONFIG[model]["label"]
    )

    levels = np.array(
        [
            int(row["b"])
            for row in results[1:]
        ]
    )

    incremental_change = np.array(
        [
            float(row["Delta_b [%]"])
            for row in results[1:]
        ]
    )

    mape_improvement = np.array(
        [
            float(
                row[
                    "Reducción MAPE [p.p.]"
                ]
            )
            for row in results[1:]
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 4.8)
    )

    axis.plot(
        levels,
        incremental_change,
        marker="o",
        linewidth=1.8,
        label=r"Cambio incremental $\Delta_b$",
    )

    axis.plot(
        levels,
        mape_improvement,
        marker="s",
        linewidth=1.8,
        label="Reducción del MAPE",
    )

    # Reference criterion established from the convergence analysis.
    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label=r"Criterio $\Delta_b=1\%$",
    )

    axis.axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Número de niveles subarmónicos b"
    )

    axis.set_ylabel(
        "Cambio [% / puntos porcentuales]"
    )

    axis.set_title(
        f"Convergencia incremental: {model_label}"
    )

    axis.set_xticks(
        levels
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "incremental_subharmonic_change.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate subharmonic convergence for an atmospheric "
            "turbulence PSD."
        )
    )

    parser.add_argument(
        "--model",
        choices=tuple(
            MODEL_CONFIG.keys()
        ),
        required=True,
        help="Turbulence PSD model.",
    )

    parser.add_argument(
        "--screens",
        type=int,
        default=NUMBER_OF_SCREENS,
        help=(
            "Number of independent realizations. "
            f"Default: {NUMBER_OF_SCREENS}."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=NUMBER_OF_WORKERS,
        help=(
            "Number of parallel worker processes. "
            f"Default: {NUMBER_OF_WORKERS}."
        ),
    )

    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help=(
            "Maximum subharmonic level. If omitted, the "
            "model-specific default is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# Execution
# ============================================================

def run(
    model: str,
    number_of_screens: int,
    number_of_workers: int,
    maximum_subharmonic_level: int | None = None,
) -> None:
    """
    Execute the complete subharmonic-selection experiment.
    """

    if number_of_screens <= 0:
        raise ValueError(
            "number_of_screens must be positive."
        )

    if number_of_workers <= 0:
        raise ValueError(
            "number_of_workers must be positive."
        )

    if maximum_subharmonic_level is None:
        maximum_subharmonic_level = int(
            MODEL_CONFIG[
                model
            ][
                "maximum_subharmonic_level"
            ]
        )

    if maximum_subharmonic_level < 0:
        raise ValueError(
            "maximum_subharmonic_level must be non-negative."
        )

    output_directory = Path(
        "results/chapter_3/subharmonic_selection"
    ) / model

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        rho,
        mean_structures,
    ) = ensemble_subharmonic_convergence(
        model=model,
        maximum_subharmonic_level=(
            maximum_subharmonic_level
        ),
        number_of_screens=number_of_screens,
        number_of_workers=number_of_workers,
    )

    results = calculate_convergence_metrics(
        model=model,
        rho=rho,
        mean_structures=mean_structures,
    )

    print_convergence_table(
        results
    )

    save_results_csv(
        results=results,
        output_directory=output_directory,
    )

    plot_structure_functions(
        model=model,
        rho=rho,
        mean_structures=mean_structures,
        output_directory=output_directory,
    )

    plot_convergence_metrics(
        results=results,
        model=model,
        output_directory=output_directory,
    )

    plot_incremental_change(
        results=results,
        model=model,
        output_directory=output_directory,
    )

    print(
        "\nResultados guardados en:"
        f"\n{output_directory.resolve()}"
    )


if __name__ == "__main__":
    arguments = parse_arguments()

    run(
        model=arguments.model,
        number_of_screens=arguments.screens,
        number_of_workers=arguments.workers,
        maximum_subharmonic_level=arguments.max_level,
    )
