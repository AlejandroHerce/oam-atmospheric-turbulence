"""
Convergence of the Kolmogorov phase structure function with
the physical phase-screen size.

The spatial sampling interval is kept fixed while the number of
grid points is varied. Therefore,

    L = N * delta

changes between simulations.

The experiment evaluates whether increasing the physical size of the
screen improves the recovery of the low-spatial-frequency content
required by the Kolmogorov phase structure function.

Independent realizations are evaluated in parallel.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_3 import (
    DEFAULT_SEED,
    DX,
    R0,
)

from src.phase_screens import (
    kolmogorov_phase_screen,
    kolmogorov_subharmonics,
)

from src.structure_functions import (
    kolmogorov_structure_function,
    structure_function_xy,
)


# ============================================================
# Experiment configuration
# ============================================================

GRID_SIZES = (
    256,
    512,
    1024,
    2048,
)

NUMBER_OF_SCREENS = 500

SUBHARMONIC_LEVEL = 9

# The same rho interval is used for every grid.
MAX_SHIFT = min(GRID_SIZES) // 4

# Conservative choice for memory usage with 2048 x 2048 screens.
NUMBER_OF_WORKERS = min(
    6,
    os.cpu_count() or 1,
)

CHUNKSIZE = 1


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIRECTORY = Path(
    "results/chapter_3/subharmonic_window_convergence"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Error metrics
# ============================================================

def mean_absolute_percentage_error(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Calculate the mean absolute percentage error.
    """

    if numerical.shape != theoretical.shape:
        raise ValueError(
            "numerical and theoretical must have the same shape."
        )

    if np.any(theoretical == 0.0):
        raise ValueError(
            "theoretical must not contain zero values."
        )

    return float(
        100.0
        * np.mean(
            np.abs(
                (
                    numerical
                    - theoretical
                )
                / theoretical
            )
        )
    )


def relative_l2_error_percentage(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    Calculate the relative L2 error in percent.
    """

    if numerical.shape != theoretical.shape:
        raise ValueError(
            "numerical and theoretical must have the same shape."
        )

    denominator = np.linalg.norm(
        theoretical
    )

    if denominator <= 0.0:
        raise ValueError(
            "The theoretical norm must be positive."
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
# One realization
# ============================================================

def analyze_one_realization(
    realization_seed: int,
    n: int,
    delta: float,
    r0: float,
    max_shift: int,
    subharmonic_level: int,
) -> np.ndarray:
    """
    Generate one phase-screen realization and calculate its
    phase structure function.
    """

    if subharmonic_level < 0:
        raise ValueError(
            "subharmonic_level must be non-negative."
        )

    rng = np.random.default_rng(
        realization_seed
    )

    # --------------------------------------------------------
    # FFT phase screen
    # --------------------------------------------------------

    phase_fft = kolmogorov_phase_screen(
        n=n,
        delta=delta,
        r0=r0,
        rng=rng,
        remove_piston=True,
    )

    phase_total = phase_fft

    # --------------------------------------------------------
    # Low-frequency subharmonics
    # --------------------------------------------------------

    if subharmonic_level > 0:
        phase_subharmonics = kolmogorov_subharmonics(
            n=n,
            delta=delta,
            r0=r0,
            n_subharmonics=subharmonic_level,
            rng=rng,
            remove_piston=False,
        )

        phase_total = (
            phase_fft
            + phase_subharmonics
        )

        phase_total -= np.mean(
            phase_total
        )

    # --------------------------------------------------------
    # Structure function
    # --------------------------------------------------------

    _, structure = structure_function_xy(
        phase=phase_total,
        delta=delta,
        max_shift=max_shift,
    )

    return structure


# ============================================================
# Ensemble for one grid size
# ============================================================

def ensemble_structure_for_size(
    number_of_screens: int,
    n: int,
    delta: float,
    r0: float,
    max_shift: int,
    subharmonic_level: int,
    realization_seeds: list[int],
    number_of_workers: int,
    chunksize: int = 1,
) -> np.ndarray:
    """
    Calculate the ensemble-averaged phase structure function
    for one grid size.
    """

    if len(realization_seeds) != number_of_screens:
        raise ValueError(
            "The number of seeds must equal number_of_screens."
        )

    worker = partial(
        analyze_one_realization,
        n=n,
        delta=delta,
        r0=r0,
        max_shift=max_shift,
        subharmonic_level=subharmonic_level,
    )

    accumulated_structure = np.zeros(
        max_shift - 1,
        dtype=np.float64,
    )

    print()
    print(
        f"Processing N={n}, "
        f"L={n * delta:.4f} m, "
        f"b={subharmonic_level}"
    )

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            worker,
            realization_seeds,
            chunksize=chunksize,
        )

        for index, current_structure in enumerate(
            results,
            start=1,
        ):
            accumulated_structure += (
                current_structure
            )

            if (
                index == 1
                or index % 25 == 0
                or index == number_of_screens
            ):
                print(
                    f"N={n}: realizations "
                    f"{index}/{number_of_screens}"
                )

    return (
        accumulated_structure
        / number_of_screens
    )


# ============================================================
# Screen-size comparison
# ============================================================

def compare_screen_sizes() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, dict[str, float]],
]:
    """
    Compare the ensemble-averaged structure function for all
    selected physical screen sizes.
    """

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        NUMBER_OF_SCREENS
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

    rho = (
        np.arange(
            1,
            MAX_SHIFT,
        )
        * DX
    )

    theoretical_structure = (
        kolmogorov_structure_function(
            rho=rho,
            r0=R0,
        )
    )

    mean_structures: dict[
        int,
        np.ndarray,
    ] = {}

    errors: dict[
        int,
        dict[str, float],
    ] = {}

    for n in GRID_SIZES:
        mean_structure = (
            ensemble_structure_for_size(
                number_of_screens=NUMBER_OF_SCREENS,
                n=n,
                delta=DX,
                r0=R0,
                max_shift=MAX_SHIFT,
                subharmonic_level=SUBHARMONIC_LEVEL,
                realization_seeds=realization_seeds,
                number_of_workers=NUMBER_OF_WORKERS,
                chunksize=CHUNKSIZE,
            )
        )

        mean_structures[n] = (
            mean_structure
        )

        errors[n] = {
            "MAPE [%]":
                mean_absolute_percentage_error(
                    numerical=mean_structure,
                    theoretical=theoretical_structure,
                ),

            "Relative L2 error [%]":
                relative_l2_error_percentage(
                    numerical=mean_structure,
                    theoretical=theoretical_structure,
                ),
        }

    return (
        rho,
        theoretical_structure,
        mean_structures,
        errors,
    )


# ============================================================
# Terminal summary
# ============================================================

def print_results(
    errors: dict[int, dict[str, float]],
) -> None:
    """
    Print the screen-size convergence table.
    """

    print()
    print(
        "Convergence with physical phase-screen size"
    )
    print(
        "==========================================="
    )

    print(
        f"delta = {DX:.6e} m"
    )

    print(
        f"r0 = {R0:.6e} m"
    )

    print(
        f"Subharmonic level b = "
        f"{SUBHARMONIC_LEVEL}"
    )

    print(
        f"Number of realizations = "
        f"{NUMBER_OF_SCREENS}"
    )

    print(
        f"Workers = "
        f"{NUMBER_OF_WORKERS}"
    )

    print()

    header = (
        f"{'N':>8}"
        f"{'L [m]':>12}"
        f"{'MAPE [%]':>16}"
        f"{'L2 error [%]':>18}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for n in GRID_SIZES:
        print(
            f"{n:>8d}"
            f"{n * DX:>12.4f}"
            f"{errors[n]['MAPE [%]']:>16.6f}"
            f"{errors[n]['Relative L2 error [%]']:>18.6f}"
        )


# ============================================================
# Structure-function comparison
# ============================================================

def plot_structure_functions(
    rho: np.ndarray,
    theoretical_structure: np.ndarray,
    mean_structures: dict[int, np.ndarray],
) -> None:
    """
    Plot the structure-function convergence with physical
    screen size using a linear scale.
    """

    figure, axis = plt.subplots(
        figsize=(7.4, 5.3)
    )

    axis.plot(
        rho,
        theoretical_structure,
        linestyle="--",
        linewidth=2.3,
        label=(
            r"Teoría de Kolmogorov "
            r"$6.88(\rho/r_0)^{5/3}$"
        ),
    )

    for n in GRID_SIZES:
        axis.plot(
            rho,
            mean_structures[n],
            linewidth=1.8,
            label=(
                rf"$N={n}$, "
                rf"$L={n * DX:.2f}$ m"
            ),
        )

    axis.set_xlabel(
        r"Separación espacial $\rho$ [m]"
    )

    axis.set_ylabel(
        r"Función de estructura "
        r"$D_\phi(\rho)$ [rad$^2$]"
    )

    axis.set_title(
        "Convergencia con el tamaño físico de la pantalla"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend(
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / (
            "structure_function_"
            f"b{SUBHARMONIC_LEVEL}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Error comparison
# ============================================================

def plot_errors(
    errors: dict[int, dict[str, float]],
) -> None:
    """
    Plot quantitative convergence metrics against physical
    screen size.
    """

    window_sizes = np.array(
        [
            n * DX
            for n in GRID_SIZES
        ],
        dtype=np.float64,
    )

    mape = np.array(
        [
            errors[n]["MAPE [%]"]
            for n in GRID_SIZES
        ],
        dtype=np.float64,
    )

    l2_error = np.array(
        [
            errors[n]["Relative L2 error [%]"]
            for n in GRID_SIZES
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 4.8)
    )

    axis.plot(
        window_sizes,
        mape,
        marker="o",
        linewidth=1.8,
        label="MAPE",
    )

    axis.plot(
        window_sizes,
        l2_error,
        marker="s",
        linewidth=1.8,
        label=r"Error relativo $L_2$",
    )

    axis.set_xlabel(
        "Tamaño físico de la pantalla L [m]"
    )

    axis.set_ylabel(
        "Error [%]"
    )

    axis.set_title(
        "Error respecto a la función de estructura teórica"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / (
            "errors_vs_screen_size_"
            f"b{SUBHARMONIC_LEVEL}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Execution
# ============================================================

def run() -> None:
    """
    Execute the physical screen-size convergence experiment.
    """

    print(
        "Kolmogorov screen-size convergence"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Grid sizes: {GRID_SIZES}"
    )

    print(
        f"Fixed spatial sampling: "
        f"{DX:.6e} m"
    )

    print(
        f"Subharmonic level: "
        f"{SUBHARMONIC_LEVEL}"
    )

    print(
        f"Number of workers: "
        f"{NUMBER_OF_WORKERS}"
    )

    (
        rho,
        theoretical_structure,
        mean_structures,
        errors,
    ) = compare_screen_sizes()

    print_results(
        errors
    )

    plot_structure_functions(
        rho=rho,
        theoretical_structure=theoretical_structure,
        mean_structures=mean_structures,
    )

    plot_errors(
        errors
    )

    print(
        "\nFigures saved to:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    run()
