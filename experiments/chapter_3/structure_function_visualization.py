"""
Visualization of the Kolmogorov phase structure function.

This script compares the ensemble-averaged phase structure function
obtained from:

    1. FFT-based Kolmogorov phase screens without subharmonics.
    2. Kolmogorov phase screens including a selected number of
       subharmonic levels.
    3. The theoretical Kolmogorov structure function.

The purpose of this script is qualitative and explanatory. It shows
why low-spatial-frequency compensation is required before the formal
subharmonic-convergence analysis is performed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_3 import (
    DEFAULT_SEED,
    DX,
    N_GRID,
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
# Visualization configuration
# ============================================================

NUMBER_OF_SCREENS = 200

MAX_SHIFT = N_GRID // 4

N_SUBHARMONICS = 5


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIRECTORY = Path(
    "results/chapter_3/structure_function_visualization"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Ensemble calculation
# ============================================================

def calculate_average_structure_functions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate ensemble-averaged structure functions with and
    without subharmonic compensation.

    Returns
    -------
    rho:
        Spatial separations [m].

    structure_fft:
        Ensemble-averaged structure function obtained from the
        FFT phase screens.

    structure_subharmonics:
        Ensemble-averaged structure function after adding the
        selected number of subharmonic levels.

    structure_theoretical:
        Theoretical Kolmogorov structure function.
    """

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        NUMBER_OF_SCREENS
    )

    structure_fft_sum = None
    structure_sub_sum = None
    rho = None

    for screen_index, child_seed in enumerate(
        child_sequences,
        start=1,
    ):
        rng = np.random.default_rng(
            child_seed
        )

        # ----------------------------------------------------
        # FFT phase screen
        # ----------------------------------------------------

        phase_fft = kolmogorov_phase_screen(
            n=N_GRID,
            delta=DX,
            r0=R0,
            rng=rng,
            remove_piston=True,
        )

        (
            current_rho,
            structure_fft,
        ) = structure_function_xy(
            phase=phase_fft,
            delta=DX,
            max_shift=MAX_SHIFT,
        )

        # ----------------------------------------------------
        # Subharmonic contribution
        # ----------------------------------------------------

        phase_sub = kolmogorov_subharmonics(
            n=N_GRID,
            delta=DX,
            r0=R0,
            n_subharmonics=N_SUBHARMONICS,
            rng=rng,
            remove_piston=False,
        )

        phase_total = (
            phase_fft
            + phase_sub
        )

        phase_total -= np.mean(
            phase_total
        )

        (
            _,
            structure_sub,
        ) = structure_function_xy(
            phase=phase_total,
            delta=DX,
            max_shift=MAX_SHIFT,
        )

        # ----------------------------------------------------
        # Accumulation
        # ----------------------------------------------------

        if structure_fft_sum is None:
            rho = current_rho

            structure_fft_sum = np.zeros_like(
                structure_fft,
                dtype=np.float64,
            )

            structure_sub_sum = np.zeros_like(
                structure_sub,
                dtype=np.float64,
            )

        structure_fft_sum += (
            structure_fft
        )

        structure_sub_sum += (
            structure_sub
        )

        if (
            screen_index == 1
            or screen_index % 25 == 0
            or screen_index == NUMBER_OF_SCREENS
        ):
            print(
                f"Screens analyzed: "
                f"{screen_index}/{NUMBER_OF_SCREENS}"
            )

    structure_fft_average = (
        structure_fft_sum
        / NUMBER_OF_SCREENS
    )

    structure_sub_average = (
        structure_sub_sum
        / NUMBER_OF_SCREENS
    )

    structure_theoretical = (
        kolmogorov_structure_function(
            rho=rho,
            r0=R0,
        )
    )

    return (
        rho,
        structure_fft_average,
        structure_sub_average,
        structure_theoretical,
    )


# ============================================================
# Plotting
# ============================================================

def plot_structure_functions(
    rho: np.ndarray,
    structure_fft: np.ndarray,
    structure_subharmonics: np.ndarray,
    structure_theoretical: np.ndarray,
) -> None:
    """
    Plot the numerical and theoretical phase structure functions.
    """

    figure, axis = plt.subplots(
        figsize=(7.2, 5.2)
    )

    axis.loglog(
        rho,
        structure_theoretical,
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Teoría de Kolmogorov "
            r"$6.88(\rho/r_0)^{5/3}$"
        ),
    )

    axis.loglog(
        rho,
        structure_fft,
        linewidth=2.0,
        label="Pantalla FFT sin subarmónicos",
    )

    axis.loglog(
        rho,
        structure_subharmonics,
        linewidth=2.0,
        label=(
            f"Pantalla FFT + "
            f"{N_SUBHARMONICS} niveles subarmónicos"
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
        "Función de estructura de fase de Kolmogorov"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "kolmogorov_structure_function.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Execution
# ============================================================

def run() -> None:
    """
    Execute the structure-function visualization.
    """

    print(
        "Kolmogorov structure-function visualization"
    )

    print(
        "-------------------------------------------"
    )

    print(
        f"Grid: {N_GRID} x {N_GRID}"
    )

    print(
        f"Spatial sampling: {DX:.6e} m"
    )

    print(
        f"Fried parameter: {R0:.6e} m"
    )

    print(
        f"Number of screens: {NUMBER_OF_SCREENS}"
    )

    print(
        f"Subharmonic levels: {N_SUBHARMONICS}"
    )

    print(
        f"Maximum shift: {MAX_SHIFT} pixels"
    )

    print()

    (
        rho,
        structure_fft,
        structure_subharmonics,
        structure_theoretical,
    ) = calculate_average_structure_functions()

    plot_structure_functions(
        rho=rho,
        structure_fft=structure_fft,
        structure_subharmonics=structure_subharmonics,
        structure_theoretical=structure_theoretical,
    )

    print(
        "\nFigure saved to:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    run()
