"""
Visualization of a Kolmogorov atmospheric phase screen.

This script generates a single phase-screen realization using the
numerical parameters adopted in Chapter 3. The resulting figure is
intended as a qualitative illustration of the atmospheric phase
perturbations used throughout the thesis.

This script is not a quantitative validation experiment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_3 import (
    DEFAULT_SEED,
    DX,
    N_GRID,
    R0,
    WINDOW_SIZE,
)

from src.phase_screens import (
    kolmogorov_phase_screen,
)


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIRECTORY = Path(
    "results/chapter_3/phase_screen_visualization"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Phase-screen generation
# ============================================================

def generate_screen() -> np.ndarray:
    """
    Generate one reproducible Kolmogorov phase-screen realization.
    """

    rng = np.random.default_rng(
        DEFAULT_SEED
    )

    phase = kolmogorov_phase_screen(
        n=N_GRID,
        delta=DX,
        r0=R0,
        rng=rng,
        remove_piston=True,
    )

    return phase


# ============================================================
# Visualization
# ============================================================

def plot_phase_screen(
    phase: np.ndarray,
) -> None:
    """
    Plot and save the atmospheric phase screen.
    """

    half_window = WINDOW_SIZE / 2.0

    extent = (
        -half_window,
        half_window,
        -half_window,
        half_window,
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    image = axis.imshow(
        phase,
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        "Kolmogorov phase-screen realization"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Phase [rad]"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "kolmogorov_phase_screen.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Execution
# ============================================================

def run() -> None:
    """
    Generate and visualize one Kolmogorov phase screen.
    """

    phase = generate_screen()

    print(
        "Kolmogorov phase-screen visualization"
    )

    print(
        f"Grid: {N_GRID} x {N_GRID}"
    )

    print(
        f"Window size: {WINDOW_SIZE:.3f} m"
    )

    print(
        f"Spatial sampling: {DX:.6e} m"
    )

    print(
        f"Fried parameter: {R0:.6e} m"
    )

    print(
        f"Seed: {DEFAULT_SEED}"
    )

    print(
        f"Mean phase: {np.mean(phase):.6e} rad"
    )

    print(
        f"Phase standard deviation: "
        f"{np.std(phase):.6e} rad"
    )

    plot_phase_screen(
        phase=phase,
    )

    print(
        "\nFigure saved to:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    run()
