"""Chapter 2: validate energy conservation during free-space ASM propagation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_2 import BEAMS, L_WINDOW, N_GRID, SCREEN_SPACING, WAVELENGTH
from src.beams import calculate_energy, create_beam
from src.grids import create_grid
from src.propagation import angular_spectrum_propagation

OUTPUT_DIRECTORY = Path("results/chapter_2/energy_conservation")


def run() -> list[dict[str, float | str]]:
    grid = create_grid(N_GRID, L_WINDOW)
    results = []

    for definition in BEAMS:
        initial = create_beam(definition, grid)
        final = angular_spectrum_propagation(initial, WAVELENGTH, SCREEN_SPACING, grid.dx)
        e0 = calculate_energy(initial, grid.dx)
        ez = calculate_energy(final, grid.dx)
        absolute_error = abs(ez - e0)
        results.append({
            "beam": definition.name or definition.family,
            "initial_energy": e0,
            "final_energy": ez,
            "absolute_error": absolute_error,
            "relative_error": absolute_error / e0,
        })
    return results


def print_results(results: list[dict[str, float | str]]) -> None:
    header = f"{'Beam':<12}{'E(0)':>16}{'E(z)':>16}{'abs. error':>18}{'rel. error':>18}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row['beam']):<12}"
            f"{float(row['initial_energy']):>16.10f}"
            f"{float(row['final_energy']):>16.10f}"
            f"{float(row['absolute_error']):>18.4e}"
            f"{float(row['relative_error']):>18.4e}"
        )


def save_plot(results: list[dict[str, float | str]]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    names = [str(row["beam"]) for row in results]
    errors = np.maximum(
        np.array([float(row["relative_error"]) for row in results]),
        np.finfo(float).eps,
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(names, errors)
    ax.set_yscale("log")
    ax.set_xlabel("Beam")
    ax.set_ylabel(r"$|E(z)-E(0)|/E(0)$")
    ax.set_title("Relative error in energy conservation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "relative_energy_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Energy conservation in free-space ASM propagation")
    print(f"Grid: {N_GRID} x {N_GRID}, L = {L_WINDOW:.3f} m")
    print(f"Propagation distance = {SCREEN_SPACING:.2f} m\n")
    results = run()
    print_results(results)
    save_plot(results)
