"""Chapter 2: compare ASM propagation with analytical beam solutions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_2 import BEAMS, L_WINDOW, N_GRID, SCREEN_SPACING, WAVELENGTH
from src.analytical import create_analytical_beam, intensity_fidelity, relative_width_error
from src.beams import create_beam
from src.grids import create_grid
from src.propagation import angular_spectrum_propagation

OUTPUT_DIRECTORY = Path("results/chapter_2/analytical_propagation")


def radial_profile_along_x(field: np.ndarray) -> np.ndarray:
    profile = np.abs(field[field.shape[0] // 2, :]) ** 2
    maximum = np.max(profile)
    return profile / maximum if maximum > 0 else profile


def run() -> list[dict[str, object]]:
    grid = create_grid(N_GRID, L_WINDOW)
    results: list[dict[str, object]] = []

    for definition in BEAMS:
        initial = create_beam(definition, grid)
        numerical = angular_spectrum_propagation(initial, WAVELENGTH, SCREEN_SPACING, grid.dx)
        analytical = create_analytical_beam(definition, grid, WAVELENGTH, SCREEN_SPACING)
        w_num, w_an, w_error = relative_width_error(numerical, analytical, grid)
        fidelity = intensity_fidelity(numerical, analytical, grid.dx)
        results.append({
            "beam": definition.name or definition.family,
            "x": grid.x,
            "numerical": numerical,
            "analytical": analytical,
            "W_numerical": w_num,
            "W_analytical": w_an,
            "width_error": w_error,
            "fidelity": fidelity,
        })
    return results


def print_results(results: list[dict[str, object]]) -> None:
    header = f"{'Beam':<12}{'W ASM [m]':>16}{'W analytical [m]':>20}{'rel. error':>16}{'fidelity':>16}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row['beam']):<12}{float(row['W_numerical']):>16.8f}"
            f"{float(row['W_analytical']):>20.8f}{float(row['width_error']):>16.4e}"
            f"{float(row['fidelity']):>16.10f}"
        )


def save_plots(results: list[dict[str, object]]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for row in results:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x_mm = np.asarray(row["x"]) * 1e3
        ax.plot(x_mm, radial_profile_along_x(np.asarray(row["analytical"])), label="Analytical", linewidth=2.0)
        ax.plot(x_mm, radial_profile_along_x(np.asarray(row["numerical"])), "--", label="ASM", linewidth=1.5)
        ax.set_xlim(-100, 100)
        ax.set_xlabel(r"$x$ [mm]")
        ax.set_ylabel("Normalized intensity")
        ax.set_title(f"Propagated profile: {row['beam']}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        safe = str(row["beam"]).replace("/", "-").replace("^", "").replace("_", "-")
        fig.savefig(OUTPUT_DIRECTORY / f"profile_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    names = [str(row["beam"]) for row in results]
    errors = np.maximum(np.array([float(row["width_error"]) for row in results]), np.finfo(float).eps)
    fidelities = np.array([float(row["fidelity"]) for row in results])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(names, errors)
    ax.set_yscale("log")
    ax.set_ylabel("Relative second-moment-radius error")
    ax.set_title("ASM vs analytical propagation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "second_moment_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(names, fidelities)
    ax.set_ylabel("Intensity fidelity")
    ax.set_ylim(max(0.0, float(np.min(fidelities)) - 0.01), 1.001)
    ax.set_title("ASM vs analytical intensity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "intensity_fidelity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("ASM comparison with analytical beam propagation\n")
    results = run()
    print_results(results)
    save_plots(results)
