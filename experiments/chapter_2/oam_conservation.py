"""Chapter 2: validate OAM conservation during free-space ASM propagation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_2 import (
    BEAMS,
    L_WINDOW,
    N_GRID,
    OAM_AZIMUTHAL_SAMPLES,
    OAM_ELL_MAX,
    OAM_ELL_MIN,
    OAM_RADIAL_SAMPLES,
    SCREEN_SPACING,
    WAVELENGTH,
)
from src.beams import create_beam
from src.grids import create_grid
from src.oam import (
    calculate_mean_oam,
    calculate_oam_spectrum,
    modal_power_at_charge,
    spectral_l1_distance,
)
from src.propagation import angular_spectrum_propagation

OUTPUT_DIRECTORY = Path("results/chapter_2/oam_conservation")


def run() -> list[dict[str, object]]:
    grid = create_grid(N_GRID, L_WINDOW)
    results: list[dict[str, object]] = []

    for definition in BEAMS:
        initial = create_beam(definition, grid)
        final = angular_spectrum_propagation(initial, WAVELENGTH, SCREEN_SPACING, grid.dx)

        ell, p0 = calculate_oam_spectrum(
            initial, grid,
            ell_min=OAM_ELL_MIN, ell_max=OAM_ELL_MAX,
            radial_samples=OAM_RADIAL_SAMPLES,
            azimuthal_samples=OAM_AZIMUTHAL_SAMPLES,
        )
        _, pz = calculate_oam_spectrum(
            final, grid,
            ell_min=OAM_ELL_MIN, ell_max=OAM_ELL_MAX,
            radial_samples=OAM_RADIAL_SAMPLES,
            azimuthal_samples=OAM_AZIMUTHAL_SAMPLES,
        )

        mean0 = calculate_mean_oam(ell, p0)
        meanz = calculate_mean_oam(ell, pz)
        results.append({
            "beam": definition.name or definition.family,
            "charge": definition.charge,
            "ell": ell,
            "p0": p0,
            "pz": pz,
            "mean0": mean0,
            "meanz": meanz,
            "mean_error": abs(meanz - mean0),
            "nominal0": modal_power_at_charge(ell, p0, definition.charge),
            "nominalz": modal_power_at_charge(ell, pz, definition.charge),
            "l1": spectral_l1_distance(p0, pz),
        })
    return results


def print_results(results: list[dict[str, object]]) -> None:
    header = f"{'Beam':<12}{'<ell> in':>14}{'<ell> out':>14}{'error':>14}{'Pnom in':>14}{'Pnom out':>14}{'L1':>14}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row['beam']):<12}{float(row['mean0']):>14.8f}{float(row['meanz']):>14.8f}"
            f"{float(row['mean_error']):>14.4e}{float(row['nominal0']):>14.8f}"
            f"{float(row['nominalz']):>14.8f}{float(row['l1']):>14.4e}"
        )


def save_plots(results: list[dict[str, object]]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for row in results:
        ell = np.asarray(row["ell"])
        p0 = np.asarray(row["p0"])
        pz = np.asarray(row["pz"])
        x = np.arange(ell.size)
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(x - width / 2, p0, width, label="Input")
        ax.bar(x + width / 2, pz, width, label="After propagation")
        ax.set_xticks(x)
        ax.set_xticklabels(ell)
        ax.set_xlabel(r"Topological charge $\ell$")
        ax.set_ylabel(r"Normalized modal power $P_\ell$")
        ax.set_title(f"OAM spectrum: {row['beam']}")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        safe = str(row["beam"]).replace("/", "-").replace("^", "").replace("_", "-")
        fig.savefig(OUTPUT_DIRECTORY / f"oam_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    print("OAM conservation in free-space ASM propagation\n")
    results = run()
    print_results(results)
    save_plots(results)
