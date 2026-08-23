"""
Illustrative visualization of Gaussian-beam propagation
through weak atmospheric turbulence.

The same Gaussian input beam is propagated over 1000 m in:

    1. Vacuum.
    2. One reproducible realization of weak Kolmogorov turbulence.

The turbulent propagation uses exactly the same split-step
configuration employed in the weak-turbulence Rytov validation:

    propagate dz/2
    -> apply phase screen
    -> propagate dz/2

repeated over all phase-screen segments.

This script is intended for qualitative visualization only.
The quantitative validation is performed through the ensemble
statistics and the comparison with Rytov theory.
"""

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
    RYTOV_R0_SCREEN,
    RYTOV_SUBHARMONIC_LEVEL,
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

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/weak_turbulence_visualization"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_FILE = (
    OUTPUT_DIRECTORY
    / "gaussian_vacuum_vs_weak_turbulence.png"
)

DATA_FILE = (
    OUTPUT_DIRECTORY
    / "gaussian_vacuum_vs_weak_turbulence.npz"
)

# Fixed a priori for reproducibility.
VISUALIZATION_SEED = 20260820


# ============================================================
# Input beam
# ============================================================

def create_gaussian_input():
    """
    Create the same Gaussian input beam used in the
    weak-turbulence Rytov validation.
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
# Optical power
# ============================================================

def calculate_power(
    field: np.ndarray,
) -> float:
    """
    Calculate the discrete optical power of a field.
    """

    return float(
        np.sum(
            np.abs(field) ** 2
        )
        * DX**2
    )


# ============================================================
# Vacuum propagation
# ============================================================

def propagate_in_vacuum(
    field: np.ndarray,
) -> np.ndarray:
    """
    Propagate the Gaussian beam directly over the complete
    propagation distance in vacuum.
    """

    return angular_spectrum_propagation(
        field=field,
        wavelength=WAVELENGTH,
        distance=TOTAL_PROPAGATION_DISTANCE,
        dx=DX,
    )


# ============================================================
# Weak-turbulence propagation
# ============================================================

def propagate_through_weak_turbulence(
    field: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Propagate one reproducible realization through weak
    Kolmogorov turbulence using the same split-step scheme
    employed in the Rytov validation.
    """

    rng = np.random.default_rng(
        seed
    )

    propagated_field = field.copy()

    for _ in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        # First half-segment.
        propagated_field = (
            angular_spectrum_propagation(
                field=propagated_field,
                wavelength=WAVELENGTH,
                distance=HALF_SCREEN_SPACING,
                dx=DX,
            )
        )

        # Phase screen at segment center.
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

        propagated_field *= np.exp(
            1j * phase_screen
        )

        # Second half-segment.
        propagated_field = (
            angular_spectrum_propagation(
                field=propagated_field,
                wavelength=WAVELENGTH,
                distance=HALF_SCREEN_SPACING,
                dx=DX,
            )
        )

    return propagated_field


# ============================================================
# Save numerical data
# ============================================================

def save_data(
    grid,
    vacuum_intensity: np.ndarray,
    turbulent_intensity: np.ndarray,
) -> None:
    """
    Save the numerical data used to construct the figure.
    """

    np.savez_compressed(
        DATA_FILE,
        x=grid.x,
        y=grid.y,
        vacuum_intensity=vacuum_intensity,
        turbulent_intensity=turbulent_intensity,
        seed=VISUALIZATION_SEED,
    )


# ============================================================
# Figure
# ============================================================

def plot_comparison(
    grid,
    vacuum_intensity: np.ndarray,
    turbulent_intensity: np.ndarray,
) -> None:
    """
    Compare the output irradiance in vacuum and through one
    realization of weak turbulence.

    Both panels use the same spatial and intensity scales.
    """

    maximum_intensity = max(
        float(np.max(vacuum_intensity)),
        float(np.max(turbulent_intensity)),
    )

    extent = (
        100.0 * grid.x[0],
        100.0 * grid.x[-1],
        100.0 * grid.y[0],
        100.0 * grid.y[-1],
    )

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(10.5, 4.5)
    )

    grid_spec = figure.add_gridspec(
        1,
        3,
        width_ratios=(
            1.0,
            1.0,
            0.045,
        ),
        wspace=0.12,
    )

    axis_vacuum = figure.add_subplot(
        grid_spec[0, 0]
    )

    axis_turbulence = figure.add_subplot(
        grid_spec[0, 1],
        sharex=axis_vacuum,
        sharey=axis_vacuum,
    )

    colorbar_axis = figure.add_subplot(
        grid_spec[0, 2]
    )

    # --------------------------------------------------------
    # Vacuum
    # --------------------------------------------------------

    vacuum_image = axis_vacuum.imshow(
        vacuum_intensity,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=maximum_intensity,
        cmap="magma",
        aspect="equal",
    )

    axis_vacuum.set_title(
        "(a) Propagación en vacío"
    )

    axis_vacuum.set_xlabel(
        "$x$ [cm]"
    )

    axis_vacuum.set_ylabel(
        "$y$ [cm]"
    )

    # --------------------------------------------------------
    # Weak turbulence
    # --------------------------------------------------------

    axis_turbulence.imshow(
        turbulent_intensity,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=maximum_intensity,
        cmap="magma",
        aspect="equal",
    )

    axis_turbulence.set_title(
        "(b) Turbulencia débil"
    )

    axis_turbulence.set_xlabel(
        "$x$ [cm]"
    )

    # Avoid duplicated y labels.
    axis_turbulence.tick_params(
        labelleft=False
    )

    # --------------------------------------------------------
    # Common colorbar
    # --------------------------------------------------------

    colorbar = figure.colorbar(
        vacuum_image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Irradiancia [u. a.]"
    )

    # --------------------------------------------------------
    # General title
    # --------------------------------------------------------

    figure.suptitle(
        "Propagación de un haz gaussiano a 1000 m",
        y=0.98,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.14,
        top=0.84,
    )

    figure.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Generate the illustrative weak-turbulence comparison.
    """

    print(
        "Visualización de turbulencia débil"
    )

    print(
        "=================================="
    )

    print(
        f"Distancia: "
        f"{TOTAL_PROPAGATION_DISTANCE:.1f} m"
    )

    print(
        f"Pantallas: "
        f"{NUMBER_OF_PHASE_SCREENS}"
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
        f"Semilla: "
        f"{VISUALIZATION_SEED}"
    )

    grid, input_field = (
        create_gaussian_input()
    )

    initial_power = (
        calculate_power(
            input_field
        )
    )

    vacuum_field = (
        propagate_in_vacuum(
            input_field
        )
    )

    turbulent_field = (
        propagate_through_weak_turbulence(
            field=input_field,
            seed=VISUALIZATION_SEED,
        )
    )

    vacuum_power = (
        calculate_power(
            vacuum_field
        )
    )

    turbulent_power = (
        calculate_power(
            turbulent_field
        )
    )

    vacuum_intensity = (
        np.abs(vacuum_field) ** 2
    )

    turbulent_intensity = (
        np.abs(turbulent_field) ** 2
    )

    print()
    print(
        "Conservación de potencia"
    )

    print(
        "-----------------------"
    )

    print(
        f"Entrada:      "
        f"{initial_power:.12e}"
    )

    print(
        f"Vacío:        "
        f"{vacuum_power:.12e}"
    )

    print(
        f"Turbulencia:  "
        f"{turbulent_power:.12e}"
    )

    vacuum_power_error = (
        100.0
        * abs(
            vacuum_power
            - initial_power
        )
        / initial_power
    )

    turbulent_power_error = (
        100.0
        * abs(
            turbulent_power
            - initial_power
        )
        / initial_power
    )

    print()
    print(
        f"Error potencia vacío: "
        f"{vacuum_power_error:.6e} %"
    )

    print(
        f"Error potencia turbulencia: "
        f"{turbulent_power_error:.6e} %"
    )

    save_data(
        grid=grid,
        vacuum_intensity=vacuum_intensity,
        turbulent_intensity=(
            turbulent_intensity
        ),
    )

    plot_comparison(
        grid=grid,
        vacuum_intensity=vacuum_intensity,
        turbulent_intensity=(
            turbulent_intensity
        ),
    )

    print()
    print(
        "Figura guardada en:"
    )

    print(
        FIGURE_FILE
    )

    print(
        "Datos guardados en:"
    )

    print(
        DATA_FILE
    )


if __name__ == "__main__":
    main()
