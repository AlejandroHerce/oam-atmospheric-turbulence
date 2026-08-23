"""
Illustrative propagation of a Bessel-Gauss beam through
moderate and strong atmospheric turbulence.

A BG^3 beam is propagated through the final Split-Step
configuration used in Chapter 4. The same random seed is used
for the moderate and strong turbulence cases, allowing a direct
qualitative comparison of the effect of turbulence strength.

Three intensity maps are generated:

    1. Input BG^3 beam.
    2. Output after moderate turbulence.
    3. Output after strong turbulence.

The script also verifies conservation of optical power.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from configs.chapter_2 import (
    BG_PARAMETERS,
)

from configs.chapter_3 import (
    KOLMOGOROV_SUBHARMONIC_LEVEL,
)

from configs.chapter_4 import (
    DX,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    MODERATE_R0_SCREEN,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    STRONG_R0_SCREEN,
    WAVELENGTH,
)

from src.beams import (
    bessel_gaussian_beam,
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
# Experiment configuration
# ============================================================

BG_CHARGE = 3

REALIZATION_SEED = 20260822

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/bg_turbulence_example"
)


# ============================================================
# Input beam
# ============================================================

def create_bg3_input():
    """
    Create the BG^3 beam used in the illustrative example.
    """

    grid = create_grid(
        n=N_GRID,
        window_size=L_WINDOW,
    )

    parameters = BG_PARAMETERS[
        BG_CHARGE
    ]

    field = bessel_gaussian_beam(
        grid=grid,
        w0=parameters["w0"],
        kr=parameters["kr"],
        charge=BG_CHARGE,
    )

    return grid, field


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
# Split-Step propagation
# ============================================================

def propagate_one_realization(
    input_field: np.ndarray,
    r0_screen: float,
    seed: int,
) -> np.ndarray:
    """
    Propagate one atmospheric realization.

    The propagation geometry is identical to that used in the
    spectral-sampling verification:

        half step -> phase screen -> half step

    for every turbulent segment.
    """

    field = input_field.copy()

    rng = np.random.default_rng(
        seed
    )

    for _ in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        # ----------------------------------------------------
        # First half of the segment
        # ----------------------------------------------------

        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        # ----------------------------------------------------
        # Phase screen
        # ----------------------------------------------------

        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                n_subharmonics=(
                    KOLMOGOROV_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        # ----------------------------------------------------
        # Second half of the segment
        # ----------------------------------------------------

        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

    return field


# ============================================================
# Intensity
# ============================================================

def calculate_intensity(
    field: np.ndarray,
) -> np.ndarray:
    """
    Calculate optical intensity.
    """

    return (
        np.abs(field) ** 2
    )


def normalize_intensities(
    input_intensity: np.ndarray,
    moderate_intensity: np.ndarray,
    strong_intensity: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Normalize all three intensity distributions using one
    common maximum.

    A common normalization preserves the relative intensity
    scale between the three panels.
    """

    common_maximum = max(
        float(
            np.max(input_intensity)
        ),
        float(
            np.max(moderate_intensity)
        ),
        float(
            np.max(strong_intensity)
        ),
    )

    if common_maximum <= 0.0:
        raise ValueError(
            "The common intensity maximum must be positive."
        )

    return (
        input_intensity
        / common_maximum,
        moderate_intensity
        / common_maximum,
        strong_intensity
        / common_maximum,
    )


# ============================================================
# Plotting
# ============================================================

def save_intensity_map(
    intensity: np.ndarray,
    filename: str,
) -> None:
    """
    Save one normalized intensity map.

    All panels use identical spatial and intensity scales so
    that they can be compared directly in the thesis.
    """

    extent = (
        -L_WINDOW / 2.0,
        L_WINDOW / 2.0,
        -L_WINDOW / 2.0,
        L_WINDOW / 2.0,
    )

    figure, axis = plt.subplots(
        figsize=(
            5.2,
            4.5,
        )
    )

    image = axis.imshow(
        intensity,
        extent=extent,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axis.set_xlabel(
        r"$x$ [m]"
    )

    axis.set_ylabel(
        r"$y$ [m]"
    )

    axis.set_aspect(
        "equal"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.03,
        fraction=0.046,
    )

    colorbar.set_label(
        "Intensidad normalizada"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Numerical checks
# ============================================================

def print_power_summary(
    input_field: np.ndarray,
    moderate_field: np.ndarray,
    strong_field: np.ndarray,
) -> None:
    """
    Print optical-power conservation for the three fields.
    """

    input_power = calculate_power(
        input_field
    )

    moderate_power = calculate_power(
        moderate_field
    )

    strong_power = calculate_power(
        strong_field
    )

    moderate_error = (
        100.0
        * abs(
            moderate_power
            - input_power
        )
        / input_power
    )

    strong_error = (
        100.0
        * abs(
            strong_power
            - input_power
        )
        / input_power
    )

    print()
    print(
        "Conservación de potencia"
    )
    print(
        "-----------------------"
    )

    print(
        f"Entrada:   "
        f"{input_power:.12e}"
    )

    print(
        f"Moderada:  "
        f"{moderate_power:.12e}"
    )

    print(
        f"Fuerte:    "
        f"{strong_power:.12e}"
    )

    print()

    print(
        "Error potencia moderada: "
        f"{moderate_error:.6e} %"
    )

    print(
        "Error potencia fuerte:   "
        f"{strong_error:.6e} %"
    )

def save_composite_figure(
    input_intensity: np.ndarray,
    moderate_intensity: np.ndarray,
    strong_intensity: np.ndarray,
) -> None:
    """
    Save a three-panel comparison with one common colorbar.
    """

    extent = (
        -L_WINDOW / 2.0,
        L_WINDOW / 2.0,
        -L_WINDOW / 2.0,
        L_WINDOW / 2.0,
    )

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(13.5, 4.2)
    )

    grid_spec = figure.add_gridspec(
        1,
        4,
        width_ratios=(
            1.0,
            1.0,
            1.0,
            0.05,
        ),
        wspace=0.18,
    )

    axes = [
        figure.add_subplot(
            grid_spec[0, index]
        )
        for index in range(3)
    ]

    colorbar_axis = figure.add_subplot(
        grid_spec[0, 3]
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    intensities = (
        input_intensity,
        moderate_intensity,
        strong_intensity,
    )

    titles = (
        "(a) Campo incidente",
        "(b) Turbulencia moderada",
        "(c) Turbulencia fuerte",
    )

    # --------------------------------------------------------
    # Intensity maps
    # --------------------------------------------------------

    image = None

    for axis, intensity, title in zip(
        axes,
        intensities,
        titles,
    ):

        image = axis.imshow(
            intensity,
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axis.set_title(
            title
        )

        axis.set_xlabel(
            r"$x$ [m]"
        )

        axis.set_aspect(
            "equal"
        )

    axes[0].set_ylabel(
        r"$y$ [m]"
    )

    # --------------------------------------------------------
    # Common colorbar
    # --------------------------------------------------------

    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Intensidad normalizada"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    figure.savefig(
        OUTPUT_DIRECTORY
        / "bg3_turbulence_comparison.png",
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
    Generate the illustrative BG^3 turbulence figures.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Ejemplo de propagación turbulenta de BG^3"
    )

    print(
        "========================================="
    )

    print(
        f"Carga topológica: "
        f"{BG_CHARGE}"
    )

    print(
        f"Pantallas de fase: "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Semilla: "
        f"{REALIZATION_SEED}"
    )

    # --------------------------------------------------------
    # Input field
    # --------------------------------------------------------

    grid, input_field = (
        create_bg3_input()
    )

    # --------------------------------------------------------
    # Moderate turbulence
    # --------------------------------------------------------

    print()
    print(
        "Propagando turbulencia moderada..."
    )

    moderate_field = (
        propagate_one_realization(
            input_field=input_field,
            r0_screen=MODERATE_R0_SCREEN,
            seed=REALIZATION_SEED,
        )
    )

    # --------------------------------------------------------
    # Strong turbulence
    # --------------------------------------------------------

    print(
        "Propagando turbulencia fuerte..."
    )

    strong_field = (
        propagate_one_realization(
            input_field=input_field,
            r0_screen=STRONG_R0_SCREEN,
            seed=REALIZATION_SEED,
        )
    )

    # --------------------------------------------------------
    # Power conservation
    # --------------------------------------------------------

    print_power_summary(
        input_field=input_field,
        moderate_field=moderate_field,
        strong_field=strong_field,
    )

    # --------------------------------------------------------
    # Intensities
    # --------------------------------------------------------

    input_intensity = (
        calculate_intensity(
            input_field
        )
    )

    moderate_intensity = (
        calculate_intensity(
            moderate_field
        )
    )

    strong_intensity = (
        calculate_intensity(
            strong_field
        )
    )

    (
        input_intensity,
        moderate_intensity,
        strong_intensity,
    ) = normalize_intensities(
        input_intensity=input_intensity,
        moderate_intensity=moderate_intensity,
        strong_intensity=strong_intensity,
    )


    save_composite_figure(
        input_intensity=input_intensity,
        moderate_intensity=moderate_intensity,
        strong_intensity=strong_intensity,
    )    
    
    # --------------------------------------------------------
    # Save figures
    # --------------------------------------------------------

    save_intensity_map(
        intensity=input_intensity,
        filename="bg3_input.png",
    )

    save_intensity_map(
        intensity=moderate_intensity,
        filename="bg3_moderate.png",
    )

    save_intensity_map(
        intensity=strong_intensity,
        filename="bg3_strong.png",
    )

    print()
    print(
        "Figuras guardadas en:"
    )

    print(
        OUTPUT_DIRECTORY
    )

    print()


if __name__ == "__main__":
    main()
