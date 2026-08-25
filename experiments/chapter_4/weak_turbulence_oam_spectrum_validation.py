"""
Weak-turbulence OAM-spectrum validation.

A transmitted LG_0^1 beam is propagated through weak
Kolmogorov turbulence using the complete Split-Step model.

The ensemble-averaged numerical OAM spectrum is compared with
the weak-turbulence rotational-coherence prediction derived by
Paterson (2005).

The full OAM interval used in the final simulations is retained
for the numerical and theoretical spectra.
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
    HALF_SCREEN_SPACING,
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    TOTAL_PROPAGATION_DISTANCE,
    WAVELENGTH,
    WEAK_R0_SCREEN,
    WEAK_R0_TOTAL,
)

from src.grids import (
    create_grid,
)

from src.oam import (
    calculate_oam_spectrum,
)

from src.oam_theory import (
    paterson_lg_oam_spectrum,
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

TRANSMITTED_ELL = 1

ELL_MIN = -240
ELL_MAX = 240

RADIAL_SAMPLES = 256
AZIMUTHAL_SAMPLES = 720

DEFAULT_NUMBER_OF_REALIZATIONS = 100

DEFAULT_NUMBER_OF_WORKERS = min(
    12,
    os.cpu_count() or 1,
)

# Use the same Kolmogorov subharmonic level adopted in the
# definitive Chapter 4 configuration.
#
# Change only this value if your final configuration stores it
# elsewhere.
SUBHARMONIC_LEVEL = 9

MASTER_SEED = 20260824

BOOTSTRAP_SEED = 20260825
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/"
    "weak_turbulence_oam_spectrum_validation"
)


# ============================================================
# Input LG_0^ell beam
# ============================================================

def create_input_beam():
    """
    Generate and normalize an LG_0^ell beam at its waist.
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
            "Grid spacing does not match Chapter 4 configuration."
        )

    radius = np.hypot(
        grid.X,
        grid.Y,
    )

    azimuth = np.arctan2(
        grid.Y,
        grid.X,
    )

    field = (
        (
            np.sqrt(2.0)
            * radius
            / W0_LG
        ) ** abs(
            TRANSMITTED_ELL
        )
        * np.exp(
            -radius**2
            / W0_LG**2
        )
        * np.exp(
            1j
            * TRANSMITTED_ELL
            * azimuth
        )
    ).astype(
        np.complex128
    )

    power = float(
        np.sum(
            np.abs(field) ** 2
        )
        * DX**2
    )

    if (
        not np.isfinite(power)
        or power <= 0.0
    ):
        raise RuntimeError(
            "Invalid LG input power."
        )

    field /= np.sqrt(
        power
    )

    return (
        grid,
        field,
    )


# ============================================================
# Seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
) -> list[int]:
    """
    Generate deterministic independent atmospheric realization
    seeds.
    """

    root_sequence = np.random.SeedSequence(
        MASTER_SEED
    )

    children = root_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in children
    ]


def generate_screen_seeds(
    realization_seed: int,
) -> list[int]:
    """
    Generate one deterministic independent stream per screen.
    """

    realization_sequence = np.random.SeedSequence(
        int(
            realization_seed
        )
    )

    children = realization_sequence.spawn(
        NUMBER_OF_PHASE_SCREENS
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in children
    ]


# ============================================================
# One atmospheric realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
) -> np.ndarray:
    """
    Propagate one realization and return its final normalized
    OAM spectrum.
    """

    grid, field = (
        create_input_beam()
    )

    screen_seeds = (
        generate_screen_seeds(
            realization_seed
        )
    )

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        screen_rng = np.random.default_rng(
            screen_seeds[
                screen_index
            ]
        )

        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=WEAK_R0_SCREEN,
                n_subharmonics=(
                    SUBHARMONIC_LEVEL
                ),
                rng=screen_rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j
            * phase_screen
        )

        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

    (
        ell_values,
        modal_power,
    ) = calculate_oam_spectrum(
        field=field,
        grid=grid,
        ell_min=ELL_MIN,
        ell_max=ELL_MAX,
        radial_samples=(
            RADIAL_SAMPLES
        ),
        azimuthal_samples=(
            AZIMUTHAL_SAMPLES
        ),
    )

    expected_ell = np.arange(
        ELL_MIN,
        ELL_MAX + 1,
        dtype=np.int64,
    )

    if not np.array_equal(
        ell_values,
        expected_ell,
    ):
        raise RuntimeError(
            "Unexpected OAM index array."
        )

    return modal_power


# ============================================================
# Ensemble
# ============================================================

def run_ensemble(
    number_of_realizations: int,
    number_of_workers: int,
) -> np.ndarray:
    """
    Run all Split-Step realizations.
    """

    seeds = generate_realization_seeds(
        number_of_realizations
    )

    number_of_modes = (
        ELL_MAX
        - ELL_MIN
        + 1
    )

    spectra = np.zeros(
        (
            number_of_realizations,
            number_of_modes,
        ),
        dtype=np.float64,
    )

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            simulate_one_realization,
            seeds,
            chunksize=1,
        )

        for index, spectrum in enumerate(
            results
        ):

            spectra[
                index
            ] = spectrum

            completed = (
                index + 1
            )

            if (
                completed == 1
                or completed % 25 == 0
                or completed
                == number_of_realizations
            ):

                print(
                    "Realizaciones completadas: "
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return spectra


# ============================================================
# Bootstrap confidence interval of mean spectrum
# ============================================================

def bootstrap_mean_spectrum(
    spectra: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Bootstrap confidence interval for the ensemble-mean OAM
    spectrum.
    """

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    number_of_realizations = (
        spectra.shape[0]
    )

    number_of_modes = (
        spectra.shape[1]
    )

    bootstrap_means = np.zeros(
        (
            BOOTSTRAP_SAMPLES,
            number_of_modes,
        ),
        dtype=np.float64,
    )

    for bootstrap_index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            number_of_realizations,
            size=number_of_realizations,
        )

        bootstrap_means[
            bootstrap_index
        ] = np.mean(
            spectra[
                indices
            ],
            axis=0,
        )

    alpha = (
        1.0
        - BOOTSTRAP_CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_means,
        alpha / 2.0,
        axis=0,
    )

    upper = np.quantile(
        bootstrap_means,
        1.0 - alpha / 2.0,
        axis=0,
    )

    return (
        lower,
        upper,
    )


# ============================================================
# Comparison metrics
# ============================================================

def calculate_l1_distance(
    numerical: np.ndarray,
    theoretical: np.ndarray,
) -> float:
    """
    L1 distance between normalized spectra.
    """

    return float(
        np.sum(
            np.abs(
                numerical
                - theoretical
            )
        )
    )


def calculate_edge_modal_power(
    spectrum: np.ndarray,
    number_of_edge_modes: int = 10,
) -> float:
    """
    Fraction of normalized OAM power contained in the modes
    closest to the two boundaries of the analyzed interval.

    This is used only as a truncation diagnostic.
    """

    return float(
        np.sum(
            spectrum[
                :number_of_edge_modes
            ]
        )
        + np.sum(
            spectrum[
                -number_of_edge_modes:
            ]
        )
    )


# ============================================================
# Save results
# ============================================================

def save_results(
    ell_values: np.ndarray,
    spectra: np.ndarray,
    numerical_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theoretical: np.ndarray,
) -> None:
    """
    Save raw ensemble spectra and the final comparison.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        OUTPUT_DIRECTORY
        / "oam_spectra_samples.csv",
        spectra,
        delimiter=",",
    )

    filename = (
        OUTPUT_DIRECTORY
        / "oam_spectrum_validation_summary.csv"
    )

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "ell",
                "split_step_mean",
                "ci95_lower",
                "ci95_upper",
                "paterson_theory",
                "absolute_difference",
            ]
        )

        for index, ell in enumerate(
            ell_values
        ):

            writer.writerow(
                [
                    ell,
                    numerical_mean[
                        index
                    ],
                    lower[
                        index
                    ],
                    upper[
                        index
                    ],
                    theoretical[
                        index
                    ],
                    abs(
                        numerical_mean[
                            index
                        ]
                        - theoretical[
                            index
                        ]
                    ),
                ]
            )


# ============================================================
# Plot
# ============================================================

def plot_validation(
    ell_values: np.ndarray,
    numerical_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theoretical: np.ndarray,
) -> None:
    """
    Plot the relevant central portion of the OAM spectrum.

    The calculation itself always uses the full interval
    [ELL_MIN, ELL_MAX].
    """

    # Automatically display modes carrying relevant power.
    relevant = (
        (numerical_mean > 1.0e-4)
        | (theoretical > 1.0e-4)
        | (upper > 1.0e-4)
    )

    positions = np.where(
        relevant
    )[0]

    if positions.size == 0:

        plot_min = max(
            ELL_MIN,
            TRANSMITTED_ELL - 10,
        )

        plot_max = min(
            ELL_MAX,
            TRANSMITTED_ELL + 10,
        )

    else:

        padding = 3

        plot_min = max(
            ELL_MIN,
            int(
                ell_values[
                    positions[0]
                ]
            )
            - padding,
        )

        plot_max = min(
            ELL_MAX,
            int(
                ell_values[
                    positions[-1]
                ]
            )
            + padding,
        )

    mask = (
        (ell_values >= plot_min)
        & (ell_values <= plot_max)
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 4.8)
    )

    axis.plot(
        ell_values[
            mask
        ],
        numerical_mean[
            mask
        ],
        marker="o",
        linewidth=1.5,
        label="Split-Step",
    )

    axis.fill_between(
        ell_values[
            mask
        ],
        lower[
            mask
        ],
        upper[
            mask
        ],
        alpha=0.2,
        label="IC bootstrap 95 %",
    )

    axis.plot(
        ell_values[
            mask
        ],
        theoretical[
            mask
        ],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="Paterson",
    )

    axis.set_xlabel(
        r"Índice OAM $\ell$"
    )

    axis.set_ylabel(
        r"Potencia modal normalizada $P_\ell$"
    )

    axis.set_title(
        r"Validación del espectro OAM de $LG_0^1$"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "weak_turbulence_oam_spectrum_validation.png",
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
        default=(
            DEFAULT_NUMBER_OF_REALIZATIONS
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=(
            DEFAULT_NUMBER_OF_WORKERS
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    arguments = parse_arguments()

    if arguments.realizations <= 0:
        raise ValueError(
            "realizations must be positive."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ell_values = np.arange(
        ELL_MIN,
        ELL_MAX + 1,
        dtype=np.int64,
    )

    print()
    print(
        "Validación del espectro OAM "
        "-- turbulencia débil"
    )

    print(
        "============================================"
    )

    print(
        rf"Haz: LG_0^{TRANSMITTED_ELL}"
    )

    print(
        f"Rango OAM: "
        f"[{ELL_MIN}, {ELL_MAX}]"
    )

    print(
        f"r0 total = "
        f"{WEAK_R0_TOTAL:.8e} m"
    )

    print(
        f"r0 por pantalla = "
        f"{WEAK_R0_SCREEN:.8e} m"
    )

    print(
        f"Pantallas = "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Subarmónicos = "
        f"{SUBHARMONIC_LEVEL}"
    )

    print(
        f"Realizaciones = "
        f"{arguments.realizations}"
    )

    print(
        f"Workers = "
        f"{arguments.workers}"
    )

    # --------------------------------------------------------
    # Numerical Split-Step spectrum
    # --------------------------------------------------------

    spectra = run_ensemble(
        number_of_realizations=(
            arguments.realizations
        ),
        number_of_workers=(
            arguments.workers
        ),
    )

    numerical_mean = np.mean(
        spectra,
        axis=0,
    )

    (
        lower,
        upper,
    ) = bootstrap_mean_spectrum(
        spectra
    )

    # --------------------------------------------------------
    # Independent theoretical spectrum
    # --------------------------------------------------------

    theoretical = (
        paterson_lg_oam_spectrum(
            ell_values=ell_values,
            transmitted_ell=(
                TRANSMITTED_ELL
            ),
            w0=W0_LG,
            wavelength=WAVELENGTH,
            propagation_distance=(
                TOTAL_PROPAGATION_DISTANCE
            ),
            r0_total=WEAK_R0_TOTAL,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    l1_distance = (
        calculate_l1_distance(
            numerical=numerical_mean,
            theoretical=theoretical,
        )
    )

    transmitted_index = int(
        np.where(
            ell_values
            == TRANSMITTED_ELL
        )[0][0]
    )

    numerical_retention = float(
        numerical_mean[
            transmitted_index
        ]
    )

    theoretical_retention = float(
        theoretical[
            transmitted_index
        ]
    )

    retention_lower = float(
        lower[
            transmitted_index
        ]
    )
    
    retention_upper = float(
        upper[
            transmitted_index
        ]
    )
    
    theory_inside_retention_ci = (
        retention_lower
        <= theoretical_retention
        <= retention_upper
    )

    retention_error = (
        100.0
        * abs(
            numerical_retention
            - theoretical_retention
        )
        / theoretical_retention
    )

    numerical_edge_power = (
        calculate_edge_modal_power(
            numerical_mean
        )
    )

    theoretical_edge_power = (
        calculate_edge_modal_power(
            theoretical
        )
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print(
        "Comparación global"
    )

    print(
        "=================="
    )

    print(
        f"Distancia L1 espectral = "
        f"{l1_distance:.6e}"
    )

    print()

    print(
        "Retención del modo transmitido"
    )
    
    print(
        "------------------------------"
    )
    
    print(
        f"Split-Step = "
        f"{numerical_retention:.8e}"
    )
    
    print(
        f"IC95%      = "
        f"[{retention_lower:.8e}, "
        f"{retention_upper:.8e}]"
    )
    
    print(
        f"Paterson   = "
        f"{theoretical_retention:.8e}"
    )
    
    print(
        f"Error      = "
        f"{retention_error:.4f} %"
    )
    
    print(
        "Paterson dentro del IC95%: "
        f"{'sí' if theory_inside_retention_ci else 'no'}"
    )

    print()

    print(
        "Diagnóstico de truncamiento modal"
    )

    print(
        "--------------------------------"
    )

    print(
        "Potencia en los 10 modos "
        "de cada extremo:"
    )

    print(
        f"Split-Step = "
        f"{numerical_edge_power:.8e}"
    )

    print(
        f"Paterson   = "
        f"{theoretical_edge_power:.8e}"
    )

    # --------------------------------------------------------
    # Save and plot
    # --------------------------------------------------------

    save_results(
        ell_values=ell_values,
        spectra=spectra,
        numerical_mean=numerical_mean,
        lower=lower,
        upper=upper,
        theoretical=theoretical,
    )

    plot_validation(
        ell_values=ell_values,
        numerical_mean=numerical_mean,
        lower=lower,
        upper=upper,
        theoretical=theoretical,
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        OUTPUT_DIRECTORY.resolve()
    )


if __name__ == "__main__":
    main()
