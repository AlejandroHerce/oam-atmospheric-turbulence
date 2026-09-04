"""
Charge-inversion control for Chapter 5.

Purpose
-------
Test whether the asymmetry of the ensemble-averaged OAM spectrum
changes consistently under

    ell_0 -> -ell_0.

The control uses

    - Laguerre-Gaussian beams;
    - Kolmogorov turbulence;
    - ell_0 = -1, -2, -3;
    - weak, moderate, and strong turbulence;
    - 500 realizations by default.

The production results under results/chapter_5/ are never modified.

Negative-charge results are stored under

    results/chapter_5/controls/charge_inversion/
        kolmogorov/<regime>/LGm01/
        kolmogorov/<regime>/LGm02/
        kolmogorov/<regime>/LGm03/

For each control scenario the script stores

    metrics.csv
    oam_spectra.npz
    mean_oam_spectrum.csv
    metadata.json

The script intentionally performs no scientific plotting.
"""

import argparse
import csv
import json
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np


# ============================================================
# Configuration imports
# ============================================================

from configs.chapter_2 import (
    W0_LG,
)

from configs.chapter_3 import (
    KOLMOGOROV_SUBHARMONIC_LEVEL,
)

from configs.chapter_4 import (
    DX,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    MODERATE_R0_SCREEN,
    MODERATE_R0_TOTAL,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    STRONG_R0_SCREEN,
    STRONG_R0_TOTAL,
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
    calculate_rms_oam_spread,
    modal_power_at_charge,
)

from src.phase_screens import (
    kolmogorov_phase_screen_with_subharmonics,
)

from src.propagation import (
    angular_spectrum_propagation,
)


# ============================================================
# Control configuration
# ============================================================

DEFAULT_NUMBER_OF_REALIZATIONS = 500

DEFAULT_NUMBER_OF_WORKERS = min(
    12,
    os.cpu_count() or 1,
)

MASTER_SEED = 20260902

ELL_MIN = -240
ELL_MAX = 240

RADIAL_SAMPLES = 256
AZIMUTHAL_SAMPLES = 720

CONTROL_ROOT = Path(
    "results/chapter_5/controls/charge_inversion"
)

PSD_NAME = "kolmogorov"

NEGATIVE_CHARGES = (
    -1,
    -2,
    -3,
)

REGIMES = (
    "weak",
    "moderate",
    "strong",
)


# ============================================================
# Turbulence configuration
# ============================================================

REGIME_PARAMETERS = {
    "weak": {
        "r0_total": WEAK_R0_TOTAL,
        "r0_screen": WEAK_R0_SCREEN,
    },
    "moderate": {
        "r0_total": MODERATE_R0_TOTAL,
        "r0_screen": MODERATE_R0_SCREEN,
    },
    "strong": {
        "r0_total": STRONG_R0_TOTAL,
        "r0_screen": STRONG_R0_SCREEN,
    },
}


REGIME_CODES = {
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


# ============================================================
# Naming
# ============================================================

def beam_name_from_charge(
    charge: int,
) -> str:

    if charge >= 0:
        raise ValueError(
            "This control script is reserved for negative charges."
        )

    return (
        f"LGm{abs(charge):02d}"
    )


def scenario_output_directory(
    charge: int,
    regime_name: str,
) -> Path:

    return (
        CONTROL_ROOT
        / PSD_NAME
        / regime_name
        / beam_name_from_charge(
            charge
        )
    )


# ============================================================
# Beam normalization
# ============================================================

def normalize_field(
    field: np.ndarray,
) -> np.ndarray:

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
            "Invalid input beam power."
        )

    return (
        field
        / np.sqrt(power)
    )


# ============================================================
# Negative-charge LG beam
# ============================================================

def create_input_beam(
    charge: int,
):

    if charge >= 0:
        raise ValueError(
            "Charge must be negative."
        )

    grid = create_grid(
        n=N_GRID,
        window_size=L_WINDOW,
    )

    if not np.isclose(
        grid.dx,
        DX,
    ):
        raise RuntimeError(
            "Grid spacing does not match configuration."
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
        ) ** abs(charge)
        * np.exp(
            -radius**2
            / W0_LG**2
        )
        * np.exp(
            1j
            * charge
            * azimuth
        )
    )

    field = np.asarray(
        field,
        dtype=np.complex128,
    )

    field = normalize_field(
        field
    )

    return (
        grid,
        field,
    )


# ============================================================
# Intensity centroid
# ============================================================

def calculate_intensity_centroid(
    field: np.ndarray,
    grid,
) -> tuple[
    float,
    float,
]:

    intensity = (
        np.abs(field) ** 2
    )

    total_intensity = float(
        np.sum(
            intensity
        )
    )

    if (
        not np.isfinite(
            total_intensity
        )
        or total_intensity <= 0.0
    ):
        raise RuntimeError(
            "Invalid field intensity."
        )

    x_centroid = float(
        np.sum(
            grid.X
            * intensity
        )
        / total_intensity
    )

    y_centroid = float(
        np.sum(
            grid.Y
            * intensity
        )
        / total_intensity
    )

    return (
        x_centroid,
        y_centroid,
    )


# ============================================================
# OAM entropy
# ============================================================

def calculate_normalized_oam_entropy(
    modal_power: np.ndarray,
) -> float:

    probabilities = np.asarray(
        modal_power,
        dtype=np.float64,
    )

    probabilities = (
        probabilities
        / np.sum(
            probabilities
        )
    )

    positive = (
        probabilities > 0.0
    )

    entropy = float(
        -np.sum(
            probabilities[
                positive
            ]
            * np.log2(
                probabilities[
                    positive
                ]
            )
        )
    )

    return float(
        entropy
        / np.log2(
            probabilities.size
        )
    )


# ============================================================
# Deterministic random hierarchy
# ============================================================

def generate_realization_seeds(
    charge: int,
    regime_name: str,
    number_of_realizations: int,
) -> list[int]:
    """
    Generate deterministic seeds for one control scenario.

    Negative-charge controls use a seed hierarchy independent
    from the Chapter 5 production campaign.
    """

    root_sequence = np.random.SeedSequence(
        [
            int(MASTER_SEED),
            int(abs(charge)),
            int(
                REGIME_CODES[
                    regime_name
                ]
            ),
        ]
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

    root = np.random.SeedSequence(
        int(
            realization_seed
        )
    )

    children = root.spawn(
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
# Kolmogorov phase screen
# ============================================================

def generate_phase_screen(
    r0_screen: float,
    rng: np.random.Generator,
) -> np.ndarray:

    return (
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


# ============================================================
# One realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
    charge: int,
    regime_name: str,
) -> dict:

    grid, field = (
        create_input_beam(
            charge
        )
    )

    r0_screen = (
        REGIME_PARAMETERS[
            regime_name
        ]["r0_screen"]
    )

    screen_seeds = (
        generate_screen_seeds(
            realization_seed
        )
    )

    # --------------------------------------------------------
    # Split-step propagation
    # --------------------------------------------------------

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
            generate_phase_screen(
                r0_screen=r0_screen,
                rng=screen_rng,
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

    # --------------------------------------------------------
    # Total power
    # --------------------------------------------------------

    total_power = float(
        np.sum(
            np.abs(field) ** 2
        )
        * DX**2
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    (
        centroid_x,
        centroid_y,
    ) = calculate_intensity_centroid(
        field=field,
        grid=grid,
    )

    # --------------------------------------------------------
    # OAM spectrum
    # --------------------------------------------------------

    (
        ell_values,
        modal_power,
    ) = calculate_oam_spectrum(
        field=field,
        grid=grid,
        ell_min=ELL_MIN,
        ell_max=ELL_MAX,
        radial_samples=RADIAL_SAMPLES,
        azimuthal_samples=AZIMUTHAL_SAMPLES,
    )

    # --------------------------------------------------------
    # Modal metrics
    # --------------------------------------------------------

    retention = modal_power_at_charge(
        ell_values=ell_values,
        modal_power=modal_power,
        charge=charge,
    )

    spread = calculate_rms_oam_spread(
        ell_values=ell_values,
        modal_power=modal_power,
        transmitted_charge=charge,
    )

    entropy = (
        calculate_normalized_oam_entropy(
            modal_power
        )
    )

    return {
        "seed":
            realization_seed,

        "retention":
            retention,

        "oam_rms_spread":
            spread,

        "normalized_entropy":
            entropy,

        "centroid_x":
            centroid_x,

        "centroid_y":
            centroid_y,

        "total_power":
            total_power,

        "modal_power":
            modal_power,
    }


# ============================================================
# Ensemble
# ============================================================

def run_ensemble(
    realization_seeds: list[int],
    charge: int,
    regime_name: str,
    number_of_workers: int,
) -> dict:

    worker = partial(
        simulate_one_realization,
        charge=charge,
        regime_name=regime_name,
    )

    n_realizations = (
        len(
            realization_seeds
        )
    )

    n_modes = (
        ELL_MAX
        - ELL_MIN
        + 1
    )

    results = {
        "seed":
            np.zeros(
                n_realizations,
                dtype=np.uint64,
            ),

        "retention":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "oam_rms_spread":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "normalized_entropy":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "centroid_x":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "centroid_y":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "total_power":
            np.zeros(
                n_realizations,
                dtype=np.float64,
            ),

        "spectra":
            np.zeros(
                (
                    n_realizations,
                    n_modes,
                ),
                dtype=np.float64,
            ),
    }

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        iterator = executor.map(
            worker,
            realization_seeds,
            chunksize=1,
        )

        for index, item in enumerate(
            iterator
        ):

            for key in (
                "seed",
                "retention",
                "oam_rms_spread",
                "normalized_entropy",
                "centroid_x",
                "centroid_y",
                "total_power",
            ):

                results[
                    key
                ][index] = item[
                    key
                ]

            results[
                "spectra"
            ][index] = item[
                "modal_power"
            ]

            completed = (
                index + 1
            )

            if (
                completed == 1
                or completed % 25 == 0
                or completed == n_realizations
            ):

                print(
                    "Realizaciones completadas: "
                    f"{completed}/"
                    f"{n_realizations}"
                )

    return results


# ============================================================
# Saving
# ============================================================

def save_metrics(
    output_directory: Path,
    results: dict,
) -> None:

    filename = (
        output_directory
        / "metrics.csv"
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
                "realization",
                "seed",
                "retention",
                "oam_rms_spread",
                "normalized_oam_entropy",
                "centroid_x_m",
                "centroid_y_m",
                "centroid_radius_m",
                "total_power",
            ]
        )

        for index in range(
            results[
                "retention"
            ].size
        ):

            radius = float(
                np.hypot(
                    results[
                        "centroid_x"
                    ][index],
                    results[
                        "centroid_y"
                    ][index],
                )
            )

            writer.writerow(
                [
                    index,
                    int(
                        results[
                            "seed"
                        ][index]
                    ),
                    results[
                        "retention"
                    ][index],
                    results[
                        "oam_rms_spread"
                    ][index],
                    results[
                        "normalized_entropy"
                    ][index],
                    results[
                        "centroid_x"
                    ][index],
                    results[
                        "centroid_y"
                    ][index],
                    radius,
                    results[
                        "total_power"
                    ][index],
                ]
            )


def save_oam_data(
    output_directory: Path,
    results: dict,
) -> None:

    ell_values = np.arange(
        ELL_MIN,
        ELL_MAX + 1,
        dtype=np.int64,
    )

    np.savez_compressed(
        output_directory
        / "oam_spectra.npz",
        ell_values=ell_values,
        modal_power=results[
            "spectra"
        ],
    )

    mean_spectrum = np.mean(
        results[
            "spectra"
        ],
        axis=0,
    )

    mean_spectrum /= np.sum(
        mean_spectrum
    )

    np.savetxt(
        output_directory
        / "mean_oam_spectrum.csv",
        np.column_stack(
            (
                ell_values,
                mean_spectrum,
            )
        ),
        delimiter=",",
        header=(
            "ell,mean_modal_power"
        ),
        comments="",
    )


def save_metadata(
    output_directory: Path,
    charge: int,
    regime_name: str,
    number_of_realizations: int,
) -> None:

    turbulence = (
        REGIME_PARAMETERS[
            regime_name
        ]
    )

    metadata = {
        "control":
            "charge_inversion",

        "beam_family":
            "LG",

        "radial_index":
            0,

        "transmitted_ell":
            charge,

        "psd":
            PSD_NAME,

        "regime":
            regime_name,

        "number_of_realizations":
            number_of_realizations,

        "master_seed":
            MASTER_SEED,

        "wavelength_m":
            WAVELENGTH,

        "total_propagation_distance_m":
            TOTAL_PROPAGATION_DISTANCE,

        "number_of_phase_screens":
            NUMBER_OF_PHASE_SCREENS,

        "r0_total_m":
            turbulence[
                "r0_total"
            ],

        "r0_screen_m":
            turbulence[
                "r0_screen"
            ],

        "grid_size":
            N_GRID,

        "window_size_m":
            L_WINDOW,

        "dx_m":
            DX,

        "ell_min":
            ELL_MIN,

        "ell_max":
            ELL_MAX,

        "radial_samples":
            RADIAL_SAMPLES,

        "azimuthal_samples":
            AZIMUTHAL_SAMPLES,

        "subharmonic_level":
            KOLMOGOROV_SUBHARMONIC_LEVEL,
    }

    with (
        output_directory
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


def save_scenario(
    output_directory: Path,
    results: dict,
    charge: int,
    regime_name: str,
) -> None:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_metrics(
        output_directory,
        results,
    )

    save_oam_data(
        output_directory,
        results,
    )

    save_metadata(
        output_directory=output_directory,
        charge=charge,
        regime_name=regime_name,
        number_of_realizations=(
            results[
                "retention"
            ].size
        ),
    )


# ============================================================
# One complete control scenario
# ============================================================

def run_control_scenario(
    charge: int,
    regime_name: str,
    number_of_realizations: int,
    number_of_workers: int,
) -> None:

    output_directory = (
        scenario_output_directory(
            charge=charge,
            regime_name=regime_name,
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Charge-inversion control"
    )
    print(
        "=" * 70
    )
    print(
        f"Beam: LG, p=0, ell={charge}"
    )
    print(
        f"PSD: {PSD_NAME}"
    )
    print(
        f"Regime: {regime_name}"
    )
    print(
        f"Realizations: {number_of_realizations}"
    )
    print(
        f"Workers: {number_of_workers}"
    )
    print()

    seeds = (
        generate_realization_seeds(
            charge=charge,
            regime_name=regime_name,
            number_of_realizations=(
                number_of_realizations
            ),
        )
    )

    results = (
        run_ensemble(
            realization_seeds=seeds,
            charge=charge,
            regime_name=regime_name,
            number_of_workers=(
                number_of_workers
            ),
        )
    )

    save_scenario(
        output_directory=(
            output_directory
        ),
        results=results,
        charge=charge,
        regime_name=regime_name,
    )

    print()
    print(
        "Saved in:"
    )
    print(
        output_directory.resolve()
    )


# ============================================================
# CLI
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

    parser.add_argument(
        "--charge",
        type=int,
        choices=NEGATIVE_CHARGES,
        default=None,
        help=(
            "Run only one negative charge. "
            "If omitted, run -1, -2 and -3."
        ),
    )

    parser.add_argument(
        "--regime",
        choices=REGIMES,
        default=None,
        help=(
            "Run only one turbulence regime. "
            "If omitted, run all regimes."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    arguments = (
        parse_arguments()
    )

    if arguments.realizations <= 0:
        raise ValueError(
            "--realizations must be positive."
        )

    if arguments.workers <= 0:
        raise ValueError(
            "--workers must be positive."
        )

    charges = (
        (arguments.charge,)
        if arguments.charge is not None
        else NEGATIVE_CHARGES
    )

    regimes = (
        (arguments.regime,)
        if arguments.regime is not None
        else REGIMES
    )

    print()
    print(
        "Chapter 5 charge-inversion control"
    )
    print(
        "=================================="
    )
    print(
        f"Charges: {charges}"
    )
    print(
        f"Regimes: {regimes}"
    )
    print(
        f"Realizations per scenario: "
        f"{arguments.realizations}"
    )
    print(
        f"Workers: {arguments.workers}"
    )

    for regime_name in regimes:

        for charge in charges:

            run_control_scenario(
                charge=charge,
                regime_name=regime_name,
                number_of_realizations=(
                    arguments.realizations
                ),
                number_of_workers=(
                    arguments.workers
                ),
            )

    print()
    print(
        "=" * 70
    )
    print(
        "All requested charge-inversion controls completed."
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
