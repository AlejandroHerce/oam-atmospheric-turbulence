"""
Run one Chapter 5 production scenario.

A scenario is uniquely defined by

    beam + turbulence PSD + turbulence regime.

Supported beams
---------------
    LG01, LG02, LG03
    BG01, BG02, BG03

Supported PSD models
--------------------
    kolmogorov
    von_karman
    modified_von_karman

Supported turbulence regimes
----------------------------
    weak
    moderate
    strong

For every atmospheric realization the script stores

    - transmitted-mode retention;
    - RMS OAM spread;
    - normalized Shannon entropy;
    - intensity centroid coordinates;
    - final total optical power;
    - complete normalized OAM spectrum.

The output of every scenario is stored independently under

    results/chapter_5/<psd>/<regime>/<beam>/

The script performs no scientific plotting. Analysis and visualization
are intentionally separated from production simulations.
"""

import argparse
import csv
import json
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from scipy.special import jv


# ============================================================
# Configuration imports
# ============================================================

from configs.chapter_2 import (
    BG_PARAMETERS,
    W0_LG,
)

from configs.chapter_3 import (
    INNER_SCALE,
    KOLMOGOROV_SUBHARMONIC_LEVEL,
    MODIFIED_VON_KARMAN_SUBHARMONIC_LEVEL,
    OUTER_SCALE,
    VON_KARMAN_SUBHARMONIC_LEVEL,
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
    modified_von_karman_phase_screen,
    modified_von_karman_subharmonics,
    von_karman_phase_screen,
    von_karman_subharmonics,
)

from src.propagation import (
    angular_spectrum_propagation,
)


# ============================================================
# Global production configuration
# ============================================================

DEFAULT_NUMBER_OF_REALIZATIONS = 1750

DEFAULT_NUMBER_OF_WORKERS = min(
    12,
    os.cpu_count() or 1,
)

MASTER_SEED = 20260825

ELL_MIN = -240
ELL_MAX = 240

RADIAL_SAMPLES = 256
AZIMUTHAL_SAMPLES = 720

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

RESULTS_ROOT = Path(
    "results/chapter_5"
)


# ============================================================
# Scenario identifiers
# ============================================================

BEAM_CODES = {
    "LG01": 1,
    "LG02": 2,
    "LG03": 3,
    "BG01": 4,
    "BG02": 5,
    "BG03": 6,
}

PSD_CODES = {
    "kolmogorov": 1,
    "von_karman": 2,
    "modified_von_karman": 3,
}

REGIME_CODES = {
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


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


# ============================================================
# Beam configuration
# ============================================================

def transmitted_charge(
    beam_name: str,
) -> int:
    """
    Return the transmitted OAM charge associated with one beam.
    """

    mapping = {
        "LG01": 1,
        "LG02": 2,
        "LG03": 3,
        "BG01": 1,
        "BG02": 2,
        "BG03": 3,
    }

    return mapping[
        beam_name
    ]


def bg_parameters(
    charge: int,
) -> tuple[
    float,
    float,
]:
    """
    Return Bessel-Gaussian w0 and kr.

    The function accepts either the dictionary format

        BG_PARAMETERS[charge]["w0"]
        BG_PARAMETERS[charge]["kr"]

    or a tuple-like format

        BG_PARAMETERS[charge] = (w0, kr).
    """

    parameters = (
        BG_PARAMETERS[
            charge
        ]
    )

    if isinstance(
        parameters,
        dict,
    ):

        return (
            float(
                parameters["w0"]
            ),
            float(
                parameters["kr"]
            ),
        )

    return (
        float(
            parameters[0]
        ),
        float(
            parameters[1]
        ),
    )


# ============================================================
# Output directory
# ============================================================

def scenario_output_directory(
    beam_name: str,
    psd_name: str,
    regime_name: str,
) -> Path:

    return (
        RESULTS_ROOT
        / psd_name
        / regime_name
        / beam_name
    )


# ============================================================
# Beam normalization
# ============================================================

def normalize_field(
    field: np.ndarray,
) -> np.ndarray:
    """
    Normalize the transverse optical field to unit power.
    """

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
# Input beam
# ============================================================

def create_input_beam(
    beam_name: str,
):
    """
    Generate one of the six definitive Chapter 5 beams.
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

    charge = transmitted_charge(
        beam_name
    )

    # --------------------------------------------------------
    # Laguerre-Gaussian, p = 0
    # --------------------------------------------------------

    if beam_name.startswith(
        "LG"
    ):

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

    # --------------------------------------------------------
    # Bessel-Gaussian
    # --------------------------------------------------------

    elif beam_name.startswith(
        "BG"
    ):

        (
            w0,
            kr,
        ) = bg_parameters(
            charge
        )

        field = (
            jv(
                charge,
                kr * radius,
            )
            * np.exp(
                -radius**2
                / w0**2
            )
            * np.exp(
                1j
                * charge
                * azimuth
            )
        )

    else:

        raise ValueError(
            f"Unknown beam: {beam_name}"
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
        raise ValueError(
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
# Random hierarchy
# ============================================================

def generate_realization_seeds(
    beam_name: str,
    psd_name: str,
    regime_name: str,
    number_of_realizations: int,
) -> list[int]:
    """
    Generate deterministic independent seeds for one complete
    Chapter 5 scenario.

    The root entropy is uniquely defined by

        master seed,
        beam,
        PSD,
        turbulence regime.

    Recreating the sequence with a larger ensemble reproduces
    exactly the previous prefix.
    """

    root_sequence = np.random.SeedSequence(
        [
            int(MASTER_SEED),
            int(
                BEAM_CODES[
                    beam_name
                ]
            ),
            int(
                PSD_CODES[
                    psd_name
                ]
            ),
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
    """
    Generate one independent deterministic stream per phase
    screen inside one atmospheric realization.
    """

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
# Phase-screen generation
# ============================================================

def generate_phase_screen(
    psd_name: str,
    r0_screen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate the complete phase screen associated with one PSD.
    """

    # --------------------------------------------------------
    # Kolmogorov
    # --------------------------------------------------------

    if psd_name == "kolmogorov":

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

    # --------------------------------------------------------
    # von Karman
    # --------------------------------------------------------

    if psd_name == "von_karman":

        phase_fft = (
            von_karman_phase_screen(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                outer_scale=OUTER_SCALE,
                rng=rng,
                remove_piston=False,
            )
        )

        phase_sub = (
            von_karman_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                outer_scale=OUTER_SCALE,
                n_subharmonics=(
                    VON_KARMAN_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=False,
            )
        )

    # --------------------------------------------------------
    # Modified von Karman
    # --------------------------------------------------------

    elif (
        psd_name
        == "modified_von_karman"
    ):

        phase_fft = (
            modified_von_karman_phase_screen(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                outer_scale=OUTER_SCALE,
                inner_scale=INNER_SCALE,
                rng=rng,
                remove_piston=False,
            )
        )

        phase_sub = (
            modified_von_karman_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=r0_screen,
                outer_scale=OUTER_SCALE,
                inner_scale=INNER_SCALE,
                n_subharmonics=(
                    MODIFIED_VON_KARMAN_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=False,
            )
        )

    else:

        raise ValueError(
            f"Unknown PSD: {psd_name}"
        )

    phase = (
        phase_fft
        + phase_sub
    )

    phase -= np.mean(
        phase
    )

    return phase


# ============================================================
# One realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
    beam_name: str,
    psd_name: str,
    regime_name: str,
) -> dict:

    grid, field = (
        create_input_beam(
            beam_name
        )
    )

    charge = transmitted_charge(
        beam_name
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
    # Split-Step propagation
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
                psd_name=psd_name,
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
    # Power
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
# Run ensemble
# ============================================================

def run_ensemble(
    realization_seeds: list[int],
    beam_name: str,
    psd_name: str,
    regime_name: str,
    number_of_workers: int,
) -> dict:

    worker = partial(
        simulate_one_realization,
        beam_name=beam_name,
        psd_name=psd_name,
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
# Bootstrap helper
# ============================================================

def bootstrap_scalar_statistic(
    number_of_realizations: int,
    statistic_function,
    seed: int,
) -> tuple[
    float,
    float,
]:

    rng = np.random.default_rng(
        seed
    )

    values = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            number_of_realizations,
            size=number_of_realizations,
        )

        values[
            index
        ] = statistic_function(
            indices
        )

    alpha = (
        1.0
        - BOOTSTRAP_CONFIDENCE_LEVEL
    )

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


# ============================================================
# Metadata
# ============================================================

def save_metadata(
    output_directory: Path,
    beam_name: str,
    psd_name: str,
    regime_name: str,
    number_of_realizations: int,
) -> None:

    charge = transmitted_charge(
        beam_name
    )

    turbulence = (
        REGIME_PARAMETERS[
            regime_name
        ]
    )

    metadata = {
        "beam":
            beam_name,

        "transmitted_ell":
            charge,

        "psd":
            psd_name,

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

        "outer_scale_m":
            (
                OUTER_SCALE
                if psd_name
                != "kolmogorov"
                else None
            ),

        "inner_scale_m":
            (
                INNER_SCALE
                if psd_name
                == "modified_von_karman"
                else None
            ),

        "subharmonic_level":
            {
                "kolmogorov":
                    KOLMOGOROV_SUBHARMONIC_LEVEL,

                "von_karman":
                    VON_KARMAN_SUBHARMONIC_LEVEL,

                "modified_von_karman":
                    MODIFIED_VON_KARMAN_SUBHARMONIC_LEVEL,
            }[
                psd_name
            ],
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


# ============================================================
# Raw metric saving
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


# ============================================================
# OAM data
# ============================================================

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


# ============================================================
# Scenario summary
# ============================================================

def save_summary(
    output_directory: Path,
    results: dict,
    transmitted_ell: int,
) -> None:
    """
    Store both

        <metric(P_j)>

    and

        metric(<P_j>).

    Beam wander is evaluated as

        sqrt(<x_c^2 + y_c^2>).
    """

    n = (
        results[
            "retention"
        ].size
    )

    ell_values = np.arange(
        ELL_MIN,
        ELL_MAX + 1,
        dtype=np.int64,
    )

    retention = (
        results[
            "retention"
        ]
    )

    spread = (
        results[
            "oam_rms_spread"
        ]
    )

    entropy = (
        results[
            "normalized_entropy"
        ]
    )

    centroid_x = (
        results[
            "centroid_x"
        ]
    )

    centroid_y = (
        results[
            "centroid_y"
        ]
    )

    total_power = (
        results[
            "total_power"
        ]
    )

    spectra = (
        results[
            "spectra"
        ]
    )

    # --------------------------------------------------------
    # Ensemble means
    # --------------------------------------------------------

    retention_mean = float(
        np.mean(
            retention
        )
    )

    spread_mean = float(
        np.mean(
            spread
        )
    )

    entropy_mean = float(
        np.mean(
            entropy
        )
    )

    retention_std = float(
        np.std(
            retention,
            ddof=1,
        )
    )

    spread_std = float(
        np.std(
            spread,
            ddof=1,
        )
    )

    entropy_std = float(
        np.std(
            entropy,
            ddof=1,
        )
    )

    # --------------------------------------------------------
    # Bootstrap means
    # --------------------------------------------------------

    retention_ci = (
        bootstrap_scalar_statistic(
            n,
            lambda i: float(
                np.mean(
                    retention[i]
                )
            ),
            seed=20260827,
        )
    )

    spread_ci = (
        bootstrap_scalar_statistic(
            n,
            lambda i: float(
                np.mean(
                    spread[i]
                )
            ),
            seed=20260828,
        )
    )

    entropy_ci = (
        bootstrap_scalar_statistic(
            n,
            lambda i: float(
                np.mean(
                    entropy[i]
                )
            ),
            seed=20260829,
        )
    )

    # --------------------------------------------------------
    # Beam wander
    # --------------------------------------------------------

    beam_wander = float(
        np.sqrt(
            np.mean(
                centroid_x**2
                + centroid_y**2
            )
        )
    )

    beam_wander_ci = (
        bootstrap_scalar_statistic(
            n,
            lambda i: float(
                np.sqrt(
                    np.mean(
                        centroid_x[i] ** 2
                        + centroid_y[i] ** 2
                    )
                )
            ),
            seed=20260831,
        )
    )

    # --------------------------------------------------------
    # Total power
    # --------------------------------------------------------

    power_mean = float(
        np.mean(
            total_power
        )
    )

    power_std = float(
        np.std(
            total_power,
            ddof=1,
        )
    )

    power_ci = (
        bootstrap_scalar_statistic(
            n,
            lambda i: float(
                np.mean(
                    total_power[i]
                )
            ),
            seed=20260830,
        )
    )

    # --------------------------------------------------------
    # Mean spectrum
    # --------------------------------------------------------

    mean_spectrum = np.mean(
        spectra,
        axis=0,
    )

    mean_spectrum /= np.sum(
        mean_spectrum
    )

    transmitted_index = int(
        np.where(
            ell_values
            == transmitted_ell
        )[0][0]
    )

    retention_ms = float(
        mean_spectrum[
            transmitted_index
        ]
    )

    spread_ms = float(
        np.sqrt(
            np.sum(
                (
                    ell_values
                    - transmitted_ell
                ) ** 2
                * mean_spectrum
            )
        )
    )

    positive = (
        mean_spectrum > 0.0
    )

    entropy_ms = float(
        -np.sum(
            mean_spectrum[
                positive
            ]
            * np.log2(
                mean_spectrum[
                    positive
                ]
            )
        )
        / np.log2(
            mean_spectrum.size
        )
    )

    # --------------------------------------------------------
    # Bootstrap nonlinear mean-spectrum metrics
    # --------------------------------------------------------

    def spread_of_mean_spectrum(
        indices,
    ):

        spectrum = np.mean(
            spectra[
                indices
            ],
            axis=0,
        )

        spectrum /= np.sum(
            spectrum
        )

        return float(
            np.sqrt(
                np.sum(
                    (
                        ell_values
                        - transmitted_ell
                    ) ** 2
                    * spectrum
                )
            )
        )

    def entropy_of_mean_spectrum(
        indices,
    ):

        spectrum = np.mean(
            spectra[
                indices
            ],
            axis=0,
        )

        spectrum /= np.sum(
            spectrum
        )

        positive = (
            spectrum > 0.0
        )

        return float(
            -np.sum(
                spectrum[
                    positive
                ]
                * np.log2(
                    spectrum[
                        positive
                    ]
                )
            )
            / np.log2(
                spectrum.size
            )
        )

    spread_ms_ci = (
        bootstrap_scalar_statistic(
            n,
            spread_of_mean_spectrum,
            seed=20260833,
        )
    )

    entropy_ms_ci = (
        bootstrap_scalar_statistic(
            n,
            entropy_of_mean_spectrum,
            seed=20260834,
        )
    )

    # Retention is linear, so the same estimate and CI apply
    # to both representations.
    retention_ms_ci = (
        retention_ci
    )

    # --------------------------------------------------------
    # Write summary
    # --------------------------------------------------------

    rows = (
        (
            "retention",
            retention_mean,
            retention_std,
            *retention_ci,
            "ensemble_mean_of_metric",
        ),
        (
            "oam_rms_spread",
            spread_mean,
            spread_std,
            *spread_ci,
            "ensemble_mean_of_metric",
        ),
        (
            "normalized_oam_entropy",
            entropy_mean,
            entropy_std,
            *entropy_ci,
            "ensemble_mean_of_metric",
        ),
        (
            "beam_wander_rms_m",
            beam_wander,
            "",
            *beam_wander_ci,
            "ensemble_statistic",
        ),
        (
            "total_power",
            power_mean,
            power_std,
            *power_ci,
            "ensemble_mean_of_metric",
        ),
        (
            "retention",
            retention_ms,
            "",
            *retention_ms_ci,
            "metric_of_mean_spectrum",
        ),
        (
            "oam_rms_spread",
            spread_ms,
            "",
            *spread_ms_ci,
            "metric_of_mean_spectrum",
        ),
        (
            "normalized_oam_entropy",
            entropy_ms,
            "",
            *entropy_ms_ci,
            "metric_of_mean_spectrum",
        ),
    )

    with (
        output_directory
        / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "metric",
                "estimate",
                "sample_std",
                "ci95_lower",
                "ci95_upper",
                "calculation",
            ]
        )

        writer.writerows(
            rows
        )


# ============================================================
# Save complete scenario
# ============================================================

def save_scenario(
    output_directory: Path,
    results: dict,
    beam_name: str,
    psd_name: str,
    regime_name: str,
) -> None:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_metadata(
        output_directory=(
            output_directory
        ),
        beam_name=beam_name,
        psd_name=psd_name,
        regime_name=regime_name,
        number_of_realizations=(
            results[
                "retention"
            ].size
        ),
    )

    save_metrics(
        output_directory,
        results,
    )

    save_oam_data(
        output_directory,
        results,
    )

    save_summary(
        output_directory=(
            output_directory
        ),
        results=results,
        transmitted_ell=(
            transmitted_charge(
                beam_name
            )
        ),
    )


# ============================================================
# Existing-result loading
# ============================================================

def load_existing_results(
    output_directory: Path,
) -> dict:
    """
    Load a previously saved Chapter 5 scenario.

    Integer seeds are read directly from their CSV string
    representation to preserve the full uint64 value exactly.
    """

    metrics_file = (
        output_directory
        / "metrics.csv"
    )

    spectra_file = (
        output_directory
        / "oam_spectra.npz"
    )

    if (
        not metrics_file.exists()
        or not spectra_file.exists()
    ):
        raise FileNotFoundError(
            "Existing scenario data are incomplete."
        )

    # ========================================================
    # Load scalar realization metrics
    #
    # Use csv.DictReader rather than np.genfromtxt because the
    # uint64 seeds may exceed the exact integer range of
    # float64. Reading them directly from strings preserves
    # every bit of the deterministic seed.
    # ========================================================

    seeds = []
    retention = []
    oam_rms_spread = []
    normalized_entropy = []
    centroid_x = []
    centroid_y = []
    total_power = []

    with metrics_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            seeds.append(
                int(
                    row["seed"]
                )
            )

            retention.append(
                float(
                    row["retention"]
                )
            )

            oam_rms_spread.append(
                float(
                    row["oam_rms_spread"]
                )
            )

            normalized_entropy.append(
                float(
                    row[
                        "normalized_oam_entropy"
                    ]
                )
            )

            centroid_x.append(
                float(
                    row[
                        "centroid_x_m"
                    ]
                )
            )

            centroid_y.append(
                float(
                    row[
                        "centroid_y_m"
                    ]
                )
            )

            total_power.append(
                float(
                    row[
                        "total_power"
                    ]
                )
            )

    # ========================================================
    # Convert to NumPy arrays
    # ========================================================

    seeds = np.asarray(
        seeds,
        dtype=np.uint64,
    )

    retention = np.asarray(
        retention,
        dtype=np.float64,
    )

    oam_rms_spread = np.asarray(
        oam_rms_spread,
        dtype=np.float64,
    )

    normalized_entropy = np.asarray(
        normalized_entropy,
        dtype=np.float64,
    )

    centroid_x = np.asarray(
        centroid_x,
        dtype=np.float64,
    )

    centroid_y = np.asarray(
        centroid_y,
        dtype=np.float64,
    )

    total_power = np.asarray(
        total_power,
        dtype=np.float64,
    )

    # ========================================================
    # Load realization-level OAM spectra
    # ========================================================

    with np.load(
        spectra_file
    ) as archive:

        spectra = np.asarray(
            archive[
                "modal_power"
            ],
            dtype=np.float64,
        )

        saved_ell_values = np.asarray(
            archive[
                "ell_values"
            ],
            dtype=np.int64,
        )

    expected_ell_values = np.arange(
        ELL_MIN,
        ELL_MAX + 1,
        dtype=np.int64,
    )

    if not np.array_equal(
        saved_ell_values,
        expected_ell_values,
    ):
        raise RuntimeError(
            "Saved OAM interval does not match the "
            "current Chapter 5 configuration."
        )

    # ========================================================
    # Consistency checks
    # ========================================================

    number_of_realizations = (
        seeds.size
    )

    scalar_sizes = (
        retention.size,
        oam_rms_spread.size,
        normalized_entropy.size,
        centroid_x.size,
        centroid_y.size,
        total_power.size,
    )

    if any(
        size != number_of_realizations
        for size in scalar_sizes
    ):
        raise RuntimeError(
            "Saved realization metrics have inconsistent sizes."
        )

    if (
        spectra.shape[0]
        != number_of_realizations
    ):
        raise RuntimeError(
            "metrics.csv and oam_spectra.npz have "
            "different ensemble sizes."
        )

    if (
        spectra.shape[1]
        != expected_ell_values.size
    ):
        raise RuntimeError(
            "Saved OAM spectra have an unexpected "
            "number of modal components."
        )

    # ========================================================
    # Return
    # ========================================================

    return {
        "seed":
            seeds,

        "retention":
            retention,

        "oam_rms_spread":
            oam_rms_spread,

        "normalized_entropy":
            normalized_entropy,

        "centroid_x":
            centroid_x,

        "centroid_y":
            centroid_y,

        "total_power":
            total_power,

        "spectra":
            spectra,
    }
# ============================================================
# Concatenate ensembles
# ============================================================

def concatenate_results(
    first: dict,
    second: dict,
) -> dict:

    return {
        key:
            np.concatenate(
                (
                    first[
                        key
                    ],
                    second[
                        key
                    ],
                ),
                axis=0,
            )
        for key in first
    }


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--beam",
        choices=tuple(
            BEAM_CODES
        ),
        required=True,
    )

    parser.add_argument(
        "--psd",
        choices=tuple(
            PSD_CODES
        ),
        required=True,
    )

    parser.add_argument(
        "--regime",
        choices=tuple(
            REGIME_CODES
        ),
        required=True,
    )

    parser.add_argument(
        "--realizations",
        type=int,
        default=(
            DEFAULT_NUMBER_OF_REALIZATIONS
        ),
    )

    parser.add_argument(
        "--extend-to",
        type=int,
        default=None,
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

    arguments = (
        parse_arguments()
    )

    output_directory = (
        scenario_output_directory(
            beam_name=arguments.beam,
            psd_name=arguments.psd,
            regime_name=arguments.regime,
        )
    )

    charge = transmitted_charge(
        arguments.beam
    )

    print()

    print(
        "Chapter 5 production scenario"
    )

    print(
        "============================="
    )

    print(
        f"Beam: {arguments.beam}"
    )

    print(
        f"PSD: {arguments.psd}"
    )

    print(
        f"Regime: {arguments.regime}"
    )

    print(
        f"Transmitted ell: {charge}"
    )

    print(
        f"Workers: {arguments.workers}"
    )

    print(
        f"OAM interval: "
        f"[{ELL_MIN}, {ELL_MAX}]"
    )

    print()

    # ========================================================
    # New scenario
    # ========================================================

    if arguments.extend_to is None:

        target_size = (
            arguments.realizations
        )

        print(
            f"Realizations: {target_size}"
        )

        seeds = (
            generate_realization_seeds(
                beam_name=arguments.beam,
                psd_name=arguments.psd,
                regime_name=arguments.regime,
                number_of_realizations=(
                    target_size
                ),
            )
        )

        results = run_ensemble(
            realization_seeds=seeds,
            beam_name=arguments.beam,
            psd_name=arguments.psd,
            regime_name=arguments.regime,
            number_of_workers=(
                arguments.workers
            ),
        )

    # ========================================================
    # Extend existing scenario
    # ========================================================

    else:

        existing = (
            load_existing_results(
                output_directory
            )
        )

        current_size = (
            existing[
                "retention"
            ].size
        )

        target_size = (
            arguments.extend_to
        )

        if target_size <= current_size:
            raise ValueError(
                f"--extend-to {target_size} must be "
                f"greater than existing ensemble "
                f"size {current_size}."
            )

        print(
            f"Existing realizations: "
            f"{current_size}"
        )

        print(
            f"Extending to: "
            f"{target_size}"
        )

        all_seeds = (
            generate_realization_seeds(
                beam_name=arguments.beam,
                psd_name=arguments.psd,
                regime_name=arguments.regime,
                number_of_realizations=(
                    target_size
                ),
            )
        )

        # Verify deterministic prefix.
        expected_prefix = np.asarray(
            all_seeds[
                :current_size
            ],
            dtype=np.uint64,
        )

        if not np.array_equal(
            expected_prefix,
            existing[
                "seed"
            ],
        ):
            raise RuntimeError(
                "Existing realization seeds do not match "
                "the deterministic scenario sequence."
            )

        new_seeds = (
            all_seeds[
                current_size:
            ]
        )

        new_results = run_ensemble(
            realization_seeds=(
                new_seeds
            ),
            beam_name=arguments.beam,
            psd_name=arguments.psd,
            regime_name=arguments.regime,
            number_of_workers=(
                arguments.workers
            ),
        )

        results = (
            concatenate_results(
                existing,
                new_results,
            )
        )

    # ========================================================
    # Save
    # ========================================================

    save_scenario(
        output_directory=(
            output_directory
        ),
        results=results,
        beam_name=arguments.beam,
        psd_name=arguments.psd,
        regime_name=arguments.regime,
    )

    print()

    print(
        "Results saved in:"
    )

    print(
        output_directory.resolve()
    )


if __name__ == "__main__":
    main()
