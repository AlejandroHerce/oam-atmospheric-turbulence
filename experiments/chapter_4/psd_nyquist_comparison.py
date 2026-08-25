"""
Comparative Nyquist test for atmospheric turbulence PSD models.

The purpose of this experiment is to verify whether the strong
Kolmogorov case used in the detailed Nyquist validation is also
spectrally conservative with respect to the von Karman and
modified von Karman models.

The same physical turbulence strength, propagation geometry,
transverse grid, number of phase screens and input beam are used
for all PSDs.

For every realization, the final field spectrum is characterized
using the same Cartesian Nyquist metrics employed in the main
validation:

    q(fx, fy)
        = max(|fx|, |fy|) / f_Nyq

    eta_edge
        = sum_{q > 0.8} S / sum S

and q_99, defined as the smallest q containing 99 % of the
spectral energy.

Random streams are unique to

    (PSD, realization, phase screen),

while remaining exactly reproducible.
"""

import argparse
import csv
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from configs.chapter_2 import (
    W0_GAUSSIAN,
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
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    STRONG_R0_TOTAL,
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
    modified_von_karman_phase_screen,
    modified_von_karman_subharmonics,
    von_karman_phase_screen,
    von_karman_subharmonics,
)

from src.propagation import (
    angular_spectrum_propagation,
)


# ============================================================
# Experiment configuration
# ============================================================

DEFAULT_NUMBER_OF_REALIZATIONS = 100

DEFAULT_NUMBER_OF_WORKERS = min(
    12,
    os.cpu_count() or 1,
)

MASTER_SEED = 20260824

BOOTSTRAP_SAMPLES = 5000

BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

EDGE_THRESHOLD = 0.80

CONTAINMENT_FRACTION = 0.99

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/"
    "psd_nyquist_comparison"
)


# ============================================================
# PSD identifiers
# ============================================================

PSD_MODELS = (
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
)

PSD_CODES = {
    "kolmogorov": 1,
    "von_karman": 2,
    "modified_von_karman": 3,
}


def psd_label(
    psd_name: str,
) -> str:
    """
    Human-readable PSD label.
    """

    labels = {
        "kolmogorov":
            "Kolmogorov",

        "von_karman":
            "von Karman",

        "modified_von_karman":
            "von Karman modificada",
    }

    return labels[
        psd_name
    ]


# ============================================================
# Fried parameter per screen
# ============================================================

def segment_fried_parameter(
    total_r0: float,
    number_of_screens: int,
) -> float:
    """
    Fried parameter associated with one equal longitudinal
    segment.

        r0_screen = r0_total * Ns^(3/5)
    """

    if total_r0 <= 0.0:
        raise ValueError(
            "total_r0 must be positive."
        )

    if number_of_screens <= 0:
        raise ValueError(
            "number_of_screens must be positive."
        )

    return float(
        total_r0
        * number_of_screens
        ** (3.0 / 5.0)
    )


STRONG_R0_SCREEN = (
    segment_fried_parameter(
        total_r0=STRONG_R0_TOTAL,
        number_of_screens=(
            NUMBER_OF_PHASE_SCREENS
        ),
    )
)


# ============================================================
# Input beam
# ============================================================

def create_input_beam():
    """
    Create the Gaussian input field.
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
# Deterministic random hierarchy
# ============================================================

def generate_realization_seeds(
    psd_name: str,
    number_of_realizations: int,
) -> list[int]:
    """
    Generate independent deterministic realization streams for
    one PSD model.

    The root sequence depends on

        (MASTER_SEED, PSD code).

    Therefore different PSD models do not share random streams.
    """

    if psd_name not in PSD_CODES:
        raise ValueError(
            f"Unknown PSD model: {psd_name}"
        )

    root_sequence = np.random.SeedSequence(
        [
            int(MASTER_SEED),
            int(
                PSD_CODES[
                    psd_name
                ]
            ),
        ]
    )

    child_sequences = root_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for sequence in child_sequences
    ]


def generate_screen_seeds(
    realization_seed: int,
) -> list[int]:
    """
    Generate one independent stream for every phase screen
    within one atmospheric realization.
    """

    realization_sequence = np.random.SeedSequence(
        int(
            realization_seed
        )
    )

    screen_sequences = (
        realization_sequence.spawn(
            NUMBER_OF_PHASE_SCREENS
        )
    )

    return [
        int(
            sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for sequence in screen_sequences
    ]


# ============================================================
# Full phase screens
# ============================================================

def generate_phase_screen(
    psd_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one complete phase screen for the selected PSD.

    The von Karman and modified von Karman implementations use
    the same convention as the existing Kolmogorov generator:

        phase = FFT component + subharmonic component.

    Both components are generated sequentially from the same RNG.
    """

    # --------------------------------------------------------
    # Kolmogorov
    # --------------------------------------------------------

    if psd_name == "kolmogorov":

        return (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=STRONG_R0_SCREEN,
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
                r0=STRONG_R0_SCREEN,
                outer_scale=OUTER_SCALE,
                rng=rng,
                remove_piston=False,
            )
        )

        phase_subharmonics = (
            von_karman_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=STRONG_R0_SCREEN,
                outer_scale=OUTER_SCALE,
                n_subharmonics=(
                    VON_KARMAN_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=False,
            )
        )

        phase = (
            phase_fft
            + phase_subharmonics
        )

        phase -= np.mean(
            phase
        )

        return phase

    # --------------------------------------------------------
    # Modified von Karman
    # --------------------------------------------------------

    if psd_name == "modified_von_karman":

        phase_fft = (
            modified_von_karman_phase_screen(
                n=N_GRID,
                delta=DX,
                r0=STRONG_R0_SCREEN,
                outer_scale=OUTER_SCALE,
                inner_scale=INNER_SCALE,
                rng=rng,
                remove_piston=False,
            )
        )

        phase_subharmonics = (
            modified_von_karman_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=STRONG_R0_SCREEN,
                outer_scale=OUTER_SCALE,
                inner_scale=INNER_SCALE,
                n_subharmonics=(
                    MODIFIED_VON_KARMAN_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=False,
            )
        )

        phase = (
            phase_fft
            + phase_subharmonics
        )

        phase -= np.mean(
            phase
        )

        return phase

    raise ValueError(
        f"Unknown PSD model: {psd_name}"
    )


# ============================================================
# Nyquist coordinate
# ============================================================

def create_normalized_frequency_coordinate() -> np.ndarray:
    """
    Construct

        q(fx, fy)
        =
        max(|fx|, |fy|) / f_Nyq.

    Thus q = 1 corresponds exactly to one of the Cartesian
    Nyquist boundaries.
    """

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N_GRID,
            d=DX,
        )
    )

    fx, fy = np.meshgrid(
        frequencies,
        frequencies,
        indexing="xy",
    )

    nyquist_frequency = (
        1.0
        / (
            2.0
            * DX
        )
    )

    q = (
        np.maximum(
            np.abs(
                fx
            ),
            np.abs(
                fy
            ),
        )
        / nyquist_frequency
    )

    return q


Q_COORDINATE = (
    create_normalized_frequency_coordinate()
)


# ============================================================
# Spectral metrics
# ============================================================

def calculate_nyquist_metrics(
    field: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """
    Calculate the two spectral metrics used in the main Nyquist
    validation.

    Returns
    -------
    eta_edge:
        Fraction of spectral energy with q > 0.8.

    q99:
        Smallest q containing 99 % of the spectral energy.
    """

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            field
        )
    )

    spectral_power = (
        np.abs(
            spectrum
        ) ** 2
    )

    total_power = float(
        np.sum(
            spectral_power
        )
    )

    if (
        not np.isfinite(
            total_power
        )
        or total_power <= 0.0
    ):
        raise ValueError(
            "Invalid spectral power."
        )

    normalized_power = (
        spectral_power
        / total_power
    )

    # --------------------------------------------------------
    # Edge energy
    # --------------------------------------------------------

    edge_mask = (
        Q_COORDINATE
        > EDGE_THRESHOLD
    )

    eta_edge = float(
        np.sum(
            normalized_power[
                edge_mask
            ]
        )
    )

    # --------------------------------------------------------
    # q_99
    # --------------------------------------------------------

    q_flat = (
        Q_COORDINATE.ravel()
    )

    power_flat = (
        normalized_power.ravel()
    )

    order = np.argsort(
        q_flat
    )

    q_sorted = (
        q_flat[
            order
        ]
    )

    power_sorted = (
        power_flat[
            order
        ]
    )

    cumulative_power = np.cumsum(
        power_sorted
    )

    containment_index = int(
        np.searchsorted(
            cumulative_power,
            CONTAINMENT_FRACTION,
            side="left",
        )
    )

    containment_index = min(
        containment_index,
        q_sorted.size - 1,
    )

    q99 = float(
        q_sorted[
            containment_index
        ]
    )

    return (
        eta_edge,
        q99,
    )


# ============================================================
# One realization
# ============================================================

def simulate_one_realization(
    realization_seed: int,
    psd_name: str,
) -> tuple[
    float,
    float,
]:
    """
    Propagate one atmospheric realization and calculate the
    final Nyquist metrics.
    """

    _, field = (
        create_input_beam()
    )

    screen_spacing = (
        TOTAL_PROPAGATION_DISTANCE
        / NUMBER_OF_PHASE_SCREENS
    )

    half_screen_spacing = (
        screen_spacing
        / 2.0
    )

    screen_seeds = (
        generate_screen_seeds(
            realization_seed
        )
    )

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        # Half step to phase screen.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=half_screen_spacing,
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
                rng=screen_rng,
            )
        )

        field *= np.exp(
            1j
            * phase_screen
        )

        # Half step to observation plane.
        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=half_screen_spacing,
            dx=DX,
        )

    return calculate_nyquist_metrics(
        field
    )


# ============================================================
# Run one PSD
# ============================================================

def run_psd(
    psd_name: str,
    number_of_realizations: int,
    number_of_workers: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Run the Nyquist mini-ensemble for one PSD.
    """

    realization_seeds = (
        generate_realization_seeds(
            psd_name=psd_name,
            number_of_realizations=(
                number_of_realizations
            ),
        )
    )

    worker = partial(
        simulate_one_realization,
        psd_name=psd_name,
    )

    eta_edge_samples = np.zeros(
        number_of_realizations,
        dtype=np.float64,
    )

    q99_samples = np.zeros(
        number_of_realizations,
        dtype=np.float64,
    )

    print()
    print(
        f"PSD: {psd_label(psd_name)}"
    )

    print(
        "-" * 50
    )

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            worker,
            realization_seeds,
            chunksize=1,
        )

        for index, (
            eta_edge,
            q99,
        ) in enumerate(
            results
        ):

            eta_edge_samples[
                index
            ] = eta_edge

            q99_samples[
                index
            ] = q99

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
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return (
        eta_edge_samples,
        q99_samples,
    )


# ============================================================
# Bootstrap mean confidence interval
# ============================================================

def bootstrap_mean_ci(
    samples: np.ndarray,
    seed: int,
) -> tuple[
    float,
    float,
]:
    """
    Percentile-bootstrap confidence interval for the ensemble
    mean.
    """

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    number_of_realizations = (
        samples.size
    )

    bootstrap_means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            low=0,
            high=number_of_realizations,
            size=number_of_realizations,
        )

        bootstrap_means[
            bootstrap_index
        ] = np.mean(
            samples[
                indices
            ]
        )

    alpha = (
        1.0
        - BOOTSTRAP_CONFIDENCE_LEVEL
    )

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


# ============================================================
# Analyze one PSD
# ============================================================

def analyze_psd(
    psd_name: str,
    eta_edge_samples: np.ndarray,
    q99_samples: np.ndarray,
) -> dict:
    """
    Calculate ensemble statistics.
    """

    psd_code = (
        PSD_CODES[
            psd_name
        ]
    )

    mean_eta_edge = float(
        np.mean(
            eta_edge_samples
        )
    )

    mean_q99 = float(
        np.mean(
            q99_samples
        )
    )

    (
        eta_lower,
        eta_upper,
    ) = bootstrap_mean_ci(
        samples=eta_edge_samples,
        seed=(
            MASTER_SEED
            + 10_000
            + psd_code
        ),
    )

    (
        q99_lower,
        q99_upper,
    ) = bootstrap_mean_ci(
        samples=q99_samples,
        seed=(
            MASTER_SEED
            + 20_000
            + psd_code
        ),
    )

    return {
        "psd":
            psd_name,

        "label":
            psd_label(
                psd_name
            ),

        "eta_edge":
            mean_eta_edge,

        "eta_edge_lower":
            eta_lower,

        "eta_edge_upper":
            eta_upper,

        "q99":
            mean_q99,

        "q99_lower":
            q99_lower,

        "q99_upper":
            q99_upper,

        "eta_edge_samples":
            eta_edge_samples,

        "q99_samples":
            q99_samples,
    }


# ============================================================
# Save raw samples
# ============================================================

def save_raw_samples(
    result: dict,
) -> None:
    """
    Save realization-level spectral metrics.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.column_stack(
        (
            result[
                "eta_edge_samples"
            ],
            result[
                "q99_samples"
            ],
        )
    )

    np.savetxt(
        OUTPUT_DIRECTORY
        / (
            result["psd"]
            + "_nyquist_samples.csv"
        ),
        data,
        delimiter=",",
        header=(
            "eta_edge,q99"
        ),
        comments="",
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    results: list[dict],
) -> None:
    """
    Save PSD comparison table.
    """

    filename = (
        OUTPUT_DIRECTORY
        / "psd_nyquist_comparison.csv"
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
                "psd",
                "mean_eta_edge",
                "eta_edge_ci95_lower",
                "eta_edge_ci95_upper",
                "mean_q99",
                "q99_ci95_lower",
                "q99_ci95_upper",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result["label"],
                    result["eta_edge"],
                    result[
                        "eta_edge_lower"
                    ],
                    result[
                        "eta_edge_upper"
                    ],
                    result["q99"],
                    result[
                        "q99_lower"
                    ],
                    result[
                        "q99_upper"
                    ],
                ]
            )


# ============================================================
# Terminal output
# ============================================================

def print_summary(
    results: list[dict],
) -> None:
    """
    Print final PSD comparison.
    """

    print()
    print(
        "Comparación Nyquist entre PSD "
        "-- turbulencia fuerte"
    )

    print(
        "=" * 112
    )

    header = (
        f"{'PSD':>24}"
        f"{'<eta_edge>':>16}"
        f"{'IC95 inf':>14}"
        f"{'IC95 sup':>14}"
        f"{'<q99>':>14}"
        f"{'IC95 inf':>14}"
        f"{'IC95 sup':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for result in results:

        print(
            f"{result['label']:>24}"
            f"{result['eta_edge']:16.6e}"
            f"{result['eta_edge_lower']:14.6e}"
            f"{result['eta_edge_upper']:14.6e}"
            f"{result['q99']:14.6f}"
            f"{result['q99_lower']:14.6f}"
            f"{result['q99_upper']:14.6f}"
        )


# ============================================================
# Plot
# ============================================================

def plot_comparison(
    results: list[dict],
) -> None:
    """
    Plot mean q99 for the three PSD models.

    The plot is only a compact visual comparison; both metrics
    remain available in the table and CSV output.
    """

    labels = [
        result["label"]
        for result in results
    ]

    q99 = np.asarray(
        [
            result["q99"]
            for result in results
        ]
    )

    lower = np.asarray(
        [
            result["q99_lower"]
            for result in results
        ]
    )

    upper = np.asarray(
        [
            result["q99_upper"]
            for result in results
        ]
    )

    x = np.arange(
        len(
            results
        )
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.errorbar(
        x,
        q99,
        yerr=np.vstack(
            (
                q99 - lower,
                upper - q99,
            )
        ),
        marker="o",
        linestyle="none",
        capsize=5,
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.4,
        label="Límite de Nyquist",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        labels
    )

    axis.set_ylabel(
        r"$q_{99}$"
    )

    axis.set_title(
        "Comparación espectral entre PSD "
        "en turbulencia fuerte"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "psd_nyquist_q99_comparison.png",
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
    """
    Parse command-line options.
    """

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
    """
    Run the PSD Nyquist mini-test.
    """

    arguments = (
        parse_arguments()
    )

    if arguments.realizations <= 0:
        raise ValueError(
            "realizations must be positive."
        )

    if arguments.workers <= 0:
        raise ValueError(
            "workers must be positive."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Mini-prueba Nyquist entre PSD"
    )

    print(
        "=============================="
    )

    print(
        "Régimen: turbulencia fuerte"
    )

    print(
        f"r0 total = "
        f"{STRONG_R0_TOTAL:.6e} m"
    )

    print(
        f"r0 por pantalla = "
        f"{STRONG_R0_SCREEN:.6e} m"
    )

    print(
        f"Pantallas = "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Realizaciones por PSD = "
        f"{arguments.realizations}"
    )

    print(
        f"Workers = "
        f"{arguments.workers}"
    )

    print(
        f"Umbral de borde: "
        f"q > {EDGE_THRESHOLD:.2f}"
    )

    print(
        f"Fracción de contención: "
        f"{CONTAINMENT_FRACTION:.2f}"
    )

    results = []

    for psd_name in PSD_MODELS:

        (
            eta_edge_samples,
            q99_samples,
        ) = run_psd(
            psd_name=psd_name,
            number_of_realizations=(
                arguments.realizations
            ),
            number_of_workers=(
                arguments.workers
            ),
        )

        result = analyze_psd(
            psd_name=psd_name,
            eta_edge_samples=(
                eta_edge_samples
            ),
            q99_samples=(
                q99_samples
            ),
        )

        results.append(
            result
        )

        save_raw_samples(
            result
        )

    print_summary(
        results
    )

    save_summary(
        results
    )

    plot_comparison(
        results
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
