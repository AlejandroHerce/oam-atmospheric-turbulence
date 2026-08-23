"""
Spectral-sampling verification of turbulent propagation.

The most spectrally demanding beam considered in the simulations,
BG^3, is propagated through moderate and strong Kolmogorov
turbulence using the final numerical configuration of Chapter 4.

At every observation plane, the spatial spectrum of the propagated
field is analyzed through two metrics:

    eta_edge:
        Fraction of spectral energy contained in the outer
        portion of the FFT domain.

    q99:
        Smallest normalized square spectral radius containing
        99 % of the spectral energy.

The purpose is to verify that turbulence-induced spectral
broadening remains sufficiently far from the Nyquist boundary
throughout the complete propagation path.
"""

import argparse
import csv
import os

from concurrent.futures import ProcessPoolExecutor
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
    SCREEN_SPACING,
    STRONG_R0_SCREEN,
    TOTAL_PROPAGATION_DISTANCE,
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

DEFAULT_NUMBER_OF_REALIZATIONS = 1000

DEFAULT_NUMBER_OF_WORKERS = min(
    8,
    os.cpu_count() or 1,
)

DEFAULT_SEED = 20260822

SPECTRAL_EDGE_THRESHOLD = 0.80

SPECTRAL_ENERGY_FRACTION = 0.99

BG_CHARGE = 3

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/turbulent_spectral_sampling"
)


# ============================================================
# Input beam
# ============================================================

def create_bg3_input():
    """
    Create the BG^3 beam used as the demanding spectral case.
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

    parameters = (
        BG_PARAMETERS[BG_CHARGE]
    )

    field = bessel_gaussian_beam(
        grid=grid,
        w0=parameters["w0"],
        kr=parameters["kr"],
        charge=BG_CHARGE,
    )

    return grid, field


# ============================================================
# Observation distances
# ============================================================

def observation_distances() -> np.ndarray:
    """
    Return observation planes:

        0, dz, 2 dz, ..., L.
    """

    return np.arange(
        NUMBER_OF_PHASE_SCREENS + 1,
        dtype=np.float64,
    ) * SCREEN_SPACING


def phase_screen_distances() -> np.ndarray:
    """
    Return the longitudinal positions of the phase screens.

    Screens are located at the center of each split-step segment.
    """

    return (
        np.arange(
            NUMBER_OF_PHASE_SCREENS,
            dtype=np.float64,
        )
        + 0.5
    ) * SCREEN_SPACING

# ============================================================
# Spectral coordinate
# ============================================================

def normalized_spectral_coordinate() -> np.ndarray:
    """
    Construct the normalized square spectral coordinate

        q = max(|fx|, |fy|) / f_Nyquist.

    Therefore q = 1 corresponds exactly to the Cartesian
    Nyquist boundary of the FFT domain.
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
        1.0 / (2.0 * DX)
    )

    q = (
        np.maximum(
            np.abs(fx),
            np.abs(fy),
        )
        / nyquist_frequency
    )

    return q


def prepare_spectral_analysis():
    """
    Precompute quantities that depend only on the numerical grid.

    These quantities are identical for every realization and
    every observation plane.
    """

    q = normalized_spectral_coordinate()

    q_flat = q.ravel()

    order = np.argsort(
        q_flat
    )

    sorted_q = (
        q_flat[order]
    )

    edge_mask = (
        q
        > SPECTRAL_EDGE_THRESHOLD
    )

    return (
        q,
        order,
        sorted_q,
        edge_mask,
    )

# ============================================================
# Spectrum
# ============================================================

def spectral_power(
    field: np.ndarray,
) -> np.ndarray:
    """
    Calculate the centered spatial spectral power.

    Absolute FFT normalization is unnecessary because all
    spectral metrics are normalized by total spectral energy.
    """

    spectrum = np.fft.fftshift(
        np.fft.fft2(field)
    )

    return (
        np.abs(spectrum) ** 2
    )


# ============================================================
# Spectral metrics
# ============================================================

def spectral_edge_fraction(
    field: np.ndarray,
    q: np.ndarray,
    threshold: float = SPECTRAL_EDGE_THRESHOLD,
) -> float:
    """
    Fraction of spectral energy located in

        q > threshold.
    """

    power = spectral_power(
        field
    )

    total_power = np.sum(
        power
    )

    if total_power <= 0.0:
        raise ValueError(
            "Spectral power must be positive."
        )

    edge_power = np.sum(
        power[q > threshold]
    )

    return float(
        edge_power / total_power
    )


def spectral_energy_quantile(
    field: np.ndarray,
    q: np.ndarray,
    energy_fraction: float = SPECTRAL_ENERGY_FRACTION,
) -> float:
    """
    Calculate q_p, the smallest normalized square spectral
    radius containing the requested fraction of spectral energy.

    For example, energy_fraction = 0.99 gives q99.
    """

    if not (
        0.0
        < energy_fraction
        <= 1.0
    ):
        raise ValueError(
            "energy_fraction must lie in (0, 1]."
        )

    power = spectral_power(
        field
    )

    q_flat = q.ravel()
    power_flat = power.ravel()

    order = np.argsort(
        q_flat
    )

    sorted_q = (
        q_flat[order]
    )

    sorted_power = (
        power_flat[order]
    )

    total_power = np.sum(
        sorted_power
    )

    if total_power <= 0.0:
        raise ValueError(
            "Spectral power must be positive."
        )

    cumulative_power = (
        np.cumsum(sorted_power)
        / total_power
    )

    index = np.searchsorted(
        cumulative_power,
        energy_fraction,
        side="left",
    )

    index = min(
        index,
        len(sorted_q) - 1,
    )

    return float(
        sorted_q[index]
    )


def calculate_spectral_metrics(
    field: np.ndarray,
    order: np.ndarray,
    sorted_q: np.ndarray,
    edge_mask: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate eta_edge and q99 using precomputed spectral-grid
    information.
    """

    spectrum = np.fft.fftshift(
        np.fft.fft2(field)
    )

    power = (
        np.abs(spectrum) ** 2
    )

    total_power = np.sum(
        power
    )

    if total_power <= 0.0:
        raise ValueError(
            "Spectral power must be positive."
        )

    # --------------------------------------------------------
    # Edge-energy fraction
    # --------------------------------------------------------

    eta_edge = float(
        np.sum(
            power[edge_mask]
        )
        / total_power
    )

    # --------------------------------------------------------
    # q99
    # --------------------------------------------------------

    power_flat = (
        power.ravel()
    )

    sorted_power = (
        power_flat[order]
    )

    cumulative_power = (
        np.cumsum(sorted_power)
        / total_power
    )

    index = np.searchsorted(
        cumulative_power,
        SPECTRAL_ENERGY_FRACTION,
        side="left",
    )

    index = min(
        index,
        len(sorted_q) - 1,
    )

    q99 = float(
        sorted_q[index]
    )

    return (
        eta_edge,
        q99,
    )

def prepare_spectral_analysis():
    """
    Precompute quantities that depend only on the numerical grid.
    """

    q = normalized_spectral_coordinate()

    q_flat = q.ravel()

    order = np.argsort(
        q_flat
    )

    sorted_q = (
        q_flat[order]
    )

    edge_mask = (
        q
        > SPECTRAL_EDGE_THRESHOLD
    )

    return (
        q,
        order,
        sorted_q,
        edge_mask,
    )

def simulate_one_realization(
    realization_seed: int,
    r0_screen: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Propagate one atmospheric realization and evaluate the
    spectral-sampling metrics at the observation planes

        z = 0, dz, 2 dz, ..., L.

    The spectral power immediately after a phase screen is
    identical, within numerical precision, to that at the next
    observation plane because vacuum ASM propagation preserves
    the spectral modulus for propagating components.
    """

    rng = np.random.default_rng(
        realization_seed
    )

    _, field = create_bg3_input()

    (
        _,
        order,
        sorted_q,
        edge_mask,
    ) = prepare_spectral_analysis()

    number_of_planes = (
        NUMBER_OF_PHASE_SCREENS
        + 1
    )

    eta_edge = np.zeros(
        number_of_planes,
        dtype=np.float64,
    )

    q99 = np.zeros(
        number_of_planes,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Input plane
    # --------------------------------------------------------

    (
        eta_edge[0],
        q99[0],
    ) = calculate_spectral_metrics(
        field=field,
        order=order,
        sorted_q=sorted_q,
        edge_mask=edge_mask,
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

        field = angular_spectrum_propagation(
            field=field,
            wavelength=WAVELENGTH,
            distance=HALF_SCREEN_SPACING,
            dx=DX,
        )

        (
            eta_edge[
                screen_index + 1
            ],
            q99[
                screen_index + 1
            ],
        ) = calculate_spectral_metrics(
            field=field,
            order=order,
            sorted_q=sorted_q,
            edge_mask=edge_mask,
        )

    return (
        eta_edge,
        q99,
    )

# ============================================================
# Seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
    regime_offset: int,
) -> list[int]:
    """
    Generate independent reproducible realization seeds.
    """

    if number_of_realizations <= 0:
        raise ValueError(
            "number_of_realizations must be positive."
        )

    seed_sequence = np.random.SeedSequence(
        [
            DEFAULT_SEED,
            regime_offset,
        ]
    )

    children = seed_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )
        for child in children
    ]


# ============================================================
# Worker
# ============================================================

def realization_worker(
    arguments: tuple[
        int,
        float,
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    (
        realization_seed,
        r0_screen,
    ) = arguments

    return simulate_one_realization(
        realization_seed=realization_seed,
        r0_screen=r0_screen,
    )


# ============================================================
# Ensemble
# ============================================================

def run_ensemble(
    regime_name: str,
    r0_screen: float,
    number_of_realizations: int,
    number_of_workers: int,
    regime_offset: int,
    realization_seeds: list[int] | None = None,
) -> dict:
    """
    Run the spectral-sampling ensemble for one turbulence regime.
    """

    if realization_seeds is None:

        seeds = generate_realization_seeds(
            number_of_realizations,
            regime_offset,
        )

    else:

        seeds = realization_seeds

        if len(seeds) != number_of_realizations:
            raise ValueError(
                "number_of_realizations must match "
                "the supplied seed sequence."
            )

    number_of_planes = (
        NUMBER_OF_PHASE_SCREENS
        + 1
    )

    eta_samples = np.zeros(
        (
            number_of_realizations,
            number_of_planes,
        ),
        dtype=np.float64,
    )

    q99_samples = np.zeros(
        (
            number_of_realizations,
            number_of_planes,
        ),
        dtype=np.float64,
    )

    worker_arguments = [
        (
            seed,
            r0_screen,
        )
        for seed in seeds
    ]

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        results = executor.map(
            realization_worker,
            worker_arguments,
        )

        for index, (
            eta_edge,
            q99,
        ) in enumerate(results):

            eta_samples[index] = (
                eta_edge
            )

            q99_samples[index] = (
                q99
            )

            completed = index + 1

            if (
                completed == 1
                or completed % 10 == 0
                or completed
                == number_of_realizations
            ):
                print(
                    f"{regime_name}: "
                    f"{completed}/"
                    f"{number_of_realizations}"
                )

    return {
        "regime": regime_name,
        "r0_screen": r0_screen,
        "z": observation_distances(),
        "eta_samples": eta_samples,
        "q99_samples": q99_samples,
    }

# ============================================================
# Summary statistics
# ============================================================

def summarize_result(
    result: dict,
) -> dict:
    """
    Calculate ensemble means and maxima at every propagation
    plane, together with global worst-case values.
    """

    eta_samples = (
        result["eta_samples"]
    )

    q99_samples = (
        result["q99_samples"]
    )

    result["eta_mean"] = np.mean(
        eta_samples,
        axis=0,
    )

    result["eta_max"] = np.max(
        eta_samples,
        axis=0,
    )

    result["q99_mean"] = np.mean(
        q99_samples,
        axis=0,
    )

    result["q99_max"] = np.max(
        q99_samples,
        axis=0,
    )

    result["eta_global_max"] = float(
        np.max(
            eta_samples
        )
    )

    result["q99_global_max"] = float(
        np.max(
            q99_samples
        )
    )

    eta_index = np.unravel_index(
        np.argmax(
            eta_samples
        ),
        eta_samples.shape,
    )

    q99_index = np.unravel_index(
        np.argmax(
            q99_samples
        ),
        q99_samples.shape,
    )

    result["eta_global_realization"] = int(
        eta_index[0]
    )

    result["eta_global_plane"] = int(
        eta_index[1]
    )

    result["q99_global_realization"] = int(
        q99_index[0]
    )

    result["q99_global_plane"] = int(
        q99_index[1]
    )

    return result

def load_raw_metrics(
    regime_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load previously saved eta_edge and q99 samples.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / regime_name
    )

    eta_file = (
        regime_directory
        / "eta_edge_samples.csv"
    )

    q99_file = (
        regime_directory
        / "q99_samples.csv"
    )

    if not eta_file.exists():
        raise FileNotFoundError(
            f"Missing file: {eta_file}"
        )

    if not q99_file.exists():
        raise FileNotFoundError(
            f"Missing file: {q99_file}"
        )

    eta_samples = np.loadtxt(
        eta_file,
        delimiter=",",
        dtype=np.float64,
    )

    q99_samples = np.loadtxt(
        q99_file,
        delimiter=",",
        dtype=np.float64,
    )

    eta_samples = np.atleast_2d(
        eta_samples
    )

    q99_samples = np.atleast_2d(
        q99_samples
    )

    if (
        eta_samples.shape
        != q99_samples.shape
    ):
        raise RuntimeError(
            "Saved eta_edge and q99 arrays "
            "have different shapes."
        )

    expected_planes = (
        NUMBER_OF_PHASE_SCREENS
        + 1
    )

    if (
        eta_samples.shape[1]
        != expected_planes
    ):
        raise RuntimeError(
            "Saved spectral arrays do not match "
            "the current number of observation planes."
        )

    return (
        eta_samples,
        q99_samples,
    )

def extend_ensemble(
    regime_name: str,
    r0_screen: float,
    target_size: int,
    number_of_workers: int,
    regime_offset: int,
) -> dict:
    """
    Extend a previously saved spectral-sampling ensemble
    without recomputing existing realizations.
    """

    (
        old_eta,
        old_q99,
    ) = load_raw_metrics(
        regime_name
    )

    current_size = (
        old_eta.shape[0]
    )

    if target_size <= current_size:
        raise ValueError(
            f"target_size={target_size} must exceed "
            f"the current ensemble size {current_size}."
        )

    additional_number = (
        target_size
        - current_size
    )

    print()
    print(
        f"Extensión del régimen: {regime_name}"
    )

    print(
        f"Ensamble existente: {current_size}"
    )

    print(
        f"Nuevas realizaciones: {additional_number}"
    )

    print(
        f"Ensamble final: {target_size}"
    )

    # Generate exactly the seed sequence that a complete
    # target_size run would have used.
    complete_seed_sequence = (
        generate_realization_seeds(
            target_size,
            regime_offset,
        )
    )

    new_seeds = (
        complete_seed_sequence[
            current_size:
            target_size
        ]
    )

    new_result = run_ensemble(
        regime_name=regime_name,
        r0_screen=r0_screen,
        number_of_realizations=(
            additional_number
        ),
        number_of_workers=(
            number_of_workers
        ),
        regime_offset=regime_offset,
        realization_seeds=new_seeds,
    )

    eta_samples = np.concatenate(
        (
            old_eta,
            new_result["eta_samples"],
        ),
        axis=0,
    )

    q99_samples = np.concatenate(
        (
            old_q99,
            new_result["q99_samples"],
        ),
        axis=0,
    )

    return {
        "regime": regime_name,
        "r0_screen": r0_screen,
        "z": observation_distances(),
        "eta_samples": eta_samples,
        "q99_samples": q99_samples,
    }

# ============================================================
# Save raw metrics
# ============================================================

def save_raw_metrics(
    result: dict,
) -> None:
    """
    Save eta_edge and q99 for every realization and plane.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    regime_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        regime_directory
        / "eta_edge_samples.csv",
        result["eta_samples"],
        delimiter=",",
    )

    np.savetxt(
        regime_directory
        / "q99_samples.csv",
        result["q99_samples"],
        delimiter=",",
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    result: dict,
) -> None:
    """
    Save plane-by-plane spectral summary.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    regime_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        regime_directory
        / "spectral_sampling_summary.csv"
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
                "z_m",
                "eta_edge_mean",
                "eta_edge_max",
                "q99_mean",
                "q99_max",
            ]
        )

        for index, z in enumerate(
            result["z"]
        ):

            writer.writerow(
                [
                    z,
                    result[
                        "eta_mean"
                    ][index],
                    result[
                        "eta_max"
                    ][index],
                    result[
                        "q99_mean"
                    ][index],
                    result[
                        "q99_max"
                    ][index],
                ]
            )


# ============================================================
# Print summary
# ============================================================

def print_summary(
    result: dict,
) -> None:
    """
    Print spectral-sampling results.
    """

    print()

    print(
        "Verificación espectral: "
        f"{result['regime']}"
    )

    print(
        "=" * 78
    )

    print(
        f"{'z [m]':>9} "
        f"{'<eta_edge>':>15} "
        f"{'max eta_edge':>15} "
        f"{'<q99>':>12} "
        f"{'max q99':>12}"
    )

    print(
        "-" * 78
    )

    for index, z in enumerate(
        result["z"]
    ):

        print(
            f"{z:9.2f} "
            f"{result['eta_mean'][index]:15.6e} "
            f"{result['eta_max'][index]:15.6e} "
            f"{result['q99_mean'][index]:12.6f} "
            f"{result['q99_max'][index]:12.6f}"
        )

    print()

    print(
        "Peor caso de todo el ensamble"
    )

    print(
        "-----------------------------"
    )

    eta_plane = (
        result[
            "eta_global_plane"
        ]
    )

    q99_plane = (
        result[
            "q99_global_plane"
        ]
    )

    print(
        "eta_edge máximo = "
        f"{result['eta_global_max']:.6e}"
    )

    print(
        "  realización = "
        f"{result['eta_global_realization'] + 1}"
    )

    print(
        "  z = "
        f"{result['z'][eta_plane]:.2f} m"
    )

    print(
        "q99 máximo = "
        f"{result['q99_global_max']:.6f}"
    )

    print(
        "  realización = "
        f"{result['q99_global_realization'] + 1}"
    )

    print(
        "  z = "
        f"{result['z'][q99_plane]:.2f} m"
    )

# ============================================================
# Plot q99
# ============================================================

def plot_q99(
    result: dict,
) -> None:
    """
    Plot mean and maximum q99 along propagation.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        result["z"],
        result["q99_mean"],
        marker="o",
        label="Promedio del ensamble",
    )

    axis.plot(
        result["z"],
        result["q99_max"],
        marker="s",
        label="Máximo del ensamble",
    )

    axis.axhline(
        SPECTRAL_EDGE_THRESHOLD,
        linestyle="--",
        label=(
            r"$q=0.8$"
        ),
    )

    axis.axhline(
        1.0,
        linestyle=":",
        label="Límite de Nyquist",
    )

    axis.set_xlabel(
        r"$z$ [m]"
    )

    axis.set_ylabel(
        r"$q_{99}$"
    )

    axis.set_title(
        "Contenido espectral durante la propagación "
        f"({result['regime']})"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        regime_directory
        / "q99_vs_distance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Plot edge-energy fraction
# ============================================================

def plot_eta_edge(
    result: dict,
) -> None:
    """
    Plot mean and maximum spectral edge-energy fraction.
    """

    regime_directory = (
        OUTPUT_DIRECTORY
        / result["regime"]
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.semilogy(
        result["z"],
        result["eta_mean"],
        marker="o",
        label="Promedio del ensamble",
    )

    axis.semilogy(
        result["z"],
        result["eta_max"],
        marker="s",
        label="Máximo del ensamble",
    )

    axis.set_xlabel(
        r"$z$ [m]"
    )

    axis.set_ylabel(
        r"$\eta_{\mathrm{edge}}$"
    )

    axis.set_title(
        "Energía próxima al límite espectral "
        f"({result['regime']})"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        regime_directory
        / "eta_edge_vs_distance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
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

    parser.add_argument(
        "--regime",
        choices=(
            "moderate",
            "strong",
            "both",
        ),
        default="both",
    )

    parser.add_argument(
        "--extend-to",
        type=int,
        default=None,
        help=(
            "Extend an existing saved ensemble to the requested "
            "total number of realizations."
        ),
    )

    return parser.parse_args()


def calculate_cumulative_worst_case(
    samples: np.ndarray,
) -> np.ndarray:
    """
    Calculate the accumulated worst-case value as the number
    of atmospheric realizations increases.

    For every realization n, the returned value is the maximum
    over all realizations 1,...,n and all propagation planes.
    """

    per_realization_maximum = np.max(
        samples,
        axis=1,
    )

    return np.maximum.accumulate(
        per_realization_maximum
    )

def print_cumulative_worst_case(
    result: dict,
) -> None:
    """
    Print the evolution of the worst observed spectral metrics
    with ensemble size.
    """

    eta_cumulative = (
        calculate_cumulative_worst_case(
            result["eta_samples"]
        )
    )

    q99_cumulative = (
        calculate_cumulative_worst_case(
            result["q99_samples"]
        )
    )

    checkpoints = (
        50,
        100,
        200,
        300,
        400,
        500,
    )

    print()
    print(
        "Evolución del peor caso con el tamaño del ensamble"
    )

    print(
        "================================================"
    )

    print(
        f"{'Nens':>7}"
        f"{'max eta_edge':>18}"
        f"{'max q99':>14}"
    )

    print(
        "-" * 39
    )

    for checkpoint in checkpoints:

        if checkpoint <= len(
            eta_cumulative
        ):

            index = (
                checkpoint - 1
            )

            print(
                f"{checkpoint:7d}"
                f"{eta_cumulative[index]:18.6e}"
                f"{q99_cumulative[index]:14.6f}"
            )

# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Execute the spectral-sampling verification.
    """

    arguments = (
        parse_arguments()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = []

    if arguments.regime in (
        "moderate",
        "both",
    ):
        configurations.append(
            (
                "moderada",
                MODERATE_R0_SCREEN,
                1,
            )
        )

    if arguments.regime in (
        "strong",
        "both",
    ):
        configurations.append(
            (
                "fuerte",
                STRONG_R0_SCREEN,
                2,
            )
        )

    print(
        "Verificación espectral de Nyquist"
    )

    print(
        "================================="
    )

    print(
        "Haz: BG^3"
    )

    print(
        f"N = {N_GRID}"
    )

    print(
        f"dx = {DX:.6e} m"
    )

    print(
        "f_Nyquist = "
        f"{1.0 / (2.0 * DX):.6e} 1/m"
    )

    print(
        f"Pantallas = "
        f"{NUMBER_OF_PHASE_SCREENS}"
    )

    print(
        f"Distancia total = "
        f"{TOTAL_PROPAGATION_DISTANCE:.2f} m"
    )

    print(
        f"Realizaciones = "
        f"{arguments.realizations}"
    )

    print(
        f"Workers = "
        f"{arguments.workers}"
    )

    print(
        f"Umbral espectral = "
        f"{SPECTRAL_EDGE_THRESHOLD:.2f} "
        "f_Nyquist"
    )

    print(
        f"Fracción q = "
        f"{SPECTRAL_ENERGY_FRACTION:.2f}"
    )

    for (
        regime_name,
        r0_screen,
        regime_offset,
    ) in configurations:

        print()

        print(
            f"Régimen: {regime_name}"
        )

        print(
            "="
            * (
                9
                + len(regime_name)
            )
        )

        if arguments.extend_to is not None:
        
            result = extend_ensemble(
                regime_name=regime_name,
                r0_screen=r0_screen,
                target_size=arguments.extend_to,
                number_of_workers=(
                    arguments.workers
                ),
                regime_offset=regime_offset,
            )
        
        else:
        
            result = run_ensemble(
                regime_name=regime_name,
                r0_screen=r0_screen,
                number_of_realizations=(
                    arguments.realizations
                ),
                number_of_workers=(
                    arguments.workers
                ),
                regime_offset=regime_offset,
            )
            
        result = summarize_result(
            result
        )

        save_raw_metrics(
            result
        )

        save_summary(
            result
        )

        print_summary(
            result
        )

        print_cumulative_worst_case(
            result
        )

        plot_q99(
            result
        )

        plot_eta_edge(
            result
        )


if __name__ == "__main__":
    main()
