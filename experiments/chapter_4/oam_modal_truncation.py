"""
OAM modal-range truncation test.

The purpose of this experiment is to determine a sufficiently
wide OAM interval for the statistical analysis performed in
Chapter 4 and for the final simulations.

A LG_0^3 beam is propagated through strong Kolmogorov turbulence,
matching the demanding configuration used in the ensemble-size
convergence experiment.

For every atmospheric realization, the OAM spectrum is first
calculated over the complete interval supported by the azimuthal
FFT. Successively smaller symmetric intervals [-ell_max, ell_max]
are then extracted from this reference spectrum.

For every candidate ell_max, the experiment evaluates:

    - spectral power omitted outside the interval;
    - error in transmitted-mode retention;
    - error in RMS OAM spread.

This separates modal-resolution convergence from statistical
ensemble convergence.
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
    W0_LG,
)

from src.beams import (
    bessel_gaussian_beam,
    laguerre_gaussian_beam,
)


from configs.chapter_4 import (
    DX,
    ENSEMBLE_BEAM_CHARGE,
    ENSEMBLE_R0_SCREEN,
    ENSEMBLE_SEED,
    ENSEMBLE_SUBHARMONIC_LEVEL,
    HALF_SCREEN_SPACING,
    L_WINDOW,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    SCREEN_SPACING,
    TOTAL_PROPAGATION_DISTANCE,
    WAVELENGTH,
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
# Experiment configuration
# ============================================================

DEFAULT_NUMBER_OF_REALIZATIONS = 20

DEFAULT_NUMBER_OF_WORKERS = min(
    8,
    os.cpu_count() or 1,
)

# Use the same master seed as the ensemble-convergence
# experiment so that the atmospheric realizations are
# reproducible and directly comparable.
DEFAULT_SEED = ENSEMBLE_SEED

RADIAL_SAMPLES = 256

AZIMUTHAL_SAMPLES = 720

MAXIMUM_RADIUS = (
    0.98
    * L_WINDOW
    / 2.0
)

FULL_ELL_MIN = (
    -AZIMUTHAL_SAMPLES // 2
)

FULL_ELL_MAX = (
    AZIMUTHAL_SAMPLES // 2
    - 1
)

ELL_MAX_CANDIDATES = (
    8,
    12,
    16,
    20,
    24,
    30,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
)

OUTSIDE_POWER_THRESHOLD = 1.0e-3

OUTPUT_DIRECTORY = Path(
    "results/chapter_4/oam_modal_truncation"
)


# ============================================================
# Input beam
# ============================================================

def create_input_beam(
    beam_family: str,
):
    """
    Create the demanding charge-3 beam for the selected family.
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

    family = beam_family.lower()

    if family == "lg":

        field = laguerre_gaussian_beam(
            grid=grid,
            w0=W0_LG,
            charge=3,
        )

    elif family == "bg":

        parameters = BG_PARAMETERS[3]

        field = bessel_gaussian_beam(
            grid=grid,
            w0=parameters["w0"],
            kr=parameters["kr"],
            charge=3,
        )

    else:

        raise ValueError(
            "beam_family must be 'lg' or 'bg'."
        )

    return (
        grid,
        field,
    )


# ============================================================
# Atmospheric propagation
# ============================================================

def propagate_one_realization(
    realization_seed: int,
    beam_family: str,
):
    """
    Propagate one atmospheric realization and return the final
    optical field.
    """

    rng = np.random.default_rng(
        realization_seed
    )

    grid, field = create_input_beam(
        beam_family=beam_family,
    )

    # --------------------------------------------------------
    # Propagation to the first screen
    # --------------------------------------------------------

    field = angular_spectrum_propagation(
        field=field,
        wavelength=WAVELENGTH,
        distance=HALF_SCREEN_SPACING,
        dx=DX,
    )

    # --------------------------------------------------------
    # Turbulent path
    # --------------------------------------------------------

    for screen_index in range(
        NUMBER_OF_PHASE_SCREENS
    ):

        phase_screen = (
            kolmogorov_phase_screen_with_subharmonics(
                n=N_GRID,
                delta=DX,
                r0=ENSEMBLE_R0_SCREEN,
                n_subharmonics=(
                    ENSEMBLE_SUBHARMONIC_LEVEL
                ),
                rng=rng,
                remove_piston=True,
            )
        )

        field *= np.exp(
            1j * phase_screen
        )

        if (
            screen_index
            < NUMBER_OF_PHASE_SCREENS - 1
        ):
            field = angular_spectrum_propagation(
                field=field,
                wavelength=WAVELENGTH,
                distance=SCREEN_SPACING,
                dx=DX,
            )

    # --------------------------------------------------------
    # Final half-step
    # --------------------------------------------------------

    field = angular_spectrum_propagation(
        field=field,
        wavelength=WAVELENGTH,
        distance=HALF_SCREEN_SPACING,
        dx=DX,
    )

    return (
        grid,
        field,
    )


# ============================================================
# Full OAM spectrum
# ============================================================

def calculate_full_spectrum(
    grid,
    field,
):
    """
    Calculate the OAM spectrum over the complete interval
    representable by the azimuthal FFT.
    """

    return calculate_oam_spectrum(
        field=field,
        grid=grid,
        ell_min=FULL_ELL_MIN,
        ell_max=FULL_ELL_MAX,
        radial_samples=RADIAL_SAMPLES,
        azimuthal_samples=AZIMUTHAL_SAMPLES,
        maximum_radius=MAXIMUM_RADIUS,
    )


# ============================================================
# Truncated-spectrum metrics
# ============================================================

def analyze_modal_window(
    ell_values: np.ndarray,
    full_power: np.ndarray,
    ell_max: int,
    transmitted_charge: int,
) -> dict:
    """
    Analyze one symmetric modal interval [-ell_max, ell_max].

    The reference spectrum is normalized over the complete FFT
    interval. The selected modal window is then renormalized,
    reproducing the behavior of calculate_oam_spectrum when a
    restricted ell interval is requested.
    """

    mask = (
        np.abs(ell_values)
        <= ell_max
    )

    included_power = float(
        np.sum(
            full_power[mask]
        )
    )

    outside_power = (
        1.0
        - included_power
    )

    if included_power <= 0.0:
        raise RuntimeError(
            "Selected OAM interval contains no power."
        )

    truncated_ell = (
        ell_values[mask]
    )

    truncated_power = (
        full_power[mask]
        / included_power
    )

    retention = modal_power_at_charge(
        ell_values=truncated_ell,
        modal_power=truncated_power,
        charge=transmitted_charge,
    )

    spread = calculate_rms_oam_spread(
        ell_values=truncated_ell,
        modal_power=truncated_power,
        transmitted_charge=transmitted_charge,
    )

    return {
        "outside_power": outside_power,
        "retention": retention,
        "spread": spread,
    }


# ============================================================
# One realization
# ============================================================

def simulate_and_analyze(
    realization_seed: int,
    beam_family: str,
) -> dict:
    """
    Propagate one realization and evaluate all modal windows.
    """

    (
        grid,
        field,
    ) = propagate_one_realization(
        realization_seed=realization_seed,
        beam_family=beam_family,
    )

    (
        ell_values,
        full_power,
    ) = calculate_full_spectrum(
        grid=grid,
        field=field,
    )

    # --------------------------------------------------------
    # Full-spectrum reference metrics
    # --------------------------------------------------------

    full_retention = modal_power_at_charge(
        ell_values=ell_values,
        modal_power=full_power,
        charge=ENSEMBLE_BEAM_CHARGE,
    )

    full_spread = calculate_rms_oam_spread(
        ell_values=ell_values,
        modal_power=full_power,
        transmitted_charge=(
            ENSEMBLE_BEAM_CHARGE
        ),
    )

    candidate_results = {}

    for ell_max in ELL_MAX_CANDIDATES:

        metrics = analyze_modal_window(
            ell_values=ell_values,
            full_power=full_power,
            ell_max=ell_max,
            transmitted_charge=(
                ENSEMBLE_BEAM_CHARGE
            ),
        )

        retention_error_absolute = abs(
            metrics["retention"]
            - full_retention
        )

        if full_retention > 0.0:
            retention_error_percent = (
                100.0
                * retention_error_absolute
                / full_retention
            )
        else:
            retention_error_percent = np.nan


        candidate_results[
            ell_max
        ] = {
            **metrics,
            "retention_error_absolute": (
                retention_error_absolute
            ),
            "retention_error_percent": (
                retention_error_percent
            ),
        }

    return {
        "full_retention": full_retention,
        "full_spread": full_spread,
        "candidates": candidate_results,
    }


# ============================================================
# Reproducible realization seeds
# ============================================================

def generate_realization_seeds(
    number_of_realizations: int,
) -> list[int]:
    """
    Generate deterministic independent realization seeds.
    """

    if number_of_realizations <= 0:
        raise ValueError(
            "number_of_realizations must be positive."
        )

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        number_of_realizations
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in child_sequences
    ]


# ============================================================
# Multiprocessing worker
# ============================================================

def realization_worker(
    arguments,
) -> dict:
    """
    Multiprocessing worker.
    """

    (
        realization_seed,
        beam_family,
    ) = arguments

    return simulate_and_analyze(
        realization_seed=realization_seed,
        beam_family=beam_family,
    )


# ============================================================
# Ensemble
# ============================================================

def run_ensemble(
    beam_family: str,
    number_of_realizations: int,
    number_of_workers: int,
) -> list[dict]:
    """
    Run the fixed modal-truncation ensemble for one beam family.
    """

    seeds = generate_realization_seeds(
        number_of_realizations
    )

    worker_arguments = [
        (
            seed,
            beam_family,
        )
        for seed in seeds
    ]

    results = []

    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        iterator = executor.map(
            realization_worker,
            worker_arguments,
        )

        for index, result in enumerate(
            iterator,
            start=1,
        ):

            results.append(
                result
            )

            if (
                index == 1
                or index % 10 == 0
                or index == number_of_realizations
            ):
                print(
                    f"{beam_family.upper()}: "
                    f"{index}/"
                    f"{number_of_realizations}"
                )

    return results

# ============================================================
# Ensemble summary
# ============================================================

def summarize_results(
    results,
):
    """
    Summarize the modal-truncation results.
    """

    summary = []

    # --------------------------------------------------------
    # Statistics for every candidate modal interval
    # --------------------------------------------------------

    for ell_max in ELL_MAX_CANDIDATES:

        outside_power_values = np.asarray(
            [
                result[
                    "candidates"
                ][ell_max][
                    "outside_power"
                ]
                for result in results
            ],
            dtype=np.float64,
        )
        
        retention_values = np.asarray(
            [
                result[
                    "candidates"
                ][ell_max][
                    "retention"
                ]
                for result in results
            ],
            dtype=np.float64,
        )
        
        spread_values = np.asarray(
            [
                result[
                    "candidates"
                ][ell_max][
                    "spread"
                ]
                for result in results
            ],
            dtype=np.float64,
        )

        summary.append(
            {
                "ell_max": ell_max,
        
                "eta_out_mean": float(
                    np.mean(
                        outside_power_values
                    )
                ),
        
                "eta_out_max": float(
                    np.max(
                        outside_power_values
                    )
                ),
        
                "retention_mean": float(
                    np.mean(
                        retention_values
                    )
                ),
        
                "spread_mean": float(
                    np.mean(
                        spread_values
                    )
                ),
        
                "spread_max": float(
                    np.max(
                        spread_values
                    )
                ),
            }
        )


    previous_spread = None
    
    for row in summary:
    
        if previous_spread is None:
    
            row[
                "spread_increment_percent"
            ] = np.nan
    
        else:
    
            row[
                "spread_increment_percent"
            ] = (
                100.0
                * abs(
                    row["spread_mean"]
                    - previous_spread
                )
                / row["spread_mean"]
            )
    
        previous_spread = (
            row["spread_mean"]
        )

    return summary


# ============================================================
# Recommended modal range
# ============================================================

def find_recommended_ell_max(
    summary: list[dict],
) -> int | None:
    """
    Return the first ell_max satisfying the outside-power
    criterion.
    """

    for row in summary:

        if (
            row["eta_out_max"]
            < OUTSIDE_POWER_THRESHOLD
        ):
            return int(
                row["ell_max"]
            )

    return None


# ============================================================
# Print results
# ============================================================

def print_summary(
    summary: list[dict],
    number_of_realizations: int,
    beam_family: str,
) -> None:
    """
    Print modal-truncation results.
    """
    family_label = (
        "LG_0^3"
        if beam_family.lower() == "lg"
        else "BG^3"
    )
    print()
    print(
        "Prueba de truncamiento modal OAM"
    )
    print(
        "================================"
    )

    print(
        f"Caso: {family_label} + Kolmogorov fuerte"
    )

    print(
        f"Realizaciones: "
        f"{number_of_realizations}"
    )

    print(
        f"Dominio completo: "
        f"[{FULL_ELL_MIN}, "
        f"{FULL_ELL_MAX}]"
    )

    print(
        f"Criterio eta_out: "
        f"{OUTSIDE_POWER_THRESHOLD:.1e}"
    )

    print()

    print(
        f"{'ell_max':>8} "
        f"{'<eta_out>':>14} "
        f"{'max eta_out':>14} "
        f"{'<sigma>':>14} "
        f"{'Delta sigma [%]':>17}"
    )

    print(
        "-" * 76
    )

    for row in summary:
    
        delta_sigma = (
            row["spread_increment_percent"]
        )
    
        if np.isnan(delta_sigma):
            delta_text = "-"
        else:
            delta_text = (
                f"{delta_sigma:.6f}"
            )
    
        print(
            f"{row['ell_max']:8d} "
            f"{row['eta_out_mean']:14.6e} "
            f"{row['eta_out_max']:14.6e} "
            f"{row['spread_mean']:14.6f} "
            f"{delta_text:>17}"
        )

    recommended = (
        find_recommended_ell_max(
            summary
        )
    )

    print()

    if recommended is None:

        print(
            "Ningún intervalo candidato satisface "
            "el criterio de potencia excluida."
        )

    else:

        print(
            "Primer intervalo que satisface "
            "eta_out < criterio:"
        )

        print(
            f"ell_max = {recommended}"
        )

        print(
            f"Intervalo = "
            f"[-{recommended}, "
            f"{recommended}]"
        )

        print(
            f"N_ell = "
            f"{2 * recommended + 1}"
        )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    summary: list[dict],
    beam_family: str,
) -> None:
    """
    Save modal-truncation summary for one beam family.
    """

    family_directory = (
        OUTPUT_DIRECTORY
        / beam_family.lower()
    )

    family_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        family_directory
        / "oam_modal_truncation_summary.csv"
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
                "ell_max",
                "eta_out_mean",
                "eta_out_max",
                "spread_mean",
                "spread_increment_percent",
            ]
        )

        for row in summary:

            writer.writerow(
                [
                    row["ell_max"],
                    row["eta_out_mean"],
                    row["eta_out_max"],
                    row["spread_mean"],
                    row[
                        "spread_increment_percent"
                    ],
                ]
            )

# ============================================================
# Plot
# ============================================================

def plot_truncation_results(
    summary: list[dict],
    beam_family: str,
) -> None:
    """
    Plot modal-truncation diagnostics for one beam family.

    The figures are saved in a family-specific directory so that
    LG and BG results remain separated.
    """

    family_directory = (
        OUTPUT_DIRECTORY
        / beam_family.lower()
    )

    family_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ell_max_values = np.array(
        [
            row["ell_max"]
            for row in summary
        ],
        dtype=np.int64,
    )

    eta_out_mean = np.array(
        [
            row["eta_out_mean"]
            for row in summary
        ],
        dtype=np.float64,
    )

    eta_out_max = np.array(
        [
            row["eta_out_max"]
            for row in summary
        ],
        dtype=np.float64,
    )

    spread_mean = np.array(
        [
            row["spread_mean"]
            for row in summary
        ],
        dtype=np.float64,
    )

    spread_increment = np.array(
        [
            row["spread_increment_percent"]
            for row in summary
        ],
        dtype=np.float64,
    )

    family_label = (
        r"$\mathrm{LG}_0^3$"
        if beam_family.lower() == "lg"
        else r"$\mathrm{BG}^3$"
    )

    # ========================================================
    # Excluded OAM power
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        ell_max_values,
        eta_out_mean,
        marker="o",
        linewidth=1.5,
        label=r"$\langle\eta_{\ell,\mathrm{out}}\rangle$",
    )

    axis.plot(
        ell_max_values,
        eta_out_max,
        marker="s",
        linewidth=1.5,
        label=r"$\max(\eta_{\ell,\mathrm{out}})$",
    )

    axis.axhline(
        1.0e-3,
        linestyle="--",
        linewidth=1.2,
        label=r"Criterio $10^{-3}$",
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        r"$\ell_{\max}$"
    )

    axis.set_ylabel(
        r"Potencia OAM excluida"
    )

    axis.set_title(
        "Truncamiento modal OAM: "
        + family_label
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        family_directory
        / "oam_truncation_excluded_power.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # ========================================================
    # RMS OAM spread
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    axis.plot(
        ell_max_values,
        spread_mean,
        marker="o",
        linewidth=1.5,
    )

    axis.set_xlabel(
        r"$\ell_{\max}$"
    )

    axis.set_ylabel(
        r"$\langle\sigma_{\Delta\ell}\rangle$"
    )

    axis.set_title(
        "Convergencia de la anchura OAM: "
        + family_label
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        family_directory
        / "oam_truncation_spread.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # ========================================================
    # Relative change in RMS spread
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    valid = np.isfinite(
        spread_increment
    )

    axis.plot(
        ell_max_values[valid],
        spread_increment[valid],
        marker="o",
        linewidth=1.5,
    )

    axis.set_xlabel(
        r"$\ell_{\max}$"
    )

    axis.set_ylabel(
        r"$\Delta\sigma_{\Delta\ell}$ [\%]"
    )

    axis.set_title(
        "Estabilización de la anchura OAM: "
        + family_label
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        family_directory
        / "oam_truncation_spread_increment.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

# ============================================================
# CLI
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

    return parser.parse_args()

def analyze_field_truncation(
    grid,
    field,
) -> list[dict]:
    """
    Evaluate modal truncation directly for one supplied field.
    """

    (
        ell_values,
        full_power,
    ) = calculate_full_spectrum(
        grid=grid,
        field=field,
    )

    full_retention = modal_power_at_charge(
        ell_values=ell_values,
        modal_power=full_power,
        charge=ENSEMBLE_BEAM_CHARGE,
    )

    full_spread = calculate_rms_oam_spread(
        ell_values=ell_values,
        modal_power=full_power,
        transmitted_charge=(
            ENSEMBLE_BEAM_CHARGE
        ),
    )

    rows = []

    for ell_max in ELL_MAX_CANDIDATES:

        metrics = analyze_modal_window(
            ell_values=ell_values,
            full_power=full_power,
            ell_max=ell_max,
            transmitted_charge=(
                ENSEMBLE_BEAM_CHARGE
            ),
        )

        rows.append(
            {
                "ell_max": ell_max,
                "outside_power": (
                    metrics["outside_power"]
                ),
                "retention": (
                    metrics["retention"]
                ),
                "spread": (
                    metrics["spread"]
                ),
                "full_retention": (
                    full_retention
                ),
                "full_spread": (
                    full_spread
                ),
            }
        )

    return rows


def print_single_field_truncation(
    title: str,
    rows: list[dict],
) -> None:
    """
    Print truncation behavior for one deterministic field.
    """

    print()
    print(title)
    print("=" * len(title))

    print(
        f"{'ell_max':>8} "
        f"{'eta_out':>14} "
        f"{'retention':>14} "
        f"{'sigma_Dell':>14}"
    )

    print(
        "-" * 55
    )

    for row in rows:

        print(
            f"{row['ell_max']:8d} "
            f"{row['outside_power']:14.6e} "
            f"{row['retention']:14.8f} "
            f"{row['spread']:14.8f}"
        )

    print()
    print(
        "Referencia espectro completo:"
    )

    print(
        "P_l0 = "
        f"{rows[0]['full_retention']:.10f}"
    )

    print(
        "sigma_Dell = "
        f"{rows[0]['full_spread']:.10f}"
    )
# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Execute the OAM modal-truncation test for LG^3 and BG^3.
    """

    arguments = parse_arguments()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for beam_family in (
        "lg",
        "bg",
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"Familia: "
            f"{beam_family.upper()}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Deterministic input / vacuum checks
        # ----------------------------------------------------

        grid, input_field = create_input_beam(
            beam_family=beam_family,
        )

        input_rows = analyze_field_truncation(
            grid=grid,
            field=input_field,
        )

        print_single_field_truncation(
            title=(
                f"{beam_family.upper()}^3 incidente"
            ),
            rows=input_rows,
        )

        vacuum_field = angular_spectrum_propagation(
            field=input_field,
            wavelength=WAVELENGTH,
            distance=(
                TOTAL_PROPAGATION_DISTANCE
            ),
            dx=DX,
        )

        vacuum_rows = analyze_field_truncation(
            grid=grid,
            field=vacuum_field,
        )

        print_single_field_truncation(
            title=(
                f"{beam_family.upper()}^3 "
                "después de 1 km en vacío"
            ),
            rows=vacuum_rows,
        )

        # ----------------------------------------------------
        # Strong-turbulence ensemble
        # ----------------------------------------------------

        results = run_ensemble(
            beam_family=beam_family,
            number_of_realizations=(
                arguments.realizations
            ),
            number_of_workers=(
                arguments.workers
            ),
        )

        summary = summarize_results(
            results
        )

        print_summary(
            summary=summary,
            number_of_realizations=(
                arguments.realizations
            ),
            beam_family=beam_family,
        )

        save_summary(
            summary=summary,
            beam_family=beam_family,
        )

        plot_truncation_results(
            summary=summary,
            beam_family=beam_family,
        )

if __name__ == "__main__":
    main()
