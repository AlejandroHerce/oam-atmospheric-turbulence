"""
Spectral and spatial sampling analysis for Kolmogorov phase screens.

This script evaluates whether the spatial discretization adopted in
Chapter 2 is also adequate for atmospheric phase screens.

The analysis includes:

    - neighboring-pixel phase differences
    - spatial phase gradients
    - fraction of spectral power close to the Nyquist limit
    - ensemble convergence of representative metrics
    - ensemble-averaged two-dimensional PSD
    - radial PSD profile compared with the Kolmogorov power law

The analysis is performed over an ensemble of independent phase-screen
realizations using the Chapter 3 numerical configuration.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_3 import (
    DEFAULT_SEED,
    DX,
    N_GRID,
    R0,
)

from src.phase_screens import (
    kolmogorov_phase_screen,
)

from src.sampling import (
    SamplingDiagnostics,
    analyze_phase_sampling,
    calculate_relative_psd,
    radial_spectral_profile,
)


# ============================================================
# Experiment configuration
# ============================================================

NUMBER_OF_SCREENS = 500

SPECTRAL_GUARD_FRACTION = 0.8

PERCENTILE = 99.9


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIRECTORY = Path(
    "results/chapter_3/spectral_sampling"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Cumulative statistics
# ============================================================

def cumulative_mean(
    values: np.ndarray,
) -> np.ndarray:
    """
    Calculate the cumulative mean of a one-dimensional sequence.
    """

    sample_number = np.arange(
        1,
        values.size + 1,
    )

    return (
        np.cumsum(values)
        / sample_number
    )


# ============================================================
# Ensemble simulation
# ============================================================

def analyze_ensemble() -> dict:
    """
    Generate and analyze an ensemble of independent Kolmogorov
    phase screens.

    Returns
    -------
    dict
        Individual diagnostics, statistical summaries, and the
        ensemble-averaged relative PSD.
    """

    seed_sequence = np.random.SeedSequence(
        DEFAULT_SEED
    )

    child_sequences = seed_sequence.spawn(
        NUMBER_OF_SCREENS
    )

    individual_results: list[
        SamplingDiagnostics
    ] = []

    total_above_pi = 0
    total_neighbor_pairs = 0

    average_psd = np.zeros(
        (N_GRID, N_GRID),
        dtype=np.float64,
    )

    for screen_index, child_seed in enumerate(
        child_sequences,
        start=1,
    ):
        rng = np.random.default_rng(
            child_seed
        )

        phase_screen = kolmogorov_phase_screen(
            n=N_GRID,
            delta=DX,
            r0=R0,
            rng=rng,
            remove_piston=True,
        )

        (
            diagnostics,
            number_above_pi,
            number_of_pairs,
        ) = analyze_phase_sampling(
            phase=phase_screen,
            delta=DX,
            spectral_guard_fraction=(
                SPECTRAL_GUARD_FRACTION
            ),
            percentile=PERCENTILE,
        )

        individual_results.append(
            diagnostics
        )

        total_above_pi += (
            number_above_pi
        )

        total_neighbor_pairs += (
            number_of_pairs
        )

        average_psd += (
            calculate_relative_psd(
                phase_screen,
                remove_mean=True,
            )
        )

        if (
            screen_index == 1
            or screen_index % 25 == 0
            or screen_index == NUMBER_OF_SCREENS
        ):
            print(
                f"Screens analyzed: "
                f"{screen_index}/{NUMBER_OF_SCREENS}"
            )

    average_psd /= NUMBER_OF_SCREENS

    metric_arrays = {
        "RMS neighbor difference [rad]":
            np.array([
                result.rms_neighbor_difference
                for result in individual_results
            ]),

        "Maximum neighbor difference [rad]":
            np.array([
                result.maximum_neighbor_difference
                for result in individual_results
            ]),

        (
            f"P{PERCENTILE} neighbor difference [rad]"
        ):
            np.array([
                result.percentile_neighbor_difference
                for result in individual_results
            ]),

        "Individual fraction |Delta phase| > pi":
            np.array([
                result.fraction_above_pi
                for result in individual_results
            ]),

        "Maximum gradient [rad/m]":
            np.array([
                result.maximum_gradient
                for result in individual_results
            ]),

        (
            f"P{PERCENTILE} gradient [rad/m]"
        ):
            np.array([
                result.percentile_gradient
                for result in individual_results
            ]),

        "Maximum pixel gradient phase change [rad]":
            np.array([
                result.maximum_gradient_phase_change
                for result in individual_results
            ]),

        (
            f"P{PERCENTILE} pixel gradient "
            f"phase change [rad]"
        ):
            np.array([
                result.percentile_gradient_phase_change
                for result in individual_results
            ]),

        "Nyquist-region spectral-power fraction":
            np.array([
                result.nyquist_power_fraction
                for result in individual_results
            ]),
    }

    summary: dict[str, dict[str, float]] = {}

    for metric_name, values in metric_arrays.items():
        mean = float(
            np.mean(values)
        )

        standard_deviation = float(
            np.std(
                values,
                ddof=1,
            )
        )

        standard_error = (
            standard_deviation
            / np.sqrt(NUMBER_OF_SCREENS)
        )

        summary[metric_name] = {
            "mean":
                mean,

            "standard_deviation":
                standard_deviation,

            "standard_error":
                standard_error,

            "ci95_lower":
                float(
                    mean
                    - 1.96 * standard_error
                ),

            "ci95_upper":
                float(
                    mean
                    + 1.96 * standard_error
                ),

            "minimum":
                float(
                    np.min(values)
                ),

            "maximum":
                float(
                    np.max(values)
                ),
        }

    pooled_fraction_above_pi = (
        total_above_pi
        / total_neighbor_pairs
    )

    return {
        "individual_results":
            individual_results,

        "metric_arrays":
            metric_arrays,

        "summary":
            summary,

        "pooled_fraction_above_pi":
            pooled_fraction_above_pi,

        "total_above_pi":
            total_above_pi,

        "total_neighbor_pairs":
            total_neighbor_pairs,

        "average_psd":
            average_psd,
    }


# ============================================================
# Statistical summary
# ============================================================

def print_summary(
    results: dict,
) -> None:
    """
    Print the main ensemble statistics.
    """

    print()
    print(
        "Phase-screen sampling analysis"
    )
    print(
        "=============================="
    )

    print(
        f"Grid: {N_GRID} x {N_GRID}"
    )

    print(
        f"Spatial sampling: "
        f"{DX:.6e} m"
    )

    print(
        f"Fried parameter: "
        f"{R0:.6e} m"
    )

    print(
        f"Number of screens: "
        f"{NUMBER_OF_SCREENS}"
    )

    print(
        f"Spectral guard: "
        f"{SPECTRAL_GUARD_FRACTION:.1f} "
        f"kappa_N"
    )

    print(
        f"Percentile: "
        f"{PERCENTILE}"
    )

    print()

    print(
        "Global fraction of neighboring differences "
        "|Delta phase| > pi:"
    )

    print(
        f"{results['pooled_fraction_above_pi']:.6e}"
    )

    print()

    header = (
        f"{'Metric':<55}"
        f"{'Mean':>14}"
        f"{'Std. dev.':>14}"
        f"{'Maximum':>14}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for metric_name, statistics in (
        results["summary"].items()
    ):
        print(
            f"{metric_name:<55}"
            f"{statistics['mean']:>14.6e}"
            f"{statistics['standard_deviation']:>14.6e}"
            f"{statistics['maximum']:>14.6e}"
        )


# ============================================================
# Ensemble convergence plots
# ============================================================

def plot_ensemble_convergence(
    results: dict,
) -> None:
    """
    Save cumulative-mean convergence plots for representative
    sampling diagnostics.
    """

    metric_arrays = (
        results["metric_arrays"]
    )

    screen_numbers = np.arange(
        1,
        NUMBER_OF_SCREENS + 1,
    )

    # --------------------------------------------------------
    # Neighbor-difference RMS
    # --------------------------------------------------------

    values = metric_arrays[
        "RMS neighbor difference [rad]"
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 4.5)
    )

    axis.plot(
        screen_numbers,
        cumulative_mean(values),
    )

    axis.set_xlabel(
        "Número de pantallas de fase"
    )

    axis.set_ylabel(
        r"Media acumulada de "
        r"$\mathrm{RMS}(|\Delta\theta|)$ [rad]"
    )

    axis.set_title(
        "Convergencia de las diferencias de fase entre píxeles vecinos"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "convergence_neighbor_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # --------------------------------------------------------
    # High-percentile phase-gradient change
    # --------------------------------------------------------

    gradient_metric_name = (
        f"P{PERCENTILE} pixel gradient "
        f"phase change [rad]"
    )

    values = metric_arrays[
        gradient_metric_name
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 4.5)
    )

    axis.plot(
        screen_numbers,
        cumulative_mean(values),
    )

    axis.axhline(
        np.pi,
        linestyle="--",
        label=r"Límite $\pi$",
    )

    axis.set_xlabel(
        "Número de pantallas de fase"
    )

    axis.set_ylabel(
        rf"Media acumulada de "
        rf"$P_{{{PERCENTILE}}}"
        rf"(\Delta x|\nabla\theta|)$ [rad]"
    )

    axis.set_title(
        "Convergencia del gradiente local de fase"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "convergence_phase_gradient.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # --------------------------------------------------------
    # Spectral power near Nyquist
    # --------------------------------------------------------

    values = metric_arrays[
        "Nyquist-region spectral-power fraction"
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 4.5)
    )

    axis.plot(
        screen_numbers,
        cumulative_mean(values),
    )

    axis.set_xlabel(
        "Número de pantallas de fase"
    )

    axis.set_ylabel(
        r"Media acumulada de $\eta_{\mathrm{N}}$"
    )

    axis.set_title(
        "Convergencia de la potencia espectral próxima a Nyquist"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "convergence_nyquist_power.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Average two-dimensional PSD
# ============================================================

def plot_average_psd(
    results: dict,
) -> None:
    """
    Save the ensemble-averaged two-dimensional relative PSD.
    """

    average_psd = (
        results["average_psd"]
    )

    maximum = float(
        np.max(average_psd)
    )

    if maximum <= 0.0:
        raise ValueError(
            "Average PSD has zero maximum."
        )

    normalized_psd = (
        average_psd
        / maximum
    )

    psd_db = (
        10.0
        * np.log10(
            normalized_psd
            + np.finfo(float).eps
        )
    )

    k = (
        2.0
        * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(
                N_GRID,
                d=DX,
            )
        )
    )

    extent = (
        k[0],
        k[-1],
        k[0],
        k[-1],
    )

    kappa_nyquist = (
        np.pi
        / DX
    )

    guard_radius = (
        SPECTRAL_GUARD_FRACTION
        * kappa_nyquist
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    image = axis.imshow(
        psd_db,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        vmin=-80.0,
        vmax=0.0,
    )

    guard_circle = plt.Circle(
        (0.0, 0.0),
        guard_radius,
        fill=False,
        linestyle="--",
        linewidth=1.5,
    )

    axis.add_patch(
        guard_circle
    )

    axis.set_xlabel(
        r"$k_x$ [rad/m]"
    )

    axis.set_ylabel(
        r"$k_y$ [rad/m]"
    )

    axis.set_title(
        "Espectro de fase promedio del ensamble"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Potencia espectral relativa [dB]"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "average_phase_psd.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Radial PSD profile
# ============================================================

def plot_radial_psd(
    results: dict,
    reference_index: int = 10,
) -> None:
    """
    Save the radial ensemble-averaged PSD and compare it with
    the Kolmogorov kappa^(-11/3) power law.
    """

    average_psd = (
        results["average_psd"]
    )

    (
        kappa,
        profile,
    ) = radial_spectral_profile(
        spectral_power=average_psd,
        delta=DX,
    )

    valid = (
        np.isfinite(kappa)
        & np.isfinite(profile)
        & (kappa > 0.0)
        & (profile > 0.0)
    )

    kappa = (
        kappa[valid]
    )

    profile = (
        profile[valid]
    )

    kappa_nyquist = (
        np.pi
        / DX
    )

    kappa_corner = (
        np.sqrt(2.0)
        * kappa_nyquist
    )

    guard_limit = (
        SPECTRAL_GUARD_FRACTION
        * kappa_nyquist
    )

    plot_region = (
        kappa
        <= kappa_corner
    )

    kappa_plot = (
        kappa[plot_region]
    )

    profile_plot = (
        profile[plot_region]
    )

    maximum_profile = float(
        np.max(profile_plot)
    )

    profile_plot = (
        profile_plot
        / maximum_profile
    )

    isotropic_indices = np.where(
        kappa_plot
        <= kappa_nyquist
    )[0]

    reference_index = int(
        np.clip(
            reference_index,
            0,
            isotropic_indices.size - 1,
        )
    )

    reference_position = (
        isotropic_indices[
            reference_index
        ]
    )

    kappa_reference = (
        kappa_plot[
            reference_position
        ]
    )

    profile_reference = (
        profile_plot[
            reference_position
        ]
    )

    kolmogorov_reference = (
        profile_reference
        * (
            kappa_plot
            / kappa_reference
        ) ** (-11.0 / 3.0)
    )

    reference_region = (
        np.arange(
            kappa_plot.size
        )
        >= reference_position
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.3)
    )

    axis.loglog(
        kappa_plot,
        profile_plot,
        linewidth=2.0,
        label="PSD radial promedio",
    )

    axis.loglog(
        kappa_plot[
            reference_region
        ],
        kolmogorov_reference[
            reference_region
        ],
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Ley de Kolmogorov "
            r"$\propto\kappa^{-11/3}$"
        ),
    )

    axis.axvline(
        guard_limit,
        linestyle="-.",
        linewidth=1.8,
        label=(
            rf"${SPECTRAL_GUARD_FRACTION:.1f}"
            rf"\,\kappa_N$"
        ),
    )

    axis.axvline(
        kappa_nyquist,
        linestyle="--",
        linewidth=1.8,
        label=(
            r"$\kappa_N=\pi/\Delta x$"
        ),
    )

    axis.axvline(
        kappa_corner,
        linestyle=":",
        linewidth=2.0,
        label=(
            r"$\kappa_{\max}="
            r"\sqrt{2}\,\kappa_N$"
        ),
    )

    axis.set_xlim(
        kappa_plot[0],
        kappa_corner,
    )

    axis.set_xlabel(
        r"Número de onda espacial radial "
        r"$\kappa$ [rad/m]"
    )

    axis.set_ylabel(
        "PSD radial promedio normalizada"
    )

    axis.set_title(
        "Perfil radial promedio del espectro de fase"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "radial_phase_psd.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Execution
# ============================================================

def run() -> None:
    """
    Execute the complete Chapter 3 sampling analysis.
    """

    print(
        "Kolmogorov phase-screen sampling analysis"
    )

    print(
        "------------------------------------------"
    )

    print(
        f"Grid: {N_GRID} x {N_GRID}"
    )

    print(
        f"Spatial sampling: {DX:.6e} m"
    )

    print(
        f"Fried parameter: {R0:.6e} m"
    )

    print(
        f"Ensemble size: {NUMBER_OF_SCREENS}"
    )

    print()

    results = analyze_ensemble()

    print_summary(
        results
    )

    plot_ensemble_convergence(
        results
    )

    plot_average_psd(
        results
    )

    plot_radial_psd(
        results
    )

    print(
        "\nFigures saved to:"
        f"\n{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    run()
