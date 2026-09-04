from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv


# ============================================================
# Production-code imports
# ============================================================

from experiments.chapter_5.run_scenario import (
    DX,
    ELL_MAX,
    ELL_MIN,
    HALF_SCREEN_SPACING,
    N_GRID,
    NUMBER_OF_PHASE_SCREENS,
    RADIAL_SAMPLES,
    AZIMUTHAL_SAMPLES,
    REGIME_PARAMETERS,
    W0_LG,
    WAVELENGTH,
    angular_spectrum_propagation,
    bg_parameters,
    calculate_oam_spectrum,
    create_grid,
    generate_phase_screen,
    generate_screen_seeds,
    normalize_field,
)


# ============================================================
# Configuration
# ============================================================

PSD_NAME = "kolmogorov"
REGIME_NAME = "strong"

FAMILY = "LG"
ORDER = 2

# Start with a diagnostic ensemble.
# If everything behaves correctly, it can later be increased.
NUMBER_OF_REALIZATIONS = 500

NUMBER_OF_WORKERS = 12

POSITIVE_RESULTS_DIRECTORY = (
    Path("results")
    / "chapter_5"
    / PSD_NAME
    / REGIME_NAME
    / f"{FAMILY}{ORDER:02d}"
)

OUTPUT_DIRECTORY = (
    Path("results")
    / "chapter_5"
    / "analysis"
    / "oam_sign_control"
    / PSD_NAME
    / REGIME_NAME
    / f"{FAMILY}{ORDER:02d}"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Explicit-charge beam
# ============================================================

def create_input_beam_with_charge(
    family: str,
    charge: int,
):
    """
    Generate LG or BG with an explicitly signed OAM charge.

    The radial profile depends on |ell|, while the sign is
    introduced only through exp(i ell phi). Therefore +ell
    and -ell have identical initial intensity distributions.
    """

    if charge == 0:
        raise ValueError(
            "This diagnostic requires a non-zero charge."
        )

    grid = create_grid(
        n=N_GRID,
        window_size=N_GRID * DX,
    )

    radius = np.hypot(
        grid.X,
        grid.Y,
    )

    azimuth = np.arctan2(
        grid.Y,
        grid.X,
    )

    order = abs(
        charge
    )

    # --------------------------------------------------------
    # LG
    # --------------------------------------------------------

    if family == "LG":

        field = (
            (
                np.sqrt(2.0)
                * radius
                / W0_LG
            ) ** order
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
    # BG
    # --------------------------------------------------------

    elif family == "BG":

        (
            w0,
            kr,
        ) = bg_parameters(
            order
        )

        field = (
            jv(
                order,
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
            f"Unknown family: {family}"
        )

    field = np.asarray(
        field,
        dtype=np.complex128,
    )

    return (
        grid,
        normalize_field(
            field
        ),
    )


# ============================================================
# One signed-charge realization
# ============================================================

def simulate_negative_realization(
    realization_seed: int,
    family: str,
    order: int,
    psd_name: str,
    regime_name: str,
) -> np.ndarray:

    charge = (
        -abs(order)
    )

    (
        grid,
        field,
    ) = create_input_beam_with_charge(
        family=family,
        charge=charge,
    )

    r0_screen = (
        REGIME_PARAMETERS[
            regime_name
        ]["r0_screen"]
    )

    screen_seeds = generate_screen_seeds(
        realization_seed
    )

    # --------------------------------------------------------
    # Same split-step realization as production
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

        rng = np.random.default_rng(
            screen_seeds[
                screen_index
            ]
        )

        phase = generate_phase_screen(
            psd_name=psd_name,
            r0_screen=r0_screen,
            rng=rng,
        )

        field *= np.exp(
            1j
            * phase
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
        radial_samples=RADIAL_SAMPLES,
        azimuthal_samples=AZIMUTHAL_SAMPLES,
    )

    return np.asarray(
        modal_power,
        dtype=np.float64,
    )


# ============================================================
# Load the exact production seeds and +ell spectra
# ============================================================

def load_positive_results(
    number_of_realizations: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    metrics_file = (
        POSITIVE_RESULTS_DIRECTORY
        / "metrics.csv"
    )

    spectra_file = (
        POSITIVE_RESULTS_DIRECTORY
        / "oam_spectra.npz"
    )

    seeds = []

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

            if (
                len(seeds)
                >= number_of_realizations
            ):
                break

    seeds = np.asarray(
        seeds,
        dtype=np.uint64,
    )

    with np.load(
        spectra_file
    ) as archive:

        ell_values = np.asarray(
            archive[
                "ell_values"
            ],
            dtype=np.int64,
        )

        positive_spectra = np.asarray(
            archive[
                "modal_power"
            ][
                :number_of_realizations
            ],
            dtype=np.float64,
        )

    if (
        positive_spectra.shape[0]
        != seeds.size
    ):

        raise RuntimeError(
            "Number of positive spectra and seeds does not match."
        )

    return (
        seeds,
        ell_values,
        positive_spectra,
    )


# ============================================================
# Local and integrated asymmetry
# ============================================================

def calculate_local_asymmetry(
    ell: np.ndarray,
    spectrum: np.ndarray,
    ell0: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    lookup = {
        int(mode): float(power)
        for mode, power in zip(
            ell,
            spectrum,
        )
    }

    maximum_delta = min(
        ell0 - int(ell[0]),
        int(ell[-1]) - ell0,
    )

    if maximum_delta <= 0:

        # For negative ell0 the previous expression must use
        # distances to both numerical bounds.
        maximum_delta = min(
            abs(
                ell0 - int(
                    ell[0]
                )
            ),
            abs(
                int(
                    ell[-1]
                )
                - ell0
            ),
        )

    delta = np.arange(
        1,
        maximum_delta + 1,
        dtype=np.int64,
    )

    asymmetry = np.asarray(
        [
            lookup[
                ell0 + d
            ]
            -
            lookup[
                ell0 - d
            ]
            for d in delta
        ],
        dtype=np.float64,
    )

    return (
        delta,
        asymmetry,
    )


def integrated_asymmetry(
    ell: np.ndarray,
    spectrum: np.ndarray,
    ell0: int,
) -> tuple[
    float,
    float,
]:

    (
        _,
        asymmetry,
    ) = calculate_local_asymmetry(
        ell=ell,
        spectrum=spectrum,
        ell0=ell0,
    )

    return (
        float(
            np.sum(
                asymmetry
            )
        ),
        float(
            np.sum(
                np.abs(
                    asymmetry
                )
            )
        ),
    )


# ============================================================
# Plot
# ============================================================

def plot_control(
    ell: np.ndarray,
    positive_mean: np.ndarray,
    negative_mean: np.ndarray,
) -> None:

    positive_charge = (
        abs(
            ORDER
        )
    )

    negative_charge = (
        -positive_charge
    )

    # Mirror the negative-charge spectrum so both transmitted
    # modes are represented in the same relative coordinate.
    delta_positive = (
        ell
        - positive_charge
    )

    delta_negative = (
        ell
        - negative_charge
    )

    display = 15

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            11.0,
            4.2,
        ),
    )

    # --------------------------------------------------------
    # Spectra
    # --------------------------------------------------------

    mask_positive = (
        np.abs(
            delta_positive
        )
        <= display
    )

    mask_negative = (
        np.abs(
            delta_negative
        )
        <= display
    )

    axes[0].plot(
        delta_positive[
            mask_positive
        ],
        positive_mean[
            mask_positive
        ],
        marker="o",
        markersize=3,
        label=rf"$\ell_0=+{ORDER}$",
    )

    # For the expected mirror comparison we reverse the
    # relative coordinate of the negative case.
    axes[0].plot(
        -delta_negative[
            mask_negative
        ][::-1],
        negative_mean[
            mask_negative
        ][::-1],
        linestyle="--",
        marker="s",
        markersize=3,
        label=rf"$\ell_0=-{ORDER}$, reflejado",
    )

    axes[0].set_xlabel(
        r"$\Delta\ell$"
    )

    axes[0].set_ylabel(
        r"$\langle P(\Delta\ell)\rangle$"
    )

    axes[0].grid(
        alpha=0.25
    )

    axes[0].legend()

    axes[0].set_title(
        "(a) Simetría bajo inversión de carga"
    )

    # --------------------------------------------------------
    # Local asymmetry
    # --------------------------------------------------------

    (
        delta_positive_a,
        asym_positive,
    ) = calculate_local_asymmetry(
        ell=ell,
        spectrum=positive_mean,
        ell0=positive_charge,
    )

    (
        delta_negative_a,
        asym_negative,
    ) = calculate_local_asymmetry(
        ell=ell,
        spectrum=negative_mean,
        ell0=negative_charge,
    )

    mask_p = (
        delta_positive_a
        <= display
    )

    mask_n = (
        delta_negative_a
        <= display
    )

    axes[1].plot(
        delta_positive_a[
            mask_p
        ],
        asym_positive[
            mask_p
        ],
        marker="o",
        label=rf"$A_{{+{ORDER}}}$",
    )

    axes[1].plot(
        delta_negative_a[
            mask_n
        ],
        -asym_negative[
            mask_n
        ],
        linestyle="--",
        marker="s",
        label=rf"$-A_{{-{ORDER}}}$",
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xlabel(
        r"$\Delta\ell$"
    )

    axes[1].set_ylabel(
        r"$A(\Delta\ell)$"
    )

    axes[1].grid(
        alpha=0.25
    )

    axes[1].legend()

    axes[1].set_title(
        "(b) Inversión de la asimetría"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "oam_sign_inversion_control.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY
        / "oam_sign_inversion_control.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    (
        seeds,
        ell_values,
        positive_spectra,
    ) = load_positive_results(
        NUMBER_OF_REALIZATIONS
    )

    print(
        f"Control de signo: "
        f"{FAMILY}, |ell|={ORDER}, "
        f"{PSD_NAME}, {REGIME_NAME}"
    )

    print(
        f"Realizaciones pareadas: "
        f"{seeds.size}"
    )

    worker = partial(
        simulate_negative_realization,
        family=FAMILY,
        order=ORDER,
        psd_name=PSD_NAME,
        regime_name=REGIME_NAME,
    )

    negative_spectra = np.zeros_like(
        positive_spectra
    )

    with ProcessPoolExecutor(
        max_workers=NUMBER_OF_WORKERS
    ) as executor:

        iterator = executor.map(
            worker,
            [
                int(seed)
                for seed in seeds
            ],
            chunksize=1,
        )

        for index, spectrum in enumerate(
            iterator
        ):

            negative_spectra[
                index
            ] = spectrum

            completed = (
                index + 1
            )

            if (
                completed == 1
                or completed % 25 == 0
                or completed == seeds.size
            ):

                print(
                    f"Completadas: "
                    f"{completed}/"
                    f"{seeds.size}"
                )

    positive_mean = np.mean(
        positive_spectra,
        axis=0,
    )

    negative_mean = np.mean(
        negative_spectra,
        axis=0,
    )

    positive_mean /= np.sum(
        positive_mean
    )

    negative_mean /= np.sum(
        negative_mean
    )

    (
        signed_positive,
        absolute_positive,
    ) = integrated_asymmetry(
        ell=ell_values,
        spectrum=positive_mean,
        ell0=ORDER,
    )

    (
        signed_negative,
        absolute_negative,
    ) = integrated_asymmetry(
        ell=ell_values,
        spectrum=negative_mean,
        ell0=-ORDER,
    )

    # --------------------------------------------------------
    # Mirror error
    # --------------------------------------------------------

    mirrored_negative = (
        negative_mean[
            ::-1
        ]
    )

    mirror_l1 = float(
        np.sum(
            np.abs(
                positive_mean
                - mirrored_negative
            )
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "RESULTADOS DEL CONTROL DE INVERSIÓN"
    )

    print(
        "=" * 70
    )

    print(
        f"A_signed(+ell) = "
        f"{signed_positive:+.8f}"
    )

    print(
        f"A_signed(-ell) = "
        f"{signed_negative:+.8f}"
    )

    print(
        f"A_signed(+ell) + "
        f"A_signed(-ell) = "
        f"{signed_positive + signed_negative:+.8e}"
    )

    print()

    print(
        f"A_abs(+ell) = "
        f"{absolute_positive:.8f}"
    )

    print(
        f"A_abs(-ell) = "
        f"{absolute_negative:.8f}"
    )

    print(
        f"ΔA_abs = "
        f"{absolute_positive - absolute_negative:+.8e}"
    )

    print()

    print(
        f"L1 entre P_+(ell) y "
        f"P_-(-ell) = "
        f"{mirror_l1:.8e}"
    )

    # --------------------------------------------------------
    # Save raw negative spectra
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_DIRECTORY
        / "negative_oam_spectra.npz",
        ell_values=ell_values,
        modal_power=negative_spectra,
        seeds=seeds,
    )

    np.savetxt(
        OUTPUT_DIRECTORY
        / "mean_negative_oam_spectrum.csv",
        np.column_stack(
            (
                ell_values,
                negative_mean,
            )
        ),
        delimiter=",",
        header="ell,mean_modal_power",
        comments="",
    )

    plot_control(
        ell=ell_values,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
    )

    print()
    print(
        f"Resultados guardados en:"
    )

    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":

    main()
