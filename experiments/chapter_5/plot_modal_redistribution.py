from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

RESULTS_DIRECTORY = Path(
    "results/chapter_5"
)

ANALYSIS_DIRECTORY = (
    RESULTS_DIRECTORY
    / "analysis"
)

OUTPUT_DIRECTORY = (
    ANALYSIS_DIRECTORY
    / "modal_redistribution"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Representative scenario
# ============================================================

REPRESENTATIVE_PSD = "kolmogorov"

REPRESENTATIVE_BEAM = "LG02"

REPRESENTATIVE_ORDER = 2

REGIMES = (
    "weak",
    "moderate",
    "strong",
)

REGIME_LABELS = {
    "weak": "Débil",
    "moderate": "Moderada",
    "strong": "Fuerte",
}


# ============================================================
# Plot configuration
# ============================================================

# Initially restrict the displayed spectrum to the region
# surrounding the transmitted mode. The complete spectrum is
# still used in all calculations.
DISPLAY_DELTA_ELL = 15


# ============================================================
# Utilities
# ============================================================

def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY
        / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def load_mean_spectrum(
    psd: str,
    regime: str,
    beam: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    filename = (
        RESULTS_DIRECTORY
        / psd
        / regime
        / beam
        / "mean_oam_spectrum.csv"
    )

    data = pd.read_csv(
        filename
    )

    ell = data[
        "ell"
    ].to_numpy(
        dtype=np.int64
    )

    power = data[
        "mean_modal_power"
    ].to_numpy(
        dtype=np.float64
    )

    power = (
        power
        / np.sum(power)
    )

    return (
        ell,
        power,
    )


def load_spectra(
    psd: str,
    regime: str,
    beam: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    filename = (
        RESULTS_DIRECTORY
        / psd
        / regime
        / beam
        / "oam_spectra.npz"
    )

    data = np.load(
        filename
    )

    ell = np.asarray(
        data["ell_values"],
        dtype=np.int64,
    )

    spectra = np.asarray(
        data["modal_power"],
        dtype=np.float64,
    )

    return (
        ell,
        spectra,
    )


# ============================================================
# Local asymmetry
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
        int(
            ell0 - ell[0]
        ),
        int(
            ell[-1] - ell0
        ),
    )

    delta_values = np.arange(
        1,
        maximum_delta + 1,
        dtype=np.int64,
    )

    asymmetry = np.asarray(
        [
            lookup[
                ell0 + delta
            ]
            -
            lookup[
                ell0 - delta
            ]
            for delta in delta_values
        ],
        dtype=np.float64,
    )

    return (
        delta_values,
        asymmetry,
    )


# ============================================================
# Entropy
# ============================================================

def normalized_entropy(
    spectrum: np.ndarray,
) -> float:

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    total = float(
        np.sum(spectrum)
    )

    if total <= 0.0:

        raise ValueError(
            "Spectrum must have positive total power."
        )

    probability = (
        spectrum
        / total
    )

    nonzero = (
        probability > 0.0
    )

    number_of_modes = (
        probability.size
    )

    return float(
        -np.sum(
            probability[nonzero]
            * np.log(
                probability[nonzero]
            )
        )
        / np.log(
            number_of_modes
        )
    )


# ============================================================
# Figure 1
# Representative OAM spectra
# ============================================================

def plot_representative_spectra() -> None:

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            12.0,
            3.8,
        ),
        sharex=True,
        sharey=True,
    )

    for axis, regime in zip(
        axes,
        REGIMES,
    ):

        (
            ell,
            power,
        ) = load_mean_spectrum(
            psd=REPRESENTATIVE_PSD,
            regime=regime,
            beam=REPRESENTATIVE_BEAM,
        )

        delta_ell = (
            ell
            - REPRESENTATIVE_ORDER
        )

        mask = (
            np.abs(
                delta_ell
            )
            <= DISPLAY_DELTA_ELL
        )

        axis.plot(
            delta_ell[mask],
            power[mask],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
        )

        axis.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

        axis.set_title(
            REGIME_LABELS[
                regime
            ]
        )

        axis.set_xlabel(
            r"$\Delta\ell=\ell-\ell_0$"
        )

        axis.grid(
            alpha=0.2
        )

    axes[0].set_ylabel(
        r"$\langle P(\Delta\ell)\rangle$"
    )

    save_figure(
        figure,
        "representative_oam_spectra",
    )


# ============================================================
# Figure 2
# Local spectral asymmetry
# ============================================================

def plot_local_asymmetry() -> None:

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            4.5,
        )
    )

    for regime in REGIMES:

        (
            ell,
            power,
        ) = load_mean_spectrum(
            psd=REPRESENTATIVE_PSD,
            regime=regime,
            beam=REPRESENTATIVE_BEAM,
        )

        (
            delta,
            asymmetry,
        ) = calculate_local_asymmetry(
            ell=ell,
            spectrum=power,
            ell0=REPRESENTATIVE_ORDER,
        )

        mask = (
            delta
            <= DISPLAY_DELTA_ELL
        )

        axis.plot(
            delta[mask],
            asymmetry[mask],
            marker="o",
            markersize=4.0,
            linewidth=1.3,
            label=REGIME_LABELS[
                regime
            ],
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    axis.set_xlabel(
        r"$\Delta\ell$"
    )

    axis.set_ylabel(
        r"$A(\Delta\ell)$"
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend()

    save_figure(
        figure,
        "representative_local_asymmetry",
    )


# ============================================================
# Figure 3
# Global asymmetry by regime
# ============================================================

def plot_global_asymmetry() -> None:

    filename = (
        ANALYSIS_DIRECTORY
        / "spectral_asymmetry_statistics.csv"
    )

    data = pd.read_csv(
        filename
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            9.0,
            4.2,
        )
    )

    positions = np.arange(
        len(REGIMES)
    )

    signed_data = [
        data.loc[
            data["regime"] == regime,
            "signed_asymmetry",
        ].to_numpy()
        for regime in REGIMES
    ]

    absolute_data = [
        data.loc[
            data["regime"] == regime,
            "absolute_asymmetry",
        ].to_numpy()
        for regime in REGIMES
    ]

    axes[0].boxplot(
        signed_data,
        positions=positions,
        widths=0.55,
    )

    axes[1].boxplot(
        absolute_data,
        positions=positions,
        widths=0.55,
    )

    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    for axis in axes:

        axis.set_xticks(
            positions
        )

        axis.set_xticklabels(
            [
                REGIME_LABELS[
                    regime
                ]
                for regime in REGIMES
            ]
        )

        axis.set_xlabel(
            "Régimen de turbulencia"
        )

        axis.grid(
            alpha=0.2,
            axis="y",
        )

    axes[0].set_ylabel(
        r"$A_{\mathrm{signed}}$"
    )

    axes[1].set_ylabel(
        r"$A_{\mathrm{abs}}$"
    )

    axes[0].set_title(
        "(a) Asimetría con signo"
    )

    axes[1].set_title(
        "(b) Magnitud de la asimetría"
    )

    save_figure(
        figure,
        "spectral_asymmetry_by_regime",
    )


# ============================================================
# Figure 4
# Entropy decomposition
# ============================================================

def calculate_entropy_components() -> pd.DataFrame:

    records = []

    for regime in REGIMES:

        entropy_mean_spectrum = []

        mean_entropy_realizations = []

        for psd in (
            "kolmogorov",
            "von_karman",
            "modified_von_karman",
        ):

            for family in (
                "LG",
                "BG",
            ):

                for order in (
                    1,
                    2,
                    3,
                ):

                    beam = (
                        f"{family}{order:02d}"
                    )

                    (
                        ell,
                        spectra,
                    ) = load_spectra(
                        psd=psd,
                        regime=regime,
                        beam=beam,
                    )

                    # Normalize each realization.
                    spectra = (
                        spectra
                        / np.sum(
                            spectra,
                            axis=1,
                            keepdims=True,
                        )
                    )

                    mean_spectrum = np.mean(
                        spectra,
                        axis=0,
                    )

                    mean_spectrum /= np.sum(
                        mean_spectrum
                    )

                    h_mean = normalized_entropy(
                        mean_spectrum
                    )

                    h_realizations = np.asarray(
                        [
                            normalized_entropy(
                                spectrum
                            )
                            for spectrum in spectra
                        ],
                        dtype=np.float64,
                    )

                    entropy_mean_spectrum.append(
                        h_mean
                    )

                    mean_entropy_realizations.append(
                        np.mean(
                            h_realizations
                        )
                    )

        records.append(
            {
                "regime":
                    regime,

                "entropy_of_mean_spectrum":
                    float(
                        np.mean(
                            entropy_mean_spectrum
                        )
                    ),

                "mean_entropy_of_realizations":
                    float(
                        np.mean(
                            mean_entropy_realizations
                        )
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def plot_entropy_gap() -> None:

    data = calculate_entropy_components()

    data[
        "entropy_gap"
    ] = (
        data[
            "entropy_of_mean_spectrum"
        ]
        -
        data[
            "mean_entropy_of_realizations"
        ]
    )

    data.to_csv(
        OUTPUT_DIRECTORY
        / "entropy_components_by_regime.csv",
        index=False,
    )

    positions = np.arange(
        len(REGIMES)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            4.8,
        )
    )

    axis.bar(
        positions - width / 2.0,
        data[
            "mean_entropy_of_realizations"
        ],
        width=width,
        label=r"$\langle H(P)\rangle$",
    )

    axis.bar(
        positions + width / 2.0,
        data[
            "entropy_of_mean_spectrum"
        ],
        width=width,
        label=r"$H(\langle P\rangle)$",
    )

    for index, row in data.iterrows():

        top = max(
            row[
                "mean_entropy_of_realizations"
            ],
            row[
                "entropy_of_mean_spectrum"
            ],
        )

        axis.text(
            positions[index],
            top + 0.008,
            (
                r"$\Delta H="
                f"{row['entropy_gap']:.3f}$"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            REGIME_LABELS[
                regime
            ]
            for regime in REGIMES
        ]
    )

    axis.set_xlabel(
        "Régimen de turbulencia"
    )

    axis.set_ylabel(
        "Entropía OAM normalizada"
    )

    axis.grid(
        alpha=0.2,
        axis="y",
    )

    axis.legend()

    save_figure(
        figure,
        "ensemble_entropy_gap",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "Generando espectros OAM representativos..."
    )

    plot_representative_spectra()

    print(
        "Generando asimetría local..."
    )

    plot_local_asymmetry()

    print(
        "Generando resumen global de asimetría..."
    )

    plot_global_asymmetry()

    print(
        "Generando análisis de brecha de entropía..."
    )

    plot_entropy_gap()

    print()
    print(
        "Resultados guardados en:"
    )
    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":

    main()
