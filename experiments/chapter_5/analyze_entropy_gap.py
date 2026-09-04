from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

RESULTS_ROOT = Path(
    "results/chapter_5"
)

ANALYSIS_DIRECTORY = (
    RESULTS_ROOT
    / "analysis"
)

OUTPUT_DIRECTORY = (
    ANALYSIS_DIRECTORY
    / "entropy_gap_diagnostic"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Scenario configuration
# ============================================================

PSDS = (
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
)

REGIMES = (
    "weak",
    "moderate",
    "strong",
)

FAMILIES = (
    "LG",
    "BG",
)

ORDERS = (
    1,
    2,
    3,
)

REGIME_LABELS = {
    "weak": "Débil",
    "moderate": "Moderada",
    "strong": "Fuerte",
}


# ============================================================
# Entropy
# ============================================================

def normalized_entropy(
    probability: np.ndarray,
) -> float:

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    probability = (
        probability
        / np.sum(
            probability
        )
    )

    positive = (
        probability > 0.0
    )

    return float(
        -np.sum(
            probability[
                positive
            ]
            * np.log2(
                probability[
                    positive
                ]
            )
        )
        / np.log2(
            probability.size
        )
    )


# ============================================================
# Jensen-Shannon divergence
# ============================================================

def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    mask = (
        p > 0.0
    )

    return float(
        np.sum(
            p[
                mask
            ]
            * np.log2(
                p[
                    mask
                ]
                / q[
                    mask
                ]
            )
        )
    )


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = (
        p
        / np.sum(p)
    )

    q = (
        q
        / np.sum(q)
    )

    mixture = (
        0.5
        * (
            p
            + q
        )
    )

    return float(
        0.5
        * kl_divergence(
            p,
            mixture,
        )
        +
        0.5
        * kl_divergence(
            q,
            mixture,
        )
    )


# ============================================================
# Load one scenario
# ============================================================

def load_scenario(
    psd: str,
    regime: str,
    family: str,
    order: int,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:

    beam = (
        f"{family}{order:02d}"
    )

    directory = (
        RESULTS_ROOT
        / psd
        / regime
        / beam
    )

    metrics = pd.read_csv(
        directory
        / "metrics.csv"
    )

    with np.load(
        directory
        / "oam_spectra.npz"
    ) as data:

        spectra = np.asarray(
            data[
                "modal_power"
            ],
            dtype=np.float64,
        )

    spectra = (
        spectra
        / np.sum(
            spectra,
            axis=1,
            keepdims=True,
        )
    )

    return (
        spectra,
        metrics,
    )


# ============================================================
# Analyze scenario
# ============================================================

def analyze_scenario(
    psd: str,
    regime: str,
    family: str,
    order: int,
) -> dict:

    (
        spectra,
        metrics,
    ) = load_scenario(
        psd=psd,
        regime=regime,
        family=family,
        order=order,
    )

    mean_spectrum = np.mean(
        spectra,
        axis=0,
    )

    mean_spectrum = (
        mean_spectrum
        / np.sum(
            mean_spectrum
        )
    )

    # --------------------------------------------------------
    # Entropy components
    # --------------------------------------------------------

    entropy_of_mean = (
        normalized_entropy(
            mean_spectrum
        )
    )

    realization_entropies = np.asarray(
        [
            normalized_entropy(
                spectrum
            )
            for spectrum in spectra
        ],
        dtype=np.float64,
    )

    mean_entropy = float(
        np.mean(
            realization_entropies
        )
    )

    entropy_gap = float(
        entropy_of_mean
        - mean_entropy
    )

    # --------------------------------------------------------
    # Jensen-Shannon distance from each realization
    # to the ensemble mean spectrum
    # --------------------------------------------------------

    js_values = np.asarray(
        [
            jensen_shannon_divergence(
                spectrum,
                mean_spectrum,
            )
            for spectrum in spectra
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Variability of previously used metrics
    # --------------------------------------------------------

    retention_std = float(
        metrics[
            "retention"
        ].std(
            ddof=1
        )
    )

    spread_std = float(
        metrics[
            "oam_rms_spread"
        ].std(
            ddof=1
        )
    )

    entropy_std = float(
        metrics[
            "normalized_oam_entropy"
        ].std(
            ddof=1
        )
    )

    # --------------------------------------------------------
    # Beam wander RMS
    # --------------------------------------------------------

    centroid_radius = metrics[
        "centroid_radius_m"
    ].to_numpy(
        dtype=np.float64
    )

    beam_wander_rms = float(
        np.sqrt(
            np.mean(
                centroid_radius**2
            )
        )
    )

    return {
        "beam":
            f"{family}{order:02d}",

        "family":
            family,

        "order":
            order,

        "psd":
            psd,

        "regime":
            regime,

        "entropy_of_mean":
            entropy_of_mean,

        "mean_entropy_realizations":
            mean_entropy,

        "entropy_gap":
            entropy_gap,

        "mean_js_divergence":
            float(
                np.mean(
                    js_values
                )
            ),

        "median_js_divergence":
            float(
                np.median(
                    js_values
                )
            ),

        "std_js_divergence":
            float(
                np.std(
                    js_values,
                    ddof=1,
                )
            ),

        "retention_std":
            retention_std,

        "spread_std":
            spread_std,

        "entropy_std":
            entropy_std,

        "beam_wander_rms_m":
            beam_wander_rms,
    }


# ============================================================
# Build complete dataset
# ============================================================

def build_dataset() -> pd.DataFrame:

    records = []

    for psd in PSDS:

        for regime in REGIMES:

            for family in FAMILIES:

                for order in ORDERS:

                    print(
                        f"Analizando: "
                        f"{psd} | "
                        f"{regime} | "
                        f"{family}{order:02d}"
                    )

                    records.append(
                        analyze_scenario(
                            psd=psd,
                            regime=regime,
                            family=family,
                            order=order,
                        )
                    )

    return pd.DataFrame(
        records
    )


# ============================================================
# Correlations
# ============================================================

def print_correlations(
    data: pd.DataFrame,
) -> None:

    variables = (
        "mean_js_divergence",
        "retention_std",
        "spread_std",
        "entropy_std",
        "beam_wander_rms_m",
    )

    print()
    print(
        "=" * 75
    )

    print(
        "CORRELACIONES DE SPEARMAN CON entropy_gap"
    )

    print(
        "=" * 75
    )

    for variable in variables:

        rho = (
            data[
                "entropy_gap"
            ].corr(
                data[
                    variable
                ],
                method="spearman",
            )
        )

        print(
            f"{variable:28s} "
            f"rho = {rho:+.6f}"
        )


# ============================================================
# Regime summary
# ============================================================

def print_regime_summary(
    data: pd.DataFrame,
) -> None:

    summary = (
        data.groupby(
            "regime"
        )
        .agg(
            entropy_mean=(
                "mean_entropy_realizations",
                "mean",
            ),
            entropy_of_mean=(
                "entropy_of_mean",
                "mean",
            ),
            entropy_gap=(
                "entropy_gap",
                "mean",
            ),
            mean_js=(
                "mean_js_divergence",
                "mean",
            ),
            retention_std=(
                "retention_std",
                "mean",
            ),
            spread_std=(
                "spread_std",
                "mean",
            ),
            entropy_std=(
                "entropy_std",
                "mean",
            ),
            wander_rms=(
                "beam_wander_rms_m",
                "mean",
            ),
        )
        .reindex(
            REGIMES
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "RESUMEN POR RÉGIMEN"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string()
    )

    summary.to_csv(
        OUTPUT_DIRECTORY
        / "entropy_gap_summary_by_regime.csv"
    )


# ============================================================
# Figure 1
# Entropy components
# ============================================================

def plot_entropy_components(
    data: pd.DataFrame,
) -> None:

    summary = (
        data.groupby(
            "regime"
        )[
            [
                "mean_entropy_realizations",
                "entropy_of_mean",
                "entropy_gap",
            ]
        ]
        .mean()
        .reindex(
            REGIMES
        )
    )

    x = np.arange(
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
        x - width / 2,
        summary[
            "mean_entropy_realizations"
        ],
        width=width,
        label=r"$\langle H(P_j)\rangle$",
    )

    axis.bar(
        x + width / 2,
        summary[
            "entropy_of_mean"
        ],
        width=width,
        label=r"$H(\langle P\rangle)$",
    )

    for index, regime in enumerate(
        REGIMES
    ):

        gap = summary.loc[
            regime,
            "entropy_gap",
        ]

        top = max(
            summary.loc[
                regime,
                "mean_entropy_realizations",
            ],
            summary.loc[
                regime,
                "entropy_of_mean",
            ],
        )

        axis.text(
            x[index],
            top + 0.008,
            rf"$\Delta H={gap:.3f}$",
            ha="center",
            va="bottom",
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            REGIME_LABELS[
                regime
            ]
            for regime in REGIMES
        ]
    )

    axis.set_ylabel(
        "Entropía OAM normalizada"
    )

    axis.set_xlabel(
        "Régimen de turbulencia"
    )

    axis.grid(
        alpha=0.25,
        axis="y",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_components_by_regime.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_components_by_regime.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Figure 2
# Entropy gap vs Jensen-Shannon divergence
# ============================================================

def plot_gap_vs_js(
    data: pd.DataFrame,
) -> None:

    figure, axis = plt.subplots(
        figsize=(
            6.5,
            5.0,
        )
    )

    for regime in REGIMES:

        subset = data[
            data[
                "regime"
            ] == regime
        ]

        axis.scatter(
            subset[
                "mean_js_divergence"
            ],
            subset[
                "entropy_gap"
            ],
            label=REGIME_LABELS[
                regime
            ],
            alpha=0.8,
        )

    axis.set_xlabel(
        "Divergencia Jensen--Shannon media"
    )

    axis.set_ylabel(
        r"$\Delta H_{\mathrm{ens}}$"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_gap_vs_js_divergence.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_gap_vs_js_divergence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Figure 3
# Diagnostic relationships
# ============================================================

def plot_gap_diagnostics(
    data: pd.DataFrame,
) -> None:

    variables = (
        (
            "retention_std",
            r"$\mathrm{std}(R_{\ell_0})$",
        ),
        (
            "spread_std",
            r"$\mathrm{std}(\sigma_{\Delta\ell})$",
        ),
        (
            "entropy_std",
            r"$\mathrm{std}(H)$",
        ),
        (
            "beam_wander_rms_m",
            r"$r_{\mathrm{wander,rms}}$ [m]",
        ),
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(
            10.0,
            8.0,
        ),
    )

    axes = axes.ravel()

    for axis, (
        variable,
        label,
    ) in zip(
        axes,
        variables,
    ):

        for regime in REGIMES:

            subset = data[
                data[
                    "regime"
                ] == regime
            ]

            axis.scatter(
                subset[
                    variable
                ],
                subset[
                    "entropy_gap"
                ],
                label=REGIME_LABELS[
                    regime
                ],
                alpha=0.8,
            )

        rho = (
            data[
                "entropy_gap"
            ].corr(
                data[
                    variable
                ],
                method="spearman",
            )
        )

        axis.set_xlabel(
            label
        )

        axis.set_ylabel(
            r"$\Delta H_{\mathrm{ens}}$"
        )

        axis.grid(
            alpha=0.25
        )

        axis.set_title(
            rf"$\rho_s={rho:.2f}$"
        )

    axes[0].legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_gap_diagnostics.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY
        / "entropy_gap_diagnostics.png",
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

    data = build_dataset()

    data.to_csv(
        OUTPUT_DIRECTORY
        / "entropy_gap_diagnostic_dataset.csv",
        index=False,
    )

    print_regime_summary(
        data
    )

    print_correlations(
        data
    )

    print()
    print(
        "Generando figuras..."
    )

    plot_entropy_components(
        data
    )

    plot_gap_vs_js(
        data
    )

    plot_gap_diagnostics(
        data
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":

    main()
