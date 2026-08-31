"""
Chapter 5 — Effect of turbulence strength
=========================================

Post-processing script for the analysis of the dependence of OAM
degradation on turbulence strength.

The script does NOT perform optical propagation simulations.

It reads the previously generated Chapter 5 results and produces:

1. A multi-panel figure showing:
   - transmitted-mode retention
   - OAM spectral RMS spread

   as functions of turbulence regime for all beam families and PSD models.

2. A CSV table quantifying the changes:
   - weak -> moderate
   - moderate -> strong
   - weak -> strong

The purpose is to separate production simulations from visualization
and scientific interpretation.

Run from the repository root:

    python -m experiments.chapter_5.plot_turbulence_effect
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

SCENARIO_SUMMARY_FILE = (
    ANALYSIS_DIRECTORY
    / "scenario_summary.csv"
)

FIGURE_DIRECTORY = (
    ANALYSIS_DIRECTORY
    / "figures"
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

PSDS = (
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
)

PSD_LABELS = {
    "kolmogorov":
        "Kolmogorov",

    "von_karman":
        "von Kármán",

    "modified_von_karman":
        "von Kármán modificado",
}

REGIMES = (
    "weak",
    "moderate",
    "strong",
)

REGIME_LABELS = {
    "weak":
        "Débil",

    "moderate":
        "Moderada",

    "strong":
        "Fuerte",
}

BEAMS = (
    "LG01",
    "LG02",
    "LG03",
    "BG01",
    "BG02",
    "BG03",
)

BEAM_LABELS = {
    "LG01":
        r"$\mathrm{LG}_0^1$",

    "LG02":
        r"$\mathrm{LG}_0^2$",

    "LG03":
        r"$\mathrm{LG}_0^3$",

    "BG01":
        r"$\mathrm{BG}^{1}$",

    "BG02":
        r"$\mathrm{BG}^{2}$",

    "BG03":
        r"$\mathrm{BG}^{3}$",
}


# ============================================================
# CSV utilities
# ============================================================

def read_csv(
    filename: Path,
) -> list[dict]:
    """
    Read a CSV file without requiring pandas.
    """

    with open(
        filename,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def write_csv(
    filename: Path,
    records: list[dict],
) -> None:
    """
    Write a list of dictionaries to CSV.
    """

    if not records:
        return

    with open(
        filename,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# Data access
# ============================================================

def find_scenario(
    rows: list[dict],
    beam: str,
    psd: str,
    regime: str,
) -> dict:
    """
    Return one scenario from scenario_summary.csv.
    """

    matches = [
        row
        for row in rows
        if (
            row["beam"] == beam
            and row["psd"] == psd
            and row["regime"] == regime
        )
    ]

    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one scenario for "
            f"{beam}, {psd}, {regime}; "
            f"found {len(matches)}."
        )

    return matches[0]


def get_metric(
    row: dict,
    metric: str,
) -> float:
    """
    Extract the ensemble estimate for one metric.
    """

    return float(
        row[metric]
    )


# ============================================================
# Confidence intervals
# ============================================================

def get_confidence_interval(
    row: dict,
    metric: str,
) -> tuple[float, float] | None:
    """
    Try to obtain the bootstrap 95 % confidence interval stored
    in scenario_summary.csv.

    Different analysis-script versions may use slightly different
    column names, so several common conventions are checked.

    If no interval is available, return None.
    """

    candidates = (
        (
            f"{metric}_ci95_lower",
            f"{metric}_ci95_upper",
        ),
        (
            f"{metric}_lower",
            f"{metric}_upper",
        ),
    )

    for lower_key, upper_key in candidates:

        if (
            lower_key in row
            and upper_key in row
            and row[lower_key] != ""
            and row[upper_key] != ""
        ):

            return (
                float(
                    row[lower_key]
                ),
                float(
                    row[upper_key]
                ),
            )

    return None


# ============================================================
# Plot
# ============================================================

def plot_turbulence_effect(
    rows: list[dict],
) -> None:
    """
    Generate the main turbulence-strength figure.

    Rows:
        retention
        OAM RMS spread

    Columns:
        turbulence PSD
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13.5, 7.5),
        sharex=True,
    )

    x = np.arange(
        len(REGIMES)
    )

    for column, psd in enumerate(
        PSDS
    ):

        # ----------------------------------------------------
        # Retention
        # ----------------------------------------------------

        ax_retention = axes[
            0,
            column,
        ]

        # ----------------------------------------------------
        # Spread
        # ----------------------------------------------------

        ax_spread = axes[
            1,
            column,
        ]

        for beam in BEAMS:

            retention = []
            spread = []

            retention_lower = []
            retention_upper = []

            spread_lower = []
            spread_upper = []

            retention_has_ci = True
            spread_has_ci = True

            for regime in REGIMES:

                row = find_scenario(
                    rows=rows,
                    beam=beam,
                    psd=psd,
                    regime=regime,
                )

                retention_value = (
                    get_metric(
                        row,
                        "retention_mean",
                    )
                )

                spread_value = (
                    get_metric(
                        row,
                        "spread_mean",
                    )
                )

                retention.append(
                    retention_value
                )

                spread.append(
                    spread_value
                )

                retention_ci = (
                    get_confidence_interval(
                        row,
                        "retention",
                    )
                )

                spread_ci = (
                    get_confidence_interval(
                        row,
                        "spread",
                    )
                )

                if retention_ci is None:

                    retention_has_ci = False

                else:

                    retention_lower.append(
                        retention_value
                        - retention_ci[0]
                    )

                    retention_upper.append(
                        retention_ci[1]
                        - retention_value
                    )

                if spread_ci is None:

                    spread_has_ci = False

                else:

                    spread_lower.append(
                        spread_value
                        - spread_ci[0]
                    )

                    spread_upper.append(
                        spread_ci[1]
                        - spread_value
                    )

            # ------------------------------------------------
            # Different line styles for LG and BG
            # ------------------------------------------------

            if beam.startswith(
                "LG"
            ):
                linestyle = "-"
                marker = "o"

            else:
                linestyle = "--"
                marker = "s"

            if retention_has_ci:

                yerr = np.asarray(
                    [
                        retention_lower,
                        retention_upper,
                    ]
                )

                ax_retention.errorbar(
                    x,
                    retention,
                    yerr=yerr,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                    capsize=3,
                    label=BEAM_LABELS[
                        beam
                    ],
                )

            else:

                ax_retention.plot(
                    x,
                    retention,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                    label=BEAM_LABELS[
                        beam
                    ],
                )

            if spread_has_ci:

                yerr = np.asarray(
                    [
                        spread_lower,
                        spread_upper,
                    ]
                )

                ax_spread.errorbar(
                    x,
                    spread,
                    yerr=yerr,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                    capsize=3,
                )

            else:

                ax_spread.plot(
                    x,
                    spread,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                )

        # ----------------------------------------------------
        # Column formatting
        # ----------------------------------------------------

        ax_retention.set_title(
            PSD_LABELS[
                psd
            ]
        )

        ax_retention.grid(
            alpha=0.25
        )

        ax_spread.grid(
            alpha=0.25
        )

        ax_spread.set_xticks(
            x
        )

        ax_spread.set_xticklabels(
            [
                REGIME_LABELS[
                    regime
                ]
                for regime in REGIMES
            ]
        )

        ax_spread.set_xlabel(
            "Régimen de turbulencia"
        )

    # ========================================================
    # Axis labels
    # ========================================================

    axes[
        0,
        0,
    ].set_ylabel(
        r"Retención del modo transmitido $R_{\ell_0}$"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        r"Anchura espectral OAM $\sigma_{\Delta\ell}$"
    )

    # ========================================================
    # Panel labels
    # ========================================================

    panel_labels = (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
    )

    for ax, label in zip(
        axes.flat,
        panel_labels,
    ):

        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    # ========================================================
    # Shared legend
    # ========================================================

    handles, labels = (
        axes[
            0,
            0,
        ].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.06,
            1.0,
            1.0,
        )
    )

    # ========================================================
    # Save
    # ========================================================

    output_pdf = (
        FIGURE_DIRECTORY
        / "turbulence_effect_retention_spread.pdf"
    )

    output_png = (
        FIGURE_DIRECTORY
        / "turbulence_effect_retention_spread.png"
    )

    figure.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    figure.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Figure saved: {output_pdf}"
    )

    print(
        f"Figure saved: {output_png}"
    )


# ============================================================
# Quantitative regime changes
# ============================================================

def calculate_regime_changes(
    rows: list[dict],
) -> list[dict]:
    """
    Quantify changes in retention and spectral spread between
    turbulence regimes.
    """

    comparisons = (
        (
            "weak",
            "moderate",
        ),
        (
            "moderate",
            "strong",
        ),
        (
            "weak",
            "strong",
        ),
    )

    metrics = (
        (
            "retention",
            "retention_mean",
        ),
        (
            "spread",
            "spread_mean",
        ),
    )

    records = []

    for beam in BEAMS:

        family = (
            "LG"
            if beam.startswith(
                "LG"
            )
            else "BG"
        )

        order = int(
            beam[-1]
        )

        for psd in PSDS:

            for (
                regime_initial,
                regime_final,
            ) in comparisons:

                initial_row = (
                    find_scenario(
                        rows,
                        beam,
                        psd,
                        regime_initial,
                    )
                )

                final_row = (
                    find_scenario(
                        rows,
                        beam,
                        psd,
                        regime_final,
                    )
                )

                for (
                    metric_name,
                    column_name,
                ) in metrics:

                    initial = float(
                        initial_row[
                            column_name
                        ]
                    )

                    final = float(
                        final_row[
                            column_name
                        ]
                    )

                    difference = (
                        final
                        - initial
                    )

                    if initial != 0.0:

                        relative_change = (
                            100.0
                            * difference
                            / initial
                        )

                    else:

                        relative_change = (
                            np.nan
                        )

                    records.append(
                        {
                            "beam":
                                beam,

                            "family":
                                family,

                            "order":
                                order,

                            "psd":
                                psd,

                            "metric":
                                metric_name,

                            "regime_initial":
                                regime_initial,

                            "regime_final":
                                regime_final,

                            "initial_value":
                                initial,

                            "final_value":
                                final,

                            "difference_final_minus_initial":
                                difference,

                            "relative_change_percent":
                                relative_change,
                        }
                    )

    return records


# ============================================================
# Terminal summary
# ============================================================

def print_change_summary(
    records: list[dict],
) -> None:
    """
    Print ranges of relative changes across scenarios.
    """

    print()
    print(
        "Relative changes across turbulence regimes"
    )

    print(
        "=" * 80
    )

    for metric in (
        "retention",
        "spread",
    ):

        print()
        print(
            metric.upper()
        )

        for (
            initial,
            final,
        ) in (
            (
                "weak",
                "moderate",
            ),
            (
                "moderate",
                "strong",
            ),
            (
                "weak",
                "strong",
            ),
        ):

            values = np.asarray(
                [
                    row[
                        "relative_change_percent"
                    ]
                    for row in records
                    if (
                        row[
                            "metric"
                        ] == metric
                        and row[
                            "regime_initial"
                        ] == initial
                        and row[
                            "regime_final"
                        ] == final
                    )
                ],
                dtype=np.float64,
            )

            print(
                f"{initial:>8} -> "
                f"{final:<8}: "
                f"min={np.min(values):8.2f}%  "
                f"median={np.median(values):8.2f}%  "
                f"max={np.max(values):8.2f}%"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not (
        SCENARIO_SUMMARY_FILE.exists()
    ):

        raise FileNotFoundError(
            "Scenario summary not found:\n"
            f"{SCENARIO_SUMMARY_FILE}\n\n"
            "Run analyze_final_results.py first."
        )

    rows = read_csv(
        SCENARIO_SUMMARY_FILE
    )

    print(
        "Chapter 5 — turbulence effect analysis"
    )

    print(
        "=" * 50
    )

    print(
        f"Scenarios loaded: {len(rows)}"
    )

    if len(rows) != 54:

        print(
            "WARNING: expected 54 scenarios."
        )

    plot_turbulence_effect(
        rows
    )

    change_records = (
        calculate_regime_changes(
            rows
        )
    )

    output_table = (
        ANALYSIS_DIRECTORY
        / "turbulence_regime_changes.csv"
    )

    write_csv(
        output_table,
        change_records,
    )

    print(
        f"Table saved: {output_table}"
    )

    print_change_summary(
        change_records
    )


if __name__ == "__main__":
    main()
