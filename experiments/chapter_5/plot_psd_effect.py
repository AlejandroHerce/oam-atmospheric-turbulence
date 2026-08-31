"""
Capítulo 5 — Influencia del modelo espectral de turbulencia
===========================================================

Este script realiza únicamente posprocesamiento.

Genera:

1. Figura de valores absolutos:
   - retención del modo transmitido
   - anchura espectral OAM
   en función del modelo PSD.

2. Figura de diferencias relativas respecto a Kolmogorov:
   - von Kármán vs Kolmogorov
   - von Kármán modificado vs Kolmogorov

   incluyendo intervalos de confianza bootstrap del 95 % calculados
   directamente para la diferencia relativa.

3. Tabla CSV con las diferencias relativas y sus IC95 %.

No se realizan nuevas propagaciones ópticas.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Rutas
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
# Configuración
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

BOOTSTRAP_SAMPLES = 5000

BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

BOOTSTRAP_SEED = 20260830


# ============================================================
# CSV
# ============================================================

def read_csv(
    filename: Path,
) -> list[dict]:

    with filename.open(
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

    if not records:
        return

    with filename.open(
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
# Acceso a escenarios
# ============================================================

def find_scenario(
    rows: list[dict],
    beam: str,
    psd: str,
    regime: str,
) -> dict:

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
            "Se esperaba exactamente un escenario para "
            f"{beam}, {psd}, {regime}; "
            f"se encontraron {len(matches)}."
        )

    return matches[0]


# ============================================================
# Carga de métricas por realización
# ============================================================

def scenario_directory(
    beam: str,
    psd: str,
    regime: str,
) -> Path:

    return (
        RESULTS_DIRECTORY
        / psd
        / regime
        / beam
    )


def load_realization_metric(
    beam: str,
    psd: str,
    regime: str,
    metric: str,
) -> np.ndarray:
    """
    Carga una métrica escalar calculada realización por
    realización desde metrics.csv.
    """

    filename = (
        scenario_directory(
            beam=beam,
            psd=psd,
            regime=regime,
        )
        / "metrics.csv"
    )

    column_map = {
        "retention":
            "retention",

        "spread":
            "oam_rms_spread",
    }

    if metric not in column_map:

        raise ValueError(
            f"Métrica no soportada: {metric}"
        )

    column_name = (
        column_map[
            metric
        ]
    )

    values = []

    with filename.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            values.append(
                float(
                    row[
                        column_name
                    ]
                )
            )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:

        raise RuntimeError(
            f"No se encontraron realizaciones en {filename}"
        )

    return values


# ============================================================
# Bootstrap de diferencia relativa
# ============================================================

def bootstrap_relative_difference(
    samples_a: np.ndarray,
    samples_reference: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Calcula

        100 * [mean(A) - mean(ref)] / mean(ref)

    junto con un IC bootstrap percentil del 95 %.

    Los dos ensambles se remuestrean independientemente porque
    corresponden a canales atmosféricos independientes.
    """

    mean_a = float(
        np.mean(
            samples_a
        )
    )

    mean_reference = float(
        np.mean(
            samples_reference
        )
    )

    if mean_reference == 0.0:

        raise ZeroDivisionError(
            "La media del escenario de referencia es cero."
        )

    observed = (
        100.0
        * (
            mean_a
            - mean_reference
        )
        / mean_reference
    )

    n_a = (
        samples_a.size
    )

    n_reference = (
        samples_reference.size
    )

    bootstrap_values = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices_a = rng.integers(
            low=0,
            high=n_a,
            size=n_a,
        )

        indices_reference = rng.integers(
            low=0,
            high=n_reference,
            size=n_reference,
        )

        bootstrap_mean_a = float(
            np.mean(
                samples_a[
                    indices_a
                ]
            )
        )

        bootstrap_mean_reference = float(
            np.mean(
                samples_reference[
                    indices_reference
                ]
            )
        )

        bootstrap_values[
            index
        ] = (
            100.0
            * (
                bootstrap_mean_a
                - bootstrap_mean_reference
            )
            / bootstrap_mean_reference
        )

    alpha = (
        1.0
        - BOOTSTRAP_CONFIDENCE_LEVEL
    )

    lower = float(
        np.quantile(
            bootstrap_values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        float(
            observed
        ),
        lower,
        upper,
    )


# ============================================================
# Figura 1 — valores absolutos
# ============================================================

def plot_absolute_psd_effect(
    rows: list[dict],
) -> None:
    """
    Filas:
        retención
        anchura espectral

    Columnas:
        débil
        moderada
        fuerte

    Eje x:
        modelo PSD
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13.5, 7.5),
        sharex=True,
    )

    x = np.arange(
        len(
            PSDS
        )
    )

    for column, regime in enumerate(
        REGIMES
    ):

        ax_retention = (
            axes[
                0,
                column,
            ]
        )

        ax_spread = (
            axes[
                1,
                column,
            ]
        )

        for beam in BEAMS:

            retention_values = []
            spread_values = []

            for psd in PSDS:

                row = find_scenario(
                    rows=rows,
                    beam=beam,
                    psd=psd,
                    regime=regime,
                )

                retention_values.append(
                    float(
                        row[
                            "retention_mean"
                        ]
                    )
                )

                spread_values.append(
                    float(
                        row[
                            "spread_mean"
                        ]
                    )
                )

            if beam.startswith(
                "LG"
            ):

                linestyle = "-"
                marker = "o"

            else:

                linestyle = "--"
                marker = "s"

            ax_retention.plot(
                x,
                retention_values,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5,
                label=(
                    BEAM_LABELS[
                        beam
                    ]
                ),
            )

            ax_spread.plot(
                x,
                spread_values,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5,
            )

        ax_retention.set_title(
            REGIME_LABELS[
                regime
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
                PSD_LABELS[
                    psd
                ]
                for psd in PSDS
            ],
            rotation=15,
            ha="right",
        )

        ax_spread.set_xlabel(
            "Modelo espectral de turbulencia"
        )

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

    panel_labels = (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
    )

    for axis, label in zip(
        axes.flat,
        panel_labels,
    ):

        axis.text(
            0.02,
            0.95,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

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
            0.07,
            1.0,
            1.0,
        )
    )

    pdf_filename = (
        FIGURE_DIRECTORY
        / "psd_effect_absolute.pdf"
    )

    png_filename = (
        FIGURE_DIRECTORY
        / "psd_effect_absolute.png"
    )

    figure.savefig(
        pdf_filename,
        bbox_inches="tight",
    )

    figure.savefig(
        png_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Figura guardada: {pdf_filename}"
    )


# ============================================================
# Calcular diferencias relativas
# ============================================================

def calculate_relative_psd_effects() -> list[dict]:
    """
    Compara:

        von Kármán - Kolmogorov
        von Kármán modificado - Kolmogorov

    para retención y anchura espectral.
    """

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    records = []

    comparisons = (
        "von_karman",
        "modified_von_karman",
    )

    metrics = (
        "retention",
        "spread",
    )

    for regime in REGIMES:

        for beam in BEAMS:

            for metric in metrics:

                reference_samples = (
                    load_realization_metric(
                        beam=beam,
                        psd="kolmogorov",
                        regime=regime,
                        metric=metric,
                    )
                )

                for psd in comparisons:

                    comparison_samples = (
                        load_realization_metric(
                            beam=beam,
                            psd=psd,
                            regime=regime,
                            metric=metric,
                        )
                    )

                    (
                        difference,
                        lower,
                        upper,
                    ) = bootstrap_relative_difference(
                        samples_a=(
                            comparison_samples
                        ),
                        samples_reference=(
                            reference_samples
                        ),
                        rng=rng,
                    )

                    records.append(
                        {
                            "beam":
                                beam,

                            "family":
                                (
                                    "LG"
                                    if beam.startswith(
                                        "LG"
                                    )
                                    else "BG"
                                ),

                            "order":
                                int(
                                    beam[-1]
                                ),

                            "regime":
                                regime,

                            "metric":
                                metric,

                            "psd":
                                psd,

                            "reference_psd":
                                "kolmogorov",

                            "relative_difference_percent":
                                difference,

                            "ci95_lower_percent":
                                lower,

                            "ci95_upper_percent":
                                upper,

                            "ci_excludes_zero":
                                (
                                    lower > 0.0
                                    or upper < 0.0
                                ),
                        }
                    )

    return records


# ============================================================
# Figura 2 — diferencias relativas
# ============================================================

def plot_relative_psd_effect(
    records: list[dict],
) -> None:
    """
    Muestra diferencias porcentuales respecto a Kolmogorov.

    Filas:
        retención
        anchura

    Columnas:
        débil
        moderada
        fuerte
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13.5, 7.5),
        sharex=True,
    )

    x = np.arange(
        len(
            BEAMS
        )
    )

    comparison_psds = (
        "von_karman",
        "modified_von_karman",
    )

    offsets = {
        "von_karman":
            -0.12,

        "modified_von_karman":
            0.12,
    }

    markers = {
        "von_karman":
            "o",

        "modified_von_karman":
            "s",
    }

    for column, regime in enumerate(
        REGIMES
    ):

        for row_index, metric in enumerate(
            (
                "retention",
                "spread",
            )
        ):

            axis = axes[
                row_index,
                column,
            ]

            axis.axhline(
                0.0,
                linestyle="--",
                linewidth=1.0,
            )

            for psd in comparison_psds:

                values = []
                lower_errors = []
                upper_errors = []

                for beam in BEAMS:

                    matches = [
                        record
                        for record in records
                        if (
                            record[
                                "beam"
                            ] == beam
                            and record[
                                "regime"
                            ] == regime
                            and record[
                                "metric"
                            ] == metric
                            and record[
                                "psd"
                            ] == psd
                        )
                    ]

                    if len(matches) != 1:

                        raise RuntimeError(
                            "Comparación PSD no encontrada para "
                            f"{beam}, {regime}, {metric}, {psd}"
                        )

                    record = (
                        matches[0]
                    )

                    value = float(
                        record[
                            "relative_difference_percent"
                        ]
                    )

                    lower = float(
                        record[
                            "ci95_lower_percent"
                        ]
                    )

                    upper = float(
                        record[
                            "ci95_upper_percent"
                        ]
                    )

                    values.append(
                        value
                    )

                    lower_errors.append(
                        value
                        - lower
                    )

                    upper_errors.append(
                        upper
                        - value
                    )

                axis.errorbar(
                    x
                    + offsets[
                        psd
                    ],
                    values,
                    yerr=np.asarray(
                        [
                            lower_errors,
                            upper_errors,
                        ]
                    ),
                    marker=markers[
                        psd
                    ],
                    linestyle="none",
                    capsize=3,
                    markersize=5,
                    label=(
                        PSD_LABELS[
                            psd
                        ]
                    ),
                )

            axis.grid(
                alpha=0.25
            )

            if row_index == 0:

                axis.set_title(
                    REGIME_LABELS[
                        regime
                    ]
                )

            axis.set_xticks(
                x
            )

            axis.set_xticklabels(
                [
                    BEAM_LABELS[
                        beam
                    ]
                    for beam in BEAMS
                ]
            )

    axes[
        0,
        0,
    ].set_ylabel(
        "Cambio relativo de la retención [%]"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "Cambio relativo de la anchura [%]"
    )

    for column in range(
        3
    ):

        axes[
            1,
            column,
        ].set_xlabel(
            "Haz transmitido"
        )

    panel_labels = (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
    )

    for axis, label in zip(
        axes.flat,
        panel_labels,
    ):

        axis.text(
            0.02,
            0.95,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

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
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.07,
            1.0,
            1.0,
        )
    )

    pdf_filename = (
        FIGURE_DIRECTORY
        / "psd_effect_relative.pdf"
    )

    png_filename = (
        FIGURE_DIRECTORY
        / "psd_effect_relative.png"
    )

    figure.savefig(
        pdf_filename,
        bbox_inches="tight",
    )

    figure.savefig(
        png_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Figura guardada: {pdf_filename}"
    )


# ============================================================
# Resumen de terminal
# ============================================================

def print_relative_summary(
    records: list[dict],
) -> None:

    print()
    print(
        "Diferencias relativas respecto a Kolmogorov"
    )

    print(
        "=" * 80
    )

    for metric in (
        "retention",
        "spread",
    ):

        print()

        if metric == "retention":

            print(
                "RETENCIÓN"
            )

        else:

            print(
                "ANCHURA ESPECTRAL"
            )

        for regime in REGIMES:

            print()

            print(
                f"Régimen: "
                f"{REGIME_LABELS[regime]}"
            )

            for psd in (
                "von_karman",
                "modified_von_karman",
            ):

                values = np.asarray(
                    [
                        record[
                            "relative_difference_percent"
                        ]
                        for record in records
                        if (
                            record[
                                "metric"
                            ] == metric
                            and record[
                                "regime"
                            ] == regime
                            and record[
                                "psd"
                            ] == psd
                        )
                    ],
                    dtype=np.float64,
                )

                print(
                    f"  {PSD_LABELS[psd]:<24} "
                    f"mín={np.min(values):8.2f}%  "
                    f"mediana={np.median(values):8.2f}%  "
                    f"máx={np.max(values):8.2f}%"
                )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not (
        SCENARIO_SUMMARY_FILE.exists()
    ):

        raise FileNotFoundError(
            "No se encontró scenario_summary.csv. "
            "Ejecute primero analyze_final_results.py."
        )

    rows = read_csv(
        SCENARIO_SUMMARY_FILE
    )

    if len(rows) != 54:

        raise RuntimeError(
            f"Se esperaban 54 escenarios; "
            f"se encontraron {len(rows)}."
        )

    print(
        "Capítulo 5 — influencia del modelo espectral"
    )

    print(
        "=" * 55
    )

    print(
        f"Escenarios cargados: {len(rows)}"
    )

    # --------------------------------------------------------
    # Valores absolutos
    # --------------------------------------------------------

    plot_absolute_psd_effect(
        rows
    )

    # --------------------------------------------------------
    # Diferencias relativas + bootstrap
    # --------------------------------------------------------

    relative_records = (
        calculate_relative_psd_effects()
    )

    output_table = (
        ANALYSIS_DIRECTORY
        / "psd_relative_effects.csv"
    )

    write_csv(
        output_table,
        relative_records,
    )

    plot_relative_psd_effect(
        relative_records
    )

    print_relative_summary(
        relative_records
    )

    print()

    print(
        f"Tabla guardada: "
        f"{output_table}"
    )


if __name__ == "__main__":
    main()
