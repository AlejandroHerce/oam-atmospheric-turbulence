"""
Análisis de la dependencia con el orden azimutal y la familia del haz.

Este script analiza los resultados estadísticos previamente obtenidos
para las 54 configuraciones del Capítulo 5.

Entradas
--------
results/chapter_5/analysis/order_comparisons.csv
results/chapter_5/analysis/lg_bg_comparisons.csv

Salidas
-------
- Resumen en terminal de la dependencia con el orden OAM.
- Resumen estadístico de las diferencias LG--BG.
- Figura de diferencias asociadas al orden.
- Figura de diferencias relativas BG--LG.

El script no requiere pandas y no modifica resultados de simulación.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Rutas
# ============================================================

ANALYSIS_DIRECTORY = (
    Path("results")
    / "chapter_5"
    / "analysis"
)

ORDER_FILE = (
    ANALYSIS_DIRECTORY
    / "order_comparisons.csv"
)

LG_BG_FILE = (
    ANALYSIS_DIRECTORY
    / "lg_bg_comparisons.csv"
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

FAMILIES = (
    "LG",
    "BG",
)

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

METRICS = (
    "retention",
    "spread",
    "entropy",
)

ORDERS = (
    1,
    2,
    3,
)


PSD_LABELS = {
    "kolmogorov":
        "Kolmogorov",

    "von_karman":
        "von Kármán",

    "modified_von_karman":
        "von Kármán modificado",
}


REGIME_LABELS = {
    "weak":
        "Débil",

    "moderate":
        "Moderada",

    "strong":
        "Fuerte",
}


METRIC_LABELS = {
    "retention":
        "Retención",

    "spread":
        "Anchura espectral",

    "entropy":
        "Entropía",
}


# ============================================================
# Lectura
# ============================================================

def parse_boolean(
    value: str,
) -> bool:

    return (
        value.strip().lower()
        == "true"
    )


def read_order_comparisons() -> list[dict]:

    rows = []

    with ORDER_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            rows.append(
                {
                    "family":
                        row["family"],

                    "psd":
                        row["psd"],

                    "regime":
                        row["regime"],

                    "metric":
                        row["metric"],

                    "order_a":
                        int(
                            row["order_a"]
                        ),

                    "order_b":
                        int(
                            row["order_b"]
                        ),

                    "difference":
                        float(
                            row[
                                "difference_a_minus_b"
                            ]
                        ),

                    "lower":
                        float(
                            row[
                                "ci95_lower"
                            ]
                        ),

                    "upper":
                        float(
                            row[
                                "ci95_upper"
                            ]
                        ),

                    "significant":
                        parse_boolean(
                            row[
                                "ci_excludes_zero"
                            ]
                        ),
                }
            )

    return rows


def read_lg_bg_comparisons() -> list[dict]:

    rows = []

    with LG_BG_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            rows.append(
                {
                    "order":
                        int(
                            row["order"]
                        ),

                    "psd":
                        row["psd"],

                    "regime":
                        row["regime"],

                    "metric":
                        row["metric"],

                    "difference":
                        float(
                            row[
                                "difference_BG_minus_LG"
                            ]
                        ),

                    "relative":
                        float(
                            row[
                                "relative_difference_percent"
                            ]
                        ),

                    "lower":
                        float(
                            row[
                                "ci95_lower"
                            ]
                        ),

                    "upper":
                        float(
                            row[
                                "ci95_upper"
                            ]
                        ),

                    "significant":
                        parse_boolean(
                            row[
                                "ci_excludes_zero"
                            ]
                        ),
                }
            )

    return rows


# ============================================================
# Utilidades
# ============================================================

def percentile_summary(
    values: list[float],
) -> tuple[
    float,
    float,
    float,
]:

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(
            np.min(
                array
            )
        ),
        float(
            np.median(
                array
            )
        ),
        float(
            np.max(
                array
            )
        ),
    )


def print_range(
    values: list[float],
    suffix: str = "",
) -> None:

    minimum, median, maximum = (
        percentile_summary(
            values
        )
    )

    print(
        f"mín={minimum:10.5f}{suffix}  "
        f"mediana={median:10.5f}{suffix}  "
        f"máx={maximum:10.5f}{suffix}"
    )


# ============================================================
# Análisis del orden
# ============================================================

def summarize_order_effect(
    rows: list[dict],
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "DEPENDENCIA CON EL ORDEN OAM"
    )
    print(
        "=" * 100
    )

    comparisons = (
        (2, 1),
        (3, 2),
    )

    for metric in METRICS:

        print()
        print(
            METRIC_LABELS[
                metric
            ].upper()
        )

        for regime in REGIMES:

            print()
            print(
                f"Régimen: "
                f"{REGIME_LABELS[regime]}"
            )

            for order_a, order_b in comparisons:

                selected = [
                    row
                    for row in rows
                    if (
                        row["metric"]
                        == metric
                        and row["regime"]
                        == regime
                        and row["order_a"]
                        == order_a
                        and row["order_b"]
                        == order_b
                    )
                ]

                values = [
                    row["difference"]
                    for row in selected
                ]

                significant = sum(
                    row["significant"]
                    for row in selected
                )

                print(
                    f"  |l|={order_b} -> "
                    f"|l|={order_a}: ",
                    end="",
                )

                print_range(
                    values
                )

                print(
                    f"      IC95 % excluye cero: "
                    f"{significant}/"
                    f"{len(selected)}"
                )


# ============================================================
# Consistencia del efecto del orden
# ============================================================

def summarize_order_consistency(
    rows: list[dict],
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "CONSISTENCIA DEL EFECTO DEL ORDEN"
    )
    print(
        "=" * 100
    )

    expected_sign = {
        "retention":
            -1.0,

        "spread":
            1.0,

        "entropy":
            1.0,
    }

    for metric in METRICS:

        metric_rows = [
            row
            for row in rows
            if (
                row["metric"]
                == metric
            )
        ]

        sign = (
            expected_sign[
                metric
            ]
        )

        expected = sum(
            (
                row["difference"]
                * sign
            )
            > 0.0
            for row in metric_rows
        )

        significant_expected = sum(
            (
                (
                    row["difference"]
                    * sign
                )
                > 0.0
            )
            and row["significant"]
            for row in metric_rows
        )

        print()
        print(
            METRIC_LABELS[
                metric
            ]
        )

        print(
            f"  Comparaciones en la dirección esperada: "
            f"{expected}/{len(metric_rows)}"
        )

        print(
            f"  Dirección esperada + IC95 % excluye cero: "
            f"{significant_expected}/{len(metric_rows)}"
        )


# ============================================================
# Análisis LG--BG
# ============================================================

def summarize_lg_bg(
    rows: list[dict],
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "COMPARACIÓN LG--BG"
    )
    print(
        "=" * 100
    )

    for metric in METRICS:

        metric_rows = [
            row
            for row in rows
            if (
                row["metric"]
                == metric
            )
        ]

        relative_values = [
            row["relative"]
            for row in metric_rows
        ]

        significant_rows = [
            row
            for row in metric_rows
            if (
                row["significant"]
            )
        ]

        bg_greater = sum(
            row["difference"] > 0.0
            for row in significant_rows
        )

        lg_greater = sum(
            row["difference"] < 0.0
            for row in significant_rows
        )

        print()
        print(
            METRIC_LABELS[
                metric
            ].upper()
        )

        print(
            "  Diferencia relativa BG respecto a LG:"
        )

        print(
            "    ",
            end="",
        )

        print_range(
            relative_values,
            suffix=" %",
        )

        print(
            f"  IC95 % de diferencia absoluta "
            f"excluye cero: "
            f"{len(significant_rows)}/"
            f"{len(metric_rows)}"
        )

        print(
            f"  Entre las diferencias distinguibles de cero:"
        )

        print(
            f"    BG > LG: "
            f"{bg_greater}"
        )

        print(
            f"    LG > BG: "
            f"{lg_greater}"
        )


# ============================================================
# Mayores diferencias LG--BG
# ============================================================

def print_largest_lg_bg_effects(
    rows: list[dict],
    number: int = 15,
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "MAYORES DIFERENCIAS RELATIVAS LG--BG"
    )
    print(
        "=" * 100
    )

    ordered = sorted(
        rows,
        key=lambda row:
            abs(
                row["relative"]
            ),
        reverse=True,
    )

    print(
        f"{'Orden':>5}  "
        f"{'PSD':>22}  "
        f"{'Régimen':>10}  "
        f"{'Métrica':>18}  "
        f"{'BG-LG [%]':>12}  "
        f"{'IC excluye 0':>13}"
    )

    print(
        "-" * 100
    )

    for row in ordered[
        :number
    ]:

        print(
            f"{row['order']:5d}  "
            f"{PSD_LABELS[row['psd']]:>22}  "
            f"{REGIME_LABELS[row['regime']]:>10}  "
            f"{METRIC_LABELS[row['metric']]:>18}  "
            f"{row['relative']:12.4f}  "
            f"{str(row['significant']):>13}"
        )


# ============================================================
# Figura 1
# Efecto incremental del orden
# ============================================================

def plot_order_effect(
    rows: list[dict],
) -> None:
    """
    Diferencias absolutas al aumentar el orden:

        |l| = 1 -> 2
        |l| = 2 -> 3

    Filas:
        retención
        anchura

    Columnas:
        débil
        moderada
        fuerte

    Marcador:
        familia

    Eje x:
        PSD
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

    family_offsets = {
        "LG":
            -0.08,

        "BG":
            0.08,
    }

    family_markers = {
        "LG":
            "o",

        "BG":
            "s",
    }

    comparison_styles = {
        (2, 1):
            "-",

        (3, 2):
            "--",
    }

    comparison_labels = {
        (2, 1):
            r"$|\ell_0|:1\rightarrow2$",

        (3, 2):
            r"$|\ell_0|:2\rightarrow3$",
    }

    metrics = (
        "retention",
        "spread",
    )

    for column, regime in enumerate(
        REGIMES
    ):

        for row_index, metric in enumerate(
            metrics
        ):

            axis = axes[
                row_index,
                column,
            ]

            axis.axhline(
                0.0,
                linestyle=":",
                linewidth=1.0,
            )

            for family in FAMILIES:

                for comparison in (
                    (2, 1),
                    (3, 2),
                ):

                    order_a, order_b = (
                        comparison
                    )

                    values = []
                    lower_errors = []
                    upper_errors = []

                    for psd in PSDS:

                        matches = [
                            row
                            for row in rows
                            if (
                                row["family"]
                                == family
                                and row["psd"]
                                == psd
                                and row["regime"]
                                == regime
                                and row["metric"]
                                == metric
                                and row["order_a"]
                                == order_a
                                and row["order_b"]
                                == order_b
                            )
                        ]

                        if len(matches) != 1:

                            raise RuntimeError(
                                "Comparación de orden no encontrada: "
                                f"{family}, {psd}, {regime}, "
                                f"{metric}, {order_a}-{order_b}"
                            )

                        record = (
                            matches[0]
                        )

                        value = (
                            record[
                                "difference"
                            ]
                        )

                        values.append(
                            value
                        )

                        lower_errors.append(
                            value
                            - record[
                                "lower"
                            ]
                        )

                        upper_errors.append(
                            record[
                                "upper"
                            ]
                            - value
                        )

                    label = (
                        f"{family}, "
                        f"{comparison_labels[comparison]}"
                    )

                    axis.errorbar(
                        x
                        + family_offsets[
                            family
                        ],
                        values,
                        yerr=np.asarray(
                            [
                                lower_errors,
                                upper_errors,
                            ]
                        ),
                        marker=family_markers[
                            family
                        ],
                        linestyle=comparison_styles[
                            comparison
                        ],
                        linewidth=1.3,
                        markersize=5,
                        capsize=3,
                        label=label,
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
                    PSD_LABELS[
                        psd
                    ]
                    for psd in PSDS
                ],
                rotation=15,
                ha="right",
            )

    axes[
        0,
        0,
    ].set_ylabel(
        r"Cambio en la retención $\Delta R_{\ell_0}$"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        r"Cambio en la anchura $\Delta\sigma_{\Delta\ell}$"
    )

    for column in range(
        3
    ):

        axes[
            1,
            column,
        ].set_xlabel(
            "Modelo espectral de turbulencia"
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
        ncol=4,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.015,
        ),
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.09,
            1.0,
            1.0,
        )
    )

    pdf_filename = (
        FIGURE_DIRECTORY
        / "order_effect.pdf"
    )

    png_filename = (
        FIGURE_DIRECTORY
        / "order_effect.png"
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
        f"Figura guardada: "
        f"{pdf_filename}"
    )


# ============================================================
# Figura 2
# Comparación relativa LG--BG
# ============================================================

def plot_lg_bg_effect(
    rows: list[dict],
) -> None:
    """
    Diferencia relativa:

        100 * (BG - LG) / LG

    Filas:
        retención
        anchura

    Columnas:
        débil
        moderada
        fuerte

    Eje x:
        orden OAM

    Curvas:
        PSD

    Nota:
        Los IC disponibles en lg_bg_comparisons.csv corresponden
        a la diferencia absoluta BG-LG, no a la diferencia
        porcentual. Por ello, esta figura muestra los estimadores
        relativos sin barras de error. La significancia se analiza
        a partir de los IC de las diferencias absolutas.
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13.5, 7.5),
        sharex=True,
    )

    x = np.asarray(
        ORDERS,
        dtype=np.float64,
    )

    psd_markers = {
        "kolmogorov":
            "o",

        "von_karman":
            "s",

        "modified_von_karman":
            "^",
    }

    metrics = (
        "retention",
        "spread",
    )

    for column, regime in enumerate(
        REGIMES
    ):

        for row_index, metric in enumerate(
            metrics
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

            for psd in PSDS:

                values = []

                significant = []

                for order in ORDERS:

                    matches = [
                        row
                        for row in rows
                        if (
                            row["order"]
                            == order
                            and row["psd"]
                            == psd
                            and row["regime"]
                            == regime
                            and row["metric"]
                            == metric
                        )
                    ]

                    if len(matches) != 1:

                        raise RuntimeError(
                            "Comparación LG-BG no encontrada: "
                            f"{order}, {psd}, "
                            f"{regime}, {metric}"
                        )

                    record = (
                        matches[0]
                    )

                    values.append(
                        record[
                            "relative"
                        ]
                    )

                    significant.append(
                        record[
                            "significant"
                        ]
                    )

                axis.plot(
                    x,
                    values,
                    marker=psd_markers[
                        psd
                    ],
                    linewidth=1.4,
                    markersize=6,
                    label=PSD_LABELS[
                        psd
                    ],
                )

                # Marcar comparaciones cuyo IC absoluto
                # excluye cero.
                for order, value, is_significant in zip(
                    ORDERS,
                    values,
                    significant,
                ):

                    if is_significant:

                        axis.scatter(
                            order,
                            value,
                            marker="*",
                            s=70,
                            zorder=5,
                        )

            axis.grid(
                alpha=0.25
            )

            axis.set_xticks(
                ORDERS
            )

            if row_index == 0:

                axis.set_title(
                    REGIME_LABELS[
                        regime
                    ]
                )

    axes[
        0,
        0,
    ].set_ylabel(
        "Diferencia relativa BG--LG\n"
        "en retención [%]"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "Diferencia relativa BG--LG\n"
        "en anchura [%]"
    )

    for column in range(
        3
    ):

        axes[
            1,
            column,
        ].set_xlabel(
            r"Orden azimutal $|\ell_0|$"
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
        ncol=3,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
    )

    figure.text(
        0.5,
        0.035,
        (
            r"$\star$: el IC del 95\,\% de la diferencia "
            r"absoluta BG--LG excluye cero"
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.10,
            1.0,
            1.0,
        )
    )

    pdf_filename = (
        FIGURE_DIRECTORY
        / "lg_bg_relative_effect.pdf"
    )

    png_filename = (
        FIGURE_DIRECTORY
        / "lg_bg_relative_effect.png"
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
        f"Figura guardada: "
        f"{pdf_filename}"
    )


# ============================================================
# Resumen por régimen de las diferencias LG--BG
# ============================================================

def summarize_lg_bg_by_regime(
    rows: list[dict],
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "DIFERENCIAS LG--BG POR RÉGIMEN"
    )
    print(
        "=" * 100
    )

    for metric in METRICS:

        print()
        print(
            METRIC_LABELS[
                metric
            ].upper()
        )

        for regime in REGIMES:

            selected = [
                row
                for row in rows
                if (
                    row["metric"]
                    == metric
                    and row["regime"]
                    == regime
                )
            ]

            values = [
                row["relative"]
                for row in selected
            ]

            significant = [
                row
                for row in selected
                if (
                    row["significant"]
                )
            ]

            print()
            print(
                f"  {REGIME_LABELS[regime]}:"
            )

            print(
                "    BG-LG relativo: ",
                end="",
            )

            print_range(
                values,
                suffix=" %",
            )

            print(
                f"    IC95 % absoluto excluye cero: "
                f"{len(significant)}/"
                f"{len(selected)}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not ORDER_FILE.exists():

        raise FileNotFoundError(
            f"No se encontró: "
            f"{ORDER_FILE}"
        )

    if not LG_BG_FILE.exists():

        raise FileNotFoundError(
            f"No se encontró: "
            f"{LG_BG_FILE}"
        )

    order_rows = (
        read_order_comparisons()
    )

    lg_bg_rows = (
        read_lg_bg_comparisons()
    )

    print()
    print(
        "Análisis de dependencia con el orden "
        "azimutal y familia del haz"
    )

    print(
        "=" * 100
    )

    print(
        f"Comparaciones de orden cargadas: "
        f"{len(order_rows)}"
    )

    print(
        f"Comparaciones LG--BG cargadas: "
        f"{len(lg_bg_rows)}"
    )

    summarize_order_effect(
        order_rows
    )

    summarize_order_consistency(
        order_rows
    )

    summarize_lg_bg(
        lg_bg_rows
    )

    summarize_lg_bg_by_regime(
        lg_bg_rows
    )

    print_largest_lg_bg_effects(
        lg_bg_rows
    )

    print()
    print(
        "=" * 100
    )
    print(
        "GENERANDO FIGURAS"
    )
    print(
        "=" * 100
    )

    plot_order_effect(
        order_rows
    )

    plot_lg_bg_effect(
        lg_bg_rows
    )

    print()
    print(
        "Análisis completado."
    )


if __name__ == "__main__":

    main()
