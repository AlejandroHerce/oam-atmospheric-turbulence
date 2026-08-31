"""
Análisis del papel del tamaño transversal efectivo en la
dependencia de las métricas OAM con el orden azimutal y
la familia del haz.

Entradas
--------
results/chapter_5/analysis/scenario_summary.csv

Salidas
-------
results/chapter_5/analysis/transverse_size_analysis/
    order_size_metric_changes.csv
    family_size_metric_changes.csv
    size_metric_diagnostic.png
    size_metric_diagnostic.pdf

Se analizan:
    - retención del modo transmitido
    - anchura espectral OAM
    - entropía modal

Comparaciones:
    1. |ell| = 1 -> 2
    2. |ell| = 2 -> 3
    3. LG -> BG a igual |ell|
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Rutas
# ============================================================

ANALYSIS_DIRECTORY = Path(
    "results"
) / "chapter_5" / "analysis"

SCENARIO_FILE = (
    ANALYSIS_DIRECTORY
    / "scenario_summary.csv"
)

OUTPUT_DIRECTORY = (
    ANALYSIS_DIRECTORY
    / "transverse_size_analysis"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Tamaño transversal efectivo
# ============================================================

R_RMS_MEAN = {
    "LG01": 25.4263e-3,
    "LG02": 31.1407e-3,
    "LG03": 35.9582e-3,
    "BG01": 25.4144e-3,
    "BG02": 31.1607e-3,
    "BG03": 35.9351e-3,
}


# ============================================================
# Configuración
# ============================================================

FAMILIES = (
    "LG",
    "BG",
)

ORDERS = (
    1,
    2,
    3,
)

REGIMES = (
    "weak",
    "moderate",
    "strong",
)

PSDS = (
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
)

METRICS = {
    "retention": "retention_mean",
    "spread": "spread_mean",
    "entropy": "entropy_mean",
}

METRIC_LABELS = {
    "retention":
        r"Retención $R_{\ell_0}$",

    "spread":
        r"Anchura $\sigma_{\Delta\ell}$",

    "entropy":
        r"Entropía modal $H$",
}


# ============================================================
# Utilidades
# ============================================================

def beam_name(
    family: str,
    order: int,
) -> str:

    return (
        f"{family}0{order}"
    )


def relative_change(
    final_value: float,
    initial_value: float,
) -> float:
    """
    Cambio relativo porcentual:

        100 * (final - initial) / initial
    """

    if initial_value == 0.0:

        raise ZeroDivisionError(
            "El valor de referencia es cero."
        )

    return (
        100.0
        * (
            final_value
            - initial_value
        )
        / initial_value
    )


def get_scenario(
    dataframe: pd.DataFrame,
    beam: str,
    psd: str,
    regime: str,
) -> pd.Series:

    matches = dataframe[
        (
            dataframe["beam"] == beam
        )
        & (
            dataframe["psd"] == psd
        )
        & (
            dataframe["regime"] == regime
        )
    ]

    if len(matches) != 1:

        raise RuntimeError(
            "No se encontró exactamente un escenario para "
            f"{beam}, {psd}, {regime}. "
            f"Coincidencias: {len(matches)}"
        )

    return matches.iloc[0]


# ============================================================
# Comparaciones entre órdenes
# ============================================================

def calculate_order_changes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    order_pairs = (
        (1, 2),
        (2, 3),
    )

    for family in FAMILIES:

        for psd in PSDS:

            for regime in REGIMES:

                for (
                    initial_order,
                    final_order,
                ) in order_pairs:

                    beam_initial = beam_name(
                        family,
                        initial_order,
                    )

                    beam_final = beam_name(
                        family,
                        final_order,
                    )

                    row_initial = get_scenario(
                        dataframe=dataframe,
                        beam=beam_initial,
                        psd=psd,
                        regime=regime,
                    )

                    row_final = get_scenario(
                        dataframe=dataframe,
                        beam=beam_final,
                        psd=psd,
                        regime=regime,
                    )

                    radius_initial = (
                        R_RMS_MEAN[
                            beam_initial
                        ]
                    )

                    radius_final = (
                        R_RMS_MEAN[
                            beam_final
                        ]
                    )

                    radius_change = (
                        relative_change(
                            radius_final,
                            radius_initial,
                        )
                    )

                    for (
                        metric_name,
                        column_name,
                    ) in METRICS.items():

                        metric_initial = float(
                            row_initial[
                                column_name
                            ]
                        )

                        metric_final = float(
                            row_final[
                                column_name
                            ]
                        )

                        metric_change = (
                            relative_change(
                                metric_final,
                                metric_initial,
                            )
                        )

                        records.append(
                            {
                                "family":
                                    family,

                                "psd":
                                    psd,

                                "regime":
                                    regime,

                                "order_initial":
                                    initial_order,

                                "order_final":
                                    final_order,

                                "beam_initial":
                                    beam_initial,

                                "beam_final":
                                    beam_final,

                                "r_rms_initial_m":
                                    radius_initial,

                                "r_rms_final_m":
                                    radius_final,

                                "r_rms_change_percent":
                                    radius_change,

                                "metric":
                                    metric_name,

                                "metric_initial":
                                    metric_initial,

                                "metric_final":
                                    metric_final,

                                "metric_change_percent":
                                    metric_change,
                            }
                        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Comparaciones LG--BG
# ============================================================

def calculate_family_changes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for order in ORDERS:

        lg_beam = beam_name(
            "LG",
            order,
        )

        bg_beam = beam_name(
            "BG",
            order,
        )

        lg_radius = (
            R_RMS_MEAN[
                lg_beam
            ]
        )

        bg_radius = (
            R_RMS_MEAN[
                bg_beam
            ]
        )

        radius_change = (
            relative_change(
                bg_radius,
                lg_radius,
            )
        )

        for psd in PSDS:

            for regime in REGIMES:

                lg_row = get_scenario(
                    dataframe=dataframe,
                    beam=lg_beam,
                    psd=psd,
                    regime=regime,
                )

                bg_row = get_scenario(
                    dataframe=dataframe,
                    beam=bg_beam,
                    psd=psd,
                    regime=regime,
                )

                for (
                    metric_name,
                    column_name,
                ) in METRICS.items():

                    lg_metric = float(
                        lg_row[
                            column_name
                        ]
                    )

                    bg_metric = float(
                        bg_row[
                            column_name
                        ]
                    )

                    metric_change = (
                        relative_change(
                            bg_metric,
                            lg_metric,
                        )
                    )

                    records.append(
                        {
                            "order":
                                order,

                            "psd":
                                psd,

                            "regime":
                                regime,

                            "lg_beam":
                                lg_beam,

                            "bg_beam":
                                bg_beam,

                            "r_rms_lg_m":
                                lg_radius,

                            "r_rms_bg_m":
                                bg_radius,

                            "r_rms_bg_minus_lg_percent":
                                radius_change,

                            "metric":
                                metric_name,

                            "metric_lg":
                                lg_metric,

                            "metric_bg":
                                bg_metric,

                            "metric_bg_minus_lg_percent":
                                metric_change,
                        }
                    )

    return pd.DataFrame(
        records
    )


# ============================================================
# Resumen numérico
# ============================================================

def print_summary(
    order_changes: pd.DataFrame,
    family_changes: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 90
    )

    print(
        "CAMBIOS DE TAMAÑO ENTRE ÓRDENES"
    )

    print(
        "=" * 90
    )

    unique_size_changes = (
        order_changes[
            [
                "family",
                "order_initial",
                "order_final",
                "r_rms_change_percent",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in unique_size_changes.iterrows():

        print(
            f"{row['family']} "
            f"|ell|={int(row['order_initial'])} -> "
            f"|ell|={int(row['order_final'])}: "
            f"{row['r_rms_change_percent']:+.4f} %"
        )

    print()
    print(
        "=" * 90
    )

    print(
        "CORRELACIÓN ENTRE CAMBIO DE TAMAÑO "
        "Y CAMBIO DE MÉTRICA"
    )

    print(
        "=" * 90
    )

    for metric in METRICS:

        subset = order_changes[
            order_changes[
                "metric"
            ] == metric
        ]

        rho, p_value = spearmanr(
            subset[
                "r_rms_change_percent"
            ],
            subset[
                "metric_change_percent"
            ],
        )

        print()
        print(
            METRIC_LABELS[
                metric
            ]
        )

        print(
            f"  rho = {rho:+.5f}"
        )

        print(
            f"  p   = {p_value:.6e}"
        )

    print()
    print(
        "=" * 90
    )

    print(
        "DIFERENCIAS LG--BG"
    )

    print(
        "=" * 90
    )

    for order in ORDERS:

        subset = family_changes[
            family_changes[
                "order"
            ] == order
        ]

        radius_difference = float(
            subset[
                "r_rms_bg_minus_lg_percent"
            ].iloc[0]
        )

        print()
        print(
            f"|ell| = {order}"
        )

        print(
            f"  diferencia r_rms BG-LG = "
            f"{radius_difference:+.5f} %"
        )

        for metric in METRICS:

            metric_subset = subset[
                subset[
                    "metric"
                ] == metric
            ]

            values = (
                metric_subset[
                    "metric_bg_minus_lg_percent"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            print(
                f"  {metric:10s}: "
                f"mín={np.min(values):+.4f} %, "
                f"mediana={np.median(values):+.4f} %, "
                f"máx={np.max(values):+.4f} %"
            )


# ============================================================
# Figura diagnóstica
# ============================================================

def create_diagnostic_figure(
    order_changes: pd.DataFrame,
    family_changes: pd.DataFrame,
) -> None:

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            12.0,
            5.2,
        ),
    )

    # --------------------------------------------------------
    # Panel (a): cambio de orden
    # --------------------------------------------------------

    axis = axes[0]

    markers = {
        "retention": "o",
        "spread": "s",
        "entropy": "^",
    }

    for metric in METRICS:

        subset = order_changes[
            order_changes[
                "metric"
            ] == metric
        ]

        axis.scatter(
            subset[
                "r_rms_change_percent"
            ],
            subset[
                "metric_change_percent"
            ],
            marker=markers[
                metric
            ],
            label=METRIC_LABELS[
                metric
            ],
            alpha=0.8,
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        r"Cambio relativo de "
        r"$\langle r_{\mathrm{rms}}\rangle_z$ [\%]"
    )

    axis.set_ylabel(
        "Cambio relativo de la métrica [%]"
    )

    axis.set_title(
        r"Cambio de orden $|\ell_0|$"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    axis.text(
        0.02,
        0.96,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Panel (b): LG--BG
    # --------------------------------------------------------

    axis = axes[1]

    for metric in METRICS:

        subset = family_changes[
            family_changes[
                "metric"
            ] == metric
        ]

        axis.scatter(
            subset[
                "r_rms_bg_minus_lg_percent"
            ],
            subset[
                "metric_bg_minus_lg_percent"
            ],
            marker=markers[
                metric
            ],
            label=METRIC_LABELS[
                metric
            ],
            alpha=0.8,
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        r"Diferencia relativa BG--LG en "
        r"$\langle r_{\mathrm{rms}}\rangle_z$ [\%]"
    )

    axis.set_ylabel(
        "Diferencia relativa BG--LG "
        "de la métrica [%]"
    )

    axis.set_title(
        "Comparación LG--BG"
    )

    axis.grid(
        alpha=0.25
    )

    axis.text(
        0.02,
        0.96,
        "(b)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    figure.tight_layout()

    pdf_file = (
        OUTPUT_DIRECTORY
        / "size_metric_diagnostic.pdf"
    )

    png_file = (
        OUTPUT_DIRECTORY
        / "size_metric_diagnostic.png"
    )

    figure.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    figure.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print(
        f"Figura guardada: {png_file}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    dataframe = pd.read_csv(
        SCENARIO_FILE
    )

    order_changes = (
        calculate_order_changes(
            dataframe
        )
    )

    family_changes = (
        calculate_family_changes(
            dataframe
        )
    )

    order_file = (
        OUTPUT_DIRECTORY
        / "order_size_metric_changes.csv"
    )

    family_file = (
        OUTPUT_DIRECTORY
        / "family_size_metric_changes.csv"
    )

    order_changes.to_csv(
        order_file,
        index=False,
    )

    family_changes.to_csv(
        family_file,
        index=False,
    )

    print_summary(
        order_changes=order_changes,
        family_changes=family_changes,
    )

    create_diagnostic_figure(
        order_changes=order_changes,
        family_changes=family_changes,
    )

    print()
    print(
        f"CSV guardado: {order_file}"
    )

    print(
        f"CSV guardado: {family_file}"
    )


if __name__ == "__main__":

    main()
