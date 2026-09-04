"""
Figuras para el análisis de la dependencia con el orden OAM
=============================================================

Este script genera las figuras utilizadas para estudiar la relación

    |ell|
        -> tamaño transversal
        -> eta = <r_rms>_z / r0
        -> degradación modal OAM

a partir de los resultados previamente calculados en el Capítulo 5.

No ejecuta nuevas simulaciones de turbulencia.

Archivos de entrada
-------------------
results/chapter_5/analysis/scenario_summary.csv

    Valores medios de las métricas OAM para cada escenario.

results/chapter_5/analysis/beam_size_evolution.csv

    Evolución de r_rms(z) en propagación libre para los
    haces LG y BG utilizados en las simulaciones.

results/chapter_5/analysis/eta_analysis/eta_dataset.csv

    Dataset que combina el tamaño efectivo de los haces,
    el parámetro de Fried y las métricas modales.

Figuras generadas
-----------------
1. order_absolute_lg
2. order_absolute_bg
3. order_rms_evolution
4. eta_modal_metrics
"""

from __future__ import annotations

import csv
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

FIGURE_DIRECTORY = (
    ANALYSIS_DIRECTORY
    / "figures"
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


SCENARIO_FILE = (
    ANALYSIS_DIRECTORY
    / "scenario_summary.csv"
)

BEAM_SIZE_FILE = (
    ANALYSIS_DIRECTORY
    / "beam_size_evolution.csv"
)

ETA_FILE = (
    ANALYSIS_DIRECTORY
    / "eta_analysis"
    / "eta_dataset.csv"
)


# ============================================================
# Configuración
# ============================================================

ORDERS = (
    1,
    2,
    3,
)

FAMILIES = (
    "LG",
    "BG",
)

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

METRICS = (
    (
        "retention_mean",
        r"Retención $R_{\ell_0}$",
    ),
    (
        "spread_mean",
        r"Anchura OAM $\sigma_{\Delta\ell}$",
    ),
    (
        "entropy_mean",
        r"Entropía modal $H$",
    ),
)


# ============================================================
# Lectura CSV
# ============================================================

def read_csv(
    filename: Path,
) -> list[dict]:
    """
    Leer un CSV como lista de diccionarios.
    """

    if not filename.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{filename}"
        )

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


# ============================================================
# Utilidades
# ============================================================

def beam_name(
    family: str,
    order: int,
) -> str:
    """
    Nombre utilizado en los resultados principales.
    """

    if family == "LG":

        return (
            f"LG0{order}"
        )

    if family == "BG":

        return (
            f"BG0{order}"
        )

    raise ValueError(
        f"Familia desconocida: {family}"
    )


def find_scenario(
    rows: list[dict],
    family: str,
    order: int,
    psd: str,
    regime: str,
) -> dict:
    """
    Encontrar un escenario único en scenario_summary.csv.
    """

    matches = [
        row
        for row in rows
        if (
            row["family"]
            == family

            and int(
                row["order"]
            )
            == order

            and row["psd"]
            == psd

            and row["regime"]
            == regime
        )
    ]

    if len(matches) != 1:

        raise RuntimeError(
            "No se encontró exactamente un escenario para "
            f"{family}, |ell|={order}, {psd}, {regime}. "
            f"Coincidencias: {len(matches)}"
        )

    return matches[0]


def save_figure(
    figure: plt.Figure,
    basename: str,
) -> None:
    """
    Guardar PDF y PNG.
    """

    pdf_file = (
        FIGURE_DIRECTORY
        / f"{basename}.pdf"
    )

    png_file = (
        FIGURE_DIRECTORY
        / f"{basename}.png"
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

    print(
        f"Figura guardada: {pdf_file}"
    )

    print(
        f"Figura guardada: {png_file}"
    )


# ============================================================
# Figura 1
# Valores absolutos de las métricas frente a |ell|
# Una figura independiente para LG y BG
# ============================================================
# ============================================================
# Figura 1
# Métricas absolutas frente al orden OAM
# LG y BG en una sola figura
# ============================================================

# ============================================================
# Figura 1
# Métricas absolutas frente al orden OAM
# LG y BG combinados
# ============================================================

def plot_absolute_order_effect_combined(
    rows: list[dict],
) -> None:

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(
            12.5,
            10.0,
        ),
        sharex=True,
    )

    x = np.asarray(
        ORDERS,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Colores asociados exclusivamente a la PSD
    # --------------------------------------------------------

    color_cycle = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    psd_colors = {
        psd:
            color_cycle[index]
        for index, psd in enumerate(
            PSDS
        )
    }

    # --------------------------------------------------------
    # Estilo asociado exclusivamente a la familia
    # --------------------------------------------------------

    family_styles = {
        "LG": {
            "linestyle": "-",
            "marker": "o",
        },

        "BG": {
            "linestyle": "--",
            "marker": "s",
        },
    }

    for column, regime in enumerate(
        REGIMES
    ):

        for row_index, (
            metric,
            ylabel,
        ) in enumerate(
            METRICS
        ):

            axis = axes[
                row_index,
                column,
            ]

            for psd in PSDS:

                for family in FAMILIES:

                    values = []

                    for order in ORDERS:

                        scenario = find_scenario(
                            rows=rows,
                            family=family,
                            order=order,
                            psd=psd,
                            regime=regime,
                        )

                        values.append(
                            float(
                                scenario[
                                    metric
                                ]
                            )
                        )

                    style = (
                        family_styles[
                            family
                        ]
                    )

                    axis.plot(
                        x,
                        values,
                        color=psd_colors[
                            psd
                        ],
                        marker=style[
                            "marker"
                        ],
                        linestyle=style[
                            "linestyle"
                        ],
                        linewidth=1.4,
                        markersize=4.5,
                    )

            axis.grid(
                alpha=0.25
            )

            axis.set_xticks(
                x
            )

            if row_index == 0:

                axis.set_title(
                    REGIME_LABELS[
                        regime
                    ]
                )

            if column == 0:

                axis.set_ylabel(
                    ylabel
                )

            if row_index == 2:

                axis.set_xlabel(
                    r"Orden azimutal $|\ell|$"
                )

    panel_labels = (
        "(a)", "(b)", "(c)",
        "(d)", "(e)", "(f)",
        "(g)", "(h)", "(i)",
    )

    for axis, label in zip(
        axes.flat,
        panel_labels,
    ):

        axis.text(
            0.03,
            0.95,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Leyenda PSD
    # --------------------------------------------------------

    psd_handles = [
        plt.Line2D(
            [],
            [],
            color=psd_colors[
                psd
            ],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label=PSD_LABELS[
                psd
            ],
        )
        for psd in PSDS
    ]

    # --------------------------------------------------------
    # Leyenda familia
    # --------------------------------------------------------

    family_handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label="LG",
        ),

        plt.Line2D(
            [],
            [],
            color="black",
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="BG",
        ),
    ]

    figure.legend(
        psd_handles
        + family_handles,
        [
            PSD_LABELS[
                psd
            ]
            for psd in PSDS
        ]
        + [
            "LG",
            "BG",
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.055,
            1.0,
            1.0,
        )
    )

    save_figure(
        figure=figure,
        basename="order_absolute_combined",
    )


# ============================================================
# Utilidades para beam_size_evolution.csv
# ============================================================

def identify_size_columns(
    rows: list[dict],
) -> tuple[
    str,
    str,
    str,
]:
    """
    Detectar automáticamente los nombres de las columnas
    principales de beam_size_evolution.csv.

    Esto permite tolerar pequeñas diferencias entre versiones
    del script que generó el CSV.
    """

    if not rows:

        raise RuntimeError(
            "beam_size_evolution.csv está vacío."
        )

    columns = set(
        rows[0].keys()
    )

    beam_candidates = (
        "beam",
        "beam_name",
        "name",
    )

    z_candidates = (
        "z_m",
        "z",
        "distance_m",
    )

    radius_candidates = (
        "r_rms_m",
        "radius_rms_m",
        "rms_radius_m",
        "r_rms",
    )

    beam_column = next(
        (
            candidate
            for candidate in beam_candidates
            if candidate in columns
        ),
        None,
    )

    z_column = next(
        (
            candidate
            for candidate in z_candidates
            if candidate in columns
        ),
        None,
    )

    radius_column = next(
        (
            candidate
            for candidate in radius_candidates
            if candidate in columns
        ),
        None,
    )

    if beam_column is None:

        raise RuntimeError(
            "No se pudo identificar la columna del haz. "
            f"Columnas disponibles: {sorted(columns)}"
        )

    if z_column is None:

        raise RuntimeError(
            "No se pudo identificar la columna de distancia z. "
            f"Columnas disponibles: {sorted(columns)}"
        )

    if radius_column is None:

        raise RuntimeError(
            "No se pudo identificar la columna de r_rms. "
            f"Columnas disponibles: {sorted(columns)}"
        )

    return (
        beam_column,
        z_column,
        radius_column,
    )


# ============================================================
# Figura 2
# Evolución de r_rms(z) frente al orden
# LG y BG en paneles independientes
# ============================================================
# ============================================================
# Figura 2
# Evolución de r_rms(z)
# LG y BG superpuestos
# ============================================================

def plot_order_rms_evolution_combined(
    rows: list[dict],
) -> None:
    """
    Evolución de r_rms(z) para los seis haces.

    Color:
        orden |ell|

    LG:
        línea continua + círculos

    BG:
        línea discontinua + cuadrados

    Esta figura muestra simultáneamente:

    1. el crecimiento del tamaño con |ell|;
    2. la persistencia de esta jerarquía durante la propagación;
    3. la casi coincidencia LG--BG para un mismo orden.
    """

    (
        beam_column,
        z_column,
        radius_column,
    ) = identify_size_columns(
        rows
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            5.8,
        )
    )

    # --------------------------------------------------------
    # Obtener colores una vez por orden
    # --------------------------------------------------------

    color_cycle = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    order_colors = {
        order:
            color_cycle[
                index
                % len(
                    color_cycle
                )
            ]
        for index, order in enumerate(
            ORDERS
        )
    }

    for order in ORDERS:

        for family in FAMILIES:

            expected_beam = beam_name(
                family=family,
                order=order,
            )

            subset = [
                row
                for row in rows
                if (
                    row[
                        beam_column
                    ]
                    == expected_beam
                )
            ]

            subset = sorted(
                subset,
                key=lambda row:
                    float(
                        row[
                            z_column
                        ]
                    ),
            )

            if not subset:

                raise RuntimeError(
                    f"No se encontraron datos para "
                    f"{expected_beam}."
                )

            z = np.asarray(
                [
                    float(
                        row[
                            z_column
                        ]
                    )
                    for row in subset
                ]
            )

            radius_mm = (
                1000.0
                * np.asarray(
                    [
                        float(
                            row[
                                radius_column
                            ]
                        )
                        for row in subset
                    ]
                )
            )

            if family == "LG":

                linestyle = "-"
                marker = "o"
                linewidth = 1.8
                markersize = 5.0

            else:

                linestyle = "--"
                marker = "s"
                linewidth = 1.3
                markersize = 4.0

            axis.plot(
                z,
                radius_mm,
                color=order_colors[
                    order
                ],
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                label=(
                    rf"{family}, $|\ell|={order}$"
                ),
            )

    axis.set_xlabel(
        r"Distancia de propagación $z$ [m]"
    )

    axis.set_ylabel(
        r"Radio RMS $r_{\mathrm{rms}}(z)$ [mm]"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        ncol=2,
        fontsize=9,
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        basename="order_rms_evolution_combined",
    )

# ============================================================
# Utilidades eta_dataset.csv
# ============================================================

def validate_eta_columns(
    rows: list[dict],
) -> None:

    if not rows:

        raise RuntimeError(
            "eta_dataset.csv está vacío."
        )

    required_columns = {
        "family",
        "order",
        "psd",
        "regime",
        "eta_total",
        "retention",
        "spread",
        "entropy",
    }

    available_columns = set(
        rows[0].keys()
    )

    missing = (
        required_columns
        - available_columns
    )

    if missing:

        raise RuntimeError(
            "Faltan columnas en eta_dataset.csv:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
            + "\n\nColumnas disponibles:\n"
            + "\n".join(
                sorted(
                    available_columns
                )
            )
        )

# ============================================================
# Figura 3
# Trayectorias de métricas OAM frente a eta
# ============================================================

# ============================================================
# Figura 3
# Caso representativo eta -> métricas modales
# ============================================================

def plot_eta_representative_case(
    rows: list[dict],
    regime: str = "moderate",
    psd: str = "von_karman",
) -> None:
    """
    Caso representativo de la relación entre eta y
    las métricas OAM.

    Por defecto:
        turbulencia moderada
        von Kármán

    Se escoge el régimen moderado porque eta ~ 1,
    es decir, el tamaño del campo es comparable a r0.

    Cada curva conecta:
        |ell| = 1 -> 2 -> 3
    """

    validate_eta_columns(
        rows
    )

    eta_metrics = (
        (
            "retention",
            r"Retención $R_{\ell_0}$",
        ),
        (
            "spread",
            r"Anchura OAM $\sigma_{\Delta\ell}$",
        ),
        (
            "entropy",
            r"Entropía modal $H$",
        ),
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            12.0,
            4.0,
        ),
    )

    family_styles = {
        "LG": {
            "linestyle": "-",
            "marker": "o",
        },

        "BG": {
            "linestyle": "--",
            "marker": "s",
        },
    }

    for column, (
        metric,
        ylabel,
    ) in enumerate(
        eta_metrics
    ):
    
        axis = axes[column]
    
        # --------------------------------------------------------
        # Guardar datos LG y BG de este panel
        # --------------------------------------------------------
    
        family_data = {}
    
        # --------------------------------------------------------
        # Dibujar ambas familias
        # --------------------------------------------------------
    
        for family in FAMILIES:
    
            subset = [
                row
                for row in rows
                if (
                    row["family"] == family
                    and row["regime"] == regime
                    and row["psd"] == psd
                )
            ]
    
            subset = sorted(
                subset,
                key=lambda row:
                    int(
                        row["order"]
                    ),
            )
    
            if len(subset) != len(ORDERS):
    
                raise RuntimeError(
                    "Se esperaban tres órdenes para "
                    f"{family}, {psd}, {regime}; "
                    f"se encontraron {len(subset)}."
                )
    
            eta = np.asarray(
                [
                    float(
                        row["eta_total"]
                    )
                    for row in subset
                ]
            )
    
            values = np.asarray(
                [
                    float(
                        row[metric]
                    )
                    for row in subset
                ]
            )
    
            # Guardar para etiquetar después
            family_data[family] = {
                "eta": eta,
                "values": values,
            }
    
            style = family_styles[
                family
            ]
    
            axis.plot(
                eta,
                values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=6,
                label=family,
            )
    
        # --------------------------------------------------------
        # Etiquetar |ell| una sola vez
        # --------------------------------------------------------
    
        lg_eta = family_data["LG"]["eta"]
        bg_eta = family_data["BG"]["eta"]
    
        lg_values = family_data["LG"]["values"]
        bg_values = family_data["BG"]["values"]
    
        eta_midpoint = (
            lg_eta
            + bg_eta
        ) / 2.0
    
        value_midpoint = (
            lg_values
            + bg_values
        ) / 2.0
    
        for order, x, y in zip(
            ORDERS,
            eta_midpoint,
            value_midpoint,
        ):
    
            axis.annotate(
                rf"$|\ell|={order}$",
                (
                    x,
                    y,
                ),
                xytext=(
                    7,
                    7,
                ),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="bottom",
            )
    
        # --------------------------------------------------------
        # Formato
        # --------------------------------------------------------
    
        axis.set_xlabel(
            r"$\eta="
            r"\langle r_{\mathrm{rms}}\rangle_z/"
            r"r_{0,\mathrm{total}}$"
        )
    
        axis.set_ylabel(
            ylabel
        )
    
        axis.grid(
            alpha=0.25
        )
    
        axis.text(
            0.03,
            0.95,
            (
                "(a)",
                "(b)",
                "(c)",
            )[column],
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )

    handles, labels = (
        axes[
            0
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
            -0.03,
        ),
    )

    figure.suptitle(
        (
            f"{REGIME_LABELS[regime]}, "
            f"{PSD_LABELS[psd]}"
        ),
        y=1.02,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.08,
            1.0,
            1.0,
        )
    )

    save_figure(
        figure=figure,
        basename=(
            "eta_modal_representative_"
            f"{regime}_{psd}"
        ),
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "FIGURAS — DEPENDENCIA CON EL ORDEN OAM"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Leer resultados
    # ========================================================

    scenario_rows = read_csv(
        SCENARIO_FILE
    )

    beam_size_rows = read_csv(
        BEAM_SIZE_FILE
    )

    eta_rows = read_csv(
        ETA_FILE
    )

    # ========================================================
    # Figura 1
    # Métricas absolutas frente al orden
    # LG y BG combinados
    # ========================================================

    print()
    print(
        "Generando métricas absolutas LG--BG..."
    )

    plot_absolute_order_effect_combined(
        rows=scenario_rows,
    )

    # ========================================================
    # Figura 2
    # Evolución de r_rms(z)
    # LG y BG superpuestos
    # ========================================================

    print()
    print(
        "Generando evolución conjunta de r_rms(z)..."
    )

    plot_order_rms_evolution_combined(
        rows=beam_size_rows,
    )

    # ========================================================
    # Figura 3
    # Caso representativo:
    # eta frente a las métricas modales
    # ========================================================

    print()
    print(
        "Generando caso representativo eta--métricas..."
    )

    plot_eta_representative_case(
        rows=eta_rows,
        regime="moderate",
        psd="von_karman",
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Figuras generadas correctamente."
    )

    print(
        f"Directorio de salida: "
        f"{FIGURE_DIRECTORY}"
    )

    print()

    print(
        "Figuras principales:"
    )

    print(
        "  - order_absolute_combined"
    )

    print(
        "  - order_rms_evolution_combined"
    )

    print(
        "  - eta_modal_representative_"
        "moderate_von_karman"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()


if __name__ == "__main__":

    main()
