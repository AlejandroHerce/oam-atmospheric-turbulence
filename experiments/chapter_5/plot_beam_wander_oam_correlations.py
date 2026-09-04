"""
Resumen gráfico de la relación entre el desplazamiento instantáneo
del centroide y la degradación OAM.

Para cada uno de los 54 escenarios de producción se utiliza la
correlación de Spearman calculada a nivel de realización individual
entre r_c y:

    - retención del modo transmitido;
    - anchura RMS del espectro OAM;
    - entropía OAM normalizada.

Cada régimen contiene 18 escenarios.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "results/chapter_5/analysis/"
    "beam_wander_oam_correlations.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/chapter_5/analysis"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_FILE = (
    OUTPUT_DIRECTORY
    / "beam_wander_oam_correlations.png"
)

PDF_FILE = (
    OUTPUT_DIRECTORY
    / "beam_wander_oam_correlations.pdf"
)


# ============================================================
# Configuration
# ============================================================

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

METRICS = (
    "retention",
    "oam_rms_spread",
    "normalized_oam_entropy",
)

METRIC_LABELS = {
    "retention":
        r"Retención $R_{\ell_0}$",

    "oam_rms_spread":
        r"Anchura $\sigma_{\Delta\ell}$",

    "normalized_oam_entropy":
        r"Entropía $H$",
}

PANEL_LABELS = (
    "(a)",
    "(b)",
    "(c)",
)


# ============================================================
# Load data
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)

required_columns = {
    "regime",
    "centroid_variable",
    "oam_metric",
    "spearman_rho",
}

missing = (
    required_columns
    - set(data.columns)
)

if missing:
    raise RuntimeError(
        f"Faltan columnas: {sorted(missing)}"
    )


# Use only radial centroid displacement.

data = data[
    data[
        "centroid_variable"
    ] == "radius"
].copy()


# ============================================================
# Numerical check
# ============================================================

print()

print(
    "=" * 90
)

print(
    "CORRELACIONES UTILIZADAS EN LA FIGURA"
)

print(
    "=" * 90
)

summary = (
    data[
        data[
            "oam_metric"
        ].isin(
            METRICS
        )
    ]
    .groupby(
        [
            "regime",
            "oam_metric",
        ],
        sort=False,
    )[
        "spearman_rho"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
)

print(
    summary.to_string()
)


# ============================================================
# Figure
# ============================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(
        11.2,
        4.3,
    ),
    sharey=True,
)

rng = np.random.default_rng(
    20260903
)

positions = np.arange(
    1,
    len(REGIMES) + 1,
)


for axis, metric, panel_label in zip(
    axes,
    METRICS,
    PANEL_LABELS,
):

    distributions = []

    for regime in REGIMES:

        values = data.loc[
            (
                data[
                    "regime"
                ] == regime
            )
            &
            (
                data[
                    "oam_metric"
                ] == metric
            ),
            "spearman_rho",
        ].to_numpy(
            dtype=np.float64
        )

        if values.size != 18:
            raise RuntimeError(
                f"{metric}, {regime}: "
                f"se esperaban 18 escenarios y "
                f"se encontraron {values.size}."
            )

        distributions.append(
            values
        )

    # --------------------------------------------------------
    # Boxplots
    # --------------------------------------------------------

    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.52,
        showfliers=False,
    )

    # --------------------------------------------------------
    # Individual scenarios + mean
    # --------------------------------------------------------

    for position, values in zip(
        positions,
        distributions,
    ):

        jitter = rng.normal(
            loc=0.0,
            scale=0.035,
            size=values.size,
        )

        axis.scatter(
            position + jitter,
            values,
            s=18,
            alpha=0.60,
            zorder=3,
        )

        axis.scatter(
            position,
            np.mean(values),
            marker="D",
            s=48,
            color="black",
            zorder=4,
            label="Media" if position == positions[0] else None,
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=0.9,
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

    axis.set_title(
        METRIC_LABELS[
            metric
        ]
    )

    axis.grid(
        axis="y",
        alpha=0.22,
    )

    axis.text(
        0.04,
        0.96,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

axes[0].set_ylabel(
    r"Correlación de Spearman $\rho_s(r_{\mathrm{c}},X)$"
)


# ============================================================
# Save
# ============================================================

figure.tight_layout()

figure.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight",
)

figure.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close(
    figure
)


print()

print(
    "Figura guardada en:"
)

print(
    PNG_FILE.resolve()
)

print(
    PDF_FILE.resolve()
)
