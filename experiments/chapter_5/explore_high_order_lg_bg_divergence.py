"""
Exploración de la divergencia LG--BG para órdenes OAM altos
===========================================================

Objetivo
--------
Comparar la evolución transversal de haces LG_p=0 y BG con
|ell| = 1,...,20 bajo propagación libre.

Para cada orden:

1. Se genera LG_0^ell con w0 fijo.
2. Se genera BG^ell.
3. Se ajusta k_r del BG para igualar el radio RMS inicial:
       r_rms_BG(0) ~= r_rms_LG(0)
4. Se propagan ambos haces en vacío hasta 1000 m mediante ASM.
5. Se calcula:
       - r_rms(z)
       - diferencia relativa BG--LG
       - diferencia máxima durante el trayecto
       - crecimiento relativo del segundo momento

No se generan pantallas de fase.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv


# ============================================================
# Configuración física y numérica
# ============================================================

WAVELENGTH = 632.8e-9

N_GRID = 512
L_WINDOW = 0.4
DX = L_WINDOW / N_GRID

TOTAL_DISTANCE = 1000.0
NUMBER_OF_INTERVALS = 16

Z_VALUES = np.linspace(
    0.0,
    TOTAL_DISTANCE,
    NUMBER_OF_INTERVALS + 1,
)

ORDERS = tuple(
    range(
        1,
        21,
    )
)

LG_W0 = 0.025


# ============================================================
# Relación estructural de la familia BG
# ============================================================

BG_KR_W0_PRODUCT = 3.0


# ============================================================
# Rutas de salida
# ============================================================

OUTPUT_DIRECTORY = (
    Path("results")
    / "chapter_5"
    / "analysis"
    / "high_order_lg_bg"
)

FIGURE_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "figures"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Malla
# ============================================================

coordinates = (
    np.arange(
        N_GRID,
        dtype=np.float64,
    )
    - N_GRID / 2.0
) * DX

X, Y = np.meshgrid(
    coordinates,
    coordinates,
    indexing="xy",
)

R = np.hypot(
    X,
    Y,
)

PHI = np.arctan2(
    Y,
    X,
)


# ============================================================
# Normalización
# ============================================================

def normalize(
    field: np.ndarray,
) -> np.ndarray:

    power = float(
        np.sum(
            np.abs(field) ** 2
        )
        * DX
        * DX
    )

    return (
        field
        / np.sqrt(power)
    )


# ============================================================
# LG
# ============================================================

def lg_field(
    ell: int,
) -> np.ndarray:

    radial = (
        (
            np.sqrt(2.0)
            * R
            / LG_W0
        )
        ** abs(ell)
        * np.exp(
            -(
                R
                / LG_W0
            ) ** 2
        )
    )

    field = (
        radial
        * np.exp(
            1j
            * ell
            * PHI
        )
    )

    return normalize(
        field
    )


# ============================================================
# BG
# ============================================================
def bg_field(
    ell: int,
    kr: float,
) -> np.ndarray:
    """
    BG^ell manteniendo la relación estructural

        kr * w0 = 3

    observada en los haces BG empleados en las
    simulaciones finales.
    """

    if kr <= 0.0:

        raise ValueError(
            "kr debe ser positivo."
        )

    w0 = (
        BG_KR_W0_PRODUCT
        / kr
    )

    radial = (
        jv(
            ell,
            kr * R,
        )
        * np.exp(
            -(
                R
                / w0
            ) ** 2
        )
    )

    field = (
        radial
        * np.exp(
            1j
            * ell
            * PHI
        )
    )

    return normalize(
        field
    )

# ============================================================
# Segundo momento
# ============================================================

def r_rms(
    field: np.ndarray,
) -> float:

    intensity = (
        np.abs(field) ** 2
    )

    power = float(
        np.sum(
            intensity
        )
        * DX
        * DX
    )

    x_centroid = float(
        np.sum(
            X
            * intensity
        )
        * DX
        * DX
        / power
    )

    y_centroid = float(
        np.sum(
            Y
            * intensity
        )
        * DX
        * DX
        / power
    )

    radius_squared = (
        (
            X
            - x_centroid
        ) ** 2
        +
        (
            Y
            - y_centroid
        ) ** 2
    )

    second_moment = float(
        np.sum(
            radius_squared
            * intensity
        )
        * DX
        * DX
        / power
    )

    return float(
        np.sqrt(
            second_moment
        )
    )


# ============================================================
# ASM
# ============================================================

frequencies = np.fft.fftfreq(
    N_GRID,
    d=DX,
)

FX, FY = np.meshgrid(
    frequencies,
    frequencies,
    indexing="xy",
)

K = (
    2.0
    * np.pi
    / WAVELENGTH
)

KX = (
    2.0
    * np.pi
    * FX
)

KY = (
    2.0
    * np.pi
    * FY
)

KZ = np.sqrt(
    np.maximum(
        K**2
        - KX**2
        - KY**2,
        0.0,
    )
)


def propagate(
    field: np.ndarray,
    z: float,
) -> np.ndarray:

    if z == 0.0:

        return field.copy()

    transfer = np.exp(
        1j
        * KZ
        * z
    )

    return np.fft.ifft2(
        np.fft.fft2(
            field
        )
        * transfer
    )


# ============================================================
# Ajuste de k_r
# ============================================================
def match_bg_kr(
    ell: int,
    target_r_rms: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Ajusta kr imponiendo simultáneamente

        kr * w0_BG = 3

    y buscando

        r_rms_BG(0) = r_rms_LG(0).
    """

    def objective(
        kr: float,
    ) -> float:

        field = bg_field(
            ell=ell,
            kr=kr,
        )

        radius = r_rms(
            field
        )

        return (
            radius
            - target_r_rms
        ) ** 2

    result = minimize_scalar(
        objective,
        bounds=(
            10.0,
            1500.0,
        ),
        method="bounded",
        options={
            "xatol": 1e-8,
        },
    )

    if not result.success:

        raise RuntimeError(
            f"No se pudo ajustar BG para ell={ell}"
        )

    kr_opt = float(
        result.x
    )

    w0_opt = (
        BG_KR_W0_PRODUCT
        / kr_opt
    )

    field = bg_field(
        ell=ell,
        kr=kr_opt,
    )

    matched_radius = r_rms(
        field
    )

    return (
        kr_opt,
        w0_opt,
        matched_radius,
    )


# ============================================================
# Evaluar orden individual
# ============================================================

def evaluate_order(
    ell: int,
) -> tuple[
    dict,
    list[dict],
]:

    lg0 = lg_field(
        ell
    )

    lg_initial_radius = r_rms(
        lg0
    )

    (
        kr_opt,
        bg_w0,
        bg_initial_radius,
    ) = match_bg_kr(
        ell=ell,
        target_r_rms=lg_initial_radius,
    )
    
    bg0 = bg_field(
        ell=ell,
        kr=kr_opt,
    )

    evolution_records = []

    relative_differences = []

    lg_radii = []
    bg_radii = []

    for z in Z_VALUES:

        lg_z = propagate(
            lg0,
            float(z),
        )

        bg_z = propagate(
            bg0,
            float(z),
        )

        lg_radius = r_rms(
            lg_z
        )

        bg_radius = r_rms(
            bg_z
        )

        relative_difference = (
            100.0
            * (
                bg_radius
                - lg_radius
            )
            / lg_radius
        )

        lg_radii.append(
            lg_radius
        )

        bg_radii.append(
            bg_radius
        )

        relative_differences.append(
            relative_difference
        )

        evolution_records.append(
            {
                "ell":
                    ell,

                "z_m":
                    float(z),

                "kr_bg_m-1":
                    kr_opt,

                "lg_r_rms_m":
                    lg_radius,

                "bg_r_rms_m":
                    bg_radius,

                "relative_difference_percent":
                    relative_difference,
            }
        )

    lg_radii = np.asarray(
        lg_radii
    )

    bg_radii = np.asarray(
        bg_radii
    )

    relative_differences = np.asarray(
        relative_differences
    )

    # --------------------------------------------------------
    # Crecimiento relativo durante el trayecto
    # --------------------------------------------------------

    lg_growth = (
        100.0
        * (
            lg_radii[-1]
            - lg_radii[0]
        )
        / lg_radii[0]
    )

    bg_growth = (
        100.0
        * (
            bg_radii[-1]
            - bg_radii[0]
        )
        / bg_radii[0]
    )

    initial_matching_error = (
        100.0
        * (
            bg_initial_radius
            - lg_initial_radius
        )
        / lg_initial_radius
    )

    summary = {
        "ell":
            ell,

        "kr_bg_m-1":
            kr_opt,

        "bg_w0_m":
            bg_w0,

        "lg_initial_r_rms_m":
            lg_initial_radius,

        "bg_initial_r_rms_m":
            bg_initial_radius,

        "initial_matching_error_percent":
            initial_matching_error,

        "lg_final_r_rms_m":
            float(
                lg_radii[-1]
            ),

        "bg_final_r_rms_m":
            float(
                bg_radii[-1]
            ),

        "lg_growth_percent":
            float(
                lg_growth
            ),

        "bg_growth_percent":
            float(
                bg_growth
            ),

        "max_abs_relative_difference_percent":
            float(
                np.max(
                    np.abs(
                        relative_differences
                    )
                )
            ),

        "median_relative_difference_percent":
            float(
                np.median(
                    relative_differences
                )
            ),
    }

    return (
        summary,
        evolution_records,
    )


# ============================================================
# CSV
# ============================================================

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
# Figura 1
# ============================================================

def plot_max_difference(
    summaries: list[dict],
) -> None:

    ell = np.asarray(
        [
            row["ell"]
            for row in summaries
        ]
    )

    difference = np.asarray(
        [
            row[
                "max_abs_relative_difference_percent"
            ]
            for row in summaries
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            5.2,
        )
    )

    axis.plot(
        ell,
        difference,
        marker="o",
        linewidth=1.5,
    )

    axis.set_xlabel(
        r"Orden azimutal $|\ell|$"
    )

    axis.set_ylabel(
        r"Máxima diferencia relativa LG--BG "
        r"en $r_{\mathrm{rms}}$ [\%]"
    )

    axis.set_xticks(
        ell
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "max_size_difference_vs_order.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "max_size_difference_vs_order.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Figura 2
# ============================================================

def plot_representative_evolution(
    evolution_records: list[dict],
) -> None:

    representative_orders = (
        1,
        3,
        5,
        10,
        15,
        20,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(
            8.5,
            8.0,
        ),
        sharex=True,
    )

    # --------------------------------------------------------
    # r_rms(z)
    # --------------------------------------------------------

    axis = axes[0]

    for ell in representative_orders:

        subset = [
            row
            for row in evolution_records
            if (
                row["ell"]
                == ell
            )
        ]

        z = np.asarray(
            [
                row["z_m"]
                for row in subset
            ]
        )

        lg_radius = (
            1000.0
            * np.asarray(
                [
                    row["lg_r_rms_m"]
                    for row in subset
                ]
            )
        )

        bg_radius = (
            1000.0
            * np.asarray(
                [
                    row["bg_r_rms_m"]
                    for row in subset
                ]
            )
        )

        axis.plot(
            z,
            lg_radius,
            linewidth=1.2,
            label=(
                rf"LG, $|\ell|={ell}$"
            ),
        )

        axis.plot(
            z,
            bg_radius,
            linestyle="--",
            linewidth=1.2,
            label=(
                rf"BG, $|\ell|={ell}$"
            ),
        )

    axis.set_ylabel(
        r"$r_{\mathrm{rms}}(z)$ [mm]"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        ncol=2,
        fontsize=8,
    )

    axis.text(
        0.02,
        0.95,
        "(a)",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # --------------------------------------------------------
    # diferencia relativa
    # --------------------------------------------------------

    axis = axes[1]

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    for ell in representative_orders:

        subset = [
            row
            for row in evolution_records
            if (
                row["ell"]
                == ell
            )
        ]

        z = np.asarray(
            [
                row["z_m"]
                for row in subset
            ]
        )

        relative = np.asarray(
            [
                row[
                    "relative_difference_percent"
                ]
                for row in subset
            ]
        )

        axis.plot(
            z,
            relative,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=(
                rf"$|\ell|={ell}$"
            ),
        )

    axis.set_xlabel(
        r"Distancia de propagación $z$ [m]"
    )

    axis.set_ylabel(
        r"$100(r_{\rm rms}^{BG}-r_{\rm rms}^{LG})/"
        r"r_{\rm rms}^{LG}$ [\%]"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        ncol=3,
    )

    axis.text(
        0.02,
        0.95,
        "(b)",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "representative_size_evolution.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "representative_size_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Terminal
# ============================================================

def print_summary(
    summaries: list[dict],
) -> None:

    print()
    print(
        "=" * 115
    )

    print(
        "BARRIDO DE DIVERGENCIA LG--BG"
    )

    print(
        "=" * 115
    )

    print(
        f"{'ell':>4} "
        f"{'kr BG [1/m]':>13} "
        f"{'error inicial [%]':>18} "
        f"{'crec. LG [%]':>14} "
        f"{'crec. BG [%]':>14} "
        f"{'máx |Δr| [%]':>15}"
    )

    print(
        "-" * 115
    )

    for row in summaries:

        print(
            f"{row['ell']:4d} "
            f"{row['kr_bg_m-1']:13.4f} "
            f"{row['initial_matching_error_percent']:18.6f} "
            f"{row['lg_growth_percent']:14.4f} "
            f"{row['bg_growth_percent']:14.4f} "
            f"{row['max_abs_relative_difference_percent']:15.4f}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    summaries = []

    evolution_records = []

    for ell in ORDERS:

        print(
            f"Evaluando |ell|={ell}..."
        )

        (
            summary,
            evolution,
        ) = evaluate_order(
            ell
        )

        summaries.append(
            summary
        )

        evolution_records.extend(
            evolution
        )

    summary_file = (
        OUTPUT_DIRECTORY
        / "high_order_summary.csv"
    )

    evolution_file = (
        OUTPUT_DIRECTORY
        / "high_order_size_evolution.csv"
    )

    write_csv(
        summary_file,
        summaries,
    )

    write_csv(
        evolution_file,
        evolution_records,
    )

    print_summary(
        summaries
    )

    plot_max_difference(
        summaries
    )

    plot_representative_evolution(
        evolution_records
    )

    print()
    print(
        f"Resultados guardados en: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":

    main()
