"""
Comparación de parametrizaciones BG con matching de segundo momento
==================================================================

Se comparan tres procedimientos para construir un haz BG^ell con
el mismo radio RMS inicial que LG_0^ell:

A) fixed_w0_adjust_kr
   - w0_BG fijo
   - se ajusta kr

B) fixed_product_adjust_kr
   - kr * w0_BG = constante
   - se ajusta kr y w0 queda determinado

C) fixed_kr_adjust_w0
   - kr fijo
   - se ajusta w0_BG

Para cada parametrización y |ell| = 1,...,20:

- se iguala r_rms_BG(0) con r_rms_LG(0);
- se propagan LG y BG en vacío hasta 1000 m;
- se calcula r_rms(z);
- se calcula la diferencia relativa BG--LG;
- se cuantifica la máxima diferencia durante el trayecto.

Los resultados de cada prueba se guardan en directorios distintos
para evitar sobrescrituras.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv


# ============================================================
# Configuración física
# ============================================================

WAVELENGTH = 632.8e-9

TOTAL_DISTANCE = 1000.0

NUMBER_OF_INTERVALS = 16

Z_VALUES = np.linspace(
    0.0,
    TOTAL_DISTANCE,
    NUMBER_OF_INTERVALS + 1,
)


# ============================================================
# Malla
# ============================================================

N_GRID = 512

L_WINDOW = 0.4

DX = (
    L_WINDOW
    / N_GRID
)

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
# Órdenes
# ============================================================

ORDERS = tuple(
    range(
        1,
        21,
    )
)


# ============================================================
# LG
# ============================================================

LG_W0 = 0.025


# ============================================================
# Parametrizaciones BG
# ============================================================

FIXED_BG_W0 = 0.040

FIXED_BG_KR = 150.0

BG_KR_W0_PRODUCT = 3.0


PARAMETERIZATIONS = (
    "fixed_w0_adjust_kr",
    "fixed_product_adjust_kr",
    "fixed_kr_adjust_w0",
)


PARAMETERIZATION_LABELS = {
    "fixed_w0_adjust_kr":
        r"$w_0^{BG}$ fijo; ajuste de $k_r$",

    "fixed_product_adjust_kr":
        r"$k_r w_0^{BG}=3$",

    "fixed_kr_adjust_w0":
        r"$k_r=150\,\mathrm{m}^{-1}$; ajuste de $w_0$",
}


# ============================================================
# Salidas
# ============================================================

OUTPUT_DIRECTORY = (
    Path("results")
    / "chapter_5"
    / "analysis"
    / "bg_parameterization_study"
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

    if power <= 0.0:

        raise RuntimeError(
            "La potencia del campo debe ser positiva."
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
    w0: float,
) -> np.ndarray:

    if kr <= 0.0:

        raise ValueError(
            "kr debe ser positivo."
        )

    if w0 <= 0.0:

        raise ValueError(
            "w0 debe ser positivo."
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


# ============================================================
# Segundo momento en el espacio espectral transversal
# ============================================================

def transverse_spectral_rms(
    field: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """
    Calcula la extensión RMS del espectro angular transversal.

    Returns
    -------
    k_perp_rms:
        Segundo momento radial RMS en el espacio (kx, ky)
        [rad/m].

    theta_rms:
        Ángulo RMS paraxial asociado [rad].
    """

    spectrum = np.fft.fft2(
        field
    )

    spectral_power = (
        np.abs(spectrum) ** 2
    )

    total_spectral_power = float(
        np.sum(
            spectral_power
        )
    )

    if total_spectral_power <= 0.0:

        raise RuntimeError(
            "La potencia espectral debe ser positiva."
        )

    k_perp_squared = (
        KX**2
        + KY**2
    )

    k_perp_rms = float(
        np.sqrt(
            np.sum(
                k_perp_squared
                * spectral_power
            )
            / total_spectral_power
        )
    )

    theta_rms = float(
        k_perp_rms
        / K
    )

    return (
        k_perp_rms,
        theta_rms,
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
# Matching A
# w0 fijo, ajustar kr
# ============================================================

def match_fixed_w0_adjust_kr(
    ell: int,
    target_radius: float,
) -> tuple[
    float,
    float,
    float,
]:

    w0 = (
        FIXED_BG_W0
    )

    def objective(
        kr: float,
    ) -> float:

        field = bg_field(
            ell=ell,
            kr=kr,
            w0=w0,
        )

        return (
            r_rms(field)
            - target_radius
        ) ** 2

    result = minimize_scalar(
        objective,
        bounds=(
            10.0,
            1500.0,
        ),
        method="bounded",
    )

    kr = float(
        result.x
    )

    field = bg_field(
        ell=ell,
        kr=kr,
        w0=w0,
    )

    return (
        kr,
        w0,
        r_rms(field),
    )


# ============================================================
# Matching B
# kr*w0 = constante
# ============================================================

def match_fixed_product_adjust_kr(
    ell: int,
    target_radius: float,
) -> tuple[
    float,
    float,
    float,
]:

    def objective(
        kr: float,
    ) -> float:

        w0 = (
            BG_KR_W0_PRODUCT
            / kr
        )

        field = bg_field(
            ell=ell,
            kr=kr,
            w0=w0,
        )

        return (
            r_rms(field)
            - target_radius
        ) ** 2

    result = minimize_scalar(
        objective,
        bounds=(
            10.0,
            1500.0,
        ),
        method="bounded",
    )

    kr = float(
        result.x
    )

    w0 = (
        BG_KR_W0_PRODUCT
        / kr
    )

    field = bg_field(
        ell=ell,
        kr=kr,
        w0=w0,
    )

    return (
        kr,
        w0,
        r_rms(field),
    )


# ============================================================
# Matching C
# kr fijo, ajustar w0
# ============================================================

def match_fixed_kr_adjust_w0(
    ell: int,
    target_radius: float,
) -> tuple[
    float,
    float,
    float,
]:

    kr = (
        FIXED_BG_KR
    )

    def objective(
        w0: float,
    ) -> float:

        field = bg_field(
            ell=ell,
            kr=kr,
            w0=w0,
        )

        return (
            r_rms(field)
            - target_radius
        ) ** 2

    result = minimize_scalar(
        objective,
        bounds=(
            0.005,
            0.20,
        ),
        method="bounded",
    )

    w0 = float(
        result.x
    )

    field = bg_field(
        ell=ell,
        kr=kr,
        w0=w0,
    )

    return (
        kr,
        w0,
        r_rms(field),
    )


# ============================================================
# Selector
# ============================================================

def match_bg(
    parameterization: str,
    ell: int,
    target_radius: float,
) -> tuple[
    float,
    float,
    float,
]:

    if parameterization == "fixed_w0_adjust_kr":

        return match_fixed_w0_adjust_kr(
            ell=ell,
            target_radius=target_radius,
        )

    if parameterization == "fixed_product_adjust_kr":

        return match_fixed_product_adjust_kr(
            ell=ell,
            target_radius=target_radius,
        )

    if parameterization == "fixed_kr_adjust_w0":

        return match_fixed_kr_adjust_w0(
            ell=ell,
            target_radius=target_radius,
        )

    raise ValueError(
        f"Parametrización desconocida: {parameterization}"
    )


# ============================================================
# Evaluación
# ============================================================

def evaluate_order(
    parameterization: str,
    ell: int,
) -> tuple[
    dict,
    list[dict],
]:

    lg0 = (
        lg_field(
            ell
        )
    )

    lg_initial_radius = (
        r_rms(
            lg0
        )
    )

    (
        kr,
        bg_w0,
        bg_initial_radius,
    ) = match_bg(
        parameterization=parameterization,
        ell=ell,
        target_radius=lg_initial_radius,
    )

    bg0 = bg_field(
        ell=ell,
        kr=kr,
        w0=bg_w0,
    )

        # ========================================================
    # Extensión espectral transversal inicial
    # ========================================================

    (
        lg_kperp_rms,
        lg_theta_rms,
    ) = transverse_spectral_rms(
        lg0
    )

    (
        bg_kperp_rms,
        bg_theta_rms,
    ) = transverse_spectral_rms(
        bg0
    )

    kperp_relative_difference = (
        100.0
        * (
            bg_kperp_rms
            - lg_kperp_rms
        )
        / lg_kperp_rms
    )

    theta_relative_difference = (
        100.0
        * (
            bg_theta_rms
            - lg_theta_rms
        )
        / lg_theta_rms
    )

    relative_differences = []

    lg_radii = []

    bg_radii = []

    evolution_records = []

    for z in Z_VALUES:

        lg_z = propagate(
            lg0,
            float(z),
        )

        bg_z = propagate(
            bg0,
            float(z),
        )

        lg_radius = (
            r_rms(
                lg_z
            )
        )

        bg_radius = (
            r_rms(
                bg_z
            )
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
                "parameterization":
                    parameterization,

                "ell":
                    ell,

                "z_m":
                    float(z),

                "kr_bg_m-1":
                    kr,

                "w0_bg_m":
                    bg_w0,

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

    initial_error = (
        100.0
        * (
            bg_initial_radius
            - lg_initial_radius
        )
        / lg_initial_radius
    )

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

    summary = {
        "parameterization":
            parameterization,

        "ell":
            ell,

        "kr_bg_m-1":
            kr,

        "w0_bg_m":
            bg_w0,

        "kr_w0_product":
            kr
            * bg_w0,

        "lg_initial_r_rms_m":
            lg_initial_radius,

        "bg_initial_r_rms_m":
            bg_initial_radius,

        "initial_matching_error_percent":
            initial_error,

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

        "lg_kperp_rms_rad_m":
            lg_kperp_rms,

        "bg_kperp_rms_rad_m":
            bg_kperp_rms,

        "kperp_relative_difference_percent":
            kperp_relative_difference,

        "lg_theta_rms_rad":
            lg_theta_rms,

        "bg_theta_rms_rad":
            bg_theta_rms,

        "theta_relative_difference_percent":
            theta_relative_difference,
            
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
        newline="",
        encoding="utf-8",
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
# Figura comparativa principal
# ============================================================

def plot_parameterization_comparison(
    summaries: list[dict],
) -> None:

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    for parameterization in PARAMETERIZATIONS:

        subset = [
            row
            for row in summaries
            if (
                row[
                    "parameterization"
                ]
                == parameterization
            )
        ]

        subset = sorted(
            subset,
            key=lambda row:
                row[
                    "ell"
                ],
        )

        ell = np.asarray(
            [
                row[
                    "ell"
                ]
                for row in subset
            ]
        )

        difference = np.asarray(
            [
                row[
                    "max_abs_relative_difference_percent"
                ]
                for row in subset
            ]
        )

        axis.plot(
            ell,
            difference,
            marker="o",
            linewidth=1.5,
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    axis.set_xlabel(
        r"Orden azimutal $|\ell|$"
    )

    axis.set_ylabel(
        r"Máxima diferencia relativa LG--BG "
        r"en $r_{\mathrm{rms}}$ [\%]"
    )

    axis.set_xticks(
        ORDERS
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    pdf_file = (
        FIGURE_DIRECTORY
        / "bg_parameterization_comparison.pdf"
    )

    png_file = (
        FIGURE_DIRECTORY
        / "bg_parameterization_comparison.png"
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


# ============================================================
# Figura de parámetros encontrados
# ============================================================

def plot_bg_parameters(
    summaries: list[dict],
) -> None:

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(
            8.0,
            7.0,
        ),
        sharex=True,
    )

    for parameterization in PARAMETERIZATIONS:

        subset = [
            row
            for row in summaries
            if (
                row[
                    "parameterization"
                ]
                == parameterization
            )
        ]

        subset = sorted(
            subset,
            key=lambda row:
                row[
                    "ell"
                ],
        )

        ell = np.asarray(
            [
                row[
                    "ell"
                ]
                for row in subset
            ]
        )

        kr = np.asarray(
            [
                row[
                    "kr_bg_m-1"
                ]
                for row in subset
            ]
        )

        w0 = (
            100.0
            * np.asarray(
                [
                    row[
                        "w0_bg_m"
                    ]
                    for row in subset
                ]
            )
        )

        axes[0].plot(
            ell,
            kr,
            marker="o",
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

        axes[1].plot(
            ell,
            w0,
            marker="o",
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    axes[0].set_ylabel(
        r"$k_r$ [$\mathrm{m}^{-1}$]"
    )

    axes[1].set_ylabel(
        r"$w_0^{BG}$ [cm]"
    )

    axes[1].set_xlabel(
        r"Orden azimutal $|\ell|$"
    )

    axes[1].set_xticks(
        ORDERS
    )

    for axis in axes:

        axis.grid(
            alpha=0.25
        )

    axes[0].legend(
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "bg_parameters_vs_order.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "bg_parameters_vs_order.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Figura resumen: parametrización espacial y espectral BG
# ============================================================

def plot_four_panel_summary(
    summaries: list[dict],
    evolution_records: list[dict],
    representative_order: int = 3,
) -> None:

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.0, 8.5),
    )

    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # ========================================================
    # (a) Máxima diferencia espacial vs orden
    # ========================================================

    for parameterization in PARAMETERIZATIONS:

        subset = sorted(
            [
                row
                for row in summaries
                if row["parameterization"]
                == parameterization
            ],
            key=lambda row: row["ell"],
        )

        ell = np.asarray(
            [
                row["ell"]
                for row in subset
            ]
        )

        difference = np.asarray(
            [
                row[
                    "max_abs_relative_difference_percent"
                ]
                for row in subset
            ]
        )

        ax_a.plot(
            ell,
            difference,
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    ax_a.set_xlabel(
        r"Orden azimutal $|\ell|$"
    )

    ax_a.set_ylabel(
        r"Máx. diferencia en $r_{\rm rms}$ [\%]"
    )

    ax_a.set_xticks(
        np.arange(1, 21, 2)
    )

    ax_a.grid(
        alpha=0.25
    )

    ax_a.legend(
        fontsize=8
    )

    ax_a.text(
        0.03,
        0.95,
        "(a)",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # ========================================================
    # (b) Diferencia espectral vs orden
    # ========================================================

    ax_b.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    for parameterization in PARAMETERIZATIONS:

        subset = sorted(
            [
                row
                for row in summaries
                if row["parameterization"]
                == parameterization
            ],
            key=lambda row: row["ell"],
        )

        ell = np.asarray(
            [
                row["ell"]
                for row in subset
            ]
        )

        difference = np.asarray(
            [
                row[
                    "kperp_relative_difference_percent"
                ]
                for row in subset
            ]
        )

        ax_b.plot(
            ell,
            difference,
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    ax_b.set_xlabel(
        r"Orden azimutal $|\ell|$"
    )

    ax_b.set_ylabel(
        r"Diferencia en $k_{\perp,\rm rms}$ [\%]"
    )

    ax_b.set_xticks(
        np.arange(1, 21, 2)
    )

    ax_b.grid(
        alpha=0.25
    )

    ax_b.text(
        0.03,
        0.95,
        "(b)",
        transform=ax_b.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # ========================================================
    # LG representativo
    # ========================================================

    lg_subset = sorted(
        [
            row
            for row in evolution_records
            if (
                row["ell"]
                == representative_order
                and row["parameterization"]
                == PARAMETERIZATIONS[0]
            )
        ],
        key=lambda row: row["z_m"],
    )

    z = np.asarray(
        [
            row["z_m"]
            for row in lg_subset
        ]
    )

    lg_radius = (
        1000.0
        * np.asarray(
            [
                row["lg_r_rms_m"]
                for row in lg_subset
            ]
        )
    )

    # ========================================================
    # (c) Evolución espacial
    # ========================================================

    ax_c.plot(
        z,
        lg_radius,
        linewidth=2.0,
        label=rf"LG, $|\ell|={representative_order}$",
    )

    for parameterization in PARAMETERIZATIONS:

        subset = sorted(
            [
                row
                for row in evolution_records
                if (
                    row["ell"]
                    == representative_order
                    and row["parameterization"]
                    == parameterization
                )
            ],
            key=lambda row: row["z_m"],
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

        ax_c.plot(
            z,
            bg_radius,
            linestyle="--",
            linewidth=1.5,
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    ax_c.set_xlabel(
        r"Distancia $z$ [m]"
    )

    ax_c.set_ylabel(
        r"$r_{\rm rms}(z)$ [mm]"
    )

    ax_c.grid(
        alpha=0.25
    )

    ax_c.legend(
        fontsize=7
    )

    ax_c.text(
        0.03,
        0.95,
        "(c)",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # ========================================================
    # (d) Diferencia relativa durante propagación
    # ========================================================

    ax_d.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    for parameterization in PARAMETERIZATIONS:

        subset = sorted(
            [
                row
                for row in evolution_records
                if (
                    row["ell"]
                    == representative_order
                    and row["parameterization"]
                    == parameterization
                )
            ],
            key=lambda row: row["z_m"],
        )

        z_local = np.asarray(
            [
                row["z_m"]
                for row in subset
            ]
        )

        relative_difference = np.asarray(
            [
                row[
                    "relative_difference_percent"
                ]
                for row in subset
            ]
        )

        ax_d.plot(
            z_local,
            relative_difference,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=PARAMETERIZATION_LABELS[
                parameterization
            ],
        )

    ax_d.set_xlabel(
        r"Distancia $z$ [m]"
    )

    ax_d.set_ylabel(
        r"Diferencia BG--LG en $r_{\rm rms}$ [\%]"
    )

    ax_d.grid(
        alpha=0.25
    )

    ax_d.legend(
        fontsize=7
    )

    ax_d.text(
        0.03,
        0.95,
        "(d)",
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # ========================================================
    # Guardado
    # ========================================================

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "bg_parameterization_four_panel_summary.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "bg_parameterization_four_panel_summary.png",
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

    for parameterization in PARAMETERIZATIONS:

        print()
        print(
            "=" * 120
        )

        print(
            PARAMETERIZATION_LABELS[
                parameterization
            ]
        )

        print(
            "=" * 120
        )

        print(
            f"{'ell':>4} "
            f"{'kr [1/m]':>12} "
            f"{'w0 BG [cm]':>12} "
            f"{'kr*w0':>10} "
            f"{'error inicial [%]':>18} "
            f"{'crec. LG [%]':>13} "
            f"{'crec. BG [%]':>13} "
            f"{'max |Δr| [%]':>14}"
        )

        print(
            "-" * 120
        )

        subset = [
            row
            for row in summaries
            if (
                row[
                    "parameterization"
                ]
                == parameterization
            )
        ]

        subset = sorted(
            subset,
            key=lambda row:
                row[
                    "ell"
                ],
        )

        for row in subset:

            print(
                f"{row['ell']:4d} "
                f"{row['kr_bg_m-1']:12.4f} "
                f"{100.0 * row['w0_bg_m']:12.4f} "
                f"{row['kr_w0_product']:10.4f} "
                f"{row['initial_matching_error_percent']:18.6f} "
                f"{row['lg_growth_percent']:13.4f} "
                f"{row['bg_growth_percent']:13.4f} "
                f"{row['max_abs_relative_difference_percent']:14.4f}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    summaries = []

    evolution_records = []

    for parameterization in PARAMETERIZATIONS:

        parameterization_directory = (
            OUTPUT_DIRECTORY
            / parameterization
        )

        parameterization_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_summaries = []

        local_evolution = []

        print()
        print(
            f"Ejecutando: {parameterization}"
        )

        for ell in ORDERS:

            print(
                f"  |ell|={ell}"
            )

            (
                summary,
                evolution,
            ) = evaluate_order(
                parameterization=parameterization,
                ell=ell,
            )

            summaries.append(
                summary
            )

            evolution_records.extend(
                evolution
            )

            local_summaries.append(
                summary
            )

            local_evolution.extend(
                evolution
            )

        write_csv(
            parameterization_directory
            / "summary.csv",
            local_summaries,
        )

        write_csv(
            parameterization_directory
            / "size_evolution.csv",
            local_evolution,
        )

    # --------------------------------------------------------
    # CSV global
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIRECTORY
        / "all_parameterizations_summary.csv",
        summaries,
    )

    write_csv(
        OUTPUT_DIRECTORY
        / "all_parameterizations_evolution.csv",
        evolution_records,
    )

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print_summary(
        summaries
    )

    # --------------------------------------------------------
    # Figuras
    # --------------------------------------------------------

    plot_parameterization_comparison(
        summaries
    )

    plot_bg_parameters(
        summaries
    )

    plot_four_panel_summary(
        summaries=summaries,
        evolution_records=evolution_records,
        representative_order=3,
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
