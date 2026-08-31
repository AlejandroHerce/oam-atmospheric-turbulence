"""
Capítulo 5 — evolución transversal de los haces LG y BG
========================================================

Objetivo
--------
Comprobar si los pares LG--BG empleados en las simulaciones finales
presentan tamaños transversales comparables:

    1. en el plano inicial z = 0;
    2. durante la propagación libre hasta z = 1000 m.

Se calculan:

    - radio del máximo de intensidad r_peak en z = 0;
    - radio RMS:
          r_rms = sqrt(<r^2>);
    - diámetro de segundo momento D_4sigma;
    - diferencia relativa BG--LG en r_rms;
    - evolución r_rms(z) en vacío.

La propagación se realiza mediante el método del espectro angular (ASM)
sobre la misma malla transversal utilizada en las simulaciones finales.

No se generan pantallas de fase y no se modifica ningún resultado del
Capítulo 5.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuración numérica
# ============================================================

WAVELENGTH = 632.8e-9        # [m]

N_GRID = 512

L_WINDOW = 0.4              # [m]

DX = (
    L_WINDOW
    / N_GRID
)

TOTAL_PROPAGATION_DISTANCE = 1000.0  # [m]

NUMBER_OF_PHASE_SCREENS = 16

SCREEN_SPACING = (
    TOTAL_PROPAGATION_DISTANCE
    / NUMBER_OF_PHASE_SCREENS
)


# ============================================================
# Parámetros de los haces
# ============================================================

LG_W0 = 0.025  # [m]


BG_PARAMETERS = {
    "BG01": {
        "order": 1,
        "w0": 0.0392,
        "kr": 76.46,
    },

    "BG02": {
        "order": 2,
        "w0": 0.0352,
        "kr": 85.34,
    },

    "BG03": {
        "order": 3,
        "w0": 0.0326,
        "kr": 91.91,
    },
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
# Salidas
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


# ============================================================
# Malla cartesiana
# ============================================================

def create_cartesian_grid() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construye la misma malla cartesiana centrada utilizada
    en las simulaciones.
    """

    coordinates = (
        np.arange(
            N_GRID,
            dtype=np.float64,
        )
        - N_GRID / 2.0
    ) * DX

    x, y = np.meshgrid(
        coordinates,
        coordinates,
        indexing="xy",
    )

    r = np.hypot(
        x,
        y,
    )

    phi = np.arctan2(
        y,
        x,
    )

    return (
        x,
        y,
        r,
        phi,
    )


# ============================================================
# Normalización
# ============================================================

def normalize_field(
    field: np.ndarray,
) -> np.ndarray:

    power = float(
        np.sum(
            np.abs(
                field
            ) ** 2
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
        / np.sqrt(
            power
        )
    )


# ============================================================
# Haces LG
# ============================================================

def laguerre_gaussian_p0(
    order: int,
    r: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    LG_p^ell con p = 0 en el plano de cintura.

    U(r,phi) ~
        (sqrt(2) r / w0)^|ell|
        exp(-r^2 / w0^2)
        exp(i ell phi)
    """

    radial = (
        (
            np.sqrt(2.0)
            * r
            / LG_W0
        )
        ** abs(
            order
        )
        * np.exp(
            -(
                r
                / LG_W0
            ) ** 2
        )
    )

    field = (
        radial
        * np.exp(
            1j
            * order
            * phi
        )
    )

    return normalize_field(
        field
    )


# ============================================================
# Haces BG
# ============================================================

def bessel_gaussian(
    order: int,
    w0: float,
    kr: float,
    r: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    BG^m en el plano inicial.

    U(r,phi) ~
        J_m(kr r)
        exp(-r^2 / w0^2)
        exp(i m phi)
    """

    # Importación local para mantener explícita la dependencia.
    from scipy.special import jv

    radial = (
        jv(
            order,
            kr * r,
        )
        * np.exp(
            -(
                r
                / w0
            ) ** 2
        )
    )

    field = (
        radial
        * np.exp(
            1j
            * order
            * phi
        )
    )

    return normalize_field(
        field
    )


# ============================================================
# Generador unificado
# ============================================================

def generate_beam(
    beam: str,
    r: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:

    if beam.startswith(
        "LG"
    ):

        order = int(
            beam[-1]
        )

        return laguerre_gaussian_p0(
            order=order,
            r=r,
            phi=phi,
        )

    if beam.startswith(
        "BG"
    ):

        parameters = (
            BG_PARAMETERS[
                beam
            ]
        )

        return bessel_gaussian(
            order=parameters[
                "order"
            ],
            w0=parameters[
                "w0"
            ],
            kr=parameters[
                "kr"
            ],
            r=r,
            phi=phi,
        )

    raise ValueError(
        f"Haz desconocido: {beam}"
    )


# ============================================================
# ASM
# ============================================================

def angular_spectrum_propagation(
    field: np.ndarray,
    propagation_distance: float,
) -> np.ndarray:
    """
    Propagación en vacío mediante método del espectro angular.
    """

    if propagation_distance == 0.0:

        return field.copy()

    frequencies = np.fft.fftfreq(
        N_GRID,
        d=DX,
    )

    fx, fy = np.meshgrid(
        frequencies,
        frequencies,
        indexing="xy",
    )

    k = (
        2.0
        * np.pi
        / WAVELENGTH
    )

    kx = (
        2.0
        * np.pi
        * fx
    )

    ky = (
        2.0
        * np.pi
        * fy
    )

    kz_squared = (
        k**2
        - kx**2
        - ky**2
    )

    # En nuestra discretización todos los componentes relevantes
    # son propagantes. El máximo evita pequeños valores negativos
    # debidos a precisión numérica.
    kz = np.sqrt(
        np.maximum(
            kz_squared,
            0.0,
        )
    )

    transfer_function = np.exp(
        1j
        * kz
        * propagation_distance
    )

    spectrum = np.fft.fft2(
        field
    )

    propagated = np.fft.ifft2(
        spectrum
        * transfer_function
    )

    return propagated


# ============================================================
# Segundo momento
# ============================================================

def calculate_second_moment_metrics(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Calcula centroide, segundo momento radial, r_rms
    y diámetro D_4sigma.
    """

    intensity = (
        np.abs(
            field
        ) ** 2
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
            x
            * intensity
        )
        * DX
        * DX
        / power
    )

    y_centroid = float(
        np.sum(
            y
            * intensity
        )
        * DX
        * DX
        / power
    )

    x_centered = (
        x
        - x_centroid
    )

    y_centered = (
        y
        - y_centroid
    )

    r_squared_centered = (
        x_centered**2
        + y_centered**2
    )

    mean_r_squared = float(
        np.sum(
            r_squared_centered
            * intensity
        )
        * DX
        * DX
        / power
    )

    r_rms = float(
        np.sqrt(
            mean_r_squared
        )
    )

    # Para simetría circular:
    #
    # <x^2> = <r^2>/2
    #
    # D_4sigma = 4 sqrt(<x^2>)
    #           = 2 sqrt(2) r_rms
    d_4sigma = float(
        2.0
        * np.sqrt(2.0)
        * r_rms
    )

    return {
        "power":
            power,

        "centroid_x_m":
            x_centroid,

        "centroid_y_m":
            y_centroid,

        "mean_r_squared_m2":
            mean_r_squared,

        "r_rms_m":
            r_rms,

        "d_4sigma_m":
            d_4sigma,
    }


# ============================================================
# Perfil radial promedio
# ============================================================

def radial_average(
    intensity: np.ndarray,
    r: np.ndarray,
    number_of_bins: int = 2000,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Promedio azimutal discreto de la intensidad para estimar
    de forma robusta el radio del máximo.
    """

    r_max = (
        L_WINDOW
        / 2.0
    )

    edges = np.linspace(
        0.0,
        r_max,
        number_of_bins + 1,
    )

    bin_index = np.digitize(
        r.ravel(),
        edges,
    ) - 1

    valid = (
        (bin_index >= 0)
        & (
            bin_index
            < number_of_bins
        )
    )

    sums = np.bincount(
        bin_index[
            valid
        ],
        weights=intensity.ravel()[
            valid
        ],
        minlength=number_of_bins,
    )

    counts = np.bincount(
        bin_index[
            valid
        ],
        minlength=number_of_bins,
    )

    profile = np.divide(
        sums,
        counts,
        out=np.zeros_like(
            sums,
            dtype=np.float64,
        ),
        where=(
            counts > 0
        ),
    )

    radii = (
        0.5
        * (
            edges[:-1]
            + edges[1:]
        )
    )

    return (
        radii,
        profile,
    )


def calculate_peak_radius(
    field: np.ndarray,
    r: np.ndarray,
) -> float:

    intensity = (
        np.abs(
            field
        ) ** 2
    )

    radii, profile = radial_average(
        intensity=intensity,
        r=r,
    )

    index = int(
        np.argmax(
            profile
        )
    )

    return float(
        radii[
            index
        ]
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
# Evaluación de todos los haces
# ============================================================

def evaluate_beams() -> tuple[
    list[dict],
    list[dict],
]:
    """
    Devuelve:

        initial_records
        propagation_records
    """

    (
        x,
        y,
        r,
        phi,
    ) = create_cartesian_grid()

    propagation_distances = np.linspace(
        0.0,
        TOTAL_PROPAGATION_DISTANCE,
        NUMBER_OF_PHASE_SCREENS + 1,
    )

    initial_records = []

    propagation_records = []

    for beam in BEAMS:

        print()
        print(
            f"Analizando {beam}..."
        )

        initial_field = (
            generate_beam(
                beam=beam,
                r=r,
                phi=phi,
            )
        )

        initial_metrics = (
            calculate_second_moment_metrics(
                field=initial_field,
                x=x,
                y=y,
            )
        )

        peak_radius = (
            calculate_peak_radius(
                field=initial_field,
                r=r,
            )
        )

        if beam.startswith(
            "LG"
        ):

            w0 = (
                LG_W0
            )

            kr = np.nan

        else:

            w0 = (
                BG_PARAMETERS[
                    beam
                ][
                    "w0"
                ]
            )

            kr = (
                BG_PARAMETERS[
                    beam
                ][
                    "kr"
                ]
            )

        initial_records.append(
            {
                "beam":
                    beam,

                "family":
                    beam[:2],

                "order":
                    int(
                        beam[-1]
                    ),

                "w0_m":
                    w0,

                "kr_m-1":
                    kr,

                "r_peak_m":
                    peak_radius,

                "r_rms_m":
                    initial_metrics[
                        "r_rms_m"
                    ],

                "d_4sigma_m":
                    initial_metrics[
                        "d_4sigma_m"
                    ],

                "total_power":
                    initial_metrics[
                        "power"
                    ],
            }
        )

        # ----------------------------------------------------
        # Propagación directa desde z=0 a cada posición.
        # ----------------------------------------------------

        for z in propagation_distances:

            propagated_field = (
                angular_spectrum_propagation(
                    field=initial_field,
                    propagation_distance=float(
                        z
                    ),
                )
            )

            metrics = (
                calculate_second_moment_metrics(
                    field=propagated_field,
                    x=x,
                    y=y,
                )
            )

            propagation_records.append(
                {
                    "beam":
                        beam,

                    "family":
                        beam[:2],

                    "order":
                        int(
                            beam[-1]
                        ),

                    "z_m":
                        float(
                            z
                        ),

                    "r_rms_m":
                        metrics[
                            "r_rms_m"
                        ],

                    "d_4sigma_m":
                        metrics[
                            "d_4sigma_m"
                        ],

                    "centroid_x_m":
                        metrics[
                            "centroid_x_m"
                        ],

                    "centroid_y_m":
                        metrics[
                            "centroid_y_m"
                        ],

                    "total_power":
                        metrics[
                            "power"
                        ],
                }
            )

    return (
        initial_records,
        propagation_records,
    )


# ============================================================
# Comparación LG--BG
# ============================================================

def calculate_pairwise_differences(
    propagation_records: list[dict],
) -> list[dict]:

    records = []

    for order in ORDERS:

        lg_name = (
            f"LG0{order}"
        )

        bg_name = (
            f"BG0{order}"
        )

        lg_rows = [
            row
            for row in propagation_records
            if (
                row["beam"]
                == lg_name
            )
        ]

        bg_rows = [
            row
            for row in propagation_records
            if (
                row["beam"]
                == bg_name
            )
        ]

        lg_rows = sorted(
            lg_rows,
            key=lambda row:
                row["z_m"],
        )

        bg_rows = sorted(
            bg_rows,
            key=lambda row:
                row["z_m"],
        )

        if len(
            lg_rows
        ) != len(
            bg_rows
        ):

            raise RuntimeError(
                "Número de posiciones z incompatible."
            )

        for lg_row, bg_row in zip(
            lg_rows,
            bg_rows,
        ):

            if not np.isclose(
                lg_row["z_m"],
                bg_row["z_m"],
            ):

                raise RuntimeError(
                    "Las posiciones longitudinales no coinciden."
                )

            lg_size = (
                lg_row[
                    "r_rms_m"
                ]
            )

            bg_size = (
                bg_row[
                    "r_rms_m"
                ]
            )

            relative_difference = (
                100.0
                * (
                    bg_size
                    - lg_size
                )
                / lg_size
            )

            records.append(
                {
                    "order":
                        order,

                    "z_m":
                        lg_row[
                            "z_m"
                        ],

                    "lg_r_rms_m":
                        lg_size,

                    "bg_r_rms_m":
                        bg_size,

                    "difference_bg_minus_lg_m":
                        (
                            bg_size
                            - lg_size
                        ),

                    "relative_difference_percent":
                        relative_difference,
                }
            )

    return records


# ============================================================
# Tabla inicial
# ============================================================

def print_initial_table(
    records: list[dict],
) -> None:

    print()
    print(
        "=" * 105
    )

    print(
        "TAMAÑO TRANSVERSAL EN z = 0"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Haz':>6} "
        f"{'w0 [mm]':>10} "
        f"{'kr [1/m]':>11} "
        f"{'r_peak [mm]':>14} "
        f"{'r_rms [mm]':>13} "
        f"{'D4sigma [mm]':>15}"
    )

    print(
        "-" * 105
    )

    for row in records:

        kr = (
            row[
                "kr_m-1"
            ]
        )

        if np.isnan(
            kr
        ):

            kr_text = (
                "---"
            )

        else:

            kr_text = (
                f"{kr:.2f}"
            )

        print(
            f"{row['beam']:>6} "
            f"{1000.0 * row['w0_m']:10.3f} "
            f"{kr_text:>11} "
            f"{1000.0 * row['r_peak_m']:14.3f} "
            f"{1000.0 * row['r_rms_m']:13.3f} "
            f"{1000.0 * row['d_4sigma_m']:15.3f}"
        )


# ============================================================
# Resumen de matching inicial
# ============================================================

def print_initial_pair_matching(
    initial_records: list[dict],
) -> None:

    lookup = {
        row["beam"]:
            row
        for row in initial_records
    }

    print()
    print(
        "=" * 95
    )

    print(
        "DIFERENCIA INICIAL BG--LG EN r_rms"
    )

    print(
        "=" * 95
    )

    for order in ORDERS:

        lg = lookup[
            f"LG0{order}"
        ]

        bg = lookup[
            f"BG0{order}"
        ]

        difference = (
            bg[
                "r_rms_m"
            ]
            - lg[
                "r_rms_m"
            ]
        )

        relative = (
            100.0
            * difference
            / lg[
                "r_rms_m"
            ]
        )

        print(
            f"|ell|={order}: "
            f"LG={1000.0 * lg['r_rms_m']:.4f} mm, "
            f"BG={1000.0 * bg['r_rms_m']:.4f} mm, "
            f"BG-LG={1000.0 * difference:+.4f} mm "
            f"({relative:+.4f} %)"
        )


# ============================================================
# Figura
# ============================================================

def plot_evolution(
    propagation_records: list[dict],
    pairwise_records: list[dict],
) -> None:

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
    # Panel (a): r_rms(z)
    # --------------------------------------------------------

    axis = axes[0]

    for beam in BEAMS:

        rows = [
            row
            for row in propagation_records
            if (
                row[
                    "beam"
                ]
                == beam
            )
        ]

        rows = sorted(
            rows,
            key=lambda row:
                row[
                    "z_m"
                ],
        )

        z = np.asarray(
            [
                row[
                    "z_m"
                ]
                for row in rows
            ]
        )

        radius = (
            1000.0
            * np.asarray(
                [
                    row[
                        "r_rms_m"
                    ]
                    for row in rows
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

        axis.plot(
            z,
            radius,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=BEAM_LABELS[
                beam
            ],
        )

    axis.set_ylabel(
        r"Radio RMS $r_{\mathrm{rms}}$ [mm]"
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
        "(a)",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Panel (b): diferencia relativa
    # --------------------------------------------------------

    axis = axes[1]

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    for order in ORDERS:

        rows = [
            row
            for row in pairwise_records
            if (
                row[
                    "order"
                ]
                == order
            )
        ]

        rows = sorted(
            rows,
            key=lambda row:
                row[
                    "z_m"
                ],
        )

        z = np.asarray(
            [
                row[
                    "z_m"
                ]
                for row in rows
            ]
        )

        relative = np.asarray(
            [
                row[
                    "relative_difference_percent"
                ]
                for row in rows
            ]
        )

        axis.plot(
            z,
            relative,
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=(
                rf"$|\ell_0|={order}$"
            ),
        )

    axis.set_xlabel(
        r"Distancia de propagación $z$ [m]"
    )

    axis.set_ylabel(
        r"Diferencia relativa "
        "\n"
        r"$100(r_{\rm rms}^{BG}-r_{\rm rms}^{LG})/"
        r"r_{\rm rms}^{LG}$ [\%]"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

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

    pdf_filename = (
        FIGURE_DIRECTORY
        / "beam_size_evolution_lg_bg.pdf"
    )

    png_filename = (
        FIGURE_DIRECTORY
        / "beam_size_evolution_lg_bg.png"
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

    print()
    print(
        f"Figura guardada: "
        f"{pdf_filename}"
    )


# ============================================================
# Resumen de diferencias durante todo el trayecto
# ============================================================

def print_pairwise_evolution_summary(
    records: list[dict],
) -> None:

    print()
    print(
        "=" * 105
    )

    print(
        "DIFERENCIA RELATIVA BG--LG DURANTE LA PROPAGACIÓN"
    )

    print(
        "=" * 105
    )

    for order in ORDERS:

        values = np.asarray(
            [
                row[
                    "relative_difference_percent"
                ]
                for row in records
                if (
                    row[
                        "order"
                    ]
                    == order
                )
            ],
            dtype=np.float64,
        )

        print(
            f"|ell|={order}: "
            f"mín={np.min(values):+.4f} %, "
            f"mediana={np.median(values):+.4f} %, "
            f"máx={np.max(values):+.4f} %, "
            f"máx |Δ|={np.max(np.abs(values)):.4f} %"
        )


# ============================================================
# Constante de órdenes
# ============================================================

ORDERS = (
    1,
    2,
    3,
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Análisis del tamaño transversal LG--BG"
    )

    print(
        "=" * 70
    )

    print(
        f"Malla: {N_GRID} x {N_GRID}"
    )

    print(
        f"Ventana: {L_WINDOW:.3f} m"
    )

    print(
        f"dx: {DX * 1e3:.5f} mm"
    )

    print(
        f"Distancia: "
        f"{TOTAL_PROPAGATION_DISTANCE:.1f} m"
    )

    print(
        f"Paso longitudinal de diagnóstico: "
        f"{SCREEN_SPACING:.1f} m"
    )

    (
        initial_records,
        propagation_records,
    ) = evaluate_beams()

    pairwise_records = (
        calculate_pairwise_differences(
            propagation_records
        )
    )

    # --------------------------------------------------------
    # Guardar datos
    # --------------------------------------------------------

    initial_filename = (
        ANALYSIS_DIRECTORY
        / "beam_initial_sizes.csv"
    )

    propagation_filename = (
        ANALYSIS_DIRECTORY
        / "beam_size_evolution.csv"
    )

    pairwise_filename = (
        ANALYSIS_DIRECTORY
        / "beam_size_lg_bg_differences.csv"
    )

    write_csv(
        initial_filename,
        initial_records,
    )

    write_csv(
        propagation_filename,
        propagation_records,
    )

    write_csv(
        pairwise_filename,
        pairwise_records,
    )

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print_initial_table(
        initial_records
    )

    print_initial_pair_matching(
        initial_records
    )

    print_pairwise_evolution_summary(
        pairwise_records
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    plot_evolution(
        propagation_records=(
            propagation_records
        ),
        pairwise_records=(
            pairwise_records
        ),
    )

    print()
    print(
        "Archivos guardados:"
    )

    print(
        initial_filename
    )

    print(
        propagation_filename
    )

    print(
        pairwise_filename
    )


if __name__ == "__main__":

    main()
