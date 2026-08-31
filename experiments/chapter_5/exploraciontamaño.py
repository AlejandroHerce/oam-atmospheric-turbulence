"""
Exploración de la relación entre tamaño transversal del haz,
parámetro de Fried y degradación modal OAM.

Se estudian los parámetros adimensionales

    eta_total  = <r_rms(z)> / r0_total

y

    eta_screen = <r_rms(z)> / r0_screen,

junto con las métricas

    - retención del modo transmitido
    - anchura espectral OAM
    - entropía modal normalizada

El objetivo es determinar si la dependencia observada con el
orden azimutal puede relacionarse con la escala transversal
efectiva del haz respecto a las escalas de turbulencia.

No requiere pandas.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuración de directorios
# ============================================================

RESULTS_DIRECTORY = Path(
    "results"
)

CHAPTER_5_DIRECTORY = (
    RESULTS_DIRECTORY
    / "chapter_5"
)

ANALYSIS_RESULTS_DIRECTORY = (
    CHAPTER_5_DIRECTORY
    / "analysis"
)

SCENARIO_SUMMARY_FILE = (
    ANALYSIS_RESULTS_DIRECTORY
    / "scenario_summary.csv"
)

ANALYSIS_DIRECTORY = (
    ANALYSIS_RESULTS_DIRECTORY
    / "eta_analysis"
)

ANALYSIS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

NUMBER_OF_PHASE_SCREENS = 16

TOTAL_DISTANCE = 1000.0  # [m]


# ============================================================
# Fried total
# ============================================================

WEAK_R0_TOTAL = 0.1066
MODERATE_R0_TOTAL = 0.0268
STRONG_R0_TOTAL = 0.0067


R0_TOTAL = {
    "weak": WEAK_R0_TOTAL,
    "moderate": MODERATE_R0_TOTAL,
    "strong": STRONG_R0_TOTAL,
}


# ============================================================
# Fried por pantalla
# ============================================================

def segment_fried_parameter(
    total_r0: float,
    number_of_screens: int,
) -> float:

    return (
        total_r0
        * number_of_screens ** (3.0 / 5.0)
    )


R0_SCREEN = {
    regime: segment_fried_parameter(
        total_r0=r0,
        number_of_screens=NUMBER_OF_PHASE_SCREENS,
    )
    for regime, r0 in R0_TOTAL.items()
}


# ============================================================
# Parámetros de los haces
# ============================================================

BEAMS = (
    "LG01",
    "LG02",
    "LG03",
    "BG01",
    "BG02",
    "BG03",
)

BEAM_ORDER = {
    "LG01": 1,
    "LG02": 2,
    "LG03": 3,
    "BG01": 1,
    "BG02": 2,
    "BG03": 3,
}

BEAM_FAMILY = {
    "LG01": "LG",
    "LG02": "LG",
    "LG03": "LG",
    "BG01": "BG",
    "BG02": "BG",
    "BG03": "BG",
}

BEAM_LABELS = {
    "LG01": r"$\mathrm{LG}_0^1$",
    "LG02": r"$\mathrm{LG}_0^2$",
    "LG03": r"$\mathrm{LG}_0^3$",
    "BG01": r"$\mathrm{BG}^{1}$",
    "BG02": r"$\mathrm{BG}^{2}$",
    "BG03": r"$\mathrm{BG}^{3}$",
}


# ============================================================
# Tamaño transversal
# ============================================================

# Valores iniciales obtenidos directamente del cálculo numérico.

R_RMS_INITIAL = {
    "LG01": 0.0250000,
    "LG02": 0.0306186,
    "LG03": 0.0353553,
    "BG01": 0.0249883,
    "BG02": 0.0306383,
    "BG03": 0.0353326,
}


# ============================================================
# Parámetros gaussianos / BG
# ============================================================

WAVELENGTH = 632.8e-9

W0 = {
    "LG01": 0.0250,
    "LG02": 0.0250,
    "LG03": 0.0250,
    "BG01": 0.0392,
    "BG02": 0.0352,
    "BG03": 0.0326,
}

KR = {
    "BG01": 76.46,
    "BG02": 85.34,
    "BG03": 91.91,
}


# ============================================================
# Leer scenario_summary.csv
# ============================================================

def read_scenario_summary(
    filename: Path,
) -> list[dict]:

    rows = []

    with filename.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            rows.append(
                row
            )

    return rows


# ============================================================
# Modelo de r_rms(z)
# ============================================================

def lg_r_rms(
    beam: str,
    z: np.ndarray,
) -> np.ndarray:
    """
    Segundo momento radial para LG_p=0.

    r_rms(z) = w(z) sqrt((|l|+1)/2)
    """

    ell = BEAM_ORDER[
        beam
    ]

    waist = W0[
        beam
    ]

    rayleigh = (
        np.pi
        * waist**2
        / WAVELENGTH
    )

    wz = (
        waist
        * np.sqrt(
            1.0
            + (
                z
                / rayleigh
            )**2
        )
    )

    return (
        wz
        * np.sqrt(
            (
                ell
                + 1.0
            )
            / 2.0
        )
    )


# ============================================================
# Evolución BG
# ============================================================

def bg_r_rms_approximation(
    beam: str,
    z: np.ndarray,
) -> np.ndarray:
    """
    Para este análisis utilizamos la evolución observada
    numéricamente en vacío.

    Dado que LG y BG emparejados presentan diferencias
    inferiores al 0.4 %, se escala la evolución LG del mismo
    orden por la razón inicial de los segundos momentos.

    Esto evita introducir una expresión analítica BG distinta
    de la utilizada en la simulación.
    """

    order = BEAM_ORDER[
        beam
    ]

    lg_beam = (
        f"LG0{order}"
    )

    lg_curve = lg_r_rms(
        beam=lg_beam,
        z=z,
    )

    scale = (
        R_RMS_INITIAL[
            beam
        ]
        / R_RMS_INITIAL[
            lg_beam
        ]
    )

    return (
        lg_curve
        * scale
    )


def calculate_r_rms(
    beam: str,
    z: np.ndarray,
) -> np.ndarray:

    if BEAM_FAMILY[
        beam
    ] == "LG":

        return lg_r_rms(
            beam=beam,
            z=z,
        )

    return bg_r_rms_approximation(
        beam=beam,
        z=z,
    )


# ============================================================
# Tamaños efectivos
# ============================================================

def calculate_effective_sizes() -> dict:

    z = np.linspace(
        0.0,
        TOTAL_DISTANCE,
        2001,
    )

    sizes = {}

    for beam in BEAMS:

        curve = calculate_r_rms(
            beam=beam,
            z=z,
        )

        mean_path = float(
            np.trapezoid(
                curve,
                z,
            )
            / TOTAL_DISTANCE
        )

        sizes[
            beam
        ] = {
            "initial":
                float(
                    curve[0]
                ),

            "final":
                float(
                    curve[-1]
                ),

            "mean_path":
                mean_path,
        }

    return sizes


# ============================================================
# Construir dataset de eta
# ============================================================

def build_eta_dataset(
    rows: list[dict],
    sizes: dict,
) -> list[dict]:

    records = []

    for row in rows:

        beam = row[
            "beam"
        ]

        regime = row[
            "regime"
        ]

        psd = row[
            "psd"
        ]

        r_effective = (
            sizes[
                beam
            ][
                "mean_path"
            ]
        )

        eta_total = (
            r_effective
            / R0_TOTAL[
                regime
            ]
        )

        eta_screen = (
            r_effective
            / R0_SCREEN[
                regime
            ]
        )

        records.append(
            {
                "beam":
                    beam,

                "family":
                    BEAM_FAMILY[
                        beam
                    ],

                "order":
                    BEAM_ORDER[
                        beam
                    ],

                "psd":
                    psd,

                "regime":
                    regime,

                "r_rms_effective":
                    r_effective,

                "r0_total":
                    R0_TOTAL[
                        regime
                    ],

                "r0_screen":
                    R0_SCREEN[
                        regime
                    ],

                "eta_total":
                    eta_total,

                "eta_screen":
                    eta_screen,

                "retention":
                    float(
                        row[
                            "retention_mean"
                        ]
                    ),

                "spread":
                    float(
                        row[
                            "spread_mean"
                        ]
                    ),

                "entropy":
                    float(
                        row[
                            "entropy_mean"
                        ]
                    ),
            }
        )

    return records


# ============================================================
# Spearman sin scipy
# ============================================================

def rankdata(
    values: np.ndarray,
) -> np.ndarray:

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    sorted_values = (
        values[
            order
        ]
    )

    start = 0

    while start < len(values):

        end = (
            start
            + 1
        )

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):

            end += 1

        mean_rank = (
            0.5
            * (
                start
                + end
                - 1
            )
            + 1.0
        )

        ranks[
            order[
                start:end
            ]
        ] = mean_rank

        start = end

    return ranks


def spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    rx = rankdata(
        x
    )

    ry = rankdata(
        y
    )

    return float(
        np.corrcoef(
            rx,
            ry,
        )[
            0,
            1,
        ]
    )


# ============================================================
# Correlaciones globales
# ============================================================

def calculate_correlations(
    records: list[dict],
) -> None:

    print()
    print(
        "=" * 90
    )

    print(
        "CORRELACIÓN GLOBAL CON ETA"
    )

    print(
        "=" * 90
    )

    for eta_name in (
        "eta_total",
        "eta_screen",
    ):

        x = np.asarray(
            [
                record[
                    eta_name
                ]
                for record in records
            ]
        )

        print()
        print(
            eta_name
        )

        for metric in (
            "retention",
            "spread",
            "entropy",
        ):

            y = np.asarray(
                [
                    record[
                        metric
                    ]
                    for record in records
                ]
            )

            rho = spearman(
                x,
                y,
            )

            print(
                f"  {metric:12s}: "
                f"rho = {rho:+.5f}"
            )


# ============================================================
# Correlaciones dentro de PSD y régimen
# ============================================================

def conditional_correlations(
    records: list[dict],
) -> None:

    print()
    print(
        "=" * 90
    )

    print(
        "CORRELACIONES DENTRO DE CADA PSD Y RÉGIMEN"
    )

    print(
        "=" * 90
    )

    psds = sorted(
        {
            record[
                "psd"
            ]
            for record in records
        }
    )

    regimes = (
        "weak",
        "moderate",
        "strong",
    )

    for regime in regimes:

        print()
        print(
            regime.upper()
        )

        for psd in psds:

            subset = [
                record
                for record in records
                if (
                    record[
                        "regime"
                    ] == regime
                    and record[
                        "psd"
                    ] == psd
                )
            ]

            x = np.asarray(
                [
                    record[
                        "eta_total"
                    ]
                    for record in subset
                ]
            )

            print(
                f"\n  {psd}"
            )

            for metric in (
                "retention",
                "spread",
                "entropy",
            ):

                y = np.asarray(
                    [
                        record[
                            metric
                        ]
                        for record in subset
                    ]
                )

                rho = spearman(
                    x,
                    y,
                )

                print(
                    f"    {metric:12s}: "
                    f"rho = {rho:+.4f}"
                )


# ============================================================
# Gráficas
# ============================================================

def plot_eta_metrics(
    records: list[dict],
    eta_name: str,
) -> None:

    metrics = (
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

    psds = (
        "kolmogorov",
        "von_karman",
        "modified_von_karman",
    )

    psd_labels = {
        "kolmogorov":
            "Kolmogorov",

        "von_karman":
            "von Kármán",

        "modified_von_karman":
            "von Kármán modificado",
    }

    markers = {
        "LG": "o",
        "BG": "s",
    }

    for metric, ylabel in metrics:

        figure, axis = plt.subplots(
            figsize=(
                7.5,
                5.5,
            )
        )

        for psd in psds:

            for family in (
                "LG",
                "BG",
            ):

                subset = [
                    record
                    for record in records
                    if (
                        record[
                            "psd"
                        ] == psd
                        and record[
                            "family"
                        ] == family
                    )
                ]

                x = np.asarray(
                    [
                        record[
                            eta_name
                        ]
                        for record in subset
                    ]
                )

                y = np.asarray(
                    [
                        record[
                            metric
                        ]
                        for record in subset
                    ]
                )

                axis.scatter(
                    x,
                    y,
                    marker=markers[
                        family
                    ],
                    s=45,
                    label=(
                        f"{family}, "
                        f"{psd_labels[psd]}"
                    ),
                )

        if eta_name == "eta_total":

            xlabel = (
                r"$\overline{\eta}_{\mathrm{total}}"
                r"=\langle r_{\mathrm{rms}}\rangle_z/"
                r"r_{0,\mathrm{total}}$"
            )

        else:

            xlabel = (
                r"$\overline{\eta}_{\mathrm{screen}}"
                r"=\langle r_{\mathrm{rms}}\rangle_z/"
                r"r_{0,\mathrm{screen}}$"
            )

        axis.set_xlabel(
            xlabel
        )

        axis.set_ylabel(
            ylabel
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend(
            fontsize=8,
            ncol=2,
        )

        figure.tight_layout()

        filename = (
            ANALYSIS_DIRECTORY
            / f"{metric}_vs_{eta_name}.png"
        )

        figure.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )


# ============================================================
# Guardar dataset
# ============================================================

def save_dataset(
    records: list[dict],
) -> None:

    filename = (
        ANALYSIS_DIRECTORY
        / "eta_dataset.csv"
    )

    fieldnames = list(
        records[0].keys()
    )

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    print()
    print(
        f"Dataset guardado: {filename}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    rows = read_scenario_summary(
        SCENARIO_SUMMARY_FILE
    )

    sizes = calculate_effective_sizes()

    records = build_eta_dataset(
        rows=rows,
        sizes=sizes,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PARÁMETROS DE FRIED"
    )

    print(
        "=" * 90
    )

    for regime in (
        "weak",
        "moderate",
        "strong",
    ):

        print(
            f"{regime:10s}: "
            f"r0 total = "
            f"{R0_TOTAL[regime]:.6f} m, "
            f"r0 pantalla = "
            f"{R0_SCREEN[regime]:.6f} m"
        )

    print()
    print(
        "=" * 90
    )

    print(
        "TAMAÑO EFECTIVO PROMEDIO"
    )

    print(
        "=" * 90
    )

    for beam in BEAMS:

        print(
            f"{beam:5s}: "
            f"<r_rms> = "
            f"{1000.0 * sizes[beam]['mean_path']:.4f} mm"
        )

    calculate_correlations(
        records
    )

    conditional_correlations(
        records
    )

    save_dataset(
        records
    )

    plot_eta_metrics(
        records=records,
        eta_name="eta_total",
    )

    plot_eta_metrics(
        records=records,
        eta_name="eta_screen",
    )

    print()
    print(
        "=" * 90
    )

    print(
        "ANÁLISIS COMPLETADO"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    main()
