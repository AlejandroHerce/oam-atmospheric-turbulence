"""
Capítulo 5 — comparación de los modelos espectrales
===================================================

Visualización de las tres PSD de fase utilizadas en las
simulaciones finales:

    - Kolmogorov
    - von Kármán
    - von Kármán modificado

Las funciones se importan directamente desde src.phase_screens
para garantizar que la figura representa exactamente los modelos
empleados en las simulaciones.

Se muestran:

1. Las PSD de fase en escala log-log.
2. La razón de cada modelo respecto a Kolmogorov.

También se indican las frecuencias angulares asociadas a las
escalas externa e interna y el intervalo espectral representable
por la malla numérica.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.chapter_4 import (
    DX,
    N_GRID,
    STRONG_R0_SCREEN,
)

from src.phase_screens import (
    kolmogorov_psd,
    von_karman_psd,
    modified_von_karman_psd,
)


# ============================================================
# Configuración
# ============================================================

OUTER_SCALE = 10.0       # [m]
INNER_SCALE = 5.0e-3     # [m]

OUTPUT_DIRECTORY = Path(
    "results/chapter_5/analysis/figures"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frecuencias características
# ============================================================

KAPPA_0 = (
    2.0
    * np.pi
    / OUTER_SCALE
)

KAPPA_M = (
    5.92
    / INNER_SCALE
)

WINDOW_SIZE = (
    N_GRID
    * DX
)

KAPPA_FUNDAMENTAL = (
    2.0
    * np.pi
    / WINDOW_SIZE
)

KAPPA_NYQUIST = (
    np.pi
    / DX
)


# ============================================================
# Mallado espectral
# ============================================================

def create_kappa_axis() -> np.ndarray:
    """
    Crear un eje logarítmico que cubra el intervalo relevante
    para las tres PSD y la malla numérica.
    """

    minimum = min(
        KAPPA_0 / 50.0,
        KAPPA_FUNDAMENTAL / 50.0,
    )

    maximum = max(
        KAPPA_M * 10.0,
        KAPPA_NYQUIST * 2.0,
    )

    return np.logspace(
        np.log10(minimum),
        np.log10(maximum),
        3000,
    )


# ============================================================
# PSD
# ============================================================

def calculate_psds(
    kappa: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluar exactamente las funciones utilizadas por el modelo
    de pantallas de fase.
    """

    kolmogorov = kolmogorov_psd(
        kappa=kappa,
        r0=STRONG_R0_SCREEN,
    )

    von_karman = von_karman_psd(
        kappa=kappa,
        r0=STRONG_R0_SCREEN,
        outer_scale=OUTER_SCALE,
    )

    modified = modified_von_karman_psd(
        kappa=kappa,
        r0=STRONG_R0_SCREEN,
        outer_scale=OUTER_SCALE,
        inner_scale=INNER_SCALE,
    )

    return (
        kolmogorov,
        von_karman,
        modified,
    )


# ============================================================
# Figura
# ============================================================

def main() -> None:

    kappa = create_kappa_axis()

    (
        kolmogorov,
        von_karman,
        modified,
    ) = calculate_psds(
        kappa
    )

    # Evitar cualquier división problemática.
    valid = (
        kolmogorov > 0.0
    )

    ratio_vk = np.full_like(
        kappa,
        np.nan,
        dtype=np.float64,
    )

    ratio_mvk = np.full_like(
        kappa,
        np.nan,
        dtype=np.float64,
    )

    ratio_vk[valid] = (
        von_karman[valid]
        / kolmogorov[valid]
    )

    ratio_mvk[valid] = (
        modified[valid]
        / kolmogorov[valid]
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.0, 8.0),
        sharex=True,
    )

    # ========================================================
    # Panel (a): PSD
    # ========================================================

    axis = axes[0]

    axis.loglog(
        kappa,
        kolmogorov,
        linewidth=1.7,
        label="Kolmogorov",
    )

    axis.loglog(
        kappa,
        von_karman,
        linewidth=1.7,
        label="von Kármán",
    )

    axis.loglog(
        kappa,
        modified,
        linewidth=1.7,
        label="von Kármán modificado",
    )

    axis.set_ylabel(
        r"PSD de fase $\Phi_{\theta}(\kappa)$"
    )

    axis.grid(
        alpha=0.25,
        which="both",
    )

    axis.legend()

    axis.text(
        0.02,
        0.95,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # ========================================================
    # Panel (b): razón respecto a Kolmogorov
    # ========================================================

    axis = axes[1]

    axis.semilogx(
        kappa,
        ratio_vk,
        linewidth=1.7,
        label="von Kármán / Kolmogorov",
    )

    axis.semilogx(
        kappa,
        ratio_mvk,
        linewidth=1.7,
        label="von Kármán modificado / Kolmogorov",
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        r"Frecuencia espacial angular $\kappa$ [rad/m]"
    )

    axis.set_ylabel(
        r"$\Phi_{\theta}^{(\mathrm{PSD})}/"
        r"\Phi_{\theta}^{(\mathrm{K})}$"
    )

    axis.grid(
        alpha=0.25,
        which="both",
    )

    axis.legend()

    axis.text(
        0.02,
        0.95,
        "(b)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

        # ========================================================
    # Escalas características
    # ========================================================

    for axis in axes:

        # Frecuencias características de los modelos de PSD
        axis.axvline(
            KAPPA_0,
            linestyle=":",
            linewidth=1.2,
        )

        axis.axvline(
            KAPPA_M,
            linestyle=":",
            linewidth=1.2,
        )

        # Límites de la banda representada por la FFT base
        axis.axvline(
            KAPPA_FUNDAMENTAL,
            linestyle="-.",
            linewidth=1.0,
        )

        axis.axvline(
            KAPPA_NYQUIST,
            linestyle="-.",
            linewidth=1.0,
        )

        # Región directamente representada por la FFT base
        axis.axvspan(
            KAPPA_FUNDAMENTAL,
            KAPPA_NYQUIST,
            alpha=0.08,
            label=None,
        )

    # ========================================================
    # Etiquetas de las frecuencias características
    # ========================================================

    # Se colocan en el panel superior para evitar saturación.
    y_top = axes[0].get_ylim()[1]

    axes[0].text(
        KAPPA_0,
        y_top / 3.0,
        r"$\kappa_0$",
        rotation=90,
        va="top",
        ha="right",
    )

    axes[0].text(
        KAPPA_M,
        y_top / 3.0,
        r"$\kappa_m$",
        rotation=90,
        va="top",
        ha="right",
    )

    axes[0].text(
        KAPPA_FUNDAMENTAL,
        y_top / 3.0,
        r"$\kappa_{\mathrm{fund}}$",
        rotation=90,
        va="top",
        ha="left",
    )

    axes[0].text(
        KAPPA_NYQUIST,
        y_top / 3.0,
        r"$\kappa_{\mathrm{Nyq}}$",
        rotation=90,
        va="top",
        ha="right",
    )

    # ========================================================
    # Etiqueta de la banda FFT
    # ========================================================

    # Centro geométrico porque el eje horizontal es logarítmico.
    kappa_fft_center = np.sqrt(
        KAPPA_FUNDAMENTAL
        * KAPPA_NYQUIST
    )

    axes[1].text(
        kappa_fft_center,
        0.08,
        "Banda FFT base",
        ha="center",
        va="bottom",
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    pdf_file = (
        OUTPUT_DIRECTORY
        / "phase_psd_comparison.pdf"
    )

    png_file = (
        OUTPUT_DIRECTORY
        / "phase_psd_comparison.png"
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

    # --------------------------------------------------------
    # Información numérica
    # --------------------------------------------------------

    print()
    print(
        "Escalas espaciales características"
    )

    print(
        "=" * 55
    )

    print(
        f"kappa_0              = "
        f"{KAPPA_0:.6e} rad/m"
    )

    print(
        f"kappa_m              = "
        f"{KAPPA_M:.6e} rad/m"
    )

    print(
        f"kappa fundamental FFT = "
        f"{KAPPA_FUNDAMENTAL:.6e} rad/m"
    )

    print(
        f"kappa Nyquist         = "
        f"{KAPPA_NYQUIST:.6e} rad/m"
    )

    print()

    print(
        f"Figura guardada: "
        f"{png_file}"
    )


if __name__ == "__main__":
    main()
