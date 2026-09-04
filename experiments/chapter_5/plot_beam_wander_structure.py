from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

RESULTS_ROOT = Path("results/chapter_5")
OUTPUT_DIR = RESULTS_ROOT / "analysis" / "beam_wander"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSD_ORDER = [
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
]

PSD_LABELS = {
    "kolmogorov": "Kolmogorov",
    "von_karman": "von Kármán",
    "modified_von_karman": "von Kármán modificado",
}

REGIME_ORDER = ["weak", "moderate", "strong"]

REGIME_LABELS = {
    "weak": "Débil",
    "moderate": "Moderado",
    "strong": "Fuerte",
}


# ============================================================
# CARGA DE DATOS
# ============================================================

records = []

for path in RESULTS_ROOT.glob("*/*/*/metrics.csv"):

    # Esperamos:
    # results/chapter_5/<psd>/<regime>/<beam>/metrics.csv

    try:
        psd = path.parts[-4]
        regime = path.parts[-3]
        beam = path.parts[-2]
    except IndexError:
        continue

    if psd not in PSD_ORDER:
        continue

    if regime not in REGIME_ORDER:
        continue

    if not (beam.startswith("LG") or beam.startswith("BG")):
        continue

    df = pd.read_csv(path)

    required = {
        "centroid_x_m",
        "centroid_y_m",
    }

    if not required.issubset(df.columns):
        print(f"Saltando {path}: faltan columnas")
        continue

    # --------------------------------------------------------
    # Beam wander RMS:
    #
    # sigma_BW = sqrt(
    #     <(xc - <xc>)^2 + (yc - <yc>)^2>
    # )
    #
    # Se elimina el pequeño offset residual del ensamble.
    # --------------------------------------------------------

    x = df["centroid_x_m"].to_numpy()
    y = df["centroid_y_m"].to_numpy()

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    wander_rms = np.sqrt(
        np.mean(x_centered**2 + y_centered**2)
    )

    family = beam[:2]
    order = int(beam[2:])

    records.append(
        {
            "beam": beam,
            "family": family,
            "order": order,
            "psd": psd,
            "regime": regime,
            "beam_wander_m": wander_rms,
        }
    )


data = pd.DataFrame(records)

if data.empty:
    raise RuntimeError(
        "No se encontraron datos válidos."
    )

print(f"\nEscenarios encontrados: {len(data)}")


# Convertimos a mm para facilitar la lectura
data["beam_wander_mm"] = 1e3 * data["beam_wander_m"]


# ============================================================
# RESÚMENES
# ============================================================

print("\n" + "=" * 90)
print("BEAM WANDER MEDIO POR PSD Y RÉGIMEN")
print("=" * 90)

summary_psd = (
    data.groupby(["regime", "psd"])["beam_wander_mm"]
    .agg(["mean", "std", "min", "max"])
)

print(summary_psd)


print("\n" + "=" * 90)
print("BEAM WANDER MEDIO POR ORDEN Y RÉGIMEN")
print("=" * 90)

summary_order = (
    data.groupby(["regime", "order"])["beam_wander_mm"]
    .agg(["mean", "std", "min", "max"])
)

print(summary_order)


# ============================================================
# FIGURA
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)


# ============================================================
# PANEL (a)
# Dependencia con PSD
# ============================================================

ax = axes[0]

x = np.arange(len(REGIME_ORDER))

for psd in PSD_ORDER:

    means = []
    stds = []

    for regime in REGIME_ORDER:

        subset = data[
            (data["psd"] == psd)
            & (data["regime"] == regime)
        ]["beam_wander_mm"]

        means.append(subset.mean())
        stds.append(subset.std())

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=PSD_LABELS[psd],
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [REGIME_LABELS[r] for r in REGIME_ORDER]
)

ax.set_xlabel("Régimen de turbulencia")
ax.set_ylabel("Beam wander RMS [mm]")

ax.set_title(
    "(a) Dependencia con el modelo espectral"
)

ax.grid(alpha=0.25)
ax.legend(frameon=False)


# ============================================================
# PANEL (b)
# Dependencia con |ell|
# ============================================================

ax = axes[1]

orders = sorted(data["order"].unique())

for regime in REGIME_ORDER:

    means = []
    stds = []

    for order in orders:

        subset = data[
            (data["regime"] == regime)
            & (data["order"] == order)
        ]["beam_wander_mm"]

        means.append(subset.mean())
        stds.append(subset.std())

    ax.errorbar(
        orders,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=REGIME_LABELS[regime],
    )

ax.set_xticks(orders)

ax.set_xlabel(r"Orden azimutal $|\ell|$")
ax.set_ylabel("Beam wander RMS [mm]")

ax.set_title(
    r"(b) Dependencia con el orden $|\ell|$"
)

ax.grid(alpha=0.25)
ax.legend(frameon=False)


# ============================================================
# GUARDAR
# ============================================================

fig.tight_layout()

png_path = OUTPUT_DIR / "beam_wander_structure.png"
pdf_path = OUTPUT_DIR / "beam_wander_structure.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(fig)

print("\nFigura guardada en:")
print(png_path)
print(pdf_path)
