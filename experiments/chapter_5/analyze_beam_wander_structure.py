from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Paths
# ============================================================

ROOT = Path("results/chapter_5")

# Buscar automáticamente los archivos por realización
files = sorted(
    f for f in ROOT.rglob("metrics.csv")
    if "controls" not in f.parts
)

if not files:
    raise RuntimeError(
        "No se encontraron archivos metrics.csv "
        "dentro de los escenarios principales de results/chapter_5"
    )

print(f"\nArchivos encontrados: {len(files)}")


# ============================================================
# Infer scenario information from path
# ============================================================

def infer_scenario(path):
    """
    Extrae beam, family, order, PSD y regime a partir de la ruta.
    Ajusta únicamente esta función si tu estructura de carpetas
    usa otros nombres.
    """

    parts = [p.lower() for p in path.parts]

    # ---------- regime ----------
    regime = None
    for r in ["weak", "moderate", "strong"]:
        if r in parts:
            regime = r
            break

    # ---------- PSD ----------
    psd = None

    if "modified_von_karman" in parts:
        psd = "modified_von_karman"
    elif "von_karman" in parts:
        psd = "von_karman"
    elif "kolmogorov" in parts:
        psd = "kolmogorov"

    # ---------- beam ----------
    beam = None

    for p in path.parts:
        u = p.upper()

        if u.startswith("LG") or u.startswith("BG"):
            if len(u) >= 4 and u[2:].isdigit():
                beam = u
                break

    if beam is None:
        return None

    family = beam[:2]

    try:
        order = int(beam[2:])
    except ValueError:
        return None

    if regime is None or psd is None:
        return None

    return {
        "beam": beam,
        "family": family,
        "order": order,
        "psd": psd,
        "regime": regime,
    }


# ============================================================
# Process every scenario
# ============================================================

rows = []

for f in files:

    info = infer_scenario(f)

    if info is None:
        continue

    df = pd.read_csv(f)

    required = [
        "centroid_x_m",
        "centroid_y_m",
        "centroid_radius_m",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Saltando {f}: faltan {missing}")
        continue

    x = df["centroid_x_m"].to_numpy(float)
    y = df["centroid_y_m"].to_numpy(float)

    # Distancia radial respecto al eje óptico
    r = np.sqrt(x**2 + y**2)

    # Centroide medio
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # --------------------------------------------------------
    # Beam-wander metrics
    # --------------------------------------------------------

    # RMS respecto al eje óptico original
    radial_rms = np.sqrt(np.mean(x**2 + y**2))

    # RMS de las fluctuaciones respecto al centroide medio
    wander_std = np.sqrt(
        np.mean(
            (x - x_mean)**2
            +
            (y - y_mean)**2
        )
    )

    # Distancia media radial
    radial_mean = np.mean(r)

    # Desplazamiento del centroide medio respecto al eje
    mean_centroid_radius = np.sqrt(
        x_mean**2 + y_mean**2
    )

    rows.append({
        **info,
        "n": len(df),
        "x_mean_m": x_mean,
        "y_mean_m": y_mean,
        "mean_centroid_radius_m": mean_centroid_radius,
        "radial_mean_m": radial_mean,
        "radial_rms_m": radial_rms,
        "wander_std_m": wander_std,
    })


result = pd.DataFrame(rows)

if result.empty:
    raise RuntimeError(
        "No fue posible reconstruir ningún escenario."
    )


# ============================================================
# Ordering
# ============================================================

regime_order = ["weak", "moderate", "strong"]

result["regime"] = pd.Categorical(
    result["regime"],
    categories=regime_order,
    ordered=True,
)

result = result.sort_values(
    ["regime", "psd", "family", "order"]
)


# ============================================================
# Full table
# ============================================================

print("\n" + "=" * 100)
print("BEAM WANDER POR ESCENARIO")
print("=" * 100)

print(
    result[
        [
            "beam",
            "family",
            "order",
            "psd",
            "regime",
            "n",
            "x_mean_m",
            "y_mean_m",
            "mean_centroid_radius_m",
            "radial_mean_m",
            "radial_rms_m",
            "wander_std_m",
        ]
    ].to_string(index=False)
)


# ============================================================
# Summary by regime
# ============================================================

print("\n" + "=" * 100)
print("BEAM WANDER POR RÉGIMEN")
print("=" * 100)

summary_regime = (
    result
    .groupby("regime", observed=True)["wander_std_m"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(summary_regime)


# ============================================================
# Summary by PSD and regime
# ============================================================

print("\n" + "=" * 100)
print("BEAM WANDER POR PSD Y RÉGIMEN")
print("=" * 100)

summary_psd = (
    result
    .groupby(
        ["regime", "psd"],
        observed=True
    )["wander_std_m"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(summary_psd)


# ============================================================
# Summary by order and regime
# ============================================================

print("\n" + "=" * 100)
print("BEAM WANDER POR ORDEN Y RÉGIMEN")
print("=" * 100)

summary_order = (
    result
    .groupby(
        ["regime", "order"],
        observed=True
    )["wander_std_m"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(summary_order)


# ============================================================
# Summary by family and regime
# ============================================================

print("\n" + "=" * 100)
print("BEAM WANDER POR FAMILIA Y RÉGIMEN")
print("=" * 100)

summary_family = (
    result
    .groupby(
        ["regime", "family"],
        observed=True
    )["wander_std_m"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(summary_family)


# ============================================================
# Isotropy / centering control
# ============================================================

print("\n" + "=" * 100)
print("CONTROL DE CENTRADO")
print("=" * 100)

centering = (
    result
    .groupby("regime", observed=True)
    .agg(
        mean_x=("x_mean_m", "mean"),
        mean_y=("y_mean_m", "mean"),
        mean_offset=("mean_centroid_radius_m", "mean"),
        mean_wander=("wander_std_m", "mean"),
    )
)

centering["offset_over_wander"] = (
    centering["mean_offset"]
    /
    centering["mean_wander"]
)

print(centering)


# ============================================================
# Relative PSD comparison
# ============================================================

print("\n" + "=" * 100)
print("MEDIAS PSD × RÉGIMEN")
print("=" * 100)

pivot_psd = (
    result
    .groupby(
        ["regime", "psd"],
        observed=True
    )["wander_std_m"]
    .mean()
    .unstack()
)

print(pivot_psd)


# ============================================================
# Monotonicity with turbulence
# ============================================================

print("\n" + "=" * 100)
print("MONOTONICIDAD CON LA INTENSIDAD DE TURBULENCIA")
print("=" * 100)

pivot_scenario = result.pivot_table(
    index=["beam", "psd"],
    columns="regime",
    values="wander_std_m",
    observed=True,
)

needed = ["weak", "moderate", "strong"]

if all(c in pivot_scenario.columns for c in needed):

    pivot_scenario["weak_lt_moderate"] = (
        pivot_scenario["weak"]
        <
        pivot_scenario["moderate"]
    )

    pivot_scenario["moderate_lt_strong"] = (
        pivot_scenario["moderate"]
        <
        pivot_scenario["strong"]
    )

    pivot_scenario["monotonic"] = (
        pivot_scenario["weak_lt_moderate"]
        &
        pivot_scenario["moderate_lt_strong"]
    )

    print(pivot_scenario)

    print(
        "\nConfiguraciones con "
        "weak < moderate < strong:"
    )

    print(
        f"{pivot_scenario['monotonic'].sum()}"
        f"/{len(pivot_scenario)}"
    )


# ============================================================
# Save
# ============================================================

outdir = ROOT / "analysis"
outdir.mkdir(parents=True, exist_ok=True)

result.to_csv(
    outdir / "beam_wander_structure.csv",
    index=False,
)

summary_regime.to_csv(
    outdir / "beam_wander_by_regime.csv"
)

summary_psd.to_csv(
    outdir / "beam_wander_by_psd_regime.csv"
)

summary_order.to_csv(
    outdir / "beam_wander_by_order_regime.csv"
)

summary_family.to_csv(
    outdir / "beam_wander_by_family_regime.csv"
)

print(
    "\nResultados guardados en:",
    outdir
)

print("\nPrimeros archivos:")
for f in files[:10]:
    print(f)
