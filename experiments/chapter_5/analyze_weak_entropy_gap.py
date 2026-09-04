"""
Diagnóstico de la estructura de la brecha de entropía en
turbulencia débil.

Objetivo
--------
Determinar si la variación de Delta H_ens entre los 18 escenarios
del régimen débil está asociada principalmente con:

    - el orden azimutal |ell|;
    - la familia del haz (LG/BG);
    - el modelo de PSD.

No realiza nuevas simulaciones.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "results/chapter_5/analysis/scenario_summary.csv"
)


def print_group_summary(
    data: pd.DataFrame,
    variable: str,
) -> None:

    summary = (
        data
        .groupby(variable)["entropy_gap"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print()
    print("=" * 80)
    print(
        f"BRECHA DE ENTROPÍA AGRUPADA POR {variable.upper()}"
    )
    print("=" * 80)

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


def main():

    data = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "beam",
        "family",
        "order",
        "psd",
        "regime",
        "entropy_gap",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas: {sorted(missing)}"
        )

    # ========================================================
    # Seleccionar únicamente turbulencia débil
    # ========================================================

    weak = (
        data.loc[
            data["regime"] == "weak"
        ]
        .copy()
    )

    if len(weak) != 18:
        raise RuntimeError(
            "Se esperaban 18 escenarios en turbulencia débil, "
            f"pero se encontraron {len(weak)}."
        )

    # ========================================================
    # Mostrar los 18 escenarios ordenados por |ell|
    # ========================================================

    weak = weak.sort_values(
        [
            "order",
            "family",
            "psd",
        ]
    )

    print()
    print("=" * 80)
    print("LOS 18 ESCENARIOS EN TURBULENCIA DÉBIL")
    print("=" * 80)

    print(
        weak[
            [
                "beam",
                "family",
                "order",
                "psd",
                "entropy_gap",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ========================================================
    # Comparaciones marginales
    # ========================================================

    print_group_summary(
        weak,
        "order",
    )

    print_group_summary(
        weak,
        "family",
    )

    print_group_summary(
        weak,
        "psd",
    )

    # ========================================================
    # Diferencias entre órdenes
    # ========================================================

    order_means = (
        weak
        .groupby("order")["entropy_gap"]
        .mean()
        .sort_index()
    )

    print()
    print("=" * 80)
    print("CAMBIO MEDIO ENTRE ÓRDENES")
    print("=" * 80)

    for first, second in [
        (1, 2),
        (2, 3),
        (1, 3),
    ]:

        difference = (
            order_means.loc[second]
            - order_means.loc[first]
        )

        print(
            f"|ell|={second} - |ell|={first}: "
            f"{difference:+.6f}"
        )

    # ========================================================
    # Consistencia de la tendencia con el orden
    #
    # Para cada combinación family + PSD comprobamos:
    #
    # DeltaH(ell=1) < DeltaH(ell=2) < DeltaH(ell=3)
    # ========================================================

    pivot = weak.pivot(
        index=[
            "family",
            "psd",
        ],
        columns="order",
        values="entropy_gap",
    )

    required_orders = {
        1,
        2,
        3,
    }

    if not required_orders.issubset(
        set(pivot.columns)
    ):
        raise RuntimeError(
            "No están disponibles los tres órdenes."
        )

    monotonic = (
        (pivot[1] < pivot[2])
        &
        (pivot[2] < pivot[3])
    )

    print()
    print("=" * 80)
    print("CONSISTENCIA DE LA DEPENDENCIA CON |ell|")
    print("=" * 80)

    output = pivot.copy()

    output[
        "1<2<3"
    ] = monotonic

    print(
        output.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()

    print(
        "Configuraciones con crecimiento monótono "
        "1 < 2 < 3:"
    )

    print(
        f"{monotonic.sum()}/{len(monotonic)}"
    )

    # ========================================================
    # Rango explicado aproximadamente por cada variable
    #
    # No es una ANOVA. Solo es un diagnóstico descriptivo:
    # diferencia entre la mayor y menor media de grupo.
    # ========================================================

    print()
    print("=" * 80)
    print("RANGO ENTRE MEDIAS DE GRUPO")
    print("=" * 80)

    for variable in [
        "order",
        "family",
        "psd",
    ]:

        means = (
            weak
            .groupby(variable)["entropy_gap"]
            .mean()
        )

        span = (
            means.max()
            - means.min()
        )

        print(
            f"{variable:>10s}: "
            f"{span:.6f}"
        )


if __name__ == "__main__":
    main()
