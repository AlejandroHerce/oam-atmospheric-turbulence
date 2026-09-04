"""
Plot the ensemble entropy gap for the 54 Chapter 5 scenarios.

Each line connects the same beam + PSD configuration across
weak, moderate, and strong turbulence.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "results/chapter_5/analysis/scenario_summary.csv"
)

OUTPUT_FILE = Path(
    "results/chapter_5/analysis/ensemble_entropy_gap.png"
)

REGIMES = [
    "weak",
    "moderate",
    "strong",
]

REGIME_LABELS = [
    "Débil",
    "Moderada",
    "Fuerte",
]

def main():

    data = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "beam",
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
            f"Missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Verify complete paired configurations
    # --------------------------------------------------------

    pivot = data.pivot(
        index=[
            "beam",
            "psd",
        ],
        columns="regime",
        values="entropy_gap",
    )

    pivot = pivot[
        REGIMES
    ]

    if pivot.isna().any().any():
        raise RuntimeError(
            "Some beam + PSD configurations do not contain "
            "all three turbulence regimes."
        )

    if len(pivot) != 18:
        raise RuntimeError(
            f"Expected 18 paired configurations, found {len(pivot)}."
        )

    x = np.arange(
        len(REGIMES)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0)
    )

    # Individual paired configurations
    for _, row in pivot.iterrows():

        values = row[
            REGIMES
        ].to_numpy(
            dtype=float
        )

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=0.8,
            markersize=3.5,
            alpha=0.35,
        )

    # --------------------------------------------------------
    # Ensemble mean across the 18 scenarios
    # --------------------------------------------------------

    means = np.asarray(
        [
            data.loc[
                data["regime"] == regime,
                "entropy_gap",
            ].mean()
            for regime in REGIMES
        ],
        dtype=float,
    )

    ax.plot(
        x,
        means,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Promedio",
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        REGIME_LABELS
    )

    ax.set_xlabel(
        "Régimen de turbulencia"
    )
    
    ax.set_ylabel(
        r"Brecha de entropía del ensamble $\Delta H_{\mathrm{ens}}$"
    )
    
    ax.legend(
        ["Promedio"],
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Numerical verification
    # --------------------------------------------------------

    moderate_is_maximum = (
        (
            pivot["moderate"]
            > pivot["weak"]
        )
        &
        (
            pivot["moderate"]
            > pivot["strong"]
        )
    )

    print()
    print(
        "Entropy-gap figure saved in:"
    )
    print(
        OUTPUT_FILE.resolve()
    )

    print()
    print(
        "Mean entropy gap by regime:"
    )

    for regime, value in zip(
        REGIMES,
        means,
    ):
        print(
            f"{regime:>10s}: {value:.6f}"
        )

    print()
    print(
        "Moderate regime is maximum:"
    )
    print(
        f"{moderate_is_maximum.sum()}/{len(pivot)}"
    )


if __name__ == "__main__":
    main()
