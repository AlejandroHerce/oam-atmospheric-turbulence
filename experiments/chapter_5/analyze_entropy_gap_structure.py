"""
Analyze the structure of the ensemble entropy gap in Chapter 5.

The script evaluates two questions:

1. Whether the entropy gap

       Delta H_ens =
           H(<P>) - <H(P)>

   reaches its maximum in the moderate turbulence regime
   for each beam + PSD combination.

2. The relative contribution

       f_inter =
           Delta H_ens / H(<P>)

   which quantifies the fraction of the entropy of the
   ensemble-averaged spectrum associated with spectral
   heterogeneity between atmospheric realizations.

No production simulations are performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ANALYSIS_DIRECTORY = Path(
    "results/chapter_5/analysis"
)

INPUT_FILE = (
    ANALYSIS_DIRECTORY
    / "scenario_summary.csv"
)

OUTPUT_COMPARISON = (
    ANALYSIS_DIRECTORY
    / "entropy_gap_regime_comparison.csv"
)

OUTPUT_REGIME_SUMMARY = (
    ANALYSIS_DIRECTORY
    / "entropy_gap_relative_summary.csv"
)


# ============================================================
# Load data
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# Check required columns
# ============================================================

required_columns = {
    "beam",
    "psd",
    "regime",
    "entropy_mean",
    "entropy_mean_spectrum",
    "entropy_gap",
}

missing = (
    required_columns
    - set(data.columns)
)

if missing:
    raise RuntimeError(
        "Missing columns in scenario_summary.csv: "
        f"{sorted(missing)}"
    )


# ============================================================
# Relative inter-realization contribution
# ============================================================

data["entropy_gap_fraction"] = (
    data["entropy_gap"]
    / data["entropy_mean_spectrum"]
)
data["entropy_gap_percent"] = (
    100.0
    * data["entropy_gap_fraction"]
)


# ============================================================
# Compare regimes for every beam + PSD combination
# ============================================================

records = []

for (
    beam,
    psd,
), group in data.groupby(
    [
        "beam",
        "psd",
    ]
):

    values = (
        group
        .set_index("regime")[
            "entropy_gap"
        ]
    )

    required_regimes = {
        "weak",
        "moderate",
        "strong",
    }

    if not required_regimes.issubset(
        values.index
    ):
        raise RuntimeError(
            f"Missing turbulence regime for "
            f"{beam}, {psd}"
        )

    weak = float(
        values["weak"]
    )

    moderate = float(
        values["moderate"]
    )

    strong = float(
        values["strong"]
    )

    moderate_gt_weak = (
        moderate > weak
    )

    moderate_gt_strong = (
        moderate > strong
    )

    moderate_is_maximum = (
        moderate_gt_weak
        and moderate_gt_strong
    )

    records.append(
        {
            "beam":
                beam,

            "psd":
                psd,

            "entropy_gap_weak":
                weak,

            "entropy_gap_moderate":
                moderate,

            "entropy_gap_strong":
                strong,

            "moderate_gt_weak":
                moderate_gt_weak,

            "moderate_gt_strong":
                moderate_gt_strong,

            "moderate_is_maximum":
                moderate_is_maximum,

            "moderate_minus_weak":
                moderate - weak,

            "moderate_minus_strong":
                moderate - strong,
        }
    )


comparison = pd.DataFrame(
    records
)

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False,
)


# ============================================================
# Overall consistency
# ============================================================

n_cases = len(
    comparison
)

n_moderate_gt_weak = int(
    comparison[
        "moderate_gt_weak"
    ].sum()
)

n_moderate_gt_strong = int(
    comparison[
        "moderate_gt_strong"
    ].sum()
)

n_moderate_maximum = int(
    comparison[
        "moderate_is_maximum"
    ].sum()
)


# ============================================================
# Relative contribution by regime
# ============================================================

regime_summary = (
    data
    .groupby("regime")
    .agg(
        n_scenarios=(
            "entropy_gap",
            "size",
        ),

        entropy_mean=(
            "entropy_mean",
            "mean",
        ),
        
        entropy_of_mean=(
            "entropy_mean_spectrum",
            "mean",
        ),

        entropy_gap_mean=(
            "entropy_gap",
            "mean",
        ),

        entropy_gap_median=(
            "entropy_gap",
            "median",
        ),

        entropy_gap_fraction_mean=(
            "entropy_gap_fraction",
            "mean",
        ),

        entropy_gap_fraction_median=(
            "entropy_gap_fraction",
            "median",
        ),

        entropy_gap_percent_mean=(
            "entropy_gap_percent",
            "mean",
        ),
    )
    .reset_index()
)

# Put regimes in physical order.

regime_order = {
    "weak": 0,
    "moderate": 1,
    "strong": 2,
}

regime_summary["_order"] = (
    regime_summary[
        "regime"
    ].map(
        regime_order
    )
)

regime_summary = (
    regime_summary
    .sort_values("_order")
    .drop(
        columns="_order"
    )
)

regime_summary.to_csv(
    OUTPUT_REGIME_SUMMARY,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print()

print(
    "=" * 90
)

print(
    "CONSISTENCY OF THE ENTROPY-GAP MAXIMUM"
)

print(
    "=" * 90
)

print(
    comparison.to_string(
        index=False
    )
)

print()

print(
    "Moderate > weak:"
)

print(
    f"{n_moderate_gt_weak}/{n_cases}"
)

print()

print(
    "Moderate > strong:"
)

print(
    f"{n_moderate_gt_strong}/{n_cases}"
)

print()

print(
    "Moderate is the maximum:"
)

print(
    f"{n_moderate_maximum}/{n_cases}"
)


print()

print(
    "=" * 90
)

print(
    "RELATIVE ENTROPY-GAP CONTRIBUTION"
)

print(
    "=" * 90
)

print(
    regime_summary.to_string(
        index=False
    )
)

print()

print(
    "Files saved:"
)

print(
    OUTPUT_COMPARISON
)

print(
    OUTPUT_REGIME_SUMMARY
)
