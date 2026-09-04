"""
Analyze the relationship between realization-level beam wander
and OAM-spectrum degradation in the 54 Chapter 5 production scenarios.

For every atmospheric realization, the script relates the centroid
displacement to

    - transmitted-mode retention;
    - RMS OAM spread;
    - normalized OAM entropy;
    - signed spectral asymmetry;
    - absolute spectral asymmetry.

The analysis is performed independently for every

    beam + PSD + turbulence regime

scenario.

In addition to the centroid radius r_c, the Cartesian centroid
coordinates x_c and y_c are tested against the signed spectral
asymmetry as an exploratory directional control.

No scientific figures are generated. The script only produces
numerical tables for subsequent interpretation and visualization.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


# ============================================================
# Configuration
# ============================================================

RESULTS_ROOT = Path(
    "results/chapter_5"
)

ANALYSIS_DIRECTORY = (
    RESULTS_ROOT
    / "analysis"
)

BEAMS = (
    "LG01",
    "LG02",
    "LG03",
    "BG01",
    "BG02",
    "BG03",
)

PSDS = (
    "kolmogorov",
    "von_karman",
    "modified_von_karman",
)

REGIMES = (
    "weak",
    "moderate",
    "strong",
)


# ============================================================
# Beam information
# ============================================================

def transmitted_charge(
    beam_name: str,
) -> int:

    mapping = {
        "LG01": 1,
        "LG02": 2,
        "LG03": 3,
        "BG01": 1,
        "BG02": 2,
        "BG03": 3,
    }

    return mapping[
        beam_name
    ]


def beam_family(
    beam_name: str,
) -> str:

    if beam_name.startswith(
        "LG"
    ):
        return "LG"

    if beam_name.startswith(
        "BG"
    ):
        return "BG"

    raise ValueError(
        f"Unknown beam: {beam_name}"
    )


# ============================================================
# Spectral asymmetry
# ============================================================

def calculate_realization_asymmetry(
    ell: np.ndarray,
    spectra: np.ndarray,
    ell0: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate signed and absolute spectral asymmetry separately
    for every atmospheric realization.

    For realization j,

        A_s(j) =
            sum_d [
                P_j(ell0+d)
                -
                P_j(ell0-d)
            ]

    and

        A_abs(j) =
            sum_d |
                P_j(ell0+d)
                -
                P_j(ell0-d)
            |.

    Only modal pairs present on both sides of ell0 are used.
    """

    ell = np.asarray(
        ell,
        dtype=np.int64,
    )

    spectra = np.asarray(
        spectra,
        dtype=np.float64,
    )

    if spectra.ndim != 2:
        raise ValueError(
            "spectra must be a two-dimensional array."
        )

    if (
        spectra.shape[1]
        != ell.size
    ):
        raise ValueError(
            "The number of spectral columns does not match ell."
        )

    maximum_delta = min(
        int(
            ell0
            - ell[0]
        ),
        int(
            ell[-1]
            - ell0
        ),
    )

    positive_modes = (
        ell0
        + np.arange(
            1,
            maximum_delta + 1,
            dtype=np.int64,
        )
    )

    negative_modes = (
        ell0
        - np.arange(
            1,
            maximum_delta + 1,
            dtype=np.int64,
        )
    )

    lookup = {
        int(mode): index
        for index, mode in enumerate(
            ell
        )
    }

    positive_indices = np.asarray(
        [
            lookup[
                int(mode)
            ]
            for mode in positive_modes
        ],
        dtype=np.int64,
    )

    negative_indices = np.asarray(
        [
            lookup[
                int(mode)
            ]
            for mode in negative_modes
        ],
        dtype=np.int64,
    )

    differences = (
        spectra[
            :,
            positive_indices
        ]
        -
        spectra[
            :,
            negative_indices
        ]
    )

    signed_asymmetry = np.sum(
        differences,
        axis=1,
    )

    absolute_asymmetry = np.sum(
        np.abs(
            differences
        ),
        axis=1,
    )

    return (
        np.asarray(
            signed_asymmetry,
            dtype=np.float64,
        ),
        np.asarray(
            absolute_asymmetry,
            dtype=np.float64,
        ),
    )


# ============================================================
# Spearman helper
# ============================================================

def calculate_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    float,
    float,
]:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if np.sum(
        valid
    ) < 3:
        return (
            np.nan,
            np.nan,
        )

    result = spearmanr(
        x[
            valid
        ],
        y[
            valid
        ],
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


# ============================================================
# Load one scenario
# ============================================================

def load_scenario(
    beam_name: str,
    psd_name: str,
    regime_name: str,
) -> dict:

    directory = (
        RESULTS_ROOT
        / psd_name
        / regime_name
        / beam_name
    )

    metrics_file = (
        directory
        / "metrics.csv"
    )

    spectra_file = (
        directory
        / "oam_spectra.npz"
    )

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Missing file: {metrics_file}"
        )

    if not spectra_file.exists():
        raise FileNotFoundError(
            f"Missing file: {spectra_file}"
        )

    metrics = pd.read_csv(
        metrics_file
    )

    required_columns = {
        "retention",
        "oam_rms_spread",
        "normalized_oam_entropy",
        "centroid_x_m",
        "centroid_y_m",
        "centroid_radius_m",
    }

    missing_columns = (
        required_columns
        - set(
            metrics.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing columns in "
            f"{metrics_file}: "
            f"{sorted(missing_columns)}"
        )

    with np.load(
        spectra_file
    ) as archive:

        ell = np.asarray(
            archive[
                "ell_values"
            ],
            dtype=np.int64,
        )

        spectra = np.asarray(
            archive[
                "modal_power"
            ],
            dtype=np.float64,
        )

    if (
        spectra.shape[0]
        != len(metrics)
    ):
        raise RuntimeError(
            f"Realization count mismatch in {directory}."
        )

    return {
        "metrics":
            metrics,

        "ell":
            ell,

        "spectra":
            spectra,
    }


# ============================================================
# Analyze one scenario
# ============================================================

def analyze_scenario(
    beam_name: str,
    psd_name: str,
    regime_name: str,
) -> list[dict]:

    scenario = load_scenario(
        beam_name=beam_name,
        psd_name=psd_name,
        regime_name=regime_name,
    )

    metrics = (
        scenario[
            "metrics"
        ]
    )

    ell = (
        scenario[
            "ell"
        ]
    )

    spectra = (
        scenario[
            "spectra"
        ]
    )

    ell0 = transmitted_charge(
        beam_name
    )

    (
        signed_asymmetry,
        absolute_asymmetry,
    ) = calculate_realization_asymmetry(
        ell=ell,
        spectra=spectra,
        ell0=ell0,
    )

    radius = metrics[
        "centroid_radius_m"
    ].to_numpy(
        dtype=np.float64
    )

    x_centroid = metrics[
        "centroid_x_m"
    ].to_numpy(
        dtype=np.float64
    )

    y_centroid = metrics[
        "centroid_y_m"
    ].to_numpy(
        dtype=np.float64
    )

    variables = {
        "retention":
            metrics[
                "retention"
            ].to_numpy(
                dtype=np.float64
            ),

        "oam_rms_spread":
            metrics[
                "oam_rms_spread"
            ].to_numpy(
                dtype=np.float64
            ),

        "normalized_oam_entropy":
            metrics[
                "normalized_oam_entropy"
            ].to_numpy(
                dtype=np.float64
            ),

        "signed_asymmetry":
            signed_asymmetry,

        "absolute_asymmetry":
            absolute_asymmetry,
    }

    records = []

    # --------------------------------------------------------
    # Main radial-wander correlations
    # --------------------------------------------------------

    for metric_name, values in (
        variables.items()
    ):

        rho, p_value = (
            calculate_spearman(
                radius,
                values,
            )
        )

        records.append(
            {
                "beam":
                    beam_name,

                "family":
                    beam_family(
                        beam_name
                    ),

                "order":
                    ell0,

                "psd":
                    psd_name,

                "regime":
                    regime_name,

                "n":
                    len(
                        metrics
                    ),

                "centroid_variable":
                    "radius",

                "oam_metric":
                    metric_name,

                "spearman_rho":
                    rho,

                "p_value":
                    p_value,
            }
        )

    # --------------------------------------------------------
    # Directional exploratory controls
    #
    # Signed asymmetry is the only metric for which the sign
    # of the centroid coordinate may carry additional
    # directional information.
    # --------------------------------------------------------

    for (
        centroid_name,
        centroid_values,
    ) in (
        (
            "x",
            x_centroid,
        ),
        (
            "y",
            y_centroid,
        ),
    ):

        rho, p_value = (
            calculate_spearman(
                centroid_values,
                signed_asymmetry,
            )
        )

        records.append(
            {
                "beam":
                    beam_name,

                "family":
                    beam_family(
                        beam_name
                    ),

                "order":
                    ell0,

                "psd":
                    psd_name,

                "regime":
                    regime_name,

                "n":
                    len(
                        metrics
                    ),

                "centroid_variable":
                    centroid_name,

                "oam_metric":
                    "signed_asymmetry",

                "spearman_rho":
                    rho,

                "p_value":
                    p_value,
            }
        )

    return records


# ============================================================
# Summaries
# ============================================================

def summarize_radial_correlations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    radial = dataframe[
        dataframe[
            "centroid_variable"
        ] == "radius"
    ].copy()

    summary = (
        radial
        .groupby(
            [
                "regime",
                "oam_metric",
            ],
            sort=False,
        )[
            "spearman_rho"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "count":
                "n_scenarios",

            "mean":
                "rho_mean",

            "median":
                "rho_median",

            "std":
                "rho_std",

            "min":
                "rho_min",

            "max":
                "rho_max",
        }
    )

    return summary


def summarize_sign_consistency(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    radial = dataframe[
        dataframe[
            "centroid_variable"
        ] == "radius"
    ].copy()

    records = []

    for (
        regime,
        metric,
    ), group in radial.groupby(
        [
            "regime",
            "oam_metric",
        ],
        sort=False,
    ):

        rho = group[
            "spearman_rho"
        ].to_numpy(
            dtype=np.float64
        )

        records.append(
            {
                "regime":
                    regime,

                "oam_metric":
                    metric,

                "n_scenarios":
                    rho.size,

                "rho_positive":
                    int(
                        np.sum(
                            rho > 0.0
                        )
                    ),

                "rho_negative":
                    int(
                        np.sum(
                            rho < 0.0
                        )
                    ),

                "rho_abs_ge_0.1":
                    int(
                        np.sum(
                            np.abs(rho)
                            >= 0.1
                        )
                    ),

                "rho_abs_ge_0.3":
                    int(
                        np.sum(
                            np.abs(rho)
                            >= 0.3
                        )
                    ),

                "rho_abs_ge_0.5":
                    int(
                        np.sum(
                            np.abs(rho)
                            >= 0.5
                        )
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def summarize_directional_controls(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    directional = dataframe[
        dataframe[
            "centroid_variable"
        ].isin(
            [
                "x",
                "y",
            ]
        )
    ].copy()

    summary = (
        directional
        .groupby(
            [
                "regime",
                "centroid_variable",
            ],
            sort=False,
        )[
            "spearman_rho"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "count":
                "n_scenarios",

            "mean":
                "rho_mean",

            "median":
                "rho_median",

            "std":
                "rho_std",

            "min":
                "rho_min",

            "max":
                "rho_max",
        }
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print()
    print(
        "=" * 90
    )
    print(
        "BEAM WANDER -- OAM ANALYSIS"
    )
    print(
        "=" * 90
    )

    for beam_name in BEAMS:

        for psd_name in PSDS:

            for regime_name in REGIMES:

                print(
                    f"Analyzing "
                    f"{beam_name:4s} | "
                    f"{psd_name:20s} | "
                    f"{regime_name}"
                )

                scenario_records = (
                    analyze_scenario(
                        beam_name=beam_name,
                        psd_name=psd_name,
                        regime_name=regime_name,
                    )
                )

                records.extend(
                    scenario_records
                )

    correlations = pd.DataFrame(
        records
    )

    # ========================================================
    # Save complete correlation table
    # ========================================================

    correlations_file = (
        ANALYSIS_DIRECTORY
        / "beam_wander_oam_correlations.csv"
    )

    correlations.to_csv(
        correlations_file,
        index=False,
    )

    # ========================================================
    # Summary by turbulence regime
    # ========================================================

    regime_summary = (
        summarize_radial_correlations(
            correlations
        )
    )

    regime_summary_file = (
        ANALYSIS_DIRECTORY
        / "beam_wander_oam_summary_by_regime.csv"
    )

    regime_summary.to_csv(
        regime_summary_file,
        index=False,
    )

    # ========================================================
    # Sign consistency
    # ========================================================

    sign_summary = (
        summarize_sign_consistency(
            correlations
        )
    )

    sign_summary_file = (
        ANALYSIS_DIRECTORY
        / "beam_wander_oam_sign_consistency.csv"
    )

    sign_summary.to_csv(
        sign_summary_file,
        index=False,
    )

    # ========================================================
    # Directional controls
    # ========================================================

    directional_summary = (
        summarize_directional_controls(
            correlations
        )
    )

    directional_file = (
        ANALYSIS_DIRECTORY
        / "beam_wander_signed_asymmetry_directional.csv"
    )

    directional_summary.to_csv(
        directional_file,
        index=False,
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print(
        "=" * 90
    )
    print(
        "RADIAL BEAM-WANDER CORRELATIONS BY REGIME"
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
        "=" * 90
    )
    print(
        "SIGN CONSISTENCY"
    )
    print(
        "=" * 90
    )

    print(
        sign_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "DIRECTIONAL CONTROL: x/y vs SIGNED ASYMMETRY"
    )
    print(
        "=" * 90
    )

    print(
        directional_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Files saved in:"
    )
    print(
        ANALYSIS_DIRECTORY.resolve()
    )


if __name__ == "__main__":
    main()
