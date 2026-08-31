"""
Final exploratory analysis for Chapter 5.

This script reads the definitive Chapter 5 simulation results and
performs cross-scenario analyses without rerunning propagation.

Main analyses
-------------
1. Scenario-level summary.
2. Pairwise PSD differences with bootstrap confidence intervals.
3. LG vs BG differences at matched OAM order.
4. Dependence on transmitted OAM order.
5. Correlation between beam wander and modal degradation.
6. Ensemble entropy gap:
       H(mean spectrum) - mean[H(realization spectrum)].
7. Spectral asymmetry around the transmitted OAM mode.
8. Automatic ranking of the largest PSD and family effects.

No thesis figures are generated here. The purpose of this script is
to identify which comparisons are scientifically relevant before
designing the final visualizations.
"""

import argparse
import csv
import json

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


# ============================================================
# Paths and configuration
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

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 20260829


# ============================================================
# Basic utilities
# ============================================================

def beam_family(
    beam: str,
) -> str:

    if beam.startswith("LG"):
        return "LG"

    if beam.startswith("BG"):
        return "BG"

    raise ValueError(
        f"Unknown beam: {beam}"
    )


def transmitted_order(
    beam: str,
) -> int:

    return int(
        beam[-2:]
    )


def scenario_directory(
    beam: str,
    psd: str,
    regime: str,
) -> Path:

    return (
        RESULTS_ROOT
        / psd
        / regime
        / beam
    )


# ============================================================
# Data loading
# ============================================================

def load_metadata(
    directory: Path,
) -> dict:

    with (
        directory
        / "metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_metrics(
    directory: Path,
) -> dict:
    """
    Load realization-level scalar metrics.
    """

    filename = (
        directory
        / "metrics.csv"
    )

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    data = np.atleast_1d(
        data
    )

    return {
        "retention":
            np.asarray(
                data["retention"],
                dtype=np.float64,
            ),

        "spread":
            np.asarray(
                data["oam_rms_spread"],
                dtype=np.float64,
            ),

        "entropy":
            np.asarray(
                data["normalized_oam_entropy"],
                dtype=np.float64,
            ),

        "centroid_x":
            np.asarray(
                data["centroid_x_m"],
                dtype=np.float64,
            ),

        "centroid_y":
            np.asarray(
                data["centroid_y_m"],
                dtype=np.float64,
            ),

        "centroid_radius":
            np.asarray(
                data["centroid_radius_m"],
                dtype=np.float64,
            ),

        "total_power":
            np.asarray(
                data["total_power"],
                dtype=np.float64,
            ),
    }


def load_spectra(
    directory: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load realization-level OAM spectra.
    """

    with np.load(
        directory
        / "oam_spectra.npz"
    ) as archive:

        ell = np.asarray(
            archive["ell_values"],
            dtype=np.int64,
        )

        spectra = np.asarray(
            archive["modal_power"],
            dtype=np.float64,
        )

    return (
        ell,
        spectra,
    )


def load_scenario(
    beam: str,
    psd: str,
    regime: str,
) -> dict:

    directory = scenario_directory(
        beam=beam,
        psd=psd,
        regime=regime,
    )

    metadata = load_metadata(
        directory
    )

    metrics = load_metrics(
        directory
    )

    (
        ell,
        spectra,
    ) = load_spectra(
        directory
    )

    if spectra.shape[0] != metrics["retention"].size:
        raise RuntimeError(
            f"Inconsistent ensemble size in {directory}"
        )

    return {
        "beam":
            beam,

        "family":
            beam_family(
                beam
            ),

        "order":
            transmitted_order(
                beam
            ),

        "psd":
            psd,

        "regime":
            regime,

        "metadata":
            metadata,

        "metrics":
            metrics,

        "ell":
            ell,

        "spectra":
            spectra,
    }


# ============================================================
# Bootstrap helpers
# ============================================================

def percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[
    float,
    float,
]:

    alpha = (
        1.0
        - confidence_level
    )

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def bootstrap_mean(
    samples: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Bootstrap CI for a sample mean.
    """

    n = samples.size

    bootstrap_values = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_values[index] = float(
            np.mean(
                samples[
                    indices
                ]
            )
        )

    lower, upper = percentile_interval(
        bootstrap_values,
        BOOTSTRAP_CONFIDENCE_LEVEL,
    )

    return (
        float(
            np.mean(samples)
        ),
        lower,
        upper,
    )


def bootstrap_difference_of_means(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Bootstrap CI for

        mean(A) - mean(B).

    The two scenario ensembles are resampled independently.
    """

    n_a = samples_a.size
    n_b = samples_b.size

    observed = float(
        np.mean(samples_a)
        - np.mean(samples_b)
    )

    bootstrap_values = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices_a = rng.integers(
            0,
            n_a,
            size=n_a,
        )

        indices_b = rng.integers(
            0,
            n_b,
            size=n_b,
        )

        bootstrap_values[index] = (
            np.mean(
                samples_a[
                    indices_a
                ]
            )
            - np.mean(
                samples_b[
                    indices_b
                ]
            )
        )

    lower, upper = percentile_interval(
        bootstrap_values,
        BOOTSTRAP_CONFIDENCE_LEVEL,
    )

    return (
        observed,
        lower,
        upper,
    )


# ============================================================
# Beam wander
# ============================================================

def calculate_beam_wander(
    metrics: dict,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                metrics["centroid_x"] ** 2
                + metrics["centroid_y"] ** 2
            )
        )
    )


# ============================================================
# Mean-spectrum metrics
# ============================================================

def normalized_entropy(
    spectrum: np.ndarray,
) -> float:

    spectrum = (
        spectrum
        / np.sum(spectrum)
    )

    positive = (
        spectrum > 0.0
    )

    entropy = float(
        -np.sum(
            spectrum[
                positive
            ]
            * np.log2(
                spectrum[
                    positive
                ]
            )
        )
    )

    return float(
        entropy
        / np.log2(
            spectrum.size
        )
    )


def mean_spectrum_metrics(
    scenario: dict,
) -> dict:

    ell = (
        scenario["ell"]
    )

    spectra = (
        scenario["spectra"]
    )

    ell0 = (
        scenario["order"]
    )

    mean_spectrum = np.mean(
        spectra,
        axis=0,
    )

    mean_spectrum /= np.sum(
        mean_spectrum
    )

    transmitted_index = np.where(
        ell == ell0
    )[0]

    if transmitted_index.size != 1:
        raise RuntimeError(
            "Transmitted OAM index not found."
        )

    retention = float(
        mean_spectrum[
            transmitted_index[0]
        ]
    )

    spread = float(
        np.sqrt(
            np.sum(
                (
                    ell
                    - ell0
                ) ** 2
                * mean_spectrum
            )
        )
    )

    entropy = normalized_entropy(
        mean_spectrum
    )

    return {
        "retention":
            retention,

        "spread":
            spread,

        "entropy":
            entropy,

        "mean_spectrum":
            mean_spectrum,
    }


# ============================================================
# Spectral asymmetry
# ============================================================

def calculate_spectral_asymmetry(
    ell: np.ndarray,
    spectrum: np.ndarray,
    ell0: int,
) -> tuple[
    float,
    float,
]:
    """
    Calculate two complementary asymmetry measures.

    signed_asymmetry:
        Sum[P(ell0+d) - P(ell0-d)].

    absolute_asymmetry:
        Sum[|P(ell0+d) - P(ell0-d)|].

    Only modal pairs present on both sides of ell0 are used.
    """

    lookup = {
        int(mode): float(power)
        for mode, power in zip(
            ell,
            spectrum,
        )
    }

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

    signed = 0.0
    absolute = 0.0

    for delta in range(
        1,
        maximum_delta + 1,
    ):

        positive = lookup[
            ell0 + delta
        ]

        negative = lookup[
            ell0 - delta
        ]

        difference = (
            positive
            - negative
        )

        signed += difference
        absolute += abs(
            difference
        )

    return (
        float(signed),
        float(absolute),
    )


# ============================================================
# Scenario summary
# ============================================================

def build_scenario_summary(
    scenario: dict,
) -> dict:

    metrics = (
        scenario["metrics"]
    )

    mean_metrics = (
        mean_spectrum_metrics(
            scenario
        )
    )

    mean_spectrum = (
        mean_metrics[
            "mean_spectrum"
        ]
    )

    (
        signed_asymmetry,
        absolute_asymmetry,
    ) = calculate_spectral_asymmetry(
        ell=scenario["ell"],
        spectrum=mean_spectrum,
        ell0=scenario["order"],
    )

    entropy_mean_realization = float(
        np.mean(
            metrics[
                "entropy"
            ]
        )
    )

    entropy_of_mean = (
        mean_metrics[
            "entropy"
        ]
    )

    entropy_gap = float(
        entropy_of_mean
        - entropy_mean_realization
    )

    return {
        "beam":
            scenario["beam"],

        "family":
            scenario["family"],

        "order":
            scenario["order"],

        "psd":
            scenario["psd"],

        "regime":
            scenario["regime"],

        "n":
            metrics[
                "retention"
            ].size,

        "retention_mean":
            float(
                np.mean(
                    metrics[
                        "retention"
                    ]
                )
            ),

        "spread_mean":
            float(
                np.mean(
                    metrics[
                        "spread"
                    ]
                )
            ),

        "entropy_mean":
            entropy_mean_realization,

        "beam_wander_m":
            calculate_beam_wander(
                metrics
            ),

        "retention_mean_spectrum":
            mean_metrics[
                "retention"
            ],

        "spread_mean_spectrum":
            mean_metrics[
                "spread"
            ],

        "entropy_mean_spectrum":
            entropy_of_mean,

        "entropy_gap":
            entropy_gap,

        "signed_asymmetry":
            signed_asymmetry,

        "absolute_asymmetry":
            absolute_asymmetry,

        "mean_total_power":
            float(
                np.mean(
                    metrics[
                        "total_power"
                    ]
                )
            ),
    }


# ============================================================
# Correlation analysis
# ============================================================

def calculate_correlations(
    scenario: dict,
) -> dict:

    metrics = (
        scenario["metrics"]
    )

    radius = (
        metrics[
            "centroid_radius"
        ]
    )

    retention = (
        metrics[
            "retention"
        ]
    )

    spread = (
        metrics[
            "spread"
        ]
    )

    entropy = (
        metrics[
            "entropy"
        ]
    )

    rho_retention, p_retention = (
        spearmanr(
            radius,
            retention,
        )
    )

    rho_spread, p_spread = (
        spearmanr(
            radius,
            spread,
        )
    )

    rho_entropy, p_entropy = (
        spearmanr(
            radius,
            entropy,
        )
    )

    return {
        "beam":
            scenario["beam"],

        "family":
            scenario["family"],

        "order":
            scenario["order"],

        "psd":
            scenario["psd"],

        "regime":
            scenario["regime"],

        "rho_wander_retention":
            float(
                rho_retention
            ),

        "p_wander_retention":
            float(
                p_retention
            ),

        "rho_wander_spread":
            float(
                rho_spread
            ),

        "p_wander_spread":
            float(
                p_spread
            ),

        "rho_wander_entropy":
            float(
                rho_entropy
            ),

        "p_wander_entropy":
            float(
                p_entropy
            ),
    }


# ============================================================
# PSD comparisons
# ============================================================

def build_psd_comparisons(
    scenarios: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Compare PSDs while holding beam and regime fixed.
    """

    comparisons = []

    pairs = (
        (
            "von_karman",
            "kolmogorov",
        ),
        (
            "modified_von_karman",
            "von_karman",
        ),
        (
            "modified_von_karman",
            "kolmogorov",
        ),
    )

    metrics = (
        "retention",
        "spread",
        "entropy",
    )

    for beam in BEAMS:

        for regime in REGIMES:

            for (
                psd_a,
                psd_b,
            ) in pairs:

                scenario_a = scenarios[
                    (
                        beam,
                        psd_a,
                        regime,
                    )
                ]

                scenario_b = scenarios[
                    (
                        beam,
                        psd_b,
                        regime,
                    )
                ]

                for metric in metrics:

                    (
                        difference,
                        lower,
                        upper,
                    ) = bootstrap_difference_of_means(
                        scenario_a[
                            "metrics"
                        ][metric],
                        scenario_b[
                            "metrics"
                        ][metric],
                        rng=rng,
                    )

                    reference = float(
                        np.mean(
                            scenario_b[
                                "metrics"
                            ][metric]
                        )
                    )

                    relative_difference = (
                        100.0
                        * difference
                        / reference
                        if reference != 0.0
                        else np.nan
                    )

                    comparisons.append(
                        {
                            "beam":
                                beam,

                            "family":
                                beam_family(
                                    beam
                                ),

                            "order":
                                transmitted_order(
                                    beam
                                ),

                            "regime":
                                regime,

                            "metric":
                                metric,

                            "psd_a":
                                psd_a,

                            "psd_b":
                                psd_b,

                            "difference_a_minus_b":
                                difference,

                            "relative_difference_percent":
                                relative_difference,

                            "ci95_lower":
                                lower,

                            "ci95_upper":
                                upper,

                            "ci_excludes_zero":
                                (
                                    lower > 0.0
                                    or upper < 0.0
                                ),
                        }
                    )

    return comparisons


# ============================================================
# LG vs BG comparisons
# ============================================================

def build_family_comparisons(
    scenarios: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Compare matched LG and BG modes while keeping PSD,
    turbulence regime, and azimuthal order fixed.
    """

    comparisons = []

    metrics = (
        "retention",
        "spread",
        "entropy",
    )

    for order in (
        1,
        2,
        3,
    ):

        lg = f"LG0{order}"
        bg = f"BG0{order}"

        for psd in PSDS:

            for regime in REGIMES:

                scenario_lg = scenarios[
                    (
                        lg,
                        psd,
                        regime,
                    )
                ]

                scenario_bg = scenarios[
                    (
                        bg,
                        psd,
                        regime,
                    )
                ]

                for metric in metrics:

                    (
                        difference,
                        lower,
                        upper,
                    ) = bootstrap_difference_of_means(
                        scenario_bg[
                            "metrics"
                        ][metric],
                        scenario_lg[
                            "metrics"
                        ][metric],
                        rng=rng,
                    )

                    reference = float(
                        np.mean(
                            scenario_lg[
                                "metrics"
                            ][metric]
                        )
                    )

                    relative_difference = (
                        100.0
                        * difference
                        / reference
                        if reference != 0.0
                        else np.nan
                    )

                    comparisons.append(
                        {
                            "order":
                                order,

                            "psd":
                                psd,

                            "regime":
                                regime,

                            "metric":
                                metric,

                            "difference_BG_minus_LG":
                                difference,

                            "relative_difference_percent":
                                relative_difference,

                            "ci95_lower":
                                lower,

                            "ci95_upper":
                                upper,

                            "ci_excludes_zero":
                                (
                                    lower > 0.0
                                    or upper < 0.0
                                ),
                        }
                    )

    return comparisons


# ============================================================
# Order dependence
# ============================================================

def build_order_comparisons(
    scenarios: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Compare consecutive transmitted OAM orders:
        2 - 1
        3 - 2
    within each family, PSD, and turbulence regime.
    """

    comparisons = []

    metrics = (
        "retention",
        "spread",
        "entropy",
    )

    for family in (
        "LG",
        "BG",
    ):

        for psd in PSDS:

            for regime in REGIMES:

                for (
                    order_a,
                    order_b,
                ) in (
                    (2, 1),
                    (3, 2),
                ):

                    beam_a = (
                        f"{family}0{order_a}"
                    )

                    beam_b = (
                        f"{family}0{order_b}"
                    )

                    scenario_a = scenarios[
                        (
                            beam_a,
                            psd,
                            regime,
                        )
                    ]

                    scenario_b = scenarios[
                        (
                            beam_b,
                            psd,
                            regime,
                        )
                    ]

                    for metric in metrics:

                        (
                            difference,
                            lower,
                            upper,
                        ) = bootstrap_difference_of_means(
                            scenario_a[
                                "metrics"
                            ][metric],
                            scenario_b[
                                "metrics"
                            ][metric],
                            rng=rng,
                        )

                        comparisons.append(
                            {
                                "family":
                                    family,

                                "psd":
                                    psd,

                                "regime":
                                    regime,

                                "metric":
                                    metric,

                                "order_a":
                                    order_a,

                                "order_b":
                                    order_b,

                                "difference_a_minus_b":
                                    difference,

                                "ci95_lower":
                                    lower,

                                "ci95_upper":
                                    upper,

                                "ci_excludes_zero":
                                    (
                                        lower > 0.0
                                        or upper < 0.0
                                    ),
                            }
                        )

    return comparisons


# ============================================================
# CSV writer
# ============================================================

def save_records(
    filename: Path,
    records: list[dict],
) -> None:

    if not records:
        return

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
# Ranking
# ============================================================

def largest_effects(
    comparisons: list[dict],
    difference_key: str,
    number: int = 20,
) -> list[dict]:

    valid = [
        record
        for record in comparisons
        if np.isfinite(
            record[
                difference_key
            ]
        )
    ]

    return sorted(
        valid,
        key=lambda record: abs(
            record[
                difference_key
            ]
        ),
        reverse=True,
    )[
        :number
    ]


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )

    return parser.parse_args()


def bootstrap_mean_spectrum_asymmetry(
    scenario: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Bootstrap confidence intervals for the signed and absolute
    asymmetry of the ensemble-averaged OAM spectrum.
    """

    spectra = (
        scenario["spectra"]
    )

    ell = (
        scenario["ell"]
    )

    ell0 = (
        scenario["order"]
    )

    n = (
        spectra.shape[0]
    )

    # --------------------------------------------------------
    # Observed values
    # --------------------------------------------------------

    mean_spectrum = np.mean(
        spectra,
        axis=0,
    )

    mean_spectrum /= np.sum(
        mean_spectrum
    )

    (
        signed_observed,
        absolute_observed,
    ) = calculate_spectral_asymmetry(
        ell=ell,
        spectrum=mean_spectrum,
        ell0=ell0,
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    signed_bootstrap = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    absolute_bootstrap = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_spectrum = np.mean(
            spectra[
                indices
            ],
            axis=0,
        )

        bootstrap_spectrum /= np.sum(
            bootstrap_spectrum
        )

        (
            signed_bootstrap[index],
            absolute_bootstrap[index],
        ) = calculate_spectral_asymmetry(
            ell=ell,
            spectrum=bootstrap_spectrum,
            ell0=ell0,
        )

    signed_lower, signed_upper = percentile_interval(
        signed_bootstrap,
        BOOTSTRAP_CONFIDENCE_LEVEL,
    )

    absolute_lower, absolute_upper = percentile_interval(
        absolute_bootstrap,
        BOOTSTRAP_CONFIDENCE_LEVEL,
    )

    return {
        "beam":
            scenario["beam"],

        "family":
            scenario["family"],

        "order":
            scenario["order"],

        "psd":
            scenario["psd"],

        "regime":
            scenario["regime"],

        "signed_asymmetry":
            signed_observed,

        "signed_ci95_lower":
            signed_lower,

        "signed_ci95_upper":
            signed_upper,

        "signed_ci_excludes_zero":
            (
                signed_lower > 0.0
                or signed_upper < 0.0
            ),

        "absolute_asymmetry":
            absolute_observed,

        "absolute_ci95_lower":
            absolute_lower,

        "absolute_ci95_upper":
            absolute_upper,
    }


def summarize_correlations_by_regime(
    correlation_records: list[dict],
) -> list[dict]:
    """
    Summarize Spearman correlations across the 18 scenarios
    belonging to each turbulence regime.

    Median and interquartile range are used because correlation
    coefficients are bounded and need not be normally distributed.
    """

    metrics = (
        "rho_wander_retention",
        "rho_wander_spread",
        "rho_wander_entropy",
    )

    records = []

    for regime in REGIMES:

        regime_rows = [
            row
            for row in correlation_records
            if row["regime"] == regime
        ]

        for metric in metrics:

            values = np.asarray(
                [
                    row[metric]
                    for row in regime_rows
                ],
                dtype=np.float64,
            )

            records.append(
                {
                    "regime":
                        regime,

                    "correlation":
                        metric,

                    "n_scenarios":
                        values.size,

                    "mean_rho":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "median_rho":
                        float(
                            np.median(
                                values
                            )
                        ),

                    "q25":
                        float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        ),

                    "q75":
                        float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        ),

                    "minimum":
                        float(
                            np.min(
                                values
                            )
                        ),

                    "maximum":
                        float(
                            np.max(
                                values
                            )
                        ),
                }
            )

    return records


def summarize_entropy_gap_by_regime(
    scenario_summary: list[dict],
) -> list[dict]:
    """
    Summarize

        Delta H =
            H(mean spectrum)
            - mean[H(realization spectrum)]

    across all scenarios of each turbulence regime.
    """

    records = []

    for regime in REGIMES:

        values = np.asarray(
            [
                row[
                    "entropy_gap"
                ]
                for row in scenario_summary
                if row[
                    "regime"
                ] == regime
            ],
            dtype=np.float64,
        )

        records.append(
            {
                "regime":
                    regime,

                "n_scenarios":
                    values.size,

                "mean_entropy_gap":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "median_entropy_gap":
                    float(
                        np.median(
                            values
                        )
                    ),

                "std_entropy_gap":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "minimum":
                    float(
                        np.min(
                            values
                        )
                    ),

                "maximum":
                    float(
                        np.max(
                            values
                        )
                    ),
            }
        )

    return records

# ============================================================
# Main
# ============================================================

def main() -> None:

    global BOOTSTRAP_SAMPLES

    arguments = parse_arguments()

    BOOTSTRAP_SAMPLES = (
        arguments.bootstrap_samples
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    # --------------------------------------------------------
    # Load all 54 scenarios
    # --------------------------------------------------------

    scenarios = {}

    for beam in BEAMS:

        for psd in PSDS:

            for regime in REGIMES:

                scenario = load_scenario(
                    beam=beam,
                    psd=psd,
                    regime=regime,
                )

                scenarios[
                    (
                        beam,
                        psd,
                        regime,
                    )
                ] = scenario

    print(
        f"Scenarios loaded: "
        f"{len(scenarios)}"
    )

    # --------------------------------------------------------
    # Scenario-level summary
    # --------------------------------------------------------

    scenario_summary = [
        build_scenario_summary(
            scenario
        )
        for scenario in scenarios.values()
    ]

    save_records(
        ANALYSIS_DIRECTORY
        / "scenario_summary.csv",
        scenario_summary,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlations = [
        calculate_correlations(
            scenario
        )
        for scenario in scenarios.values()
    ]

    save_records(
        ANALYSIS_DIRECTORY
        / "wander_modal_correlations.csv",
        correlations,
    )

    # --------------------------------------------------------
    # Bootstrap spectral asymmetry
    # --------------------------------------------------------

    asymmetry_statistics = [
        bootstrap_mean_spectrum_asymmetry(
            scenario=scenario,
            rng=rng,
        )
        for scenario in scenarios.values()
    ]

    save_records(
        ANALYSIS_DIRECTORY
        / "spectral_asymmetry_statistics.csv",
        asymmetry_statistics,
    )

    # --------------------------------------------------------
    # Correlation summaries by turbulence regime
    # --------------------------------------------------------

    correlation_summary = (
        summarize_correlations_by_regime(
            correlations
        )
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "correlation_summary_by_regime.csv",
        correlation_summary,
    )

    # --------------------------------------------------------
    # Entropy-gap summaries by turbulence regime
    # --------------------------------------------------------

    entropy_gap_summary = (
        summarize_entropy_gap_by_regime(
            scenario_summary
        )
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "entropy_gap_summary_by_regime.csv",
        entropy_gap_summary,
    )

    # --------------------------------------------------------
    # PSD comparisons
    # --------------------------------------------------------

    psd_comparisons = (
        build_psd_comparisons(
            scenarios=scenarios,
            rng=rng,
        )
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "psd_comparisons.csv",
        psd_comparisons,
    )

    # --------------------------------------------------------
    # LG vs BG
    # --------------------------------------------------------

    family_comparisons = (
        build_family_comparisons(
            scenarios=scenarios,
            rng=rng,
        )
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "lg_bg_comparisons.csv",
        family_comparisons,
    )

    # --------------------------------------------------------
    # Order dependence
    # --------------------------------------------------------

    order_comparisons = (
        build_order_comparisons(
            scenarios=scenarios,
            rng=rng,
        )
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "order_comparisons.csv",
        order_comparisons,
    )

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    largest_psd = largest_effects(
        psd_comparisons,
        difference_key=(
            "relative_difference_percent"
        ),
        number=25,
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "largest_psd_effects.csv",
        largest_psd,
    )

    largest_family = largest_effects(
        family_comparisons,
        difference_key=(
            "relative_difference_percent"
        ),
        number=25,
    )

    save_records(
        ANALYSIS_DIRECTORY
        / "largest_lg_bg_effects.csv",
        largest_family,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print()

    print(
        "Final Chapter 5 exploratory analysis"
    )

    print(
        "===================================="
    )

    print(
        f"Scenario summaries: "
        f"{len(scenario_summary)}"
    )

    print(
        f"PSD comparisons: "
        f"{len(psd_comparisons)}"
    )

    print(
        f"LG/BG comparisons: "
        f"{len(family_comparisons)}"
    )

    print(
        f"Order comparisons: "
        f"{len(order_comparisons)}"
    )

    print()

    print(
        "Analysis saved in:"
    )

    print(
        ANALYSIS_DIRECTORY.resolve()
    )


    print(
        f"Asymmetry analyses: "
        f"{len(asymmetry_statistics)}"
    )

    print(
        f"Correlation regime summaries: "
        f"{len(correlation_summary)}"
    )

    print(
        f"Entropy-gap regime summaries: "
        f"{len(entropy_gap_summary)}"
    )

if __name__ == "__main__":
    main()
