from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

RESULTS_ROOT = Path(
    "results/chapter_5"
)

CONTROL_ROOT = (
    RESULTS_ROOT
    / "controls"
    / "charge_inversion"
    / "kolmogorov"
)

OUTPUT_DIRECTORY = (
    RESULTS_ROOT
    / "analysis"
    / "charge_inversion"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

REGIMES = (
    "weak",
    "moderate",
    "strong",
)

ORDERS = (
    1,
    2,
    3,
)

NUMBER_OF_REALIZATIONS = 500


# ============================================================
# Loading
# ============================================================

def load_positive_spectra(
    regime: str,
    order: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    filename = (
        RESULTS_ROOT
        / "kolmogorov"
        / regime
        / f"LG{order:02d}"
        / "oam_spectra.npz"
    )

    with np.load(
        filename
    ) as data:

        ell = np.asarray(
            data["ell_values"],
            dtype=np.int64,
        )

        spectra = np.asarray(
            data["modal_power"][
                :NUMBER_OF_REALIZATIONS
            ],
            dtype=np.float64,
        )

    return (
        ell,
        spectra,
    )


def load_negative_spectra(
    regime: str,
    order: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    filename = (
        CONTROL_ROOT
        / regime
        / f"LGm{order:02d}"
        / "oam_spectra.npz"
    )

    with np.load(
        filename
    ) as data:

        ell = np.asarray(
            data["ell_values"],
            dtype=np.int64,
        )

        spectra = np.asarray(
            data["modal_power"],
            dtype=np.float64,
        )

    if spectra.shape[0] != NUMBER_OF_REALIZATIONS:

        raise RuntimeError(
            f"Expected {NUMBER_OF_REALIZATIONS} "
            f"negative realizations, got "
            f"{spectra.shape[0]}."
        )

    return (
        ell,
        spectra,
    )


# ============================================================
# Mean spectrum
# ============================================================

def mean_normalized_spectrum(
    spectra: np.ndarray,
) -> np.ndarray:

    spectrum = np.mean(
        spectra,
        axis=0,
    )

    total = float(
        np.sum(
            spectrum
        )
    )

    if total <= 0.0:

        raise RuntimeError(
            "Invalid mean spectrum."
        )

    return (
        spectrum
        / total
    )


# ============================================================
# Spectral asymmetry
# ============================================================

def calculate_asymmetry(
    ell: np.ndarray,
    spectrum: np.ndarray,
    ell0: int,
) -> tuple[
    float,
    float,
]:

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

        difference = (
            lookup[
                ell0 + delta
            ]
            -
            lookup[
                ell0 - delta
            ]
        )

        signed += (
            difference
        )

        absolute += abs(
            difference
        )

    return (
        float(
            signed
        ),
        float(
            absolute
        ),
    )


# ============================================================
# Safe relative mismatch
# ============================================================

def symmetric_relative_error(
    first: float,
    second: float,
) -> float:

    denominator = (
        0.5
        * (
            abs(first)
            + abs(second)
        )
    )

    if denominator == 0.0:

        return 0.0

    return float(
        abs(
            first
            - second
        )
        / denominator
    )


# ============================================================
# One comparison
# ============================================================

def analyze_case(
    regime: str,
    order: int,
) -> dict:

    (
        ell_positive,
        positive_spectra,
    ) = load_positive_spectra(
        regime=regime,
        order=order,
    )

    (
        ell_negative,
        negative_spectra,
    ) = load_negative_spectra(
        regime=regime,
        order=order,
    )

    if not np.array_equal(
        ell_positive,
        ell_negative,
    ):

        raise RuntimeError(
            "Positive and negative OAM grids differ."
        )

    ell = (
        ell_positive
    )

    positive_mean = (
        mean_normalized_spectrum(
            positive_spectra
        )
    )

    negative_mean = (
        mean_normalized_spectrum(
            negative_spectra
        )
    )

    (
        signed_positive,
        absolute_positive,
    ) = calculate_asymmetry(
        ell=ell,
        spectrum=positive_mean,
        ell0=order,
    )

    (
        signed_negative,
        absolute_negative,
    ) = calculate_asymmetry(
        ell=ell,
        spectrum=negative_mean,
        ell0=-order,
    )

    # --------------------------------------------------------
    # Charge-inversion diagnostics
    # --------------------------------------------------------

    signed_sum = (
        signed_positive
        + signed_negative
    )

    signed_inversion_error = (
        symmetric_relative_error(
            signed_positive,
            -signed_negative,
        )
    )

    absolute_relative_error = (
        symmetric_relative_error(
            absolute_positive,
            absolute_negative,
        )
    )

    # Because ell runs symmetrically from -240 to +240,
    # reversing the array implements ell -> -ell.
    mirrored_negative = (
        negative_mean[
            ::-1
        ]
    )

    mirror_l1 = float(
        np.sum(
            np.abs(
                positive_mean
                - mirrored_negative
            )
        )
    )

    return {
        "regime":
            regime,

        "order":
            order,

        "signed_positive":
            signed_positive,

        "signed_negative":
            signed_negative,

        "signed_sum":
            signed_sum,

        "signed_inversion_error":
            signed_inversion_error,

        "absolute_positive":
            absolute_positive,

        "absolute_negative":
            absolute_negative,

        "absolute_relative_error":
            absolute_relative_error,

        "mirror_l1":
            mirror_l1,
    }


# ============================================================
# CSV
# ============================================================

def write_csv(
    filename: Path,
    records: list[dict],
) -> None:

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
# Terminal
# ============================================================

def print_results(
    records: list[dict],
) -> None:

    print()
    print(
        "=" * 132
    )

    print(
        "CONTROL DE INVERSIÓN DE CARGA OAM"
    )

    print(
        "=" * 132
    )

    print(
        f"{'régimen':>10} "
        f"{'|ell|':>6} "
        f"{'As(+)':>11} "
        f"{'As(-)':>11} "
        f"{'As(+)+As(-)':>14} "
        f"{'err As [%]':>12} "
        f"{'Aabs(+)':>11} "
        f"{'Aabs(-)':>11} "
        f"{'err Aabs [%]':>14} "
        f"{'L1 espejo':>11}"
    )

    print(
        "-" * 132
    )

    for row in records:

        print(
            f"{row['regime']:>10} "
            f"{row['order']:6d} "
            f"{row['signed_positive']:11.6f} "
            f"{row['signed_negative']:11.6f} "
            f"{row['signed_sum']:14.6e} "
            f"{100.0 * row['signed_inversion_error']:12.3f} "
            f"{row['absolute_positive']:11.6f} "
            f"{row['absolute_negative']:11.6f} "
            f"{100.0 * row['absolute_relative_error']:14.3f} "
            f"{row['mirror_l1']:11.6f}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    records = []

    for regime in REGIMES:

        for order in ORDERS:

            record = (
                analyze_case(
                    regime=regime,
                    order=order,
                )
            )

            records.append(
                record
            )

    write_csv(
        OUTPUT_DIRECTORY
        / "charge_inversion_summary.csv",
        records,
    )

    print_results(
        records
    )

    print()
    print(
        "Resultados guardados en:"
    )
    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":

    main()
