"""
Batch runner for Chapter 5 production simulations.

This script orchestrates the complete Chapter 5 scenario matrix
without duplicating any simulation physics.

Each scenario is defined by

    beam + PSD + turbulence regime

and is delegated to

    experiments.chapter_5.run_scenario

The batch runner can:

    - run missing scenarios;
    - extend incomplete scenarios;
    - skip completed scenarios;
    - filter by beam, PSD, or regime;
    - perform a dry run without executing simulations.
"""

import argparse
import json
import subprocess
import sys

from pathlib import Path


# ============================================================
# Scenario space
# ============================================================

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

DEFAULT_TARGET_REALIZATIONS = 1750
DEFAULT_NUMBER_OF_WORKERS = 12

RESULTS_ROOT = Path(
    "results/chapter_5"
)


# ============================================================
# Scenario utilities
# ============================================================

def scenario_directory(
    beam: str,
    psd: str,
    regime: str,
) -> Path:
    """
    Return the output directory associated with one scenario.
    """

    return (
        RESULTS_ROOT
        / psd
        / regime
        / beam
    )


def metadata_path(
    beam: str,
    psd: str,
    regime: str,
) -> Path:
    """
    Return the metadata file associated with one scenario.
    """

    return (
        scenario_directory(
            beam=beam,
            psd=psd,
            regime=regime,
        )
        / "metadata.json"
    )


def current_ensemble_size(
    beam: str,
    psd: str,
    regime: str,
) -> int:
    """
    Determine the current ensemble size from metadata.json.

    Returns
    -------
    int
        Number of saved realizations.

        Returns zero when the scenario does not exist yet.
    """

    filename = metadata_path(
        beam=beam,
        psd=psd,
        regime=regime,
    )

    if not filename.exists():
        return 0

    with filename.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if (
        "number_of_realizations"
        not in metadata
    ):
        raise RuntimeError(
            f"Missing number_of_realizations in {filename}"
        )

    number_of_realizations = int(
        metadata[
            "number_of_realizations"
        ]
    )

    if number_of_realizations < 0:
        raise RuntimeError(
            f"Invalid ensemble size in {filename}"
        )

    return number_of_realizations


# ============================================================
# Scenario integrity
# ============================================================

def scenario_files_complete(
    beam: str,
    psd: str,
    regime: str,
) -> bool:
    """
    Check whether the principal scenario output files exist.
    """

    directory = scenario_directory(
        beam=beam,
        psd=psd,
        regime=regime,
    )

    required_files = (
        "metadata.json",
        "metrics.csv",
        "summary.csv",
        "mean_oam_spectrum.csv",
        "oam_spectra.npz",
    )

    return all(
        (
            directory
            / filename
        ).exists()
        for filename in required_files
    )


# ============================================================
# Scenario selection
# ============================================================

def selected_values(
    requested: str | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Return either one requested value or the complete set.
    """

    if requested is None:
        return available

    return (
        requested,
    )


def build_scenario_list(
    beam_filter: str | None,
    psd_filter: str | None,
    regime_filter: str | None,
) -> list[
    tuple[
        str,
        str,
        str,
    ]
]:
    """
    Build the filtered Chapter 5 scenario matrix.
    """

    beams = selected_values(
        beam_filter,
        BEAMS,
    )

    psds = selected_values(
        psd_filter,
        PSDS,
    )

    regimes = selected_values(
        regime_filter,
        REGIMES,
    )

    scenarios = []

    for beam in beams:
        for psd in psds:
            for regime in regimes:

                scenarios.append(
                    (
                        beam,
                        psd,
                        regime,
                    )
                )

    return scenarios


# ============================================================
# Command construction
# ============================================================

def build_command(
    beam: str,
    psd: str,
    regime: str,
    current_size: int,
    target_size: int,
    workers: int,
) -> list[str]:
    """
    Construct the run_scenario command for one scenario.
    """

    command = [
        sys.executable,
        "-m",
        "experiments.chapter_5.run_scenario",
        "--beam",
        beam,
        "--psd",
        psd,
        "--regime",
        regime,
        "--workers",
        str(
            workers
        ),
    ]

    if current_size == 0:

        command.extend(
            [
                "--realizations",
                str(
                    target_size
                ),
            ]
        )

    else:

        command.extend(
            [
                "--extend-to",
                str(
                    target_size
                ),
            ]
        )

    return command


# ============================================================
# Execute one scenario
# ============================================================

def run_scenario(
    beam: str,
    psd: str,
    regime: str,
    current_size: int,
    target_size: int,
    workers: int,
    dry_run: bool,
) -> None:
    """
    Execute or extend one scenario.
    """

    command = build_command(
        beam=beam,
        psd=psd,
        regime=regime,
        current_size=current_size,
        target_size=target_size,
        workers=workers,
    )

    print()

    print(
        "Command:"
    )

    print(
        " ".join(
            command
        )
    )

    if dry_run:

        print(
            "Dry run: command not executed."
        )

        return

    completed_process = subprocess.run(
        command,
        check=False,
    )

    if completed_process.returncode != 0:

        raise RuntimeError(
            "Scenario failed: "
            f"{beam}, {psd}, {regime}"
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse batch-runner command-line options.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--beam",
        choices=BEAMS,
        default=None,
        help=(
            "Run only one beam. "
            "If omitted, all beams are included."
        ),
    )

    parser.add_argument(
        "--psd",
        choices=PSDS,
        default=None,
        help=(
            "Run only one PSD. "
            "If omitted, all PSDs are included."
        ),
    )

    parser.add_argument(
        "--regime",
        choices=REGIMES,
        default=None,
        help=(
            "Run only one turbulence regime. "
            "If omitted, all regimes are included."
        ),
    )

    parser.add_argument(
        "--target",
        type=int,
        default=(
            DEFAULT_TARGET_REALIZATIONS
        ),
        help=(
            "Target ensemble size for every selected scenario."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=(
            DEFAULT_NUMBER_OF_WORKERS
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned actions without executing them."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Execute the selected Chapter 5 production campaign.
    """

    arguments = (
        parse_arguments()
    )

    if arguments.target <= 0:
        raise ValueError(
            "--target must be positive."
        )

    if arguments.workers <= 0:
        raise ValueError(
            "--workers must be positive."
        )

    scenarios = build_scenario_list(
        beam_filter=arguments.beam,
        psd_filter=arguments.psd,
        regime_filter=arguments.regime,
    )

    print()

    print(
        "Chapter 5 batch runner"
    )

    print(
        "======================"
    )

    print(
        f"Selected scenarios: "
        f"{len(scenarios)}"
    )

    print(
        f"Target realizations: "
        f"{arguments.target}"
    )

    print(
        f"Workers per scenario: "
        f"{arguments.workers}"
    )

    print(
        f"Dry run: "
        f"{arguments.dry_run}"
    )

    completed_count = 0
    extended_count = 0
    new_count = 0
    skipped_count = 0

    for (
        scenario_index,
        (
            beam,
            psd,
            regime,
        ),
    ) in enumerate(
        scenarios,
        start=1,
    ):

        print()

        print(
            "=" * 72
        )

        print(
            f"Scenario "
            f"{scenario_index}/"
            f"{len(scenarios)}"
        )

        print(
            f"Beam   : {beam}"
        )

        print(
            f"PSD    : {psd}"
        )

        print(
            f"Regime : {regime}"
        )

        current_size = (
            current_ensemble_size(
                beam=beam,
                psd=psd,
                regime=regime,
            )
        )

        files_complete = (
            scenario_files_complete(
                beam=beam,
                psd=psd,
                regime=regime,
            )
        )

        print(
            f"Current realizations: "
            f"{current_size}"
        )

        # ----------------------------------------------------
        # Scenario already complete
        # ----------------------------------------------------

        if (
            current_size
            == arguments.target
            and files_complete
        ):

            print(
                "Status: complete -> skipped"
            )

            skipped_count += 1

            continue

        # ----------------------------------------------------
        # Scenario unexpectedly larger than requested target
        # ----------------------------------------------------

        if (
            current_size
            > arguments.target
        ):

            print(
                "Status: ensemble already exceeds "
                "requested target -> skipped"
            )

            skipped_count += 1

            continue

        # ----------------------------------------------------
        # Metadata exists but output is incomplete
        # ----------------------------------------------------

        if (
            current_size > 0
            and not files_complete
        ):

            raise RuntimeError(
                "Scenario has metadata but incomplete output files: "
                f"{beam}, {psd}, {regime}"
            )

        # ----------------------------------------------------
        # New scenario
        # ----------------------------------------------------

        if current_size == 0:

            print(
                "Status: new scenario"
            )

            new_count += 1

        # ----------------------------------------------------
        # Extend scenario
        # ----------------------------------------------------

        else:

            print(
                "Status: incomplete -> extend"
            )

            extended_count += 1

        run_scenario(
            beam=beam,
            psd=psd,
            regime=regime,
            current_size=current_size,
            target_size=(
                arguments.target
            ),
            workers=arguments.workers,
            dry_run=arguments.dry_run,
        )

        if not arguments.dry_run:

            completed_count += 1

    # ========================================================
    # Final campaign summary
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "Batch summary"
    )

    print(
        "=" * 72
    )

    print(
        f"Selected scenarios : "
        f"{len(scenarios)}"
    )

    print(
        f"New               : "
        f"{new_count}"
    )

    print(
        f"Extended          : "
        f"{extended_count}"
    )

    print(
        f"Already complete  : "
        f"{skipped_count}"
    )

    if not arguments.dry_run:

        print(
            f"Executed this run : "
            f"{completed_count}"
        )


if __name__ == "__main__":
    main()
