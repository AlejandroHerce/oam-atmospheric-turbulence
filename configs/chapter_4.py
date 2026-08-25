"""
Shared configuration for Chapter 4 integrated propagation validation.
"""

from configs.chapter_2 import (
    L_WINDOW,
    N_GRID,
    WAVELENGTH,
)

from configs.chapter_3 import (
    KOLMOGOROV_SUBHARMONIC_LEVEL,
)


# ============================================================
# Numerical grid
# ============================================================

DX = L_WINDOW / N_GRID


# ============================================================
# Propagation geometry
# ============================================================

TOTAL_PROPAGATION_DISTANCE = 1000.0  # [m]

NUMBER_OF_PHASE_SCREENS = 16

SCREEN_SPACING = (
    TOTAL_PROPAGATION_DISTANCE
    / NUMBER_OF_PHASE_SCREENS
)  # [m]

HALF_SCREEN_SPACING = (
    SCREEN_SPACING / 2.0
)  # [m]


# ============================================================
# Turbulence regimes
# ============================================================

# Fried parameters integrated over the complete 1000 m path.

WEAK_R0_TOTAL = 0.1066       # [m]
MODERATE_R0_TOTAL = 0.0268   # [m]
STRONG_R0_TOTAL = 0.0067     # [m]


def segment_fried_parameter(
    total_r0: float,
    number_of_screens: int,
) -> float:
    """
    Convert the Fried parameter of the complete homogeneous path
    into the Fried parameter represented by each equal segment.

        r0_screen = r0_total * N_screens^(3/5)
    """

    if total_r0 <= 0.0:
        raise ValueError(
            "total_r0 must be positive."
        )

    if number_of_screens <= 0:
        raise ValueError(
            "number_of_screens must be positive."
        )

    return (
        total_r0
        * number_of_screens ** (3.0 / 5.0)
    )


WEAK_R0_SCREEN = segment_fried_parameter(
    WEAK_R0_TOTAL,
    NUMBER_OF_PHASE_SCREENS,
)

MODERATE_R0_SCREEN = segment_fried_parameter(
    MODERATE_R0_TOTAL,
    NUMBER_OF_PHASE_SCREENS,
)

STRONG_R0_SCREEN = segment_fried_parameter(
    STRONG_R0_TOTAL,
    NUMBER_OF_PHASE_SCREENS,
)


# ============================================================
# Ensemble-convergence test
# ============================================================

ENSEMBLE_REFERENCE_SIZE = 2000

ENSEMBLE_CHECKPOINTS = (
    25,
    50,
    100,
    150,
    200,
    300,
    400,
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    2000,
)

ENSEMBLE_SEED = 20260816


# ============================================================
# Test beam
# ============================================================

ENSEMBLE_BEAM_FAMILY = "LG"
ENSEMBLE_BEAM_CHARGE = 3


# ============================================================
# Turbulence model
# ============================================================

ENSEMBLE_TURBULENCE_MODEL = "kolmogorov"

ENSEMBLE_R0_SCREEN = STRONG_R0_SCREEN

ENSEMBLE_SUBHARMONIC_LEVEL = (
    KOLMOGOROV_SUBHARMONIC_LEVEL
)


# ============================================================
# OAM analysis
# ============================================================

OAM_ELL_MIN = -240
OAM_ELL_MAX = 240

# ============================================================
# Weak-turbulence Rytov validation
# ============================================================

WEAK_CN2 = 1.0e-15  # [m^(-2/3)]

RYTOV_NUMBER_OF_REALIZATIONS = 1750

RYTOV_R0_SCREEN = WEAK_R0_SCREEN

RYTOV_SUBHARMONIC_LEVEL = (
    KOLMOGOROV_SUBHARMONIC_LEVEL
)

RYTOV_NUMBER_OF_OBSERVATION_PLANES = (
    NUMBER_OF_PHASE_SCREENS + 1
)
# ============================================================
# Bootstrap uncertainty for Rytov validation
# ============================================================

RYTOV_BOOTSTRAP_SAMPLES = 10_000

RYTOV_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

RYTOV_BOOTSTRAP_SEED = 20260820

# ============================================================
# Moderate/strong split-step convergence
# ============================================================

SCREEN_CONVERGENCE_LEVELS = (
    8,
    16,
    32,
    64,
)

SCREEN_CONVERGENCE_NUMBER_OF_REALIZATIONS = 1000

SCREEN_CONVERGENCE_SEED = 20260820

SCREEN_CONVERGENCE_SUBHARMONIC_LEVEL = (
    KOLMOGOROV_SUBHARMONIC_LEVEL
)

SCREEN_CONVERGENCE_BOOTSTRAP_SAMPLES = 10_000

SCREEN_CONVERGENCE_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

SCREEN_CONVERGENCE_BOOTSTRAP_SEED = 20260821

# ============================================================
# Weak-turbulence beam-wander validation
# ============================================================

BEAM_WANDER_NUMBER_OF_REALIZATIONS = 1750

BEAM_WANDER_R0_SCREEN = WEAK_R0_SCREEN

BEAM_WANDER_SUBHARMONIC_LEVEL = (
    KOLMOGOROV_SUBHARMONIC_LEVEL
)

BEAM_WANDER_BOOTSTRAP_SAMPLES = 10_000

BEAM_WANDER_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

BEAM_WANDER_BOOTSTRAP_SEED = 20260824
