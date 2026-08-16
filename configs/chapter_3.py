"""
Shared configuration for Chapter 3 phase-screen validation.

This module contains the numerical and turbulence parameters shared
by the atmospheric phase-screen experiments.
"""


# ============================================================
# Numerical grid
# ============================================================

N_GRID = 512

WINDOW_SIZE = 0.4  # [m]

DX = WINDOW_SIZE / N_GRID  # [m]


# ============================================================
# Turbulence strength
# ============================================================

# Fried parameter used in the original Chapter 3 validation
# experiments.
#
# This value is retained here to reproduce the Chapter 3
# validation results. The final turbulence parameters used for
# cross-PSD comparisons will later be adjusted so that the
# different spectra represent equivalent turbulence strengths.

R0 = 6.7e-3  # [m]


# ============================================================
# von Kármán parameters
# ============================================================

OUTER_SCALE = 10.0  # L0 [m]


# ============================================================
# Modified von Kármán parameters
# ============================================================

INNER_SCALE = 5.0e-3  # l0 [m]


# ============================================================
# Reproducibility
# ============================================================

DEFAULT_SEED = 12345

# ============================================================
# Subharmonic levels selected from convergence analysis
# ============================================================

# Convergence criterion:
#
#     Delta_b < 1 %
#
# where Delta_b measures the relative change between the
# structure functions obtained with b and b-1 subharmonic levels.

KOLMOGOROV_SUBHARMONIC_LEVEL = 9
VON_KARMAN_SUBHARMONIC_LEVEL = 4
MODIFIED_VON_KARMAN_SUBHARMONIC_LEVEL = 4
