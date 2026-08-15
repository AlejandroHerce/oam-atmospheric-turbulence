"""Shared numerical and beam parameters established in Chapter 2."""

from src.beams import BeamDefinition

WAVELENGTH = 632.8e-9
N_GRID = 512
L_WINDOW = 0.4
SCREEN_SPACING = 62.5
TOTAL_DISTANCE = 1000.0

W0_GAUSSIAN = 0.025
W0_LG = 0.025

BG_MATCHING_Q = 3.0
BG_WINDOW_ALPHA = 1.2
BG_WINDOW_RING = 3
# Rounded values used in the Chapter 2 propagation experiments.
# The matching experiment reports the corresponding high-precision values.
BG_PARAMETERS = {
    1: {"w0": 0.0392, "kr": 76.46},
    2: {"w0": 0.0352, "kr": 85.34},
    3: {"w0": 0.0326, "kr": 91.91},
}

OAM_ELL_MIN = -8
OAM_ELL_MAX = 8
OAM_RADIAL_SAMPLES = 256
OAM_AZIMUTHAL_SAMPLES = 720

BEAMS = (
    BeamDefinition("Gaussian", 0, W0_GAUSSIAN, name="Gaussian"),
    BeamDefinition("LG", 1, W0_LG, name="LG_0^1"),
    BeamDefinition("LG", 2, W0_LG, name="LG_0^2"),
    BeamDefinition("LG", 3, W0_LG, name="LG_0^3"),
    BeamDefinition("BG", 1, BG_PARAMETERS[1]["w0"], BG_PARAMETERS[1]["kr"], "BG^1"),
    BeamDefinition("BG", 2, BG_PARAMETERS[2]["w0"], BG_PARAMETERS[2]["kr"], "BG^2"),
    BeamDefinition("BG", 3, BG_PARAMETERS[3]["w0"], BG_PARAMETERS[3]["kr"], "BG^3"),
)
