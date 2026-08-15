from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import jv, jnp_zeros

from src.grids import Grid


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


# ============================================================
# Beam definition
# ============================================================

@dataclass(frozen=True)
class BeamDefinition:
    """
    Definition of an optical beam.

    Parameters
    ----------
    family:
        Beam family: "Gaussian", "LG", or "BG".
    charge:
        Topological charge ell (or m for BG).
    w0:
        Beam waist [m].
    kr:
        Radial wave number [1/m], required for BG beams.
    name:
        Optional descriptive name.
    """

    family: str
    charge: int
    w0: float
    kr: float | None = None
    name: str | None = None


# ============================================================
# Energy and normalization
# ============================================================

def calculate_energy(
    field: ComplexArray,
    dx: float,
) -> float:
    """
    Calculate the transverse optical energy/power:

        E = ∫∫ |U(x,y)|² dx dy.
    """

    return float(
        np.sum(np.abs(field) ** 2) * dx**2
    )


def normalize_field(
    field: ComplexArray,
    dx: float,
) -> ComplexArray:
    """
    Normalize a field to unit transverse energy.
    """

    energy = calculate_energy(
        field,
        dx,
    )

    if not np.isfinite(energy) or energy <= 0:
        raise ValueError(
            "The field has an invalid energy."
        )

    return (
        field / np.sqrt(energy)
    ).astype(np.complex128)


# ============================================================
# Laguerre-Gauss beam size
# ============================================================

def lg_second_moment_radius(
    charge: int,
    w0: float,
) -> float:
    """
    Calculate the second-moment radius of an LG beam with p = 0.

        W_LG = w0 * sqrt(|ell| + 1)
    """

    return (
        w0
        * np.sqrt(abs(charge) + 1)
    )


# ============================================================
# Bessel-Gauss beam size
# ============================================================

def bessel_gaussian_intensity(
    r: float,
    charge: int,
    kr: float,
    w0: float,
) -> float:
    """
    Radial intensity profile of a Bessel-Gauss beam.

        I(r) = J_m²(k_r r) exp(-2 r²/w0²)
    """

    return (
        jv(abs(charge), kr * r) ** 2
        * np.exp(-2.0 * r**2 / w0**2)
    )


def bg_second_moment_radius(
    charge: int,
    kr: float,
    w0: float,
) -> float:
    """
    Calculate the second-moment radius of a Bessel-Gauss beam.
    """

    numerator, _ = quad(
        lambda r: (
            r**3
            * bessel_gaussian_intensity(
                r,
                charge,
                kr,
                w0,
            )
        ),
        0.0,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=300,
    )

    denominator, _ = quad(
        lambda r: (
            r
            * bessel_gaussian_intensity(
                r,
                charge,
                kr,
                w0,
            )
        ),
        0.0,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=300,
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid Bessel-Gauss intensity integral."
        )

    return np.sqrt(
        2.0 * numerator / denominator
    )


# ============================================================
# BG-LG matching
# ============================================================

def match_bg_waist_to_lg(
    charge: int,
    lg_waist: float,
    q: float,
) -> float:
    """
    Determine the Bessel-Gauss waist by matching its
    second-moment radius to the corresponding LG beam.

    The constraint is

        q = k_r w0_BG.
    """

    target_radius = lg_second_moment_radius(
        charge,
        lg_waist,
    )

    def objective(
        w0_bg: float,
    ) -> float:

        kr = q / w0_bg

        return (
            bg_second_moment_radius(
                charge,
                kr,
                w0_bg,
            )
            - target_radius
        )

    return brentq(
        objective,
        1e-3,
        0.20,
        xtol=1e-12,
        rtol=1e-12,
    )


def calculate_bg_parameters(
    charge: int,
    lg_waist: float,
    q: float,
) -> tuple[float, float]:
    """
    Calculate matched Bessel-Gauss parameters.

    Returns
    -------
    w0_bg:
        Bessel-Gauss waist [m].

    kr:
        Radial wave number [1/m].
    """

    w0_bg = match_bg_waist_to_lg(
        charge,
        lg_waist,
        q,
    )

    kr = q / w0_bg

    return w0_bg, kr


# ============================================================
# Computational window
# ============================================================

def gaussian_waist_at_z(
    w0: float,
    wavelength: float,
    z: float,
) -> float:
    """
    Gaussian beam waist after propagating a distance z.
    """

    return w0 * np.sqrt(
        1.0
        + (
            wavelength * z
            / (np.pi * w0**2)
        ) ** 2
    )


def bg_computational_window(
    charge: int,
    kr: float,
    w0: float,
    wavelength: float,
    z: float,
    alpha: float = 1.2,
    n_ring: int = 3,
) -> tuple[float, float, float]:
    """
    Calculate the computational-window criteria for a
    Bessel-Gauss beam.

    Returns
    -------
    L_rings:
        Window required by the selected radial rings.

    L_gaussian:
        Window required by the Gaussian envelope.

    L:
        Maximum of the two criteria.
    """

    beta_n_m = jnp_zeros(
        abs(charge),
        n_ring,
    )[-1]

    L_rings = (
        2.0
        * alpha
        * beta_n_m
        / kr
    )

    propagated_waist = gaussian_waist_at_z(
        w0,
        wavelength,
        z,
    )

    L_gaussian = (
        6.0 * propagated_waist
    )

    L = max(
        L_rings,
        L_gaussian,
    )

    return L_rings, L_gaussian, L


# ============================================================
# Individual beam generators
# ============================================================

def gaussian_beam(
    grid: Grid,
    w0: float,
) -> ComplexArray:
    """
    Generate a Gaussian beam at its waist.

        U(r) = exp(-r²/w0²)
    """

    field = np.exp(
        -(grid.r**2) / w0**2
    )

    return normalize_field(
        field.astype(np.complex128),
        grid.dx,
    )


def laguerre_gaussian_beam(
    grid: Grid,
    w0: float,
    charge: int,
) -> ComplexArray:
    """
    Generate an LG beam with radial index p = 0.

        U(r,phi) ∝
            (sqrt(2) r / w0)^|ell|
            exp(-r²/w0²)
            exp(i ell phi)
    """

    radial_factor = (
        (
            np.sqrt(2.0)
            * grid.r
            / w0
        ) ** abs(charge)
    )

    gaussian_envelope = np.exp(
        -(grid.r**2) / w0**2
    )

    azimuthal_phase = np.exp(
        1j * charge * grid.phi
    )

    field = (
        radial_factor
        * gaussian_envelope
        * azimuthal_phase
    )

    return normalize_field(
        field.astype(np.complex128),
        grid.dx,
    )


def bessel_gaussian_beam(
    grid: Grid,
    w0: float,
    kr: float,
    charge: int,
) -> ComplexArray:
    """
    Generate a Bessel-Gauss beam.

        U(r,phi) ∝
            J_m(k_r r)
            exp(-r²/w0²)
            exp(i m phi)
    """

    bessel_profile = jv(
        abs(charge),
        kr * grid.r,
    )

    gaussian_envelope = np.exp(
        -(grid.r**2) / w0**2
    )

    azimuthal_phase = np.exp(
        1j * charge * grid.phi
    )

    field = (
        bessel_profile
        * gaussian_envelope
        * azimuthal_phase
    )

    return normalize_field(
        field.astype(np.complex128),
        grid.dx,
    )


# ============================================================
# General beam factory
# ============================================================

def create_beam(
    definition: BeamDefinition,
    grid: Grid,
) -> ComplexArray:
    """
    Generate a beam from its BeamDefinition.
    """

    family = definition.family.lower()

    if family == "gaussian":
        return gaussian_beam(
            grid,
            definition.w0,
        )

    if family == "lg":
        return laguerre_gaussian_beam(
            grid,
            definition.w0,
            definition.charge,
        )

    if family == "bg":
        if definition.kr is None:
            raise ValueError(
                "BG beams require kr."
            )

        return bessel_gaussian_beam(
            grid,
            definition.w0,
            definition.kr,
            definition.charge,
        )

    raise ValueError(
        f"Unknown beam family: {definition.family}"
    )
