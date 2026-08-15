"""Analytical reference solutions and comparison metrics for Chapter 2."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv

from src.beams import BeamDefinition, normalize_field
from src.grids import Grid

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def rayleigh_range(w0: float, wavelength: float) -> float:
    return np.pi * w0**2 / wavelength


def gaussian_width(w0: float, wavelength: float, z: float) -> float:
    zr = rayleigh_range(w0, wavelength)
    return float(w0 * np.sqrt(1.0 + (z / zr) ** 2))


def wavefront_radius(w0: float, wavelength: float, z: float) -> float:
    if np.isclose(z, 0.0):
        return np.inf
    zr = rayleigh_range(w0, wavelength)
    return float(z * (1.0 + (zr / z) ** 2))


def gouy_phase(w0: float, wavelength: float, z: float) -> float:
    zr = rayleigh_range(w0, wavelength)
    return float(np.arctan(z / zr))


def gaussian_analytical(grid: Grid, w0: float, wavelength: float, z: float) -> ComplexArray:
    k = 2.0 * np.pi / wavelength
    wz = gaussian_width(w0, wavelength, z)
    rz = wavefront_radius(w0, wavelength, z)
    psi = gouy_phase(w0, wavelength, z)

    envelope = (w0 / wz) * np.exp(-(grid.r**2) / wz**2)
    curvature = np.ones_like(grid.r, dtype=np.complex128) if np.isinf(rz) else np.exp(1j * k * grid.r**2 / (2.0 * rz))
    field = envelope * curvature * np.exp(-1j * psi)
    return normalize_field(field.astype(np.complex128), grid.dx)


def lg_analytical(grid: Grid, w0: float, charge: int, wavelength: float, z: float) -> ComplexArray:
    k = 2.0 * np.pi / wavelength
    wz = gaussian_width(w0, wavelength, z)
    rz = wavefront_radius(w0, wavelength, z)
    psi = gouy_phase(w0, wavelength, z)

    radial = (w0 / wz) * (np.sqrt(2.0) * grid.r / wz) ** abs(charge) * np.exp(-(grid.r**2) / wz**2)
    curvature = np.ones_like(grid.r, dtype=np.complex128) if np.isinf(rz) else np.exp(1j * k * grid.r**2 / (2.0 * rz))
    field = radial * np.exp(1j * charge * grid.phi) * curvature * np.exp(-1j * (abs(charge) + 1) * psi)
    return normalize_field(field.astype(np.complex128), grid.dx)


def bg_analytical(grid: Grid, w0: float, kr: float, charge: int, wavelength: float, z: float) -> ComplexArray:
    k = 2.0 * np.pi / wavelength
    if kr >= k:
        raise ValueError("kr must be smaller than k for a propagating BG component.")

    kz = np.sqrt(k**2 - kr**2)
    zr = kz * w0**2 / 2.0

    if np.isclose(z, 0.0):
        from src.beams import bessel_gaussian_beam
        return bessel_gaussian_beam(grid, w0, kr, charge)

    width_z = w0 * np.sqrt(1.0 + (z / zr) ** 2)
    radius_z = z * (1.0 + (zr / z) ** 2)
    psi = np.arctan(z / zr)
    scaling = 1.0 + 1j * z / zr

    field = (
        jv(abs(charge), grid.r * kr / scaling)
        * np.exp(1j * z * (kz - kr**2 / (2.0 * kz)))
        * np.exp(-1j * psi)
        * np.exp(-(grid.r**2) / width_z**2)
        * np.exp((-1j * kz / (2.0 * radius_z)) * (grid.r**2 + kr**2 * zr / kz**2))
        * np.exp(1j * charge * grid.phi)
    )
    return normalize_field(field.astype(np.complex128), grid.dx)


def create_analytical_beam(definition: BeamDefinition, grid: Grid, wavelength: float, z: float) -> ComplexArray:
    family = definition.family.lower()
    if family == "gaussian":
        return gaussian_analytical(grid, definition.w0, wavelength, z)
    if family == "lg":
        return lg_analytical(grid, definition.w0, definition.charge, wavelength, z)
    if family == "bg":
        if definition.kr is None:
            raise ValueError("BG beams require kr.")
        return bg_analytical(grid, definition.w0, definition.kr, definition.charge, wavelength, z)
    raise ValueError(f"Unknown beam family: {definition.family}")


def second_moment_radius(field: ComplexArray, grid: Grid) -> float:
    intensity = np.abs(field) ** 2
    denominator = np.sum(intensity) * grid.dx**2
    if denominator <= 0:
        raise ValueError("Field has zero intensity.")
    numerator = np.sum(grid.r**2 * intensity) * grid.dx**2
    return float(np.sqrt(2.0 * numerator / denominator))


def relative_width_error(numerical_field: ComplexArray, analytical_field: ComplexArray, grid: Grid) -> tuple[float, float, float]:
    numerical_width = second_moment_radius(numerical_field, grid)
    analytical_width = second_moment_radius(analytical_field, grid)
    error = abs(numerical_width - analytical_width) / analytical_width
    return numerical_width, analytical_width, float(error)


def intensity_fidelity(numerical_field: ComplexArray, analytical_field: ComplexArray, dx: float) -> float:
    i_num = np.abs(numerical_field) ** 2
    i_an = np.abs(analytical_field) ** 2
    overlap = np.sum(np.sqrt(i_num * i_an)) * dx**2
    p_num = np.sum(i_num) * dx**2
    p_an = np.sum(i_an) * dx**2
    return float(overlap**2 / (p_num * p_an))
