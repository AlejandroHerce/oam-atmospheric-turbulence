# OAM Propagation Through Atmospheric Turbulence

Numerical framework for studying the propagation of structured optical beams carrying orbital angular momentum (OAM) through atmospheric turbulence.

This repository contains the computational methods, numerical validation procedures, and simulation experiments developed as part of my undergraduate physics thesis. The project investigates how different atmospheric turbulence power spectral density (PSD) models affect the propagation and modal coupling of OAM-carrying optical beams.

## Research Overview

Atmospheric turbulence produces random spatial fluctuations in the refractive index that distort the amplitude and phase of propagating optical fields. For beams carrying orbital angular momentum, these perturbations can redistribute optical power among different OAM modes, producing modal crosstalk and degradation of the transmitted state.

The numerical framework developed in this project combines:

* Gaussian, Laguerre–Gaussian (LG), and Bessel–Gaussian (BG) beams
* Angular Spectrum Method (ASM) for free-space propagation
* Split-Step propagation through atmospheric turbulence
* Atmospheric phase-screen generation
* Kolmogorov, von Kármán, and modified von Kármán turbulence PSD models
* Subharmonic compensation of low spatial frequencies
* Orbital angular momentum spectrum decomposition
* Statistical ensemble simulations
* Bootstrap-based uncertainty estimation
* Numerical and theoretical validation procedures

The final objective is to quantify how the choice of atmospheric turbulence PSD influences spatial beam degradation and OAM modal coupling under weak, moderate, and strong turbulence regimes.

## Repository Structure

```text
.
├── configs/
│   ├── chapter_2.py
│   ├── chapter_3.py
│   └── chapter_4.py
│
├── src/
│   ├── analytical.py
│   ├── beams.py
│   ├── grids.py
│   ├── oam.py
│   ├── phase_screens.py
│   └── propagation.py
│
├── experiments/
│   ├── chapter_2/
│   ├── chapter_3/
│   └── chapter_4/
│
└── tests/
```

### `src/`

Contains the reusable numerical and physical methods used throughout the project, including beam generation, propagation algorithms, phase-screen synthesis, analytical reference solutions, and OAM-spectrum analysis.

### `configs/`

Contains the physical and numerical parameters shared by the experiments. Parameters established through validation are centralized here to ensure consistency and reproducibility across the simulation framework.

### `experiments/`

Contains the numerical experiments and validation procedures associated with each thesis chapter. These scripts are kept separate from the reusable numerical methods in `src/`.

### `tests/`

Reserved for software-level tests used to verify individual components of the numerical implementation as the framework evolves.

## Numerical Validation

The numerical framework is validated progressively, from individual numerical components to the complete turbulent-propagation model.

### Chapter 2 — Free-Space Propagation

The free-space propagation framework includes validation of:

* computational window size and spatial sampling
* beam parameter selection and beam-size matching
* optical energy conservation
* conservation of orbital angular momentum
* numerical propagation against analytical beam solutions
* OAM-spectrum numerical representation and modal truncation

These tests establish the spatial and modal discretization used in the subsequent simulations.

### Chapter 3 — Atmospheric Phase Screens

The atmospheric turbulence implementation includes validation and characterization of:

* Kolmogorov, von Kármán, and modified von Kármán PSD models
* phase-screen spatial-frequency sampling
* low-frequency subharmonic compensation
* phase structure functions
* statistical properties of the generated turbulence

These tests establish the phase-screen generation procedure used by the Split-Step propagation model.

### Chapter 4 — Integrated Turbulent Propagation

The complete turbulent-propagation framework combines the previously validated free-space propagation and atmospheric phase-screen models.

The integrated validation includes:

* statistical convergence of ensemble-based observables
* weak-turbulence validation against the Rytov approximation
* beam-wander validation against theoretical predictions
* OAM-spectrum validation against the weak-turbulence model of Paterson
* longitudinal robustness analysis of the Split-Step implementation
* spatial-frequency and Nyquist verification under moderate and strong turbulence
* cross-PSD spectral-sampling verification
* establishment of the final numerical configuration for production simulations

Together, these tests assess the behavior of the complete propagation framework before its application to the comparative study of atmospheric turbulence models.

### Chapter 5 — OAM Propagation Through Turbulence

The production simulations investigate the influence of atmospheric turbulence on structured optical beams across:

* Gaussian, LG, and BG beam families
* Kolmogorov, von Kármán, and modified von Kármán turbulence models
* weak, moderate, and strong turbulence regimes
* multiple transmitted OAM states

The analysis focuses on complementary spatial and modal observables, including:

* transmitted-mode retention
* beam wander
* OAM-spectrum broadening
* complete OAM modal distributions
* spectral asymmetry
* normalized Shannon entropy of the OAM spectrum

These quantities are used to characterize not only the magnitude of turbulence-induced degradation, but also the structure of the resulting modal redistribution.

## Methodological Workflow

The computational workflow follows a progressive validation strategy:

1. **Free-space propagation:** establish the transverse grid, propagation method, beam parameters, and OAM representation.
2. **Atmospheric turbulence:** implement and validate the phase-screen models independently.
3. **Integrated propagation:** combine phase screens with Split-Step propagation and validate the complete numerical framework.
4. **Production simulations:** apply the validated configuration to compare the effects of different atmospheric turbulence PSD models on OAM propagation.

This separation between reusable numerical methods, validation experiments, and production simulations is intended to make the computational methodology transparent and reproducible.

## Current Status

The numerical framework for free-space propagation, atmospheric phase-screen generation, and integrated turbulent propagation has been implemented and validated.

The project is currently transitioning to the production simulations used to compare the influence of the different turbulence PSD models on spatial beam propagation and OAM-mode coupling.

## Requirements

The numerical framework is written in Python and relies primarily on:

* NumPy
* SciPy
* Matplotlib

The simulations are designed for reproducible ensemble calculations and support parallel execution for computationally intensive experiments.

Installation and reproducibility instructions will be expanded as the project develops.

## Author

**Alejandro Hernández Celis**  
Undergraduate Physics Thesis  
Colombia
