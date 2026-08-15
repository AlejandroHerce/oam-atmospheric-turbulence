# OAM Propagation Through Atmospheric Turbulence

Numerical framework for studying the propagation of structured optical beams carrying orbital angular momentum (OAM) through atmospheric turbulence.

This repository contains the computational methods, numerical validations, and simulation experiments developed as part of my undergraduate physics thesis. The project investigates how different atmospheric turbulence power spectral density (PSD) models affect the propagation and modal coupling of OAM-carrying optical beams.

## Research Overview

Atmospheric turbulence introduces spatial fluctuations in the refractive index that distort the amplitude and phase of propagating optical fields. For beams carrying orbital angular momentum, these distortions can redistribute optical power among different OAM modes.

The numerical framework developed in this project combines:

* Gaussian, Laguerre–Gaussian (LG), and Bessel–Gaussian (BG) beams
* Angular Spectrum Method (ASM) propagation
* Atmospheric phase-screen generation
* Kolmogorov and modified turbulence power spectral density models
* Subharmonic compensation of low spatial frequencies
* Orbital angular momentum spectrum analysis
* Statistical ensemble simulations

The final objective is to quantify how the choice of atmospheric turbulence PSD influences OAM mode coupling under different turbulence regimes.

## Repository Structure

```text
.
├── configs/
│   └── chapter_2.py
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
│   └── chapter_3/
│
└── tests/
```

### `src/`

Contains the reusable numerical and physical methods used throughout the project.

### `configs/`

Contains simulation parameters shared by the numerical experiments. Parameters established through validation are centralized here to ensure consistency and reproducibility.

### `experiments/`

Contains the numerical experiments and validation procedures associated with each thesis chapter.

### `tests/`

Reserved for software-level tests used to verify the numerical implementation as the framework evolves.

## Numerical Validation

The numerical framework is validated progressively before atmospheric-turbulence simulations are performed.

### Chapter 2 — Free-Space Propagation

The free-space propagation framework includes validation of:

* computational window size and spatial sampling
* beam parameter selection and beam-size matching
* optical energy conservation
* conservation of orbital angular momentum
* numerical propagation against analytical beam solutions

These tests establish the numerical parameters used in subsequent simulations.

### Chapter 3 — Atmospheric Phase Screens

The atmospheric turbulence implementation includes validation of:

* turbulence power spectral density models
* phase-screen spatial-frequency sampling
* low-frequency subharmonic compensation
* phase structure functions
* statistical convergence of phase-screen properties

### Chapter 4 — Integrated Turbulent Propagation

Currently under development.

This stage combines the previously validated propagation and atmospheric models to assess the complete turbulent-propagation framework and establish the final simulation configuration.

### Chapter 5 — OAM Propagation Through Turbulence

Planned production simulations will investigate OAM-mode coupling across different beam families, turbulence models, and turbulence regimes.

## Current Status

The repository is under active development alongside the thesis.

Free-space propagation and atmospheric phase-screen methods have been developed and validated. The current work focuses on integrating these components and validating full optical propagation through atmospheric turbulence.

## Requirements

The numerical framework is written in Python and currently relies primarily on:

* NumPy
* SciPy
* Matplotlib

Installation and reproducibility instructions will be expanded as the project develops.

## Author
Alejandro Hernández Celis
Undergraduate Physics Thesis
Colombia

