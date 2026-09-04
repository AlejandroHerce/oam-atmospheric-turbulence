# OAM Propagation Through Atmospheric Turbulence

Numerical framework for studying the propagation of structured optical beams carrying orbital angular momentum (OAM) through atmospheric turbulence.

This repository contains the computational methods, numerical validation procedures, production simulations, and statistical analyses developed as part of my undergraduate physics thesis.

The project investigates how the spatial structure of optical beams and the power spectral density (PSD) of atmospheric turbulence influence the propagation, modal coupling, and degradation of OAM-carrying optical fields.

---

## Research Overview

Atmospheric turbulence produces random spatial fluctuations in the refractive index that distort the amplitude and phase of propagating optical fields. For beams carrying orbital angular momentum, these perturbations redistribute optical power among different OAM modes, producing modal crosstalk and degradation of the transmitted state.

The numerical framework developed in this project combines:

- Gaussian, Laguerre–Gaussian (LG), and Bessel–Gaussian (BG) beams
- Angular Spectrum Method (ASM) for free-space propagation
- Split-Step propagation through atmospheric turbulence
- Atmospheric phase-screen synthesis
- Kolmogorov, von Kármán, and modified von Kármán turbulence PSD models
- Low-spatial-frequency subharmonic compensation
- Orbital angular momentum spectrum decomposition
- Statistical ensemble simulations
- Bootstrap-based uncertainty estimation
- Numerical and theoretical validation procedures
- Analysis of modal retention, spectral broadening, entropy, asymmetry, and beam wander
- Realization-level analysis of the relationship between spatial and modal degradation

The final study compares the response of OAM-carrying beams under weak, moderate, and strong turbulence and examines how the turbulence PSD, azimuthal order, transverse beam scale, and beam family affect the resulting spatial and modal degradation.

---

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
│   ├── chapter_4/
│   └── chapter_5/
│
├── results/
│   └── chapter_5/
│
└── tests/
```

### `src/`

Contains the reusable numerical and physical methods used throughout the project, including beam generation, propagation algorithms, phase-screen synthesis, analytical reference solutions, and OAM-spectrum analysis.

### `configs/`

Contains the physical and numerical parameters established during the different stages of the thesis. Centralizing these parameters ensures consistency and reproducibility across validation and production simulations.

### `experiments/`

Contains the numerical experiments associated with each thesis chapter. Validation procedures, production simulations, statistical analyses, and visualization scripts are kept separate from the reusable methods in `src/`.

### `results/`

Contains the outputs of the production simulations and their subsequent statistical analyses. Results are organized according to turbulence PSD, turbulence regime, and transmitted beam.

### `tests/`

Contains software-level tests used to verify individual components of the numerical implementation.

---

## Numerical Validation

The numerical framework was validated progressively, from individual numerical components to the complete turbulent-propagation model.

### Chapter 2 — Free-Space Propagation

The free-space propagation framework includes validation of:

- computational window size and spatial sampling
- beam parameter selection and transverse-size matching
- optical energy conservation
- conservation of orbital angular momentum
- numerical propagation against analytical beam solutions
- numerical representation of the OAM spectrum
- spatial and modal truncation criteria

These tests establish the spatial and modal discretization used throughout the subsequent simulations.

### Chapter 3 — Atmospheric Phase Screens

The atmospheric turbulence implementation includes validation and characterization of:

- Kolmogorov, von Kármán, and modified von Kármán PSD models
- spatial-frequency sampling of the phase screens
- low-frequency subharmonic compensation
- phase structure functions
- statistical properties of the generated turbulence

These tests establish the phase-screen synthesis procedure subsequently used by the Split-Step propagation model.

### Chapter 4 — Integrated Turbulent Propagation

The complete turbulent-propagation framework combines the previously validated free-space propagation and atmospheric phase-screen models.

The integrated validation includes:

- statistical convergence of ensemble-based observables
- weak-turbulence validation against the Rytov approximation
- beam-wander comparison with theoretical predictions
- OAM-spectrum comparison with the weak-turbulence model of Paterson
- longitudinal robustness analysis of the Split-Step implementation
- spatial-frequency and Nyquist verification under moderate and strong turbulence
- cross-PSD spectral-sampling verification
- establishment of the final numerical configuration for production simulations

Together, these tests establish the validity and numerical robustness of the propagation framework used in the final comparative study.

---

## Chapter 5 — OAM Propagation Through Atmospheric Turbulence

The final production study considers:

- Laguerre–Gaussian beams with azimuthal orders \(|\ell| = 1, 2, 3\)
- Bessel–Gaussian beams with corresponding OAM orders
- Kolmogorov turbulence
- von Kármán turbulence
- modified von Kármán turbulence
- weak, moderate, and strong turbulence regimes

The factorial design contains **54 atmospheric propagation scenarios**, corresponding to the combinations of:

- 2 structured-beam families
- 3 azimuthal orders
- 3 turbulence PSD models
- 3 turbulence regimes

Each production scenario uses an ensemble of atmospheric realizations and stores the OAM spectrum together with complementary spatial and modal observables.

The main quantities analyzed are:

- transmitted-mode retention
- RMS OAM spectral spread
- normalized Shannon entropy
- complete ensemble-averaged OAM spectrum
- spectral asymmetry around the transmitted mode
- ensemble entropy gap
- intensity-centroid displacement
- RMS beam wander
- realization-level correlations between beam wander and OAM degradation

The analysis therefore distinguishes between different manifestations of turbulence-induced degradation rather than treating modal robustness as a single scalar quantity.

---

## Main Physical Questions

The numerical study addresses several related questions:

1. How does increasing turbulence strength modify OAM-mode retention and modal spreading?

2. Does the turbulence PSD influence OAM degradation even when the overall turbulence regime is comparable?

3. How does the transmitted OAM order affect modal robustness?

4. To what extent are apparent differences between LG and BG beams related to their spatial scale and diffraction rather than their modal family?

5. Does atmospheric turbulence redistribute OAM power symmetrically around the transmitted state?

6. How much of the modal variability arises within individual atmospheric realizations and how much arises from differences between realizations?

7. How is transverse beam wander related to the degradation of the OAM spectrum?

These questions are investigated using complementary metrics because no single observable completely characterizes the interaction between a structured optical field and atmospheric turbulence.

---

## Methodological Workflow

The computational workflow follows a progressive validation strategy:

1. **Free-space propagation**  
   Establish the transverse grid, propagation method, beam parameters, and OAM representation.

2. **Atmospheric turbulence**  
   Implement and independently validate the phase-screen models.

3. **Integrated propagation**  
   Combine phase screens with Split-Step propagation and validate the complete numerical framework.

4. **Production simulations**  
   Propagate the selected structured beams through the different atmospheric turbulence models.

5. **Statistical analysis**  
   Quantify modal retention, spectral broadening, entropy, spectral asymmetry, ensemble variability, beam wander, and their relationships.

This separation between reusable numerical methods, validation experiments, production simulations, and final statistical analysis is intended to make the computational methodology transparent and reproducible.

---

## Key Findings

Within the parameter space investigated in this thesis:

- Increasing turbulence strength systematically decreases transmitted-mode retention and increases OAM spectral broadening and modal entropy.

- The turbulence PSD affects the resulting spatial and modal degradation, showing that turbulence intensity alone does not completely determine the behavior of an OAM channel.

- Higher OAM orders exhibit systematically stronger modal degradation under the beam parameterization used in this work.

- The observed order dependence cannot be separated from the simultaneous change in the transverse scale and diffractive evolution of the optical fields. The ratio between the effective transverse beam size and the turbulence coherence scale provides a useful description of this behavior.

- LG and BG beams with comparable transverse evolution exhibit very similar OAM degradation, with no systematic advantage of one family over the other under the conditions studied.

- Under moderate and strong turbulence, the ensemble-averaged OAM spectrum develops a systematic asymmetry. For the positive transmitted charges considered in the main simulations, the redistribution is preferentially oriented toward lower OAM indices. Charge-inversion control simulations support an interpretation in terms of preferential redistribution toward lower \(|\ell|\), rather than toward a fixed sign of the OAM axis.

- The ensemble entropy gap distinguishes modal diversity within individual atmospheric realizations from spectral variability between realizations. This quantity reaches its maximum in the moderate-turbulence regime for all configurations studied.

- Beam wander increases strongly with turbulence intensity and is sensitive to the turbulence PSD, particularly to differences associated with the low-spatial-frequency region of the turbulence spectrum.

- Under the parameterization considered here, beam wander decreases modestly but systematically with increasing OAM order, while modal degradation simultaneously increases. This demonstrates that spatial beam stability and OAM modal robustness are not equivalent observables.

- At the realization level, larger centroid excursions are systematically associated with lower transmitted-mode retention and broader, more entropic OAM spectra.

- The orientation of the OAM spectral asymmetry is essentially uncorrelated with the magnitude or direction of centroid displacement, indicating that the observed spectral bias cannot be reduced to a trivial consequence of transverse beam wander.

Overall, the results support a picture in which OAM robustness is governed by the interaction between the spatial scales of the optical field and the spatial scales represented in the turbulence spectrum, rather than by OAM order or beam-family label alone.

---

## OAM Metrics

Several complementary observables are used to characterize the modal response of the propagated fields.

### Transmitted-Mode Retention

The fraction of modal power remaining in the initially transmitted OAM channel provides a direct measure of channel preservation.

### RMS OAM Spectral Spread

The RMS spread around the transmitted mode characterizes the characteristic modal distance over which turbulence redistributes optical power.

### Normalized OAM Entropy

The normalized Shannon entropy measures the diversity of the OAM distribution and provides a complementary description of modal spreading that does not depend explicitly on modal distance.

### Spectral Asymmetry

Signed and absolute asymmetry metrics are used to determine whether power transfer toward modes located symmetrically around the transmitted state occurs with equal strength.

This analysis reveals that turbulence-induced OAM redistribution is not necessarily symmetric around the transmitted mode.

### Ensemble Entropy Gap

The ensemble entropy gap is defined as the difference between the entropy of the ensemble-averaged spectrum and the average entropy of individual atmospheric realizations,

\[
\Delta H_{\mathrm{ens}}
=
H\!\left(\langle P_j\rangle\right)
-
\left\langle H(P_j)\right\rangle.
\]

For equiprobable realizations, this quantity is related to the generalized Jensen–Shannon divergence of the ensemble of OAM spectra.

It distinguishes modal diversity occurring within individual atmospheric realizations from spectral variability produced by changes between realizations.

---

## Beam Wander and Modal Degradation

In addition to modal redistribution, the framework tracks the transverse intensity centroid of the propagated field.

The ensemble beam wander is characterized through the RMS fluctuation of the centroid around its ensemble-averaged position.

The analysis shows that beam wander and OAM degradation are statistically related at the realization level:

- larger centroid excursions are associated with lower transmitted-mode retention;
- larger centroid excursions are associated with broader OAM spectra;
- larger centroid excursions are associated with higher modal entropy.

However, beam wander and OAM degradation remain physically distinct observables. In particular, the direction of the OAM spectral asymmetry is not determined by the direction of centroid displacement.

This distinction is important when assessing the robustness of structured optical fields: reduced transverse motion does not necessarily imply improved preservation of the transmitted OAM state.

---

## Reproducibility

Production simulations use deterministic hierarchical random seeds so that each beam + PSD + turbulence-regime scenario can be reproduced independently.

The production framework also supports deterministic extension of existing ensembles while preserving the original realization sequence.

Statistical uncertainty is estimated through bootstrap resampling where appropriate.

Because complete realization-level OAM spectra and large atmospheric ensembles can generate substantial amounts of data, users reproducing the study may choose to regenerate raw simulation outputs locally rather than storing all intermediate data in the repository.

---

## Requirements

The numerical framework is written in Python and relies primarily on:

- NumPy
- SciPy
- Matplotlib
- pandas

The simulations support parallel execution for computationally intensive ensemble calculations.

A Python virtual environment is recommended to isolate the project dependencies.

---

## Thesis Status

The numerical framework, validation studies, production simulations, and final statistical analyses associated with the undergraduate thesis have been completed.

The repository represents the computational framework used to obtain the results reported in the final thesis.

Future extensions of the project may include:

- experimental validation under controlled atmospheric turbulence;
- independent variation of turbulence inner and outer scales;
- separation of OAM order from transverse beam size and diffraction;
- optimization of LG and BG beam parameters for specific atmospheric channels;
- joint radial and azimuthal modal decompositions;
- temporally correlated atmospheric channels;
- adaptive-optics and modal-tracking strategies;
- realistic longitudinal \(C_n^2\) profiles;
- and end-to-end OAM communication-system simulations.

---

## Author

**Alejandro Hernández Celis**  
Undergraduate Physics Thesis  
Colombia
