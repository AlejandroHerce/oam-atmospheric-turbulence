## LAOP 2026 Poster

**Influence of Atmospheric Turbulence Power Spectral Density Models on Orbital Angular Momentum Mode Coupling** ·
Alejandro Hernández, Yezid T. Moreno, Zayda Paola Reyes Quijano ·
Grupo de Óptica y Tratamiento de Señales (GOTS), Universidad Industrial de Santander, Colombia ·
Optica Latin America Optics and Photonics Conference (LAOP 2026), Bogotá, Colombia · Joint Poster Session, 10 November 2026 · Paper ID [Tu5A.28]

📄 **Poster (PDF):** [LAOP2026/LAOP2026_poster.pdf](LAOP2026/LAOP2026_poster.pdf)
✉️ **Contact:** alejandro2190739@correo.uis.edu.co

### Key results

| Result | Value |
| --- | --- |
| Same $C_n^2$, different PSD → change in OAM metrics | up to ~25 % |
| Beam wander vs. Kolmogorov (von Kármán and modified von Kármán) | ≈ −15 % for both → set by the outer scale $L_0$ |
| OAM spectral width vs. Kolmogorov | −22 % only with the inner scale $l_0$ (von Kármán alone: −3 %) |
| Mode retention vs. Kolmogorov | +10 % (von Kármán), +13 % (modified von Kármán) |
| Matched-size LG vs. BG | < 1 % difference in OAM width and entropy; no systematic winner |
| Degradation vs. beam size | tracks $\bar{\eta} = r_{\mathrm{rms}}/r_0$ (Spearman $\rho \approx -0.99$ with retention) |
| Ensemble entropy gap | peaks at moderate turbulence in 18/18 configurations |
| Spectral asymmetry | strong turbulence shifts power toward lower $\vert\ell\vert$; the sign flips for $-\ell_0$ |

Relative changes are means over 18 scenarios (6 beams × 3 regimes), 1750 realizations each.

### Simulation parameters

| Category | Parameter | Value |
| --- | --- | --- |
| Propagation | Wavelength | $\lambda = 632.8$ nm (He–Ne) |
| | Path | 1 km, horizontal, homogeneous $C_n^2$ |
| | Method | Symmetric split-step, angular spectrum method |
| | Phase screens | 16 ($\Delta z = 62.5$ m) |
| Transverse grid | Samples | $512 \times 512$ |
| | Window | $0.4 \times 0.4$ m ($\Delta x = 0.781$ mm) |
| Turbulence | PSD models | Kolmogorov, von Kármán, modified von Kármán |
| | $C_n^2$ | $10^{-15}$, $10^{-14}$, $10^{-13}\ \mathrm{m}^{-2/3}$ (weak, moderate, strong) |
| | Fried parameter $r_0$ | 106.6, 26.8, 6.7 mm |
| | Outer / inner scale | $L_0 = 10$ m, $l_0 = 5$ mm |
| | Subharmonic levels | 9 (Kolmogorov), 4 (von Kármán models) |
| Beams | Laguerre–Gauss | $\mathrm{LG}_0^1$, $\mathrm{LG}_0^2$, $\mathrm{LG}_0^3$, $w_0 = 2.5$ cm |
| | Bessel–Gauss | $\ell = 1, 2, 3$, $k_r w_0 = 3$, RMS radius matched to LG |
| OAM analysis | Decomposition | Azimuthal FFT on a polar grid (256 radii × 720 angles) |
| | Modal range | $-240 \le \ell \le 240$ |
| Statistics | Ensemble | 1750 realizations per scenario |
| | Design | 3 PSD × 3 regimes × 6 beams = 54 scenarios (94 500 propagations) |
| | Uncertainty | Bootstrap 95 % confidence intervals |

### Numerical verification and validation

The code was checked in four stages before running the production simulations: free-space propagation, phase-screen statistics, numerical settings in turbulence, and physical validation against analytic theory in weak turbulence.

#### 1. Free-space propagation (no turbulence)

Test beams: Gaussian, $\mathrm{LG}_0^1$–$\mathrm{LG}_0^3$ and BG $\ell = 1$–$3$, propagated over 1 km in 16 steps of 62.5 m.

| Check | What was done | Result |
| --- | --- | --- |
| Transverse sampling | Window ≥ 6 RMS radii at 1 km; $\Delta x$ bounded by phase, radial-profile and Bessel-ring sampling criteria for every beam | Most restrictive criterion (radial, $\mathrm{LG}_0^1$) → $512 \times 512$, 0.4 m window |
| Transfer-function sampling | Nyquist condition of the angular-spectrum transfer function for $\Delta z = 62.5$ m | Required padding ≈ 33 samples ≪ 512 → no zero padding needed |
| Energy conservation | Total power before and after 1 km | Relative error $\le 6 \times 10^{-16}$ |
| OAM conservation | OAM spectrum before and after 1 km: $\langle \ell \rangle$, power in $\ell_0$ and $L_1$ distance between spectra | $\langle \ell \rangle$ and $P(\ell_0) = 1$ preserved; $L_1$ distance $\le 2.4 \times 10^{-12}$ |
| Analytic solutions | RMS radius and intensity fidelity vs. analytic LG and BG propagation | RMS-radius error $\le 3 \times 10^{-10}$; fidelity = 1.0000000000 |

#### 2. Phase screens

| Check | What was done | Result |
| --- | --- | --- |
| Screen sampling | Phase differences between neighboring pixels and spectral energy near the Nyquist limit (worst case: Kolmogorov, strong turbulence, $r_0 = 35$ mm per screen) | RMS difference 0.39 rad; 99.9th percentile 1.28 rad $< \pi$; energy above $0.8\,\kappa_N \approx 4 \times 10^{-5}$ |
| Kolmogorov structure function | Ensemble-averaged phase structure function vs. $D_\theta(\rho) = 6.88\,(\rho/r_0)^{5/3}$, adding subharmonic levels $b = 0$–$12$ | MAPE 50.8 % ($b = 0$) → 7.4 % ($b = 9$); change between consecutive levels < 1 % from $b = 9$ → **$b = 9$ adopted** |
| Finite-screen effect | Same test with screen sizes $N = 256$–$2048$ | Residual deficit decreases with screen size (MAPE 3.4 % at $N = 2048$, $b = 9$): a finite-window effect, not a method error |
| von Kármán and modified von Kármán | Structure function vs. the corresponding analytic expressions, $b = 0$–$4$ | MAPE 36 % ($b = 0$) → 5.4 % ($b = 4$); change between levels 0.19 % → **$b = 4$ adopted** |

#### 3. Numerical settings in turbulence

Worst-case configuration (Kolmogorov, strong turbulence, largest beams) unless stated otherwise.

| Check | What was done | Result |
| --- | --- | --- |
| OAM modal range | Fraction of OAM power outside $-\ell_{\max} \le \ell \le \ell_{\max}$ | $\ell_{\max} = 240$ leaves $< 8.5 \times 10^{-4}$ of the power outside |
| Ensemble size | Retention and spectral width vs. number of realizations (up to 2000), with relative standard error of the mean | 1750 realizations → relative SEM 0.99 % (retention) and 0.30 % (width) |
| Number of phase screens | Rytov-variance criterion per screen; scintillation index with 8, 16, 32 and 64 screens (moderate turbulence) | Criterion requires ≥ 11 screens in strong turbulence; no systematic change beyond the statistical uncertainty → **16 screens adopted** |
| Spectral content of the field | Fraction of field energy near the Nyquist limit and 99th-percentile spatial frequency after 1 km (moderate and strong, all PSDs) | ≤ 0.5 % of the energy above $0.8\,\kappa_N$; 99th percentile $\le 0.66\,\kappa_N$; energy conserved to $\sim 10^{-12}$ % |
| Reproducibility | Deterministic seed hierarchy (PSD → regime → realization → screen) | All beams in a scenario see the same atmosphere; every run can be reproduced exactly |

#### 4. Physical validation in weak turbulence

Kolmogorov PSD, $C_n^2 = 10^{-15}\ \mathrm{m}^{-2/3}$ (plane-wave Rytov variance $\sigma_R^2 \approx 0.057 \ll 1$), 1 km, 16 screens, 1750 realizations. The scintillation and beam-wander tests use a Gaussian beam ($w_0 = 2.5$ cm) and compare the simulation with theory at each of the 16 propagation planes, not only at the receiver.

| Test | Theory | Result |
| --- | --- | --- |
| On-axis scintillation index $\sigma_I^2(z)$ | Rytov theory for a Gaussian beam in weak fluctuations | Global $L_2$ error 3.4 %; theory inside the 95 % CI in 14/16 planes. At 1 km: 0.0324 [0.0303, 0.0345] vs. 0.0338 (4.3 %) |
| RMS beam wander (centroid) | Analytic prediction for a collimated beam on a horizontal path | Global $L_2$ error 0.47 %. At 1 km: 2.161 mm [2.087, 2.232] vs. 2.151 mm (0.44 %); largest deviation at the first plane |
| OAM spectrum of $\mathrm{LG}_0^1$ | Paterson's phase-perturbation model with Kolmogorov azimuthal coherence | Retention 0.666 vs. 0.644 (3.4 %); $L_1$ distance between spectra 0.044 |

The slightly higher simulated OAM retention is consistent with Paterson's model using the plane-wave coherence radius and ignoring diffraction after each screen.

### Computational environment

| Component | Specification |
| --- | --- |
| OS | Ubuntu 24.04 LTS (x86_64) |
| CPU | Intel Core i5-12500H (12 cores, 16 threads), no GPU |
| RAM | 15 GiB |
| Software | Python 3.12.3, NumPy 2.5.2, SciPy 1.18.0, Matplotlib 3.11.1 |
| Parallelization | `ProcessPoolExecutor`, up to 12 processes |

### Reproducing one scenario

```bash
# Example: LG beam with ℓ = 3, Kolmogorov PSD, strong turbulence
python -m experiments.chapter_5.run_scenario --beam LG03 --psd kolmogorov --regime strong --realizations 1750

# Statistical analysis of all scenarios
python -m experiments.chapter_5.analyze_final_results
```

Beams: `LG01`–`LG03`, `BG01`–`BG03` · PSD: `kolmogorov`, `von_karman`, `modified_von_karman` · Regimes: `weak`, `moderate`, `strong`.

### References

1. L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random Media*, 2nd ed. (SPIE Press, 2005).
2. C. Paterson, "Atmospheric turbulence and orbital angular momentum of single photons for optical communication," *Phys. Rev. Lett.* **94**, 153901 (2005).
3. C. Peters, V. Cocotos, and A. Forbes, "Structured light in atmospheric turbulence—a guide to its digital implementation: tutorial," *Adv. Opt. Photon.* **17**, 113–184 (2025).
4. A. Klug, I. Nape, and A. Forbes, "The orbital angular momentum of a turbulent atmosphere and its impact on propagating structured light fields," *New J. Phys.* **23**, 093012 (2021).
5. N. A. Ferlic, M. van Iersel, and C. C. Davis, "Weak turbulence effects on different beams carrying orbital angular momentum," *J. Opt. Soc. Am. A* **38**, 1423–1437 (2021).
6. K. Khare, P. Lochab, and P. Senthilkumaran, *Orbital Angular Momentum States of Light: Propagation through Atmospheric Turbulence* (IOP Publishing, 2020).
