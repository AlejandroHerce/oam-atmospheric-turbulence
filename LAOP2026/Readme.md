## LAOP 2026 Poster

**Influence of Atmospheric Turbulence Power Spectral Density Models on Orbital Angular Momentum Mode Coupling**
Alejandro Hernández, Yezid T. Moreno, Zayda Paola Reyes Quijano
Grupo de Óptica y Tratamiento de Señales (GOTS), Universidad Industrial de Santander, Colombia
Optica Latin America Optics and Photonics Conference (LAOP 2026), Bogotá, Colombia · Joint Poster Session, 10 November 2026 · Paper ID [Tu5A.28]

📄 **Poster (PDF):** [docs/LAOP2026_poster.pdf](docs/LAOP2026_poster.pdf)
📘 **Undergraduate thesis:** []
✉️ **Contact:** alejandro2190739@correo.uis.edu.co

### Key results

| Result | Value |
| --- | --- |
| Same $$C<sub>n</sub><sup>2</sup>$$, different PSD → change in OAM metrics | up to ~25 % |
| Beam wander vs. Kolmogorov (von Kármán and modified von Kármán) | ≈ −15 % for both → set by the outer scale L₀ |
| OAM spectral width vs. Kolmogorov | −22 % only with the inner scale l₀ (von Kármán alone: −3 %) |
| Mode retention vs. Kolmogorov | +10 % (von Kármán), +13 % (modified von Kármán) |
| Matched-size LG vs. BG | < 1 % difference in OAM width and entropy; no systematic winner |
| Degradation vs. beam size | tracks η̄ = r<sub>rms</sub>/r₀ (Spearman ρ ≈ −0.99 with retention) |
| Ensemble entropy gap | peaks at moderate turbulence in 18/18 configurations |
| Spectral asymmetry | strong turbulence shifts power toward lower \|ℓ\|; the sign flips for −ℓ₀ |

Relative changes are means over 18 scenarios (6 beams × 3 regimes), 1750 realizations each.

### Simulation parameters

| Category | Parameter | Value |
| --- | --- | --- |
| Propagation | Wavelength | 632.8 nm (He–Ne) |
| | Path | 1 km, horizontal, homogeneous C<sub>n</sub><sup>2</sup> |
| | Method | Symmetric split-step, angular spectrum method |
| | Phase screens | 16 (Δz = 62.5 m) |
| Transverse grid | Samples | 512 × 512 |
| | Window | 0.4 m × 0.4 m (Δx = 0.781 mm) |
| Turbulence | PSD models | Kolmogorov, von Kármán, modified von Kármán |
| | C<sub>n</sub><sup>2</sup> | 10⁻¹⁵, 10⁻¹⁴, 10⁻¹³ m⁻²ᐟ³ (weak, moderate, strong) |
| | Fried parameter r₀ | 106.6, 26.8, 6.7 mm |
| | Outer / inner scale | L₀ = 10 m, l₀ = 5 mm |
| | Subharmonic levels | 9 (Kolmogorov), 4 (von Kármán models) |
| Beams | Laguerre–Gauss | LG₀¹, LG₀², LG₀³, w₀ = 2.5 cm |
| | Bessel–Gauss | ℓ = 1, 2, 3, k<sub>r</sub>w₀ = 3, RMS radius matched to LG |
| OAM analysis | Decomposition | Azimuthal FFT on a polar grid (256 radii × 720 angles) |
| | Modal range | −240 ≤ ℓ ≤ 240 |
| Statistics | Ensemble | 1750 realizations per scenario |
| | Design | 3 PSD × 3 regimes × 6 beams = 54 scenarios (94 500 propagations) |
| | Uncertainty | Bootstrap 95 % confidence intervals |

### Validation (weak Kolmogorov turbulence)

| Test | Reference | Agreement |
| --- | --- | --- |
| On-axis scintillation index | Rytov theory (Gaussian beam) | 4.3 % at 1 km |
| RMS beam wander | Analytic prediction | 0.4 % at 1 km |
| OAM retention (LG₀¹) | Paterson model | 3.4 % |
| Energy and OAM spectrum in vacuum | Exact conservation | machine precision |

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
4. Asher Klug, Isaac Nape, and Andrew Forbes. The orbital angular momentum of a turbulent
atmosphere and its impact on propagating structured light fields. New Journal of Physics, 23
(9):093012, 2021.
5. N. A. Ferlic, M. van Iersel, and C. C. Davis, "Weak turbulence effects on different beams carrying orbital angular momentum," *J. Opt. Soc. Am. A* **38**, 1423–1437 (2021).
6. K. Khare, P. Lochab, and P. Senthilkumaran, *Orbital Angular Momentum States of Light: Propagation through Atmospheric Turbulence* (IOP Publishing, 2020).
