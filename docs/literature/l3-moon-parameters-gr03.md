# L3: Moon parameters against GR03a and GR03b

Written 2026-11-03 (task L3, role A1). Question: do the masses and elements of Prometheus and Pandora in `params/` explain the factor 1.4 between our Lyapunov rate (0.22 per year, A1-ANALYSIS) and GR03 (0.30 to 0.35 per year)? Short answer: the literature alone cannot decide. The mass difference is the main candidate. One model run decides (section 6).

Keys: [GR03a] Goldreich and Rappaport 2003, Icarus 162, 391 (arXiv astro-ph/0205330, v1 only). [GR03b] Goldreich and Rappaport 2003, Icarus 166, 320 (arXiv astro-ph/0307259, v1). Other keys as in `params/sources.md`.

## 1. Values

| Item | GR03a | GR03b | Ours (`params/`) |
|---|---|---|---|
| Prometheus m/M | 1.19e-9 [GR03a, Table I] | 5.80e-10 [GR03b, Table 1] | 2.82e-10 (derived: GM 0.01071 / GM(Saturn) 37931206.234) |
| Pandora m/M | 7.65e-10 [GR03a, Table I] | 3.43e-10 [GR03b, Table 1] | 2.44e-10 (derived: 0.00926 / 37931206.234) |
| Ours / GR03b | | | 0.49 and 0.71 (derived) |
| Source of mass | assumed density 1.3 g/cm3 [GR03a, abstract] | assumed density 0.63 g/cm3 [GR03b, sec 2.1] | fitted GM, SAT415 via SSD [SSD_PHYS] |
| e Prometheus, e Pandora | 2.29e-3, 4.37e-3 [GR03a, Table I] | same [GR03b, Table 2] | 2.23e-3, 4.24e-3 (50-year track, `docs/results/a1-analysis.md` sec 1); SSD mean 0.002, 0.004 [SSD_ELEM] |
| Apse precession | from the Saturn gravity field [GR03a, sec 1] | 3.1911e-5 and 3.0082e-5 deg/s [GR03b, Table 2], from Campbell and Anderson 1989 | not stored; follows from J2 to J6 in the model |
| Mean motion | 1.1864e-4 and 1.1571e-4 rad/s [GR03a, Table I] | 6.797331e-3 and 6.629506e-3 deg/s [GR03b, Table 1] | states only |
| Epoch | Voyager fit (Jacobson, personal communication) [GR03a, sec 2.1] | 1981 Aug 23 04:02:12 UTC [GR03b, sec 3]; Jacobson 2001, JED 2444839.6682 [GR03b, references] | 2008-01-01 TDB, SAT415 |
| Bodies and forces | Saturn J2, J4, J6 and the two moons [GR03a, eq. 1] | same for FSHEP [GR03b, sec 2.2] | Saturn J2 to J6, the two moons, no other moons, no collisions (`docs/results/a1-analysis.md`) |

Open point. GR03a v1 gives density 1.3 g/cm3 in the abstract and Table I masses. GR03b sec 1 says that GR03a "assumed satellite masses based on a nominal density of 0.63". Only v1 of astro-ph/0205330 exists on arXiv. The inconsistency stays open. The 0.35 per year in GR03a (abstract) and "of order 0.3" in GR03b (sec 1) may come from these two mass sets.

Our masses as an equivalent density (derived, scaling 0.63 g/cm3 by the mass ratio): Prometheus 0.31, Pandora 0.45 g/cm3.

## 2. How GR03 measured the exponent

- Two shadow bodies, reset when the longitude separation is about 1e-4 rad [GR03a, sec 2.1].
- One integration of 3000 years (Bulirsch-Stoer, variable step) [GR03a, sec 2.1 and 2.2, Fig. 7].
- The exponent is shown against time, with a "constant + (log t)/t" curve fitted to the last point as the no-chaos case [GR03a, sec 2.2]. Result: "of order 0.35 yr^-1" [GR03a, abstract and sec 2.2]; "of order 0.3" [GR03b, sec 1 and sec 3, Fig. 3].
- The unit is per year. The log base is not stated. We assume natural.
- There is no per-run statistic and no mean over chaotic runs only.
- [GR03b, sec 3]: the reduced programs FSHEPRES and FSHEPSIM give similar results. FSHEPSIM "yields a slightly smaller Lyapunov exponent".

Ours (`docs/results/a1-analysis.md` sec 3): 100 runs of 100 years, a 2 km position kick, a tangent vector with IAS15. Rate = slope of ln(d) over years 30 to 100. Chaotic group (26 runs): 0.223 per year, spread 0.061. Nominal state: 0.175 per year; 0.31 in years 30 to 60 and 0.13 in years 60 to 100. The chaos is intermittent.

## 3. Scaling with mass and eccentricity

What GR03b gives:
- Disturbing function with four terms [GR03b, eq. 3]. The coefficients go as e^3 f82, e^2 e' f83, e e'^2 f84, e'^3 f85 [GR03b, eq. 4 to 7]. Values in Table 3: -0.001, 0.006, -0.01, 0.007.
- The resonance argument obeys eq. 18: the acceleration of psi is 3 (121 n')^2 (m/M) [1 + alpha m'/m] sum of C_k sin(psi + delta_k).
- Half widths 1.5, 3.7, 5.1, 4.1 rad/yr against a splitting of about 1.0 rad/yr [GR03b, sec 4]. The widths exceed the splitting, so the components overlap.
- With masses reduced by 10, the surface of section is regular [GR03b, Fig. 7 caption].
- Notation. [GR03b, sec 2.1, footnote 2] says "Primed and unprimed symbols refer to Prometheus and Pandora, respectively". The rest of the paper uses the opposite: the argument 121λ′ − 118λ is slow only if the primed moon is the slower one, Pandora (Table 1: 6.629506e-3 against 6.797331e-3 deg/s, and Ω′/Ω ≈ 118/121 in sec 2.3, after eq. 18); α = a/a′ must be below 1; and the same text calls m′ "the lighter body", which is Pandora (Table 1). So footnote 2 is the error. Below, unprimed is Prometheus and primed is Pandora (A0 review).

GR03b gives no equation for the Lyapunov rate against mass or e.

Estimate by A1 (not from GR03). For overlapping resonances, the Lyapunov rate is of the order of the libration frequency. The libration frequency goes like the half width, which goes like sqrt((m + alpha m') C_k) by eq. 18. Steps, with alpha = a_Prometheus / a_Pandora = 139,400 / 141,700 = 0.98 [SSD_ELEM]; I use 0.98:
1. Effective mass GR03b (Table 1), m = Prometheus, m′ = Pandora: 5.80e-10 + 0.98 x 3.43e-10 = 9.16e-10.
2. Effective mass ours: 2.824e-10 + 0.98 x 2.441e-10 = 5.22e-10.
3. Ratio 5.22 / 9.16 = 0.569. Half width ratio from mass: sqrt(0.569) = 0.754.
4. Largest term e e′^2 (Table 3), e = Prometheus, e′ = Pandora. With our e: (2.23 / 2.29) x (4.24 / 4.37)^2 = 0.974 x 0.941 = 0.917. Half width ratio from e: sqrt(0.917) = 0.958.
5. Total: 0.754 x 0.958 = 0.72. For comparison, the mass factor alone for GR03b masses scaled to 0.3 g/cm3 (factor 0.48): sqrt(0.48) = 0.69.
6. If lambda scales like the half width, GR's 0.30 per year becomes 0.30 x 0.72 = 0.22 per year. The inverse, 1/0.72 = 1.38, lies in the range 0.30/0.22 = 1.36 to 0.35/0.22 = 1.59.

## 4. Two readings

(a) Scaling estimate (section 3). The mass difference CAN explain the whole factor. The estimate is a proportionality argument, not a result of GR03.

(b) [GR03b, sec 1] says the exponent is "insensitive to the assumed density above a critical value of approximately 0.3 g/cm3". No figure or number supports it in the text. Our Prometheus density equivalent is 0.31 g/cm3 and Pandora 0.45. Prometheus is at the threshold, where "insensitive above" does not clearly apply. Reading (b) alone would say that the masses do not matter. At the threshold it cannot be applied with confidence.

## 5. Differences that are not about `params/`

- Definition. GR: mean over one 3000-year trajectory. Ours: slope over years 30 to 100, chaotic runs only (26 of 100). Our nominal state is 0.31 and then 0.13 in the two halves. The window changes the number by a factor of more than 2.
- Epoch and phase. GR start at the Voyager epoch (1981), where the mean motions were near their extremes [GR03b, sec 3]. We start in 2008 from SAT415.
- Method. GR use shadow bodies. We use the tangent vector, because shadow orbits gave a rate that depended on the offset (M20).
- Body set. The same: Saturn J2 to J6 and two moons (GR03a eq. 1; our model). This difference is not present.
- Cannot be decided from the literature: the values in GR's Fig. 3 and Fig. 7 (the figures are not in the text), the density behind each GR number, and the size of the window effect.

## 6. Verdict and follow-up

Verdict. The literature alone cannot decide. Candidate 1: the mass difference (reading a). Candidate 2: the method and window difference. A run by A8 decides. A1 does not run it. No value in `params/` changes (that needs an AJ decision).

Follow-ups for A8 (`fring-analyse lyapunov`, tangent method):
1. Baseline of 100 runs of 100 years (same seed, 2 km kick, rate over years 30 to 100), with the GR03b masses: GM(Prometheus) = 5.80e-10 x 37931206.234 = 0.02200 and GM(Pandora) = 3.43e-10 x 37931206.234 = 0.01301 km3/s2. These are our values times 2.054 and 1.405 (5.80 / 2.824 and 3.43 / 2.441). Predicted outcome: about 0.30 per year for the chaotic group if reading (a) is right; about 0.22 if reading (b) is right. Compare also the fraction of chaotic runs (26 % at present).
2. Re-analysis of the existing 100 runs, no new integration: rate over years 0 to 100 (shear subtracted as GR do with the (log t)/t curve), and the mean over all 100 runs. This tests the definition difference.
3. Optional: one 3000-year run of the two moons with the nominal parameters, the shadow-free tangent method, to compare with GR's single trajectory.

Decision rule: the mass difference explains the factor if run 1 gives the chaotic-group rate between 0.30 and 0.35 per year (within the error of the mean, about 0.012 per year at 26 runs). It does not explain the factor if run 1 stays near 0.22.
