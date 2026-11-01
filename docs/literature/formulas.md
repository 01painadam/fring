# Analytical formulas for near-circular equatorial orbits (PLAN 7.2)

Task L2. Written 2026-11-01. Parameters: `params/saturn.toml` (Jacobson 2022 field, R = 60330 km) and `params/ring.toml`.

## 1. Confirmation of PLAN 7.2

The expressions of PLAN 7.2 were derived again with sympy from the potential of PLAN 7.1 (spherical radius, Legendre polynomials P2 and P4). The script is `docs/literature/derivations/n_kappa_nu.py`. Rerun it with:

```sh
python3 docs/literature/derivations/n_kappa_nu.py     # needs only sympy (pip install sympy)
```

With q = R/r and Omega_K^2 = GM/r^3, in the equatorial plane:

| Frequency | Definition | Result |
|---|---|---|
| n^2 | (1/r) dPhi/dr | Omega_K^2 [1 + (3/2) J2 q^2 - (15/8) J4 q^4] |
| kappa^2 | Phi_rr + (3/r) Phi_r | Omega_K^2 [1 - (3/2) J2 q^2 + (45/8) J4 q^4] |
| nu^2 | Phi_zz | Omega_K^2 [1 + (9/2) J2 q^2 - (75/8) J4 q^4] |

All three agree with PLAN 7.2. Two remarks.

- For the truncated potential (point mass, J2 and J4) these are exact for a circular equatorial orbit, not first-order approximations. Higher harmonics add terms in q^6 and above.
- The apse precession is varpi_dot = n - kappa and the nodal regression is node_dot = -(nu - n), both as in PLAN 7.2.

**Status of the published equation numbers: UNVERIFIED.** The textbook source named in PLAN (Murray and Dermott 1999) and Renner and Sicardy 2006 could not be opened in this session, and none of the open papers searched (arXiv: Tiscareno 2013, Latter 2017, Hedman and Nicholson 2013 and 2014, Longaretti 2016) writes these expansions. The confirmation here is the derivation. A0 decided on 2026-11-01 that the derivation serves as the confirmation. The equation numbers should be added when a book or paper is available.

## 2. Apse precession at the core, computed and observed

Input: a = 140222.43 km [Cuzzi24], GM = 37931206.234 km^3 s^-2, R = 60330 km, J2 = 1.6290615e-2, J4 = -9.35119e-4, and for the last rows J6 = 8.6764e-5, J8 = -1.3931e-5, J10 = 5.426e-6, J12 = -5.37e-7 [J22_Zenodo]. Rates are in deg/day.

| Field | n | kappa | varpi_dot = n - kappa | minus observed | node_dot |
|---|---|---|---|---|---|
| J2 only | 581.953771 | 579.327316 | 2.626455 | -0.073795 | -2.614655 |
| J2 + J4 (adopted) | 581.971174 | 579.274867 | 2.696307 | -0.003943 | -2.683872 |
| to J6 | 581.971523 | 579.273116 | 2.698407 | -0.001843 | -2.685953 |
| to J8 | 581.971535 | 579.273034 | 2.698501 | -0.001749 | -2.686046 |
| to J12 | 581.971536 | 579.273025 | 2.698510 | -0.001740 | -2.686056 |

Observed apse precession of the core: 2.70025 ± 0.00029 deg/day (Albers et al. 2012, quoted in [Hedman17]; verified (secondary)).

Result, as it is:

- The adopted J2 + J4 field gives 2.69631 deg/day. It is below the observed rate by 0.00394 deg/day (0.15 %, 13.6 times the quoted uncertainty).
- Adding J6 to J12 brings it to 2.69851, still below the observation by 0.00174 deg/day (0.064 %, 6.0 times the uncertainty).
- J2 alone is wrong by 0.074 deg/day, so J4 is needed.

What the gap means over 30 years (10957.5 d):

- J2 + J4: 0.00394 deg/day x 10957.5 d = 43.2 degrees of apse longitude.
- To J12: 0.00174 deg/day x 10957.5 d = 19.1 degrees.
- A model that uses J2 and J4 only and a point-mass Saturn will let a freely precessing core apse drift by about 43 degrees relative to the real one in 30 years. That is 0.12 of a full turn. Compare the apse-alignment cycle with Prometheus (17.1 yr, section 4).
- Not included in the computed rate: the secular terms from Prometheus and Pandora, the F ring's own mass and ring self-gravity, and the Titan term. These are probably the cause of the rest of the gap. They were not computed.

Effect of J6 on the core location. The mean motion at a given radius rises by Delta n = +3.487e-4 deg/day (relative +5.99e-7) when J6 is added to J2 + J4, and by +3.613e-4 deg/day (+6.21e-7) with J6 to J12. A resonance at a fixed mean motion (such as the CER, set by Prometheus) moves by Delta a = (2/3) a Delta n / n:

- J6: 0.056 km.
- J6 to J12: 0.058 km.

This is of the size of the quoted uncertainty of the core location (0.05 km, [Cuzzi24]) and about 6 % of the upper limit of the local core width (under 1 km). Leaving out J6 shifts resonance locations by an amount that matters only if the model resolves the core at the 0.1 km level. A0 raises J6 with AJ at G0.

## 3. Check of the corotation resonance of Cuzzi et al. 2024

[Cuzzi24] states that the true core sits at the m = 110 CER with Prometheus, with mean motion n_F = n_P - kappa_P / m [Cuzzi14dps] and Prometheus mean motion n_P = 587.285415 deg/day. Steps (in the script):

1. Find the radius where the field gives n = n_P: 139377.9 km.
2. Compute kappa_P from the field at that radius.
3. n_F = n_P - kappa_P / 110, and find the radius where the field gives n = n_F.

| Field | Prometheus radius (km) | kappa_P (deg/day) | n_F (deg/day) | CER radius (km) |
|---|---|---|---|---|
| J2 + J4 | 139377.86 | 584.530621 | 581.971500 | 140222.378 |
| to J12 | 139377.92 | 584.528318 | 581.971521 | 140222.432 |

The J2 to J12 field reproduces the quoted CER radius, 140222.43 km, to 0.002 km. The adopted J2 + J4 field is 0.053 km lower. This supports the gravity field, the core location of [Cuzzi24] and the cited mean motion together. The agreement at the 0.002 km level suggests that the authors of [Cuzzi24] used a field with the higher harmonics (J6 to J12) or one very close to J22_Zenodo. This is an inference and was not checked in the paper (supplement blocked).

## 4. Derived time scale

Apse drift of Prometheus relative to the core: Prometheus 2.758 deg/day [Hedman17, citing Spitale 2006 and Jacobson 2008] and the core 2.70025 deg/day give 360 / (2.758 - 2.70025) = 6234 d = 17.1 yr.

## 5. Open items

- Equation numbers in a published source (see section 1).
- Secular contributions of Prometheus, Pandora and ring self-gravity to the apse precession (section 2).
