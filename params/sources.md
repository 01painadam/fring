# Sources of the numbers in params/

Task L2. Written 2026-11-01. Every number in `saturn.toml`, `moons.toml` and `ring.toml` is listed here with a source key.

Status values:

- **verified (primary)**: read in this session in the primary source (paper, official data page or data file).
- **verified (secondary, quoted in X)**: read in this session in source X, which quotes the primary. The primary source was not opened.
- **UNVERIFIED**: not read in any opened source, or only approximate. Listed again under open questions.
- **derived**: computed here from verified numbers; the formula is given.

## Epoch, time scale and frame

- Epoch: 2008-01-01 00:00:00 TDB = JD 2454466.5 TDB. `time_s = 0` at this epoch. Choice: Albers et al. 2012 was not opened, so its fit epoch is unknown. The fallback is a Cassini-era date in the middle of the observing period of [Cuzzi24] (2005 to 2013) and inside the span of the SAT415 ephemeris (1990-06-14 to 2018-01-18, from the JPL SSD page [SSD_EPHEM]). AJ confirms at gate G0.
- Time scale: TDB. Horizons was queried with TIME_TYPE = TDB, so no UTC conversion enters.
- Frame: Saturn-centred, Saturn's equator at the epoch (kept fixed), z along the spin axis. The x axis is the ascending node of Saturn's equator on the ICRF equator at the epoch. This is the Horizons convention for REF_PLANE = BODY (the response text states it; see `params/horizons/`).
- Pole precession: Saturn's pole moves by -0.04229 deg/century (RA) and -0.00444 deg/century (Dec) [Coop15_T3]. Over 30 yr that is 0.013 deg and 0.0013 deg. Fixing the frame at the epoch ignores this. The effect is far below the other uncertainties. Verified (secondary, quoted in [Coop15_T3]); the primary is Jacobson 2004, not opened.
- The pole used by Horizons is the IAU one (IAU_SATURN, "of date"), not the pole fitted in [J22]. The difference was not quantified. UNVERIFIED.

## Saturn (`saturn.toml`)

| Key | Value | Source key | Status |
|---|---|---|---|
| `gm_km3_s2` | 37931206.234 | J22_Zenodo (3.7931206234e16 m^3 s^-2 in the file header). SSD page gives 37931206.23 ± 0.24 for SAT441 [SSD_PHYS] | verified (primary: data file of the paper, transcribed by M. Wieczorek); agrees with SSD |
| `reference_radius_km` | 60330 | J22_Zenodo (file header, 6.033e7 m); the same radius is used in [Iess19] and [Coop15_T3] | verified (primary: data file) |
| `j2` | 1.6290615e-2 (± 2.5e-8) | J22_Zenodo: unnormalised C20 = -1.6290615e-2, and J2 = -C20 | verified (primary: data file) |
| `j4` | -9.35119e-4 (± 9.3e-8) | J22_Zenodo: unnormalised C40 = +9.35119e-4, and J4 = -C40 | verified (primary: data file) |
| `j6` | 8.6764e-5 (± 1.62e-7) | J22_Zenodo: unnormalised C60 = -8.6764e-5, and J6 = -C60. Added at G0 (ICR-001) | verified (primary: data file) |

Notes.

- The paper of Jacobson 2022 (AJ 164, 199, DOI 10.3847/1538-3881/ac90c9) could not be opened (bot check on the IOP site). J22_Zenodo is the archive "Spherical harmonic models of the gravity field of Saturn", Zenodo record 10878627, DOI 10.5281/zenodo.10878627, file `Jacobson2022.sh`. The record says the coefficients are for unnormalised functions without the Condon-Shortley phase, in the pyshtools layout, with the reference radius, GM, GM uncertainty and maximum degree (12) in the header. The sign convention J_n = -C_n0 is the standard one for unnormalised zonal terms. The convention is confirmed by the J2 and J4 values of [Iess19] (J2 > 0, J4 < 0).
- The GM uncertainty in the file header (0.726e9 m^3 s^-2 = 0.726 km^3 s^-2) differs from the 0.24 on the SSD page [SSD_PHYS]. Both are small (relative 2e-8). The reason is unknown.
- The moon states come from SAT415 (Horizons) and the field from SAT441 (Jacobson 2022). The two solutions differ. Accepted by A0 on 2026-11-01. The size of the differences between field solutions is in the next table.

### Comparison of field solutions (all for R = 60330 km)

| Solution | J2 | J4 | J6 | Source key |
|---|---|---|---|---|
| Jacobson 2022 (SAT441), used here | 0.016290615 | -9.35119e-4 | 8.6764e-5 | J22_Zenodo |
| Iess et al. 2019 (Grand Finale) | 0.016290573 ± 2.8e-8 | -9.35314e-4 ± 3.7e-8 | 8.6340e-5 ± 8.7e-8 | Iess19 (Table 1) |
| JPL SAT351, used by Cooper et al. 2015 | 0.01629084747 | -9.33698e-4 | 9.6437e-5 | Coop15_T3 |

- J2 differences: 2022 minus Iess = +4.2e-8 (relative 2.6e-6). 2022 minus SAT351 = -2.3e-7.
- J4 differences: 2022 minus Iess = +1.95e-7 (about 5 times the Iess uncertainty). 2022 minus SAT351 = -1.42e-6 (relative 0.15 %).
- The J2 of [Iess19] includes a constant tidal term from the satellites (Table 1 caption). Jacobson's treatment is UNVERIFIED.
- Effect on the apse precession at the core: see `docs/literature/formulas.md` (J4 and J6 matter at the level of 0.002 deg/day).

### Higher harmonics

`SaturnParams` has J2, J4 and J6 (J6 added at gate G0, ICR-001; the change to this file was made by A0 under that ICR). The Zenodo file also gives (unnormalised C, so J = -C): J3 = +9.5e-8 (± 4.0e-8), J5 = -6.0e-8 (± 1.37e-7), J6 = +8.6764e-5 (± 1.62e-7), J7 = +4.81e-7, J8 = -1.3931e-5 (± 2.17e-7), J9 = +8.92e-7, J10 = +5.426e-6 (± 2.08e-7), J11 = +1.47e-7, J12 = -5.37e-7 (± 1.80e-7). Odd harmonics are zero at the level of the uncertainty and are ignored. J8 to J12: not used in v1 (together they change the apse precession at the core by about 0.0001 deg/day). The effect of J6 on the resonance location is in `formulas.md`. Status: verified (primary: data file).

### Ring mass

- Iess et al. 2019 measure a total ring mass of 0.41 ± 0.13 Mimas masses [Iess19, Table 1]. With GM(Mimas) = 2.502784 km^3 s^-2 [Coop15_T4, SAT351 value], the ring GM is about 1.03 km^3 s^-2, which is 2.7e-8 of GM(Saturn) (derived).
- A ring of that mass at radii of about 75,000 to 137,000 km would add a quadrupole of order (1/2)(M_ring/M_Saturn)(r/R)^2, which is about 5e-8 of the J2 scale or 3e-6 of J2 (order of magnitude, derived here). The change in apse precession at the core is about 1e-5 deg/day, below the uncertainty of the observed rate (0.00029 deg/day).
- Whether the fitted field of [J22_Zenodo] contains the ring mass in its harmonics is UNVERIFIED. If it does, the point-mass plus J2 and J4 model already includes it on average. If it does not, the omitted effect is of the size given above.

## Moons (`moons.toml`)

| Key | Value | Source key | Status |
|---|---|---|---|
| Prometheus `gm_km3_s2` | 0.01071 ± 0.00001 | SSD_PHYS (ref SAT415) | verified (primary: official page) |
| Pandora `gm_km3_s2` | 0.00926 ± 0.00002 | SSD_PHYS (ref SAT415) | verified (primary: official page) |
| Prometheus `radius_km` | 43.09 | Horizons semi-axes 68.2 x 41.6 x 28.2 km (response header, "Revised: Oct 03, 2018"); radius = cube root of the product (derived) | verified (primary); derived |
| Pandora `radius_km` | 40.63 | Horizons semi-axes 52.2 x 40.8 x 31.5 km; cube root of the product (derived) | verified (primary); derived |
| Prometheus state | see `moons.toml` | HORIZONS_SAT415: `params/horizons/prometheus_616_2454466.5.txt` | verified (primary) |
| Pandora state | see `moons.toml` | HORIZONS_SAT415: `params/horizons/pandora_617_2454466.5.txt` | verified (primary) |

Notes.

- GM disagreement. Cooper et al. 2015 [Coop15] fit GM(Prometheus) = 0.010677 ± 0.000006 and GM(Pandora) = 0.009133 ± 0.000009 km^3 s^-2. SAT415 values differ by +0.3 % and +1.4 %. The older Jacobson et al. 2008 values (0.01064 ± 0.00010, 0.00915 ± 0.00013) are in [Coop15_T6]. The SSD values are used because they belong to the ephemeris of the states. The mass ratio matters for the strength of the chaos [GR03a].
- The Horizons query (all parameters) is in `params/horizons/fetch.sh`. The query string for each object is: COMMAND = '616' or '617', CENTER = '500@699', REF_PLANE = 'BODY', REF_SYSTEM = 'ICRF', TLIST = '2454466.5', TLIST_TYPE = 'JD', TIME_TYPE = 'TDB', EPHEM_TYPE = 'VECTORS', VEC_TABLE = '2', OUT_UNITS = 'KM-S', VEC_CORR = 'NONE'. The response states: target body from `sat415_merged_DE437`, reference frame "IAU_SATURN body equator and node of date", output "GEOMETRIC cartesian states".
- The moon radii are physical sizes for context. The code uses GM only.
- Osculating elements from the states with point-mass GM(Saturn) are not meaningful (a differs from the mean value by hundreds of km because of J2). `moons.toml` therefore gives states only. Mean elements from the SSD page [SSD_ELEM] (epoch 2000-01-01.5 TDB, Laplace frame; the page warns that they are not for ephemeris use): Prometheus a = 139,400 km, e = 0.002; Pandora a = 141,700 km, e = 0.004. Verified (primary), not used in the files.

### Cross-check of the states against an independent fit

[Coop15_T5] gives state vectors in the ICRF at JD 2454252.50075446 TDB (2007-06-01 00:00:00 UTC). Horizons was queried at that time with REF_PLANE = FRAME and REF_SYSTEM = ICRF (`params/horizons/crosscheck_*`). Differences (Horizons SAT415 minus Cooper):

| Moon | dx, dy, dz (km) | |dr| (km) | |dv| (km/s) |
|---|---|---|---|
| Prometheus | +1.79, -11.61, +1.47 | 11.8 | 0.0013 |
| Pandora | +2.12, -2.36, +1.13 | 3.4 | 0.0001 |

The two fits (both from Cassini data) agree to a few km for Pandora and to 12 km for Prometheus, against a distance of 1.4 x 10^5 km. The Prometheus offset is mostly along the orbit (dy) and reflects the chaotic changes in its mean motion between the two fits (the fit of Cooper uses data to 2013, SAT415 to 2018). The state vectors of [Coop15_T5] have formal errors of 0.1 to 1.5 km (position) and at most 0.0002 km/s. The difference is larger than these. A run that needs the moon phases to better than 0.01 degree must treat the moon realisation as uncertain. The chaotic motion makes this unavoidable over 30 yr.

## Ring (`ring.toml`)

| Key | Value | Source key | Status |
|---|---|---|---|
| `core.semi_major_axis_km` | 140222.43 ± 0.05 | Cuzzi24 (text: "F ring true core semimajor axis (location of the m = 110 CER) was 140222.43 ± 0.05 km (table S1)") | verified (primary); table S1 not opened |
| `core.eccentricity` | 0.00235 | Albers12_via_Hedman17 | verified (secondary, quoted in Hedman17) |
| `core.apse_rate_deg_per_day` | 2.70025 ± 0.00029 | Albers12_via_Hedman17 | verified (secondary, quoted in Hedman17) |
| `core.apse_longitude_deg` | 353.78 ± 5 | Hedman17: ringlet pericentre about 50 deg at J2000, "about 25 deg in front of the F-ring's pericenter" (Albers 2012). F ring: 25 deg at J2000. Propagation: 25 + 2.70025 × 2921.5 d = 7913.78 deg, mod 360 = 353.78 deg | UNVERIFIED (approximate; J2000 taken as JD 2451545.0 TDB, not stated in Hedman17) |
| `core.inclination_deg` | 0.0065 | Bosh02_via_Charnoz09 | verified (secondary, quoted in Charnoz09); the primary was not opened |
| `targets.radial_width_max_km` | 1.0 | Cuzzi24 | verified (primary) |
| `targets.dust_strand_width_*_km` | 10 to 30 | Cuzzi24 | verified (primary) |
| `targets.cer_order_m`, `lobe_width_deg` | 110, 3.27 | Cuzzi24 | verified (primary) |
| `targets.prometheus_mean_motion_deg_per_day` | 587.285415 ± 0.00029 | Cuzzi24 | verified (primary) |

Notes.

- Other values for the core a (spread of about 2 km): 140,221 km (Albers via Hedman17), 140223 (Bosh via Charnoz09), 140221.6 (Bosh fit 2 and Cooper 2013, via Benet17). Cuzzi24 is adopted because it is the best-defined number, it has an uncertainty, and `formulas.md` shows that it follows from the cited field and the Prometheus mean motion.
- The apse longitude at the epoch is the least certain number in `ring.toml`. The rate error alone gives ±0.85 deg after 2921.5 d. The J2000 value is "about" and has no uncertainty in the source.
- The e of Bosh (0.00254, via Charnoz09) differs from the Albers value (0.00235) by 8 %.

## Collisions: restitution law and method references (task K1-REFS)

Written 2026-11-07. These values are not in a TOML file. A run config sets them in `[collisions.restitution]` (`law = "power_law"`, `exponent`, `reference_speed_km_s`). The test `crates/fring-collide/tests/k1_pair.rs` uses `reference_speed_km_s = 7.7e-8`.

| Key | Value | Source key | Status |
|---|---|---|---|
| `exponent` p | 0.234 | Bridges84_via_LO06 (Latter and Ogilvie 2006, section 1) | verified (secondary, quoted in LO06). The primary was not opened |
| `reference_speed_km_s` v_c | 0.0077 cm/s = 7.7e-8 km/s | Bridges84_via_LO06 | verified (secondary, quoted in LO06). Unit conversion: 1 cm/s = 1e-5 km/s, so 0.0077 cm/s = 7.7e-8 km/s |
| Law | eps = (v_n / v_c)^(-p) for v_n > v_c, eps = 1 for v_n <= v_c | LO06, Eq. (1) | verified (secondary, quoted in LO06). The cut at v_c is part of the quoted step-wise power law, but LO06 gives it for all the authors it cites (Bridges, Hatzes, Supulver), not for Bridges alone. Whether Bridges et al. 1984 print it in this form is UNVERIFIED. v_n is the normal impact speed (LO06) |
| eps at 1 mm/s, 1 cm/s, 1 m/s | 0.549, 0.320, 0.109 | derived | derived: (v / 0.0077 cm/s)^(-0.234) with v/v_c = 12.99, 129.9, 12987. The SPEC values 0.55, 0.32, 0.11 agree |

Conditions of the Bridges values. LO06 says: "Bridges' data admit p = 0.234 and vc = 0.0077 cm s-1 for frosted particles of radius 2.5cm at a temperature of 210K (significantly higher than the appropriate conditions)." So the temperature is 210 K, not a low temperature. `fring-collide/SPEC.md` section 6 says "frosty ice at low temperature". This differs from LO06 (reported to A0). Other values quoted in LO06 for the same law: Hatzes et al. 1988, p = 0.20 and v_c = 0.025 cm/s (frosted particles, 123 K); Supulver et al. 1995, p = 0.19 and v_c = 0.029 cm/s (about 100 K). LO06 also says that a power law fits smoother particles worse than an exponential law, and that the coefficient of restitution changes much with the surface state.

Opened in this session for K1-REFS (at most two attempts per reference):

| Key | Reference | What was opened | What it supports |
|---|---|---|---|
| Bridges84 | Bridges, Hatzes, Lin 1984, Nature 309, 333-335 (title in the LO06 reference list: "Structure, Stability and Evolution of Saturn's Rings"). DOI 10.1038/309333a0 (not opened) | Not opened: the ADS page returned HTTP 405 and the Nature page was not tried. Values are from LO06 | p and v_c above, as quoted |
| LO06 | Latter, Ogilvie 2006, "The linear stability of dilute particulate rings", Icarus; arXiv astro-ph/0609164 | Full text | The Bridges values, Hatzes and Supulver values and the form of the law (Eq. 1) |
| Bird94 | Bird 1994, Molecular Gas Dynamics and the Direct Simulation of Gas Flows, Clarendon Press, Oxford | Not opened (a book). Search results only: secondary pages (arXiv 0709.3221 and a DSMC tutorial at particleincell.com) describe the NTC method | NTC: a candidate count from the maximum of sigma times relative speed, accepted with probability proportional to sigma times relative speed. The book chapter and equation numbers were not checked |
| CS69 | Carnahan, Starling 1969, J. Chem. Phys. 51(2), 635-636, DOI 10.1063/1.1672048 | Only the bibliographic record and abstract (via search). Full text not opened | The equation of state Z = (1 + eta + eta^2 - eta^3)/(1 - eta)^3 (formula from a textbook page, not from the paper). The SPEC uses it for a rate correction of 2.5 D. This is derived here: Z = 1 + 4 eta g gives g(contact) = (1 - eta/2)/(1 - eta)^3, which is 1 + 2.5 eta + ... for small eta. The paper gives the equation of state, not the contact value |
| Haff83 | Haff 1983, J. Fluid Mech. 134, 401-430 | Only bibliographic data and the abstract (Caltech record). Full text paywalled | The abstract says that grain collisions are inelastic and the energy equation is needed. The law that the kinetic energy falls as t^-2 is attributed to Haff by Ernst and Brito (as reported by a search result; that paper was not opened). The formula T = T0/(1 + t/t0)^2 and t0 = 6/((1 - eps^2) omega_0) in the SPEC are K1's own derivation (SPEC appendix A), not a quote of Haff |
| vNE98 | van Noije, Ernst 1998, Granular Matter 1, 57-64, DOI 10.1007/s100350050009; arXiv cond-mat/9803042 | arXiv abstract | Velocity distribution of inelastic hard spheres in free cooling and in the heated case, with the fourth cumulant and the high-energy tail. The abstract says that the non-Gaussian corrections have a small effect on the cooling rate. This supports a Maxwellian approximation in the SPEC for small inelasticity. No value from this paper is used |

## Variant: GR03b moon masses (task L3-PARAMS)

Written 2026-11-07. File: `params/variants/gr03b-moons.toml`. It is a copy of `moons.toml` with only the two GM values changed. The nominal values do not change.

| Key | Value | Source key | Status |
|---|---|---|---|
| Prometheus m/M | 5.80e-10 | GR03b, Table 1 | verified (primary), full text read by A1 in task L3 and re-checked by A0 on 2026-11-03 (`docs/handoffs/L3.md`). Not opened again in L3-PARAMS |
| Pandora m/M | 3.43e-10 | GR03b, Table 1 | same |
| Prometheus `gm_km3_s2` | 0.02200 | derived: 5.80e-10 x 37931206.234 = 0.0220001 | derived. Nominal 0.01071; ratio 2.054 |
| Pandora `gm_km3_s2` | 0.01301 | derived: 3.43e-10 x 37931206.234 = 0.0130104 | derived. Nominal 0.00926; ratio 1.405 |

Notes.

- GR03b is Goldreich and Rappaport 2003, Icarus 166, 320, arXiv astro-ph/0307259 (v1). GR03b sec 2.1 gives the masses from an assumed density of 0.63 g/cm3. Moon labels follow `docs/literature/l3-moon-parameters-gr03.md` (unprimed is Prometheus).
- The product uses GM(Saturn) of `saturn.toml`. GR03b used m/M, so the GM of the variant depends on this choice of GM(Saturn). The effect is 1e-8 relative or less for any current GM(Saturn).
- Purpose: A4-ANALYSIS (the deciding run of L3). The states are the nominal ones, so only the masses differ.

## Source list

| Key | Reference | DOI or URL | Opened in this session |
|---|---|---|---|
| J22_Zenodo | Jacobson 2022 gravity field, Zenodo transcription by M. Wieczorek (2024-03-26), file `Jacobson2022.sh` | 10.5281/zenodo.10878627, https://zenodo.org/api/records/10878627 | Yes: record and file |
| J22 | Jacobson 2022, AJ 164, 199 | 10.3847/1538-3881/ac90c9 | No (bot check). Bibliographic data from Crossref |
| SSD_PHYS | JPL SSD, Planetary Satellite Physical Parameters | https://ssd.jpl.nasa.gov/sats/phys_par/ | Yes |
| SSD_ELEM | JPL SSD, Planetary Satellite Mean Elements | https://ssd.jpl.nasa.gov/sats/elem/ | Yes |
| SSD_EPHEM | JPL SSD, Planetary Satellite Ephemerides (SAT415, SAT441 listing) | https://ssd.jpl.nasa.gov/sats/ephem/ | Yes |
| HORIZONS_SAT415 | JPL Horizons API 1.2 | https://ssd.jpl.nasa.gov/api/horizons.api (queries in `params/horizons/fetch.sh`) | Yes |
| Iess19 | Iess et al. 2019, Science 364, eaat2965 (Table 1) | 10.1126/science.aat2965; table read at https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Cassini/logs/Iess-Science-2019.pdf | Yes: tables of the paper |
| Coop15 | Cooper, Renner, Murray, Evans 2015, AJ 149, 27 (Tables 3 to 6) | 10.1088/0004-6256/149/1/27; arXiv 1406.6492 | Yes: full text |
| Cuzzi24 | Cuzzi et al. 2024, Sci. Adv. 10, eadl6601 | 10.1126/sciadv.adl6601; PMC11086617 | Yes: full text (supplement blocked) |
| Hedman17 | Hedman 2017, Icarus 281, 322 | 10.1016/j.icarus.2016.08.024; arXiv 1608.07844 | Yes: full text |
| Charnoz09 | Charnoz 2009, Icarus 201, 191 | 10.1016/j.icarus.2008.12.036; arXiv 0901.0482 | Yes: full text |
| Albers12 | Albers et al. 2012, Icarus 217, 367 | 10.1016/j.icarus.2011.11.016 | No |
| Bosh02 | Bosh et al. 2002, Icarus 157, 57 | 10.1006/icar.2002.6791 | No |
| GR03a | Goldreich and Rappaport 2003, Icarus 162, 391 | 10.1016/s0019-1035(02)00080-5; arXiv astro-ph/0205330 | Yes: full text |

Table keys used above: Coop15_T3, Coop15_T4, Coop15_T5, Coop15_T6 are Tables 3 to 6 of Coop15. Albers12_via_Hedman17 and Bosh02_via_Charnoz09 are numbers quoted in the opened papers. Full literature list: `docs/literature/summary.md`, section 8.

## Open questions

1. The epoch 2008-01-01 TDB is a fallback (Albers' epoch unknown). AJ approved it at G0.
2. J6 is used since G0 (ICR-001). J8 to J12 are not used in v1. Effect on the apse precession and the resonance location: `docs/literature/formulas.md`.
3. Apse longitude at the epoch is approximate and UNVERIFIED. Albers et al. 2012 (primary) is needed.
4. Core eccentricity and apse rate are verified only in a secondary source (Hedman 2017, quoting Albers et al. 2012). The inclination is verified only in Charnoz 2009 (quoting Bosh et al. 2002).
5. Whether the Jacobson 2022 field includes the ring mass, and the sign and size of the tidal term in J2, are UNVERIFIED (paper not opened).
6. The pole of Horizons (IAU) and the pole of Jacobson 2022 may differ. Not quantified.
7. The difference between the SAT415 and Cooper 2015 states for Prometheus (11.8 km) is larger than the formal errors. Use of one realisation of the moons is a modelling choice.
8. Cuzzi 2024 supplement (tables S1 and S2) could not be opened. The value 140222.43 ± 0.05 km is taken from the main text.
