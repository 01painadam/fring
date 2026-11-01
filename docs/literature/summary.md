# F ring dynamics since 2000: literature summary

Task L1. Role A1. Written 2026-11-01.

Scope: shepherding, chaos, the location of the core, jets and spirals, moonlets, collisional models. The first question is the project question: why is the F ring core narrow and long-lived in a region where Prometheus makes orbits chaotic?

## How to read this file

- Every number has a source key in square brackets, for example [Cuzzi24]. Section 8 lists the keys with DOI and the way each source was opened.
- "Opened" means that the text, abstract or table was read in this session from the DOI page, arXiv, PubMed, PMC or an official data page. Where only a second paper reports a number, the entry says "via [key]". The primary source is then UNVERIFIED for that number.
- Section 7 lists what could not be opened. No number in this file comes from memory.
- Angles are in degrees. Rates are in degrees per day unless stated. Distances are in km.

## 1. The core question: a narrow ring in a chaotic zone

### 1.1 What was expected, and what went wrong

1. Shepherding. After Voyager, the F ring was explained by two moonlets on either side that confine it by oppositely directed torques [Cuzzi24, introduction, citing Goldreich and Tremaine 1979]. Prometheus is the more massive moon and it is the nearer one, so a simple torque balance fails [Cuzzi24, citing Showalter and Burns 1982].
2. Chaos. The whole F ring region is chaotic. Prometheus perturbs any particle there strongly enough that orbits change substantially and unpredictably on time scales of weeks to months [Cuzzi24, introduction, citing Scargle et al. 1993 and Winter et al. 2007, 2010].
3. Persistence. Despite this, the ring has stayed in essentially the same orbit from the Voyager encounters in 1980 to 1981 to the end of Cassini [Cuzzi24, citing Cooper et al. 2013; Benet17, quoting Tiscareno 2013].

### 1.2 The chaos of the moons themselves

- Prometheus and Pandora are chaotic because of the 121:118 mean motion resonance. Differential precession splits it into four components equally spaced in frequency. The libration width of each component exceeds the splitting, so they overlap [GR03b, abstract].
- Mean motions wander chaotically in zones of width 1.8 deg/yr (Prometheus) and 3.1 deg/yr (Pandora) [GR03b, abstract].
- The Lyapunov exponent of the pair is of order 0.3 to 0.35 per year [GR03a, abstract; GR03b, introduction]. Conjunctions every 24.8 d drive the perturbations, and the orbits come closest every 6.2 yr, when the apses are anti-aligned [GR03a, abstract].
- Cassini fits show that the changes in mean motion are not confined to the times of apse anti-alignment [Coop15, abstract].
- Masses from the Cassini fit: GM(Prometheus) = 10.677 ± 0.006 and GM(Pandora) = 9.133 ± 0.009, both in 10^-3 km^3 s^-2 [Coop15, abstract].
- Consequence for the model: the moon longitudes cannot be predicted beyond a few years. A 30 year run compares statistics, not trajectories (AGENTS.md rule). The moon table is integrated from one epoch, and the choice of epoch is part of the experiment.

### 1.3 Hypotheses for the stability of the core

| ID | Hypothesis | Status in the literature | Source |
|---|---|---|---|
| H1 | Torque balance between Prometheus and Pandora | Doubtful, because the nearer moon is the more massive | [Cuzzi24] |
| H2 | Stable pockets in the chaotic sea, found by long N-body runs. They lie close to first-order outer Lindblad resonances (LR) of Prometheus, and one of them matches the observed core | Narrow stable zones in simulations; one agrees with the core within the uncertainty | [Cuzzi24], describing Cuzzi et al. 2014 |
| H3 | "Antiresonance": apse precession between encounters cancels the semi-major axis kicks | An error was found in the formulation and a corrigendum was issued (Icarus 413, 116011). Do not use H3 as a mechanism | [Cuzzi24, ref. 18 and text] |
| H4 | Corotation-eccentricity resonance (CER). The core sits in the m = 110 CER with Prometheus. Conjunctions with the arcs happen when Prometheus is at periapse, which is farthest from the ring, so the kick is weakest | Supported by Cassini radio occultation (RSS) data, with a chance probability of 0.001 to 0.01 | [Cuzzi24] |
| H5 | Test-particle scattering. Only some initial conditions stay trapped between the moons; those that survive form a narrow, eccentric, apse-aligned ring | A J2-only planar model with Prometheus, Pandora and Titan finds stripes tied to Prometheus outer LRs and CRs. Pandora inner LR and CR coincide with gaps | [Benet17] |
| H6 | Collisional damping and self-gravity help to hold the core together | Not tested on the core by the sources opened here. Hedman17 finds collisions may lock the apse precession of a nearby ringlet. This is the gap that `fring` can fill | [Hedman17] |

### 1.4 Key results with numbers

**The true core [Cuzzi24].**

- The F ring was detected in 23 of 66 Cassini RSS ingress or egress occultations (0.94, 3.6 and 13 cm), May 2005 to August 2013. That is about 35 %.
- The detections show a single strand, a few hundred metres to about 1 km wide, with peak normal optical depth of a few to several tenths. The radio signal senses particles larger than a few millimetres, up to several centimetres.
- The imaging strands of micron dust are 10 to 30 km wide and are transparent to the radio signal.
- The mass of the core (width under 1 km, particle radius of a few millimetres or more) is at least 1000 times the mass of the dust strands.
- The arcs of the core sit at longitudes set by the m = 110 CER. One cycle covers 360/110 = 3.27 degrees of longitude. About 70 % of each cycle is populated. Detections cluster within ±1 degree of the longitude of Prometheus periapse.
- Semi-major axis of the CER, taken as the true core: 140222.43 ± 0.05 km. Prometheus mean motion in the analysis period: 587.285415 ± 0.00029 deg/day.
- A "small chaotic glitch" in Prometheus' orbit near the end of the mission temporarily broke the confinement. The arcs appeared to adapt.
- Not resolved: how disconnected arcs keep a common, uniformly precessing, eccentric orbit [Cuzzi24].

**Stable zones and the earlier theory [Cuzzi14dps; Cuzzi24].**

- Particles that meet Prometheus near its apoapse, especially at apse anti-alignment, get a semi-major axis kick of tens of km that is strongly non-sinusoidal. It cannot be cancelled, so diffusion is chaotic [Cuzzi14dps, abstract]. The antiresonance explanation built on this is withdrawn (H3).
- The hypothesis for the true core mean motion is n_F = n_P − κ_P/m, with m = 110 [Cuzzi14dps]. `docs/literature/formulas.md` (L2) tests this with the cited gravity field: the relation gives 140222.43 km (J2 to J12), the value quoted in [Cuzzi24].

**Test-particle scattering [Benet17].**

- Planar restricted five-body model (Saturn with J2, Prometheus, Pandora, Titan). About 2.5 million test particles, followed for up to 6 × 10^6 Prometheus periods (more than 10,000 yr).
- The surviving stable subset forms a narrow, eccentric ring. The best-fit semi-major axis is about 140213 km, against the observed 140221.6 km. The eccentricity is 2.9 × 10^-4 and 1.5 × 10^-4 in the two snapshots, about an order of magnitude below observation.
- The 90 % width of the model ring is about 241 km. The paper says this overestimates the core width, which it gives as 1 to 40 km (citing French et al. 2012).

**Ringlet in Prometheus' orbit [Hedman17].**

- A narrow dusty ringlet has mean radius 139,300 to 139,400 km and e = 0.0012, consistent with a 1:1 resonance with Prometheus. Its apse precesses at 2.702 ± 0.003 deg/day (the rate of the F ring) and not at the 2.758 deg/day expected near Prometheus. The author attributes the difference to collisions with F ring material.
- This is evidence that collisions can lock the apse precession of material over a range of semi-major axes.

## 2. Location and orbit of the core

| Quantity | Value | Source |
|---|---|---|
| Semi-major axis | 140,221 km (Albers) | via [Hedman17], quoting Albers et al. 2012 |
| | 140223 km (Bosh) | via [Charnoz09], used there as an initial condition |
| | 140221.6 km (Bosh fit 2; Cooper et al. 2013) | via [Benet17] |
| | 140222.43 ± 0.05 km (CER location, RSS data) | [Cuzzi24] |
| Eccentricity | 0.00235 | via [Hedman17], quoting Albers et al. 2012 |
| | 0.00254 | via [Charnoz09], quoting Bosh et al. 2002 |
| Inclination | 0.0065 degrees | via [Charnoz09], quoting Bosh et al. 2002 |
| Apse precession rate | 2.70025 ± 0.00029 deg/day | via [Hedman17], quoting Albers et al. 2012 |
| Pericentre longitude | about 25 degrees at the J2000 epoch. Inferred: [Hedman17] puts the ringlet pericentre at about 50 degrees, "about 25 degrees in front of" the F ring pericentre | via [Hedman17]; approximate |

Notes.

- The primary papers (Bosh 2002, Albers 2012, Cooper 2013) were not opened. See section 7.
- The sources differ by about 2 km in a. The CER location (140222.43 ± 0.05) is for the massive core and is the best-defined number. It is also the value that follows from the cited gravity field and the Prometheus mean motion (`formulas.md`). Use it as a0 for the rotating frame (PLAN 4.2).
- Prometheus: a = 139,380 km, e = 0.0022, apse precession 2.758 deg/day [Hedman17, citing Spitale 2006 and Jacobson 2008].
- With apse precession of 2.758 for Prometheus and 2.70025 for the core, the apses drift relative to each other with a period of 360 / (2.758 − 2.70025) d = 6234 d = 17.1 yr (derived here). [Hedman17] states that "every 17 years" the moon can come within 200 km of the core. This is the long time scale of the system. The 6.2 yr period is that of Prometheus and Pandora.

## 3. Prometheus: channels and the streamer

- Prometheus approaches and recedes from the ring every 14.7 h. As it recedes it draws out ring material. After one orbital period, keplerian shear turns the disturbed region into a channel [Murray05].
- The effect should grow as the orbits near their minimum separation in 2009 [Murray05].
- Gravitational scattering gives vorticity above background for 1 to 3 orbital periods after an encounter. Radial dispersion at channel edges is 20 to 50 cm/s, up to 1 m/s [Sutton16].
- A model with self-gravity and a multi-strand ring gives transient clumps of about 20 × 100 km, which can combine into a vortex sheet [Sutton13].

## 4. Jets, spirals and mini-jets

- Spiral. The strands on both sides of the core form one trailing, one-arm spiral winding at least three times around Saturn. It rotates with the orbital motion of its particles. It looks like a cloud of particles scattered from the core and stretched by differential motion [Charnoz05].
- The moonlet candidate S/2004 S6 crosses the core where the spiral meets the ring [Charnoz05].
- Jets. Murray et al. report moonlets embedded in the bright core. Most of the F ring morphology comes from the continual gravitational and collisional effects of small satellites, often with Prometheus [Murray08].
- Collisional origin of spirals and jets [Charnoz09]:
  - Scattering by Prometheus-mass bodies is not available, because there are none in the region. The paper proposes physical collisions of km-size moonlets with loose clumps.
  - The radial extent of the structures (about 300 km, up to ±200 km around the core) needs a coefficient of restitution of 0.18 or less. The N-body runs used 0.05 to 0.1 and 10,000 test particles. Encounter velocities are about 25 m/s (e of about 0.002).
- Mini-jets [Attree14]:
  - More than 800 features about 50 km across, with a typical life of about 1 d, at random longitudes and times, with little correlation with Prometheus.
  - They imply objects of radius 1 km or less, with internal strength, with semi-major axes spread over ±100 km relative to the ring and mostly within 20 km. The collision speed is about 1 m/s (from the introduction of [Attree14], citing Attree et al. 2012).

## 5. Collisional models: what is known and what is missing

- The coefficient of restitution needed for jets and spirals is low, 0.05 to 0.18, for impacts of about 25 m/s on solid moonlets [Charnoz09]. This is for moonlet-core collisions, not for ring particle collisions at cm/s.
- Relative speeds for mini-jet collisions are of order 1 m/s [Attree14].
- No source opened here gives a velocity-dependent restitution law for core particles. PLAN 7.5 allows a Bridges-type law. Bridges et al. 1984 was not opened. UNVERIFIED.
- No source opened here gives a collisional N-body or DSMC model of the narrow core over decades with Prometheus included. [Benet17] notes that its test-particle approach ignores collisions and self-gravity. This is the opening for `fring`.

## 6. Observations a model must reproduce

Format: ID, observation, value, source, model diagnostic (PLAN 7.7), priority. Priority A decides the project question, B supports it, C is for completeness. AJ sets the final priorities (PLAN 13, question 4).

| ID | Observation | Value | Source | Model diagnostic | Priority |
|---|---|---|---|---|---|
| O1 | Semi-major axis of the core | 140221 to 140223 km; best 140222.43 ± 0.05 km | [Cuzzi24]; via [Hedman17]; via [Charnoz09] | Mean a of core particles | A |
| O2 | Core eccentricity | 0.00235 (Albers), 0.00254 (Bosh) | via [Hedman17]; via [Charnoz09] | Mean e of core particles | A |
| O3 | Common apse precession | 2.70025 ± 0.00029 deg/day | via [Hedman17] | Mean apse longitude against time | A |
| O4 | Radial width of the massive core | under 1 km locally; 1 to 40 km (French et al. 2012, via [Benet17]); dust strands 10 to 30 km | [Cuzzi24]; via [Benet17] | Core width (rms and 90 %) | A |
| O5 | Persistence | same orbit from 1980 to 2013 and later | [Cuzzi24] | Fraction of particles in the core band against time | A |
| O6 | Arc structure of the massive core | 110 lobes of 3.27 degrees, about 70 % populated, tied to the longitude of Prometheus periapse; seen in 35 % of occultations | [Cuzzi24] | Azimuthal density against longitude relative to Prometheus periapse | A |
| O7 | Response to a Prometheus glitch | arcs adapt to a small chaotic change of Prometheus' mean motion | [Cuzzi24] | Runs with two moon realisations; compare core retention | B |
| O8 | Channels behind Prometheus | repeat every 14.7 h; form after one orbit of shear | [Murray05] | Density map in the rotating frame | B |
| O9 | Mass ratio core to dust | at least 1000 | [Cuzzi24] | Mass in particles above a few mm | B |
| O10 | Spiral | one-arm trailing spiral, at least 3 windings | [Charnoz05] | Image of a clump run | C |
| O11 | Jets | extent about 300 km around the core; need low restitution | [Charnoz09]; [Murray08] | Radial extent of debris after a moonlet-core collision | C |
| O12 | Mini-jets | over 800 seen; 50 km; life about 1 d; random longitude | [Attree14] | Rate and lifetime of collision clumps | C |
| O13 | Prometheus ringlet | e = 0.0012, a = 139,300 to 139,400 km, precession 2.702 ± 0.003 deg/day | [Hedman17] | Test of apse locking by collisions | C |
| O14 | Chaos of the moons | Lyapunov exponent about 0.3 per year; mean motion zones 1.8 and 3.1 deg/yr | [GR03a]; [GR03b] | Reproduced by the moon integrator (C3), not by the ring | A (for C3) |

## 7. What could not be opened

These are cited second-hand above or not at all. Anything taken from them is UNVERIFIED.

- Cuzzi et al. 2014 (Icarus 232, 157) and the 2024 corrigendum (Icarus 413, 116011): the publisher pages were not opened. The 2014 content comes from [Cuzzi24] and the conference abstract [Cuzzi14dps].
- Albers et al. 2012 (Icarus 217, 367), Bosh et al. 2002 (Icarus 157, 57) and Cooper et al. 2013 (AJ 145, 161): the numbers come from [Hedman17], [Charnoz09] and [Benet17].
- The supplement of [Cuzzi24] (tables S1 and S2, the Keplerian core model): the publisher blocked the download.
- Murray and French 2018 (book chapter), French et al. 2003, Spitale et al. 2006, Jacobson et al. 2008.
- Jacobson 2022 (AJ 164, 199): the IOP page is behind a bot check. The gravity field is read from a Zenodo transcription (see `params/sources.md`).
- Bridges et al. 1984 restitution law.
- Winter et al. (2007, 2010) on confined chaos of the F ring, cited only through [Cuzzi24].

## 8. Sources

"Route" says how the source was opened in this session. Crossref was used to check DOIs and bibliographic data.

| Key | Reference | DOI or URL | Route |
|---|---|---|---|
| Cuzzi24 | Cuzzi, Marouf, French, Murray, Cooper 2024, Saturn's F ring is intermittently shepherded by Prometheus, Sci. Adv. 10, eadl6601 | 10.1126/sciadv.adl6601 | Full text on PMC (PMC11086617); supplement blocked |
| Cuzzi14dps | Cuzzi, Marouf, French, Jacobson 2014, F Ring Core Stability: Corotation Resonance Plus Antiresonance, DPS 46, 402.01 | https://ntrs.nasa.gov/api/citations/20140017740/downloads/20140017740.pdf | Abstract on NTRS |
| GR03a | Goldreich and Rappaport 2003, Chaotic motions of Prometheus and Pandora, Icarus 162, 391 | 10.1016/s0019-1035(02)00080-5; arXiv astro-ph/0205330 | arXiv full text |
| GR03b | Goldreich and Rappaport 2003, Origin of chaos in the Prometheus-Pandora system, Icarus 166, 320 | 10.1016/j.icarus.2003.09.002; arXiv astro-ph/0307259 | arXiv full text |
| Coop15 | Cooper, Renner, Murray, Evans 2015, Saturn's inner satellites: orbits, masses and the chaotic motion of Atlas, AJ 149, 27 | 10.1088/0004-6256/149/1/27; arXiv 1406.6492 | arXiv full text |
| Benet17 | Benet and Jorba 2017, A simple model for the location of Saturn's F ring, Icarus 285, 224 | 10.1016/j.icarus.2016.12.006; arXiv 1612.01451 | arXiv full text |
| Hedman17 | Hedman 2017, A curious ringlet that shares Prometheus' orbit but precesses like the F ring, Icarus 281, 322 | 10.1016/j.icarus.2016.08.024; arXiv 1608.07844 | arXiv full text |
| Charnoz09 | Charnoz 2009, Physical collisions of moonlets and clumps with the Saturn's F-ring core, Icarus 201, 191 | 10.1016/j.icarus.2008.12.036; arXiv 0901.0482 | arXiv full text |
| Attree14 | Attree, Murray, Williams, Cooper 2014, A survey of low-velocity collisional features in Saturn's F ring, Icarus 227, 56 | 10.1016/j.icarus.2013.09.008; arXiv 1309.3119 | arXiv full text |
| Charnoz05 | Charnoz et al. 2005, Cassini discovers a kinematic spiral ring around Saturn, Science 310, 1300 | 10.1126/science.1119387 | Abstract on PubMed (PMID 16311328) |
| Murray05 | Murray et al. 2005, How Prometheus creates structure in Saturn's F ring, Nature 437, 1326 | 10.1038/nature04212 | Abstract on PubMed (PMID 16251957) |
| Murray08 | Murray et al. 2008, The determination of the structure of Saturn's F ring by nearby moonlets, Nature 453, 739 | 10.1038/nature06999 | Abstract on PubMed (PMID 18528389) |
| Sutton13 | Sutton and Kusmartsev 2013, Gravitational vortices and clump formation in Saturn's F ring during an encounter with Prometheus, Sci. Rep. 3, 1276 | 10.1038/srep01276 | Abstract on PubMed (PMID 23429480) |
| Sutton16 | Sutton and Kusmartsev 2016, Prometheus induced vorticity in Saturn's F ring, Earth, Moon, and Planets | arXiv 1609.01086 | arXiv full text |
