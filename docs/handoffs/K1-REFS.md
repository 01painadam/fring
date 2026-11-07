# K1-REFS: open the five references of fring-collide/SPEC.md
Role: A1  Branch: ws/a1-k1-refs  Status: review
Updated: 2026-11-07

## Goal and acceptance criteria
Open the five references of `fring-collide/SPEC.md` (Bird 1994, Bridges et al. 1984, Carnahan and Starling 1969, Haff 1983, van Noije and Ernst 1998). Cite the Bridges et al. power-law values in `params/sources.md`. No run may use `power_law` before this citation exists.

## Done
- New section "Collisions: restitution law and method references" in `params/sources.md`.
- Bridges et al. values p = 0.234 and v_c = 0.0077 cm/s = 7.7e-8 km/s: verified (secondary, quoted in Latter and Ogilvie 2006, arXiv astro-ph/0609164, full text read). The Nature paper was not opened (ADS page returned HTTP 405).
- eps check, (v/v_c)^(-0.234): 1 mm/s: ratio 12.987, eps 0.5488; 1 cm/s: ratio 129.87, eps 0.3202; 1 m/s: ratio 12987, eps 0.1090. SPEC says 0.55, 0.32, 0.11: agrees.
- Form: LO06 Eq. (1) is eps = (v_n/v_c)^(-p) for v_n > v_c, 1 for v_n <= v_c, v_n the normal speed. It is given for Bridges, Hatzes and Supulver together. That Bridges et al. print the cut themselves is UNVERIFIED.

## Findings for A0 (SPEC disagreements)
1. SPEC section 6 says "frosty ice at low temperature". LO06 says the Bridges values are for frosted particles of radius 2.5 cm at 210 K, "significantly higher than the appropriate conditions". Another review (search result, not opened) puts the experiments near 170 K. Suggest A4 words it as "frosted ice spheres, 210 K (per Latter and Ogilvie 2006)".
2. Same sentence: LO06 gives other fits: Hatzes 1988 p = 0.20, v_c = 0.025 cm/s (123 K); Supulver 1995 p = 0.19, v_c = 0.029 cm/s (about 100 K). These are closer to ring temperature and are less dissipative. The Bridges law is the most dissipative. A4/AJ may want a sensitivity run.
3. Carnahan-Starling: the paper gives the equation of state only. The "2.5 D" rate correction in SPEC is the contact value g = (1 - eta/2)/(1 - eta)^3 = 1 + 2.5 eta + ..., which follows from Z = 1 + 4 eta g. Correct, but derived, not in the paper. Full text not opened.
4. Haff 1983: full text paywalled. The t0 = 6/((1 - eps^2) omega_0) form is K1's own derivation. The SPEC wording "Haff (1983)" in K1-T7 is fair for the law T ~ t^-2, but the constants are not from Haff.
5. Bird 1994 (book) and Nature 1984 were not opened. NTC description checked only in secondary pages.
6. van Noije and Ernst: the abstract supports the claim that non-Gaussian corrections change the cooling rate little.

## Next step (one concrete action)
A0 re-runs the checks and opens the pull request. Then L3-PARAMS.

## How to verify the current state
`grep -n "Collisions: restitution" params/sources.md` shows the section. `python3 -c "print((0.1/0.0077)**-0.234)"` prints 0.5488.

## Open questions / blockers
- Primary text of Bridges et al. 1984 (Nature) not read. Status stays "verified (secondary)".

## Decisions made (link ADRs)
- None.
