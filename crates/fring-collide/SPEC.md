# Collision specification

Owner: A4. Task K1. Status: **confirmed by AJ on 2026-11-07** (draft 0.3, which applies the A0 review of PR #38: sections 5.1, 5.3, 5.6, 8, 9, 11, 12 and the references). AJ chose the four options of section 13 on 2026-11-07.

This file is a shared interface (`AGENTS.md`). The stochastic model on the CPU (K2), the hard-sphere model on the CPU (K3), the stochastic model on the GPU (K5) and the cross-check (K4) all follow it. To change it after AJ confirms it, write an ICR.

## 1. Summary

- Ring particles are identical smooth hard spheres. A collision reverses the normal part of the relative velocity and multiplies it by the coefficient of restitution ε(vₙ). The tangential part does not change. Momentum is conserved exactly.
- **Stochastic model (primary).** Superparticles are binned into cells in local coordinates. In each cell, pairs are picked with Bird's no-time-counter (NTC) method and collide with a random impact geometry. The relative velocity is the difference of **peculiar velocities** (velocity minus the local circular velocity), so the Keplerian shear and the turn of the orbital direction across a cell do not count as random motion.
- **Hard-sphere model (check).** Real particles, real overlaps, found with a cell list at the end of each step.
- **Shared tests (section 11).** Nine tests, K1-T1 to K1-T9. The main ones: momentum exact to rounding; energy loss per collision against ε; collision rate in a uniform box against kinetic theory (within 2 %); cooling of a uniform box against Haff's law.

## 2. Scope (v1)

In scope: the collision operator, the two models, where they act in the time step, their random numbers, diagnostics and tests.

Not in v1 (AJ, Q1):

- particle spin and surface friction (tangential restitution is 1),
- a size distribution (all particles have one radius s and one weight W),
- the Enskog factor for dense packing (the stochastic model uses the dilute Boltzmann rate; section 10 gives the check),
- sticking, fragmentation and self-gravity (PLAN section 2).

## 3. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| s | radius of one real particle | km |
| W | real particles per superparticle (stochastic model); 1 for hard spheres | 1 |
| σ = π (2s)² | collision cross-section of two real particles | km² |
| uᵢ | peculiar velocity of particle i (section 5.2) | km s⁻¹ |
| g = uᵢ − uⱼ | relative velocity of i with respect to j | km s⁻¹ |
| n̂ | unit normal, from particle j to particle i at contact | 1 |
| gₙ = g · n̂ | normal relative velocity; gₙ < 0 means the pair approaches | km s⁻¹ |
| vₙ = \|gₙ\| | normal impact speed | km s⁻¹ |
| c | one-dimensional velocity dispersion (each component) | km s⁻¹ |
| Ω | local mean motion n at the particle's radius (`physics::circular`) | rad s⁻¹ |
| h | integrator step; Δt_c the collision step (section 7) | s |

All particles have the same mass, so the mass cancels. Energies in this spec are **per unit mass of one superparticle** (km² s⁻²).

## 4. Pair operator (shared by all models)

For a pair (i, j) with g and n̂, and only if gₙ < 0:

```
uᵢ ← uᵢ − ½ (1 + ε) gₙ n̂
uⱼ ← uⱼ + ½ (1 + ε) gₙ n̂
```

with ε = ε(vₙ) from section 6. If gₙ ≥ 0, nothing changes and the pair does not count as a collision.

Results (exact algebra; K1-T1 tests them in floating point):

- uᵢ + uⱼ does not change (momentum).
- The new normal relative velocity is −ε gₙ. The tangential part g − gₙ n̂ does not change.
- The kinetic energy of the pair falls by **ΔK = ¼ (1 − ε²) gₙ²** (per unit mass of one particle). The model adds ΔK to its energy-loss tally (section 9).

The same Δu is added to the full state velocity of each particle (inertial, Saturn-centred). Positions do not change.

Rust reference: `fring_collide::pair::collide` (K1). The WGSL twin (K5) must agree with it to 10⁻⁶ relative (PLAN section 10).

## 5. Stochastic superparticle model (primary)

### 5.1 Cells

The cell of a particle comes from its local coordinates (PLAN 4.2; `physics::frames`): x = ρ − a₀ (radial), y = a₀ φ (along the reference circle, φ the longitude in the rotating frame), z (vertical).

- Three cell sizes: L_x (radial), L_y (azimuthal), L_z (vertical) (AJ, Q2): `cell_radial_km`, `cell_azimuthal_km`, `cell_vertical_km` in `[collisions]` (ICR-009). `every_steps` sets the cadence (section 7).
- Cell index: (⌊x / L_x⌋, ⌊y_c / L_y⌋, ⌊z / L_z⌋), with y_c = a₀ ((φ − φ₀) mod 2π) the arc length from the start of the domain: φ₀ = 0 for the full ring, and φ₀ = arc centre − Δθ/2 (rotating frame, after the D1 wrap) for the arc. So y_c is in [0, D).
- Do not use a₀ (φ mod 2π) wrapped into [0, D) for the arc. It is correct only when 360°/Δθ is an integer. Example: Δθ = 7° with the centre at 0°. Then φ mod 7° maps both [−3°, 0°) and [0°, 3°) into [0°, 3°), so those cells hold particles from two places 3° (7,342 km) apart. The cells for [3.5°, 6.5°) stay empty, and the pair count in the shared cells is 4 times too large.
- L_y is adjusted so that a whole number of cells fills the domain: n_y = round(D / L_y,config), L_y = D / n_y, where D = 2π a₀ for the full ring or the arc length for the arc domain (D1). No cell crosses the wrap.
- Cell volume V_c = L_x L_y L_z.
- Only active particles take part (not dropped or frozen: ICR-006, ICR-004).

### 5.2 Peculiar velocity

For particle i with Saturn-centred inertial position rᵢ and velocity vᵢ:

```
ρᵢ = √(xᵢ² + yᵢ²),   φ̂ᵢ = (−yᵢ, xᵢ, 0) / ρᵢ
uᵢ = vᵢ − v_circ(ρᵢ) φ̂ᵢ
```

where v_circ is `physics::circular::circular_speed` (Saturn with J2, J4, J6, in the equatorial plane). The pair operator works on uᵢ; the same Δu goes to vᵢ.

Why: the model treats the members of a cell as if they were at one point. Two particles at one point see the same circular flow, so their true relative velocity is uᵢ − uⱼ. A plain velocity difference would add the flow difference across the cell as false random motion. At the core (Ω = 1.1756 × 10⁻⁴ rad s⁻¹, orbit 14.85 h, v_circ = 16.48 km s⁻¹), per 100 m of separation:

- Keplerian shear, radial separation: (3/2) Ω Δx = 1.76 cm s⁻¹.
- Turn of the orbital direction, azimuthal separation (inertial or rotating differences): Ω Δy = 1.18 cm s⁻¹.

The core's velocity dispersion is not known; its radial width (under 1 km, `params/ring.toml`) gives c ≲ Ω × 0.5 km = 5.9 cm s⁻¹, and it can be much less. So these false speeds would be as large as the real ones. With peculiar velocities they are zero. Section 10 gives what is left (the core's own eccentricity).

### 5.3 Pair selection (no-time-counter)

In each cell with N_c ≥ 2 members, at each collision step:

1. Members are in order of particle id (determinism, section 5.5).
2. g_max: an upper bound of |uᵢ − uⱼ| over all pairs of the cell at the start of the cell's collision step. Reference: g_max = 2 max |uᵢ − ū| with ū the mean peculiar velocity of the cell (triangle inequality). Any value that is a proven upper bound at the start is allowed. A value that is not is a defect, because it biases the rate.
3. Expected number of candidates: M = ½ N_c (N_c − 1) W σ g_max Δt_c / V_c. Use ⌊M⌋ candidates, plus one more with probability M − ⌊M⌋.
4. For each candidate, in order: pick i uniformly from the N_c members, and j uniformly from the other N_c − 1. Compute g with the current velocities (earlier candidates of this step may have changed them). Accept with probability min(1, |g| / g_max).
5. For an accepted pair, draw the impact geometry (section 5.4) and apply section 4.

**|g| > g_max.** A collision keeps ū, but it can move one particle further from ū. So after an earlier collision in the same step, a candidate can have |g| > g_max. Example: four members with u = (±1, 0, 0) and (0, ±1, 0), so ū = 0 and g_max = 2. An elastic collision of (1, 0, 0) and (0, 1, 0) with n̂ = (0, 1, 0) gives (1, 1, 0) and (0, 0, 0). Then the candidate (1, 1, 0) and (−1, 0, 0) has |g| = √5 = 2.24. Such a candidate is accepted (probability 1) and counted as `gmax_exceeded` (section 9). The rate bias is very small: a particle must collide twice in one step (at most 0.1 collisions per particle per step, section 5.6) and then exceed the bound. If K4 finds that the count is not negligible, 2 (Σₖ |uₖ − ū|²)^½ is a bound for the whole step (the peculiar kinetic energy of a cell cannot increase), but it gives more candidates.

This gives each pair the collision probability P_ij = W σ |g_ij| Δt_c / V_c, and the kinetic-theory rate (appendix A). Note N_c (N_c − 1), not N_c²: with Poisson occupancy, N_c (N_c − 1) is the unbiased estimate of ⟨N_c⟩².

### 5.4 Impact geometry

For an accepted pair, the impact parameter is uniform over the disc of radius 2s normal to g. So, with ĝ = g / |g| and two uniform numbers u₁, u₂ in (0, 1):

```
cos θ = √u₁,  sin θ = √(1 − u₁),  ψ = 2π u₂
n̂ = −cos θ ĝ + sin θ (cos ψ ê₁ + sin ψ ê₂)
```

with (ê₁, ê₂, ĝ) orthonormal. Reference basis: ê₁ = normalise(ĝ × ẑ) if |ĝ_z| < 0.9, else normalise(ĝ × x̂); ê₂ = ĝ × ê₁. Then gₙ = −|g| cos θ < 0, cos² θ is uniform on (0, 1), and ⟨gₙ²⟩ = ½ |g|².

An isotropic n̂ (as in Bird's variable-hard-sphere model for gases) is wrong here: it is equivalent for elastic collisions but gives the wrong energy loss for ε < 1.

### 5.5 Random numbers and determinism

Philox stream `Stream::COLLISIONS` (`fring_model::rng`). For a cell:

- `index` = the smallest particle id in the cell. Each particle is in one cell, so the index is unique in a step and does not depend on the order in which cells are visited.
- `step` = the step count of the run after the integrator step that ends at the collision time.
- `draw 0`: words 0 and 1 give the uniform for the rounding of M.
- `draw 1 + 2k`, candidate k: word 0 gives i, word 1 gives j, words 2 and 3 give the acceptance uniform.
- `draw 2 + 2k`, candidate k: words 0 and 1 give u₁, words 2 and 3 give u₂.
- Integers: i = ⌊word₀ × N_c / 2³²⌋; j' = ⌊word₁ × (N_c − 1) / 2³²⌋, j = j' + 1 if j' ≥ i, else j'. (WGSL: `mulhilo`.)
- f64 uniforms: `u01_f64(hi, lo)` on the two words. f32 uniforms (GPU): `u01_f32` of the first word.
- `draw` has 16 bits, so a cell can have at most 32,767 candidates in one step. More is an error (the cells are too large).

**Determinism (AJ, Q4):** the same config and seed on the same backend give bit-identical results. On the GPU the cell members must be in particle-id order before pair selection (K5-BIN). The CPU and the GPU do not have to agree bit for bit (rounding in the acceptance test differs); they agree statistically (K5).

### 5.6 Limits that a run reports

| Check | Rule | Why |
|---|---|---|
| Overfull cell | W σ g_max Δt_c / V_c ≤ 1 | Else a pair would need P > 1 |
| Collisions per particle per collision step | 2 × collisions / active particles ≤ 0.1 | Repeated collisions in one step are then rare |
| Particles in cells with N_c < 2 | report | They cannot collide |
| Particle-weighted occupancy Σ N_c² / Σ N_c | report; target ≥ 10 | Correlations at low occupancy (K4 measures) |
| Candidates with \|g\| > g_max | report | Rate bias of section 5.3 (K4 measures) |

These are diagnostics, not stops, except the 32,767-candidate limit.

## 6. Restitution laws

From `[collisions.restitution]` (`fring_model::config::Restitution`), with vₙ = |gₙ|:

- `constant`: ε = epsilon.
- `power_law`: ε = (vₙ / v_c)^(−p) if vₙ > v_c, else 1. Bridges, Hatzes and Lin (1984) give p = 0.234 and v_c = 0.0077 cm s⁻¹ for frosted ice particles (radius 2.5 cm) at 210 K (Latter and Ogilvie 2006, eq. 1; `params/sources.md`, section "Collisions: restitution law and method references"); this gives ε = 0.55 at 1 mm s⁻¹, 0.32 at 1 cm s⁻¹ and 0.11 at 1 m s⁻¹. A1 must cite the values in `params/sources.md` before a run uses them.

ε depends on vₙ, not on |g|. K1-T5 checks this.

## 7. Place in the time step

- The collision step acts on the synchronised state at the end of an integrator step (after the full drift-kick-drift, and after all sub-steps of E1). It does not act inside a step.
- It acts every k integrator steps (`every_steps`, default 1; AJ, Q3), with Δt_c = k h. The hard-sphere model always uses k = 1.
- In the arc domain (D1), the wrap happens before the cells are found.

At 150 steps per orbit and k = 1, h = 356 s. For an optical depth τ, the collision frequency of a particle in a thin layer is about 2.55 τ Ω (appendix A), so about 16 τ collisions per orbit: 4.8 per orbit and 0.032 per step for τ = 0.3. So k = 1 is well inside the 0.1 limit of section 5.6, and k = 2 or 3 is possible.

## 8. Hard-sphere model (check)

- Real particles: W = 1, radius s.
- At the end of each integrator step, find all pairs with |rᵢ − rⱼ| < 2s, with a cell list of cell edge ≥ 2s in local coordinates (each pair once; across the y wrap). Along the ring the edge must be ≥ 2s a₀ / ρ_min, with ρ_min the smallest radius of an active particle, because an arc length a₀ φ is longer than the distance at a radius below a₀; the distance of a pair is always the inertial distance of the positions.
- For each overlapping pair, with d = rᵢ − rⱼ and g = vᵢ − vⱼ (inertial; for a touching pair the frame does not change gₙ, because the rotation term Ω × Δr is normal to n̂): skip the pair if d · g ≥ 0 (it separates).
- **Contact normal.** Go back along the straight relative path to the contact: τ > 0 with |d − g τ| = 2s, that is τ = (d·g + √((d·g)² − |g|² (|d|² − 4s²))) / |g|². If τ ≤ h, n̂ = (d − g τ) / |d − g τ|; else (the pair overlapped before this step) n̂ = d / |d|. Then collide (section 4). The relative path over one step is straight to a very good approximation (the tidal acceleration across 2s is tiny), and the contact normal removes a bias: with n̂ = d / |d| a pair that is found deep inside the overlap gets a too grazing normal, so too little energy loss.
- Order: pairs sorted by (min id, max id); each pair at most once per step; d · g is computed again with the current velocities when the pair is processed.
- No position correction. An overlapping pair that separates is left alone. Initial conditions must not overlap: a pair that overlaps from the start, without an approach, is outside the model (the straight path back to the contact is then wrong, and a grazing pair gets a tilted normal).
- Overlap depth δ = 2s − |rᵢ − rⱼ|. A collision with δ > 0.2 s (10 % of the contact distance) counts as **deep**. A run with more than 1 % deep collisions has too large a step for its speeds.
- **Step size.** A pair is found only if a step ends while the pair is on the approach half of its chord through the contact sphere (pairs with d · g ≥ 0 are skipped). For the impact parameter b uniform over the disc, that half has the length 2s √p, with p = 1 − b² / 4s² uniform on (0, 1). The step ends at a random point of the path, and the pair moves |g| h in one step, so the pair is missed with probability max(0, 1 − 2s √p / (|g| h)). Over p (for |g| h ≤ 2s) the missed fraction is (1/3)(|g| h / 2s)² = (4/3)(|g| h / 4s)² for each g. Averaged over collisions (weight g, so a factor π/2) it is 2.09 (⟨g⟩ h / 4s)². So ⟨g⟩ h = 0.1 s (the setup of the K1 tests) misses 0.13 %, and ⟨g⟩ h = 0.4 s misses 2.1 %.

## 9. Diagnostics

Per stats interval, each model reports (ICR-009 adds them to `StepStats`):

- collisions (accepted pairs),
- energy loss: Σ ¼ (1 − ε²) gₙ² over the collisions, per unit superparticle mass (km² s⁻²),
- stochastic: candidates, occupied cells, Σ N_c and Σ N_c² over cells and steps, particles in cells with N_c < 2, overfull cells, candidates with |g| > g_max,
- hard-sphere: deep collisions.

## 10. Choosing the parameters (guidance for K4, D2 and S1)

**Cell size and rate.** A cell averages the density over its size. For a Gaussian profile with RMS width w along one axis, the rate is low by about L² / (24 w²) (appendix A): 4.2 % for L = w, 1.0 % for L = w/2, 0.46 % for L = w/3. Use L_x ≤ w_x / 2 and L_z ≤ H / 2, with H the RMS thickness (about c / Ω: 85 m for c = 1 cm s⁻¹).

**What peculiar velocities do not remove.** The core has a forced eccentricity e = 0.00235. Its streamline is tilted against the reference circle by up to e: across L_y it moves radially by e L_y, and its velocity changes by about e Ω L_y. Per 10 km of L_y: 23.5 m and 2.8 mm s⁻¹. So use e L_y ≤ L_x / 2 and e Ω L_y ≤ 0.3 c. For L_x = 50 m and c = 1 cm s⁻¹, both give L_y ≲ 10 km. (Cells that follow the core's streamline would remove this; not in v1.)

**Occupancy.** In a 5° arc (12,237 km) with 10⁶ superparticles there are 81.7 per km of arc; in the full ring (881,044 km) only 1.1 per km. For a Gaussian core with w_x = 300 m, L_x = 50 m, L_z = H/2 and L_y = 10 km, the particle-weighted occupancy is about 817 × 50/(√(4π) 300) × 0.5/√(4π) ≈ 5.4. So a resolved core needs the arc domain, and 10⁷ superparticles (occupancy about 54) are much better than 10⁶. Column cells (x and y only) would raise the occupancy about 7 times; AJ chose box cells for v1 (Q2), so column cells would need an ICR.

**Superparticle weight.** Only W σ enters the stochastic model. For a core of width 1 km and τ = 0.3, the number of real particles of radius 1 cm is τ × width × circumference / (π s²) = 0.3 × 1 × 881,044 / (π × 10⁻¹⁰) ≈ 8.4 × 10¹⁴, so W ≈ 8 × 10⁸ for 10⁶ superparticles in the full ring.

**Dilute limit.** The stochastic model has no Enskog factor. The volume filling factor D = n_real (4/3) π s³ must be small: the rate is low by about 2.5 D (Carnahan and Starling 1969), so D ≤ 0.004 keeps it under 1 %. Each run reports D at the midplane (from W, s and the cell counts).

## 11. Shared statistical tests

The tests are in `fring_collide::testkit` (K1). Each model runs them through one trait, so K2, K3 and K5 use the same code and numbers.

**Box.** A periodic cube with no forces. Positions are uniform, velocities are Maxwellian with dispersion c and zero mean. Positions stream freely (x += u Δt, then wrap). The stochastic model divides the box into cells; the hard-sphere model uses its cell list. A box test calls the model with box positions as local coordinates and box velocities as peculiar velocities.

**Sample-size rule (M17).** Fixed seeds. Each statistical test asserts a minimum sample so that the standard error (SE) is at most one fifth of its tolerance, and prints the sample size, the measured value, the theory value and the SE. These are the **full** tests; they take seconds to minutes in release mode, so they are `#[ignore]` and run with `--release -- --ignored`. Each also has a **quick** form for every `cargo test` (SE ≤ tolerance / 2, error within 5 SE), which finds large mistakes only.

**Setups** (`testkit::Setup`; c = 1 cm s⁻¹ at the start in all of them). Stochastic: cubic cells of 1 km, a step in which a particle moves 0.1 cell, and W chosen so that a particle collides with probability 0.01 to 0.02 per step (0.05 in T9). Hard spheres: s = 1 m, ⟨g⟩ h = 0.1 s, filling factor D = 0.005 (T4, T6), 0.01 (T7, T9) or 0.05 (T5, where D does not enter the theory).

| ID | Test | Theory | Tolerance |
|---|---|---|---|
| K1-T1 | Pair operator on 10⁵ random pairs and edge cases (ε = 0 and 1, gₙ = 0, \|u\| = 16.5 km s⁻¹ with differences of 1 mm s⁻¹) | Section 4 | Momentum: each component of Δ(uᵢ + uⱼ) ≤ 4 ε_mach max(\|uᵢ\|, \|uⱼ\|). Normal reversal, tangential part and ΔK: 10⁻¹² relative plus the same rounding term. Receding pairs: unchanged, bit for bit |
| K1-T2 | Impact geometry, 10⁶ draws for fixed g | cos² θ and ψ / 2π uniform on (0, 1); \|n̂\| = 1 | Kolmogorov-Smirnov statistic below the 0.1 % critical value (1.95 / √N) for each; \|n̂\| − 1 ≤ 10⁻¹⁴ |
| K1-T3 | Restitution laws | Section 6 | Exact values at chosen points; ε = 1 at and below v_c; continuous at v_c |
| K1-T4 | Collision rate in a box, ε = 1 | Events per step = Σ_cells ½ N_c (N_c − 1) W σ ⟨g⟩ Δt / V_c, ⟨g⟩ = 4c/√π. Hard spheres: ½ N (N − 1) σ ⟨g⟩ χ Δt / V with χ = (1 − D/2)/(1 − D)³ and D ≤ 0.005 | 2 % (PLAN section 10). Stochastic: mean occupancy 2, 5 and 20 (2 catches the N_c² mistake, which is 50 % high there) |
| K1-T5 | Energy loss per collision against ε. New Maxwellian for each trial, at most 1 % of particles collide per trial | Constant ε: (1 − ε²) c² per collision (appendix A), for ε in {0, 0.5, 0.9, 1}. One power-law case: the integral of appendix A, by quadrature | 2 %; ε = 1 gives 0 exactly |
| K1-T6 | Momentum and energy over a box run (10³ steps) | Total momentum constant; ε = 1: energy constant | Momentum: ≤ 4 ε_mach max\|u\| × collisions per component. Energy: ε = 1, 10⁻¹² relative; ε = 0.5, K₀ − K_end = Σ ΔK (the tally) to 10⁻¹⁰ relative |
| K1-T7 | Homogeneous cooling, ε = 0.9. The step grows as 1/c, so each step has about the same number of collisions | Haff (1983): d(c²)/dt = −(1/3)(1 − ε²) ω c², so T(t) = T₀ / (1 + t / t₀)² with t₀ = 6 / ((1 − ε²) ω₀) and ω₀ = n σ 4c₀/√π (× χ for hard spheres). The theory is integrated over the same steps; for the stochastic model with the pairs of each cell at each step. The non-Gaussian correction is 0.3 % at ε = 0.9 (van Noije and Ernst 1998) | 3 % on T / T₀ at t = t₀ (T / T₀ = 1/4). SE from Σ ΔK² / K₀. The full form uses about 7.2 × 10⁴ particles: SE ≈ 1.4 / √N of T(t₀), so about 0.51 % against the limit 0.6 % |
| K1-T8 | Frames (production path, each backend) | (a) a particle on a circular equatorial orbit has u = 0; (b) a cold ring of circular orbits stays cold with collisions on; (c) cell index at the y wrap, at negative x and z, and n_y L_y = D | (a) \|u\| ≤ 10⁻¹³ km s⁻¹ (f64); (b) after 10 orbits the peculiar velocity dispersion with collisions is within 10⁻¹² km s⁻¹ of the run without; (c) exact |
| K1-T9 | Determinism | Same seed: same velocities, bit for bit. Another seed: other velocities, and K1-T4 still passes | Exact |

The GPU (K5) runs K1-T1 (through the WGSL twin), T4 to T7 and T9 on its own path, and T8 on the GPU state. "CPU and GPU agree statistically" means: each GPU result is within its tolerance, and each CPU and GPU pair of results is within 4 combined SE.

Box velocities have a zero mean, so the box does not test the precision rules of section 12. K5 therefore also runs T6 with a bulk velocity of 16.5 km/s added to the double-single velocities (the adapter adds it before the upload and removes it after the download), with ε_mach in the bound replaced by the unit round-off of the double-single arithmetic (ADR 0006). An f32 velocity update fails this test at once: one f32 step at 16.5 km/s is 1.9 mm/s. T4, T5 and T7 cannot see such errors at their tolerances.

K1 also adds two slow reference models (`testkit::reference`, test use only): `AllPairs` (each pair of a cell collides with probability P_ij, without the NTC method) and `HardSphere` (section 8 in the box). The testkit passes with both, so a failing K2, K3 or K5 points to the model, not the test. K1-T8 (b) needs the integrator, so K2 and K5 write it.

## 12. Precision

- CPU: f64 throughout.
- GPU: the speed of a particle is 16.5 km s⁻¹ and one f32 step at that speed is 1.9 mm s⁻¹, which is as large as the peculiar velocities. So the GPU must compute uᵢ = vᵢ − v_circ φ̂ᵢ from the double-single state (error ≤ 10⁻³ c of the run), and add Δu to the double-single velocity. The same holds for the cell coordinates: f32 resolves ρ only to 15.6 m at 140,222 km, so x = ρ − a₀ and y come from the double-single position.
- GPU, f32 velocities: also subtract the cell mean ū in double-single, and convert only uᵢ − ū to f32. Then uᵢ − ū, g and Δu may be f32. Reason: the core has a forced eccentricity, so |uᵢ| is up to e v_circ = 0.00235 × 16.48 km s⁻¹ = 38.7 m s⁻¹, not about c. The f32 spacing there is 2⁻²⁸ km s⁻¹ = 3.7 µm s⁻¹, which is more than 10⁻³ c for c < 3.7 mm s⁻¹. The effect is very small: a random rounding δ of g heats the pair by ¼ (1 + ε)² δₙ², which is (1 + ε)/(1 − ε) ⟨δₙ²⟩ / ⟨gₙ²⟩ of the loss. With rms δₙ ≈ 1.5 µm s⁻¹ and ⟨gₙ²⟩ = 4c² (appendix A), that is 2 × 10⁻⁶ at c = 1 mm s⁻¹ and ε = 0.5. But the subtraction is almost free, because 5.3 needs ū for g_max, and it keeps the rule of 10⁻³ c true.

## 13. AJ decisions (2026-11-07)

AJ chose the recommended option of each question in the Claude Code session of 2026-11-07.

- **Q1, physics scope: smooth identical spheres.** No spin or friction, one size, no Enskog factor (section 2). Not chosen: friction and spin (a spin state per particle), a size distribution (weight and radius per particle).
- **Q2, cell shape: box cells with three sizes** L_x, L_y, L_z (ICR-009). Not chosen: column cells (about 7 times the occupancy, but a Gaussian vertical profile is assumed in each cell); one cube size (too few particles per cell).
- **Q3, collision cadence: `every_steps`, default 1.** Not chosen: every step only.
- **Q4, determinism: required on each backend.** The GPU sorts cell members by id. Not chosen: statistics only.

## 14. Interface changes

ICR-009 (`docs/interfaces/ICR-009.md`, accepted with amendments by A0 on 2026-11-07): three cell sizes and `every_steps` in `[collisions]`; collision diagnostics in `StepStats` (with `gmax_exceeded`, amendment A2); `fring-collide` as a dependency of `fring-cpu` and `fring-gpu` (it must then build for wasm32).

## Appendix A. Derivations

**Rate.** A superparticle represents W real particles, so the real density in a cell is n = W N_c / V_c. The real collision rate in the cell is ½ n² σ ⟨g⟩ V_c, and one superparticle event stands for W real collisions. So the expected number of events is ½ W N_c² σ ⟨g⟩ Δt / V_c; with self-pairs removed, ½ N_c (N_c − 1) W σ ⟨g⟩ Δt / V_c, which is what 5.3 gives: M candidates, each a given pair with probability 1 / (½ N_c (N_c − 1)), accepted with |g_ij| / g_max. For Maxwellian particles with dispersion c, g is Maxwellian with dispersion √2 c, so ⟨g⟩ = √(8/π) √2 c = 4c / √π = 2.2568 c.

**Energy loss per collision.** Collisions select pairs in proportion to |g|. For a Maxwellian g with dispersion σ_g, ⟨g³⟩ / ⟨g⟩ = 4 σ_g² = 8 c². With cos² θ uniform, ⟨gₙ²⟩ = ½ × 8 c² = 4 c², so the mean ΔK per collision is ¼ (1 − ε²) × 4 c² = (1 − ε²) c². For ε(vₙ), the mean is ∫∫ ¼ (1 − ε(g μ)²) g² μ² f(g) g dg dμ² / ∫ f(g) g dg with μ = cos θ, μ² uniform, and f the Maxwellian of g.

**Haff's law.** The kinetic energy is (3/2) N c². Collision events happen at ½ N ω, each with loss (1 − ε²) c², so d(c²)/dt = −(1/3)(1 − ε²) ω c². With ω = n σ 4c/√π proportional to c, c(t) = c₀ / (1 + t/t₀) with t₀ = 6 / ((1 − ε²) ω₀), and T ∝ c².

**Collision frequency in a thin layer.** τ = (number per area) π s². With a Gaussian layer of thickness H = c/Ω, the midplane density is τ / (π s² √(2π) H), so ω = n σ ⟨g⟩ = 16 τ Ω / (√2 π) = 3.60 τ Ω at the midplane, and 3.60 / √2 = 2.55 τ Ω averaged over the particles of the layer.

**Cell averaging.** For a smooth density n along one axis and cells of length L, Σ_cells (cell mean)² L = ∫ n² − (L²/12) ∫ n'² to leading order (average over grid offsets). For a Gaussian with RMS w, ∫ n'² / ∫ n² = 1 / (2 w²), so the rate is low by L² / (24 w²).

## References

The K1 session did not open these; they are standard results. A1 opens them and cites the Bridges et al. values in `params/sources.md` in task K1-REFS, before gate G4 (A0 decision, 2026-11-07). No run uses the power law before that citation (section 6).

- Bird, G. A. 1994. Molecular Gas Dynamics and the Direct Simulation of Gas Flows. Clarendon Press, Oxford. (NTC method.)
- Bridges, F. G., Hatzes, A., Lin, D. N. C. 1984. Structure, stability and evolution of Saturn's rings. Nature 309, 333.
- Carnahan, N. F., Starling, K. E. 1969. Equation of state for nonattracting rigid spheres. J. Chem. Phys. 51, 635.
- Haff, P. K. 1983. Grain flow as a fluid-mechanical phenomenon. J. Fluid Mech. 134, 401.
- van Noije, T. P. C., Ernst, M. H. 1998. Velocity distributions in homogeneous granular fluids: the free and the heated case. Granular Matter 1, 57.
