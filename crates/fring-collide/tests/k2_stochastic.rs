//! K2: the stochastic model on the CPU (`SPEC.md` sections 5 and 11).
//!
//! The quick tests run in every `cargo test`. The full tests (sample-size rule
//! of SPEC 11) are long; run them with
//! `cargo test -p fring-collide --release -- --ignored --nocapture`.

use std::f64::consts::{PI, TAU};

use fring_collide::frame::peculiar_velocity;
use fring_collide::stochastic::{
    CellList, CellScratch, Domain, MAX_CANDIDATES, State, StochasticModel,
};
use fring_collide::testkit::{
    ModelKind, Sample, t4_rate, t5_energy_loss, t6_budget, t7_haff, t9_determinism,
};
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::circular::circular_speed;
use fring_model::physics::frames::{reference_mean_motion, rotating_to_inertial};
use fring_model::physics::gravity::SaturnField;
use fring_model::rng::{Counter, Philox, Stream, box_muller, u01_f64};

const S: ModelKind = ModelKind::Stochastic;
const A0_KM: f64 = 140_222.43;

fn model(p: fring_collide::testkit::ModelParams) -> StochasticModel {
    StochasticModel::from_params(p)
}

fn power_law() -> Restitution {
    Restitution::PowerLaw {
        reference_speed_km_s: 1e-6,
        exponent: 0.234,
    }
}

fn saturn() -> SaturnField<f64> {
    let p = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    SaturnField::from_params(&p)
}

// ---- K1-T4 to T7 and T9 on the production model, quick ----

#[test]
fn quick_t4_rate() {
    for occupancy in [2.0, 5.0] {
        t4_rate(S, occupancy, Sample::Quick, model).assert_ok();
    }
}

#[test]
fn quick_t5_energy_loss() {
    for e in [0.5, 1.0] {
        let law = Restitution::Constant { epsilon: e };
        t5_energy_loss(S, law, Sample::Quick, model).assert_ok();
    }
}

#[test]
fn quick_t6_budget() {
    t6_budget(S, 0.5, model).assert_ok();
    t6_budget(S, 1.0, model).assert_ok();
}

#[test]
fn quick_t7_haff() {
    t7_haff(S, Sample::Quick, model).assert_ok();
}

#[test]
fn k1_t9_determinism() {
    t9_determinism(S, model);
}

// ---- full (long) ----

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t4_rate() {
    for occupancy in [2.0, 5.0, 20.0] {
        t4_rate(S, occupancy, Sample::Full, model).assert_ok();
    }
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t5_energy_loss() {
    for e in [0.0, 0.5, 0.9, 1.0] {
        let law = Restitution::Constant { epsilon: e };
        t5_energy_loss(S, law, Sample::Full, model).assert_ok();
    }
    t5_energy_loss(S, power_law(), Sample::Full, model).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t6_budget() {
    t6_budget(S, 0.5, model).assert_ok();
    t6_budget(S, 1.0, model).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t7_haff() {
    t7_haff(S, Sample::Full, model).assert_ok();
}

// ---- the cell function ----

fn u_cloud(n: usize, c: f64, seed: u64) -> Vec<Vec3<f64>> {
    let rng = Philox::from_seed(seed);
    (0..n)
        .map(|i| {
            let a = rng.block(Counter {
                index: i as u32,
                step: 0,
                stream: Stream::TEST,
                draw: 0,
            });
            let b = rng.block(Counter {
                index: i as u32,
                step: 0,
                stream: Stream::TEST,
                draw: 1,
            });
            let (x, y) = box_muller(u01_f64(a[0], a[1]), u01_f64(a[2], a[3]));
            let (z, _) = box_muller(u01_f64(b[0], b[1]), u01_f64(b[2], b[3]));
            Vec3::new(x, y, z) * c
        })
        .collect()
}

fn g_max_of(u: &[Vec3<f64>]) -> f64 {
    let mut m = Vec3::new(0.0, 0.0, 0.0);
    for &v in u {
        m += v;
    }
    m = m * (1.0 / u.len() as f64);
    2.0 * u.iter().map(|&v| (v - m).norm()).fold(0.0, f64::max)
}

#[test]
fn cell_edge_cases() {
    let m = StochasticModel::new(1e-6, 1e9, Restitution::Constant { epsilon: 0.5 }, 1);
    let ids = [3_u32, 5, 9];
    let zero = Vec3::new(0.0, 0.0, 0.0);

    // Empty cell.
    let t = m.process_cell(1.0, 1e4, 0, &[], &mut [], &mut []).unwrap();
    assert_eq!(t, Default::default());

    // One particle: counted as lonely, no candidates.
    let mut u = [Vec3::new(1e-5, 0.0, 0.0)];
    let mut d = [zero];
    let t = m
        .process_cell(1.0, 1e4, 0, &ids[..1], &mut u, &mut d)
        .unwrap();
    assert_eq!(t.stats.lonely_particles, 1);
    assert_eq!(t.stats.occupied_cells, 1);
    assert_eq!(t.stats.occupancy_sq_sum, 1);
    assert_eq!((t.stats.candidates, t.collisions), (0, 0));
    assert_eq!(d[0], zero);

    // Equal velocities: g_max = 0, no candidates, nothing changes.
    let mut u = [Vec3::new(1e-5, 2e-5, 0.0); 3];
    let mut d = [zero; 3];
    let t = m.process_cell(1.0, 1e4, 0, &ids, &mut u, &mut d).unwrap();
    assert_eq!((t.stats.candidates, t.collisions), (0, 0));
    assert_eq!(t.stats.occupancy_sum, 3);
    assert_eq!(d, [zero; 3]);
    assert_eq!(u, [Vec3::new(1e-5, 2e-5, 0.0); 3]);

    // Two particles, certain collision (W σ g_max Δt / V_c = 1.5 > 1: overfull,
    // reported, and the cell still runs). Momentum and delta agree.
    let m1 = StochasticModel::new(1e-6, 1e9, Restitution::Constant { epsilon: 0.0 }, 1);
    let u0 = [Vec3::new(1e-5, 0.0, 0.0), Vec3::new(-1e-5, 0.0, 0.0)];
    let w_sigma_dt = 1e9 * m1.cross_section_km2() * 2e-5 * 1e4;
    let volume = w_sigma_dt / 1.5;
    let (mut u, mut d) = (u0, [zero; 2]);
    let t = m1
        .process_cell(volume, 1e4, 7, &ids[..2], &mut u, &mut d)
        .unwrap();
    assert_eq!(t.stats.overfull_cells, 1);
    assert!(t.stats.candidates >= 1);
    assert_eq!(d[0] + d[1], zero);
    assert_eq!(u[0], u0[0] + d[0]);
    assert_eq!(u[1], u0[1] + d[1]);
    assert!(t.collisions >= 1);
}

#[test]
fn candidate_limit_is_an_error_and_the_boundary_is_allowed() {
    let n = 300;
    let u = u_cloud(n, 1e-5, 5);
    let ids: Vec<u32> = (0..n as u32).collect();
    let m0 = StochasticModel::new(1e-6, 1.0, Restitution::Constant { epsilon: 1.0 }, 1);
    let dt = 1.0;
    let per_unit_w = m0.cross_section_km2() * g_max_of(&u) * dt;
    let pairs = 0.5 * (n * (n - 1)) as f64;
    // Weight so that the expected number of candidates is `target`.
    let with_m = |target: f64| {
        StochasticModel::new(
            1e-6,
            target / (pairs * per_unit_w),
            Restitution::Constant { epsilon: 1.0 },
            1,
        )
    };
    let run = |m: &StochasticModel| {
        let (mut uu, mut d) = (u.clone(), vec![Vec3::new(0.0, 0.0, 0.0); n]);
        m.process_cell(1.0, dt, 0, &ids, &mut uu, &mut d)
    };
    // The limit: 32,767 candidates is the largest allowed (draw 2 × 32,766 + 2 = 65,534 fits 16 bits).
    for step in [32_700.4, 32_766.0] {
        let t = run(&with_m(step)).expect("below the limit");
        assert!(t.stats.candidates <= MAX_CANDIDATES);
        assert!(t.stats.candidates as f64 >= step.floor());
    }
    let err = run(&with_m(32_767.5)).expect_err("above the limit");
    assert_eq!(err.smallest_id, 0);
    assert_eq!(err.members, n);
    assert!(err.to_string().contains("too large"));
}

// ---- the production path in local coordinates ----

/// A ring at the core: `n` particles in a band of `width_km` around `a0`, over
/// `arc_km` of longitude (rotating frame, at t = 0), with random peculiar
/// velocities of dispersion `c_km_s`.
fn ring_state(
    field: &SaturnField<f64>,
    n: usize,
    width_km: f64,
    arc_km: f64,
    c: f64,
) -> Vec<State> {
    let rng = Philox::from_seed(77);
    let u = u_cloud(n, c, 11);
    (0..n)
        .map(|i| {
            let w = rng.block(Counter {
                index: i as u32,
                step: 0,
                stream: Stream::TEST,
                draw: 5,
            });
            let r = A0_KM + (u01_f64(w[0], w[1]) - 0.5) * width_km;
            let phi = u01_f64(w[2], w[3]) * arc_km / A0_KM;
            let vc = circular_speed(field, r);
            let (cp, sp) = (phi.cos(), phi.sin());
            let pos = Vec3::new(r * cp, r * sp, 0.0);
            let vel = Vec3::new(-sp, cp, 0.0) * vc + u[i];
            [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z]
        })
        .collect()
}

fn domain(field: &SaturnField<f64>) -> Domain {
    let n0 = reference_mean_motion(field, A0_KM);
    Domain::full_ring(0.05, 10.0, 0.04, A0_KM, n0).unwrap()
}

#[test]
fn cell_keys_follow_the_rotating_frame() {
    let field = saturn();
    let d = domain(&field);
    // A point at a cell centre of the rotating frame; at time t the inertial
    // position is the rotated one. The key must not depend on t, also for
    // very large n0 t.
    let (x, z) = (0.025, 0.02);
    for cell in [0_u32, 1, 100, d.grid.azimuthal_cells - 1] {
        let y_c = (f64::from(cell) + 0.5) * d.grid.azimuthal_km;
        // y_c is the arc from φ0 = 0; the longitude may be above π (then y is negative).
        let phi = y_c / A0_KM;
        let rot = Vec3::new((A0_KM + x) * phi.cos(), (A0_KM + x) * phi.sin(), z);
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let mut first = None;
        for t in [0.0, 1.0e3, 3.15e7, 3.0e8, 1.0e9] {
            let (p, _) = rotating_to_inertial(rot, zero, d.n0_rad_s, t);
            let k = d.cell_key(p, t);
            assert_eq!((k.radial, k.vertical, k.azimuthal), (0, 0, cell), "t = {t}");
            first = first.or(Some(k));
            assert_eq!(Some(k), first);
        }
    }
}

#[test]
fn inactive_particles_take_no_part() {
    let field = saturn();
    let d = domain(&field);
    let state = ring_state(&field, 400, 0.2, 40.0, 1e-5);
    let ids: Vec<u32> = (0..400).collect();
    let active: Vec<bool> = (0..400).map(|i| i % 3 != 0).collect();
    let cells = d.build_cells(0.0, &ids, Some(&active), |i| {
        Vec3::new(state[i][0], state[i][1], state[i][2])
    });
    let all = d.build_cells(0.0, &ids, None, |i| {
        Vec3::new(state[i][0], state[i][1], state[i][2])
    });
    let count = |c: &CellList| (0..c.len()).map(|k| c.members(k).len()).sum::<usize>();
    assert_eq!(count(&cells), active.iter().filter(|&&a| a).count());
    assert_eq!(count(&all), 400);
    for k in 0..cells.len() {
        let m = cells.members(k);
        assert!(m.iter().all(|&i| active[i as usize]));
        assert!(
            m.windows(2)
                .all(|w| ids[w[0] as usize] < ids[w[1] as usize])
        );
    }
    // A step leaves the inactive particles unchanged.
    let model = StochasticModel::new(1e-5, 2e6, Restitution::Constant { epsilon: 0.5 }, 3);
    let mut s2 = state.clone();
    let t = model
        .collide_step(&d, &field, 0.0, 1, 356.0, &ids, Some(&active), &mut s2)
        .unwrap();
    assert!(t.collisions > 0, "the test needs collisions");
    for i in (0..400).filter(|i| !active[*i]) {
        assert_eq!(s2[i], state[i]);
    }
}

#[test]
fn cell_order_does_not_matter_and_ids_decide() {
    let field = saturn();
    let d = domain(&field);
    let n = 600;
    let state = ring_state(&field, n, 0.2, 40.0, 1e-5);
    let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + 7 * i).collect();
    let model = StochasticModel::new(1e-5, 2e7, Restitution::Constant { epsilon: 0.5 }, 9);
    let (t_s, step, dt) = (5.0e5, 1234, 356.0);
    let cells = d.build_cells(t_s, &ids, None, |i| {
        Vec3::new(state[i][0], state[i][1], state[i][2])
    });
    let volume = d.grid.volume_km3();
    let mut deltas: Vec<Option<Vec<Vec3<f64>>>> = vec![None; cells.len()];
    let mut total_rev = fring_collide::tally::Tally::default();
    let mut scratch = CellScratch::default();
    for c in (0..cells.len()).rev() {
        total_rev += model
            .process_cell_states(
                &field,
                volume,
                dt,
                step,
                &ids,
                &state,
                cells.members(c),
                &mut scratch,
            )
            .unwrap();
        deltas[c] = Some(scratch.delta_km_s.clone());
    }
    let mut s_fwd = state.clone();
    let total_fwd = model
        .collide_step(&d, &field, t_s, step, dt, &ids, None, &mut s_fwd)
        .unwrap();
    assert!(
        total_fwd.collisions > 20,
        "collisions: {}",
        total_fwd.collisions
    );
    // Same sums (the energy sum may differ in the last bits: other order).
    assert_eq!(total_fwd.collisions, total_rev.collisions);
    assert_eq!(total_fwd.stats.candidates, total_rev.stats.candidates);
    // Same velocity changes, bit for bit.
    let mut s_rev = state.clone();
    for (c, delta) in deltas.iter().enumerate() {
        for (&m, dv) in cells.members(c).iter().zip(delta.as_ref().unwrap()) {
            let s = &mut s_rev[m as usize];
            s[3] += dv.x;
            s[4] += dv.y;
            s[5] += dv.z;
        }
    }
    assert_eq!(s_fwd, s_rev);
    // Particles in another array order (same ids) get the same velocities.
    let perm: Vec<usize> = (0..n).rev().collect();
    let state_p: Vec<State> = perm.iter().map(|&i| state[i]).collect();
    let ids_p: Vec<u32> = perm.iter().map(|&i| ids[i]).collect();
    let mut s_p = state_p.clone();
    model
        .collide_step(&d, &field, t_s, step, dt, &ids_p, None, &mut s_p)
        .unwrap();
    for (k, &i) in perm.iter().enumerate() {
        assert_eq!(s_p[k], s_fwd[i], "particle {i}");
    }
    // Positions never change.
    for i in 0..n {
        assert_eq!(s_fwd[i][..3], state[i][..3]);
    }
}

#[test]
fn full_velocity_momentum_is_conserved() {
    // Δv is added to the inertial velocity of 16.5 km/s; the sum over the
    // ring must stay constant to the rounding of that addition (SPEC 12).
    let field = saturn();
    let d = domain(&field);
    let n = 2000;
    let state = ring_state(&field, n, 0.2, 40.0, 1e-5);
    let ids: Vec<u32> = (0..n as u32).collect();
    let model = StochasticModel::new(1e-5, 2e6, Restitution::Constant { epsilon: 0.5 }, 4);
    let mut s = state.clone();
    let t = model
        .collide_step(&d, &field, 0.0, 1, 356.0, &ids, None, &mut s)
        .unwrap();
    assert!(t.collisions > 50, "collisions: {}", t.collisions);
    // Sum of the changes, not the change of a sum: a sum of 2000 velocities of
    // 16.5 km/s has a rounding of its own that is larger than the bound.
    let dp = s
        .iter()
        .zip(&state)
        .fold(Vec3::new(0.0, 0.0, 0.0), |a, (x, y)| {
            a + Vec3::new(x[3] - y[3], x[4] - y[4], x[5] - y[5])
        });
    // 2 additions of rounding per collision at most, each 0.5 ulp of 16.5 km/s.
    let bound = 2.0 * t.collisions as f64 * 16.5 * f64::EPSILON;
    println!(
        "K2 momentum: |Δ Σv| = {:e}, bound {:e}, collisions {}",
        dp.norm(),
        bound,
        t.collisions
    );
    assert!(dp.x.abs() <= bound && dp.y.abs() <= bound && dp.z.abs() <= bound);
}

/// K1-T8b. A cold ring of circular orbits stays cold with collisions on.
///
/// The orbits are advanced by an exact test-only step: a particle on a circular
/// equatorial orbit of radius r turns at ω = v_circ(r) / r, so its position and
/// velocity rotate about z by ω Δt. (`fring_model::physics` has no exact
/// circular drift: the leapfrog orbit differs from `circular_speed` by
/// (n h)², about 10⁻³ relative, which is larger than the 10⁻¹² km/s of the test.)
/// The shear moves neighbours through the cells, so the cells, the frame
/// conversion and the peculiar velocity are all in use.
#[test]
fn k1_t8b_cold_ring_stays_cold() {
    let field = saturn();
    let d = domain(&field);
    let n = 3000;
    let ids: Vec<u32> = (0..n as u32).collect();
    let cold = ring_state(&field, n, 0.3, 30.0, 0.0);
    // W is large: any g of 10⁻¹² km/s or more would give collisions.
    let model = StochasticModel::new(1e-5, 1e14, Restitution::Constant { epsilon: 0.5 }, 21);
    let h = 356.0;
    let orbits = 10.0;
    let steps = (orbits * TAU / (d.n0_rad_s * h)).round() as u64;
    let rotate = |s: &mut State, dt: f64| {
        let r = s[0].hypot(s[1]);
        let a = circular_speed(&field, r) / r * dt;
        let (c, sn) = (a.cos(), a.sin());
        let rot = |x: f64, y: f64| (c * x - sn * y, sn * x + c * y);
        (s[0], s[1]) = rot(s[0], s[1]);
        (s[3], s[4]) = rot(s[3], s[4]);
    };
    let mut with = cold.clone();
    let mut without = cold.clone();
    let mut collisions = 0;
    let mut candidates = 0;
    let mut worst = 0.0_f64;
    for step in 1..=steps {
        let t = step as f64 * h;
        for s in with.iter_mut().chain(without.iter_mut()) {
            rotate(s, h);
        }
        let tally = model
            .collide_step(&d, &field, t, step, h, &ids, None, &mut with)
            .unwrap();
        collisions += tally.collisions;
        candidates += tally.stats.candidates;
        if step % 50 == 0 || step == steps {
            for (a, b) in with.iter().zip(&without) {
                let u = |s: &State| {
                    peculiar_velocity(
                        &field,
                        Vec3::new(s[0], s[1], s[2]),
                        Vec3::new(s[3], s[4], s[5]),
                    )
                };
                worst = worst.max(u(a).norm()).max((u(a) - u(b)).norm());
            }
        }
    }
    println!(
        "K1-T8b: {steps} steps ({orbits} orbits), {n} particles: candidates {candidates}, \
         collisions {collisions}, worst |u| or |u_with - u_without| = {worst:e} km/s"
    );
    assert!(worst <= 1e-12, "worst = {worst:e}");
}

/// The mean collision rate of a warm ring through the production path, from the
/// pair counts of the cells (SPEC 5.3 and appendix A), as a cross-check of the
/// frame code: events per step = Σ_cells ½ N_c (N_c − 1) W σ ⟨g⟩ Δt / V_c.
#[test]
fn quick_rate_in_the_ring_geometry() {
    let field = saturn();
    let d = domain(&field);
    let n = 4000;
    let c = 1e-5;
    let state = ring_state(&field, n, 0.4, 40.0, c);
    let ids: Vec<u32> = (0..n as u32).collect();
    let (w, s, dt) = (3.0e5, 1e-5, 100.0);
    let model = StochasticModel::new(s, w, Restitution::Constant { epsilon: 1.0 }, 8);
    let cells = d.build_cells(0.0, &ids, None, |i| {
        Vec3::new(state[i][0], state[i][1], state[i][2])
    });
    let pairs: f64 = (0..cells.len())
        .map(|k| {
            let m = cells.members(k).len() as f64;
            0.5 * m * (m - 1.0)
        })
        .sum();
    let sigma = PI * (2.0 * s).powi(2);
    let g_mean = 4.0 * c / PI.sqrt();
    let expected_per_step = pairs * w * sigma * g_mean * dt / d.grid.volume_km3();
    let mut total = 0;
    let trials = 200;
    for step in 0..trials {
        let mut st = state.clone();
        total += model
            .collide_step(&d, &field, 0.0, step, dt, &ids, None, &mut st)
            .unwrap()
            .collisions;
    }
    let measured = total as f64 / trials as f64;
    let se = (total as f64).sqrt() / trials as f64;
    println!("ring rate: measured {measured:.3} +- {se:.3}, expected {expected_per_step:.3}");
    // The pair counts use the actual cells; the velocity cloud is Maxwellian
    // only to the sampling of 4000 particles, so allow 5 SE and 5 %.
    assert!((measured - expected_per_step).abs() <= 5.0 * se + 0.05 * expected_per_step);
}
