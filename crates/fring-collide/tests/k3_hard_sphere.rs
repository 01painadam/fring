//! K3: the hard-sphere model on the CPU (`SPEC.md` sections 8 and 11).
//!
//! The quick tests run in every `cargo test`. The full tests (sample-size rule
//! of SPEC 11) are long; run them with
//! `cargo test -p fring-collide --release -- --ignored --nocapture`.

use std::f64::consts::TAU;

use fring_collide::hard_sphere::HardSphereModel;
use fring_collide::stochastic::State;
use fring_collide::testkit::{
    ModelKind, Sample, t4_rate, t5_energy_loss, t6_budget, t7_haff, t9_determinism,
};
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::circular::circular_speed;
use fring_model::physics::frames::{reference_mean_motion, rotating_to_inertial};
use fring_model::physics::gravity::SaturnField;
use fring_model::rng::{Counter, Philox, Stream, box_muller};

const H: ModelKind = ModelKind::HardSphere;
const A0_KM: f64 = 140_222.43;

fn model(p: fring_collide::testkit::ModelParams) -> HardSphereModel {
    HardSphereModel::from_params(p)
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

// T4, T6 and T7 take about a minute each without optimisation: they run in
// `cargo test --release` (and are skipped in a debug `cargo test`).
#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug: run with --release")]
fn quick_t4_rate() {
    t4_rate(H, 0.0, Sample::Quick, model).assert_ok();
}

#[test]
fn quick_t5_energy_loss() {
    let law = Restitution::Constant { epsilon: 0.5 };
    t5_energy_loss(H, law, Sample::Quick, model).assert_ok();
    t5_energy_loss(H, power_law(), Sample::Quick, model).assert_ok();
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug: run with --release")]
fn quick_t6_budget() {
    t6_budget(H, 0.5, model).assert_ok();
    t6_budget(H, 1.0, model).assert_ok();
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug: run with --release")]
fn quick_t7_haff() {
    t7_haff(H, Sample::Quick, model).assert_ok();
}

#[test]
fn k1_t9_determinism() {
    t9_determinism(H, model);
}

// ---- full (long) ----

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t4_rate() {
    t4_rate(H, 0.0, Sample::Full, model).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t5_energy_loss() {
    for e in [0.0, 0.5, 0.9, 1.0] {
        let law = Restitution::Constant { epsilon: e };
        t5_energy_loss(H, law, Sample::Full, model).assert_ok();
    }
    t5_energy_loss(H, power_law(), Sample::Full, model).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t6_budget() {
    t6_budget(H, 0.5, model).assert_ok();
    t6_budget(H, 1.0, model).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t7_haff() {
    t7_haff(H, Sample::Full, model).assert_ok();
}

// ---- pair finding in the ring ----

fn uniform(seed: u64, i: usize, draw: u16) -> [f64; 2] {
    Philox::from_seed(seed).uniform_f64(Counter {
        index: i as u32,
        step: 0,
        stream: Stream::TEST,
        draw,
    })
}

fn gauss(seed: u64, i: usize) -> Vec3<f64> {
    let a = uniform(seed, i, 100);
    let b = uniform(seed, i, 101);
    let (x, y) = box_muller(a[0], a[1]);
    let (z, _) = box_muller(b[0], b[1]);
    Vec3::new(x, y, z)
}

/// `n` inertial positions at time `t_s` in a clump of edge `w_km` around the
/// rotating-frame point (radius `rho_km`, longitude `phi`).
fn clump(
    n: usize,
    seed: u64,
    rho_km: f64,
    phi: f64,
    w_km: f64,
    n0: f64,
    t_s: f64,
) -> Vec<Vec3<f64>> {
    (0..n)
        .map(|i| {
            let u = uniform(seed, i, 0);
            let z = uniform(seed, i, 1)[0];
            let rho = rho_km + (u[0] - 0.5) * w_km;
            let ph = phi + (u[1] - 0.5) * w_km / rho_km;
            let p = Vec3::new(rho * ph.cos(), rho * ph.sin(), (z - 0.5) * w_km);
            rotating_to_inertial(p, Vec3::new(0.0, 0.0, 0.0), n0, t_s).0
        })
        .collect()
}

fn brute_force(pos: &[Vec3<f64>], ids: &[u32], active: &[bool], contact: f64) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for i in 0..pos.len() {
        for j in 0..pos.len() {
            if i != j
                && active[i]
                && active[j]
                && ids[i] < ids[j]
                && (pos[i] - pos[j]).norm_squared() < contact * contact
            {
                out.push((i as u32, j as u32));
            }
        }
    }
    out.sort_unstable_by_key(|&(i, j)| (ids[i as usize], ids[j as usize]));
    out
}

fn check_against_brute_force(s_km: f64, pos: &[Vec3<f64>], t_s: f64, n0: f64, what: &str) -> usize {
    let m = HardSphereModel::new(s_km, Restitution::Constant { epsilon: 0.5 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    // Ids in another order than the indices, and some inactive particles.
    let ids: Vec<u32> = (0..pos.len() as u32).map(|i| 100_000 - 3 * i).collect();
    let active: Vec<bool> = (0..pos.len()).map(|i| i % 7 != 3).collect();
    let found: Vec<(u32, u32)> = m
        .find_pairs(&d, t_s, &ids, Some(&active), |i| pos[i])
        .iter()
        .map(|p| {
            assert_eq!(p.shift, 0);
            (p.i, p.j)
        })
        .collect();
    let want = brute_force(pos, &ids, &active, 2.0 * s_km);
    assert_eq!(found, want, "{what}, t = {t_s}");
    want.len()
}

#[test]
fn pairs_match_brute_force_across_the_wrap_and_at_low_radius() {
    let n0 = reference_mean_motion(&saturn(), A0_KM);
    let s = 0.005; // contact 10 m
    for t in [0.0, 1.0e5, 3.0e8] {
        // A clump on the wrap (rotating longitude 0), 0.25 km wide.
        let pos = clump(1500, 1, A0_KM, 0.0, 0.25, n0, t);
        let found = check_against_brute_force(s, &pos, t, n0, "wrap clump");
        assert!(found > 50, "the test needs pairs: {found}");
        // A clump far inside a0 (curvature: arc length is 1/0.6 times the distance).
        let pos = clump(1500, 2, 0.6 * A0_KM, 1.0, 0.25, n0, t);
        let found = check_against_brute_force(s, &pos, t, n0, "low radius");
        assert!(found > 100, "the test needs pairs: {found}");
        // Both clumps at once: the cell grid must serve the smaller radius.
        let mut both = clump(800, 3, A0_KM, 0.0, 0.25, n0, t);
        both.extend(clump(800, 4, 0.6 * A0_KM, 1.0, 0.25, n0, t));
        check_against_brute_force(s, &both, t, n0, "both");
    }
}

#[test]
fn pairs_with_fewer_than_three_cells_around_the_ring() {
    // Contact 3 x 10^5 km: only 2 cells along the ring, and few radially.
    let n0 = reference_mean_motion(&saturn(), A0_KM);
    let s = 1.5e5;
    let pos: Vec<Vec3<f64>> = (0..80)
        .map(|i| {
            let u = uniform(9, i, 0);
            let z = uniform(9, i, 1)[0];
            let (rho, ph) = (A0_KM * (0.5 + u[0]), TAU * u[1]);
            Vec3::new(rho * ph.cos(), rho * ph.sin(), (z - 0.5) * 2.0e5)
        })
        .collect();
    let found = check_against_brute_force(s, &pos, 0.0, n0, "two cells");
    assert!(found > 100, "the test needs pairs: {found}");
}

// ---- collisions ----

fn state_of(pos: Vec3<f64>, vel: Vec3<f64>) -> State {
    [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z]
}

/// Two particles that touch across the wrap of the full ring (rotating
/// longitude +/- a few metres around 0), approaching each other on the line
/// between them. Returns the state, the unit vector from j to i and the base
/// velocity.
fn wrap_pair(
    field: &SaturnField<f64>,
    n0: f64,
    t_s: f64,
    separation_km: f64,
    approach_km_s: f64,
) -> ([State; 2], Vec3<f64>, Vec3<f64>) {
    let zero = Vec3::new(0.0, 0.0, 0.0);
    let h = 0.5 * separation_km / A0_KM;
    // i at longitude +h, j at -h (rotating frame); d = r_i - r_j points along +y there.
    let (pi, vi) = rotating_to_inertial(
        Vec3::new(A0_KM * h.cos(), A0_KM * h.sin(), 0.0),
        zero,
        n0,
        t_s,
    );
    let (pj, _) = rotating_to_inertial(
        Vec3::new(A0_KM * h.cos(), -A0_KM * h.sin(), 0.0),
        zero,
        n0,
        t_s,
    );
    let _ = vi;
    let d = pi - pj;
    let d_hat = d / d.norm();
    // The same circular velocity for both (rigid rotation about Saturn at
    // the rate of i), plus the approach along d.
    let theta = n0 * t_s;
    let phi_hat = Vec3::new(-(theta + h).sin(), (theta + h).cos(), 0.0);
    let base = phi_hat * circular_speed(field, A0_KM);
    let vi = base - d_hat * (0.5 * approach_km_s);
    let vj = base + d_hat * (0.5 * approach_km_s);
    ([state_of(pi, vi), state_of(pj, vj)], d_hat, base)
}

#[test]
fn a_pair_across_the_wrap_collides() {
    let field = saturn();
    let n0 = reference_mean_motion(&field, A0_KM);
    let m = HardSphereModel::new(0.01, Restitution::Constant { epsilon: 1.0 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let ids = [10_u32, 11];
    for t in [0.0, 7.7e4, 2.5e8] {
        // Separation 15 m: overlap 5 m = 0.25 of the contact distance: deep.
        // Separation 19.5 m: overlap 0.5 m = 0.025: not deep.
        for (sep, deep) in [(0.015, 1), (0.0195, 0)] {
            let (mut st, d_hat, base) = wrap_pair(&field, n0, t, sep, 1.0e-3);
            let before = st;
            let pairs = m.find_pairs(&d, t, &ids, None, |i| {
                Vec3::new(st[i][0], st[i][1], st[i][2])
            });
            let pairs: Vec<(u32, u32)> = pairs.iter().map(|p| (p.i, p.j)).collect();
            assert_eq!(pairs, vec![(0, 1)], "t = {t}, separation {sep}");
            let tally = m.collide_step(&d, t, 356.0, &ids, None, &mut st);
            assert_eq!(tally.collisions, 1);
            assert_eq!(tally.stats.deep_collisions, deep, "separation {sep}");
            // Elastic head-on: the velocities along d swap.
            let vi = Vec3::new(st[0][3], st[0][4], st[0][5]);
            let vj = Vec3::new(st[1][3], st[1][4], st[1][5]);
            let want_i = base + d_hat * 0.5e-3;
            let err = (vi - want_i)
                .norm()
                .max((vj - (base - d_hat * 0.5e-3)).norm());
            // 16.5 km/s base: the round-off floor is a few 10^-15 km/s.
            assert!(err < 1e-13, "t = {t}: |error| = {err:e} km/s");
            assert_eq!(tally.stats.energy_loss_km2_s2, 0.0);
            // Positions do not change.
            assert_eq!(st[0][..3], before[0][..3]);
            assert_eq!(st[1][..3], before[1][..3]);
        }
    }
    // A receding pair across the wrap is found but not collided.
    let (mut st, _, _) = wrap_pair(&field, n0, 0.0, 0.015, -1.0e-3);
    let tally = m.collide_step(&d, 0.0, 356.0, &ids, None, &mut st);
    assert_eq!(tally.collisions, 0);
}

/// A dense clump with Maxwellian velocities: many pairs, and repeated
/// collisions of one particle in one step.
fn warm_clump(field: &SaturnField<f64>, n: usize, s_km: f64) -> (Vec<State>, f64, f64) {
    let n0 = reference_mean_motion(field, A0_KM);
    let t = 1.0e5;
    let pos = clump(n, 21, A0_KM, 0.0, 0.2, n0, t);
    let state = pos
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let rho = p.x.hypot(p.y);
            let phi_hat = Vec3::new(-p.y / rho, p.x / rho, 0.0);
            let v = phi_hat * circular_speed(field, rho) + gauss(22, i) * 1e-5;
            state_of(p, v)
        })
        .collect();
    let _ = s_km;
    (state, n0, t)
}

#[test]
fn collisions_in_the_ring_conserve_momentum_and_account_for_energy() {
    let field = saturn();
    let s = 0.002; // contact 4 m
    let (state, n0, t) = warm_clump(&field, 3000, s);
    let m = HardSphereModel::new(s, Restitution::Constant { epsilon: 0.5 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let ids: Vec<u32> = (0..state.len() as u32).collect();
    let mut st = state.clone();
    // A step of 356 s with c = 1 cm/s moves a particle 3.6 m: a pair that
    // overlaps is not at its contact, so deep collisions occur.
    let tally = m.collide_step(&d, t, 356.0, &ids, None, &mut st);
    assert!(tally.collisions > 20, "collisions: {}", tally.collisions);
    // Sums of the changes (the change of a sum of 16.5 km/s values has its own rounding).
    let mut dp = Vec3::new(0.0, 0.0, 0.0);
    let mut de = 0.0;
    for (a, b) in st.iter().zip(&state) {
        let dv = Vec3::new(a[3] - b[3], a[4] - b[4], a[5] - b[5]);
        let vs = Vec3::new(a[3] + b[3], a[4] + b[4], a[5] + b[5]);
        dp += dv;
        de += 0.5 * dv.dot(vs);
    }
    let bound = 2.0 * tally.collisions as f64 * 16.5 * f64::EPSILON;
    println!(
        "K3 ring: collisions {}, deep {}, |dp| = {:e} (bound {:e}), energy {:e} vs tally {:e}",
        tally.collisions,
        tally.stats.deep_collisions,
        dp.norm(),
        bound,
        -de,
        tally.stats.energy_loss_km2_s2
    );
    assert!(dp.x.abs() <= bound && dp.y.abs() <= bound && dp.z.abs() <= bound);
    // The rounding of 16.5 km/s velocities (3.6e-15 km/s each) in ½ dv · (v + v')
    // is about 3.6e-15 x 16.5 per collision, against about 1e-10 km² s⁻² of loss.
    let tol = 1e-9 * tally.stats.energy_loss_km2_s2 + 1e-13 * tally.collisions as f64 * 1e-3;
    assert!((-de - tally.stats.energy_loss_km2_s2).abs() <= tol);
    assert!(tally.stats.deep_collisions > 0);
}

#[test]
fn the_result_does_not_depend_on_the_array_order() {
    let field = saturn();
    let s = 0.002;
    let (state, n0, t) = warm_clump(&field, 3000, s);
    let m = HardSphereModel::new(s, Restitution::Constant { epsilon: 0.5 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let ids: Vec<u32> = (0..state.len() as u32).map(|i| 17 + 5 * i).collect();
    let mut a = state.clone();
    let ta = m.collide_step(&d, t, 356.0, &ids, None, &mut a);
    assert!(ta.collisions > 20);
    // Same call again: no randomness, so the same bits.
    let mut a2 = state.clone();
    let ta2 = m.collide_step(&d, t, 356.0, &ids, None, &mut a2);
    assert_eq!(a, a2);
    assert_eq!(ta, ta2);
    // Reversed arrays with the same ids: every particle gets the same state.
    let perm: Vec<usize> = (0..state.len()).rev().collect();
    let mut b: Vec<State> = perm.iter().map(|&i| state[i]).collect();
    let ids_b: Vec<u32> = perm.iter().map(|&i| ids[i]).collect();
    let tb = m.collide_step(&d, t, 356.0, &ids_b, None, &mut b);
    assert_eq!(ta.collisions, tb.collisions);
    for (k, &i) in perm.iter().enumerate() {
        assert_eq!(b[k], a[i], "particle {i}");
    }
}

#[test]
fn inactive_particles_take_no_part() {
    let field = saturn();
    let s = 0.002;
    let (state, n0, t) = warm_clump(&field, 3000, s);
    let m = HardSphereModel::new(s, Restitution::Constant { epsilon: 0.5 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let ids: Vec<u32> = (0..state.len() as u32).collect();
    let active: Vec<bool> = (0..state.len()).map(|i| i % 3 != 0).collect();
    let mut st = state.clone();
    let tally = m.collide_step(&d, t, 356.0, &ids, Some(&active), &mut st);
    assert!(tally.collisions > 10);
    for i in (0..state.len()).filter(|&i| !active[i]) {
        assert_eq!(st[i], state[i]);
    }
    // No active particle: nothing to do.
    let none = vec![false; state.len()];
    let mut st = state.clone();
    assert_eq!(
        m.collide_step(&d, t, 356.0, &ids, Some(&none), &mut st)
            .collisions,
        0
    );
    assert_eq!(st, state);
}

/// K1-T8 (b) for hard spheres. Hard spheres see real velocity differences, also
/// the shear (neighbours at other radii collide: that is physics, not a defect),
/// so the "cold ring" is a chain at one radius. All its particles turn at one
/// rate, so the chain is a rigid body: d . g = 0 for every pair, and nothing
/// may change. The spacing is 1.01 contact distances: close enough that the
/// cell search must look at every neighbour, far enough that the rounding of
/// the positions (3e-11 km at 1.4e5 km, 1.5e-9 of the contact distance) cannot
/// make an overlap. (A chain that starts overlapped is not a rigid body in the
/// sense of SPEC 8: the straight path back to the contact is then wrong, and a
/// grazing pair with tau <= h gets a tilted normal. An initial overlap is
/// outside the model.)
///
/// The orbit step is a test-only exact rotation, as in K2-T8b. The limit is
/// 10^-12 km/s (K1-T8 b); the expected change is exactly 0, so the margin is
/// not a ratio.
#[test]
fn k1_t8b_rigid_chain_stays_cold() {
    let field = saturn();
    let n0 = reference_mean_motion(&field, A0_KM);
    let s = 0.01; // contact 20 m
    let m = HardSphereModel::new(s, Restitution::Constant { epsilon: 0.5 });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let spacing = 1.01 * 2.0 * s;
    let n = 300;
    let ids: Vec<u32> = (0..n as u32).collect();
    let vc = circular_speed(&field, A0_KM);
    // The chain straddles the wrap of the rotating frame (longitude 0).
    let mut st: Vec<State> = (0..n)
        .map(|i| {
            let phi = (i as f64 - 0.5 * n as f64) * spacing / A0_KM;
            let p = Vec3::new(A0_KM * phi.cos(), A0_KM * phi.sin(), 0.0);
            state_of(p, Vec3::new(-phi.sin(), phi.cos(), 0.0) * vc)
        })
        .collect();
    let h = 356.0;
    let steps = (10.0 * TAU / (n0 * h)).round() as u64;
    let a = vc / A0_KM * h;
    let (c, sn) = (a.cos(), a.sin());
    let rot = |x: f64, y: f64| (c * x - sn * y, sn * x + c * y);
    let mut collisions = 0;
    let mut worst = 0.0_f64;
    for step in 1..=steps {
        for s in st.iter_mut() {
            (s[0], s[1]) = rot(s[0], s[1]);
            (s[3], s[4]) = rot(s[3], s[4]);
        }
        let before = st.clone();
        collisions += m
            .collide_step(&d, step as f64 * h, h, &ids, None, &mut st)
            .collisions;
        for (a, b) in st.iter().zip(&before) {
            let dv = Vec3::new(a[3] - b[3], a[4] - b[4], a[5] - b[5]);
            worst = worst.max(dv.norm());
        }
    }
    println!(
        "K1-T8b (hard spheres): {steps} steps, {n} particles, collisions {collisions}, \
         largest velocity change {worst:e} km/s"
    );
    assert_eq!(collisions, 0);
    assert!(worst <= 1e-12, "worst = {worst:e}");
}

/// The contact normal of SPEC 8 in a ring state. j is at rest in a local patch
/// (x radial, y along the ring); i is at d = (14 m, 12 m) from j and moves with
/// g = (-1 m/s, 0, 0). The contact is at d_x = sqrt(20² - 12²) = 16 m, so the
/// pair is 2 m deep and the path back to the contact takes tau = 2 s.
///
/// - With h = 356 s (tau <= h): n = (16, 12) / 20, the normal at contact.
/// - With h = 1 s (tau > h): the pair overlapped before this step, n = d / |d|.
#[test]
fn contact_normal_from_the_path_back() {
    let field = saturn();
    let n0 = reference_mean_motion(&field, A0_KM);
    let eps = 0.5;
    let m = HardSphereModel::new(0.01, Restitution::Constant { epsilon: eps });
    let d = m.full_ring_domain(A0_KM, n0).unwrap();
    let ids = [0_u32, 1];
    let pj = Vec3::new(A0_KM, 0.0, 0.0);
    let vj = Vec3::new(0.0, circular_speed(&field, A0_KM), 0.0);
    let g = Vec3::new(-1.0e-3, 0.0, 0.0);
    let (dx, dy) = (0.014, 0.012);
    // d from the stored positions (3e-11 km resolution at 1.4e5 km), and tau
    // from the textbook form (b + sqrt(b² + |g|² (4s² - |d|²))) / |g|², which
    // is not the cancellation-free form of the model.
    let pi = pj + Vec3::new(dx, dy, 0.0);
    let dd = pi - pj;
    let b = dd.dot(g);
    let tau = (b + (b * b + g.norm_squared() * (0.02_f64.powi(2) - dd.norm_squared())).sqrt())
        / g.norm_squared();
    assert!((tau - 2.0).abs() < 0.2, "tau = {tau}");
    for (h, n_hat) in [(356.0, (dd - g * tau) / 0.02), (1.0, dd / dd.norm())] {
        let mut st = [state_of(pi, vj + g), state_of(pj, vj)];
        let t = m.collide_step(&d, 0.0, h, &ids, None, &mut st);
        assert_eq!(t.collisions, 1);
        // Change of v_i: -(1 + eps)/2 gn n, with gn = g . n < 0.
        let gn = g.dot(n_hat);
        let want = n_hat * (-0.5 * (1.0 + eps) * gn);
        let got = Vec3::new(st[0][3] - (vj.x + g.x), st[0][4] - (vj.y + g.y), 0.0);
        assert!(
            (got - want).norm() < 1e-13,
            "h = {h}: got {got:?}, want {want:?}"
        );
        let loss = 0.25 * (1.0 - eps * eps) * gn * gn;
        assert!((t.stats.energy_loss_km2_s2 - loss).abs() < 1e-18);
        // Overlap depth 20 - 18.4 = 1.6 m is below 0.2 s = 2 m: not deep.
        assert_eq!(t.stats.deep_collisions, 0);
    }
}
