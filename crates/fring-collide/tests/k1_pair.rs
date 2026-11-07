//! K1-T1, K1-T2, K1-T3 and K1-T8 (a, c): the pair operator, the impact
//! geometry, the restitution laws, peculiar velocities and cells
//! (`SPEC.md` section 11).

use std::f64::consts::PI;

use fring_collide::frame::{CellGrid, CellKey, peculiar_velocity};
use fring_collide::pair::{collide, epsilon, impact_normal};
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::circular::circular_speed;
use fring_model::physics::gravity::SaturnField;
use fring_model::rng::{Counter, Philox, Stream, box_muller, u01_f64};

const EPS: f64 = f64::EPSILON;

fn rng_block(rng: &Philox, index: u32, draw: u16) -> [u32; 4] {
    rng.block(Counter {
        index,
        step: 0,
        stream: Stream::TEST,
        draw,
    })
}

fn normal3(rng: &Philox, index: u32, draw: u16) -> Vec3<f64> {
    let a = rng_block(rng, index, draw);
    let b = rng_block(rng, index, draw + 1);
    let (x, y) = box_muller(u01_f64(a[0], a[1]), u01_f64(a[2], a[3]));
    let (z, _) = box_muller(u01_f64(b[0], b[1]), u01_f64(b[2], b[3]));
    Vec3::new(x, y, z)
}

fn unit(v: Vec3<f64>) -> Vec3<f64> {
    v / v.norm()
}

/// Checks one collision against SPEC section 4. Returns true if the pair collided.
fn check_pair(ui: Vec3<f64>, uj: Vec3<f64>, n: Vec3<f64>, law: &Restitution) -> bool {
    let g = ui - uj;
    let gn = g.dot(n);
    let umax = ui.norm().max(uj.norm());
    let round = 8.0 * EPS * umax;
    let Some(imp) = collide(g, n, law) else {
        assert!(gn >= 0.0, "an approaching pair did not collide");
        return false;
    };
    assert!(gn < 0.0);
    let (mut ui2, mut uj2) = (ui, uj);
    imp.apply(&mut ui2, &mut uj2);
    // Momentum, each component.
    let dp = (ui2 + uj2) - (ui + uj);
    for c in dp.to_array() {
        assert!(
            c.abs() <= 4.0 * EPS * umax,
            "momentum change {c:e} for |u| {umax:e}"
        );
    }
    // Normal reversal and tangential part, from the velocities.
    let eps = epsilon(law, gn.abs());
    assert_eq!(imp.epsilon, eps);
    let g2 = ui2 - uj2;
    let tol = 1e-12 * g.norm() + round;
    assert!(
        (g2.dot(n) + eps * gn).abs() <= tol,
        "normal: {:e} vs {:e}",
        g2.dot(n),
        -eps * gn
    );
    let tang = |v: Vec3<f64>| v - n * v.dot(n);
    assert!(
        (tang(g2) - tang(g)).norm() <= tol,
        "tangential part changed"
    );
    // Energy: the tally against ¼ (|g|² − |g'|²) with g' = g + 2Δ (no rounding of u).
    let g_exact = g + imp.delta_ui_km_s * 2.0;
    let de = 0.25 * (g.norm_squared() - g_exact.norm_squared());
    assert!(
        (imp.energy_loss_km2_s2 - de).abs() <= 1e-12 * 0.25 * g.norm_squared(),
        "energy {:e} vs {:e}",
        imp.energy_loss_km2_s2,
        de
    );
    assert!(imp.energy_loss_km2_s2 >= 0.0);
    true
}

#[test]
fn k1_t1_pair_operator_random_pairs() {
    let rng = Philox::from_seed(1);
    let laws = [
        Restitution::Constant { epsilon: 0.0 },
        Restitution::Constant { epsilon: 0.5 },
        Restitution::Constant { epsilon: 1.0 },
        Restitution::PowerLaw {
            reference_speed_km_s: 7.7e-8,
            exponent: 0.234,
        },
    ];
    let mut collided = 0;
    let n_pairs: u32 = 100_000;
    for k in 0..n_pairs {
        let law = &laws[(k % 4) as usize];
        // Half the pairs: peculiar speeds of 1 cm/s. Half: orbital speed 16.5 km/s
        // with differences of 1 mm/s, as in the GPU state.
        let base = if k % 2 == 0 {
            Vec3::zero()
        } else {
            unit(normal3(&rng, k, 0)) * 16.5
        };
        let scale = if k % 2 == 0 { 1e-5 } else { 1e-6 };
        let ui = base + normal3(&rng, k, 2) * scale;
        let uj = base + normal3(&rng, k, 4) * scale;
        let n = unit(normal3(&rng, k, 6));
        if check_pair(ui, uj, n, law) {
            collided += 1;
        }
    }
    // About half the pairs approach.
    assert!(collided > 45_000 && collided < 55_000, "{collided}");
    println!("K1-T1: {n_pairs} pairs, {collided} collided, all checks passed");
}

#[test]
fn k1_t1_pair_operator_edge_cases() {
    let ui = Vec3::new(1e-5, 2e-6, -3e-6);
    let uj = Vec3::new(-2e-6, 1e-6, 4e-6);
    let g = ui - uj;
    let n_head_on = unit(-g);
    // ε = 0: the normal relative velocity becomes 0.
    let zero = Restitution::Constant { epsilon: 0.0 };
    let imp = collide(g, n_head_on, &zero).unwrap();
    assert!((g + imp.delta_ui_km_s * 2.0).dot(n_head_on).abs() <= 1e-20);
    assert!(check_pair(ui, uj, n_head_on, &zero));
    // ε = 1: no energy loss, exactly.
    let one = Restitution::Constant { epsilon: 1.0 };
    assert_eq!(collide(g, n_head_on, &one).unwrap().energy_loss_km2_s2, 0.0);
    // gₙ = 0 and gₙ > 0: no collision.
    let perp = unit(g.cross(Vec3::new(0.0, 0.0, 1.0)));
    assert!(collide(g, perp, &zero).is_none() || g.dot(perp) < 0.0);
    assert!(collide(g, Vec3::new(0.0, 0.0, 0.0), &zero).is_none());
    assert!(collide(g, -n_head_on, &zero).is_none());
    // Not a number: no collision.
    assert!(collide(Vec3::new(f64::NAN, 0.0, 0.0), n_head_on, &zero).is_none());
    // f32 gives the same result to f32 rounding.
    let imp32 = collide(g.cast::<f32>(), n_head_on.cast::<f32>(), &zero).unwrap();
    let d64 = imp.delta_ui_km_s;
    let d32 = imp32.delta_ui_km_s.cast::<f64>();
    assert!((d32 - d64).norm() <= 1e-6 * d64.norm());
}

/// Kolmogorov-Smirnov statistic of samples against the uniform distribution on (0, 1).
fn ks_uniform(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len() as f64;
    x.iter()
        .enumerate()
        .map(|(i, &v)| ((i as f64 + 1.0) / n - v).max(v - i as f64 / n))
        .fold(0.0, f64::max)
}

#[test]
fn k1_t2_impact_geometry() {
    let rng = Philox::from_seed(2);
    let g = Vec3::new(1e-5, -2e-5, 3e-5);
    let g_hat = unit(g);
    let (e1, e2) = fring_collide::pair::basis(g_hat);
    let n_draws: u32 = 1_000_000;
    let mut mu2 = Vec::with_capacity(n_draws as usize);
    let mut psi = Vec::with_capacity(n_draws as usize);
    let mut max_norm_err: f64 = 0.0;
    for k in 0..n_draws {
        let w = rng_block(&rng, k, 0);
        let n = impact_normal(g, u01_f64(w[0], w[1]), u01_f64(w[2], w[3]));
        max_norm_err = max_norm_err.max((n.norm() - 1.0).abs());
        let cos_theta = -n.dot(g_hat);
        assert!(cos_theta > 0.0, "the pair must approach");
        mu2.push(cos_theta * cos_theta);
        psi.push((n.dot(e2).atan2(n.dot(e1)) / (2.0 * PI)).rem_euclid(1.0));
    }
    let critical = 1.95 / f64::from(n_draws).sqrt();
    let (d_mu, d_psi) = (ks_uniform(mu2), ks_uniform(psi));
    println!(
        "K1-T2: N = {n_draws}, KS cos²θ {d_mu:.2e}, KS ψ {d_psi:.2e}, critical {critical:.2e}, max ||n|-1| {max_norm_err:.1e}"
    );
    assert!(d_mu < critical && d_psi < critical);
    assert!(max_norm_err <= 1e-14);
}

#[test]
fn k1_t3_restitution_laws() {
    assert_eq!(epsilon(&Restitution::Constant { epsilon: 0.3 }, 1.0), 0.3);
    let vc = 7.7e-8; // Bridges et al. (1984): 0.0077 cm/s
    let law = Restitution::PowerLaw {
        reference_speed_km_s: vc,
        exponent: 0.234,
    };
    assert_eq!(epsilon(&law, 0.0), 1.0);
    assert_eq!(epsilon(&law, 0.5 * vc), 1.0);
    assert_eq!(epsilon(&law, vc), 1.0);
    assert!((epsilon(&law, vc * (1.0 + 1e-12)) - 1.0).abs() < 1e-12);
    assert!((epsilon(&law, 2.0 * vc) - 2.0_f64.powf(-0.234)).abs() < 1e-15);
    // SPEC section 6: 0.55 at 1 mm/s, 0.32 at 1 cm/s, 0.11 at 1 m/s.
    for (v, e) in [(1e-6, 0.5488), (1e-5, 0.3202), (1e-3, 0.1090)] {
        assert!(
            (epsilon(&law, v) - e).abs() < 1e-4,
            "{v}: {}",
            epsilon(&law, v)
        );
    }
}

fn saturn() -> SaturnField<f64> {
    let p = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    SaturnField::from_params(&p)
}

#[test]
fn k1_t8a_peculiar_velocity_of_a_circular_orbit_is_zero() {
    let field = saturn();
    let a = 140_222.43;
    let v = circular_speed(&field, a);
    for k in 0..12 {
        let phi = 0.5 + k as f64 * PI / 6.0;
        let pos = Vec3::new(a * phi.cos(), a * phi.sin(), 0.0);
        let vel = Vec3::new(-phi.sin(), phi.cos(), 0.0) * v;
        let u = peculiar_velocity(&field, pos, vel);
        assert!(u.norm() <= 1e-13, "{phi}: |u| = {:e}", u.norm());
        // An added velocity comes back unchanged.
        let du = Vec3::new(3e-6, -2e-6, 1e-6);
        let u2 = peculiar_velocity(&field, pos, vel + du);
        assert!((u2 - du).norm() <= 1e-13);
    }
}

#[test]
fn k1_t8c_cell_keys() {
    let a0 = 140_222.43;
    let d = 2.0 * PI * a0;
    let grid = CellGrid::new(0.05, 10.0, 0.04, d).unwrap();
    assert_eq!(grid.azimuthal_cells, (d / 10.0).round() as u32);
    assert!((f64::from(grid.azimuthal_cells) * grid.azimuthal_km - d).abs() <= 1e-9);
    let last = grid.azimuthal_cells - 1;
    let key = |x, y, z| grid.key(x, y, z);
    assert_eq!(
        key(0.0, 0.0, 0.0),
        CellKey {
            radial: 0,
            azimuthal: 0,
            vertical: 0
        }
    );
    assert_eq!(
        key(-0.01, 5.0, -1e-12),
        CellKey {
            radial: -1,
            azimuthal: 0,
            vertical: -1
        }
    );
    assert_eq!(key(0.12, -1e-9, 0.05).azimuthal, last);
    assert_eq!(key(0.0, -1e-30, 0.0).azimuthal, last);
    assert_eq!(key(0.0, d, 0.0).azimuthal, 0);
    assert_eq!(key(0.0, d - 1e-9, 0.0).azimuthal, last);
    assert_eq!(key(0.0, 2.0 * d + 15.0, 0.0), key(0.0, 15.0, 0.0));
    assert_eq!(key(0.12, 0.0, 0.0).radial, 2);
    assert!(CellGrid::new(0.0, 10.0, 0.04, d).is_none());
    assert!(CellGrid::new(0.05, f64::NAN, 0.04, d).is_none());
    // A requested L_y longer than the domain gives one cell.
    assert_eq!(
        CellGrid::new(0.05, 2.0 * d, 0.04, d)
            .unwrap()
            .azimuthal_cells,
        1
    );
}
