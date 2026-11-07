//! ARC-CELLS (CPU part): the cell index of an arc and the hard-sphere pairs
//! across the arc seam (`SPEC.md` 5.1 and 8).

use std::f64::consts::{PI, TAU};

use fring_collide::hard_sphere::{HardSphereModel, Pair};
use fring_collide::stochastic::Domain;
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::arc::{ArcGeometry, EDGE_TOL};
use fring_model::physics::circular::circular_speed;
use fring_model::physics::frames::{reference_mean_motion, rotating_to_inertial};
use fring_model::physics::gravity::SaturnField;

const A0_KM: f64 = 140_222.43;
/// Requested cell edges for the cell index tests, km.
const EDGE_KM: f64 = 100.0;

fn saturn() -> SaturnField<f64> {
    let p = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    SaturnField::from_params(&p)
}

fn n0() -> f64 {
    reference_mean_motion(&saturn(), A0_KM)
}

fn arc(width_deg: f64, centre_deg: f64) -> ArcGeometry {
    ArcGeometry::new(width_deg.to_radians(), centre_deg.to_radians(), n0()).unwrap()
}

/// Inertial position at radius `rho_km`, inertial longitude `lon`, height `z_km`.
fn at(rho_km: f64, lon: f64, z_km: f64) -> Vec3<f64> {
    Vec3::new(rho_km * lon.cos(), rho_km * lon.sin(), z_km)
}

/// Position at angle `s` from the arc centre (at time `t_s`) on the core circle.
fn at_angle(arc: &ArcGeometry, t_s: f64, s: f64) -> Vec3<f64> {
    at(A0_KM, arc.centre_at_rad(t_s) + s, 0.0)
}

fn azimuthal_cells(d: &Domain) -> u32 {
    d.grid.azimuthal_cells
}

/// The cell number that `y = a0 (s + width/2)` gives, with `y` taken modulo D.
fn expected_cell(d: &Domain, arc: &ArcGeometry, s: f64) -> u32 {
    let y = (A0_KM * (s + 0.5 * arc.width_rad())).rem_euclid(d.grid.domain_length_km);
    ((y / d.grid.azimuthal_km).floor() as u32).min(azimuthal_cells(d) - 1)
}

#[test]
fn the_seven_degree_arc_across_180_degrees() {
    // 360 / 7 is not a whole number: the old a0 (phi - phi0) was wrong here.
    let a = arc(7.0, 180.0);
    let d = Domain::arc_domain(EDGE_KM, EDGE_KM, EDGE_KM, A0_KM, a).unwrap();
    for t in [0.0, 1.0e5, 3.0e8] {
        // At t = 0 the particle at 181 degrees is 1 degree from the centre;
        // y = a0 (1 + 3.5) degrees = a0 x 4.5 degrees.
        for deg in [-3.4, -1.0, -1e-3, 0.0, 1.0, 3.4] {
            let s = f64::to_radians(deg);
            let key = d.cell_key(at_angle(&a, t, s), t);
            assert_eq!(
                key.azimuthal,
                expected_cell(&d, &a, s),
                "t = {t}, s = {deg} degrees"
            );
        }
    }
    // The example of the task: the particle at 181 degrees at t = 0.
    let key = d.cell_key(at(A0_KM, PI + 1.0_f64.to_radians(), 0.0), 0.0);
    let y = A0_KM * 4.5_f64.to_radians();
    assert_eq!(key.azimuthal, (y / d.grid.azimuthal_km).floor() as u32);
    // Not the old value, a0 x 1.5 degrees.
    let old = (A0_KM * 1.5_f64.to_radians() / d.grid.azimuthal_km).floor() as u32;
    assert_ne!(key.azimuthal, old);
}

#[test]
fn the_cell_index_is_monotonic_through_180_degrees() {
    // Walk the 7 degree arc from the start edge to the end edge across the
    // branch cut of atan2: the cell number never decreases.
    let a = arc(7.0, 180.0);
    let d = Domain::arc_domain(EDGE_KM, EDGE_KM, EDGE_KM, A0_KM, a).unwrap();
    let h = a.width_rad() / 2.0;
    let mut last = 0;
    for n in 0..=7000 {
        let s = -h + a.width_rad() * (f64::from(n) + 0.5) / 7001.0;
        let c = d.cell_key(at_angle(&a, 0.0, s), 0.0).azimuthal;
        assert!(c >= last && c <= last + 1, "n = {n}: {last} -> {c}");
        last = c;
    }
    assert_eq!(last, azimuthal_cells(&d) - 1);
}

#[test]
fn edges_inside_at_slack_and_beyond_on_both_edges() {
    let slack = EDGE_TOL * (1.0 + 1.0); // a bit more than the worst EDGE_TOL (cos h + sin h)
    for (width, centre) in [(5.0, 0.0), (7.0, 180.0), (10.0, 90.0), (30.0, -120.0)] {
        let a = arc(width, centre);
        let d = Domain::arc_domain(EDGE_KM, EDGE_KM, EDGE_KM, A0_KM, a).unwrap();
        let n_y = azimuthal_cells(&d);
        let h = a.width_rad() / 2.0;
        for t in [0.0, 4.4e7] {
            let cell = |s: f64| d.cell_key(at_angle(&a, t, s), t).azimuthal;
            let ctx = format!("width {width}, centre {centre}, t = {t}");
            // Inside: the first and the last cell, 1e-3 rad from each edge.
            assert_eq!(cell(-h + 1e-5), expected_cell(&d, &a, -h + 1e-5), "{ctx}");
            assert_eq!(cell(h - 1e-5), expected_cell(&d, &a, h - 1e-5), "{ctx}");
            assert_eq!(
                cell(-h + 1e-5),
                (A0_KM * 1e-5 / d.grid.azimuthal_km) as u32,
                "{ctx}"
            );
            assert_eq!(cell(h - 1e-5), n_y - 1, "{ctx}");
            // Exactly at an edge: the cell inside the edge, or the periodic
            // image (the same place): the first or the last cell.
            for s in [-h, h] {
                let c = cell(s);
                assert!(c == 0 || c == n_y - 1, "{ctx}: edge cell {c} of {n_y}");
            }
            // Inside the slack, outside the edge: the periodic image, which is
            // the cell just inside the other edge (here: the first cell for the
            // end edge, the last cell for the start edge). Never out of range.
            for k in [0.1, 0.5, 1.0] {
                let out = k * slack;
                assert_eq!(cell(h + out), 0, "{ctx}: end edge, {out:e} rad out");
                assert_eq!(
                    cell(-h - out),
                    n_y - 1,
                    "{ctx}: start edge, {out:e} rad out"
                );
            }
            // Beyond: the cell of the image, `s -+ width`.
            for out in [1e-3, 0.3 * h, 0.9 * h] {
                assert_eq!(cell(h + out), cell(h + out - a.width_rad()), "{ctx}: end");
                assert_eq!(
                    cell(-h - out),
                    cell(-h - out + a.width_rad()),
                    "{ctx}: start"
                );
            }
        }
    }
}

#[test]
fn the_full_ring_is_unchanged() {
    // The full ring has no arc: y = a0 (phi - 0) with phi from the rotating frame.
    let d = Domain::full_ring(EDGE_KM, EDGE_KM, EDGE_KM, A0_KM, n0()).unwrap();
    assert!(d.arc.is_none());
    let cells = azimuthal_cells(&d);
    let t = 0.0;
    for k in 0..360 {
        let lon = f64::from(k).to_radians() - PI + 0.123;
        let key = d.cell_key(at(A0_KM, lon, 0.0), t);
        let phi = lon.rem_euclid(TAU);
        let want = ((A0_KM * phi / d.grid.azimuthal_km).floor() as u32).min(cells - 1);
        assert!(key.azimuthal.abs_diff(want) <= 1, "k = {k}");
    }
}

// ---- hard-sphere pairs across the seam ----

fn uniform(seed: u64, i: usize, k: u32) -> f64 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(k) << 32;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn rot(v: Vec3<f64>, a: f64) -> Vec3<f64> {
    let (c, s) = (a.cos(), a.sin());
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

#[test]
fn pairs_across_the_seam_match_brute_force_with_images() {
    let n0 = n0();
    let s = 0.1; // contact 200 m
    let m = HardSphereModel::new(s, Restitution::Constant { epsilon: 0.5 });
    for (width, centre) in [(7.0, 180.0), (10.0, 0.0)] {
        let a = ArcGeometry::new(f64::to_radians(width), f64::to_radians(centre), n0).unwrap();
        let d = m.arc_domain(A0_KM, a).unwrap();
        for t in [0.0, 1.0e5, 3.0e8] {
            // A clump of 2 km around the start edge, then wrapped into the arc
            // by the D1 wrap: about half of it lands at the end edge.
            let n = 2000;
            let lon0 = a.rotating_start_rad();
            let pos: Vec<Vec3<f64>> = (0..n)
                .map(|i| {
                    let rho = A0_KM + (uniform(1, i, 0) - 0.5) * 2.0;
                    let ph = lon0 + (uniform(1, i, 1) - 0.5) * 2.0 / A0_KM;
                    let p = Vec3::new(
                        rho * ph.cos(),
                        rho * ph.sin(),
                        (uniform(1, i, 2) - 0.5) * 2.0,
                    );
                    let q = rotating_to_inertial(p, Vec3::new(0.0, 0.0, 0.0), n0, t).0;
                    a.wrap(q, Vec3::new(0.0, 0.0, 0.0), t).unwrap().pos_km
                })
                .collect();
            let ids: Vec<u32> = (0..n as u32).map(|i| 50_000 - 3 * i).collect();
            let active: Vec<bool> = (0..n).map(|i| i % 7 != 3).collect();
            let found = m.find_pairs(&d, t, &ids, Some(&active), |i| pos[i]);
            // Brute force: the nearest of the three images of j.
            let c2 = (2.0 * s) * (2.0 * s);
            let mut want = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    if i == j || !active[i] || !active[j] || ids[i] >= ids[j] {
                        continue;
                    }
                    let best = [0_i8, 1, -1]
                        .into_iter()
                        .map(|k| {
                            let q = rot(pos[j], f64::from(k) * a.width_rad());
                            ((pos[i] - q).norm_squared(), k)
                        })
                        .min_by(|x, y| x.0.total_cmp(&y.0))
                        .unwrap();
                    if best.0 < c2 {
                        want.push(Pair {
                            i: i as u32,
                            j: j as u32,
                            shift: best.1,
                        });
                    }
                }
            }
            want.sort_unstable_by_key(|p| (ids[p.i as usize], ids[p.j as usize]));
            let missing: Vec<_> = want.iter().filter(|p| !found.contains(p)).collect();
            let extra: Vec<_> = found.iter().filter(|p| !want.contains(p)).collect();
            assert!(
                missing.is_empty() && extra.is_empty(),
                "width {width}, t = {t}: missing {missing:?}, extra {extra:?}"
            );
            let across = want.iter().filter(|p| p.shift != 0).count();
            assert!(
                across > 20,
                "the test needs pairs across the seam: {across}"
            );
            assert!(want.len() > across + 20, "and pairs inside");
        }
    }
}

/// State of a particle at radius `rho`, inertial longitude `lon`, height `z`,
/// with the circular speed plus the velocity `w` = (radial, tangential, z).
fn particle(rho: f64, lon: f64, z: f64, w: [f64; 3], vc: f64) -> [f64; 6] {
    let (c, s) = (lon.cos(), lon.sin());
    let vt = vc + w[1];
    [
        rho * c,
        rho * s,
        z,
        w[0] * c - vt * s,
        w[0] * s + vt * c,
        w[2],
    ]
}

/// Local components (radial, tangential, z) of a velocity change at longitude `lon`.
fn local(dv: Vec3<f64>, lon: f64) -> [f64; 3] {
    let (c, s) = (lon.cos(), lon.sin());
    [dv.x * c + dv.y * s, -dv.x * s + dv.y * c, dv.z]
}

#[test]
fn a_pair_across_the_seam_collides_like_the_same_pair_in_the_middle() {
    let field = saturn();
    let n0 = n0();
    let m = HardSphereModel::new(0.01, Restitution::Constant { epsilon: 0.5 });
    let vc = circular_speed(&field, A0_KM);
    let tan_a = 0.012 / A0_KM; // tangential offset of the pair, rad
    let w_a = [1.0e-3, 2.0e-3, -1.0e-3];
    let w_b = [-1.0e-3, -1.0e-3, 5.0e-4];
    let ids = [10_u32, 11];
    for (width, centre) in [(7.0, 180.0), (10.0, 0.0)] {
        let a = ArcGeometry::new(f64::to_radians(width), f64::to_radians(centre), n0).unwrap();
        let d = m.arc_domain(A0_KM, a).unwrap();
        let h = a.width_rad() / 2.0;
        for t in [0.0, 7.7e4, 2.5e8] {
            let lc = a.centre_at_rad(t);
            let make = |lon_a: f64, lon_b: f64| {
                [
                    particle(A0_KM, lon_a, 0.0, w_a, vc),
                    particle(A0_KM + 0.004, lon_b, 0.003, w_b, vc),
                ]
            };
            // Middle of the arc.
            let mid = make(lc, lc + tan_a);
            // On the seam: A 0.3 steps before the end edge, B (wrapped) on the
            // other side, 0.7 steps beyond the end edge = start edge + 0.7.
            let mut seam = make(lc + h - 0.3 * tan_a, lc - h + 0.7 * tan_a);
            let mut st = mid;
            let tm = m.collide_step(&d, t, 356.0, &ids, None, &mut st);
            assert_eq!(tm.collisions, 1);
            let dm: Vec<[f64; 3]> = (0..2)
                .map(|k| {
                    let dv = Vec3::new(
                        st[k][3] - mid[k][3],
                        st[k][4] - mid[k][4],
                        st[k][5] - mid[k][5],
                    );
                    local(dv, if k == 0 { lc } else { lc + tan_a })
                })
                .collect();
            let pairs = m.find_pairs(&d, t, &ids, None, |i| {
                Vec3::new(seam[i][0], seam[i][1], seam[i][2])
            });
            assert_eq!(
                pairs,
                vec![Pair {
                    i: 0,
                    j: 1,
                    shift: 1
                }],
                "width {width}, t = {t}"
            );
            let before = seam;
            let ts = m.collide_step(&d, t, 356.0, &ids, None, &mut seam);
            assert_eq!(ts.collisions, 1, "width {width}, t = {t}");
            assert_eq!(ts.stats.deep_collisions, tm.stats.deep_collisions);
            let lons = [lc + h - 0.3 * tan_a, lc - h + 0.7 * tan_a];
            let mut worst = 0.0_f64;
            for k in 0..2 {
                let dv = Vec3::new(
                    seam[k][3] - before[k][3],
                    seam[k][4] - before[k][4],
                    seam[k][5] - before[k][5],
                );
                let ls = local(dv, lons[k]);
                for c in 0..3 {
                    worst = worst
                        .max((ls[c] - dm[k][c]).abs() / dm[k][0].hypot(dm[k][1]).hypot(dm[k][2]));
                }
            }
            let e_rel = (ts.stats.energy_loss_km2_s2 - tm.stats.energy_loss_km2_s2).abs()
                / tm.stats.energy_loss_km2_s2;
            println!(
                "ARC seam pair width {width} t = {t}: worst relative difference of the velocity change {worst:e}, energy loss {e_rel:e}"
            );
            // The inertial coordinates are 1.4e5 km with a rounding of 3e-11 km,
            // and the pair is 13 mm apart: 3e-11 / 1.3e-2 = 2e-9 limits the
            // direction of d, the rotation of the state adds nothing above that.
            assert!(worst < 2.0e-8, "worst = {worst:e}");
            assert!(e_rel < 2.0e-8, "energy {e_rel:e}");
            // Positions do not change.
            assert_eq!(seam[0][..3], before[0][..3]);
            assert_eq!(seam[1][..3], before[1][..3]);
            // The ids the other way round: the partner is rotated the other way.
            let ids_swapped = [11_u32, 10];
            let pairs = m.find_pairs(&d, t, &ids_swapped, None, |i| {
                Vec3::new(before[i][0], before[i][1], before[i][2])
            });
            assert_eq!(
                pairs,
                vec![Pair {
                    i: 1,
                    j: 0,
                    shift: -1
                }]
            );
            let mut sw = before;
            let tw = m.collide_step(&d, t, 356.0, &ids_swapped, None, &mut sw);
            assert_eq!(tw.collisions, 1);
            for k in 0..2 {
                for c in 3..6 {
                    assert!(
                        (sw[k][c] - seam[k][c]).abs() < 1e-11,
                        "swapped ids, k = {k}"
                    );
                }
            }
        }
    }
    // A receding pair across the seam is found but not collided.
    let a = arc(10.0, 0.0);
    let d = m.arc_domain(A0_KM, a).unwrap();
    let h = a.width_rad() / 2.0;
    let mut st = [
        particle(A0_KM, h - 0.3 * tan_a, 0.0, [-1e-3, -2e-3, 1e-3], vc),
        particle(
            A0_KM + 0.004,
            -h + 0.7 * tan_a,
            0.003,
            [1e-3, 1e-3, -5e-4],
            vc,
        ),
    ];
    let before = st;
    let tally = m.collide_step(&d, 0.0, 356.0, &ids, None, &mut st);
    assert_eq!(tally.collisions, 0);
    assert_eq!(st, before);
}
