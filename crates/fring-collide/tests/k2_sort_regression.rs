//! K2-SORT: the faster `CellList::group_with_keys` and `find_pairs` give the
//! same result, bit for bit, as the first versions of K2 and K3.
//!
//! This file keeps copies of the old code (the serial sort; the 27-key
//! neighbour list with a binary search per key) as references. They are
//! slow and simple on purpose. The edge cases: 1, 2 and 3 cells along the
//! ring (neighbours repeat), the arc seam, negative radial and vertical keys,
//! all particles in one cell, no particles, one particle, all inactive.

use std::f64::consts::TAU;

use fring_collide::frame::{CellGrid, CellKey};
use fring_collide::hard_sphere::{HardSphereModel, Pair};
use fring_collide::stochastic::{CellList, Domain};
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::physics::arc::ArcGeometry;

const A0_KM: f64 = 140_222.43;
const N0: f64 = 4.5e-4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

// ---- references: the old code ----

fn old_group<K: Ord + Copy>(mut items: Vec<(K, u32, u32)>) -> (Vec<u32>, Vec<usize>, Vec<K>) {
    items.sort_unstable_by_key(|&(k, id, _)| (k, id));
    let mut order = Vec::new();
    let mut starts = Vec::new();
    let mut keys = Vec::new();
    for (n, &(k, _, index)) in items.iter().enumerate() {
        if n == 0 || items[n - 1].0 != k {
            starts.push(n);
            keys.push(k);
        }
        order.push(index);
    }
    starts.push(items.len());
    (order, starts, keys)
}

fn old_neighbours(key: CellKey, ny: u32) -> Vec<CellKey> {
    let mut out = Vec::new();
    for dx in -1..=1_i64 {
        for dy in -1..=1_i64 {
            for dz in -1..=1_i64 {
                out.push(CellKey {
                    radial: key.radial + dx,
                    azimuthal: (i64::from(key.azimuthal) + dy).rem_euclid(i64::from(ny)) as u32,
                    vertical: key.vertical + dz,
                });
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn rot_z(v: Vec3<f64>, c: f64, s: f64) -> Vec3<f64> {
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// The first K3 `find_pairs` (edge = 2s (1 + 1e-6)), with the old search.
fn old_find_pairs(
    s_km: f64,
    domain: &Domain,
    ids: &[u32],
    active: Option<&[bool]>,
    pos: &[Vec3<f64>],
) -> Vec<Pair> {
    let rho_min = (0..ids.len())
        .filter(|&i| active.is_none_or(|a| a[i]))
        .map(|i| pos[i].x.hypot(pos[i].y))
        .fold(f64::INFINITY, f64::min);
    if !rho_min.is_finite() {
        return Vec::new();
    }
    let edge = 2.0 * s_km * (1.0 + 1.0e-6);
    let d_km = domain.grid.domain_length_km;
    let rho_min = rho_min.max(1.0e-9 * domain.a0_km);
    let n_y = (d_km * rho_min / (edge * domain.a0_km)).floor().max(1.0);
    let grid = CellGrid::new(edge, d_km / n_y, edge, d_km).unwrap();
    let local = Domain { grid, ..*domain };
    let (cells, keys) = local.build_keyed(0.0, ids, active, |i| pos[i]);
    let c2 = (2.0 * s_km).powi(2);
    let width = domain.arc.map(|a| a.width_cos_sin());
    let close = |i: usize, j: usize| -> Option<i8> {
        let (pi, pj) = (pos[i], pos[j]);
        if (pi - pj).norm_squared() < c2 {
            return Some(0);
        }
        let (cw, sw) = width?;
        let mut best: Option<(f64, i8)> = None;
        for shift in [1_i8, -1] {
            let d2 = (pi - rot_z(pj, cw, sw * f64::from(shift))).norm_squared();
            if d2 < c2 && best.is_none_or(|(b, _)| d2 < b) {
                best = Some((d2, shift));
            }
        }
        best.map(|(_, shift)| shift)
    };
    let mut pairs = Vec::new();
    for (c, &key) in keys.iter().enumerate() {
        for nb in old_neighbours(key, grid.azimuthal_cells) {
            let Ok(c2i) = keys.binary_search(&nb) else {
                continue;
            };
            for &i in cells.members(c) {
                for &j in cells.members(c2i) {
                    if ids[i as usize] < ids[j as usize]
                        && let Some(shift) = close(i as usize, j as usize)
                    {
                        pairs.push(Pair { i, j, shift });
                    }
                }
            }
        }
    }
    pairs.sort_unstable_by_key(|p| (ids[p.i as usize], ids[p.j as usize]));
    pairs
}

// ---- group ----

fn check_group<K: Ord + Copy + std::fmt::Debug>(items: Vec<(K, u32, u32)>) {
    let (order, starts, keys) = old_group(items.clone());
    let (list, new_keys) = CellList::group_with_keys(items.clone());
    assert_eq!(new_keys, keys);
    assert_eq!(list.len(), keys.len());
    for c in 0..list.len() {
        assert_eq!(
            list.members(c),
            &order[starts[c]..starts[c + 1]],
            "cell {c}"
        );
    }
    assert_eq!(CellList::group(items), list);
}

#[test]
fn group_matches_the_old_sort() {
    check_group::<CellKey>(Vec::new());
    let key = |r, a, v| CellKey {
        radial: r,
        azimuthal: a,
        vertical: v,
    };
    check_group(vec![(key(0, 0, 0), 5, 0)]);
    let mut rng = Rng(7);
    for (n, span) in [
        (50, 1),
        (500, 3),
        (5000, 6),
        (20000, 200),
        (20000, 4_000_000),
    ] {
        let items: Vec<_> = (0..n)
            .map(|i| {
                let k = |rng: &mut Rng| (rng.next() * span as f64) as i64 - span as i64 / 2;
                let (r, a, v) = (k(&mut rng), k(&mut rng).unsigned_abs() as u32, k(&mut rng));
                // Ids are a shuffled range, so that the id order differs from the index order.
                (key(r, a, v), (i * 7919 + 13) as u32 % 1_000_003, i as u32)
            })
            .collect();
        check_group(items);
    }
    // A whole ring in one cell, ids in reverse index order.
    check_group((0..1000).map(|i| (key(-1, 2, -3), 999 - i, i)).collect());
    check_group((0..1000).map(|i| (i % 7, 999 - i, i)).collect());
}

// ---- find_pairs ----

struct Case {
    name: String,
    s_km: f64,
    domain: Domain,
    pos: Vec<Vec3<f64>>,
}

fn arc_case(n_cells: f64, n: usize, seed: u64) -> Case {
    let s_km = 1.0;
    let edge = 2.0 * s_km * (1.0 + 1.0e-6);
    let width = (n_cells + 0.3) * edge / A0_KM;
    let arc = ArcGeometry::new(width, 0.7, N0).unwrap();
    let domain = HardSphereModel::new(s_km, Restitution::Constant { epsilon: 0.5 })
        .arc_domain(A0_KM, arc)
        .unwrap();
    let mut rng = Rng(seed);
    let centre = arc.centre_at_rad(0.0);
    let pos = (0..n)
        .map(|_| {
            // A little beyond the edges too: the wrap slack.
            let lon = centre + rng.range(-0.5, 0.5) * width * (1.0 + 2.0e-6);
            let rho = A0_KM + rng.range(-3.0, 3.0);
            Vec3::new(rho * lon.cos(), rho * lon.sin(), rng.range(-3.0, 3.0))
        })
        .collect();
    Case {
        name: format!("arc of {n_cells} cells, N = {n}"),
        s_km,
        domain,
        pos,
    }
}

fn ring_case(s_km: f64, n: usize, spread_km: f64, seed: u64) -> Case {
    let domain = HardSphereModel::new(s_km, Restitution::Constant { epsilon: 0.5 })
        .full_ring_domain(A0_KM, N0)
        .unwrap();
    let mut rng = Rng(seed);
    let pos = (0..n)
        .map(|_| {
            let lon = rng.range(0.0, TAU);
            let rho = A0_KM + rng.range(-spread_km, spread_km);
            Vec3::new(
                rho * lon.cos(),
                rho * lon.sin(),
                rng.range(-spread_km, spread_km),
            )
        })
        .collect();
    Case {
        name: format!("ring s = {s_km} km, N = {n}"),
        s_km,
        domain,
        pos,
    }
}

fn compare(case: &Case, ids: &[u32], active: Option<&[bool]>) -> Vec<Pair> {
    let model = HardSphereModel::new(case.s_km, Restitution::Constant { epsilon: 0.5 });
    let new = model.find_pairs(&case.domain, 0.0, ids, active, |i| case.pos[i]);
    let old = old_find_pairs(case.s_km, &case.domain, ids, active, &case.pos);
    assert_eq!(new, old, "{}", case.name);
    new
}

fn shuffled_ids(n: usize, seed: u64) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..n as u32).map(|i| i * 3 + 1).collect();
    let mut rng = Rng(seed);
    for i in (1..n).rev() {
        ids.swap(i, (rng.next() * (i + 1) as f64) as usize);
    }
    ids
}

#[test]
fn find_pairs_matches_the_old_search_on_arcs() {
    let mut seam = 0;
    for (k, n_cells) in [1.0, 2.0, 3.0, 4.0, 5.0, 40.0].into_iter().enumerate() {
        let n = if n_cells < 6.0 { 300 } else { 4000 };
        let case = arc_case(n_cells, n, 100 + k as u64);
        let ids = shuffled_ids(n, 5 + k as u64);
        let pairs = compare(&case, &ids, None);
        assert!(
            !pairs.is_empty(),
            "{}: no pairs, the test is empty",
            case.name
        );
        seam += pairs.iter().filter(|p| p.shift != 0).count();
        // Half the particles inactive.
        let active: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        compare(&case, &ids, Some(&active));
    }
    assert!(
        seam > 0,
        "no pair across the seam: the test misses the shift branch"
    );
}

#[test]
fn find_pairs_matches_the_old_search_on_the_full_ring() {
    // Few cells along the ring (huge particles): 1, 2 and 3 cells.
    for (s_km, seed) in [(2.5e5, 1), (1.8e5, 2), (1.2e5, 3)] {
        let case = ring_case(s_km, 60, 50.0, seed);
        let pairs = compare(&case, &shuffled_ids(60, seed), None);
        assert!(!pairs.is_empty());
    }
    // Many cells: dense, with negative radial and vertical keys.
    let case = ring_case(2000.0, 6000, 40.0, 11);
    let n = case.pos.len();
    let pairs = compare(&case, &shuffled_ids(n, 9), None);
    assert!(!pairs.is_empty());
    // All in one cell: a tight clump on the +x axis.
    let mut rng = Rng(12);
    let pos = (0..200)
        .map(|_| {
            Vec3::new(
                A0_KM + rng.range(0.0, 0.5),
                rng.range(0.0, 0.5),
                rng.range(0.0, 0.5),
            )
        })
        .collect();
    let case = Case {
        pos,
        ..ring_case(5.0, 0, 1.0, 12)
    };
    assert!(!compare(&case, &shuffled_ids(200, 4), None).is_empty());
}

#[test]
fn find_pairs_matches_on_empty_and_inactive_input() {
    let case = ring_case(2.0, 100, 10.0, 21);
    let ids = shuffled_ids(100, 3);
    let none = vec![false; 100];
    assert!(compare(&case, &ids, Some(&none)).is_empty());
    let empty = Case {
        pos: Vec::new(),
        ..ring_case(2.0, 0, 10.0, 1)
    };
    assert!(compare(&empty, &[], None).is_empty());
    let one = ring_case(2.0, 1, 10.0, 2);
    assert!(compare(&one, &[7], None).is_empty());
}
