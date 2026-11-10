//! K2-SORT benchmark: time per stage of the cell grouping and of the
//! hard-sphere pair search on a synthetic ring (serial, one thread).
//!
//! Run: `cargo run -p fring-collide --release --example k2_sort_bench [N ...]`
//! (default N = 100000 and 1000000). Each stage is timed 3 times; the table
//! shows the smallest time. The particles are a ring of 100 km width and
//! 0.5 km height at a0 = 140222.43 km. The radius s is set so that a cell
//! (edge 2s) holds about one particle, as in the production rings.
//!
//! The stage "pair search" is `find_pairs` minus the scan and `build_keyed`:
//! the neighbour search, the pair tests and the final pair sort.
//! (This bench uses a small local generator, not the model RNG: it makes
//! positions only and nothing in a model depends on it.)

use std::f64::consts::TAU;
use std::time::Instant;

use fring_collide::hard_sphere::HardSphereModel;
use fring_collide::stochastic::CellList;
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::real::sin_cos;

const A0_KM: f64 = 140_222.43;
const WIDTH_KM: f64 = 100.0;
const HEIGHT_KM: f64 = 0.5;

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn best_of<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..3 {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (best, out.unwrap())
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![100_000, 1_000_000]
    } else {
        sizes
    };
    println!("N, stage, ms");
    for n in sizes {
        let mut rng = 12345_u64;
        let pos: Vec<Vec3<f64>> = (0..n)
            .map(|_| {
                let rho = A0_KM + WIDTH_KM * (splitmix(&mut rng) - 0.5);
                let phi = TAU * splitmix(&mut rng);
                let z = HEIGHT_KM * (splitmix(&mut rng) - 0.5);
                let (sin_phi, cos_phi) = sin_cos(phi);
                Vec3::new(rho * cos_phi, rho * sin_phi, z)
            })
            .collect();
        let ids: Vec<u32> = (0..n as u32).collect();
        let s_km = (TAU * A0_KM * WIDTH_KM / (4.0 * n as f64)).sqrt();
        let model = HardSphereModel::new(s_km, Restitution::Constant { epsilon: 0.5 });
        let domain = model.full_ring_domain(A0_KM, 4.5e-4).unwrap();
        let position = |i: usize| pos[i];

        let (t_scan, _) = best_of(|| {
            (0..n)
                .map(|i| position(i).x.hypot(position(i).y))
                .fold(f64::INFINITY, f64::min)
        });
        // The same cell grid as find_pairs, for the grouping stage.
        let edge = 2.0 * s_km * (1.0 + 1.0e-6);
        let rho_min = A0_KM - 0.5 * WIDTH_KM;
        let d = domain.grid.domain_length_km;
        let n_y = (d * rho_min / (edge * A0_KM)).floor().max(1.0);
        let grid = fring_collide::frame::CellGrid::new(edge, d / n_y, edge, d).unwrap();
        let local = fring_collide::stochastic::Domain { grid, ..domain };
        let (t_group, cells): (f64, (CellList, _)) =
            best_of(|| local.build_keyed(0.0, &ids, None, position));
        let (t_all, pairs) = best_of(|| model.find_pairs(&domain, 0.0, &ids, None, position));
        println!("{n}, scan, {t_scan:.2}");
        println!("{n}, build_keyed (group), {t_group:.2}");
        println!(
            "{n}, pair search (find_pairs - scan - group), {:.2}",
            t_all - t_scan - t_group
        );
        println!("{n}, find_pairs total, {t_all:.2}");
        println!(
            "# N={n}: s={s_km:.3} km, cells={}, pairs={}",
            cells.0.len(),
            pairs.len()
        );
    }
}
