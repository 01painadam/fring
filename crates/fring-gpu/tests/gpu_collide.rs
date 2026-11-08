//! K5: the stochastic collisions on the GPU (`fring_gpu::collide`, WGSL in
//! `shaders/collide/`). Every test needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_collide -- --ignored --nocapture`
//! (quick forms). The full statistical forms have names that start with
//! `full_`; run them with the filter `full_`.
//!
//! What is exact and what has a tolerance is in the module documentation of
//! `fring_gpu::collide`. Tolerances of the K1 tests are those of `SPEC.md`
//! section 11 and are not relaxed. The bounds that f32 cannot meet at the f64
//! value are in `docs/interfaces/ICR-010.md` (proposed); each test prints the
//! measured value and the bound.

mod common;

use fring_collide::frame::peculiar_velocity;
use fring_collide::pair;
use fring_gpu::bin::Dsm;
use fring_gpu::collide::{COLLIDE_TEST_WGSL, collide_uniform};
use fring_gpu::gpu::Gpu;
use fring_gpu::probe::self_test;
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::physics::{SaturnField, circular_speed};
use fring_model::rng::{mulhilo, philox4x32_10, u01_f32};

fn gpu() -> Option<Gpu> {
    let gpu = Gpu::new()?;
    self_test(&gpu).expect("DS self-test");
    Some(gpu)
}

/// A deterministic stream of u32 words (Philox with a test key).
struct Words {
    n: u32,
}

impl Words {
    fn new() -> Self {
        Words { n: 0 }
    }
    fn u32(&mut self) -> u32 {
        let w = philox4x32_10([self.n, 0, 0, 0xAB], [0x1234_5678, 0x9ABC_DEF0]);
        self.n += 1;
        w[0]
    }
    fn unit(&mut self) -> f64 {
        f64::from(self.u32() >> 8) / f64::from(1u32 << 24)
    }
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
}

fn run_test_kernel(
    gpu: &Gpu,
    entry: &str,
    input: Vec<u32>,
    out_words: usize,
    uniform: &[u32],
    items: usize,
) -> Vec<u32> {
    let mut bufs = vec![input, vec![0u32; out_words]];
    gpu.run(
        COLLIDE_TEST_WGSL,
        entry,
        &mut bufs,
        Some(uniform),
        (items as u32).div_ceil(64),
        1,
    );
    bufs.pop().unwrap()
}

fn f(w: u32) -> f32 {
    f32::from_bits(w)
}

#[test]
#[ignore = "needs a GPU"]
fn philox_and_mulhilo_equal_the_rust_functions() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let mut rng = Words::new();
    let n = 100_000;
    let mut input = Vec::new();
    let mut cases: Vec<([u32; 4], [u32; 2])> = vec![
        ([0; 4], [0; 2]),
        ([u32::MAX; 4], [u32::MAX; 2]),
        ([0xFFFF_FFFF, 0, 0, 0x0002_FFFF], [0xDEAD_BEEF, 1]),
    ];
    while cases.len() < n {
        let c = [
            rng.u32(),
            rng.u32(),
            rng.u32(),
            (2 << 16) | (rng.u32() & 0xFFFF),
        ];
        cases.push((c, [rng.u32(), rng.u32()]));
    }
    for (c, k) in &cases {
        input.extend(c);
        input.extend(k);
    }
    let out = run_test_kernel(
        &gpu,
        "t_philox",
        input,
        4 * n,
        &[n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        n,
    );
    let bad = cases
        .iter()
        .enumerate()
        .filter(|(i, (c, k))| philox4x32_10(*c, *k)[..] != out[4 * i..4 * i + 4])
        .count();
    println!("Philox: {n} counters, {bad} differ (must be 0)");
    assert_eq!(bad, 0);

    let mut input = Vec::new();
    let mut pairs: Vec<(u32, u32)> = vec![
        (0, 0),
        (u32::MAX, u32::MAX),
        (u32::MAX, 1),
        (0xFFFF, 0xFFFF),
        (0x1_0000, 0xFFFF_0000),
        (0x8000_0000, 2),
    ];
    while pairs.len() < n {
        pairs.push((rng.u32(), rng.u32()));
    }
    for (a, b) in &pairs {
        input.extend([*a, *b]);
    }
    let out = run_test_kernel(
        &gpu,
        "t_mulhilo",
        input,
        3 * n,
        &[n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        n,
    );
    let bad = pairs
        .iter()
        .enumerate()
        .filter(|(i, (a, b))| {
            let (hi, lo) = mulhilo(*a, *b);
            out[3 * i] != hi || out[3 * i + 1] != lo || out[3 * i + 2] != u01_f32(*a).to_bits()
        })
        .count();
    println!("mulhilo and u01_f32: {n} cases, {bad} differ (must be 0)");
    assert_eq!(bad, 0);
}

/// The pair operator and the impact geometry (K1-T1 through the WGSL twin at
/// 1e-6 relative, SPEC 4; K1-T3 values).
fn pair_cases(rng: &mut Words, n: usize) -> Vec<(Vec3<f32>, f32, f32)> {
    let mut v = vec![
        // g along z (basis branch |g_z| >= 0.9), along x, tiny, large.
        (Vec3::new(0.0, 0.0, 1.0e-5), 0.3, 0.7),
        (Vec3::new(0.0, 0.0, -1.0e-5), 0.9, 0.1),
        (Vec3::new(1.0e-5, 0.0, 0.0), 0.5, 0.5),
        (Vec3::new(1.0e-9, 2.0e-9, 3.0e-9), 0.2, 0.2),
        (Vec3::new(1.0, -2.0, 3.0), 0.99, 0.99),
        (Vec3::new(1.0e-5, 1.0e-5, 0.9e-4), 1.0e-3, 0.5),
    ];
    while v.len() < n {
        let s = 10f64.powf(rng.uniform(-7.0, -2.0));
        let g = Vec3::new(
            (s * rng.uniform(-1.0, 1.0)) as f32,
            (s * rng.uniform(-1.0, 1.0)) as f32,
            (s * rng.uniform(-1.0, 1.0)) as f32,
        );
        let u1 = u01_f32(rng.u32());
        let u2 = u01_f32(rng.u32());
        v.push((g, u1, u2));
    }
    v
}

fn pair_run(gpu: &Gpu, cases: &[(Vec3<f32>, f32, f32)], law: &Restitution) -> Vec<[f32; 12]> {
    let field = SaturnField::<f64> {
        gm_km3_s2: 1.0,
        reference_radius_km: 1.0,
        j2: 0.0,
        j4: 0.0,
        j6: 0.0,
    };
    let cp = collide_uniform(&field, law);
    let kind = u32::from(matches!(law, Restitution::PowerLaw { .. }));
    let mut uniform = vec![0u32; 16];
    uniform[0] = cases.len() as u32;
    uniform[1] = kind;
    uniform[2] = cp[2].to_bits();
    uniform[3] = cp[3].to_bits();
    uniform[4] = cp[4].to_bits();
    let mut input = Vec::new();
    for (g, u1, u2) in cases {
        input.extend([g.x, g.y, g.z, *u1, *u2].map(f32::to_bits));
    }
    let out = run_test_kernel(
        gpu,
        "t_pair",
        input,
        12 * cases.len(),
        &uniform,
        cases.len(),
    );
    out.chunks(12)
        .map(|c| {
            let mut a = [0f32; 12];
            for (x, &w) in a.iter_mut().zip(c) {
                *x = f(w);
            }
            a
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn pair_operator_and_impact_normal_equal_the_f32_twin() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut rng = Words::new();
    let cases = pair_cases(&mut rng, 100_000);
    let laws = [
        Restitution::Constant { epsilon: 0.0 },
        Restitution::Constant { epsilon: 0.5 },
        Restitution::Constant { epsilon: 1.0 },
        // Bridges et al.: p = 0.234, v_c = 0.0077 cm/s = 7.7e-8 km/s.
        Restitution::PowerLaw {
            reference_speed_km_s: 7.7e-8,
            exponent: 0.234,
        },
    ];
    for law in &laws {
        let out = pair_run(&gpu, &cases, law);
        let (mut e_n, mut e_d, mut e_l, mut e_norm) = (0f64, 0f64, 0f64, 0f64);
        let mut hits = 0;
        for ((g, u1, u2), o) in cases.iter().zip(&out) {
            let n_hat = pair::impact_normal(*g, *u1, *u2);
            let gm = f64::from(g.dot(*g).sqrt());
            let gpu_n = Vec3::new(o[0], o[1], o[2]);
            e_n = e_n.max(f64::from((gpu_n - n_hat).dot(gpu_n - n_hat).sqrt()));
            e_norm = e_norm.max(f64::from((gpu_n.dot(gpu_n).sqrt() - 1.0).abs()));
            let imp = pair::collide(*g, n_hat, law);
            let hit = o[8] == f32::from_bits(1) || o[8].to_bits() == 1;
            assert_eq!(hit, imp.is_some(), "hit flag for {g:?}");
            if let Some(i) = imp {
                hits += 1;
                let d = Vec3::new(o[3], o[4], o[5]);
                let dd = d - i.delta_ui_km_s;
                e_d = e_d.max(f64::from(dd.dot(dd).sqrt()) / gm);
                e_l = e_l.max(f64::from((o[6] - i.energy_loss_km2_s2).abs()) / (gm * gm));
            }
        }
        println!(
            "pair {law:?}: {hits} collisions of {}; max |n_gpu - n_twin| {e_n:.2e}, \
             | |n| - 1 | {e_norm:.2e} (own extra check, bound 4 f32 ulp), max |dU| error / |g| {e_d:.2e}, \
             max loss error / g^2 {e_l:.2e} (bound 1e-6 relative to |g|, SPEC 4)",
            cases.len()
        );
        assert!(e_n <= 1e-6, "normal differs by {e_n}");
        assert!(
            e_norm <= 4.0 * f64::from(f32::EPSILON),
            "|n| - 1 = {e_norm}"
        );
        assert!(e_d <= 1e-6, "delta differs by {e_d} |g|");
        assert!(e_l <= 1e-6, "loss differs by {e_l} g^2");
    }
    // Receding and grazing pairs: nothing changes (flag u2 < 0 sends n along g).
    let cases: Vec<(Vec3<f32>, f32, f32)> = (0..64)
        .map(|k| (Vec3::new(1e-5 * (k as f32 - 31.5), 2e-6, -3e-6), 0.5, -1.0))
        .collect();
    let out = pair_run(&gpu, &cases, &Restitution::Constant { epsilon: 0.5 });
    for o in &out {
        assert_eq!(o[8].to_bits(), 0, "a receding pair must not collide");
        assert_eq!((o[3], o[4], o[5], o[6]), (0.0, 0.0, 0.0, 0.0));
    }
    println!("receding pairs: 64 of 64 unchanged (bit for bit)");
}

/// K1-T2 on the GPU (an extra check; SPEC 11 lists T2 for the CPU): cos^2 theta
/// and psi/2 pi uniform; KS statistic below 1.95 / sqrt(N) at 10^6 draws.
#[test]
#[ignore = "needs a GPU"]
fn impact_geometry_is_uniform_on_the_gpu() {
    let Some(gpu) = gpu() else {
        return;
    };
    let n = 1_000_000;
    let g = Vec3::new(3.0e-6f32, -4.0e-6, 5.0e-6);
    let cases: Vec<_> = (0..n)
        .map(|k| {
            let w = philox4x32_10([k as u32, 0, 0, 7], [11, 13]);
            (g, u01_f32(w[0]), u01_f32(w[1]))
        })
        .collect();
    let out = pair_run(&gpu, &cases, &Restitution::Constant { epsilon: 0.5 });
    let g64 = [f64::from(g.x), f64::from(g.y), f64::from(g.z)];
    let gn = (g64[0] * g64[0] + g64[1] * g64[1] + g64[2] * g64[2]).sqrt();
    let gh = Vec3::new(g.x / gn as f32, g.y / gn as f32, g.z / gn as f32);
    let (e1, e2) = pair::basis(gh);
    let (mut c2, mut psi) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for o in &out {
        let nh = Vec3::new(o[0], o[1], o[2]);
        let c = f64::from(nh.dot(gh));
        c2.push(c * c);
        let a = f64::from(nh.dot(e2)).atan2(f64::from(nh.dot(e1)));
        psi.push(a.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU);
    }
    let ks = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let m = v.len() as f64;
        v.iter()
            .enumerate()
            .map(|(i, &x)| {
                ((i as f64 + 1.0) / m - x)
                    .abs()
                    .max((x - i as f64 / m).abs())
            })
            .fold(0.0, f64::max)
    };
    let (k1, k2) = (ks(&mut c2), ks(&mut psi));
    let crit = 1.95 / (n as f64).sqrt();
    println!(
        "K1-T2 on the GPU: KS(cos^2 theta) {k1:.3e}, KS(psi/2pi) {k2:.3e}, bound {crit:.3e} (N = {n})"
    );
    assert!(k1 < crit && k2 < crit);
}

fn field() -> SaturnField<f64> {
    let system = common::system(356.0, 10, false);
    SaturnField::<f64>::from_params(&system.saturn)
}

fn peculiar_input(
    field: &SaturnField<f64>,
    rng: &mut Words,
    n: usize,
    circular: bool,
) -> Vec<([f64; 3], [f64; 3])> {
    (0..n)
        .map(|_| {
            let rho = rng.uniform(1.30e5, 1.50e5);
            let phi = rng.uniform(0.0, std::f64::consts::TAU);
            let z = rng.uniform(-0.05, 0.05);
            let pos = [rho * phi.cos(), rho * phi.sin(), z];
            let vc = circular_speed(field, rho);
            // Peculiar velocity: 0 (circular) or up to 0.04 km/s (forced eccentricity, SPEC 12).
            let s = if circular {
                0.0
            } else {
                10f64.powf(rng.uniform(-7.0, -1.5))
            };
            let (ux, uy, uz) = (
                s * rng.uniform(-1.0, 1.0),
                s * rng.uniform(-1.0, 1.0),
                s * rng.uniform(-1.0, 1.0),
            );
            let vel = [-vc * phi.sin() + ux, vc * phi.cos() + uy, uz];
            (pos, vel)
        })
        .collect()
}

fn peculiar_gpu(
    gpu: &Gpu,
    field: &SaturnField<f64>,
    items: &[([f64; 3], [f64; 3])],
) -> Vec<([f64; 3], f64)> {
    let mut input = Vec::new();
    for (p, v) in items {
        for x in [p[0], p[1], v[0], v[1], v[2]] {
            let d = Dsm::split(x);
            input.extend([d.hi.to_bits(), d.lo.to_bits()]);
        }
    }
    let cp = collide_uniform(field, &Restitution::Constant { epsilon: 1.0 });
    let mut uniform = vec![0u32; 16];
    uniform[0] = items.len() as u32;
    uniform[1] = 1.0f32.to_bits();
    for k in 0..10 {
        uniform[4 + k] = cp[8 + k].to_bits();
    }
    let out = run_test_kernel(
        gpu,
        "t_peculiar",
        input,
        8 * items.len(),
        &uniform,
        items.len(),
    );
    out.chunks(8)
        .map(|c| {
            let ds = |k: usize| f64::from(f(c[2 * k])) + f64::from(f(c[2 * k + 1]));
            ([ds(0), ds(1), ds(2)], ds(3))
        })
        .collect()
}

/// The peculiar velocity from the DS state against the f64 function
/// (`frame::peculiar_velocity`), and K1-T8a (circular orbits: u = 0).
#[test]
#[ignore = "needs a GPU"]
fn peculiar_velocity_equals_the_f64_function() {
    let Some(gpu) = gpu() else {
        return;
    };
    let field = field();
    let mut rng = Words::new();
    for circular in [true, false] {
        let items = peculiar_input(&field, &mut rng, 100_000, circular);
        let out = peculiar_gpu(&gpu, &field, &items);
        let (mut e_u, mut e_vc, mut u_max) = (0f64, 0f64, 0f64);
        for ((p, v), (u, vc)) in items.iter().zip(&out) {
            let u64_ = peculiar_velocity(
                &field,
                Vec3::new(p[0], p[1], p[2]),
                Vec3::new(v[0], v[1], v[2]),
            );
            let d = ((u[0] - u64_.x).powi(2) + (u[1] - u64_.y).powi(2) + (u[2] - u64_.z).powi(2))
                .sqrt();
            e_u = e_u.max(d);
            u_max = u_max.max((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt());
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            e_vc = e_vc.max((vc / circular_speed(&field, rho) - 1.0).abs());
        }
        println!(
            "peculiar velocity ({}): max |u_gpu - u_f64| {e_u:.2e} km/s, max |u_gpu| {u_max:.2e} km/s, \
             v_circ relative error {e_vc:.2e} (T8a bound 1e-8 km/s = 1e-3 c at c = 1 cm/s, SPEC 12)",
            if circular {
                "circular orbits, K1-T8a"
            } else {
                "random u"
            }
        );
        assert!(e_u <= 1e-8, "u differs by {e_u} km/s");
        assert!(e_vc <= 1e-12, "v_circ relative error {e_vc}");
        if circular {
            assert!(u_max <= 1e-8, "a circular orbit has |u| = {u_max} km/s");
        }
    }
}
