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
use fring_collide::stochastic::StochasticModel;
use fring_collide::tally::Tally;
use fring_collide::testkit::{
    BoxModel, BoxSpec, Check, ModelKind, ModelParams, Sample, Setup, dispersion, kinetic_energy,
    momentum, t4_rate, t5_energy_loss, t7_haff, t9_determinism,
};
use fring_gpu::bin::Dsm;
use fring_gpu::collide::{COLLIDE_TEST_WGSL, CollideParams, GpuCollider, collide_uniform};
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

// ---------------------------------------------------------------- box tests --

/// A box model on the GPU for the K1 drivers (`testkit::BoxModel`): the host
/// sorts the particles into cells (id order) and writes the sorted ids and keys
/// in the layout of `GpuBinner`; `GpuCollider` runs in box mode (u = v). With
/// `bulk` the velocity of every particle carries a bulk velocity in the
/// double-single state (K1-T6 with 16.5 km/s). The GPU keeps the state between
/// calls when the caller passes back the velocities it got; so the bulk part
/// is not rounded again at each step.
struct GpuBox {
    gpu: Gpu,
    params: ModelParams,
    bulk: Vec3<f64>,
    inner: Option<Inner>,
    last: Vec<Vec3<f64>>,
}

struct Inner {
    collider: GpuCollider,
    state: wgpu::Buffer,
    sorted: wgpu::Buffer,
    skeys: wgpu::Buffer,
    cap: usize,
}

fn buffer(gpu: &Gpu, bytes: usize, copy_src: bool) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes.max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | if copy_src {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            },
        mapped_at_creation: false,
    })
}

impl GpuBox {
    fn new(gpu: &Gpu, params: ModelParams) -> Self {
        Self::with_bulk(gpu, params, Vec3::zero())
    }

    fn with_bulk(gpu: &Gpu, params: ModelParams, bulk: Vec3<f64>) -> Self {
        Self {
            gpu: gpu.clone(),
            params,
            bulk,
            inner: None,
            last: Vec::new(),
        }
    }
}

fn words_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl BoxModel for GpuBox {
    fn collide(
        &mut self,
        cells: &BoxSpec,
        pos_km: &[Vec3<f64>],
        vel_km_s: &mut [Vec3<f64>],
        step: u64,
        dt_s: f64,
    ) -> Tally {
        let n = pos_km.len();
        let cap = n.next_multiple_of(64);
        if self.inner.is_none() {
            let state = buffer(&self.gpu, 6 * cap * 8, true);
            let sorted = buffer(&self.gpu, 4 * n, false);
            let skeys = buffer(&self.gpu, 8 * n, false);
            let params = CollideParams {
                particle_radius_km: self.params.particle_radius_km,
                weight: self.params.weight,
                restitution: self.params.restitution,
                seed: self.params.seed,
            };
            let collider = GpuCollider::new(
                &self.gpu,
                &state,
                &sorted,
                &skeys,
                n,
                cap,
                &field(),
                params,
                cells.cell_volume_km3(),
                true,
            )
            .expect("collider");
            self.inner = Some(Inner {
                collider,
                state,
                sorted,
                skeys,
                cap,
            });
        }
        let inner = self.inner.as_ref().unwrap();
        let q = self.gpu.queue();
        // Cells in id order.
        let mut order: Vec<(usize, u32)> = pos_km
            .iter()
            .enumerate()
            .map(|(i, &p)| (cells.cell_of(p), i as u32))
            .collect();
        order.sort_unstable();
        let sorted: Vec<u32> = order.iter().map(|&(_, i)| i).collect();
        let keys: Vec<u32> = order.iter().flat_map(|&(c, _)| [c as u32, 0]).collect();
        q.write_buffer(&inner.sorted, 0, &words_bytes(&sorted));
        q.write_buffer(&inner.skeys, 0, &words_bytes(&keys));
        if self.last.as_slice() != &*vel_km_s {
            let mut w = vec![0f32; 12 * cap];
            for (i, v) in vel_km_s.iter().enumerate() {
                let u = *v + self.bulk;
                for (c, x) in [u.x, u.y, u.z].into_iter().enumerate() {
                    let d = Dsm::split(x);
                    w[2 * ((3 + c) * cap + i)] = d.hi;
                    w[2 * ((3 + c) * cap + i) + 1] = d.lo;
                }
            }
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            q.write_buffer(&inner.state, 0, &bytes);
        }
        let mut enc = self
            .gpu
            .device()
            .create_command_encoder(&Default::default());
        inner.collider.encode(&mut enc, step, dt_s);
        q.submit([enc.finish()]);
        let tally = inner.collider.read_tally().expect("tally");
        let w = inner
            .collider
            .read_words(&inner.state, 2 * 3 * inner.cap, 2 * 3 * inner.cap);
        for (i, v) in vel_km_s.iter_mut().enumerate() {
            let get = |c: usize| {
                let k = 2 * (c * inner.cap + i);
                f64::from(f(w[k])) + f64::from(f(w[k + 1]))
            };
            *v = Vec3::new(get(0), get(1), get(2)) - self.bulk;
        }
        self.last = vel_km_s.to_vec();
        tally
    }
}

const S: ModelKind = ModelKind::Stochastic;

fn power_law() -> Restitution {
    Restitution::PowerLaw {
        reference_speed_km_s: 1e-6,
        exponent: 0.234,
    }
}

/// CPU against GPU: the relative errors against the own theory differ by at most
/// 4 combined standard errors (SPEC 11, last paragraph of the table).
fn compare(g: &Check, c: &Check) {
    let (eg, ec) = (g.relative_error(), c.relative_error());
    let se = ((g.standard_error / g.theory).powi(2) + (c.standard_error / c.theory).powi(2)).sqrt();
    let z = if se > 0.0 { (eg - ec) / se } else { 0.0 };
    println!("  CPU against GPU: GPU {eg:+.4e}, CPU {ec:+.4e}, difference {z:+.2} combined SE");
    assert!(z.abs() <= 4.0, "CPU and GPU differ by {z} combined SE");
}

fn cpu(p: ModelParams) -> StochasticModel {
    StochasticModel::from_params(p)
}

fn t4(sample: Sample, occupancies: &[f64]) {
    let Some(gpu) = gpu() else {
        return;
    };
    for &o in occupancies {
        let g = t4_rate(S, o, sample, |p| GpuBox::new(&gpu, p));
        g.assert_ok();
        let c = t4_rate(S, o, sample, cpu);
        compare(&g, &c);
    }
}

fn t5(sample: Sample, laws: &[Restitution]) {
    let Some(gpu) = gpu() else {
        return;
    };
    for law in laws {
        let g = t5_energy_loss(S, *law, sample, |p| GpuBox::new(&gpu, p));
        g.assert_ok();
        let c = t5_energy_loss(S, *law, sample, cpu);
        if g.theory != 0.0 {
            compare(&g, &c);
        }
    }
}

fn t7(sample: Sample) {
    let Some(gpu) = gpu() else {
        return;
    };
    let g = t7_haff(S, sample, |p| GpuBox::new(&gpu, p));
    g.assert_ok();
    let c = t7_haff(S, sample, cpu);
    compare(&g, &c);
}

#[test]
#[ignore = "needs a GPU"]
fn quick_t4_rate() {
    t4(Sample::Quick, &[2.0, 5.0]);
}

#[test]
#[ignore = "needs a GPU"]
fn quick_t5_energy_loss() {
    t5(
        Sample::Quick,
        &[
            Restitution::Constant { epsilon: 0.5 },
            Restitution::Constant { epsilon: 1.0 },
        ],
    );
}

#[test]
#[ignore = "needs a GPU"]
fn quick_t7_haff() {
    t7(Sample::Quick);
}

#[test]
#[ignore = "needs a GPU"]
fn k1_t9_determinism_on_the_gpu() {
    let Some(gpu) = gpu() else {
        return;
    };
    t9_determinism(S, |p| GpuBox::new(&gpu, p));
    println!("K1-T9 on the GPU: same seed bit for bit, another seed other velocities: ok");
}

#[test]
#[ignore = "needs a GPU; long"]
fn full_t4_rate() {
    t4(Sample::Full, &[2.0, 5.0, 20.0]);
}

#[test]
#[ignore = "needs a GPU; long"]
fn full_t5_energy_loss() {
    let mut laws: Vec<Restitution> = [0.0, 0.5, 0.9, 1.0]
        .iter()
        .map(|&e| Restitution::Constant { epsilon: e })
        .collect();
    laws.push(power_law());
    t5(Sample::Full, &laws);
}

#[test]
#[ignore = "needs a GPU; long"]
fn full_t7_haff() {
    t7(Sample::Full);
}

/// K1-T6 on the GPU with the bounds of `ICR-010` (proposed). Momentum: 4 u_DS
/// max|u| collisions, u_DS = 2^-48 (ADR 0006), as `SPEC.md` 11 says for the bulk
/// case. Energy: the f64 bounds (1e-12, 1e-10) cannot hold with f32 velocity
/// changes. Each collision changes the energy by about the rounding of Delta-u,
/// 2 u_f32 |g|^2 with u_f32 = 2^-24; the bound adds 4 of these per collision in a
/// straight sum (no cancellation): 8 u_f32 N_coll <g^2> / K0, <g^2> = 8 c0^2.
fn t6(epsilon: f64, bulk: Vec3<f64>) {
    let Some(gpu) = gpu() else {
        return;
    };
    let law = Restitution::Constant { epsilon };
    let setup = Setup::stochastic(8, 5.0, 0.02, law, 61);
    let mut state = setup.initial_state(0);
    let mut model = GpuBox::with_bulk(&gpu, setup.params, bulk);
    let p0 = momentum(&state.vel_km_s);
    let k0 = kinetic_energy(&state.vel_km_s);
    let c0 = dispersion(&state.vel_km_s);
    let max_speed = |v: &[Vec3<f64>]| v.iter().map(|u| (*u + bulk).norm()).fold(0.0, f64::max);
    let mut umax = max_speed(&state.vel_km_s);
    let mut tally = Tally::default();
    for step in 1..=1000 {
        setup.stream(&mut state, setup.dt_s);
        tally += model.collide(
            &setup.cells,
            &state.pos_km,
            &mut state.vel_km_s,
            step,
            setup.dt_s,
        );
        umax = umax.max(max_speed(&state.vel_km_s));
    }
    let dp = momentum(&state.vel_km_s) - p0;
    let dp = dp.x.abs().max(dp.y.abs()).max(dp.z.abs());
    let u_ds = 2f64.powi(-48);
    let p_bound = 4.0 * u_ds * umax * tally.collisions as f64;
    let k1 = kinetic_energy(&state.vel_km_s);
    let elastic = epsilon == 1.0;
    let e_err = if elastic {
        (k1 - k0).abs() / k0
    } else {
        (k0 - k1 - tally.stats.energy_loss_km2_s2).abs() / k0
    };
    let e_bound = 8.0 * 2f64.powi(-24) * tally.collisions as f64 * 8.0 * c0 * c0 / k0;
    println!(
        "K1-T6 GPU eps {epsilon}, bulk {bulk:?}: {} collisions; momentum error {dp:.2e} km/s (bound {p_bound:.2e}); \
         energy error {e_err:.2e} (ICR-010 bound {e_bound:.2e}; f64 bound {:e})",
        tally.collisions,
        if elastic { 1e-12 } else { 1e-10 }
    );
    assert!(tally.collisions > 0);
    assert!(dp <= p_bound, "momentum");
    assert!(e_err <= e_bound, "energy");
}

#[test]
#[ignore = "needs a GPU"]
fn k1_t6_budget_on_the_gpu() {
    for eps in [0.5, 1.0] {
        t6(eps, Vec3::zero());
    }
}

/// K1-T6 with 16.5 km/s added to the double-single velocities (SPEC 11).
#[test]
#[ignore = "needs a GPU"]
fn k1_t6_budget_with_bulk_velocity_on_the_gpu() {
    for eps in [0.5, 1.0] {
        t6(eps, Vec3::new(16.5, -3.0, 1.0));
    }
}

// ------------------------------------------------- kernel against its twin --

/// `col_cells` against `cell_twin`, cell by cell. Setup: 8^3 cells, about 5
/// particles per cell, a collision probability of 0.3 per step (so some cells
/// have several collisions in one step and the later candidates see the changed
/// velocities). Exact: the number of candidates (summed over all cells, and
/// per step), occupancy sums. With a tolerance: the velocity change of each
/// particle (f32 against f32; a cell is "equal" if all its particles agree to
/// 1e-4 c, which allows a different rounding of an f32 acceptance test only
/// when it does not flip).
#[test]
#[ignore = "needs a GPU"]
fn cell_kernel_equals_its_twin() {
    use fring_gpu::collide::cell_twin;
    use fring_model::rng::Philox;
    let Some(gpu) = gpu() else {
        return;
    };
    let law = Restitution::Constant { epsilon: 0.5 };
    let setup = Setup::stochastic(8, 5.0, 0.3, law, 77);
    let params = CollideParams {
        particle_radius_km: setup.params.particle_radius_km,
        weight: setup.params.weight,
        restitution: law,
        seed: setup.params.seed,
    };
    let rng = Philox::from_seed(params.seed);
    let k_rate = params.rate_factor(setup.dt_s, setup.cells.cell_volume_km3());
    let mut model = GpuBox::new(&gpu, setup.params);
    let (mut cells_total, mut cells_equal, mut cand_gpu, mut cand_twin) = (0u64, 0u64, 0u64, 0u64);
    let (mut coll_gpu, mut coll_twin, mut worst_equal) = (0u64, 0u64, 0f64);
    for trial in 0..20u64 {
        let mut state = setup.initial_state(trial);
        setup.stream(&mut state, setup.dt_s);
        let before = state.vel_km_s.clone();
        let step = 100 + trial;
        let t = model.collide(
            &setup.cells,
            &state.pos_km,
            &mut state.vel_km_s,
            step,
            setup.dt_s,
        );
        let c = dispersion(&before);
        // Group by cell, ids ascending.
        let mut order: Vec<(usize, usize)> = state
            .pos_km
            .iter()
            .enumerate()
            .map(|(i, &p)| (setup.cells.cell_of(p), i))
            .collect();
        order.sort_unstable();
        let mut cand = 0u64;
        for run in order.chunk_by(|a, b| a.0 == b.0) {
            let ids: Vec<usize> = run.iter().map(|r| r.1).collect();
            if ids.len() < 2 {
                continue;
            }
            let mean = ids.iter().fold(Vec3::zero(), |a, &i| a + before[i]) / ids.len() as f64;
            let mut w: Vec<Vec3<f32>> = ids
                .iter()
                .map(|&i| {
                    let d = before[i] - mean;
                    Vec3::new(d.x as f32, d.y as f32, d.z as f32)
                })
                .collect();
            let w0 = w.clone();
            let ct = cell_twin(&rng, &law, k_rate, ids[0] as u32, step, &mut w);
            cand += u64::from(ct.candidates);
            coll_twin += u64::from(ct.collisions);
            let err = ids
                .iter()
                .enumerate()
                .map(|(q, &i)| {
                    let dg = state.vel_km_s[i] - before[i];
                    let dt_ = Vec3::new(
                        f64::from(w[q].x - w0[q].x),
                        f64::from(w[q].y - w0[q].y),
                        f64::from(w[q].z - w0[q].z),
                    );
                    (dg - dt_).norm()
                })
                .fold(0.0, f64::max);
            cells_total += 1;
            if err <= 1e-4 * c {
                cells_equal += 1;
                worst_equal = worst_equal.max(err / c);
            }
        }
        cand_gpu += t.stats.candidates;
        cand_twin += cand;
        coll_gpu += t.collisions;
    }
    println!(
        "cell kernel against twin: {cells_total} cells with 2 or more members, {cells_equal} equal to 1e-4 c \
         (worst equal cell {worst_equal:.2e} c); candidates GPU {cand_gpu}, twin {cand_twin}; \
         collisions GPU {coll_gpu}, twin {coll_twin}"
    );
    assert_eq!(cand_gpu, cand_twin, "the candidates are exact");
    assert!(
        cells_equal as f64 >= 0.999 * cells_total as f64,
        "{cells_equal} of {cells_total}"
    );
    assert!((coll_gpu as f64 - coll_twin as f64).abs() <= 0.005 * coll_twin as f64 + 2.0);
}
