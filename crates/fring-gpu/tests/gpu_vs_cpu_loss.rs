//! E1-GPU, test G4 (PLAN section 10, ICR-004 amendment 2): `GpuBackend` against
//! `CpuBackend`, statistics only (chaotic orbits). Same particles, real moons,
//! `[loss]` with freeze and the radius rule on both. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_vs_cpu_loss -- --ignored --nocapture`.
//!
//! Environment (for a quick run): `E1_G4_N` particles (default 20,000),
//! `E1_G4_YEARS` (default 2.0), `E1_G4_SAMPLE` steps between samples (default 88).
//!
//! Checks, each within sampling error:
//! 1. the number lost by reason (unbound, inside Prometheus, inside Pandora):
//!    `|a - b| < 3 sqrt(a + b)`;
//! 2. the `drop_step` of the lost particles: two-sample KS, p > 0.01;
//! 3. the number of samples inside 1 and 3 R_H of Prometheus and of Pandora
//!    (live particles, `E1_G4_SAMPLE` steps apart, at least 1,000 sample times,
//!    the same times on both backends), `|a - b| < 3 sqrt(a + b)`. The samples
//!    of one encounter are correlated, so the number of distinct particles
//!    that were ever inside is printed too, with the same limit;
//! 4. power: the GPU run with all radii x 1.1 is printed against the CPU, to
//!    show how much a 10 % mistake of the radius changes the numbers.

// Small loops over [moon][radius] tables and particle indexes read better with indexes.
#![allow(clippy::needless_range_loop)]

mod common;

use common::*;
use fring_cpu::CpuBackend;
use fring_gpu::backend::GpuBackend;
use fring_gpu::gpu::Gpu;
use fring_model::physics::{EpicycleElements, SaturnField, epicycle_state};
use fring_model::{Backend, DropReport, InitialState, Particles, RunConfig, System};

const RADII_KM: [f64; 2] = [43.09, 40.63];
const LOSS: &str = "[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n";

struct Sm(u64);
impl Sm {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / 18_446_744_073_709_551_616.0
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `n` particles with a uniform in 139,700 to 141,100 km (the band of
/// A3-ANALYSIS), e 0.0005 to 0.004, small inclinations, random angles.
fn band(system: &System, n: usize) -> InitialState {
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let mut rng = Sm(20_261_108);
    let mut p = Particles::with_capacity(n);
    for i in 0..n {
        let el = EpicycleElements {
            a_km: 139_700.0 + 1_400.0 * rng.next(),
            e: 0.0005 + 0.0035 * rng.next(),
            i_rad: 2.0e-4 * rng.next(),
            mean_anomaly_rad: std::f64::consts::TAU * rng.next(),
            z_phase_rad: std::f64::consts::TAU * rng.next(),
            mean_longitude_rad: std::f64::consts::TAU * rng.next(),
        };
        let (x, v) = epicycle_state(&field, &el);
        p.push(i as u32, x, v);
    }
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

/// Hill radius of moon `k` at the start, km: a (GM_m / 3 GM_S)^(1/3).
fn hill_km(system: &System, k: usize) -> f64 {
    let m = &fring_model::physics::interpolate_moons(&system.moons, 0.0)
        .unwrap()
        .moons[k];
    let gm_s = system.saturn.gm_km3_s2;
    m.position_km.norm() * (system.moons.tracks[k].gm_km3_s2 / (3.0 * gm_s)).cbrt()
}

/// Two-sample Kolmogorov-Smirnov test: (D, asymptotic p).
fn ks(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (n, m) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / n - j as f64 / m).abs());
    }
    let en = (n * m / (n + m)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    if lambda < 1e-6 {
        return (d, 1.0);
    }
    let mut p = 0.0;
    for k in 1..=100 {
        let kf = f64::from(k);
        p += 2.0 * (-1.0f64).powi(k - 1) * (-2.0 * kf * kf * lambda * lambda).exp();
    }
    (d, p.clamp(0.0, 1.0))
}

/// What one backend gives.
struct Result {
    report: DropReport,
    /// Samples inside 1 and 3 R_H, [moon][0: 1 R_H, 1: 3 R_H].
    samples: [[u64; 2]; 2],
    /// Distinct live particles that were ever inside.
    distinct: [[u64; 2]; 2],
    n_samples: usize,
}

fn run_pair(
    config: &RunConfig,
    system: &System,
    init: &InitialState,
    steps: u64,
    every: u64,
    gpu_radius_scale: &[f64],
) -> (Result, Vec<Result>) {
    let mut cpu = CpuBackend::new(config, system, init).unwrap();
    cpu.set_moon_radii_km(&RADII_KM).unwrap();
    let mut gpus: Vec<GpuBackend> = gpu_radius_scale
        .iter()
        .map(|s| {
            let mut g = GpuBackend::new(config, system, init).unwrap();
            g.set_moon_radii_km(&RADII_KM.map(|r| r * s)).unwrap();
            g
        })
        .collect();
    let hill = [hill_km(system, 0), hill_km(system, 1)];
    let n = init.particles.len();
    let acc = |b: &mut dyn Backend, r: &mut Result, ever: &mut [[Vec<bool>; 2]; 2]| {
        let rep = b.dropped().unwrap().unwrap();
        let t = b.moons();
        let v = b.particles().unwrap();
        let p = v.particles;
        for k in 0..2 {
            let m = t.moons[k].position_km;
            for i in 0..n {
                if rep.reason[i] != 0 {
                    continue;
                }
                let d = (p.position_km(i) - m).norm();
                for (c, f) in [1.0, 3.0].iter().enumerate() {
                    if d < f * hill[k] {
                        r.samples[k][c] += 1;
                        ever[k][c][i] = true;
                    }
                }
            }
        }
        r.n_samples += 1;
    };
    let mk = || Result {
        report: DropReport::default(),
        samples: [[0; 2]; 2],
        distinct: [[0; 2]; 2],
        n_samples: 0,
    };
    let fresh = || {
        [
            [vec![false; n], vec![false; n]],
            [vec![false; n], vec![false; n]],
        ]
    };
    let (mut rc, mut ec) = (mk(), fresh());
    let mut rg: Vec<(Result, [[Vec<bool>; 2]; 2])> = gpus.iter().map(|_| (mk(), fresh())).collect();
    let mut done = 0;
    let t0 = std::time::Instant::now();
    while done < steps {
        let s = every.min(steps - done);
        cpu.step(s).unwrap();
        for g in &mut gpus {
            g.step(s).unwrap();
        }
        done += s;
        acc(&mut cpu, &mut rc, &mut ec);
        for (g, (r, e)) in gpus.iter_mut().zip(rg.iter_mut()) {
            acc(g, r, e);
        }
        if (done / every).is_multiple_of(200) {
            eprintln!(
                "step {done} of {steps}, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let finish = |r: &mut Result, e: &[[Vec<bool>; 2]; 2], b: &mut dyn Backend| {
        r.report = b.dropped().unwrap().unwrap();
        for k in 0..2 {
            for c in 0..2 {
                r.distinct[k][c] = e[k][c].iter().filter(|x| **x).count() as u64;
            }
        }
    };
    finish(&mut rc, &ec, &mut cpu);
    for (g, (r, e)) in gpus.iter_mut().zip(rg.iter_mut()) {
        finish(r, e, g);
    }
    (rc, rg.into_iter().map(|x| x.0).collect())
}

fn counts(r: &DropReport) -> [u64; 3] {
    let n = |c: u8| r.reason.iter().filter(|x| **x == c).count() as u64;
    [n(1), n(2), n(3)]
}

fn z(a: u64, b: u64) -> f64 {
    if a + b == 0 {
        0.0
    } else {
        (a as f64 - b as f64) / ((a + b) as f64).sqrt()
    }
}

#[test]
#[ignore = "needs a GPU; about 2 minutes"]
fn g4_gpu_against_cpu_statistics_20000_particles_2_years() {
    if Gpu::new().is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let n: usize = env_or("E1_G4_N", 20_000);
    let years: f64 = env_or("E1_G4_YEARS", 2.0);
    let every: u64 = env_or("E1_G4_SAMPLE", 88);
    let step_s = 360.0;
    let steps = (years * YEAR_S / step_s).floor() as u64;
    let system = system(step_s, steps, true);
    let config = RunConfig::from_toml_str(&config_text(step_s, steps, true, LOSS, None)).unwrap();
    let init = band(&system, n);
    let (cpu, gpus) = run_pair(&config, &system, &init, steps, every, &[1.0, 1.1]);
    let gpu = &gpus[0];
    let n_samples = cpu.n_samples;
    println!(
        "{n} particles, {steps} steps ({years} years), {n_samples} sample times every {every} steps"
    );
    println!(
        "R_H: Prometheus {:.1} km, Pandora {:.1} km",
        hill_km(&system, 0),
        hill_km(&system, 1)
    );
    assert!(
        n_samples >= 1000 || n < 20_000,
        "at least 1,000 sample times"
    );

    // 1. Lost by reason.
    let (cc, cg) = (counts(&cpu.report), counts(&gpu.report));
    println!("lost [unbound, Prometheus, Pandora]: CPU {cc:?}, GPU {cg:?}");
    for (j, name) in ["unbound", "inside Prometheus", "inside Pandora"]
        .iter()
        .enumerate()
    {
        let zz = z(cc[j], cg[j]);
        println!("  {name}: z = {zz:.2}");
        assert!(zz.abs() < 3.0, "{name}: CPU {} GPU {}", cc[j], cg[j]);
    }
    assert_eq!(cpu.report.n_unbound, cc[0]);
    assert_eq!(gpu.report.n_unbound, cg[0]);
    // 2. drop_step of lost particles.
    let steps_of = |r: &DropReport| -> Vec<f64> {
        r.reason
            .iter()
            .zip(&r.drop_step)
            .filter(|(c, _)| **c != 0)
            .map(|(_, s)| *s as f64)
            .collect()
    };
    let (dc, dg) = (steps_of(&cpu.report), steps_of(&gpu.report));
    let (d, p) = ks(&dc, &dg);
    println!(
        "drop_step KS: {} CPU and {} GPU lost, D = {d:.4}, p = {p:.3}",
        dc.len(),
        dg.len()
    );
    assert!(p > 0.01, "drop_step distributions differ: p = {p}");
    // 3. Samples inside 1 and 3 R_H.
    for (k, name) in ["Prometheus", "Pandora"].iter().enumerate() {
        for (c, f) in ["1 R_H", "3 R_H"].iter().enumerate() {
            let (a, b) = (cpu.samples[k][c], gpu.samples[k][c]);
            let (da, db) = (cpu.distinct[k][c], gpu.distinct[k][c]);
            println!(
                "samples inside {f} of {name}: CPU {a}, GPU {b}, z = {:.2}; distinct particles CPU {da}, GPU {db}, z = {:.2}",
                z(a, b),
                z(da, db)
            );
            assert!(z(a, b).abs() < 3.0, "{name} {f} samples");
            assert!(z(da, db).abs() < 3.0, "{name} {f} distinct");
        }
    }
    // 4. Power: all radii x 1.1.
    let g11 = &gpus[1];
    let c11 = counts(&g11.report);
    println!(
        "power check, GPU radii x 1.1: lost [unbound, Prometheus, Pandora] {c11:?} (CPU {cc:?}); z = [{:.2}, {:.2}, {:.2}]",
        z(cc[0], c11[0]),
        z(cc[1], c11[1]),
        z(cc[2], c11[2])
    );
    let (d11, p11) = ks(&dc, &steps_of(&g11.report));
    println!("  drop_step KS against CPU: D = {d11:.4}, p = {p11:.3}");
    let total = |c: [u64; 3]| c[0] + c[1] + c[2];
    println!(
        "  total lost: CPU {}, GPU {}, GPU x 1.1 {} (z of the total {:.2})",
        total(cc),
        total(cg),
        total(c11),
        z(total(cc), total(c11))
    );
}
