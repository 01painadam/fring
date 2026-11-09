//! Task G3, done when: the GPU chaos indicator matches the CPU indicator for
//! 1,000 particles (ADR 0008). Chaotic orbits are compared by statistics, and
//! every limit is tied to a baseline: a CPU run against a second CPU run in
//! which every particle starts 1e-4 km away (the size of the GPU-CPU
//! deviation; A0 and AJ, 2026-11-02, ADR 0008).
//!
//! Particles: circular, uniform in a over 139,600 to 140,600 km and in
//! longitude, both moons, h = 356.2537 s, 150,000 steps (about 1000 orbits),
//! shadow offset d0 = 1 km, renormalisation every K = 100 steps. Classes
//! from the mean MEGNO: regular <= 2.5, chaotic >= 5, else uncertain.
//!
//! Environment (for the sweep of ADR 0008): FRING_G3_D0 (km), FRING_G3_K,
//! FRING_G3_N (particles). Needs a GPU, about 1 minute: `#[ignore]`:
//! `cargo test -p fring-gpu --release --test chaos_done_when -- --ignored --nocapture`.

mod common;

use common::*;
use fring_cpu::CpuBackend;
use fring_gpu::backend::{GpuBackend, Options};
use fring_gpu::wh::Mode;
use fring_model::config::{ChaosConfig, ChaosMethod, DiagnosticsConfig, Distribution};
use fring_model::physics::{SaturnField, generate_particles};
use fring_model::rng::{Counter, Philox, Stream};
use fring_model::{Backend, ChaosReport, InitialState, RunConfig, System};
use std::time::Instant;

const STEPS: u64 = 150_000;
const REGULAR_MAX: f64 = 2.5;
const CHAOTIC_MIN: f64 = 5.0;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Regular,
    Uncertain,
    Chaotic,
}

fn class(y: f64) -> Class {
    if y <= REGULAR_MAX {
        Class::Regular
    } else if y >= CHAOTIC_MIN {
        Class::Chaotic
    } else {
        Class::Uncertain
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Every particle moved by `size_km` in a fixed direction per particle, from
/// the TEST stream (not the shadow stream).
fn displaced(init: &InitialState, size_km: f64) -> InitialState {
    let rng = Philox::from_seed(99);
    let mut out = InitialState {
        time_s: init.time_s,
        particles: init.particles.clone(),
    };
    for i in 0..init.particles.len() {
        let [u, w] = rng.uniform_f64(Counter {
            index: i as u32,
            step: 0,
            stream: Stream::TEST,
            draw: 0,
        });
        let z = 2.0 * u - 1.0;
        let r = (1.0 - z * z).sqrt();
        let phi = std::f64::consts::TAU * w;
        out.particles.x_km[i] += size_km * r * phi.cos();
        out.particles.y_km[i] += size_km * r * phi.sin();
        out.particles.z_km[i] += size_km * z;
    }
    out
}

/// ICR-011: both arms drift about mu_eff (CPU `new`, GPU `GpuBackend`).
fn run(config: &RunConfig, system: &System, init: &InitialState) -> (ChaosReport, f64) {
    let t0 = Instant::now();
    let mut b = CpuBackend::new(config, system, init).unwrap();
    b.step(STEPS).unwrap();
    let r = b.chaos().unwrap().expect("chaos table is set");
    (r, t0.elapsed().as_secs_f64())
}

struct Stats {
    agree: f64,
    n_compared: usize,
    reg_dl: Vec<f64>,
    reg_dy: Vec<f64>,
    q_rel: [f64; 3],
    n_chaotic_both: usize,
}

/// Compares report `b` with the reference `a`.
fn compare(a: &ChaosReport, b: &ChaosReport) -> Stats {
    let n = a.megno_mean.len();
    let (mut same, mut compared) = (0usize, 0usize);
    let mut reg_dl = Vec::new();
    let mut reg_dy = Vec::new();
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    for i in 0..n {
        let (ca, cb) = (class(a.megno_mean[i]), class(b.megno_mean[i]));
        if ca == Class::Uncertain || cb == Class::Uncertain {
            continue;
        }
        compared += 1;
        if ca == cb {
            same += 1;
        }
        if ca == Class::Regular && cb == Class::Regular {
            reg_dl.push((a.log_growth[i] - b.log_growth[i]).abs());
            reg_dy.push((a.megno_mean[i] - b.megno_mean[i]).abs());
        }
        if ca == Class::Chaotic && cb == Class::Chaotic {
            fa.push(a.ftle_per_s[i]);
            fb.push(b.ftle_per_s[i]);
        }
    }
    fa.sort_by(f64::total_cmp);
    fb.sort_by(f64::total_cmp);
    let mut q_rel = [f64::NAN; 3];
    if !fa.is_empty() {
        for (k, q) in [0.1, 0.5, 0.9].into_iter().enumerate() {
            let (x, y) = (quantile(&fa, q), quantile(&fb, q));
            q_rel[k] = (y - x).abs() / x.abs();
        }
    }
    reg_dl.sort_by(f64::total_cmp);
    reg_dy.sort_by(f64::total_cmp);
    Stats {
        reg_dy,
        agree: same as f64 / compared.max(1) as f64,
        n_compared: compared,
        reg_dl,
        q_rel,
        n_chaotic_both: fa.len(),
    }
}

fn print_table(name: &str, s: &Stats) {
    let dy = &s.reg_dy;
    println!(
        "{name:12} regular-both |d<Y>| median {:.2e} p95 {:.2e} max {:.2e}",
        quantile(dy, 0.5),
        quantile(dy, 0.95),
        dy.last().copied().unwrap_or(f64::NAN)
    );
    let dl = &s.reg_dl;
    println!(
        "{name:12} compared {:4}  same class {:6.2} %  regular-both {:4}: |dL| median {:.2e} p95 {:.2e} max {:.2e}  \
         chaotic-both {:4}: FTLE quantile diff 10/50/90 % = {:.3} {:.3} {:.3}",
        s.n_compared,
        100.0 * s.agree,
        dl.len(),
        dl.get(dl.len() / 2).copied().unwrap_or(f64::NAN),
        dl.get(((dl.len().max(1) - 1) as f64 * 0.95).round() as usize)
            .copied()
            .unwrap_or(f64::NAN),
        dl.last().copied().unwrap_or(f64::NAN),
        s.n_chaotic_both,
        s.q_rel[0],
        s.q_rel[1],
        s.q_rel[2],
    );
}

/// |dL| of the regular-on-both particles by the MEGNO of the reference (diagnostic).
fn dl_by_megno(name: &str, a: &ChaosReport, b: &ChaosReport) {
    for (lo, hi) in [(0.0, 2.05), (2.05, 2.2), (2.2, 2.5)] {
        let mut v: Vec<f64> = (0..a.megno_mean.len())
            .filter(|&i| {
                let y = a.megno_mean[i];
                y > lo && y <= hi && class(b.megno_mean[i]) == Class::Regular
            })
            .map(|i| (a.log_growth[i] - b.log_growth[i]).abs())
            .collect();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            continue;
        }
        println!(
            "{name:12} <Y> in ({lo}, {hi}]: {:4} particles, |dL| median {:.2e} p95 {:.2e} max {:.2e}",
            v.len(),
            quantile(&v, 0.5),
            quantile(&v, 0.95),
            v.last().unwrap()
        );
    }
}

fn summary(name: &str, r: &ChaosReport) {
    let mut y = r.megno_mean.clone();
    y.sort_by(f64::total_cmp);
    let mut f = r.ftle_per_s.clone();
    f.sort_by(f64::total_cmp);
    let count = |c: Class| r.megno_mean.iter().filter(|&&v| class(v) == c).count();
    println!(
        "{name:12} regular {:4} uncertain {:4} chaotic {:4}   <Y> quantiles 10/50/90 %: {:.3} {:.3} {:.3}   FTLE per day 50/90 %: {:.3e} {:.3e}",
        count(Class::Regular),
        count(Class::Uncertain),
        count(Class::Chaotic),
        quantile(&y, 0.1),
        quantile(&y, 0.5),
        quantile(&y, 0.9),
        quantile(&f, 0.5) * 86_400.0,
        quantile(&f, 0.9) * 86_400.0,
    );
}

/// The rules of the A0 decision. Returns the failed ones, so that one run
/// reports every mode.
fn check(
    name: &str,
    cpu: &ChaosReport,
    base_report: &ChaosReport,
    base: &Stats,
    gpu: &ChaosReport,
) -> (Vec<String>, [f64; 3]) {
    let s = compare(cpu, gpu);
    print_table(name, &s);
    dl_by_megno(name, cpu, gpu);
    summary(name, gpu);
    let mut failed = Vec::new();
    // 1. Same class.
    if s.agree < 0.97 {
        failed.push(format!("{name}: class agreement {:.4} < 0.97", s.agree));
    }
    if s.agree < base.agree - 0.01 {
        failed.push(format!(
            "{name}: agreement {:.4} is worse than the baseline {:.4} by more than 1 point",
            s.agree, base.agree
        ));
    }
    // 2. Regular on both: |dL|.
    let dl = &s.reg_dl;
    let p95 = quantile(dl, 0.95);
    if p95 > 0.1 {
        failed.push(format!("{name}: 95 % of |dL| is {p95:.3e}, limit 0.1"));
    }
    // Each particle regular on both: GPU |dL| <= max(0.3, 2 x the |dL| of the
    // same particle in the CPU baseline run) (ADR 0009: a single particle on
    // a weak orbit moves by 2.4 under a 1e-4 km offset).
    for i in 0..cpu.megno_mean.len() {
        if class(cpu.megno_mean[i]) == Class::Regular && class(gpu.megno_mean[i]) == Class::Regular
        {
            let g = (cpu.log_growth[i] - gpu.log_growth[i]).abs();
            let b = (cpu.log_growth[i] - base_report.log_growth[i]).abs();
            if g > f64::max(0.3, 2.0 * b) {
                failed.push(format!(
                    "{name}: particle {i}: |dL| {g:.3e} > max(0.3, 2 x baseline {b:.3e})"
                ));
            }
        }
    }
    // 3. Chaotic on both: FTLE quantiles.
    if s.n_chaotic_both < 20 {
        failed.push(format!("{name}: too few chaotic particles"));
    }
    for k in 0..3 {
        if s.q_rel[k] > 0.15 {
            failed.push(format!(
                "{name}: FTLE quantile {k} differs by {:.3} > 0.15",
                s.q_rel[k]
            ));
        }
    }
    (failed, s.q_rel)
}

/// The five regular-on-both particles with the largest |dL| (CPU against GPU),
/// with the class and MEGNO of each, the start radius and |dL| of the CPU
/// baseline run (diagnostic, WH-EFF-GPU).
fn top_dl(name: &str, init: &InitialState, cpu: &ChaosReport, base: &ChaosReport, g: &ChaosReport) {
    let mut v: Vec<(f64, usize)> = (0..cpu.megno_mean.len())
        .filter(|&i| {
            class(cpu.megno_mean[i]) == Class::Regular && class(g.megno_mean[i]) == Class::Regular
        })
        .map(|i| ((cpu.log_growth[i] - g.log_growth[i]).abs(), i))
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (d, i) in v.into_iter().take(5) {
        let r = init.particles.position_km(i).norm();
        println!(
            "{name:8} top |dL| {d:.3e}: particle {i}, start r {r:.1} km, <Y> cpu {:.3} gpu {:.3}, cpu L {:.2}, |dL| of the CPU baseline {:.3e}",
            cpu.megno_mean[i],
            g.megno_mean[i],
            cpu.log_growth[i],
            (cpu.log_growth[i] - base.log_growth[i]).abs()
        );
    }
}

#[test]
#[ignore = "needs a GPU, about 2 minutes"]
fn gpu_chaos_indicator_matches_cpu_for_1000_particles() {
    let n = env_f64("FRING_G3_N", 1000.0) as u32;
    let d0 = env_f64("FRING_G3_D0", 1.0);
    let k = env_f64("FRING_G3_K", 100.0) as u32;
    let system = system(H_S, STEPS + k as u64, true);
    let mut config = config(H_S, STEPS + k as u64, true);
    config.diagnostics = Some(DiagnosticsConfig {
        chaos: Some(ChaosConfig {
            method: ChaosMethod::Shadow,
            offset_km: d0,
            renorm_steps: k,
        }),
    });
    config.validate().unwrap();
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_600.0,
        max_semi_major_axis_km: 140_600.0,
    };
    // ADR 0009: four independent samples. FRING_G3_SEED runs one (then the
    // median rules are the single-sample rules).
    let seeds: Vec<u64> = match std::env::var("FRING_G3_SEED") {
        Ok(v) => vec![v.parse().expect("FRING_G3_SEED")],
        Err(_) => vec![20_261_102, 101, 202, 303],
    };
    println!("N = {n}, d0 = {d0} km, K = {k}, steps = {STEPS}, h = {H_S} s, seeds {seeds:?}");

    let mut failed = Vec::new();
    // FTLE quantile differences by seed: the baseline and the GPU modes.
    let mut q_base: Vec<[f64; 3]> = Vec::new();
    let mut q_gpu: [Vec<[f64; 3]>; 2] = [Vec::new(), Vec::new()];
    for &seed in &seeds {
        println!("== seed {seed}");
        let init = InitialState {
            time_s: 0.0,
            particles: generate_particles(&field, &dist, n, seed).unwrap(),
        };
        let (cpu, t_cpu) = run(&config, &system, &init);
        let (base, _) = run(
            &config,
            &system,
            &displaced(&init, env_f64("FRING_G3_BASE_KM", 1e-4)),
        );
        println!("CPU run {t_cpu:.1} s, n_renorm {}", cpu.n_renorm);
        summary("cpu", &cpu);
        summary("cpu baseline", &base);
        let bs = compare(&cpu, &base);
        print_table("baseline", &bs);
        dl_by_megno("baseline", &cpu, &base);

        // Classes of the reference must not be empty (M17).
        let count = |c: Class| cpu.megno_mean.iter().filter(|&&v| class(v) == c).count();
        assert!(count(Class::Regular) >= 100, "too few regular particles");
        assert!(count(Class::Chaotic) >= 100, "too few chaotic particles");
        assert!(bs.agree >= 0.97, "baseline agreement {:.4}", bs.agree);
        q_base.push(bs.q_rel);

        for (name, mode, asserted) in [
            ("gpu D+", Mode::Dp, Some(0)),
            ("gpu C", Mode::C, Some(1)),
            ("gpu D", Mode::D, None),
        ] {
            let opts = Options {
                mode,
                ..Options::default()
            };
            let t0 = Instant::now();
            let mut b = GpuBackend::with_options(&config, &system, &init, opts).unwrap();
            b.step(STEPS).unwrap();
            let g = b.chaos().unwrap().unwrap();
            println!("{name}: {:.1} s", t0.elapsed().as_secs_f64());
            assert_eq!(g.n_renorm, cpu.n_renorm);
            if let Some(slot) = asserted {
                top_dl(name, &init, &cpu, &base, &g);
                let (f, q) = check(name, &cpu, &base, &bs, &g);
                failed.extend(f.into_iter().map(|m| format!("seed {seed}: {m}")));
                q_gpu[slot].push(q);
            } else {
                // Mode D is reported only (ADR 0009, ADR 0006 addendum).
                let s = compare(&cpu, &g);
                print_table(name, &s);
                summary(name, &g);
            }
        }
    }

    // FTLE quantiles, chaotic on both: the MEDIAN over the seeds of the GPU
    // difference must be at most 2 x the MEDIAN over the seeds of the baseline
    // difference, for each of the 10, 50 and 90 % quantiles (ADR 0009).
    let median = |v: &[[f64; 3]], k: usize| {
        let mut x: Vec<f64> = v.iter().map(|q| q[k]).collect();
        x.sort_by(f64::total_cmp);
        let m = x.len();
        if m % 2 == 1 {
            x[m / 2]
        } else {
            0.5 * (x[m / 2 - 1] + x[m / 2])
        }
    };
    for (name, q) in [("gpu D+", &q_gpu[0]), ("gpu C", &q_gpu[1])] {
        for k in 0..3 {
            let (g, b) = (median(q, k), median(&q_base, k));
            println!(
                "{name}: FTLE quantile {k}: median GPU difference {g:.4}, median baseline {b:.4}, limit {:.4}",
                2.0 * b
            );
            if g > 2.0 * b {
                failed.push(format!(
                    "{name}: FTLE quantile {k}: median GPU difference {g:.4} > 2 x median baseline {b:.4}"
                ));
            }
        }
    }
    assert!(failed.is_empty(), "failed rules:\n{}", failed.join("\n"));
}
