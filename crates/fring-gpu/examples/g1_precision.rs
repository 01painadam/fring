//! G1 precision matrix (ADR 0006): every option against the f64 CPU code.
//!
//! ```sh
//! cargo run -p fring-gpu --release --example g1_precision -- --orbits 10000 --moons none
//! cargo run -p fring-gpu --release --example g1_precision -- --orbits 10000 --moons both
//! ```
//!
//! Options: `--orbits N` (default 10000; one orbit = 150 steps of
//! h = 356.25 s), `--moons none|both`, `--no-gpu`, `--out DIR` (default
//! `docs/adr/0006`). Writes `DIR/g1_<moons>_series.csv` (log-spaced in orbits,
//! particles regular up to 10³ orbits), `DIR/g1_<moons>_summary.csv` (10³
//! and 10⁴ orbits, regular and all particles, growth exponents from 100
//! orbits on) and `DIR/g1_<moons>_particles.csv` (every particle).
//!
//! Ensemble: three orbit types at a = 140,222.43 km (circular; core
//! e = 0.0024, i = 1.1e-4; stress e = 0.01, i = 1e-3), 32 particles each at
//! different longitudes and epicycle phases. Series per particle:
//! - `cpu_own`: the f64 CPU at h against the f64 CPU at h/10 (the CPU's own
//!   error against the true orbit; ADR 0005);
//! - `rust_m4`: the Rust twins with the M4 division model;
//! - `rust_wgsl`: the Rust twins with the WGSL worst-case model (C, D);
//! - `gpu`: the WGSL kernels on the GPU.
//!
//! Chaos test: an f64 shadow starts 1 mm further out. A particle is regular
//! up to a horizon if the shadow's |Δa| stays below 10 times its start value
//! and its phase difference below 10 times the linear Kepler shear. The
//! `regular` rows follow PLAN section 10 (trajectories of non-chaotic orbits
//! only); the `all` rows and `median_dphase_over_shadow` (the GPU-CPU phase
//! difference divided by the 1 mm shadow's) are the statistics for the rest.

use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{
    Consts, Opt, State, Tables, advance, phase_difference, semi_major_axis, state_from_f64,
    state_to_f64,
};
use fring_gpu::spike_gpu::GpuSpike;
use fring_gpu::wf32::model;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    EpicycleElements, KickTable, SaturnField, build_ephemeris, epicycle_state, wh_steps_moons,
};
use fring_model::{MoonEphemeris, Vec3};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::time::Instant;

const M4: u8 = model::M4;
const WG: u8 = model::WGSL;
const H_S: f64 = 356.25;
const STEPS_PER_ORBIT: usize = 150;
const BLOCK: usize = 1500;
const FINE: usize = 10;
const N_PER_TYPE: usize = 32;
const A0_KM: f64 = 140_222.43;
const ORBIT_TYPES: [(&str, f64, f64); 3] = [
    ("circular", 0.0, 0.0),
    ("core", 0.0024, 1.1e-4),
    ("stress", 0.01, 1.0e-3),
];
const WGSL_OPTS: [Opt; 2] = [Opt::D, Opt::C];

/// A series: where the numbers come from and which option.
#[derive(Clone, Debug)]
struct Series {
    source: &'static str,
    opt: &'static str,
}

struct Particle {
    kind: usize,
    p: Vec3<f64>,
    v: Vec3<f64>,
    shadow: (Vec3<f64>, Vec3<f64>),
    fine: (Vec3<f64>, Vec3<f64>),
    m4: Vec<State<M4>>,
    wg: Vec<State<WG>>,
    gpu: Vec<State<M4>>,
    /// Per block: (Δa km, Δλ rad) for every series (index 1: the shadow).
    rec: Vec<Vec<(f64, f64)>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let orbits: usize = arg("--orbits").map_or(10_000, |s| s.parse().unwrap());
    let moons_mode = arg("--moons").unwrap_or_else(|| "none".into());
    let out_dir = arg("--out").unwrap_or_else(|| "docs/adr/0006".into());
    let use_gpu = !args.iter().any(|a| a == "--no-gpu");
    let n_steps = orbits * STEPS_PER_ORBIT;
    let n_blocks = n_steps.div_ceil(BLOCK);

    let sp = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let field = SaturnField::<f64>::from_params(&sp);
    let eph = if moons_mode == "both" {
        let moons = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
        let t0 = Instant::now();
        let e = build_ephemeris(&sp, &moons, n_steps as f64 * H_S + 7200.0, 1800.0).unwrap();
        eprintln!("ephemeris: {:.1} s", t0.elapsed().as_secs_f64());
        e
    } else {
        MoonEphemeris::none()
    };
    let c4 = Consts::<M4>::new(&sp, &eph, H_S);
    let cw = Consts::<WG>::new(&sp, &eph, H_S);

    let gpu = if use_gpu { Gpu::new() } else { None };
    let spike = gpu.as_ref().map(|g| match GpuSpike::new(g) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    });

    let mut series = vec![
        Series {
            source: "cpu_own",
            opt: "f64",
        },
        Series {
            source: "shadow",
            opt: "f64",
        },
    ];
    for o in Opt::ALL {
        series.push(Series {
            source: "rust_m4",
            opt: o.name(),
        });
    }
    for o in WGSL_OPTS {
        series.push(Series {
            source: "rust_wgsl",
            opt: o.name(),
        });
    }
    if spike.is_some() {
        for o in Opt::ALL {
            series.push(Series {
                source: "gpu",
                opt: o.name(),
            });
        }
    }

    // Initial states.
    let mut parts: Vec<Particle> = Vec::new();
    for (kind, &(_, e, inc)) in ORBIT_TYPES.iter().enumerate() {
        for k in 0..N_PER_TYPE {
            let el = EpicycleElements {
                a_km: A0_KM,
                e,
                i_rad: inc,
                mean_anomaly_rad: 2.399_963 * k as f64,
                z_phase_rad: 1.3 * k as f64,
                mean_longitude_rad: 0.1 + std::f64::consts::TAU * k as f64 / N_PER_TYPE as f64,
            };
            let (p, v) = epicycle_state(&field, &el);
            let ps = p * (1.0 + 1e-6 / p.norm());
            parts.push(Particle {
                kind,
                p,
                v,
                shadow: (ps, v),
                fine: (p, v),
                m4: vec![state_from_f64(p, v); Opt::ALL.len()],
                wg: vec![state_from_f64(p, v); WGSL_OPTS.len()],
                gpu: vec![state_from_f64(p, v); Opt::ALL.len()],
                rec: Vec::with_capacity(n_blocks),
            });
        }
    }

    let mut t64 = KickTable::<f64>::for_ephemeris(&eph);
    let mut tfine = KickTable::<f64>::for_ephemeris(&eph);
    let mut tab4 = Tables::<M4>::new(&eph);
    let mut tabw = Tables::<WG>::new(&eph);
    let t_start = Instant::now();
    let mut gpu_time = 0.0;
    for b in 0..n_blocks {
        let first = b * BLOCK;
        let n = BLOCK.min(n_steps - first);
        t64.fill(&eph, 0.0, H_S, first as u64, n).unwrap();
        tfine
            .fill(
                &eph,
                0.0,
                H_S / FINE as f64,
                (first * FINE) as u64,
                n * FINE,
            )
            .unwrap();
        tab4.fill(&eph, 0.0, H_S, first as u64, n).unwrap();
        tabw.fill(&eph, 0.0, H_S, first as u64, n).unwrap();

        parts.par_iter_mut().for_each(|q| {
            wh_steps_moons(&field, &t64, &mut q.p, &mut q.v, H_S).unwrap();
            wh_steps_moons(&field, &t64, &mut q.shadow.0, &mut q.shadow.1, H_S).unwrap();
            wh_steps_moons(
                &field,
                &tfine,
                &mut q.fine.0,
                &mut q.fine.1,
                H_S / FINE as f64,
            )
            .unwrap();
            for (s, o) in q.m4.iter_mut().zip(Opt::ALL) {
                advance(o, &c4, &tab4, s).unwrap();
            }
            for (s, o) in q.wg.iter_mut().zip(WGSL_OPTS) {
                advance(o, &cw, &tabw, s).unwrap();
            }
        });
        if let Some(sp_gpu) = &spike {
            for (k, o) in Opt::ALL.into_iter().enumerate() {
                let mut st: Vec<State<M4>> = parts.iter().map(|q| q.gpu[k]).collect();
                gpu_time += sp_gpu.advance(o, &c4, &tab4, &mut st).as_secs_f64();
                for (q, s) in parts.iter_mut().zip(st) {
                    q.gpu[k] = s;
                }
            }
        }
        let gm = sp.gm_km3_s2;
        for q in &mut parts {
            let ar = semi_major_axis(gm, q.p, q.v);
            let d = |p: Vec3<f64>, v: Vec3<f64>| {
                (semi_major_axis(gm, p, v) - ar, phase_difference(q.p, p))
            };
            let mut row = Vec::with_capacity(series.len());
            // CPU's own error: the step-h result minus the h/10 result.
            let af = semi_major_axis(gm, q.fine.0, q.fine.1);
            row.push((ar - af, phase_difference(q.fine.0, q.p)));
            let sh = d(q.shadow.0, q.shadow.1);
            row.push(sh);
            for s in &q.m4 {
                let (p, v) = state_to_f64(s);
                row.push(d(p, v));
            }
            for s in &q.wg {
                let (p, v) = state_to_f64(s);
                row.push(d(p, v));
            }
            if spike.is_some() {
                for s in &q.gpu {
                    let (p, v) = state_to_f64(s);
                    row.push(d(p, v));
                }
            }
            q.rec.push(row);
        }
        if b % 50 == 49 || b + 1 == n_blocks {
            eprintln!(
                "{} of {} orbits, {:.0} s (GPU {:.1} s)",
                (first + n) / STEPS_PER_ORBIT,
                orbits,
                t_start.elapsed().as_secs_f64(),
                gpu_time
            );
        }
    }

    // Chaos test (PLAN 10: compare trajectories only for non-chaotic orbits).
    // The f64 shadow starts 1 mm further out. On a regular orbit its Δa stays
    // near the start value and its phase difference grows linearly (Kepler
    // shear): Δλ_lin = 1.5 (Δa₀/a) 2π N. A particle is regular up to block
    // b if max |Δa_sh| < 10 |Δa_sh(0)| and |Δλ_sh| < 10 Δλ_lin up to b.
    let orbits_at = |b: usize| ((b + 1) * BLOCK / STEPS_PER_ORBIT) as f64;
    let amp = |q: usize, b: usize| {
        let r = &parts[q].rec;
        let da0 = r[0][1].0.abs().max(1e-12);
        let amp_a = (0..=b).map(|j| r[j][1].0.abs()).fold(0.0, f64::max) / da0;
        let amp_l = (0..=b)
            .map(|j| r[j][1].1.abs() / (1.5 * da0 / A0_KM * std::f64::consts::TAU * orbits_at(j)))
            .fold(0.0, f64::max);
        (amp_a, amp_l)
    };
    let regular = |q: usize, b: usize| {
        let (aa, al) = amp(q, b);
        aa < 10.0 && al < 10.0
    };
    let horizon_block = |o: usize| (o * STEPS_PER_ORBIT / BLOCK).min(n_blocks) - 1;
    let b1000 = horizon_block(orbits.min(1000));
    let mut csv = String::from(
        "source,option,orbit_type,moons,orbits,n_particles,rms_da_km,max_da_km,rms_dphase_rad,max_dphase_rad\n",
    );
    let mut summary = String::from(
        "source,option,orbit_type,moons,orbits,filter,n_particles,rms_da_km,max_da_km,rms_dphase_rad,max_dphase_rad,median_dphase_over_shadow,exp_da,exp_dphase\n",
    );
    let mut per_particle = String::from(
        "moons,orbit_type,particle,orbits,regular,shadow_amp_da,shadow_amp_dphase,source,option,da_km,dphase_rad\n",
    );
    for (si, s) in series.iter().enumerate() {
        for (kind, &(tname, _, _)) in ORBIT_TYPES.iter().enumerate() {
            let members: Vec<usize> = (0..parts.len())
                .filter(|&q| parts[q].kind == kind)
                .collect();
            // Statistics at block b over the particles in `set`.
            let stats = |b: usize, set: &[usize]| {
                let (mut sa, mut ma, mut sl, mut ml) = (0.0, 0.0f64, 0.0, 0.0f64);
                let mut ratios: Vec<f64> = Vec::new();
                for &q in set {
                    let (da, dl) = parts[q].rec[b][si];
                    sa += da * da;
                    sl += dl * dl;
                    ma = ma.max(da.abs());
                    ml = ml.max(dl.abs());
                    ratios.push(dl.abs() / parts[q].rec[b][1].1.abs().max(1e-300));
                }
                ratios.sort_by(f64::total_cmp);
                let med = ratios.get(ratios.len() / 2).copied().unwrap_or(f64::NAN);
                let n = set.len().max(1) as f64;
                (set.len(), (sa / n).sqrt(), ma, (sl / n).sqrt(), ml, med)
            };
            // Time series and growth exponents over the particles that are
            // regular up to 10³ orbits (all of them without moons).
            let reg1000: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&q| regular(q, b1000))
                .collect();
            // Log-spaced output: every 10 orbits to 100, 50 to 1000, 500 after.
            let keep = |o: f64| {
                let o = o as usize;
                (o <= 100 && o.is_multiple_of(10))
                    || (o <= 1000 && o.is_multiple_of(50))
                    || o.is_multiple_of(500)
            };
            for b in (0..n_blocks).filter(|&b| keep(orbits_at(b))) {
                let (n, ra, ma, rl, ml, _) = stats(b, &reg1000);
                writeln!(
                    csv,
                    "{},{},{tname},{moons_mode},{},{n},{ra:.4e},{ma:.4e},{rl:.4e},{ml:.4e}",
                    s.source,
                    s.opt,
                    orbits_at(b)
                )
                .unwrap();
            }
            let fit = |col: usize, set: &[usize], until: usize| {
                let pts: Vec<(f64, f64)> = (0..=until)
                    .filter_map(|b| {
                        let (_, ra, _, rl, _, _) = stats(b, set);
                        let y = if col == 0 { ra } else { rl };
                        (orbits_at(b) >= 100.0 && y > 0.0).then(|| (orbits_at(b).ln(), y.ln()))
                    })
                    .collect();
                if pts.len() < 3 {
                    return f64::NAN;
                }
                let n = pts.len() as f64;
                let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
                let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
                let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
                let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
                sxy / sxx
            };
            for horizon in [1000usize, 10_000] {
                if horizon > orbits {
                    continue;
                }
                let b = horizon_block(horizon);
                let reg: Vec<usize> = members.iter().copied().filter(|&q| regular(q, b)).collect();
                for (filter, set) in [("regular", &reg), ("all", &members)] {
                    let (n, ra, ma, rl, ml, med) = stats(b, set);
                    let (ea, el) = (fit(0, set, b), fit(1, set, b));
                    writeln!(
                        summary,
                        "{},{},{tname},{moons_mode},{horizon},{filter},{n},{ra:.4e},{ma:.4e},{rl:.4e},{ml:.4e},{med:.3e},{ea:.2},{el:.2}",
                        s.source, s.opt
                    )
                    .unwrap();
                }
                for &q in &members {
                    let (aa, al) = amp(q, b);
                    let (da, dl) = parts[q].rec[b][si];
                    writeln!(
                        per_particle,
                        "{moons_mode},{tname},{q},{horizon},{},{aa:.3e},{al:.3e},{},{},{da:.4e},{dl:.4e}",
                        regular(q, b),
                        s.source,
                        s.opt
                    )
                    .unwrap();
                }
            }
        }
    }
    std::fs::create_dir_all(&out_dir).unwrap();
    let base = format!("{out_dir}/g1_{moons_mode}");
    std::fs::write(format!("{base}_series.csv"), csv).unwrap();
    std::fs::write(format!("{base}_summary.csv"), &summary).unwrap();
    std::fs::write(format!("{base}_particles.csv"), per_particle).unwrap();
    println!("{summary}");
    eprintln!(
        "done: {:.0} s wall, GPU {:.1} s; wrote {base}_series.csv and {base}_summary.csv",
        t_start.elapsed().as_secs_f64(),
        gpu_time
    );
}
