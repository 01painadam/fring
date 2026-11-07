//! K5-BIN benchmark: the cost of one bin pass (`GpuBinner`) next to the cost
//! of one Wisdom-Holman block, for N particles (default 10^5, 10^6, 10^7) in
//! the band of `g3_bench`, with the cell size scaled so that the mean
//! occupancy of an occupied cell is about 5.
//!
//! Each timing records several passes in one command buffer (so the cost of a
//! submit is shared) and is repeated `--reps` times (default 7; the GPU is shared with
//! other runs), after a warm-up, and the minimum and the median are printed.
//!
//! `cargo run -p fring-gpu --release --example k5_bin_bench [-- --n 100000,1000000 --reps 7]`

use fring_gpu::bin::{BinGeometry, GpuBinner, Stage};
use fring_gpu::gpu::Gpu;
use fring_gpu::probe::self_test;
use fring_gpu::stepper::{GpuStepper, MAX_BLOCK_STEPS};
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_model::config::Distribution;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    SaturnField, build_ephemeris, generate_particles, reference_mean_motion,
};
use fring_model::{InitialState, RunConfig, System};
use std::f64::consts::TAU;
use std::time::Duration;

const H: f64 = 356.2537;
const TARGET_OCCUPANCY: f64 = 5.0;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// (minimum, median) in ms of `reps` runs of `f`, after one warm-up.
fn time(reps: usize, mut f: impl FnMut() -> Duration) -> (f64, f64) {
    let _ = f();
    let mut v: Vec<f64> = (0..reps).map(|_| ms(f())).collect();
    v.sort_by(f64::total_cmp);
    (v[0], v[reps / 2])
}

/// Wall time per pass of `k` passes of the first `stages` stages, recorded in
/// one command buffer (so the cost of a submit is shared).
fn run_k(b: &GpuBinner, gpu: &Gpu, t: f64, stages: usize, k: usize) -> Duration {
    let d = gpu.device();
    let mut enc = d.create_command_encoder(&Default::default());
    for _ in 0..k {
        b.encode_stages(&mut enc, t, stages);
    }
    let cmd = enc.finish();
    let t0 = std::time::Instant::now();
    gpu.queue().submit([cmd]);
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    t0.elapsed() / k as u32
}

/// Mean occupancy of the occupied cells after one pass: binned / runs.
fn occupancy(b: &GpuBinner, n: usize, t: f64) -> (f64, usize) {
    let _ = b.run_timed(t, Stage::ALL.len());
    let info = b.read_words(b.info_buffer(), b.info_offset_words(), 3);
    let keys = b.read_words(b.sorted_keys_buffer(), 0, 2 * n);
    let binned = info[0] as usize;
    let mut runs = 0usize;
    let mut last = None;
    for p in 0..binned {
        let k = (keys[2 * p], keys[2 * p + 1]);
        if last != Some(k) {
            runs += 1;
            last = Some(k);
        }
    }
    (binned as f64 / runs.max(1) as f64, binned)
}

fn main() {
    let sizes: Vec<usize> = arg("--n")
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![100_000, 1_000_000, 10_000_000]);
    let reps: usize = arg("--reps").and_then(|s| s.parse().ok()).unwrap_or(7);
    let gpu = Gpu::new().expect("no GPU");
    self_test(&gpu).expect("DS self-test");
    println!("{} ({:?}), reps {reps}", gpu.info.name, gpu.info.backend);
    let text = include_str!("../../fring-model/tests/data/example-run.toml")
        .replace("step_s = 360.0", &format!("step_s = {H}"))
        .replace("duration_years = 1.0", "duration_years = 0.0335");
    let cfg = RunConfig::from_toml_str(&text).unwrap();
    let saturn = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let all = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let max_steps = 6 * MAX_BLOCK_STEPS;
    let eph = build_ephemeris(&saturn, &all, max_steps as f64 * H + 7200.0, 1800.0).unwrap();
    let system = System { saturn, moons: eph };
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let a0 = cfg.frame.reference_semi_major_axis_km;
    let n0 = reference_mean_motion(&field, a0);
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).unwrap();
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_600.0,
        max_semi_major_axis_km: 140_600.0,
    };
    println!(
        "| N | M | cells (x,y,z) m | occupancy | clear | keys | scan | scatter | rank | emit | bin pass min / median ms | WH step ms | WH block ms (steps) | bin / WH step | bin / WH block |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for n in sizes {
        let init = InitialState {
            time_s: 0.0,
            particles: generate_particles(&field, &dist, n as u32, 1).unwrap(),
        };
        let block = (200_000_000 / n.max(1)).clamp(8, MAX_BLOCK_STEPS);
        let mut stepper =
            GpuStepper::new(&gpu, &kernels, &system, &init, H, block, Mode::Dp, None).unwrap();
        // One WH block.
        let d = gpu.device();
        let mut run_block = || {
            let mut enc = d.create_command_encoder(&Default::default());
            stepper.encode_steps(&mut enc, block as u64).unwrap();
            let cmd = enc.finish();
            let t0 = std::time::Instant::now();
            gpu.queue().submit([cmd]);
            d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            t0.elapsed()
        };
        let wh = time(reps.min(5), &mut run_block);
        let t_s = stepper.time_s();
        // Scale the cells to the target occupancy (occupancy grows about as s^3).
        let base = [0.05, 10.0, 0.04];
        let mut s = 1.0f64;
        let mut best: Option<(GpuBinner, f64, f64)> = None;
        let mut last: Option<(f64, f64)> = None; // (ln s, ln occupancy)
        for _ in 0..12 {
            let g = BinGeometry::new(
                a0,
                n0,
                [base[0] * s, base[1] * s, base[2] * s],
                TAU * a0,
                0.0,
            )
            .unwrap();
            let b = GpuBinner::for_stepper(&gpu, &stepper, g).unwrap();
            let (occ, _) = occupancy(&b, n, t_s);
            let err = (occ / TARGET_OCCUPANCY).ln().abs();
            if best
                .as_ref()
                .is_none_or(|(_, _, o)| err < (o / TARGET_OCCUPANCY).ln().abs())
            {
                best = Some((b, s, occ));
            }
            if err < 0.1 {
                break;
            }
            // Secant in log-log; the first slope guess is 2 (a thin band).
            let now = (s.ln(), occ.ln());
            let slope = match last {
                Some((ls, lo)) if (now.0 - ls).abs() > 1e-9 && (now.1 - lo).abs() > 1e-9 => {
                    ((now.1 - lo) / (now.0 - ls)).clamp(0.3, 3.0)
                }
                _ => 2.0,
            };
            last = Some(now);
            s *= (TARGET_OCCUPANCY / occ).powf(1.0 / slope);
        }
        let (b, s, occ) = best.unwrap();
        // Prefix timings: the cost of each stage is the difference.
        let stages = Stage::ALL.len();
        let passes = (2_000_000 / n).clamp(2, 20);
        let mut prefix = Vec::new();
        for k in 1..=stages {
            prefix.push(time(reps, || run_k(&b, &gpu, t_s, k, passes)).1);
        }
        let full = time(reps, || run_k(&b, &gpu, t_s, stages, passes));
        let st = |a: usize, b2: usize| {
            (prefix[b2 - 1] - if a == 0 { 0.0 } else { prefix[a - 1] }).max(0.0)
        };
        let cells = b.geometry();
        println!(
            "| {n} | {} | {:.1}, {:.0}, {:.1} | {occ:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} / {:.2} | {:.3} | {:.1} ({block}) | {:.3} | {:.4} |",
            b.buckets(),
            cells.lx_km * 1e3,
            cells.ly_km * 1e3,
            cells.lz_km * 1e3,
            st(0, 1),
            st(1, 2),
            st(2, 5),
            st(5, 6),
            st(6, 7),
            st(7, 8),
            full.0,
            full.1,
            wh.0 / block as f64,
            wh.0,
            full.0 / (wh.0 / block as f64),
            full.0 / wh.0,
        );
        let _ = s;
    }
}
