//! K5-BIN: the GPU binning (`GpuBinner`) against its Rust twin and against
//! the standard-library sort. Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_bin -- --ignored --nocapture`.
//!
//! Rules (A0 decision):
//! - GPU against the DS twin (`bin::cell_ds`, `bin::bin_twin`): exact. The
//!   same key for each particle (when the particle is not within 1e-6 km of
//!   an edge) and the same sorted arrays.
//! - Twin or GPU against the f64 reference: a particle may get another cell
//!   only if its f64 coordinate is within `DELTA` = 1.01e-7 km of a cell edge on
//!   an axis where the cells differ (10 times the largest error measured in
//!   the unit test `twin_agrees_with_f64_up_to_the_edge_rule`, which is
//!   1.001e-8 km, rounded up; at most 1e-6 km).

mod common;

use fring_gpu::bin::{
    BinGeometry, BinOutput, Binned, Cell, Dsm, GpuBinner, NO_ID, NO_KEY, Stage, bin_twin, cell_ds,
    cell_f64, cells_reference, edge_distance_km, key_of, pack_key, params, runs, unpack_key,
};
use fring_gpu::gpu::Gpu;
use fring_gpu::probe::self_test;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, WhKernels};
use fring_gpu::wh_gpu::{pack_states, status_init_bytes};
use fring_model::physics::{SaturnField, reference_mean_motion};
use fring_model::rng::{philox4x32_10, u01_f64};
use std::f64::consts::TAU;

const A0: f64 = 140_222.43;
const N0: f64 = 1.1756e-4;
/// 10 x 1.001e-8 km (the largest error of the unit test), rounded up.
const DELTA: f64 = 1.01e-7;

fn uniform(seed: u32, stream: u32, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let w = philox4x32_10([i as u32, stream, 0, 0], [seed, 0x4b35]);
            u01_f64(w[0], w[1])
        })
        .collect()
}

fn inertial(g: &BinGeometry, alpha: f64, local: [f64; 3]) -> [f64; 3] {
    let rho = g.a0_km + local[0];
    let phi = local[1] / g.a0_km + alpha;
    [rho * phi.cos(), rho * phi.sin(), local[2]]
}

/// Buffers of a GPU state for the inertial positions `pos` (km), with the
/// drop reasons `reasons` (0 = live), and the binner.
struct Setup {
    gpu: Gpu,
    binner: GpuBinner,
    /// The DS positions as the GPU holds them.
    ds: Vec<[Dsm; 3]>,
    reasons: Vec<u32>,
}

fn setup(
    gpu: &Gpu,
    g: BinGeometry,
    pos: &[[f64; 3]],
    reasons: &[u32],
    buckets: Option<usize>,
) -> Setup {
    let n = pos.len();
    let cap = n.next_multiple_of(64).max(64);
    let states: Vec<[Dsm; 6]> = pos
        .iter()
        .map(|p| {
            [
                Dsm::split(p[0]),
                Dsm::split(p[1]),
                Dsm::split(p[2]),
                Dsm::split(0.0),
                Dsm::split(0.0),
                Dsm::split(0.0),
            ]
        })
        .collect();
    let words = pack_states(&states, cap);
    let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d = gpu.device();
    let mk = |size: u64| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let state = mk(bytes.len() as u64);
    gpu.queue().write_buffer(&state, 0, &bytes);
    let mut sb = status_init_bytes(cap);
    for (i, &r) in reasons.iter().enumerate() {
        sb[4 * (8 + i)..4 * (9 + i)].copy_from_slice(&r.to_le_bytes());
    }
    let status = mk(sb.len() as u64);
    gpu.queue().write_buffer(&status, 0, &sb);
    let binner = GpuBinner::new(gpu, &state, &status, n, cap, g, buckets).expect("binner");
    Setup {
        gpu: gpu.clone(),
        binner,
        ds: states.iter().map(|s| [s[0], s[1], s[2]]).collect(),
        reasons: reasons.to_vec(),
    }
}

impl Setup {
    /// Runs one bin pass at `time_s` and returns the GPU result and keys.
    fn bin(&self, time_s: f64) -> (BinOutput, Vec<[u32; 2]>) {
        let mut enc = self
            .gpu
            .device()
            .create_command_encoder(&Default::default());
        self.binner.encode(&mut enc, time_s);
        self.gpu.queue().submit([enc.finish()]);
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        (self.binner.read_output(), self.binner.read_particle_keys())
    }

    /// The twin's cell of each particle (a dropped particle is not binned).
    fn twin_cells(&self, time_s: f64) -> Vec<Binned> {
        let g = self.binner.geometry();
        let prm = params(g);
        let alpha = Dsm::split(g.alpha_rad(time_s));
        self.ds
            .iter()
            .zip(&self.reasons)
            .map(|(p, &r)| {
                if r != 0 {
                    Binned::NotBinned
                } else {
                    cell_ds(&prm, g.ny, alpha, *p).0
                }
            })
            .collect()
    }
}

fn gpu() -> Option<Gpu> {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return None;
    };
    self_test(&gpu).expect("DS self-test");
    Some(gpu)
}

/// Positions near a ring sector (clustered, many particles per cell) and
/// spread over the domain, as inertial km, with the f64 local coordinates.
fn particles(g: &BinGeometry, time_s: f64, n: usize, seed: u32) -> Vec<[f64; 3]> {
    let alpha = g.alpha_rad(time_s);
    let (a, b, c, k) = (
        uniform(seed, 1, n),
        uniform(seed, 2, n),
        uniform(seed, 3, n),
        uniform(seed, 4, n),
    );
    (0..n)
        .map(|i| {
            let local = if k[i] < 0.7 {
                // A clump: 8 cells in x, 40 km in y, 6 cells in z.
                [
                    8.0 * g.lx_km * a[i],
                    0.4 * g.domain_km * 0.0 + 40.0 * b[i] + 0.25 * g.domain_km,
                    6.0 * g.lz_km * c[i] - 3.0 * g.lz_km,
                ]
            } else {
                [
                    200.0 * a[i] - 100.0,
                    1.04 * g.domain_km * b[i],
                    2.0 * c[i] - 1.0,
                ]
            };
            inertial(g, alpha, local)
        })
        .collect()
}

/// Checks a GPU result against the twin (exact) and against the standard
/// library sort of the twin's cells; returns the number of particles whose
/// cell differs from f64.
fn check_against_twin(s: &Setup, time_s: f64, m: usize, pos: &[[f64; 3]]) -> usize {
    let (out, keys) = s.bin(time_s);
    let cells = s.twin_cells(time_s);
    let twin_keys: Vec<[u32; 2]> = cells.iter().map(|&c| key_of(c)).collect();
    // Keys: exact (the particles are not placed on edges here).
    let bad_keys: Vec<usize> = (0..keys.len())
        .filter(|&i| keys[i] != twin_keys[i])
        .collect();
    assert!(
        bad_keys.is_empty(),
        "{} keys differ from the twin, first {:?}",
        bad_keys.len(),
        bad_keys.first().map(|&i| (i, keys[i], twin_keys[i]))
    );
    // Sorted arrays: exact, because the order inside a bucket is (key, id).
    let want = bin_twin(&cells, m);
    assert_eq!(out.n_binned, want.n_binned);
    assert_eq!(out.n_not_binned, want.n_not_binned);
    assert_eq!(out.n_out_of_range, want.n_out_of_range);
    assert_eq!(out.bucket_start, want.bucket_start, "bucket starts");
    assert_eq!(out.sorted, want.sorted, "sorted ids");
    assert_eq!(out.sorted_keys, want.sorted_keys, "sorted keys");
    // Against the standard sort: the runs of equal keys are the cells.
    let mut got = runs(&out);
    let n_runs = got.len();
    got.sort_by_key(|r| pack_key(r.0));
    assert_eq!(got.len(), n_runs);
    let reference = cells_reference(&cells);
    assert_eq!(got, reference, "cells and members");
    assert!(
        got.windows(2).all(|w| w[0].0 != w[1].0),
        "a cell in two runs"
    );
    // Against f64: the edge rule.
    let g = s.binner.geometry();
    let alpha = g.alpha_rad(time_s);
    let mut differ = 0;
    for (i, p) in pos.iter().enumerate() {
        if s.reasons[i] != 0 {
            continue;
        }
        let (bf, lf) = cell_f64(g, alpha, *p);
        let bg = unpack_key(keys[i]).map_or_else(
            || {
                if bf == Binned::OutOfRange {
                    Binned::OutOfRange
                } else {
                    Binned::NotBinned
                }
            },
            Binned::Cell,
        );
        if bf != bg {
            differ += 1;
            let near = edge_distance_km(g, lf)
                .into_iter()
                .chain([(lf[1] - g.domain_km).abs()])
                .fold(f64::INFINITY, f64::min);
            assert!(
                near <= DELTA,
                "particle {i}: {bf:?} (f64) vs {bg:?} (GPU), edge distance {near}"
            );
        }
    }
    differ
}

#[test]
#[ignore = "needs a GPU"]
fn gpu_sort_agrees_with_twin_and_std_sort_on_random_particles() {
    let Some(gpu) = gpu() else { return };
    let cell = [0.05, 10.0, 0.04];
    for (name, domain, phi0, t_s, n) in [
        ("ring, t = 0", TAU * A0, 0.0, 0.0, 100_000usize),
        ("ring, t = 3.1e8 s", TAU * A0, 0.0, 3.1e8, 100_000),
        ("arc 5 deg", A0 * 5f64.to_radians(), 0.2, 1.0e7, 60_001),
        ("one particle", TAU * A0, 0.0, 5.0, 1),
        ("63 particles", TAU * A0, 0.0, 5.0, 63),
    ] {
        let g = BinGeometry::new(A0, N0, cell, domain, phi0).unwrap();
        let pos = particles(&g, t_s, n, 11);
        // Some particles are dropped (reason 1 or 2).
        let reasons: Vec<u32> = (0..n)
            .map(|i| match i % 101 {
                3 => 1,
                50 => 2,
                _ => 0,
            })
            .collect();
        for buckets in [None, Some(1usize << 20)] {
            let s = setup(&gpu, g, &pos, &reasons, buckets);
            let m = s.binner.buckets();
            let differ = check_against_twin(&s, t_s, m, &pos);
            let out = s.bin(t_s).0;
            eprintln!(
                "{name}: n = {n}, M = {m}, binned {}, dropped/bad {}, out of range {}, \
                 runs {}, cells differing from f64: {differ}",
                out.n_binned,
                out.n_not_binned,
                out.n_out_of_range,
                runs(&out).len()
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn few_buckets_keep_different_cells_in_separate_runs() {
    let Some(gpu) = gpu() else { return };
    let g = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], TAU * A0, 0.0).unwrap();
    let n = 6000;
    let pos = particles(&g, 0.0, n, 3);
    let reasons = vec![0u32; n];
    for m in [1usize, 2, 4, 64] {
        let s = setup(&gpu, g, &pos, &reasons, Some(m));
        assert_eq!(s.binner.buckets(), m);
        check_against_twin(&s, 0.0, m, &pos);
        let (out, _) = s.bin(0.0);
        // More cells than buckets: some bucket holds several cells.
        let r = runs(&out);
        assert!(r.len() > m, "{} cells in {m} buckets", r.len());
        eprintln!("M = {m}: {} cells, largest bucket {}", r.len(), {
            out.bucket_start
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap()
        });
    }
}

#[test]
#[ignore = "needs a GPU"]
fn edge_cases_none_binned_all_out_of_range_and_all_in_one_cell() {
    let Some(gpu) = gpu() else { return };
    let g = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], TAU * A0, 0.0).unwrap();
    // All dropped.
    let pos = particles(&g, 0.0, 300, 4);
    let s = setup(&gpu, g, &pos, &vec![1; 300], None);
    let (out, keys) = s.bin(0.0);
    assert_eq!((out.n_binned, out.n_not_binned), (0, 300));
    assert!(keys.iter().all(|&k| k == NO_KEY));
    assert!(out.sorted.iter().all(|&i| i == NO_ID));
    assert!(out.sorted_keys.iter().all(|&k| k == NO_KEY));
    assert_eq!(out.bucket_start.last(), Some(&0));
    // The origin, and a particle far outside the key range.
    let far = [A0 + 0.05 * 3.0e6, 0.0, 0.0];
    let pos = vec![[0.0; 3], far, [A0 + 0.01, 0.0, 0.0]];
    let s = setup(&gpu, g, &pos, &[0, 0, 0], None);
    let (out, _) = s.bin(0.0);
    assert_eq!(
        (out.n_binned, out.n_not_binned, out.n_out_of_range),
        (1, 1, 1)
    );
    assert_eq!(out.sorted[0], 2);
    assert_eq!(&out.sorted[1..], &[NO_ID, NO_ID]);
    // An arc: a particle behind the end of the arc is out of range.
    let arc = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], 12_237.0, 0.3).unwrap();
    let alpha = arc.alpha_rad(0.0);
    let pos: Vec<[f64; 3]> = [100.0, 12_000.0, 12_300.0, 50_000.0]
        .iter()
        .map(|&y| inertial(&arc, alpha, [0.0123, y, 0.0123]))
        .collect();
    let s = setup(&gpu, arc, &pos, &[0; 4], None);
    let (out, _) = s.bin(0.0);
    assert_eq!((out.n_binned, out.n_out_of_range), (2, 2));
    check_against_twin(&s, 0.0, s.binner.buckets(), &pos);
    // All particles in one cell: ids in order, whatever the atomic order.
    for n in [4096usize, 16_384, 32_767] {
        let p = inertial(&g, 0.0, [0.0123, 3.0, 0.0123]);
        let pos = vec![p; n];
        let s = setup(&gpu, g, &pos, &vec![0; n], None);
        let _ = s.binner.run_timed(0.0, Stage::ALL.len()); // warm-up
        let t = (0..3)
            .map(|_| s.binner.run_timed(0.0, Stage::ALL.len()))
            .min()
            .unwrap();
        let (out, _) = s.bin(0.0);
        assert_eq!(out.n_binned as usize, n);
        assert_eq!(out.sorted, (0..n as u32).collect::<Vec<_>>());
        let r = runs(&out);
        assert_eq!(r.len(), 1);
        eprintln!(
            "all {n} particles in one cell: best of 3 passes (incl. submit) {:.2} ms",
            t.as_secs_f64() * 1e3
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn particles_next_to_an_edge_agree_with_the_twin_and_obey_the_rule() {
    let Some(gpu) = gpu() else { return };
    let offsets = [
        -1e-5, -1e-6, -1e-7, -1e-8, -1e-9, 0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
    ];
    let (mut total, mut gpu_vs_twin, mut gpu_vs_f64) = (0usize, 0usize, 0usize);
    for (domain, phi0, t_s) in [
        (TAU * A0, 0.0, 0.0),
        (TAU * A0, 0.0, 7.0e7),
        (12_237.0, 0.3, 1e6),
    ] {
        let g = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], domain, phi0).unwrap();
        let alpha = g.alpha_rad(t_s);
        let u = uniform(5, 1, 3 * 200);
        for axis in 0..3 {
            let mut pos = Vec::new();
            let mut offs = Vec::new();
            for &off in &offsets {
                for r in 0..200 {
                    let (ix, iy, iz) = (
                        (u[3 * r] * 400.0) as i32 - 200,
                        (u[3 * r + 1] * f64::from(g.ny - 2)) as u32 + 1,
                        (u[3 * r + 2] * 200.0) as i32 - 100,
                    );
                    let mut local = [
                        (f64::from(ix) + 0.5) * g.lx_km,
                        (f64::from(iy) + 0.5) * g.ly_km,
                        (f64::from(iz) + 0.5) * g.lz_km,
                    ];
                    local[axis] += off - 0.5 * [g.lx_km, g.ly_km, g.lz_km][axis];
                    pos.push(inertial(&g, alpha, local));
                    offs.push(off);
                }
            }
            let n = pos.len();
            let s = setup(&gpu, g, &pos, &vec![0; n], None);
            let (_, keys) = s.bin(t_s);
            let twin = s.twin_cells(t_s);
            for i in 0..n {
                total += 1;
                let (bf, lf) = cell_f64(&g, alpha, pos[i]);
                if keys[i] != key_of(twin[i]) {
                    gpu_vs_twin += 1;
                    assert!(
                        offs[i].abs() < 1e-6,
                        "GPU and twin differ at offset {}",
                        offs[i]
                    );
                }
                if keys[i] != key_of(bf) {
                    gpu_vs_f64 += 1;
                    let near = edge_distance_km(&g, lf)[axis];
                    assert!(near <= DELTA, "axis {axis}: {near}");
                    assert!(offs[i].abs() < 1e-6);
                }
            }
        }
    }
    eprintln!(
        "edge cases: {total} particles; GPU key differs from the twin for {gpu_vs_twin}, from f64 for {gpu_vs_f64} (all within {DELTA:e} km of an edge)"
    );
    let _ = Cell {
        ix: 0,
        iy: 0,
        iz: 0,
    };
}

/// The binner behind a real stepper in one command encoder: the time given to
/// the binner is `stepper.time_s()` after `encode_steps`, and the cells agree
/// with the f64 cells of the state that the GPU holds (a wrong time of one
/// step, 356 s, would turn the ring by 0.04 rad = 5900 km and fail at once).
#[test]
#[ignore = "needs a GPU"]
fn binner_follows_the_stepper_in_one_encoder() {
    let Some(gpu) = gpu() else { return };
    let steps = 40u64;
    let system = common::system(common::H_S, steps + 10, true);
    let n = 20_000;
    let (init, _) = common::ensemble(&system, n);
    let kernels = WhKernels::new(&gpu, DEFAULT_WORKGROUP).expect("kernels");
    let mut stepper = GpuStepper::new(
        &gpu,
        &kernels,
        &system,
        &init,
        common::H_S,
        8,
        Mode::Dp,
        None,
    )
    .expect("stepper");
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let n0 = reference_mean_motion(&field, common::A0_KM);
    let g = BinGeometry::new(
        common::A0_KM,
        n0,
        [0.05, 10.0, 0.04],
        TAU * common::A0_KM,
        0.0,
    )
    .unwrap();
    let binner = GpuBinner::for_stepper(&gpu, &stepper, g).expect("binner");
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    stepper.encode_steps(&mut enc, steps).expect("steps");
    binner.encode(&mut enc, stepper.time_s());
    gpu.queue().submit([enc.finish()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let t = stepper.time_s();
    assert!((t - steps as f64 * common::H_S).abs() < 1e-6);
    // The state the GPU holds, as f64 (planes x, y, z of hi + lo).
    let cap = stepper.capacity();
    let w = binner.read_words(stepper.state_buffer(), 0, 12 * cap);
    let f = |c: usize, i: usize| {
        f64::from(f32::from_bits(w[2 * (c * cap + i)]))
            + f64::from(f32::from_bits(w[2 * (c * cap + i) + 1]))
    };
    let keys = binner.read_particle_keys();
    let alpha = g.alpha_rad(t);
    let mut cells_seen = std::collections::HashSet::new();
    let mut differ = 0;
    for (i, &key) in keys.iter().enumerate().take(n) {
        let (bf, lf) = cell_f64(&g, alpha, [f(0, i), f(1, i), f(2, i)]);
        if key != key_of(bf) {
            differ += 1;
            let near = edge_distance_km(&g, lf)
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            assert!(near <= DELTA, "particle {i}: edge distance {near}");
        }
        cells_seen.insert(key);
    }
    // Not a trivial binning: many cells, and a time that matters.
    assert!(cells_seen.len() > n / 4, "{} cells", cells_seen.len());
    let wrong = g.alpha_rad(t - common::H_S);
    let off = (0..n)
        .filter(|&i| key_of(cell_f64(&g, wrong, [f(0, i), f(1, i), f(2, i)]).0) != keys[i])
        .count();
    assert!(
        off > n / 2,
        "a time wrong by one step must change most keys: {off}"
    );
    let out = binner.read_output();
    assert_eq!(
        out.n_binned as usize + out.n_not_binned as usize + out.n_out_of_range as usize,
        n
    );
    eprintln!(
        "stepper + binner: t = {t:.1} s, {} cells, {differ} keys differ from f64 (within {DELTA:e} km of an edge), \
         a time wrong by one step changes {off} of {n} keys",
        cells_seen.len()
    );
}
