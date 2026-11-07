//! ARC-CELLS (GPU part): the cell of a particle in an arc domain, after the D1
//! wrap. The plain tests need no GPU; the GPU tests are `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_bin_arc -- --include-ignored --nocapture`.
//!
//! Rules (the K5-BIN rules of A0): GPU against the DS twin exactly; the twin
//! against f64 only for a particle within `DELTA` of a cell edge (the edges of
//! the arc included).
//!
//! Choice made: a particle outside an edge by at most `ARC_SLACK_RAD` goes into
//! the edge cell (clamp), not into its periodic image. That is the simpler
//! rule: the two ends of the arc are not neighbours in the cell grid.

mod common;

use fring_gpu::bin::{
    ARC_SLACK_RAD, BinGeometry, Binned, Dsm, GpuBinner, bin_twin, cell_ds, cell_f64, key_of, params,
};
use fring_gpu::gpu::Gpu;
use fring_gpu::probe::self_test;
use fring_gpu::wh_gpu::{pack_states, status_init_bytes};
use fring_model::config::{Collisions, Domain, Restitution};
use fring_model::physics::arc::EDGE_TOL;
use fring_model::physics::{ArcGeometry, SaturnField, reference_mean_motion};
use fring_model::{Backend, RunConfig};
use std::f64::consts::TAU;

/// 10 x the largest DS error of K5-BIN, rounded up (as in `gpu_bin.rs`).
const DELTA: f64 = 1.01e-7;

fn stochastic_config(width_deg: f64, centre_deg: f64) -> RunConfig {
    let mut cfg = common::config(common::H_S, 400, false);
    cfg.collisions = Collisions::Stochastic {
        particle_radius_km: 1e-3,
        weight: 1.0,
        cell_radial_km: 0.05,
        cell_azimuthal_km: 10.0,
        cell_vertical_km: 0.04,
        every_steps: 1,
        restitution: Restitution::Constant { epsilon: 0.5 },
    };
    cfg.domain = Domain::Arc {
        width_deg,
        centre_longitude_deg: centre_deg,
    };
    cfg
}

fn n0() -> f64 {
    let system = common::system(common::H_S, 400, false);
    let field = SaturnField::<f64>::from_params(&system.saturn);
    reference_mean_motion(&field, 140_220.0)
}

fn arc_geometry(width_deg: f64, centre_deg: f64) -> BinGeometry {
    let g = BinGeometry::from_config(&stochastic_config(width_deg, centre_deg), n0()).unwrap();
    assert!(g.arc);
    g
}

/// The inertial position at the signed angle `s` rad from the arc centre,
/// radius a0 + x, height z, when the centre is at `alpha`.
fn at_angle(g: &BinGeometry, alpha: f64, s: f64, x: f64, z: f64) -> [f64; 3] {
    let (rho, phi) = (g.a0_km + x, alpha + s);
    [rho * phi.cos(), rho * phi.sin(), z]
}

#[derive(Debug, PartialEq)]
enum Want {
    First,
    Last,
    Inside,
    Out,
}

/// Particles at the angles `offset` rad outside the edges (negative: inside),
/// on both edges, with the cell expected.
fn edge_cases(g: &BinGeometry) -> Vec<(f64, Want, &'static str)> {
    let h = 0.5 * g.domain_km / g.a0_km;
    let wrap_slack = EDGE_TOL * (h.cos() + h.sin()); // what the D1 wrap leaves
    assert!(wrap_slack < ARC_SLACK_RAD);
    let mut v = Vec::new();
    for (sign, name) in [(-1.0, "start"), (1.0, "end")] {
        let mk = |off: f64| sign * (h + off);
        v.push((mk(-1.0e-4), Want::Inside, name)); // inside
        v.push((mk(0.0), edge_of(sign), name)); // exactly at the edge
        v.push((mk(0.5 * wrap_slack), edge_of(sign), name));
        v.push((mk(wrap_slack), edge_of(sign), name)); // the largest the wrap leaves
        v.push((mk(2.0 * ARC_SLACK_RAD), Want::Out, name)); // beyond the slack
        v.push((mk(1.0e-3), Want::Out, name));
    }
    v
}

fn edge_of(sign: f64) -> Want {
    if sign < 0.0 { Want::First } else { Want::Last }
}

fn check_cell(g: &BinGeometry, b: Binned, want: &Want, what: &str) {
    let ok = match (want, b) {
        (Want::First, Binned::Cell(c)) => c.iy == 0,
        (Want::Last, Binned::Cell(c)) => c.iy == g.ny - 1,
        (Want::Inside, Binned::Cell(c)) => c.iy < g.ny,
        (Want::Out, Binned::OutOfRange) => true,
        _ => false,
    };
    assert!(ok, "{what}: want {want:?}, got {b:?}");
}

#[test]
fn arc_edges_inside_slack_and_beyond_on_both_edges() {
    for (w, c) in [
        (7.0, 180.0),
        (7.0, 0.0),
        (5.0, 30.0),
        (10.0, -90.0),
        (30.0, 175.0),
    ] {
        let g = arc_geometry(w, c);
        let prm = params(&g);
        for t in [0.0, 12_345.6, 3.0e7] {
            let alpha = g.alpha_rad(t);
            for (s, want, name) in edge_cases(&g) {
                let p = at_angle(&g, alpha, s, 0.0123, 0.0123);
                let what = format!("{w} deg at {c}, t {t}, {name} edge, s {s:e}");
                check_cell(&g, cell_f64(&g, alpha, p).0, &want, &format!("f64 {what}"));
                let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
                check_cell(
                    &g,
                    cell_ds(&prm, g.ny, Dsm::split(alpha), ps).0,
                    &want,
                    &format!("DS {what}"),
                );
            }
        }
    }
}

/// The y coordinate runs 0..D across the arc, also across the branch cut of
/// atan2 (arc of 7 degrees centred at 180 degrees; 360/7 is not a whole
/// number), and the cell index follows the angle from the start edge.
#[test]
fn arc_across_the_branch_cut_has_cells_in_order() {
    let g = arc_geometry(7.0, 180.0);
    let prm = params(&g);
    let h = 0.5 * g.domain_km / g.a0_km;
    let t = 0.0; // alpha is the centre: 180 degrees
    let alpha = g.alpha_rad(t);
    assert!((alpha - std::f64::consts::PI).abs() < 1e-12);
    let mut last = 0u32;
    for k in 0..=1000 {
        let s = -h + 2.0 * h * (f64::from(k) + 0.5) / 1001.0;
        let p = at_angle(&g, alpha, s, 0.0, 0.0);
        let Binned::Cell(c) = cell_f64(&g, alpha, p).0 else {
            panic!("s {s}")
        };
        assert!(c.iy >= last, "cells must not go back");
        last = c.iy;
        let y = g.a0_km * (s + h);
        assert!((f64::from(c.iy) - (y / g.ly_km).floor()).abs() <= 1.0);
        let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
        let (_, local) = cell_ds(&prm, g.ny, Dsm::split(alpha), ps);
        assert!((local[1] - y).abs() < 1e-7, "y {} against {y}", local[1]);
    }
    assert!(last > g.ny - 5, "the last cells are reached: {last}");
}

/// The full ring is unchanged: same parameters (arc words zero), same alpha.
#[test]
fn full_ring_parameters_are_unchanged() {
    let a0 = 140_220.0;
    let g = BinGeometry::new(a0, n0(), [0.05, 10.0, 0.04], TAU * a0, 0.0).unwrap();
    assert!(!g.arc);
    let w = params(&g);
    assert_eq!(&w[0..4], &[1.0, 0.0, 0.0, 0.0]);
    assert!(w[16..20].iter().all(|&x| x == 0.0), "row 4 is unused");
    assert_eq!(g.alpha_rad(777.0), (n0() * 777.0).rem_euclid(TAU));
    assert_eq!(g.slack_km(), 0.0);
    // The full ring from the config is not an arc either.
    let mut cfg = stochastic_config(7.0, 180.0);
    cfg.domain = Domain::FullRing;
    assert!(!BinGeometry::from_config(&cfg, n0()).unwrap().arc);
}

struct Setup {
    gpu: Gpu,
    binner: GpuBinner,
    ds: Vec<[Dsm; 3]>,
}

fn gpu() -> Option<Gpu> {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return None;
    };
    self_test(&gpu).expect("DS self-test");
    Some(gpu)
}

fn setup(gpu: &Gpu, g: BinGeometry, pos: &[[f64; 3]]) -> Setup {
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
    let bytes: Vec<u8> = pack_states(&states, cap)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
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
    let sb = status_init_bytes(cap);
    let status = mk(sb.len() as u64);
    gpu.queue().write_buffer(&status, 0, &sb);
    let binner = GpuBinner::new(gpu, &state, &status, n, cap, g, None).expect("binner");
    Setup {
        gpu: gpu.clone(),
        binner,
        ds: states.iter().map(|s| [s[0], s[1], s[2]]).collect(),
    }
}

/// Dist of y to the nearest cell edge or arc end, km.
fn edge_km(g: &BinGeometry, y: f64) -> f64 {
    let r = y - g.ly_km * (y / g.ly_km).round();
    r.abs().min((y - g.domain_km).abs()).min(y.abs())
}

/// GPU = twin exactly, twin = f64 except within DELTA of an edge. Returns
/// (n_binned, n_out_of_range).
fn check(s: &Setup, t: f64, pos: &[[f64; 3]]) -> (u32, u32) {
    let g = *s.binner.geometry();
    let mut enc = s.gpu.device().create_command_encoder(&Default::default());
    s.binner.encode(&mut enc, t);
    s.gpu.queue().submit([enc.finish()]);
    s.gpu
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let (out, keys) = (s.binner.read_output(), s.binner.read_particle_keys());
    let prm = params(&g);
    let alpha = g.alpha_rad(t);
    let cells: Vec<Binned> =
        s.ds.iter()
            .map(|p| cell_ds(&prm, g.ny, Dsm::split(alpha), *p).0)
            .collect();
    for (i, c) in cells.iter().enumerate() {
        assert_eq!(keys[i], key_of(*c), "particle {i}: GPU against twin");
    }
    let want = bin_twin(&cells, s.binner.buckets());
    assert_eq!(out.sorted, want.sorted);
    assert_eq!(out.sorted_keys, want.sorted_keys);
    assert_eq!(out.bucket_start, want.bucket_start);
    assert_eq!(out.n_out_of_range, want.n_out_of_range);
    for (i, p) in pos.iter().enumerate() {
        let (bf, lf) = cell_f64(&g, alpha, *p);
        if bf != cells[i] {
            assert!(
                edge_km(&g, lf[1]) <= DELTA
                    || (lf[0] / g.lx_km - (lf[0] / g.lx_km).round()).abs() * g.lx_km <= DELTA
                    || (lf[2] / g.lz_km - (lf[2] / g.lz_km).round()).abs() * g.lz_km <= DELTA,
                "particle {i}: {bf:?} (f64) against {:?} (twin), local {lf:?}",
                cells[i]
            );
        }
    }
    (out.n_binned, out.n_out_of_range)
}

#[test]
#[ignore = "needs a GPU"]
fn gpu_sort_equals_twin_with_particles_on_and_beyond_the_edges() {
    let Some(gpu) = gpu() else { return };
    for (w, c) in [(7.0, 180.0), (5.0, 30.0), (30.0, 175.0)] {
        let g = arc_geometry(w, c);
        let h = 0.5 * g.domain_km / g.a0_km;
        let t = 98_765.4;
        let alpha = g.alpha_rad(t);
        let mut pos = Vec::new();
        let (mut n_out, mut n_in) = (0u32, 0u32);
        for (s, want, _) in edge_cases(&g) {
            for k in 0..40 {
                let x = 0.05 * f64::from(k % 8) + 0.01;
                pos.push(at_angle(&g, alpha, s, x, 0.002 * f64::from(k)));
                if want == Want::Out {
                    n_out += 1;
                } else {
                    n_in += 1;
                }
            }
        }
        // And a spread over the arc.
        for k in 0..2000 {
            let s = -h + 2.0 * h * (f64::from(k) + 0.37) / 2000.0;
            pos.push(at_angle(&g, alpha, s, 0.021 * f64::from(k % 11), 0.0));
            n_in += 1;
        }
        let s = setup(&gpu, g, &pos);
        let (binned, out) = check(&s, t, &pos);
        assert_eq!((binned, out), (n_in, n_out), "{w} deg at {c}");
        eprintln!(
            "{w} deg arc at {c}: {binned} binned, {out} out of range (beyond the slack): as expected"
        );
    }
}

/// GpuBackend with the arc on, a few hundred steps: the binner finds no
/// particle out of range, while the old plain-domain rule does.
#[test]
#[ignore = "needs a GPU"]
fn backend_run_with_the_arc_has_no_out_of_range_particle() {
    use fring_gpu::backend::{GpuBackend, Options};
    use fring_gpu::wh::Mode;
    let Some(gpu) = gpu() else { return };
    let _ = gpu;
    let steps = 300u64;
    let (width, centre) = (7.0, 180.0);
    let mut cfg = common::config(common::H_S, steps + 10, false);
    cfg.domain = Domain::Arc {
        width_deg: width,
        centre_longitude_deg: centre,
    };
    cfg.validate().unwrap();
    let system = common::system(common::H_S, steps + 10, false);
    let n = 100_000;
    let (init, _) = common::ensemble(&system, n);
    let mut b = GpuBackend::with_options(
        &cfg,
        &system,
        &init,
        Options {
            mode: Mode::Dp,
            ..Default::default()
        },
    )
    .unwrap();
    let mut bcfg = cfg.clone();
    bcfg.collisions = stochastic_config(width, centre).collisions;
    let g = BinGeometry::from_config(&bcfg, n0_of(&cfg, &system)).unwrap();
    let old = BinGeometry { arc: false, ..g };
    let (mut worst_new, mut worst_old, mut binned_min) = (0u32, 0u32, u32::MAX);
    let binner = GpuBinner::for_stepper(b.gpu(), b.stepper(), g).unwrap();
    let binner_old = GpuBinner::for_stepper(b.gpu(), b.stepper(), old).unwrap();
    for _ in 0..3 {
        b.step(steps / 3).unwrap();
        let t = b.stepper().time_s();
        for (bn, worst) in [(&binner, &mut worst_new), (&binner_old, &mut worst_old)] {
            let mut enc = b.gpu().device().create_command_encoder(&Default::default());
            bn.encode(&mut enc, t);
            b.gpu().queue().submit([enc.finish()]);
            b.gpu()
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            let o = bn.read_output();
            *worst = (*worst).max(o.n_out_of_range);
            if std::ptr::eq(bn, &binner) {
                binned_min = binned_min.min(o.n_binned);
            }
        }
    }
    eprintln!(
        "{n} particles, {steps} steps, 7 deg arc at 180 deg: out of range, arc rule {worst_new}, plain-domain rule {worst_old}; binned at least {binned_min}"
    );
    assert_eq!(worst_new, 0);
    assert_eq!(binned_min as usize, n);
}

fn n0_of(cfg: &RunConfig, system: &fring_model::System) -> f64 {
    let field = SaturnField::<f64>::from_params(&system.saturn);
    reference_mean_motion(&field, cfg.frame.reference_semi_major_axis_km)
}

#[test]
fn arc_geometry_matches_the_wrap_geometry() {
    // The binning centre is the wrap centre: a particle on the wrap's edge is
    // on the binning's edge.
    let g = arc_geometry(7.0, 180.0);
    let a = ArcGeometry::new(7f64.to_radians(), 180f64.to_radians(), n0()).unwrap();
    assert!((a.width_rad() * g.a0_km - g.domain_km).abs() < 1e-9);
}
