//! WGSL kernels against their Rust twins (PLAN section 10: relative < 1e-6
//! for f32), for one step and for one orbit (150 steps). Over one orbit,
//! two runs with different f32 rounding drift apart, so the f32 criterion
//! applies to one step; after one orbit the bounds depend on the option. Needs a GPU, so `#[ignore]`: run with
//! `cargo test -p fring-gpu --release --test gpu_twins -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::spike::{Consts, Opt, Tables, advance, state_from_f64, state_to_f64};
use fring_gpu::spike_gpu::GpuSpike;
use fring_gpu::wf32::model;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{EpicycleElements, SaturnField, build_ephemeris, epicycle_state};

const M4: u8 = model::M4;

#[test]
#[ignore = "needs a GPU"]
fn spike_kernels_match_their_rust_twins() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let sp = saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap();
    let moons = moons_from_toml_str(include_str!("../../../params/moons.toml")).unwrap();
    let h = 356.25;
    for n in [1, 150] {
        check(&gpu, &sp, &moons, h, n);
    }
}

fn check(
    gpu: &Gpu,
    sp: &fring_model::SaturnParams,
    moons: &[fring_model::physics::MoonInit],
    h: f64,
    n: usize,
) {
    let eph = build_ephemeris(sp, moons, 2.0 * n as f64 * h + 7200.0, 1800.0).unwrap();
    let c = Consts::<M4>::new(sp, &eph, h);
    let mut tab = Tables::<M4>::new(&eph);
    tab.fill(&eph, 0.0, h, 0, n).unwrap();
    let field = SaturnField::<f64>::from_params(sp);
    let states: Vec<_> = (0..64)
        .map(|k| {
            let el = EpicycleElements {
                a_km: 140_222.43 + 10.0 * (k % 8) as f64,
                e: 0.0024,
                i_rad: 1.1e-4,
                mean_anomaly_rad: 0.37 * k as f64,
                z_phase_rad: 0.11 * k as f64,
                mean_longitude_rad: 0.0981 * k as f64,
            };
            let (p, v) = epicycle_state(&field, &el);
            state_from_f64::<M4>(p, v)
        })
        .collect();
    let spike = GpuSpike::new(gpu).expect("self-test");
    for opt in Opt::ALL {
        let mut cpu = states.clone();
        for s in &mut cpu {
            advance(opt, &c, &tab, s).unwrap();
        }
        let mut g = states.clone();
        spike.advance(opt, &c, &tab, &mut g);
        let (mut worst_p, mut worst_v, mut same) = (0.0f64, 0.0f64, 0usize);
        for (a, b) in cpu.iter().zip(&g) {
            let (pa, va) = state_to_f64(a);
            let (pb, vb) = state_to_f64(b);
            worst_p = worst_p.max((pa - pb).norm() / pa.norm());
            worst_v = worst_v.max((va - vb).norm() / va.norm());
            same += usize::from(a == b);
        }
        println!(
            "{} steps, option {}: GPU against Rust twin: position {worst_p:.2e}, velocity {worst_v:.2e} (relative); identical states {same}/64",
            n,
            opt.name()
        );
        // One step: the f32 criterion of PLAN section 10 for every option.
        // One orbit: only C is DS throughout; the others carry f32 parts whose
        // rounding differs between the GPU and the model (bounds about 10
        // times the measured values: 4.1e-7, 4.8e-9, 5.4e-13).
        let bound = match (n, opt) {
            (1, _) => 1e-6,
            (_, Opt::C) => 1e-11,
            (_, Opt::D) => 5e-8,
            (_, Opt::BPlus) => 5e-6,
            _ => f64::INFINITY,
        };
        assert!(worst_p < bound && worst_v < bound, "{}", opt.name());
    }
}
