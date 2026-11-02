//! Shared set-up of the GPU tests: the G1 ensemble (ADR 0006), a config and a
//! system with the real moons.

#![allow(dead_code)]

use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{EpicycleElements, SaturnField, build_ephemeris, epicycle_state};
use fring_model::{InitialState, Particles, RunConfig, System};

/// Semi-major axis of the G1 ensemble, km.
pub const A0_KM: f64 = 140_222.43;
/// A step of about 150 per orbit that is not exact in f32 (ADR 0007), s.
pub const H_S: f64 = 356.2537;
/// Seconds in a Julian year.
pub const YEAR_S: f64 = 365.25 * 86_400.0;
/// The three orbit types of G1: (name, e, i).
pub const ORBIT_TYPES: [(&str, f64, f64); 3] = [
    ("circular", 0.0, 0.0),
    ("core", 0.0024, 1.1e-4),
    ("stress", 0.01, 1.0e-3),
];

/// Config of the example run with the given step and number of steps.
pub fn config(step_s: f64, steps: u64, moons: bool) -> RunConfig {
    let text = include_str!("../../../fring-model/tests/data/example-run.toml");
    let mut t = text.replace("step_s = 360.0", &format!("step_s = {step_s}"));
    // A little more than the steps, so that the run end is inside the table.
    t = t.replace(
        "duration_years = 1.0",
        &format!("duration_years = {}", steps as f64 * step_s / YEAR_S),
    );
    if !moons {
        t = t.replace(r#"include = ["prometheus", "pandora"]"#, "include = []");
    }
    RunConfig::from_toml_str(&t).unwrap()
}

/// Saturn and (if `moons`) Prometheus and Pandora, with a table that covers
/// `steps` steps of `step_s`.
pub fn system(step_s: f64, steps: u64, moons: bool) -> System {
    let saturn = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
    let all = moons_from_toml_str(include_str!("../../../../params/moons.toml")).unwrap();
    let eph = if moons {
        build_ephemeris(&saturn, &all, steps as f64 * step_s + 7200.0, 1800.0).unwrap()
    } else {
        fring_model::MoonEphemeris::none()
    };
    System { saturn, moons: eph }
}

/// `n` particles, a third of each orbit type of G1, at `A0_KM` with golden
/// ratio angles. Returns the particles and the orbit type of each.
pub fn ensemble(system: &System, n: usize) -> (InitialState, Vec<usize>) {
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let mut p = Particles::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    let golden = 0.618_033_988_749_894_9_f64;
    for i in 0..n {
        let kind = i % 3;
        let k = (i / 3) as f64;
        let (_, e, inc) = ORBIT_TYPES[kind];
        let el = EpicycleElements {
            a_km: A0_KM,
            e,
            i_rad: inc,
            mean_anomaly_rad: std::f64::consts::TAU * ((k * golden + 0.17) % 1.0),
            z_phase_rad: std::f64::consts::TAU * ((k * 0.754_877_666_246_693 + 0.31) % 1.0),
            mean_longitude_rad: std::f64::consts::TAU
                * ((k * 0.569_840_290_998_053_2 + 0.07) % 1.0),
        };
        let (pos, vel) = epicycle_state(&field, &el);
        p.push(i as u32, pos, vel);
        kinds.push(kind);
    }
    (
        InitialState {
            time_s: 0.0,
            particles: p,
        },
        kinds,
    )
}

/// Copies a buffer to the host (blocking; for tests).
pub fn read_buffer(gpu: &fring_gpu::gpu::Gpu, buf: &wgpu::Buffer) -> Vec<u8> {
    let d = gpu.device();
    let stage = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buf.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &stage, 0, buf.size());
    gpu.queue().submit([enc.finish()]);
    stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v = stage.get_mapped_range(..).expect("range").to_vec();
    stage.unmap();
    v
}

/// FNV-1a 64-bit hash of bytes (for the golden-bits tests).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
