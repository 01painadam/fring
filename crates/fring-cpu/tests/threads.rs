//! The result of `CpuBackend` must not depend on the number of threads:
//! bit-identical for 1, 3 and 8 threads, and equal to a serial loop that does
//! not use rayon. Also the throughput (ignored test).

use fring_cpu::CpuBackend;
use fring_cpu::integrate::{State, advance_chunk};
use fring_model::config::Distribution;
use fring_model::physics::{KickTable, SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, Particles, RunConfig, System};

fn system(duration_s: f64) -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let moons =
        fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
            .unwrap();
    let moons = build_ephemeris(&saturn, &moons, duration_s, 1800.0).unwrap();
    System { saturn, moons }
}

fn config() -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    RunConfig::from_toml_str(&text.replace("duration_years = 1.0", "duration_years = 0.04"))
        .unwrap()
}

fn bits(p: &Particles) -> Vec<u64> {
    [
        &p.x_km, &p.y_km, &p.z_km, &p.vx_km_s, &p.vy_km_s, &p.vz_km_s,
    ]
    .iter()
    .flat_map(|v| v.iter().map(|x| x.to_bits()))
    .collect()
}

/// A band across both moons, so that many particles meet a moon closely.
fn band(sys: &System, count: u32) -> InitialState {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_000.0,
        max_semi_major_axis_km: 141_800.0,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 20_261_101).unwrap(),
    }
}

const CALLS: [u64; 3] = [1000, 1000, 1000];

fn run_backend(sys: &System, ini: &InitialState, threads: Option<usize>) -> Particles {
    let work = || {
        let mut b = CpuBackend::new(&config(), sys, ini).unwrap();
        for n in CALLS {
            b.step(n).unwrap();
        }
        b.particles().unwrap().particles.clone()
    };
    match threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(work),
        None => work(),
    }
}

/// The same work in one thread, without rayon, with the same blocks.
fn run_serial(sys: &System, ini: &InitialState) -> Particles {
    let field = SaturnField::<f64>::from_params(&sys.saturn)
        // ICR-011: the backend drifts about mu_eff at a0 (config a0 = 140,220 km).
        .with_drift_radius(140_220.0);
    let p = &ini.particles;
    let mut state: Vec<State> = (0..p.len())
        .map(|i| {
            [
                p.x_km[i],
                p.y_km[i],
                p.z_km[i],
                p.vx_km_s[i],
                p.vy_km_s[i],
                p.vz_km_s[i],
            ]
        })
        .collect();
    let h = config().time.step_s;
    let mut table = KickTable::<f64>::for_ephemeris(&sys.moons);
    let mut done = 0;
    for n in CALLS {
        table
            .fill(&sys.moons, ini.time_s, h, done, n as usize)
            .unwrap();
        assert!(advance_chunk(&field, Some(&table), &mut state, h, n).is_none());
        done += n;
    }
    let mut out = Particles::default();
    for (i, s) in state.iter().enumerate() {
        out.push(
            p.id[i],
            fring_model::Vec3::new(s[0], s[1], s[2]),
            fring_model::Vec3::new(s[3], s[4], s[5]),
        );
    }
    out
}

#[test]
fn one_and_many_threads_give_identical_bits() {
    let sys = system(1.3e6);
    let ini = band(&sys, 1000);
    let serial = bits(&run_serial(&sys, &ini));
    let moved = bits(&ini.particles) != serial;
    assert!(moved);
    for threads in [Some(1), Some(3), Some(8), None] {
        let got = bits(&run_backend(&sys, &ini, threads));
        assert!(
            got == serial,
            "threads {threads:?} differ from the serial loop"
        );
    }
}

#[test]
#[ignore = "throughput: run in release with --ignored --nocapture"]
fn throughput() {
    let sys = system(1.3e6);
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Core {
        semi_major_axis_km: 140_220.0,
        sigma_semi_major_axis_km: 5.0,
        eccentricity: 0.0024,
        apse_longitude_deg: 0.0,
        inclination_deg: 0.0,
    };
    let n_p = 10_000;
    let ini = InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, n_p, 1).unwrap(),
    };
    let steps = 1460_u64;
    for threads in [1, rayon::current_num_threads()] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut b = CpuBackend::new(&config(), &sys, &ini).unwrap();
        let t = std::time::Instant::now();
        pool.install(|| b.step(steps).unwrap());
        let s = t.elapsed().as_secs_f64();
        let rate = f64::from(n_p) * steps as f64 / s;
        eprintln!(
            "throughput: {threads} threads, {n_p} particles x {steps} steps in {s:.2} s = {:.2e} particle-steps/s",
            rate
        );
    }
}
