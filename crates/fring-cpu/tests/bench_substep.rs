//! ICR-004 test C8: the cost of `[substep]` on the band of the ICR table.
//!
//! `fring-bench` belongs to A5 and does not set the new keys, so the CPU rate
//! is measured here: 12,800 particles of the band 139,700 to 141,100 km with
//! the real moons, h = 356.25 s, one year (88,600 steps), `[substep]` off and
//! on, three runs each, alternating. The rate is the best (smallest time) of
//! the three. The limit is 5 % (proposed in ICR-004); the test prints the
//! raw times and reports the ratio.
//!
//! Run: `cargo test -p fring-cpu --release --test bench_substep -- --ignored --nocapture`

use fring_cpu::CpuBackend;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, RunConfig, System};

const STEPS: u64 = 88_600; // one year at 356.25 s
const N: u32 = 12_800;

fn config(sub: bool) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let mut t = text
        .replace("step_s = 360.0", "step_s = 356.25")
        .replace("duration_years = 1.0", "duration_years = 1.02");
    if sub {
        t += "\n[substep]\naccuracy = 0.1\nmax_level = 4\n";
    }
    RunConfig::from_toml_str(&t).unwrap()
}

#[test]
#[ignore = "release: about 3 minutes; ICR-004 test C8"]
fn c8_substep_cost_on_the_band() {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let all = fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .unwrap();
    let moons = build_ephemeris(&saturn, &all, 1.03 * 365.25 * 86_400.0, 1800.0).unwrap();
    let sys = System { saturn, moons };
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_700.0,
        max_semi_major_axis_km: 141_100.0,
    };
    let ini = InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, N, 20_261_108).unwrap(),
    };
    let mut times = [Vec::new(), Vec::new()];
    for _ in 0..3 {
        for (k, sub) in [false, true].into_iter().enumerate() {
            let mut b = CpuBackend::new(&config(sub), &sys, &ini).unwrap();
            b.set_moon_radii_km(&[43.09, 40.63]).unwrap();
            let t = std::time::Instant::now();
            b.step(STEPS).unwrap();
            let s = t.elapsed().as_secs_f64();
            eprintln!("C8 substep {}: {s:.3} s", if sub { "on " } else { "off" });
            times[k].push(s);
        }
    }
    let best = |v: &Vec<f64>| v.iter().cloned().fold(f64::MAX, f64::min);
    let (off, on) = (best(&times[0]), best(&times[1]));
    let rate = |t: f64| f64::from(N) * STEPS as f64 / t;
    eprintln!(
        "C8 best off {off:.3} s ({:.3e} particle-steps/s), best on {on:.3} s ({:.3e}); on/off time ratio {:.4}",
        rate(off),
        rate(on),
        on / off
    );
    assert!(
        on / off <= 1.05,
        "[substep] costs {:.1} % more",
        100.0 * (on / off - 1.0)
    );
}
