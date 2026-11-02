//! The CPU chaos indicator (ICR-003, ADR 0008): thread independence, step
//! splitting, and the two analytic checks (regular orbit: MEGNO about 2).

use fring_cpu::CpuBackend;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, ChaosReport, InitialState, MoonEphemeris, RunConfig, System};

const K: u32 = 100;

fn system(moons: bool, duration_s: f64) -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let all = fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .unwrap();
    let moons = if moons {
        build_ephemeris(&saturn, &all, duration_s, 1800.0).unwrap()
    } else {
        MoonEphemeris::none()
    };
    System { saturn, moons }
}

fn config(chaos: bool, moons: bool) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let mut t = text.replace("duration_years = 1.0", "duration_years = 0.1");
    if !moons {
        t = t.replace(r#"include = ["prometheus", "pandora"]"#, "include = []");
    }
    if chaos {
        t += &format!(
            "\n[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = {K}\n"
        );
    }
    RunConfig::from_toml_str(&t).unwrap()
}

fn band(sys: &System, count: u32, lo: f64, hi: f64) -> InitialState {
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: lo,
        max_semi_major_axis_km: hi,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 11).unwrap(),
    }
}

fn run(
    cfg: &RunConfig,
    sys: &System,
    ini: &InitialState,
    calls: &[u64],
    threads: usize,
) -> ChaosReport {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut b = CpuBackend::new(cfg, sys, ini).unwrap();
        for &n in calls {
            b.step(n).unwrap();
        }
        b.chaos().unwrap().unwrap()
    })
}

fn bits(r: &ChaosReport) -> Vec<u64> {
    [&r.ftle_per_s, &r.log_growth, &r.megno_mean]
        .iter()
        .flat_map(|v| v.iter().map(|x| x.to_bits()))
        .collect()
}

#[test]
fn no_table_no_indicator() {
    let sys = system(false, 1e6);
    let ini = band(&sys, 4, 139_000.0, 141_000.0);
    let mut b = CpuBackend::new(&config(false, false), &sys, &ini).unwrap();
    b.step(10).unwrap();
    assert!(b.chaos().unwrap().is_none());
}

#[test]
fn threads_and_call_splits_do_not_change_the_bits() {
    let sys = system(true, 4e6);
    let ini = band(&sys, 300, 139_600.0, 140_600.0);
    let cfg = config(true, true);
    let a = run(&cfg, &sys, &ini, &[3000], 1);
    let b = run(&cfg, &sys, &ini, &[3000], 8);
    assert_eq!(bits(&a), bits(&b), "thread count changed the result");
    // Calls that end at multiples of K give the same blocks.
    let c = run(&cfg, &sys, &ini, &[1000, 500, 1500], 3);
    assert_eq!(bits(&a), bits(&c), "call splitting changed the result");
    assert_eq!(a.n_renorm, 30);
}

#[test]
fn regular_orbit_has_megno_near_two() {
    // Circular orbits in Saturn's field only: integrable. 200 orbits.
    let sys = system(false, 1e6);
    let ini = band(&sys, 50, 139_000.0, 141_000.0);
    let cfg = config(true, false);
    let r = run(&cfg, &sys, &ini, &[30_000], 4);
    let mut worst = 0.0f64;
    for &y in &r.megno_mean {
        worst = worst.max((y - 2.0).abs());
    }
    assert!(worst < 0.5, "worst |<Y> - 2| = {worst}");
    assert!(r.flags.iter().all(|&f| f == 0));
}
