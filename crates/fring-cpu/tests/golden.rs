//! C1 of ICR-004: "off means off". Golden hashes of the state after 2000
//! steps, recorded on the code before E1-CPU changed `advance_chunk`. The new
//! keys `[loss]` and `[substep]` are absent here, so the bits must not change.
//!
//! The hashes are f64 bit patterns of this machine's libm (macOS). Change
//! them only with a decision of A0.

use fring_cpu::CpuBackend;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, build_ephemeris, generate_particles};
use fring_model::{Backend, InitialState, MoonEphemeris, RunConfig, System};

const STEPS: u64 = 2000;
const GOLDEN_NO_MOONS: u64 = 0x7ed4_9723_2da4_4f96;
const GOLDEN_MOONS: u64 = 0x14f5_b688_39fd_1b87;
const GOLDEN_MOONS_CHAOS: u64 = 0x0e57_b85b_5b10_bcf4;

fn system(moons: bool) -> System {
    let saturn =
        fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
            .unwrap();
    let all = fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
        .unwrap();
    let moons = if moons {
        build_ephemeris(&saturn, &all, 1.3e6, 1800.0).unwrap()
    } else {
        MoonEphemeris::none()
    };
    System { saturn, moons }
}

fn config(moons: bool, chaos: bool) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let mut t = text.replace("duration_years = 1.0", "duration_years = 0.04");
    if !moons {
        t = t.replace(r#"include = ["prometheus", "pandora"]"#, "include = []");
    }
    if chaos {
        t += "\n[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 100\n";
    }
    RunConfig::from_toml_str(&t).unwrap()
}

fn fnv(h: &mut u64, x: u64) {
    for b in x.to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn run_hash(moons: bool, chaos: bool) -> u64 {
    let sys = system(moons);
    let field = SaturnField::<f64>::from_params(&sys.saturn);
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_000.0,
        max_semi_major_axis_km: 141_800.0,
    };
    let ini = InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, 300, 20_261_108).unwrap(),
    };
    let mut b = CpuBackend::new(&config(moons, chaos), &sys, &ini).unwrap();
    b.step(777).unwrap();
    b.step(STEPS - 777).unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let p = &b.particles().unwrap().particles;
    for v in [
        &p.x_km, &p.y_km, &p.z_km, &p.vx_km_s, &p.vy_km_s, &p.vz_km_s,
    ] {
        v.iter().for_each(|x| fnv(&mut h, x.to_bits()));
    }
    if chaos {
        let r = b.chaos().unwrap().unwrap();
        for v in [&r.ftle_per_s, &r.log_growth, &r.megno_mean] {
            v.iter().for_each(|x| fnv(&mut h, x.to_bits()));
        }
        r.flags.iter().for_each(|&f| fnv(&mut h, u64::from(f)));
    }
    h
}

#[test]
fn off_means_off_golden_hashes() {
    let a = run_hash(false, false);
    let b = run_hash(true, false);
    let c = run_hash(true, true);
    eprintln!("hashes {a:#018x} {b:#018x} {c:#018x}");
    assert_eq!(a, GOLDEN_NO_MOONS, "no moons");
    assert_eq!(b, GOLDEN_MOONS, "moons");
    assert_eq!(c, GOLDEN_MOONS_CHAOS, "moons with chaos");
}
