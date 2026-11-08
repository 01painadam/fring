//! ICR-004 tests C2 (freeze, unbound) and C3 (moon-radius rule), with fixed
//! "moons" that make the geometry exact.

use fring_cpu::CpuBackend;
use fring_model::config::Distribution;
use fring_model::physics::{SaturnField, generate_particles, kepler_drift};
use fring_model::{
    Backend, BodyState, Error, InitialState, MoonEphemeris, MoonTrack, Particles, RunConfig,
    System, Vec3,
};

const H: f64 = 360.0; // step of the example config

fn saturn() -> fring_model::SaturnParams {
    fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap()
}

/// Moons that stand still at `positions`, with GM `gm`, for 1e8 s.
fn fixed_moons(positions: &[Vec3<f64>], gm: f64) -> MoonEphemeris {
    let state = |p: Vec3<f64>| BodyState {
        position_km: p,
        velocity_km_s: Vec3::zero(),
    };
    MoonEphemeris {
        start_s: 0.0,
        interval_s: 1.0e8,
        tracks: positions
            .iter()
            .enumerate()
            .map(|(i, p)| MoonTrack {
                name: ["prometheus", "pandora"][i].to_string(),
                gm_km3_s2: gm,
                samples: vec![state(*p), state(*p)],
                accelerations_km_s2: vec![Vec3::zero(); 2],
            })
            .collect(),
    }
}

fn system(moons: MoonEphemeris) -> System {
    System {
        saturn: saturn(),
        moons,
    }
}

fn config(extra: &str, n_moons: usize) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let mut t = text.replace("duration_years = 1.0", "duration_years = 0.04");
    if n_moons == 0 {
        t = t.replace(r#"include = ["prometheus", "pandora"]"#, "include = []");
    } else if n_moons == 1 {
        t = t.replace(r#"["prometheus", "pandora"]"#, r#"["prometheus"]"#);
    }
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

fn bits(p: &Particles, i: usize) -> [u64; 6] {
    [
        p.x_km[i].to_bits(),
        p.y_km[i].to_bits(),
        p.z_km[i].to_bits(),
        p.vx_km_s[i].to_bits(),
        p.vy_km_s[i].to_bits(),
        p.vz_km_s[i].to_bits(),
    ]
}

fn circular(r_km: f64, lon: f64, id: u32, p: &mut Particles) {
    let field = SaturnField::<f64>::from_params(&saturn());
    let v = fring_model::physics::circular_speed(&field, r_km);
    let (c, s) = (lon.cos(), lon.sin());
    p.push(
        id,
        Vec3::new(r_km * c, r_km * s, 0.0),
        Vec3::new(-v * s, v * c, 0.0),
    );
}

/// State of a particle after `steps` steps in Saturn's field only.
fn sync_state(init: &InitialState, steps: u64) -> Particles {
    let mut b = CpuBackend::new(&config("", 0), &system(MoonEphemeris::none()), init).unwrap();
    b.step(steps).unwrap();
    b.particles().unwrap().particles.clone()
}

/// Position at the kick of step `k` (after the first half drift of the step).
fn kick_position(init: &InitialState, i: usize, k: u64) -> Vec3<f64> {
    let p = sync_state(init, k);
    let gm = saturn().gm_km3_s2;
    kepler_drift(gm, p.position_km(i), p.velocity_km_s(i), 0.5 * H)
        .unwrap()
        .0
}

fn band(count: u32) -> InitialState {
    let field = SaturnField::<f64>::from_params(&saturn());
    let dist = Distribution::Band {
        min_semi_major_axis_km: 139_000.0,
        max_semi_major_axis_km: 141_800.0,
    };
    InitialState {
        time_s: 0.0,
        particles: generate_particles(&field, &dist, count, 20_261_108).unwrap(),
    }
}

/// C2: an unbound particle is frozen with reason 1 and the exact step; the
/// other particles are bit for bit those of a run without it.
#[test]
fn c2_freeze_unbound() {
    let init = band(24);
    let victim = 7usize;
    let k = 100u64; // before the first return of the orbit (148 steps)
    // A heavy fixed moon 0.01 km from the victim's position at kick `k`:
    // the kick is huge and the next drift fails.
    let q = kick_position(&init, victim, k) + Vec3::new(0.01, 0.0, 0.0);
    let sys = system(fixed_moons(&[q], 1.0e3));
    let cfg = config("[loss]\non_unbound = \"freeze\"\n", 1);

    // "error" mode: the existing error.
    let mut e = CpuBackend::new(&config("", 1), &sys, &init).unwrap();
    let err = e.step(1000).unwrap_err();
    assert!(
        matches!(&err, Error::Backend(m) if m.contains("has an unbound or invalid orbit")),
        "{err:?}"
    );
    // [loss] with the default keys takes the old path and the same error.
    let mut e = CpuBackend::new(&config("[loss]\n", 1), &sys, &init).unwrap();
    let err2 = e.step(1000).unwrap_err();
    assert_eq!(format!("{err:?}"), format!("{err2:?}"));

    let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
    assert!(b.dropped().unwrap().is_some());
    b.step(100).unwrap();
    let r0 = b.dropped().unwrap().unwrap();
    assert!(r0.reason.iter().all(|&r| r == 0) && r0.n_unbound == 0);
    assert!(r0.drop_step.iter().all(|&s| s == u64::MAX));
    let before = b.particles().unwrap().particles.clone();
    b.step(900).unwrap();
    let d = b.dropped().unwrap().unwrap();
    assert_eq!(d.reason[victim], 1);
    assert_eq!(d.drop_step[victim], k);
    assert_eq!(d.n_unbound, 1);
    assert_eq!(d.n_inside_moon, vec![0]);
    assert_eq!(d.reason.iter().filter(|&&r| r != 0).count(), 1);
    assert_eq!(d.time_s, 1000.0 * H);
    let after = b.particles().unwrap().particles.clone();
    // The state of the lost particle is that of the start of its block.
    assert_eq!(bits(&after, victim), bits(&before, victim));
    b.step(100).unwrap();
    let later = b.particles().unwrap().particles.clone();
    assert_eq!(bits(&later, victim), bits(&before, victim));
    assert_eq!(b.dropped().unwrap().unwrap().drop_step[victim], k);

    // The others: a run without the victim (old path) gives the same bits.
    let mut rest = Particles::with_capacity(23);
    for i in (0..24).filter(|&i| i != victim) {
        rest.push(
            init.particles.id[i],
            init.particles.position_km(i),
            init.particles.velocity_km_s(i),
        );
    }
    let init_rest = InitialState {
        time_s: 0.0,
        particles: rest,
    };
    let mut r = CpuBackend::new(&config("", 1), &sys, &init_rest).unwrap();
    for n in [100, 900, 100] {
        r.step(n).unwrap();
    }
    let rp = r.particles().unwrap().particles.clone();
    for (j, i) in (0..24).filter(|&i| i != victim).enumerate() {
        assert_eq!(bits(&later, i), bits(&rp, j), "particle {i}");
        assert_eq!(d.reason[i], 0);
    }
}

/// C2 for the no-moon case: freeze with an unbound start.
#[test]
fn c2_unbound_at_the_start_is_step_zero_without_moons() {
    let mut p = Particles::with_capacity(3);
    circular(140_000.0, 0.0, 10, &mut p);
    p.push(
        11,
        Vec3::new(140_000.0, 0.0, 0.0),
        Vec3::new(0.0, 40.0, 0.0), // above escape speed
    );
    circular(140_500.0, 1.0, 12, &mut p);
    let init = InitialState {
        time_s: 0.0,
        particles: p,
    };
    let sys = system(MoonEphemeris::none());
    let mut b =
        CpuBackend::new(&config("[loss]\non_unbound = \"freeze\"\n", 0), &sys, &init).unwrap();
    b.step(50).unwrap();
    b.step(50).unwrap();
    let d = b.dropped().unwrap().unwrap();
    assert_eq!(d.reason, vec![0, 1, 0]);
    assert_eq!(d.drop_step, vec![u64::MAX, 0, u64::MAX]);
    assert!(d.n_inside_moon.is_empty());
    // The two live particles equal an old-path run of themselves.
    let mut live = Particles::with_capacity(2);
    for i in [0usize, 2] {
        live.push(
            init.particles.id[i],
            init.particles.position_km(i),
            init.particles.velocity_km_s(i),
        );
    }
    let mut r = CpuBackend::new(
        &config("", 0),
        &sys,
        &InitialState {
            time_s: 0.0,
            particles: live,
        },
    )
    .unwrap();
    r.step(50).unwrap();
    r.step(50).unwrap();
    let rp = r.particles().unwrap().particles.clone();
    let got = b.particles().unwrap().particles.clone();
    assert_eq!(bits(&got, 0), bits(&rp, 0));
    assert_eq!(bits(&got, 2), bits(&rp, 1));
}

/// A fixed moon `offset` km from the chord of the drift after kick `k`
/// (perpendicular to the chord, in the orbital plane).
fn moon_beside_chord(init: &InitialState, i: usize, k: u64, offset_km: f64) -> Vec3<f64> {
    let a = kick_position(init, i, k);
    let b = kick_position(init, i, k + 1);
    let chord = (b - a) / (b - a).norm();
    let w = chord.cross(Vec3::new(0.0, 0.0, 1.0));
    let w = w / w.norm();
    (a + b) * 0.5 + w * offset_km
}

fn two_particles() -> InitialState {
    let mut p = Particles::with_capacity(2);
    circular(140_000.0, 0.0, 0, &mut p);
    circular(140_100.0, 1.5, 1, &mut p);
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

const RADII: [f64; 2] = [43.09, 40.63];

fn rule_run(offsets: [f64; 2], steps: &[u64]) -> fring_model::DropReport {
    let init = two_particles();
    let (ka, kb) = (60u64, 120u64); // both before the first return (148 steps)
    let ma = moon_beside_chord(&init, 0, ka, offsets[0]);
    let mb = moon_beside_chord(&init, 1, kb, offsets[1]);
    let sys = system(fixed_moons(&[ma, mb], 1.0e-9));
    let cfg = config(
        "[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n",
        2,
    );
    let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
    // Missing radii are a config error at the first step.
    assert!(matches!(b.step(1), Err(Error::Config(_))));
    let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
    assert!(matches!(b.set_moon_radii_km(&[1.0]), Err(Error::Config(_))));
    assert!(matches!(
        b.set_moon_radii_km(&[1.0, f64::NAN]),
        Err(Error::Config(_))
    ));
    b.set_moon_radii_km(&RADII).unwrap();
    for &n in steps {
        b.step(n).unwrap();
    }
    b.dropped().unwrap().unwrap()
}

/// C3: a chord that passes inside the radius between two kicks (both ends far
/// outside) is dropped in the right step, with the right reason; radius + 1 km
/// is not dropped. Both moons. Block splits do not matter.
#[test]
fn c3_radius_rule() {
    for steps in [&[1000u64][..], &[59, 1, 40, 900]] {
        let hit = rule_run([RADII[0] - 1.0, RADII[1] - 1.0], steps);
        assert_eq!(hit.reason, vec![2, 3], "{steps:?}");
        assert_eq!(hit.drop_step, vec![60, 120], "{steps:?}");
        assert_eq!(hit.n_inside_moon, vec![1, 1]);
        assert_eq!(hit.n_unbound, 0);
        // Only up to step 140: a particle returns after one orbit (148 steps)
        // and may come closer on the second pass.
        let short: Vec<u64> = if steps.len() == 1 {
            vec![140]
        } else {
            vec![59, 1, 80]
        };
        let miss = rule_run([RADII[0] + 1.0, RADII[1] + 1.0], &short);
        assert_eq!(miss.reason, vec![0, 0], "{steps:?} {miss:?}");
        assert_eq!(miss.drop_step, vec![u64::MAX; 2]);
    }
}

/// C7 (flags): a lost particle gets `ChaosReport.flags` bit 1 and its
/// accumulators stop; the others keep adding.
#[test]
fn c7_dropped_particle_is_flagged_and_its_accumulators_stop() {
    let init = band(24);
    let victim = 7usize;
    let q = kick_position(&init, victim, 100) + Vec3::new(0.01, 0.0, 0.0);
    let sys = system(fixed_moons(&[q], 1.0e3));
    let cfg = config(
        "[loss]\non_unbound = \"freeze\"\n[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1e-3\nrenorm_steps = 50\n",
        1,
    );
    let mut b = CpuBackend::new(&cfg, &sys, &init).unwrap();
    b.step(100).unwrap();
    let c1 = b.chaos().unwrap().unwrap();
    assert!(c1.flags.iter().all(|&f| f & 2 == 0));
    assert_eq!(c1.n_renorm, 2);
    b.step(400).unwrap();
    let c2 = b.chaos().unwrap().unwrap();
    assert_eq!(c2.n_renorm, 10);
    for i in 0..24 {
        if i == victim {
            assert_eq!(c2.flags[i] & 2, 2);
            // Dropped in step 100, the block [100, 150): only the two
            // renormalisations before it were added.
            assert_eq!(c2.log_growth[i].to_bits(), c1.log_growth[i].to_bits());
            assert_eq!(c2.megno_mean[i].to_bits(), c1.megno_mean[i].to_bits());
        } else {
            assert_eq!(c2.flags[i] & 2, 0);
            assert_ne!(c2.log_growth[i].to_bits(), c1.log_growth[i].to_bits());
        }
    }
}
