//! Tests of the run config schema (`fring_model::config`).

use fring_model::Error;
use fring_model::config::{BackendKind, Collisions, Distribution, Domain, Restitution, RunConfig};

const EXAMPLE: &str = include_str!("data/example-run.toml");

fn example() -> RunConfig {
    RunConfig::from_toml_str(EXAMPLE).expect("example config must be valid")
}

/// Asserts that `result` is a config error whose message contains `needle`.
fn assert_config_error<T: std::fmt::Debug>(result: fring_model::Result<T>, needle: &str) {
    match result {
        Err(Error::Config(msg)) => assert!(
            msg.contains(needle),
            "message {msg:?} does not contain {needle:?}"
        ),
        other => panic!("expected a config error with {needle:?}, got {other:?}"),
    }
}

#[test]
fn example_parses_with_expected_values() {
    let c = example();
    assert_eq!(c.run.name, "example-core");
    assert_eq!(c.run.backend, BackendKind::Cpu);
    assert_eq!(c.run.seed, 20_261_101);
    assert_eq!(c.run.params_dir, "params");
    assert_eq!(c.domain, Domain::FullRing);
    assert_eq!(c.moons.include, ["prometheus", "pandora"]);
    assert_eq!(c.particles.count, 10_000);
    assert!(matches!(
        c.particles.distribution,
        Distribution::Core { eccentricity, .. } if eccentricity == 0.0024
    ));
    assert_eq!(c.collisions, Collisions::None);
    assert!(!c.output.velocities);
}

#[test]
fn step_count_covers_the_duration() {
    let c = example();
    // 1 Julian year = 31,557,600 s; 31,557,600 / 360 = 87,660 exactly.
    assert_eq!(c.time.duration_s(), 31_557_600.0);
    assert_eq!(c.time.n_steps(), 87_660);
    // A step that does not divide the duration: round up.
    let mut t = c.time;
    t.step_s = 7.0e6;
    assert_eq!(t.n_steps(), 5); // 31,557,600 / 7e6 = 4.5
}

#[test]
fn toml_round_trip() {
    let c = example();
    let text = c.to_toml_string().unwrap();
    assert_eq!(RunConfig::from_toml_str(&text).unwrap(), c);
}

#[test]
fn defaults_for_optional_fields() {
    let text = EXAMPLE
        .replace("params_dir = \"params\"", "")
        .replace("velocities = false", "");
    let c = RunConfig::from_toml_str(&text).unwrap();
    assert_eq!(c.run.params_dir, "params");
    assert!(!c.output.velocities);
}

#[test]
fn every_variant_parses_and_round_trips() {
    let cases = [
        (
            "kind = \"full_ring\"",
            "kind = \"arc\"\nwidth_deg = 10.0\ncentre_longitude_deg = 45.0",
        ),
        (
            "kind = \"core\"\nsemi_major_axis_km = 140220.0\nsigma_semi_major_axis_km = 5.0\n\
             eccentricity = 0.0024\napse_longitude_deg = 0.0",
            "kind = \"band\"\nmin_semi_major_axis_km = 139500.0\nmax_semi_major_axis_km = 141500.0",
        ),
        (
            "kind = \"core\"\nsemi_major_axis_km = 140220.0\nsigma_semi_major_axis_km = 5.0\n\
             eccentricity = 0.0024\napse_longitude_deg = 0.0",
            "kind = \"clump\"\nsemi_major_axis_km = 140220.0\nlongitude_deg = 90.0\nsigma_km = 2.0",
        ),
        (
            "model = \"none\"",
            "model = \"stochastic\"\nparticle_radius_km = 0.001\nweight = 1.0e9\ncell_size_km = 1.0\n\
             [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5",
        ),
        (
            "model = \"none\"",
            "model = \"hard_sphere\"\nparticle_radius_km = 0.001\n\
             [collisions.restitution]\nlaw = \"power_law\"\nreference_speed_km_s = 7.7e-8\nexponent = 0.234",
        ),
    ];
    let mut seen = Vec::new();
    for (from, to) in cases {
        assert!(
            EXAMPLE.contains(from),
            "test setup: {from:?} not in example"
        );
        let c = RunConfig::from_toml_str(&EXAMPLE.replacen(from, to, 1))
            .unwrap_or_else(|e| panic!("{to:?}: {e}"));
        let again = RunConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
        assert_eq!(again, c);
        seen.push(c);
    }
    assert!(matches!(seen[0].domain, Domain::Arc { width_deg, .. } if width_deg == 10.0));
    assert!(matches!(
        seen[1].particles.distribution,
        Distribution::Band { .. }
    ));
    assert!(matches!(
        seen[2].particles.distribution,
        Distribution::Clump { .. }
    ));
    assert!(matches!(
        seen[3].collisions,
        Collisions::Stochastic {
            restitution: Restitution::Constant { epsilon },
            ..
        } if epsilon == 0.5
    ));
    assert!(matches!(
        seen[4].collisions,
        Collisions::HardSphere {
            restitution: Restitution::PowerLaw { .. },
            ..
        }
    ));
}

#[test]
fn eccentric_band_parses_round_trips_and_is_checked() {
    const CORE: &str = "kind = \"core\"\nsemi_major_axis_km = 140220.0\nsigma_semi_major_axis_km = 5.0\n\
                        eccentricity = 0.0024\napse_longitude_deg = 0.0";
    assert!(
        EXAMPLE.contains(CORE),
        "test setup: core block not in example"
    );
    let with = |min: &str, e: &str, apse: &str| {
        EXAMPLE.replacen(
            CORE,
            &format!(
                "kind = \"eccentric_band\"\nmin_semi_major_axis_km = {min}\n\
                 max_semi_major_axis_km = 140700.0\neccentricity = {e}\napse_longitude_deg = {apse}"
            ),
            1,
        )
    };
    let c = RunConfig::from_toml_str(&with("140150.0", "0.0024", "353.78")).unwrap();
    assert_eq!(
        c.particles.distribution,
        Distribution::EccentricBand {
            min_semi_major_axis_km: 140_150.0,
            max_semi_major_axis_km: 140_700.0,
            eccentricity: 0.0024,
            apse_longitude_deg: 353.78,
        }
    );
    let again = RunConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
    assert_eq!(again, c);
    // e = 0 is allowed (the circular control of ICR-007).
    assert!(RunConfig::from_toml_str(&with("140150.0", "0.0", "0.0")).is_ok());
    assert_config_error(
        RunConfig::from_toml_str(&with("140150.0", "1.0", "0.0")),
        "eccentricity",
    );
    assert_config_error(
        RunConfig::from_toml_str(&with("140150.0", "-0.1", "0.0")),
        "eccentricity",
    );
    assert_config_error(
        RunConfig::from_toml_str(&with("140150.0", "0.0024", "nan")),
        "apse_longitude_deg",
    );
    assert_config_error(
        RunConfig::from_toml_str(&with("140700.0", "0.0024", "0.0")),
        "min_semi_major_axis_km",
    );
}

#[test]
fn unknown_field_is_an_error() {
    let text = EXAMPLE.replace("step_s = 360.0", "step_sec = 360.0");
    assert_config_error(RunConfig::from_toml_str(&text), "step_sec");
}

#[test]
fn unknown_field_in_tagged_section_is_an_error() {
    let text = EXAMPLE.replace("eccentricity = 0.0024", "ecc = 0.0024");
    assert_config_error(RunConfig::from_toml_str(&text), "ecc");
}

#[test]
fn unknown_section_is_an_error() {
    let text = format!("{EXAMPLE}\n[extra]\nx = 1\n");
    assert_config_error(RunConfig::from_toml_str(&text), "extra");
}

#[test]
fn missing_field_is_an_error() {
    let text = EXAMPLE.replace("count = 10000", "");
    assert_config_error(RunConfig::from_toml_str(&text), "count");
}

#[test]
fn unknown_backend_is_an_error() {
    let text = EXAMPLE.replace("backend = \"cpu\"", "backend = \"cuda\"");
    assert_config_error(RunConfig::from_toml_str(&text), "cuda");
}

#[test]
fn out_of_range_values_are_errors() {
    let cases = [
        ("step_s = 360.0", "step_s = -1.0", "time.step_s"),
        (
            "duration_years = 1.0",
            "duration_years = 0.0",
            "time.duration_years",
        ),
        ("name = \"example-core\"", "name = \"bad/name\"", "run.name"),
        ("name = \"example-core\"", "name = \"\"", "run.name"),
        ("count = 10000", "count = 0", "particles.count"),
        (
            "eccentricity = 0.0024",
            "eccentricity = 1.0",
            "eccentricity",
        ),
        (
            "sigma_semi_major_axis_km = 5.0",
            "sigma_semi_major_axis_km = -5.0",
            "sigma_semi_major_axis_km",
        ),
        (
            "include = [\"prometheus\", \"pandora\"]",
            "include = [\"pandora\", \"pandora\"]",
            "twice",
        ),
        (
            "frame_every_steps = 1460",
            "frame_every_steps = 0",
            "frame_every_steps",
        ),
        (
            "reference_semi_major_axis_km = 140220.0",
            "reference_semi_major_axis_km = nan",
            "reference_semi_major_axis_km",
        ),
        (
            "kind = \"full_ring\"",
            "kind = \"arc\"\nwidth_deg = 400.0\ncentre_longitude_deg = 0.0",
            "width_deg",
        ),
        (
            "model = \"none\"",
            "model = \"hard_sphere\"\nparticle_radius_km = 0.001\n\
             [collisions.restitution]\nlaw = \"constant\"\nepsilon = 1.5",
            "epsilon",
        ),
        (
            "kind = \"core\"\nsemi_major_axis_km = 140220.0\nsigma_semi_major_axis_km = 5.0\n\
             eccentricity = 0.0024\napse_longitude_deg = 0.0",
            "kind = \"band\"\nmin_semi_major_axis_km = 141500.0\nmax_semi_major_axis_km = 139500.0",
            "min_semi_major_axis_km",
        ),
    ];
    for (from, to, needle) in cases {
        assert!(
            EXAMPLE.contains(from),
            "test setup: {from:?} not in example"
        );
        assert_config_error(
            RunConfig::from_toml_str(&EXAMPLE.replacen(from, to, 1)),
            needle,
        );
    }
}

#[test]
fn seed_above_i64_max_is_rejected_by_validate() {
    let mut c = example();
    c.run.seed = u64::MAX;
    assert_config_error(c.validate(), "run.seed");
}
