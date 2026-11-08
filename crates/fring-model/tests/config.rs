//! Tests of the run config schema (`fring_model::config`).

use fring_model::Error;
use fring_model::config::{
    BackendKind, Collisions, Distribution, Domain, MoonletVelocity, Restitution, RunConfig,
};

const EXAMPLE: &str = include_str!("data/example-run.toml");

/// The `[collisions]` example of ICR-009, without `every_steps` (default 1).
/// It replaces `model = "none"` in `EXAMPLE`.
const STOCHASTIC: &str = "model = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 8.0e8\n\
                          cell_radial_km = 0.05\ncell_azimuthal_km = 10.0\ncell_vertical_km = 0.04\n\
                          [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5";

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
        ("model = \"none\"", STOCHASTIC),
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
    assert_eq!(
        seen[3].collisions,
        Collisions::Stochastic {
            particle_radius_km: 1.0e-5,
            weight: 8.0e8,
            cell_radial_km: 0.05,
            cell_azimuthal_km: 10.0,
            cell_vertical_km: 0.04,
            every_steps: 1,
            restitution: Restitution::Constant { epsilon: 0.5 },
        }
    );
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

/// ICR-009: three cell sizes, each finite and > 0; `every_steps` ≥ 1 with
/// default 1; the old single `cell_size_km` is an unknown field.
#[test]
fn stochastic_cells_and_cadence_are_checked() {
    let stochastic = |from: &str, to: &str| {
        assert!(
            STOCHASTIC.contains(from),
            "test setup: {from:?} not in STOCHASTIC"
        );
        EXAMPLE.replacen("model = \"none\"", &STOCHASTIC.replacen(from, to, 1), 1)
    };

    let every = "cell_vertical_km = 0.04\nevery_steps = 3";
    let c = RunConfig::from_toml_str(&stochastic("cell_vertical_km = 0.04", every)).unwrap();
    assert!(matches!(
        c.collisions,
        Collisions::Stochastic { every_steps: 3, .. }
    ));
    let again = RunConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
    assert_eq!(again, c);

    let cases = [
        (
            "cell_radial_km = 0.05",
            "cell_radial_km = 0.0",
            "cell_radial_km",
        ),
        (
            "cell_azimuthal_km = 10.0",
            "cell_azimuthal_km = -10.0",
            "cell_azimuthal_km",
        ),
        (
            "cell_vertical_km = 0.04",
            "cell_vertical_km = inf",
            "cell_vertical_km",
        ),
        (
            "cell_vertical_km = 0.04",
            "cell_vertical_km = 0.04\nevery_steps = 0",
            "every_steps",
        ),
        (
            "cell_radial_km = 0.05",
            "cell_radial_km = 0.05\ncell_size_km = 1.0",
            "cell_size_km",
        ),
    ];
    for (from, to, needle) in cases {
        assert_config_error(RunConfig::from_toml_str(&stochastic(from, to)), needle);
    }
}

#[test]
fn seed_above_i64_max_is_rejected_by_validate() {
    let mut c = example();
    c.run.seed = u64::MAX;
    assert_config_error(c.validate(), "run.seed");
}

#[test]
fn moonlet_parses_round_trips_and_is_checked() {
    const CORE: &str = "kind = \"core\"\nsemi_major_axis_km = 140220.0\nsigma_semi_major_axis_km = 5.0\n\
                        eccentricity = 0.0024\napse_longitude_deg = 0.0";
    assert!(
        EXAMPLE.contains(CORE),
        "test setup: core block not in example"
    );
    let with = |radius: &str, velocity: &str| {
        EXAMPLE.replacen(
            CORE,
            &format!(
                "kind = \"moonlet\"\nsemi_major_axis_km = 140222.43\nlongitude_deg = 180.0\n\
                 packing_radius_km = {radius}\nvelocity = \"{velocity}\""
            ),
            1,
        )
    };
    for (name, mode) in [
        ("synchronous", MoonletVelocity::Synchronous),
        ("centre", MoonletVelocity::Centre),
        ("circular", MoonletVelocity::Circular),
    ] {
        let c = RunConfig::from_toml_str(&with("20.0", name)).unwrap();
        assert_eq!(
            c.particles.distribution,
            Distribution::Moonlet {
                semi_major_axis_km: 140_222.43,
                longitude_deg: 180.0,
                packing_radius_km: 20.0,
                velocity: mode,
            }
        );
        let again = RunConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
        assert_eq!(again, c);
    }
    assert_config_error(
        RunConfig::from_toml_str(&with("0.0", "centre")),
        "packing_radius_km",
    );
    assert_config_error(
        RunConfig::from_toml_str(&with("140222.43", "centre")),
        "must be less than",
    );
    assert!(RunConfig::from_toml_str(&with("20.0", "spin")).is_err());
}

// ICR-004: [loss] and [substep].

fn with(extra: &str) -> fring_model::Result<RunConfig> {
    RunConfig::from_toml_str(&format!("{EXAMPLE}\n{extra}"))
}

#[test]
fn loss_and_substep_absent_means_off() {
    let c = example();
    assert!(c.loss.is_none() && c.substep.is_none());
    assert!(!c.loss_or_substep_active());
    assert!(!c.freeze_unbound() && !c.moon_radius_rule());
}

#[test]
fn loss_and_substep_parse_with_defaults_and_round_trip() {
    let c = with("[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n[substep]\n").unwrap();
    assert!(c.freeze_unbound() && c.moon_radius_rule() && c.loss_or_substep_active());
    let s = c.substep.unwrap();
    assert_eq!((s.accuracy, s.max_level), (0.1, 4));
    let back = RunConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
    assert_eq!(back, c);
    // An empty [loss] is "error" and "off": it changes nothing.
    let c = with("[loss]\n").unwrap();
    assert!(!c.loss_or_substep_active());
    let c = with("[loss]\non_unbound = \"error\"\nmoon_radius = \"off\"\n").unwrap();
    assert!(!c.loss_or_substep_active());
    let c = with("[loss]\non_unbound = \"freeze\"\n").unwrap();
    assert!(c.loss_or_substep_active() && !c.moon_radius_rule());
}

#[test]
fn loss_and_substep_are_checked() {
    assert_config_error(
        with("[loss]\nmoon_radius = \"params\"\n"),
        "needs loss.on_unbound",
    );
    assert_config_error(
        with("[loss]\non_unbound = \"freeze\"\nmoon_radius = \"maybe\"\n"),
        "moon_radius",
    );
    assert_config_error(with("[loss]\nwhat = 1\n"), "what");
    assert_config_error(with("[substep]\naccuracy = 0.06\n"), "substep.accuracy");
    assert_config_error(with("[substep]\naccuracy = 1.01\n"), "substep.accuracy");
    assert_config_error(with("[substep]\naccuracy = nan\n"), "substep.accuracy");
    assert_config_error(with("[substep]\nmax_level = 7\n"), "substep.max_level");
    assert!(with("[substep]\naccuracy = 0.07\nmax_level = 6\n").is_ok());
    assert!(with("[substep]\naccuracy = 1.0\nmax_level = 0\n").is_ok());
    // Without moons.
    let no_moons = EXAMPLE.replace(r#"include = ["prometheus", "pandora"]"#, "include = []");
    let r = RunConfig::from_toml_str(&format!("{no_moons}\n[substep]\n"));
    assert_config_error(r, "at least one moon");
    let r = RunConfig::from_toml_str(&format!(
        "{no_moons}\n[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n"
    ));
    assert_config_error(r, "at least one moon");
    // freeze without the radius rule is fine without moons.
    let r = RunConfig::from_toml_str(&format!("{no_moons}\n[loss]\non_unbound = \"freeze\"\n"));
    assert!(r.is_ok());
    // More than 4 moons.
    let five = EXAMPLE.replace(
        r#"include = ["prometheus", "pandora"]"#,
        r#"include = ["a", "b", "c", "d", "e"]"#,
    );
    assert_config_error(
        RunConfig::from_toml_str(&format!("{five}\n[substep]\n")),
        "at most 4 moons",
    );
}
