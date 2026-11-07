//! Tests of `fring-run`: small runs in debug builds, and the acceptance run
//! (`#[ignore]`, release: `cargo test -p fring-run --release -- --ignored`).
//! Every test removes its temporary directory, also when it fails.

use fring_io::{RunDir, read_stats_csv_report};
use fring_model::RunConfig;
use fring_model::Vec3;
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::{SaturnField, generate_particles, rotating_to_inertial};
use fring_run::cli::Args;
use fring_run::{RunError, run};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

const CONFIG: &str = include_str!("../configs/acceptance-1y-1e4.toml");
static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A unique temporary directory that is removed on drop.
struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("fring-run-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Tmp(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn params_dir() -> String {
    format!("{}/../../params", env!("CARGO_MANIFEST_DIR"))
}

/// Replaces `from` by `to` and checks that `from` exists (mistakes.md M10).
fn replace(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "config has no {from:?}");
    text.replacen(from, to, 1)
}

/// The acceptance config made small: 200 particles, 0.02 years (about 7 days),
/// frames every 600 steps and stats every 500 steps. Neither cadence divides
/// the step count, so the last step must add a frame and a stats row.
fn small_config() -> String {
    let t = replace(
        CONFIG,
        "params_dir = \"params\"",
        &format!("params_dir = {:?}", params_dir()),
    );
    let t = replace(&t, "duration_years = 1.0", "duration_years = 0.02");
    let t = replace(&t, "count = 10000", "count = 200");
    let t = replace(&t, "frame_every_steps = 1460", "frame_every_steps = 600");
    replace(&t, "stats_every_steps = 146", "stats_every_steps = 500")
}

fn run_config(
    tmp: &Tmp,
    text: &str,
    name: &str,
    quiet: bool,
) -> Result<(RunDir, fring_run::Summary), RunError> {
    let cfg = tmp.path().join(format!("{name}.toml"));
    std::fs::write(&cfg, text).unwrap();
    let out = tmp.path().join(name);
    let summary = run(&Args {
        config: cfg,
        out: Some(out.clone()),
        quiet,
    })?;
    Ok((RunDir::new(out), summary))
}

fn expected_frame_steps(n: u64, every: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..=n).step_by(every as usize).collect();
    if v.last() != Some(&n) {
        v.push(n);
    }
    v
}

#[test]
fn small_run_with_moons_writes_a_complete_directory() {
    let tmp = Tmp::new();
    let text = small_config();
    let config = RunConfig::from_toml_str(&text).unwrap();
    let n = config.time.n_steps();
    let (dir, summary) = run_config(&tmp, &text, "a", true).unwrap();

    // config.toml is a byte-for-byte copy.
    assert_eq!(dir.read_config_copy().unwrap(), text.as_bytes());

    // Frames: 0, 600, 1200, then the last step.
    let steps = expected_frame_steps(n, 600);
    assert_eq!(summary.frames as usize, steps.len());
    assert_eq!(
        dir.list_frames().unwrap(),
        (0..steps.len() as u64).collect::<Vec<_>>()
    );
    let n0 = fring_model::physics::reference_mean_motion(
        &SaturnField::<f64>::from_params(
            &saturn_from_toml_str(
                &std::fs::read_to_string(format!("{}/saturn.toml", params_dir())).unwrap(),
            )
            .unwrap(),
        ),
        140_220.0,
    );
    for (i, &s) in steps.iter().enumerate() {
        let f = dir.read_frame_file(i as u64).unwrap();
        assert_eq!(f.step, s);
        assert_eq!(f.time_s, s as f64 * 360.0);
        assert_eq!(f.len(), 200);
        assert_eq!((f.a0_km, f.t_ref_s), (140_220.0, 0.0));
        assert_eq!(f.n0_rad_s, n0);
        assert!(f.velocities.is_none());
    }

    // Frame 0, rotated back, is the initial state (f32 rounding only).
    let saturn = saturn_from_toml_str(
        &std::fs::read_to_string(format!("{}/saturn.toml", params_dir())).unwrap(),
    )
    .unwrap();
    let init = generate_particles(
        &SaturnField::<f64>::from_params(&saturn),
        &config.particles.distribution,
        200,
        config.run.seed,
    )
    .unwrap();
    let f0 = dir.read_frame_file(0).unwrap();
    for i in 0..200 {
        let (p, _) = rotating_to_inertial::<f64>(
            Vec3::new(f0.x_km[i] as f64, f0.y_km[i] as f64, f0.z_km[i] as f64),
            Vec3::new(0.0, 0.0, 0.0),
            f0.n0_rad_s,
            f0.time_s,
        );
        let d = p - init.position_km(i);
        assert!(
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt() < 0.1,
            "particle {i}"
        );
        assert_eq!(f0.id[i], i as u32);
    }

    // moons.bin: two moons, one record per frame, same times.
    let moons = dir.read_moons_file().unwrap();
    assert_eq!(moons.names, ["prometheus", "pandora"]);
    let times: Vec<f64> = moons.records.iter().map(|r| r.time_s).collect();
    assert_eq!(
        times,
        steps.iter().map(|&s| s as f64 * 360.0).collect::<Vec<_>>()
    );
    assert!(moons.records.iter().all(|r| r.bodies.len() == 2));

    // stats.csv: steps 0, 500, 1000, 1500, last. Complete file, no dropped line.
    let report =
        read_stats_csv_report(&mut std::fs::File::open(dir.stats_path()).unwrap()).unwrap();
    assert!(!report.dropped_partial);
    let rows: Vec<u64> = report.rows.iter().map(|r| r.steps_total).collect();
    assert_eq!(rows, vec![0, 500, 1000, 1500, n]);
    assert_eq!(summary.stats_rows, 5);
    assert!(
        report.rows.iter().all(|r| r.max_rel_energy_error.is_none()),
        "moons: no energy check"
    );

    // meta.json.
    let meta = dir.read_meta_file().unwrap();
    assert_eq!(meta.code_version, env!("CARGO_PKG_VERSION"));
    assert!(!meta.git_hash.is_empty() && !meta.machine.is_empty());
    assert!(meta.wall_time_s > 0.0);
}

#[test]
fn small_run_without_moons_is_deterministic() {
    let tmp = Tmp::new();
    let text = replace(
        &small_config(),
        "include = [\"prometheus\", \"pandora\"]",
        "include = []",
    );
    let text = replace(&text, "velocities = false", "velocities = true");
    let (a, _) = run_config(&tmp, &text, "a", true).unwrap();
    let (b, _) = run_config(&tmp, &text, "b", true).unwrap();
    let moons = a.read_moons_file().unwrap();
    assert!(moons.names.is_empty() && moons.records.iter().all(|r| r.bodies.is_empty()));
    for i in a.list_frames().unwrap() {
        let (fa, fb) = (a.read_frame_file(i).unwrap(), b.read_frame_file(i).unwrap());
        assert!(fa.velocities.is_some());
        assert!(fa.bit_identical(&fb), "frame {i}");
    }
    let rows = a.read_stats_file().unwrap();
    assert!(rows.iter().all(|r| r.max_rel_energy_error.is_some()));
    assert_eq!(rows, b.read_stats_file().unwrap());
}

#[test]
fn refuses_features_that_are_not_built_yet() {
    let base = small_config();
    let cases = [
        (
            replace(
                &base,
                "kind = \"full_ring\"",
                "kind = \"arc\"\nwidth_deg = 10.0\ncentre_longitude_deg = 0.0",
            ),
            "D1",
        ),
        (
            replace(
                &base,
                "model = \"none\"",
                "model = \"stochastic\"\nparticle_radius_km = 1.0e-5\nweight = 8.0e8\n\
                 cell_radial_km = 0.05\ncell_azimuthal_km = 10.0\ncell_vertical_km = 0.04\n\
                 [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5",
            ),
            "K2",
        ),
        (
            replace(
                &base,
                "model = \"none\"",
                "model = \"hard_sphere\"\nparticle_radius_km = 1.0e-5\n\
                 [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5",
            ),
            "K3",
        ),
    ];
    for (text, task) in cases {
        let tmp = Tmp::new();
        let e = run_config(&tmp, &text, "x", true).unwrap_err();
        assert!(
            matches!(&e, RunError::Setup(m) if m.contains("not implemented") && m.contains(task)),
            "{e}"
        );
        assert_eq!(e.exit_code(), 2);
        assert!(!tmp.path().join("x").exists(), "nothing is written");
    }
}

#[test]
fn never_overwrites_a_non_empty_directory() {
    let tmp = Tmp::new();
    let out = tmp.path().join("a");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("precious.txt"), b"keep").unwrap();
    let e = run_config(&tmp, &small_config(), "a", true).unwrap_err();
    assert_eq!(e.exit_code(), 2);
    assert!(e.to_string().contains(&out.display().to_string()), "{e}");
    assert_eq!(std::fs::read(out.join("precious.txt")).unwrap(), b"keep");
    // An empty directory is fine.
    std::fs::remove_file(out.join("precious.txt")).unwrap();
    run_config(
        &tmp,
        &replace(
            &small_config(),
            "duration_years = 0.02",
            "duration_years = 0.002",
        ),
        "a",
        true,
    )
    .unwrap();
}

#[test]
fn unknown_moon_and_missing_params_are_setup_errors() {
    let tmp = Tmp::new();
    let e = run_config(
        &tmp,
        &replace(&small_config(), "\"pandora\"", "\"atlas\""),
        "a",
        true,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("atlas") && e.to_string().contains("prometheus"),
        "{e}"
    );
    let bad = replace(&small_config(), &params_dir(), "/nonexistent-params");
    let e = run_config(&tmp, &bad, "b", true).unwrap_err();
    assert!(
        matches!(e, RunError::Setup(_)) && e.to_string().contains("saturn.toml"),
        "{e}"
    );
}

#[test]
fn binary_exit_codes() {
    let exe = env!("CARGO_BIN_EXE_fring-run");
    let code = |args: &[&str]| {
        Command::new(exe)
            .args(args)
            .output()
            .unwrap()
            .status
            .code()
            .unwrap()
    };
    assert_eq!(code(&["--help"]), 0);
    assert_eq!(code(&[]), 2);
    assert_eq!(code(&["--bogus"]), 2);
    assert_eq!(code(&["--config", "/nonexistent.toml"]), 2);
    let tmp = Tmp::new();
    let cfg = tmp.path().join("c.toml");
    std::fs::write(&cfg, small_config()).unwrap();
    let out = tmp.path().join("o");
    let ok = Command::new(exe)
        .args([
            "--config",
            cfg.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(ok.status.code(), Some(0));
    let log = String::from_utf8_lossy(&ok.stderr);
    assert!(log.contains("[100%]") && log.contains("done in"), "{log}");
    assert!(RunDir::new(&out).meta_path().exists());
}

/// Acceptance for R1: 10^4 particles, 1 year, two moons, from one config file.
#[test]
#[ignore = "release build, about 15 s"]
fn acceptance_1e4_particles_1_year() {
    let tmp = Tmp::new();
    let text = replace(
        CONFIG,
        "params_dir = \"params\"",
        &format!("params_dir = {:?}", params_dir()),
    );
    let (dir, s) = run_config(&tmp, &text, "acc", false).unwrap();
    assert_eq!(s.steps, 87_660);
    assert_eq!((s.frames, s.stats_rows), (62, 602));
    assert_eq!(dir.list_frames().unwrap().len(), 62);
    let last = dir.read_frame_file(61).unwrap();
    assert_eq!((last.step, last.len()), (87_660, 10_000));
    assert!(last.x_km.iter().chain(&last.y_km).all(|v| v.is_finite()));
    let bytes: u64 = std::fs::read_dir(dir.root())
        .unwrap()
        .chain(std::fs::read_dir(dir.frames_dir()).unwrap())
        .map(|e| e.unwrap().metadata().unwrap())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();
    eprintln!(
        "acceptance: wall {:.1} s, {} frames, {} stats rows, {:.1} MB, {:.2} M particle-steps/s",
        s.wall_time_s,
        s.frames,
        s.stats_rows,
        bytes as f64 / 1e6,
        s.particle_steps_per_s / 1e6
    );
}

/// `backend = "gpu"` runs `GpuBackend` and agrees with the CPU run: the
/// frames are f32 and the GPU differs from the f64 CPU by far less than
/// their resolution. Needs a GPU: `cargo test -p fring-run --release -- --ignored`.
#[test]
#[ignore = "needs a GPU"]
fn gpu_backend_run_agrees_with_the_cpu_run() {
    let base = small_config();
    let tmp = Tmp::new();
    let (cpu, _) = run_config(&tmp, &base, "cpu", true).unwrap();
    let gpu_text = replace(&base, "backend = \"cpu\"", "backend = \"gpu\"");
    let (gpu, summary) = match run_config(&tmp, &gpu_text, "gpu", true) {
        Ok(r) => r,
        Err(RunError::Setup(m)) if m.contains("no GPU adapter") => {
            eprintln!("no GPU: skipped");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert!(summary.particle_steps_per_s > 0.0);
    let frames = cpu.list_frames().unwrap();
    assert_eq!(frames, gpu.list_frames().unwrap());
    let mut worst = 0.0f32;
    for i in frames {
        let (a, b) = (
            cpu.read_frame_file(i).unwrap(),
            gpu.read_frame_file(i).unwrap(),
        );
        assert_eq!(a.id, b.id);
        assert_eq!(a.step, b.step);
        for (x, y) in [(&a.x_km, &b.x_km), (&a.y_km, &b.y_km), (&a.z_km, &b.z_km)] {
            for (p, q) in x.iter().zip(y) {
                worst = worst.max((p - q).abs());
            }
        }
    }
    // The core particles move 4e5 km in a few orbits; the f32 frames have a
    // resolution of 0.03 km at that size.
    assert!(worst < 0.1, "largest frame difference {worst} km");
    let (ra, rb) = (
        cpu.read_stats_file().unwrap(),
        gpu.read_stats_file().unwrap(),
    );
    assert_eq!(ra.len(), rb.len());
}

/// `backend = "gpu"` is accepted: with a GPU the run works, without one it
/// fails with a backend error (never "not implemented"). Runs in CI.
#[test]
fn gpu_backend_is_selected_not_refused() {
    let text = replace(&small_config(), "backend = \"cpu\"", "backend = \"gpu\"");
    let tmp = Tmp::new();
    match run_config(&tmp, &text, "g", true) {
        Ok(_) => {}
        Err(RunError::Setup(m)) => {
            assert!(!m.contains("not implemented"), "{m}");
            assert!(m.contains("backend error"), "{m}");
        }
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn chaos_table_writes_chaos_csv_and_the_meta_field() {
    let tmp = Tmp::new();
    // Steps of one block of the indicator: 0.02 years is about 4,800 steps.
    let base = small_config();
    let text = format!(
        "{base}\n[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1.0\nrenorm_steps = 100\n"
    );
    let n = RunConfig::from_toml_str(&text).unwrap().time.n_steps();
    let (dir, summary) = run_config(&tmp, &text, "chaos", true).unwrap();
    let rows = dir.read_chaos_file().unwrap();
    assert_eq!(rows.len(), 200);
    assert!(rows.iter().enumerate().all(|(i, r)| r.id == i as u32));
    assert!(
        rows.iter()
            .all(|r| r.a0_km > 139_000.0 && r.a0_km < 141_500.0)
    );
    assert!(rows.iter().all(|r| r.lambda_rel_prometheus_deg >= -180.0));
    assert!(
        rows.iter()
            .all(|r| r.megno_mean.is_finite() && r.flags == 0)
    );
    let meta = dir.read_meta_file().unwrap();
    let chaos = meta.chaos.expect("meta.json has the chaos field");
    assert_eq!((chaos.method.as_str(), chaos.renorm_steps), ("shadow", 100));
    assert_eq!(chaos.offset_km, 1.0);
    assert_eq!(chaos.n_renorm, n / 100);
    assert_eq!(summary.chaos, Some(chaos));

    // Without the table: no file, no field.
    let (dir, summary) = run_config(&tmp, &base, "plain", true).unwrap();
    assert!(!dir.chaos_path().exists());
    assert!(dir.read_meta_file().unwrap().chaos.is_none());
    assert!(summary.chaos.is_none());
}
