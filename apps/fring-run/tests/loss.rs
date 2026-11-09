//! `[loss]` in `fring-run` (ICR-004 section 5, RUN-LOSS): `drops.csv`, the moon
//! radii and the chaos flag bit 1. Small runs; the GPU tests skip without a GPU.

use fring_io::{DropRow, RunDir};
use fring_model::RunConfig;
use fring_run::cli::Args;
use fring_run::{RunError, run};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("fring-loss-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Tmp(dir)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const STEP_S: f64 = 360.0;

/// A clump of `count` particles with a Gaussian spread of `sigma_km` around the
/// start position of Prometheus (`params/moons.toml`: (3294.7, 139120.9, -16.7)
/// km, radius 139,160 km, longitude 87.6 degrees). The shear (1.5 n da) moves
/// the particles through the moon: some start inside it, some cross it later.
fn config(backend: &str, extra: &str, count: u32, sigma_km: f64) -> String {
    format!(
        "[run]\nname = \"loss\"\nbackend = \"{backend}\"\nseed = 20261109\nparams_dir = {:?}\n\
         [time]\nstep_s = {STEP_S}\nduration_years = 0.0025\n\
         [frame]\nreference_semi_major_axis_km = 140220.0\n\
         [domain]\nkind = \"full_ring\"\n\
         [moons]\ninclude = [\"prometheus\"]\n\
         [particles]\ncount = {count}\n\
         [particles.distribution]\nkind = \"clump\"\nsemi_major_axis_km = 139160.0\n\
         longitude_deg = 87.6\nsigma_km = {sigma_km}\n\
         [collisions]\nmodel = \"none\"\n\
         [output]\nframe_every_steps = 100\nstats_every_steps = 100\nvelocities = false\n\
         {extra}\n",
        format!("{}/../../params", env!("CARGO_MANIFEST_DIR"))
    )
}

fn run_text(tmp: &Tmp, text: &str, name: &str) -> Result<RunDir, RunError> {
    let cfg = tmp.0.join(format!("{name}.toml"));
    std::fs::write(&cfg, text).unwrap();
    let out = tmp.0.join(name);
    run(&Args {
        config: cfg,
        out: Some(out.clone()),
        quiet: true,
    })?;
    Ok(RunDir::new(out))
}

fn run_gpu(tmp: &Tmp, text: &str, name: &str) -> Option<RunDir> {
    match run_text(tmp, text, name) {
        Ok(d) => Some(d),
        Err(RunError::Setup(m)) if m.contains("no GPU adapter") => {
            eprintln!("no GPU: skipped");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

const FREEZE: &str = "[loss]\non_unbound = \"freeze\"\nmoon_radius = \"params\"\n";

fn n_steps(text: &str) -> u64 {
    RunConfig::from_toml_str(text).unwrap().time.n_steps()
}

fn check_rows(rows: &[DropRow], n: u64, count: u32) {
    assert!(!rows.is_empty(), "the clump must lose particles");
    assert!(rows.len() < count as usize, "some particles must survive");
    assert!(rows.windows(2).all(|w| w[0].id < w[1].id), "ordered by id");
    for r in rows {
        assert_eq!(r.reason, 2, "inside moon 0 (prometheus)");
        assert!(r.drop_step < n);
        assert_eq!(r.time_s, r.drop_step as f64 * STEP_S);
        assert!(
            r.position_km
                .iter()
                .chain(&r.velocity_km_s)
                .all(|v| v.is_finite())
        );
        // Saturn-centred inertial state on the orbit of Prometheus.
        let rho = r.position_km[0].hypot(r.position_km[1]);
        assert!(
            (rho - 139_160.0).abs() < 400.0,
            "frozen position {:?}",
            r.position_km
        );
    }
}

fn all_files(root: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let name = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                out.push((name, std::fs::read(&p).unwrap()));
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.retain(|(n, _)| n != "config.toml" && n != "meta.json");
    v.sort();
    v
}

#[test]
fn loss_absent_or_default_changes_nothing_and_writes_no_drops_file() {
    let tmp = Tmp::new();
    let base = config("cpu", "", 60, 60.0);
    let a = run_text(&tmp, &base, "a").unwrap();
    let defaults = config(
        "cpu",
        "[loss]\non_unbound = \"error\"\nmoon_radius = \"off\"\n",
        60,
        60.0,
    );
    let b = run_text(&tmp, &defaults, "b").unwrap();
    let (fa, fb) = (all_files(a.root()), all_files(b.root()));
    assert!(fa.len() >= 5);
    assert_eq!(
        fa, fb,
        "all files except config.toml and meta.json are identical"
    );
    assert!(!a.drops_path().exists() && !b.drops_path().exists());
}

#[test]
fn freeze_without_a_loss_writes_a_header_only_file() {
    let tmp = Tmp::new();
    // Far from the moon: nobody is lost.
    let text = config("cpu", "[loss]\non_unbound = \"freeze\"\n", 20, 60.0);
    let d = run_text(&tmp, &text, "f").unwrap();
    assert!(d.drops_path().exists());
    assert!(d.read_drops_file().unwrap().is_empty());
    let raw = std::fs::read_to_string(d.drops_path()).unwrap();
    assert_eq!(
        raw,
        "id,reason,drop_step,time_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n"
    );
}

#[test]
fn moon_radius_rule_without_freeze_is_a_config_error() {
    let tmp = Tmp::new();
    let text = config("cpu", "[loss]\nmoon_radius = \"params\"\n", 5, 60.0);
    assert!(matches!(
        run_text(&tmp, &text, "x"),
        Err(RunError::Setup(_))
    ));
}

#[test]
fn cpu_radius_rule_writes_the_lost_particles() {
    let tmp = Tmp::new();
    let text = config(
        "cpu",
        &format!(
            "{FREEZE}[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1.0\nrenorm_steps = 100\n"
        ),
        100,
        60.0,
    );
    let d = run_text(&tmp, &text, "c").unwrap();
    let rows = d.read_drops_file().unwrap();
    check_rows(&rows, n_steps(&text), 300);
    // chaos.csv: flags bit 1 is set for exactly the lost particles.
    let chaos = d.read_chaos_file().unwrap();
    for c in &chaos {
        let lost = rows.iter().any(|r| r.id == c.id);
        assert_eq!(c.flags & 2 != 0, lost, "id {}", c.id);
    }
    assert!(d.read_meta_file().is_ok());
}

/// The same run on the GPU: the same particles are lost in the same steps.
#[test]
fn gpu_radius_rule_agrees_with_the_cpu_on_reason_and_drop_step() {
    let tmp = Tmp::new();
    let extra = format!(
        "{FREEZE}[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1.0\nrenorm_steps = 100\n"
    );
    let cpu_text = config("cpu", &extra, 300, 40.0);
    let cpu = run_text(&tmp, &cpu_text, "cpu").unwrap();
    let Some(gpu) = run_gpu(&tmp, &config("gpu", &extra, 300, 40.0), "gpu") else {
        return;
    };
    let (rc, rg) = (
        cpu.read_drops_file().unwrap(),
        gpu.read_drops_file().unwrap(),
    );
    check_rows(&rg, n_steps(&cpu_text), 300);
    let key = |r: &DropRow| (r.id, r.reason, r.drop_step);
    let (kc, kg): (Vec<_>, Vec<_>) = (rc.iter().map(key).collect(), rg.iter().map(key).collect());
    eprintln!(
        "cpu lost {}, gpu lost {}; steps {:?}",
        kc.len(),
        kg.len(),
        kc.iter().map(|k| k.2).collect::<Vec<_>>()
    );
    assert_eq!(kc, kg, "id, reason and drop_step");
    for (c, g) in cpu
        .read_chaos_file()
        .unwrap()
        .iter()
        .zip(gpu.read_chaos_file().unwrap())
    {
        assert_eq!(c.flags & 2, g.flags & 2, "chaos flag bit 1, id {}", c.id);
    }
}

#[test]
fn cpu_substep_alone_gets_the_radii_and_runs() {
    let tmp = Tmp::new();
    let text = config("cpu", "[substep]\n", 20, 60.0);
    let d = run_text(&tmp, &text, "s").unwrap();
    assert!(!d.drops_path().exists());
    assert!(d.read_meta_file().is_ok());
}

#[test]
fn gpu_refuses_substep() {
    let tmp = Tmp::new();
    if run_gpu(&tmp, &config("gpu", "", 5, 60.0), "probe").is_none() {
        return;
    }
    let text = config("gpu", "[substep]\n", 5, 60.0);
    assert!(matches!(
        run_text(&tmp, &text, "s"),
        Err(RunError::Setup(_))
    ));
}
