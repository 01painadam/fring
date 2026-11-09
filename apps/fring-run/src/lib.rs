//! `fring-run`: reads a config file, runs a backend, writes a run directory
//! (`docs/snapshot-format.md`) and shows progress on stderr.
//!
//! Owner: A5 (see `AGENTS.md`). The steps, in order:
//!
//! 1. read and check the config; refuse features that are not built yet;
//! 2. load `params/`, build the moon ephemeris, make the initial particles;
//! 3. make the backend; only then create the output directory, so a failure
//!    in steps 1 to 3 leaves nothing behind;
//! 4. write `config.toml`, `moons.bin`, then frames and `stats.csv` while
//!    stepping, and `meta.json` last (its presence means "finished").
//!    With `[loss] on_unbound = "freeze"` it also writes `drops.csv` at the end
//!    of the run, and when a step fails (ICR-004, `docs/snapshot-format.md` 8b).

pub mod cli;
pub mod e1;

use cli::Args;
use fring_cpu::CpuBackend;
use fring_gpu::backend::GpuBackend;
use fring_io::{
    CHAOS_VERSION, ChaosMeta, ChaosRow, DropRow, Frame, MoonRecord, MoonsFile, RotatingVelocities,
    RunDir, RunMeta,
};
use fring_model::config::{BackendKind, Collisions};
use fring_model::params::{moon_radii_from_toml_str, moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    DEFAULT_INTERVAL_S, MoonInit, SaturnField, build_ephemeris, generate_particles,
    inertial_to_rotating, interpolate_moons, reference_mean_motion,
};
use fring_model::{Backend, DropReport, InitialState, ParticleView, Particles, RunConfig, System};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Why a run stopped. Decides the exit code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// Bad usage, config, parameters, or a feature that is not built yet
    /// (exit code 2). Nothing was written.
    Setup(String),
    /// Failure while running or writing (exit code 1).
    Run(String),
}

impl RunError {
    /// Process exit code.
    pub fn exit_code(&self) -> u8 {
        match self {
            RunError::Setup(_) => 2,
            RunError::Run(_) => 1,
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Setup(m) | RunError::Run(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RunError {}

fn setup<T>(r: fring_model::Result<T>) -> Result<T, RunError> {
    r.map_err(|e| RunError::Setup(e.to_string()))
}

fn during<T>(r: fring_model::Result<T>) -> Result<T, RunError> {
    r.map_err(|e| RunError::Run(e.to_string()))
}

/// What a finished run reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// The run directory.
    pub out_dir: PathBuf,
    /// Steps done.
    pub steps: u64,
    /// Frame files written.
    pub frames: u64,
    /// Rows written to `stats.csv`.
    pub stats_rows: u64,
    /// Wall-clock time of the whole run in seconds.
    pub wall_time_s: f64,
    /// Particle-steps per second while stepping (output time excluded).
    pub particle_steps_per_s: f64,
    /// The chaos indicator settings and result, if the config asked for it.
    pub chaos: Option<ChaosMeta>,
}

/// Refuses what the chosen backend would ignore or cannot do, with a clear
/// message. A backend must never ignore a setting without an error.
pub fn check_supported(config: &RunConfig) -> Result<(), RunError> {
    let gpu = config.run.backend == BackendKind::Gpu;
    // Arc domains, and stochastic collisions on either backend, are accepted.
    // The GPU backend itself refuses arc or mode C with collisions, and chaos
    // with collisions. Those errors come back through `setup` as a clean
    // setup error (exit code 2), so no second list is kept here.
    match config.collisions {
        Collisions::HardSphere { .. } if gpu => Err(RunError::Setup(
            "collisions.model = \"hard_sphere\" is not available on the GPU backend \
             (K3 is CPU only); use backend = \"cpu\""
                .to_string(),
        )),
        _ => Ok(()),
    }
}

/// Checks that the output directory does not exist or is empty. Never
/// overwrites anything.
fn check_out_dir(dir: &Path) -> Result<(), RunError> {
    if !dir.exists() {
        return Ok(());
    }
    let not_empty = || {
        RunError::Setup(format!(
            "output directory {} exists and is not empty; fring-run never overwrites a run. \
             Choose another --out or remove it yourself",
            dir.display()
        ))
    };
    if !dir.is_dir() {
        return Err(not_empty());
    }
    let mut entries = std::fs::read_dir(dir)
        .map_err(|e| RunError::Setup(format!("cannot read {}: {e}", dir.display())))?;
    if entries.next().is_some() {
        return Err(not_empty());
    }
    Ok(())
}

fn read_text(path: &Path) -> Result<String, RunError> {
    std::fs::read_to_string(path)
        .map_err(|e| RunError::Setup(format!("cannot read {}: {e}", path.display())))
}

/// Selects the moons of `[moons] include`, in that order.
fn select_moons(include: &[String], all: Vec<MoonInit>) -> Result<Vec<MoonInit>, RunError> {
    let mut chosen: Vec<MoonInit> = Vec::new();
    for name in include {
        if chosen.iter().any(|m| &m.name == name) {
            return Err(RunError::Setup(format!(
                "moons.include lists {name:?} twice"
            )));
        }
        let moon = all.iter().find(|m| &m.name == name).ok_or_else(|| {
            let known: Vec<&str> = all.iter().map(|m| m.name.as_str()).collect();
            RunError::Setup(format!(
                "moons.include: unknown moon {name:?}; params/moons.toml has {known:?}"
            ))
        })?;
        chosen.push(moon.clone());
    }
    Ok(chosen)
}

/// The radii of the moons in the track order of the ephemeris (which is the
/// `[moons] include` order), from the text of `moons.toml`. Reason `2 + k` of a
/// lost particle uses the same order.
pub(crate) fn moon_radii_for(moons_text: &str, system: &System) -> Result<Vec<f64>, RunError> {
    let radii = setup(moon_radii_from_toml_str(moons_text))?;
    system
        .moons
        .tracks
        .iter()
        .map(|t| {
            radii
                .iter()
                .find(|(name, _)| *name == t.name)
                .map(|(_, r)| *r)
                .ok_or_else(|| RunError::Setup(format!("no radius_km for moon {:?}", t.name)))
        })
        .collect()
}

/// The `drops.csv` rows: one per lost particle, ordered by id, with the frozen
/// inertial state from `particles` (the same order as `report`). `time_s` is
/// `start_s + drop_step x step_s`.
pub fn drop_rows(
    report: &DropReport,
    particles: &Particles,
    start_s: f64,
    step_s: f64,
) -> Result<Vec<DropRow>, RunError> {
    let n = particles.len();
    if report.reason.len() != n || report.drop_step.len() != n {
        return Err(RunError::Run(format!(
            "the drop report has {} reasons and {} steps for {n} particles",
            report.reason.len(),
            report.drop_step.len()
        )));
    }
    let mut rows = Vec::new();
    for (i, &reason) in report.reason.iter().enumerate() {
        if reason == 0 {
            continue;
        }
        let drop_step = report.drop_step[i];
        if drop_step == u64::MAX {
            return Err(RunError::Run(format!(
                "particle {} is lost (reason {reason}) but has no drop step",
                particles.id[i]
            )));
        }
        let (r, v) = (particles.position_km(i), particles.velocity_km_s(i));
        rows.push(DropRow {
            id: particles.id[i],
            reason,
            drop_step,
            time_s: start_s + drop_step as f64 * step_s,
            position_km: [r.x, r.y, r.z],
            velocity_km_s: [v.x, v.y, v.z],
        });
    }
    rows.sort_by_key(|r| r.id);
    Ok(rows)
}

/// Reads the lost particles from the backend (`None` without freeze). The GPU
/// backend reads buffers back here: call it once, not every step.
fn collect_drops<B: Backend>(
    backend: &mut B,
    start_s: f64,
    step_s: f64,
) -> Result<Option<Vec<DropRow>>, RunError> {
    let Some(report) = during(backend.dropped())? else {
        return Ok(None);
    };
    let view = during(backend.particles())?;
    drop_rows(&report, view.particles, start_s, step_s).map(Some)
}

/// True at multiples of `every` and at the last step `n`.
fn is_event(step: u64, every: u64, n: u64) -> bool {
    step.is_multiple_of(every) || step == n
}

/// The next step after `step` that is a multiple of `every`, or `n`.
fn next_multiple(step: u64, every: u64, n: u64) -> u64 {
    ((step / every + 1) * every).min(n)
}

/// Converts a backend view (inertial, f64) to a frame (rotating, f32).
/// The rotation is done in f64 and only the result is rounded to f32.
pub fn make_frame(
    view: &ParticleView<'_>,
    step: u64,
    a0_km: f64,
    n0_rad_s: f64,
    velocities: bool,
) -> Frame {
    let p = view.particles;
    let n = p.len();
    let mut f = Frame {
        step,
        time_s: view.time_s,
        a0_km,
        n0_rad_s,
        t_ref_s: 0.0,
        id: p.id.clone(),
        x_km: Vec::with_capacity(n),
        y_km: Vec::with_capacity(n),
        z_km: Vec::with_capacity(n),
        velocities: None,
    };
    let mut v = RotatingVelocities::default();
    for i in 0..n {
        let (pos, vel) = inertial_to_rotating::<f64>(
            p.position_km(i),
            p.velocity_km_s(i),
            n0_rad_s,
            view.time_s,
        );
        f.x_km.push(pos.x as f32);
        f.y_km.push(pos.y as f32);
        f.z_km.push(pos.z as f32);
        if velocities {
            v.vx_km_s.push(vel.x as f32);
            v.vy_km_s.push(vel.y as f32);
            v.vz_km_s.push(vel.z as f32);
        }
    }
    if velocities {
        f.velocities = Some(v);
    }
    f
}

struct Progress {
    quiet: bool,
    start: Instant,
    last: Instant,
}

impl Progress {
    fn line(&self, msg: &str) {
        if !self.quiet {
            eprintln!("{msg}");
        }
    }

    fn update(&mut self, step: u64, n_steps: u64, time_s: f64, frames: u64, force: bool) {
        if self.quiet || (!force && self.last.elapsed().as_secs_f64() < 1.0) {
            return;
        }
        self.last = Instant::now();
        let elapsed = self.start.elapsed().as_secs_f64();
        let frac = step as f64 / n_steps as f64;
        let eta = if frac > 0.0 {
            elapsed * (1.0 - frac) / frac
        } else {
            0.0
        };
        eprintln!(
            "[{:3.0}%] step {step}/{n_steps}  t={:.3} yr  frames {frames}  elapsed {elapsed:.1} s  eta {eta:.1} s",
            100.0 * frac,
            time_s / fring_model::units::SECONDS_PER_JULIAN_YEAR,
        );
    }
}

/// Runs the simulation described by `args`. See the module documentation.
pub fn run(args: &Args) -> Result<Summary, RunError> {
    let start = Instant::now();

    // 1. Config.
    let config_bytes = std::fs::read(&args.config)
        .map_err(|e| RunError::Setup(format!("cannot read {}: {e}", args.config.display())))?;
    let config_text = String::from_utf8(config_bytes.clone())
        .map_err(|_| RunError::Setup("the config file is not UTF-8 text".into()))?;
    let config = setup(RunConfig::from_toml_str(&config_text))?;
    check_supported(&config)?;
    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs").join(&config.run.name));
    check_out_dir(&out_dir)?;

    // 2. Parameters, ephemeris, initial particles.
    let params_dir = Path::new(&config.run.params_dir);
    let saturn = setup(saturn_from_toml_str(&read_text(
        &params_dir.join("saturn.toml"),
    )?))?;
    let (moons, moons_text) = if config.moons.include.is_empty() {
        (Vec::new(), None)
    } else {
        let text = read_text(&params_dir.join("moons.toml"))?;
        let all = setup(moons_from_toml_str(&text))?;
        (select_moons(&config.moons.include, all)?, Some(text))
    };
    let n_steps = config.time.n_steps();
    let step_s = config.time.step_s;
    let progress_start = Instant::now();
    let mut progress = Progress {
        quiet: args.quiet,
        start: progress_start,
        last: progress_start,
    };
    progress.line(&format!(
        "run {:?}: {} particles, {n_steps} steps of {step_s} s, {} moons -> {}",
        config.run.name,
        config.particles.count,
        moons.len(),
        out_dir.display()
    ));
    // The last sample must be at or after the end of the last step, which can
    // be later than `duration_s` because the step count is rounded up.
    let eph = setup(build_ephemeris(
        &saturn,
        &moons,
        n_steps as f64 * step_s + DEFAULT_INTERVAL_S,
        DEFAULT_INTERVAL_S,
    ))?;
    let field = SaturnField::<f64>::from_params(&saturn);
    let a0_km = config.frame.reference_semi_major_axis_km;
    let n0_rad_s = reference_mean_motion(&field, a0_km);
    let particles = setup(generate_particles(
        &field,
        &config.particles.distribution,
        config.particles.count,
        config.run.seed,
    ))?;
    let init = InitialState {
        time_s: 0.0,
        particles,
    };
    let system = System { saturn, moons: eph };
    // Moon radii for the radius rule (`moon_radius = "params"`) and for
    // `[substep]` (ICR-004 section 5). Read only when one of them is on, so a
    // config without them behaves exactly as before.
    let radii_km = match &moons_text {
        Some(text) if config.moon_radius_rule() || config.substep.is_some() => {
            Some(moon_radii_for(text, &system)?)
        }
        _ => None,
    };

    // 3. Backend, chosen by `[run] backend`.
    let ctx = Context {
        config: &config,
        system: &system,
        config_bytes: &config_bytes,
        a0_km,
        n0_rad_s,
        n_steps,
        start_s: init.time_s,
        out: RunDir::new(&out_dir),
    };
    let (mut summary, machine) = match config.run.backend {
        BackendKind::Cpu => {
            let mut backend = setup(CpuBackend::new(&config, &system, &init))?;
            if let Some(r) = &radii_km {
                setup(backend.set_moon_radii_km(r))?;
            }
            let machine = fring_io::sysinfo::machine("cpu");
            (execute(backend, &ctx, &mut progress)?, machine)
        }
        BackendKind::Gpu => {
            let mut backend = setup(GpuBackend::new(&config, &system, &init))?;
            if let Some(r) = &radii_km {
                setup(backend.set_moon_radii_km(r))?;
            }
            let machine = fring_io::sysinfo::machine(&format!(
                "gpu {}, mode {}",
                backend.adapter_name(),
                backend.mode().name()
            ));
            (execute(backend, &ctx, &mut progress)?, machine)
        }
    };

    // 4. meta.json, last.
    summary.wall_time_s = start.elapsed().as_secs_f64();
    let git = fring_io::sysinfo::git_hash(
        &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    );
    let mut meta = RunMeta::new(
        env!("CARGO_PKG_VERSION"),
        &git,
        &machine,
        summary.wall_time_s,
    );
    meta.chaos = summary.chaos.clone();
    during(ctx.out.write_meta_file(&meta))?;
    progress.line(&format!(
        "done in {:.1} s: {} frames, {} stats rows, {:.2} M particle-steps/s (stepping only) -> {}",
        summary.wall_time_s,
        summary.frames,
        summary.stats_rows,
        summary.particle_steps_per_s / 1e6,
        out_dir.display()
    ));
    Ok(summary)
}

struct Context<'a> {
    config: &'a RunConfig,
    system: &'a System,
    config_bytes: &'a [u8],
    a0_km: f64,
    n0_rad_s: f64,
    n_steps: u64,
    /// Simulation time of step 0, s.
    start_s: f64,
    out: RunDir,
}

/// Steps at which frames are written: 0, every `frame_every_steps`, and the last step.
fn frame_steps(every: u64, n_steps: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..=n_steps).step_by(every as usize).collect();
    if v.last() != Some(&n_steps) {
        v.push(n_steps);
    }
    v
}

fn moons_file(ctx: &Context<'_>, steps: &[u64]) -> Result<MoonsFile, RunError> {
    let eph = &ctx.system.moons;
    let names = eph.tracks.iter().map(|t| t.name.clone()).collect();
    let mut records = Vec::with_capacity(steps.len());
    for &s in steps {
        let t = s as f64 * ctx.config.time.step_s;
        let bodies = if eph.tracks.is_empty() {
            Vec::new()
        } else {
            during(interpolate_moons(eph, t))?.moons
        };
        records.push(MoonRecord { time_s: t, bodies });
    }
    Ok(MoonsFile { names, records })
}

/// Appends one `stats.csv` row with a single `write_all` (the crash rule).
fn append_stats(file: &mut std::fs::File, stats: &fring_model::StepStats) -> Result<(), RunError> {
    let mut buf = Vec::new();
    during(fring_io::write_stats_row(&mut buf, stats))?;
    file.write_all(&buf)
        .and_then(|()| file.flush())
        .map_err(|e| RunError::Run(format!("writing stats.csv: {e}")))
}

/// Per particle: id, Keplerian a (point mass) and longitude relative to
/// Prometheus (degrees, [-180, 180), NaN without Prometheus) at the start.
fn chaos_initial_fn<B: Backend>(
    ctx: &Context<'_>,
    backend: &mut B,
) -> Result<Vec<(u32, f64, f64)>, RunError> {
    let gm = ctx.system.saturn.gm_km3_s2;
    let prometheus = ctx
        .system
        .moons
        .tracks
        .iter()
        .position(|t| t.name == "prometheus");
    let moon_lon = prometheus.and_then(|k| {
        let m = backend.moons();
        m.moons
            .get(k)
            .map(|b| b.position_km.y.atan2(b.position_km.x))
    });
    let view = during(backend.particles())?;
    let p = view.particles;
    Ok((0..p.len())
        .map(|i| {
            let (r, v) = (p.position_km(i), p.velocity_km_s(i));
            let a = 1.0 / (2.0 / r.norm() - v.norm_squared() / gm);
            let lambda = moon_lon.map_or(f64::NAN, |ml| {
                let d = (r.y.atan2(r.x) - ml).to_degrees();
                (d + 180.0).rem_euclid(360.0) - 180.0
            });
            (p.id[i], a, lambda)
        })
        .collect())
}

fn execute<B: Backend>(
    mut backend: B,
    ctx: &Context<'_>,
    progress: &mut Progress,
) -> Result<Summary, RunError> {
    let cfg = ctx.config;
    let out = &ctx.out;
    let n = ctx.n_steps;
    let (fe, se) = (cfg.output.frame_every_steps, cfg.output.stats_every_steps);

    during(out.create())?;
    during(out.write_config_copy(ctx.config_bytes))?;
    during(out.write_moons_file(&moons_file(ctx, &frame_steps(fe, n))?))?;
    let mut stats_file = std::fs::File::create(out.stats_path())
        .map_err(|e| RunError::Run(format!("creating stats.csv: {e}")))?;
    let mut header = Vec::new();
    during(fring_io::write_stats_header(&mut header))?;
    stats_file
        .write_all(&header)
        .map_err(|e| RunError::Run(format!("writing stats.csv: {e}")))?;

    let mut frames = 0u64;
    let mut stats_rows = 0u64;
    let mut stepping_s = 0.0;
    let write_frame = |backend: &mut B, step: u64, frames: &mut u64| -> Result<(), RunError> {
        let view = during(backend.particles())?;
        let frame = make_frame(&view, step, ctx.a0_km, ctx.n0_rad_s, cfg.output.velocities);
        during(out.write_frame_file(*frames, &frame))?;
        *frames += 1;
        Ok(())
    };

    // The stepping loop. If it fails and `drops.csv` is on, the lost
    // particles are still written (best effort) before the error is returned.
    let mut chaos_initial = None;
    let loop_result = (|| -> Result<(), RunError> {
        if cfg.diagnostics.as_ref().is_some_and(|d| d.chaos.is_some()) {
            chaos_initial = Some(chaos_initial_fn(ctx, &mut backend)?);
        }
        let mut step = 0u64;
        write_frame(&mut backend, 0, &mut frames)?;
        append_stats(&mut stats_file, &during(backend.stats())?)?;
        stats_rows += 1;
        while step < n {
            let next = next_multiple(step, fe, n).min(next_multiple(step, se, n));
            let t0 = Instant::now();
            during(backend.step(next - step))?;
            stepping_s += t0.elapsed().as_secs_f64();
            step = next;
            if is_event(step, fe, n) {
                write_frame(&mut backend, step, &mut frames)?;
            }
            if is_event(step, se, n) {
                append_stats(&mut stats_file, &during(backend.stats())?)?;
                stats_rows += 1;
            }
            progress.update(step, n, backend.time_s(), frames, step == n);
        }
        Ok(())
    })();
    if let Err(e) = loop_result {
        // The state is the one after the last completed `step` call. A failure
        // to write here must not hide the first error.
        if let Ok(Some(rows)) = collect_drops(&mut backend, ctx.start_s, cfg.time.step_s) {
            let _ = out.write_drops_file(&rows);
        }
        return Err(e);
    }
    // Once, at the end: the GPU reads its status buffer back here.
    let drops = collect_drops(&mut backend, ctx.start_s, cfg.time.step_s)?;
    if let Some(rows) = &drops {
        // Header only if nobody was lost: the file exists if and only if freeze is on.
        during(out.write_drops_file(rows))?;
    }
    let chaos = match (
        chaos_initial,
        cfg.diagnostics.as_ref().and_then(|d| d.chaos),
    ) {
        (Some(initial), Some(c)) => {
            let report = during(backend.chaos())?
                .ok_or_else(|| RunError::Run("the backend returned no chaos indicator".into()))?;
            let day_s = 86_400.0;
            // Bit 1 (value 2) = lost. The backends set it at their next
            // renormalisation, so a particle lost after the last one has no
            // bit yet (found on the GPU). Setting it here from the drop report
            // makes the CPU and GPU files agree.
            let lost: std::collections::HashSet<u32> =
                drops.iter().flatten().map(|r| r.id).collect();
            let rows: Vec<ChaosRow> = initial
                .iter()
                .enumerate()
                .map(|(i, &(id, a0_km, lambda))| ChaosRow {
                    id,
                    a0_km,
                    lambda_rel_prometheus_deg: lambda,
                    ftle_per_day: report.ftle_per_s[i] * day_s,
                    log_growth: report.log_growth[i],
                    megno_mean: report.megno_mean[i],
                    flags: report.flags[i] | if lost.contains(&id) { 2 } else { 0 },
                })
                .collect();
            during(out.write_chaos_file(&rows))?;
            Some(ChaosMeta {
                method: "shadow".to_string(),
                offset_km: c.offset_km,
                renorm_steps: c.renorm_steps,
                n_renorm: report.n_renorm,
                chaos_format_version: CHAOS_VERSION,
            })
        }
        _ => None,
    };
    let particle_steps = cfg.particles.count as f64 * n as f64;
    Ok(Summary {
        chaos,
        out_dir: out.root().to_path_buf(),
        steps: n,
        frames,
        stats_rows,
        wall_time_s: 0.0,
        particle_steps_per_s: particle_steps / stepping_s.max(1e-9),
    })
}
