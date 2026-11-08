//! E1-RUN: runs a backend on the E1-REF files and writes the "WH result"
//! files that `validation/scripts/e1_compare.py` reads.
//!
//! Owner: A5. This is a validation protocol, not the run format of
//! `docs/snapshot-format.md`, so the CSV code lives here and not in
//! `fring-io`. The file contract is in `docs/handoffs/E1-REF.md`.
//!
//! - Layer 1: every encounter of `e1_l1_states.csv` runs for 172,800 s from
//!   its own `t0_s`; the output is the state at `t0_s + 172800 s`.
//! - Layer 2: every particle of `e1_l2_states.csv` runs for `days` of
//!   `e1_l2.json` from t = 0; the output is the closest distance to
//!   Prometheus, the final state and the mean MEGNO.
//!
//! The functions are generic over [`Backend`]; the caller gives a closure
//! that makes the backend (so that the GPU can use the same code).

use crate::{RunError, read_text, select_moons, setup};
use fring_cpu::CpuBackend;
use fring_model::params::{moon_radii_from_toml_str, moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{DEFAULT_INTERVAL_S, build_ephemeris, interpolate_moons};
use fring_model::rng::{Counter, Philox, Stream, box_muller};
use fring_model::units::SECONDS_PER_JULIAN_YEAR;
use fring_model::{Backend, InitialState, Particles, RunConfig, System, Vec3};
use std::fmt::Write as _;
use std::path::Path;

/// Length of the layer 1 window in seconds (2 days).
pub const WINDOW_S: f64 = 172_800.0;
/// Number of steps in the layer 1 window (172,800 s / 356.2537 s = 485.05).
pub const L1_STEPS: u64 = 485;
/// Nominal step in seconds (ICR-004, ADR 0008).
pub const NOMINAL_STEP_S: f64 = 356.2537;

/// One run setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `[loss] on_unbound = "freeze"`, no radius rule, no sub-stepping.
    /// Production setting; the file for `e1_compare.py --run`.
    Plain,
    /// `plain` plus `[substep]` (accuracy 0.1, max_level 4); `--substep`.
    Substep,
    /// `plain` plus `moon_radius = "params"`. Information only.
    Loss,
}

impl Variant {
    /// Name used for the output directory.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Plain => "plain",
            Variant::Substep => "substep",
            Variant::Loss => "loss",
        }
    }

    /// Parses a variant name.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "plain" => Ok(Variant::Plain),
            "substep" => Ok(Variant::Substep),
            "loss" => Ok(Variant::Loss),
            _ => Err(format!("unknown variant {s:?} (plain, substep or loss)")),
        }
    }
}

/// One layer 1 encounter.
#[derive(Clone, Debug, PartialEq)]
pub struct L1Row {
    /// Particle id.
    pub id: u32,
    /// Stratum of the measured closest distance (0 to 3).
    pub stratum: u8,
    /// Epoch index j (`t0 = j x 10 d`).
    pub t0_idx: u32,
    /// Start time in s after the epoch.
    pub t0_s: f64,
    /// Inertial state (km, km/s) at `t0_s`.
    pub state: [f64; 6],
}

/// A particle that a backend froze.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frozen {
    /// Particle id.
    pub id: u32,
    /// Stratum (layer 1) or 255 (layer 2).
    pub stratum: u8,
    /// Reason code (1 unbound, 2 + k inside moon k).
    pub reason: u8,
    /// 0-based step of the loss.
    pub drop_step: u64,
}

/// One layer 2 result row.
#[derive(Clone, Debug, PartialEq)]
pub struct L2Row {
    /// Particle id.
    pub id: u32,
    /// Closest distance to Prometheus over the run, km.
    pub min_prometheus_km: f64,
    /// Final state (km, km/s).
    pub state: [f64; 6],
    /// Mean MEGNO of the run.
    pub megno_mean: f64,
}

// ---------------------------------------------------------------- CSV

/// Numeric rows of a CSV with `#` comment lines. Each row must have `ncols`.
pub fn parse_numeric_csv(text: &str, ncols: usize, what: &str) -> Result<Vec<Vec<f64>>, RunError> {
    let mut rows = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let vals: Result<Vec<f64>, _> = line.split(',').map(|s| s.trim().parse::<f64>()).collect();
        let vals = vals.map_err(|e| RunError::Setup(format!("{what} line {}: {e}", n + 1)))?;
        if vals.len() != ncols {
            return Err(RunError::Setup(format!(
                "{what} line {}: {} columns, expected {ncols}",
                n + 1,
                vals.len()
            )));
        }
        rows.push(vals);
    }
    Ok(rows)
}

fn id_of(v: f64, what: &str) -> Result<u32, RunError> {
    if v >= 0.0 && v.fract() == 0.0 && v <= f64::from(u32::MAX) {
        Ok(v as u32)
    } else {
        Err(RunError::Setup(format!("{what}: bad id {v}")))
    }
}

/// Reads `e1_l1_states.csv`.
pub fn parse_l1_states(text: &str) -> Result<Vec<L1Row>, RunError> {
    parse_numeric_csv(text, 10, "e1_l1_states.csv")?
        .into_iter()
        .map(|r| {
            Ok(L1Row {
                id: id_of(r[0], "e1_l1_states.csv")?,
                stratum: r[1] as u8,
                t0_idx: r[2] as u32,
                t0_s: r[3],
                state: [r[4], r[5], r[6], r[7], r[8], r[9]],
            })
        })
        .collect()
}

/// Reads `e1_l2_states.csv` as particles at t = 0.
pub fn parse_l2_states(text: &str) -> Result<Particles, RunError> {
    let mut p = Particles::default();
    for r in parse_numeric_csv(text, 7, "e1_l2_states.csv")? {
        p.push(
            id_of(r[0], "e1_l2_states.csv")?,
            Vec3::new(r[1], r[2], r[3]),
            Vec3::new(r[4], r[5], r[6]),
        );
    }
    Ok(p)
}

fn num(x: f64) -> String {
    // `{:e}` with no precision prints the shortest string that round-trips.
    format!("{x:e}")
}

/// Layer 1 result text, rows sorted by id.
pub fn format_l1(rows: &[(u32, [f64; 6])], header: &str) -> String {
    let mut sorted: Vec<_> = rows.to_vec();
    sorted.sort_by_key(|r| r.0);
    let mut s = String::new();
    for l in header.lines() {
        let _ = writeln!(s, "# {l}");
    }
    s.push_str("# columns: id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n");
    for (id, v) in sorted {
        let _ = writeln!(
            s,
            "{id},{},{},{},{},{},{}",
            num(v[0]),
            num(v[1]),
            num(v[2]),
            num(v[3]),
            num(v[4]),
            num(v[5])
        );
    }
    s
}

/// Layer 2 result text, rows sorted by id.
pub fn format_l2(rows: &[L2Row], header: &str) -> String {
    let mut sorted: Vec<_> = rows.to_vec();
    sorted.sort_by_key(|r| r.id);
    let mut s = String::new();
    for l in header.lines() {
        let _ = writeln!(s, "# {l}");
    }
    s.push_str(
        "# columns: id,min_prometheus_km,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,megno_mean\n",
    );
    for r in sorted {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{}",
            r.id,
            num(r.min_prometheus_km),
            num(r.state[0]),
            num(r.state[1]),
            num(r.state[2]),
            num(r.state[3]),
            num(r.state[4]),
            num(r.state[5]),
            num(r.megno_mean)
        );
    }
    s
}

/// The frozen list as CSV.
pub fn format_frozen(layer: u8, rows: &[Frozen]) -> String {
    let mut sorted: Vec<_> = rows.to_vec();
    sorted.sort_by_key(|r| r.id);
    let mut s = String::from(
        "# layer,id,stratum,reason,drop_step (reason 1 unbound, 2 + k inside moon k; stratum 255 = layer 2)\n",
    );
    for r in sorted {
        let _ = writeln!(
            s,
            "{layer},{},{},{},{}",
            r.id, r.stratum, r.reason, r.drop_step
        );
    }
    s
}

// ---------------------------------------------------------------- perturbation

/// Hill radius of Prometheus used by E1-REF layer 2 (`e1_l2.json`), km.
pub const R_HILL_KM: f64 = 63.300_144_434_718_87;

/// Adds to each start position an offset of length `length_km` in a random
/// direction (isotropic, from the seeded Philox RNG, stream `ANALYSIS`,
/// counter index = particle id). With `length_km == 0` nothing changes.
pub fn perturb_positions(p: &mut Particles, length_km: f64, seed: u64) {
    if length_km == 0.0 {
        return;
    }
    let rng = Philox::from_seed(seed);
    for i in 0..p.len() {
        let c = |draw| Counter {
            index: p.id[i],
            step: 0,
            stream: Stream::ANALYSIS,
            draw,
        };
        let (a, b) = (rng.uniform_f64(c(0)), rng.uniform_f64(c(1)));
        let (g0, g1) = box_muller(a[0], a[1]);
        let (g2, _) = box_muller(b[0], b[1]);
        let norm = (g0 * g0 + g1 * g1 + g2 * g2).sqrt();
        p.x_km[i] += length_km * g0 / norm;
        p.y_km[i] += length_km * g1 / norm;
        p.z_km[i] += length_km * g2 / norm;
    }
}

/// Counts inside 1, 3 and 6.3 R_H of Prometheus and the chaotic fraction
/// (mean MEGNO at least 5, ADR 0008) of a layer 2 result.
pub fn l2_summary(rows: &[L2Row]) -> ([usize; 3], f64) {
    let n = |k: f64| {
        rows.iter()
            .filter(|r| r.min_prometheus_km < k * R_HILL_KM)
            .count()
    };
    let chaotic = rows.iter().filter(|r| r.megno_mean >= 5.0).count();
    ([n(1.0), n(3.0), n(6.3)], chaotic as f64 / rows.len() as f64)
}

// ---------------------------------------------------------------- config

/// Steps and step size for a window of `window_s` seconds: `n` whole steps
/// of `window_s / n` seconds, with `n` the nearest whole number to
/// `window_s / nominal_s`.
pub fn exact_step(window_s: f64, nominal_s: f64) -> (u64, f64) {
    let n = (window_s / nominal_s).round().max(1.0) as u64;
    (n, window_s / n as f64)
}

/// The config text for one run. `duration` is (n - 0.5) steps so that
/// `ceil(duration / step)` is exactly `n`.
pub fn config_text(
    variant: Variant,
    count: usize,
    n_steps: u64,
    step_s: f64,
    chaos: bool,
) -> String {
    let duration_years = (n_steps as f64 - 0.5) * step_s / SECONDS_PER_JULIAN_YEAR;
    let mut t = format!(
        "[run]\nname = \"e1\"\nbackend = \"cpu\"\nseed = 1\nparams_dir = \"params\"\n\
         [time]\nstep_s = {step_s:e}\nduration_years = {duration_years:e}\n\
         [frame]\nreference_semi_major_axis_km = 140220.0\n\
         [domain]\nkind = \"full_ring\"\n\
         [moons]\ninclude = [\"prometheus\", \"pandora\"]\n\
         [particles]\ncount = {count}\n\
         [particles.distribution]\nkind = \"band\"\nmin_semi_major_axis_km = 139500.0\nmax_semi_major_axis_km = 141500.0\n\
         [collisions]\nmodel = \"none\"\n\
         [output]\nframe_every_steps = 1000000\nstats_every_steps = 1000000\nvelocities = false\n\
         [loss]\non_unbound = \"freeze\"\nmoon_radius = \"{}\"\n",
        if variant == Variant::Loss {
            "params"
        } else {
            "off"
        }
    );
    if variant == Variant::Substep {
        t.push_str("[substep]\naccuracy = 0.1\nmax_level = 4\n");
    }
    if chaos {
        t.push_str(
            "[diagnostics.chaos]\nmethod = \"shadow\"\noffset_km = 1.0\nrenorm_steps = 100\n",
        );
    }
    t
}

// ---------------------------------------------------------------- inputs

/// Saturn, moons (as a ready `System`) and the moon radii by track order.
pub struct Setup {
    /// Saturn and the ephemeris (Prometheus, Pandora).
    pub system: System,
    /// `radius_km` of the tracks, in track order.
    pub radii_km: Vec<f64>,
}

/// Builds the system from `params_dir`, with an ephemeris that covers
/// `[0, cover_s]` seconds.
pub fn build_setup(params_dir: &Path, cover_s: f64) -> Result<Setup, RunError> {
    let saturn = setup(saturn_from_toml_str(&read_text(
        &params_dir.join("saturn.toml"),
    )?))?;
    let moons_text = read_text(&params_dir.join("moons.toml"))?;
    let all = setup(moons_from_toml_str(&moons_text))?;
    let names = vec!["prometheus".to_string(), "pandora".to_string()];
    let moons = select_moons(&names, all)?;
    let radii = setup(moon_radii_from_toml_str(&moons_text))?;
    let radii_km = names
        .iter()
        .map(|n| {
            radii
                .iter()
                .find(|(m, _)| m == n)
                .map(|(_, r)| *r)
                .ok_or_else(|| RunError::Setup(format!("no radius_km for {n}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eph = setup(build_ephemeris(
        &saturn,
        &moons,
        cover_s + 2.0 * DEFAULT_INTERVAL_S,
        DEFAULT_INTERVAL_S,
    ))?;
    Ok(Setup {
        system: System { saturn, moons: eph },
        radii_km,
    })
}

/// Makes a CPU backend and gives it the moon radii (needed by `[substep]`
/// and by the radius rule; harmless otherwise).
pub fn cpu_backend(
    radii_km: Vec<f64>,
) -> impl FnMut(&RunConfig, &System, &InitialState) -> Result<CpuBackend, RunError> {
    move |config, system, init| {
        let mut b = setup(CpuBackend::new(config, system, init))?;
        setup(b.set_moon_radii_km(&radii_km))?;
        Ok(b)
    }
}

/// Moon position and velocity differences against `e1_moons_t0.csv`.
/// Returns the CSV text and the maximum position (km) and velocity (km/s)
/// differences over both moons and all epochs.
pub fn moon_difference(
    eph: &fring_model::MoonEphemeris,
    csv: &str,
) -> Result<(String, f64, f64), RunError> {
    let rows = parse_numeric_csv(csv, 14, "e1_moons_t0.csv")?;
    let mut out = String::from(
        "# j,t0_s,dr_prometheus_km,dv_prometheus_km_s,dr_pandora_km,dv_pandora_km_s (Rust ephemeris minus E1-REF IAS15)\n",
    );
    let (mut max_r, mut max_v) = (0.0f64, 0.0f64);
    for r in rows {
        let m = setup(interpolate_moons(eph, r[1]))?;
        let mut line = format!("{},{}", r[0], r[1]);
        for (k, b) in m.moons.iter().enumerate() {
            let o = 2 + 6 * k;
            let dr = ((b.position_km.x - r[o]).powi(2)
                + (b.position_km.y - r[o + 1]).powi(2)
                + (b.position_km.z - r[o + 2]).powi(2))
            .sqrt();
            let dv = ((b.velocity_km_s.x - r[o + 3]).powi(2)
                + (b.velocity_km_s.y - r[o + 4]).powi(2)
                + (b.velocity_km_s.z - r[o + 5]).powi(2))
            .sqrt();
            max_r = max_r.max(dr);
            max_v = max_v.max(dv);
            let _ = write!(line, ",{dr:e},{dv:e}");
        }
        out.push_str(&line);
        out.push('\n');
    }
    Ok((out, max_r, max_v))
}

// ---------------------------------------------------------------- closest distance

/// Smallest distance of the cubic Hermite curve through `(r0, v0)` and
/// `(r1, v1)` over an interval of `h_s` seconds. r in km, v in km/s.
pub fn hermite_min_distance(
    r0: [f64; 3],
    v0: [f64; 3],
    r1: [f64; 3],
    v1: [f64; 3],
    h_s: f64,
) -> f64 {
    const N: usize = 32;
    let at = |s: f64| -> f64 {
        let (s2, s3) = (s * s, s * s * s);
        let (h00, h10, h01, h11) = (
            2.0 * s3 - 3.0 * s2 + 1.0,
            s3 - 2.0 * s2 + s,
            -2.0 * s3 + 3.0 * s2,
            s3 - s2,
        );
        let mut d2 = 0.0;
        for k in 0..3 {
            let x = h00 * r0[k] + h10 * h_s * v0[k] + h01 * r1[k] + h11 * h_s * v1[k];
            d2 += x * x;
        }
        d2.sqrt()
    };
    let d: Vec<f64> = (0..=N).map(|i| at(i as f64 / N as f64)).collect();
    let (best, &dmin) = d
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty");
    if best == 0 || best == N {
        return dmin;
    }
    // Parabola through the best sample and its neighbours.
    let (a, b, c) = (d[best - 1], d[best], d[best + 1]);
    let den = a - 2.0 * b + c;
    if den <= 0.0 {
        return dmin;
    }
    let off = 0.5 * (a - c) / den; // in samples, within +-0.5
    let s = (best as f64 + off.clamp(-1.0, 1.0)) / N as f64;
    at(s).min(dmin)
}

// ---------------------------------------------------------------- layers

fn drops<B: Backend>(
    backend: &mut B,
    ids: &[u32],
    stratum: impl Fn(usize) -> u8,
    out: &mut Vec<Frozen>,
) -> Result<(), RunError> {
    if let Some(d) = backend
        .dropped()
        .map_err(|e| RunError::Run(e.to_string()))?
    {
        for (i, &r) in d.reason.iter().enumerate() {
            if r != 0 {
                out.push(Frozen {
                    id: ids[i],
                    stratum: stratum(i),
                    reason: r,
                    drop_step: d.drop_step[i],
                });
            }
        }
    }
    Ok(())
}

/// Result of layer 1: final states and frozen particles.
pub struct L1Result {
    /// (id, state at `t0_s + 172800 s`).
    pub finals: Vec<(u32, [f64; 6])>,
    /// Frozen particles.
    pub frozen: Vec<Frozen>,
    /// Step size used, s.
    pub step_s: f64,
}

/// Runs layer 1: one backend per `t0_idx`.
pub fn layer1<B: Backend>(
    rows: &[L1Row],
    system: &System,
    variant: Variant,
    mut make: impl FnMut(&RunConfig, &System, &InitialState) -> Result<B, RunError>,
) -> Result<L1Result, RunError> {
    let (n, h) = (L1_STEPS, WINDOW_S / L1_STEPS as f64);
    let mut idxs: Vec<u32> = rows.iter().map(|r| r.t0_idx).collect();
    idxs.sort_unstable();
    idxs.dedup();
    let mut finals = Vec::with_capacity(rows.len());
    let mut frozen = Vec::new();
    for j in idxs {
        let group: Vec<&L1Row> = rows.iter().filter(|r| r.t0_idx == j).collect();
        let t0_s = group[0].t0_s;
        if group.iter().any(|r| r.t0_s != t0_s) {
            return Err(RunError::Setup(format!("t0_idx {j}: different t0_s")));
        }
        let mut p = Particles::with_capacity(group.len());
        for r in &group {
            p.push(
                r.id,
                Vec3::new(r.state[0], r.state[1], r.state[2]),
                Vec3::new(r.state[3], r.state[4], r.state[5]),
            );
        }
        let text = config_text(variant, group.len(), n, h, false);
        let config = setup(RunConfig::from_toml_str(&text))?;
        if config.time.n_steps() != n {
            return Err(RunError::Setup("n_steps mismatch".into()));
        }
        let init = InitialState {
            time_s: t0_s,
            particles: p,
        };
        let mut backend = make(&config, system, &init)?;
        backend
            .step(n)
            .map_err(|e| RunError::Run(format!("t0_idx {j}: {e}")))?;
        let ids: Vec<u32> = group.iter().map(|r| r.id).collect();
        let strata: Vec<u8> = group.iter().map(|r| r.stratum).collect();
        drops(&mut backend, &ids, |i| strata[i], &mut frozen)?;
        let view = backend
            .particles()
            .map_err(|e| RunError::Run(e.to_string()))?;
        let q = view.particles;
        for i in 0..q.len() {
            finals.push((
                q.id[i],
                [
                    q.x_km[i],
                    q.y_km[i],
                    q.z_km[i],
                    q.vx_km_s[i],
                    q.vy_km_s[i],
                    q.vz_km_s[i],
                ],
            ));
        }
    }
    Ok(L1Result {
        finals,
        frozen,
        step_s: h,
    })
}

/// Which particles to trace in layer 2 and how often.
pub struct Trace {
    /// Particle ids.
    pub ids: Vec<u32>,
    /// Sample every this many steps of the undivided step h.
    pub every_steps: u64,
}

/// One trace sample: id, time (s), inertial state, Prometheus position.
pub type TraceRow = (u32, f64, [f64; 6], [f64; 3]);

/// Result of layer 2.
pub struct L2Result {
    /// Trace samples (empty without a `Trace`).
    pub trace: Vec<TraceRow>,
    /// One row per particle.
    pub rows: Vec<L2Row>,
    /// Frozen particles.
    pub frozen: Vec<Frozen>,
    /// Steps and step size used.
    pub n_steps: u64,
    /// Step size, s.
    pub step_s: f64,
}

/// Particle states, Prometheus position (km) and velocity (km/s).
type Snap = (Vec<[f64; 6]>, [f64; 3], [f64; 3]);

fn vec3(a: Vec3<f64>) -> [f64; 3] {
    [a.x, a.y, a.z]
}

/// Runs layer 2: `days` from t = 0, one step at a time, measuring the
/// closest distance to Prometheus with a cubic Hermite between step ends.
pub fn layer2<B: Backend>(
    init_particles: Particles,
    days: f64,
    nominal_step_s: f64,
    div: u64,
    sample_every: u64,
    trace: Option<&Trace>,
    system: &System,
    variant: Variant,
    mut make: impl FnMut(&RunConfig, &System, &InitialState) -> Result<B, RunError>,
) -> Result<L2Result, RunError> {
    let (n, h) = exact_step(days * 86_400.0, nominal_step_s);
    let (n, h) = (n * div, h / div as f64);
    let count = init_particles.len();
    let text = config_text(variant, count, n, h, true);
    let config = setup(RunConfig::from_toml_str(&text))?;
    if config.time.n_steps() != n {
        return Err(RunError::Setup("n_steps mismatch".into()));
    }
    let prom = system
        .moons
        .tracks
        .iter()
        .position(|t| t.name == "prometheus")
        .ok_or_else(|| RunError::Setup("no Prometheus track".into()))?;
    let init = InitialState {
        time_s: 0.0,
        particles: init_particles,
    };
    let ids = init.particles.id.clone();
    let mut backend = make(&config, system, &init)?;
    let mut min_d = vec![f64::INFINITY; count];
    // Previous step end: particle state and Prometheus state.
    let snapshot = |b: &mut B| -> Result<Snap, RunError> {
        let m = b.moons().moons[prom];
        let view = b.particles().map_err(|e| RunError::Run(e.to_string()))?;
        let p = view.particles;
        Ok((
            (0..p.len())
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
                .collect(),
            vec3(m.position_km),
            vec3(m.velocity_km_s),
        ))
    };
    let rel = |s: &[f64; 6], mp: [f64; 3], mv: [f64; 3]| -> ([f64; 3], [f64; 3]) {
        (
            [s[0] - mp[0], s[1] - mp[1], s[2] - mp[2]],
            [s[3] - mv[0], s[4] - mv[1], s[5] - mv[2]],
        )
    };
    let (mut prev, mut pp, mut pv) = snapshot(&mut backend)?;
    for (i, s) in prev.iter().enumerate() {
        let (r, _) = rel(s, pp, pv);
        min_d[i] = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    }
    let mut live = vec![true; count];
    let mut last = prev.clone();
    let mut trace_rows: Vec<TraceRow> = Vec::new();
    let mut record = |step: u64, st: &[[f64; 6]], mp: [f64; 3]| {
        if let Some(t) = trace {
            for &id in &t.ids {
                if let Some(i) = ids.iter().position(|&x| x == id) {
                    trace_rows.push((id, step as f64 * h, st[i], mp));
                }
            }
        }
    };
    record(0, &prev, pp);
    for k in 0..n {
        backend
            .step(1)
            .map_err(|e| RunError::Run(format!("layer 2 step {k}: {e}")))?;
        let (cur, cp, cv) = snapshot(&mut backend)?;
        if let Some(t) = trace
            && (k + 1) % (t.every_steps * div) == 0
        {
            record(k + 1, &cur, cp);
        }
        if let Some(d) = backend
            .dropped()
            .map_err(|e| RunError::Run(e.to_string()))?
        {
            for (i, &r) in d.reason.iter().enumerate() {
                if r != 0 {
                    live[i] = false;
                }
            }
        }
        if (k + 1) % sample_every == 0 {
            for i in 0..count {
                if !live[i] {
                    continue;
                }
                let (r0, v0) = rel(&prev[i], pp, pv);
                let (r1, v1) = rel(&cur[i], cp, cv);
                let d = hermite_min_distance(r0, v0, r1, v1, h * sample_every as f64);
                if d < min_d[i] {
                    min_d[i] = d;
                }
            }
            prev = cur.clone();
            pp = cp;
            pv = cv;
        }
        last = cur;
    }
    let mut frozen = Vec::new();
    drops(&mut backend, &ids, |_| 255, &mut frozen)?;
    let chaos = backend
        .chaos()
        .map_err(|e| RunError::Run(e.to_string()))?
        .ok_or_else(|| RunError::Run("the backend returned no chaos indicator".into()))?;
    let rows = (0..count)
        .map(|i| L2Row {
            id: ids[i],
            min_prometheus_km: min_d[i],
            state: last[i],
            megno_mean: chaos.megno_mean[i],
        })
        .collect();
    Ok(L2Result {
        trace: trace_rows,
        rows,
        frozen,
        n_steps: n,
        step_s: h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation() {
        let mut base = Particles::default();
        for i in 0..20u32 {
            base.push(
                i,
                Vec3::new(1.0e5 + f64::from(i), -2.0e4, 3.0),
                Vec3::new(0.1, 16.0, 0.0),
            );
        }
        let mut zero = base.clone();
        perturb_positions(&mut zero, 0.0, 7);
        assert_eq!(zero, base, "x = 0 must change nothing");
        let (mut a, mut b, mut c) = (base.clone(), base.clone(), base.clone());
        perturb_positions(&mut a, 1e-6, 7);
        perturb_positions(&mut b, 1e-6, 7);
        perturb_positions(&mut c, 1e-6, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
        for i in 0..20 {
            let d = a.position_km(i) - base.position_km(i);
            let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            assert!((len - 1e-6).abs() < 1e-9, "length {len}");
            assert_eq!(a.velocity_km_s(i), base.velocity_km_s(i));
        }
    }

    #[test]
    fn exact_steps() {
        let (n, h) = exact_step(WINDOW_S, NOMINAL_STEP_S);
        assert_eq!(n, 485);
        assert!((h - 356.2886597938).abs() < 1e-6);
        let (n, h) = exact_step(365.25 * 86_400.0, NOMINAL_STEP_S);
        assert_eq!(n, 88_582);
        assert!((h - 356.2527).abs() < 1e-3);
        for (n, h) in [
            (485, WINDOW_S / 485.0),
            (88_582, 365.25 * 86_400.0 / 88_582.0),
        ] {
            let c = RunConfig::from_toml_str(&config_text(Variant::Plain, 3, n, h, true)).unwrap();
            assert_eq!(c.time.n_steps(), n);
        }
    }

    #[test]
    fn variants_make_valid_configs() {
        for v in [Variant::Plain, Variant::Substep, Variant::Loss] {
            RunConfig::from_toml_str(&config_text(v, 3, 485, 356.3, false)).unwrap();
        }
    }

    #[test]
    fn csv_round_trip() {
        let rows = vec![
            (4u32, [1.0 / 3.0, -2.5e5, 0.0, 1e-17, -16.25, 3.0e-3]),
            (1, [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            (3, [1.0; 6]),
            (0, [2.0; 6]),
            (2, [3.0; 6]),
        ];
        let text = format_l1(&rows, "test");
        let back = parse_numeric_csv(&text, 7, "t").unwrap();
        assert_eq!(back.len(), 5);
        let ids: Vec<f64> = back.iter().map(|r| r[0]).collect();
        assert_eq!(ids, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(back[4][1..], rows[0].1);
        let l2 = vec![
            L2Row {
                id: 2,
                min_prometheus_km: 12.5,
                state: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                megno_mean: 2.0,
            },
            L2Row {
                id: 0,
                min_prometheus_km: 1.0 / 7.0,
                state: [0.0; 6],
                megno_mean: 7.5,
            },
        ];
        let b2 = parse_numeric_csv(&format_l2(&l2, "x"), 9, "t").unwrap();
        assert_eq!(b2[0][0], 0.0);
        assert_eq!(b2[0][1], 1.0 / 7.0);
        assert_eq!(b2[1][8], 2.0);
        assert!(parse_numeric_csv("1,2\n", 3, "t").is_err());
    }

    #[test]
    fn hermite_straight_pass() {
        // Relative motion r(t) = (-20 + 0.04 t, 7, 0) km over 356 s: closest 7 km at t = 500 s,
        // so start the interval so that the minimum lies inside it.
        let (v, b, h) = (0.04, 7.0, 356.25);
        let t_start = 500.0 - 0.4 * h;
        let r = |t: f64| [-20.0 + v * t, b, 0.0];
        let d = hermite_min_distance(r(t_start), [v, 0.0, 0.0], r(t_start + h), [v, 0.0, 0.0], h);
        assert!((d - b).abs() < 1e-9, "{d}");
        // Minimum at the end of the interval.
        let d = hermite_min_distance(r(0.0), [v, 0.0, 0.0], r(h), [v, 0.0, 0.0], h);
        assert!((d - (r(h)[0].powi(2) + b * b).sqrt()).abs() < 1e-9);
    }
}
