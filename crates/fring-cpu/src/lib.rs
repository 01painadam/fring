//! f64 reference backend.
//!
//! Owner: A2 (see `AGENTS.md`).
//!
//! C1, C2: Saturn's field (point mass + J2 + J4 + J6) and the fixed-step
//! Wisdom-Holman integrator (ADR 0003; the plain leapfrog stays as a reference).
//! C4: the moons' direct and indirect pull in the kick (ADR 0004), from the
//! ephemeris, and rayon over the particles. The physics lives in
//! `fring_model::physics`; this crate holds the particle loop and the
//! [`CpuBackend`].
//!
//! # Parallel and deterministic
//!
//! One call of [`Backend::step`] works in blocks of at most [`BLOCK_STEPS`]
//! steps. For each block the moon positions and the indirect term are
//! computed once (a `KickTable`), then the particles run in parallel
//! chunks, each particle through the whole block. A particle never reads
//! another particle, so the result does not depend on the number of threads
//! or on the chunk size. Set the number of threads with `RAYON_NUM_THREADS`
//! or a rayon thread pool.

pub mod collide;
pub mod extended;
pub mod integrate;

use collide::CollisionState;
use extended::{ExtParams, LEVEL_UNKNOWN, MAX_MOONS, advance_ext};
use fring_model::Vec3;
use fring_model::config::ChaosConfig;
use fring_model::physics::chaos::{ChaosAccum, renormalise, shadow_offset_km};
use fring_model::physics::{ArcGeometry, KickTable, SaturnField, interpolate_moons};
use fring_model::{
    Backend, ChaosReport, DropReport, Error, InitialState, MoonEphemeris, MoonState, ParticleView,
    Particles, Result, RunConfig, StepStats, System,
};
use integrate::{State, advance_chunk, particle_energy};
use rayon::iter::Either;
use rayon::prelude::*;

/// Largest number of steps for which the moon table is made at once. The
/// table of 4096 steps for two moons is about 300 kB, so it stays in the
/// cache of every core.
pub const BLOCK_STEPS: u64 = 4096;

/// Particles per rayon task. Only the scheduling depends on it, not the result.
const CHUNK: usize = 128;

/// Shadow particles and sums of the chaos indicator (ICR-003, ADR 0008).
struct ChaosState {
    cfg: ChaosConfig,
    shadow: Vec<State>,
    acc: Vec<ChaosAccum>,
}

/// State of the keys `[loss]` and `[substep]` (ICR-004). `Some` only when a
/// key changes the behaviour; without it the old loop runs.
struct ExtState {
    freeze: bool,
    radius_rule: bool,
    /// `(accuracy, max_level)` with `[substep]`.
    substep: Option<(f64, u8)>,
    /// Set by `set_moon_radii_km`.
    radii_km: Option<Vec<f64>>,
    /// Per particle: 0 live, 1 unbound, 2 + k inside moon k.
    reason: Vec<u8>,
    /// Per particle: global step of the loss, `u64::MAX` if live.
    drop_step: Vec<u64>,
    /// Per particle: level of the next step (`LEVEL_UNKNOWN` at the start).
    level: Vec<u8>,
}

/// The CPU reference backend (f64, Saturn's field and the moons of the
/// ephemeris, particles in parallel).
pub struct CpuBackend {
    field: SaturnField<f64>,
    eph: MoonEphemeris,
    table: KickTable<f64>,
    step_s: f64,
    start_s: f64,
    steps_total: u64,
    ids: Vec<u32>,
    state: Vec<State>,
    /// Copy of `state` in the form of the interface; made by `particles`.
    snapshot: Particles,
    snapshot_stale: bool,
    /// Energy per unit mass of each particle at the start, km² s⁻²
    /// (only without moons; with moons energy is not conserved).
    energy0: Vec<f64>,
    /// Largest |dE/E| seen at any call of `stats`.
    max_rel_energy_error: f64,
    /// `Some` only with `[diagnostics.chaos]`; without it there is no extra work.
    chaos: Option<ChaosState>,
    /// `Some` only with `[loss]` or `[substep]` keys that change the behaviour.
    ext: Option<ExtState>,
    /// `Some` only with `domain.kind = "arc"` (D1-CPU). The step then runs in
    /// blocks of one step and the wrap follows every step.
    arc: Option<ArcGeometry>,
    /// `Some` only with `model = "stochastic"` (K2-CPU).
    collide: Option<CollisionState>,
}

/// Wraps every live particle into the arc at time `t_s` (the state after
/// global step `step`, counted from 1). `shadow` gets the same rotation as its
/// main particle (never a wrap of its own). A particle that cannot be wrapped
/// (more than `MAX_WRAPS` widths in one step, or not finite) is treated as an
/// unbound orbit, like the GPU kernel (`arc.wgsl` `arc_run`: `drop_particle`
/// with reason 1 and step `first_step + seg - 1`): with `freeze` it gets
/// reason 1 and `drop_step = step - 1`, else the call returns
/// `Error::Backend` with the lowest such particle.
#[allow(clippy::too_many_arguments)]
pub fn wrap_states(
    arc: &ArcGeometry,
    t_s: f64,
    step: u64,
    ids: &[u32],
    state: &mut [State],
    shadow: Option<&mut [State]>,
    mut ext: Option<(&mut [u8], &mut [u64], bool)>,
) -> Result<()> {
    const FAILED: i32 = i32::MAX;
    let live = ext.as_ref().map(|(r, _, _)| &**r);
    let ks: Vec<i32> = state
        .par_iter_mut()
        .enumerate()
        .map(|(i, s)| {
            if live.is_some_and(|r| r[i] != 0) {
                return 0;
            }
            let pos = Vec3::new(s[0], s[1], s[2]);
            let vel = Vec3::new(s[3], s[4], s[5]);
            match arc.wrap(pos, vel, t_s) {
                Some(w) => {
                    if w.k != 0 {
                        *s = [
                            w.pos_km.x,
                            w.pos_km.y,
                            w.pos_km.z,
                            w.vel_km_s.x,
                            w.vel_km_s.y,
                            w.vel_km_s.z,
                        ];
                    }
                    w.k
                }
                None => FAILED,
            }
        })
        .collect();
    if let Some(sh) = shadow {
        let (cw, sw) = arc.width_cos_sin();
        sh.par_iter_mut().zip(ks.par_iter()).for_each(|(s, &k)| {
            if k == 0 || k == FAILED {
                return;
            }
            // Same operations as the wrap of the main particle: |k| turns of
            // one width, k > 0 rotating by +width (see `ArcGeometry::wrap`).
            let sn = if k > 0 { sw } else { -sw };
            for _ in 0..k.unsigned_abs() {
                for o in [0, 3] {
                    let (x, y) = (s[o], s[o + 1]);
                    s[o] = cw * x - sn * y;
                    s[o + 1] = sn * x + cw * y;
                }
            }
        });
    }
    let Some(first) = ks.iter().position(|&k| k == FAILED) else {
        return Ok(());
    };
    if let Some((reason, drop_step, true)) = ext.as_mut() {
        for (i, _) in ks.iter().enumerate().filter(|(_, k)| **k == FAILED) {
            reason[i] = fring_model::DROP_UNBOUND;
            drop_step[i] = step - 1;
        }
        return Ok(());
    }
    Err(Error::Backend(format!(
        "particle {} (index {first}) has an unbound or invalid orbit",
        ids[first]
    )))
}

impl CpuBackend {
    /// Sets the radius of each moon in km, in the order of the `System`
    /// (ICR-004). The backend reads no files: `fring-run` reads
    /// `<params_dir>/moons.toml` with `moon_radii_from_toml_str`, matches the
    /// names and calls this. The radii are needed with `moon_radius =
    /// "params"` and with `[substep]` (the floor of the effective distance).
    ///
    /// Returns `Error::Config` if the length is not the number of moons or a
    /// radius is not finite and > 0.
    pub fn set_moon_radii_km(&mut self, radii_km: &[f64]) -> Result<()> {
        let n = self.eph.tracks.len();
        if radii_km.len() != n {
            return Err(Error::Config(format!(
                "set_moon_radii_km: {} radii for {n} moons",
                radii_km.len()
            )));
        }
        if let Some(r) = radii_km.iter().find(|r| !(r.is_finite() && **r > 0.0)) {
            return Err(Error::Config(format!(
                "set_moon_radii_km: radius {r} is not finite and > 0"
            )));
        }
        if let Some(e) = &mut self.ext {
            e.radii_km = Some(radii_km.to_vec());
        }
        Ok(())
    }

    /// The collision step after global step `step` (the count after the
    /// integrator step), if `step` is a multiple of `every_steps`.
    fn collide_after(&mut self, step: u64) -> Result<()> {
        let Some(c) = &mut self.collide else {
            return Ok(());
        };
        if !step.is_multiple_of(c.every) {
            return Ok(());
        }
        let reason = self.ext.as_ref().map(|e| e.reason.as_slice());
        let t_s = self.start_s + step as f64 * self.step_s;
        c.step(
            &self.field,
            t_s,
            step,
            self.step_s,
            &self.ids,
            reason,
            &mut self.state,
        )?;
        self.snapshot_stale = true;
        Ok(())
    }

    fn has_moons(&self) -> bool {
        !self.eph.tracks.is_empty()
    }

    /// One block with the extended loop (ICR-004). The chaos shadow is
    /// stepped with its main particle.
    fn step_block_ext(&mut self, first_step: u64, block: u64) -> Result<()> {
        let ext = self.ext.as_mut().expect("extended state");
        if (ext.radius_rule || ext.substep.is_some()) && ext.radii_km.is_none() {
            return Err(Error::Config(
                "moon_radius = \"params\" or [substep] needs the moon radii: call \
                 CpuBackend::set_moon_radii_km before the first step"
                    .to_string(),
            ));
        }
        self.table.fill_ext(
            &self.eph,
            self.start_s,
            self.step_s,
            first_step,
            block as usize,
        )?;
        let prm = ExtParams {
            field: &self.field,
            table: &self.table,
            h_s: self.step_s,
            first_step,
            radii_km: ext.radii_km.as_deref().unwrap_or(&[]),
            radius_rule: ext.radius_rule,
            substep: ext.substep,
        };
        let n_chunks = self.state.len().div_ceil(CHUNK);
        let shadows = match &mut self.chaos {
            Some(c) => Either::Left(c.shadow.par_chunks_mut(CHUNK).map(Some)),
            None => Either::Right((0..n_chunks).into_par_iter().map(|_| None)),
        };
        self.state
            .par_chunks_mut(CHUNK)
            .zip(ext.reason.par_chunks_mut(CHUNK))
            .zip(ext.drop_step.par_chunks_mut(CHUNK))
            .zip(ext.level.par_chunks_mut(CHUNK))
            .zip(shadows)
            .for_each(|((((st, re), dr), lv), mut sh)| {
                for i in 0..st.len() {
                    let shadow = sh.as_mut().map(|x| &mut x[i]);
                    advance_ext(&prm, &mut st[i], shadow, &mut re[i], &mut dr[i], &mut lv[i]);
                }
            });
        self.snapshot_stale = true;
        if !ext.freeze
            && let Some(index) = ext.reason.iter().position(|&r| r != 0)
        {
            return Err(Error::Backend(format!(
                "particle {} (index {index}) has an unbound or invalid orbit",
                self.ids[index]
            )));
        }
        if let Some(arc) = &self.arc {
            let step = first_step + block;
            let t_s = self.start_s + step as f64 * self.step_s;
            let shadow = self.chaos.as_mut().map(|c| c.shadow.as_mut_slice());
            let e = Some((
                ext.reason.as_mut_slice(),
                ext.drop_step.as_mut_slice(),
                ext.freeze,
            ));
            wrap_states(arc, t_s, step, &self.ids, &mut self.state, shadow, e)?;
        }
        if let Some(c) = &mut self.chaos
            && (first_step + block).is_multiple_of(u64::from(c.cfg.renorm_steps))
        {
            let d0 = c.cfg.offset_km;
            self.state
                .par_iter()
                .zip(c.shadow.par_iter_mut())
                .zip(c.acc.par_iter_mut())
                .zip(ext.reason.par_iter())
                .for_each(|(((m, sh), a), &r)| {
                    if r == 0 {
                        a.add(renormalise(m, sh, d0));
                    }
                });
        }
        Ok(())
    }

    fn sync_snapshot(&mut self) {
        if !self.snapshot_stale {
            return;
        }
        let p = &mut self.snapshot;
        for (i, s) in self.state.iter().enumerate() {
            p.x_km[i] = s[0];
            p.y_km[i] = s[1];
            p.z_km[i] = s[2];
            p.vx_km_s[i] = s[3];
            p.vy_km_s[i] = s[4];
            p.vz_km_s[i] = s[5];
        }
        self.snapshot_stale = false;
    }
}

impl Backend for CpuBackend {
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self> {
        config.validate()?;
        system.validate()?;
        init.particles.validate()?;
        let eph = &system.moons;
        if !eph.tracks.is_empty() {
            let end = eph.end_s();
            let run_end = init.time_s + config.time.duration_s();
            if init.time_s < eph.start_s || run_end > end {
                return Err(Error::Input(format!(
                    "the moon ephemeris covers [{}, {end}] s, the run needs [{}, {run_end}] s",
                    eph.start_s, init.time_s
                )));
            }
        }
        // ICR-011: the Wisdom-Holman drift is about mu_eff at a0.
        let field = SaturnField::<f64>::from_params(&system.saturn)
            .with_drift_radius(config.frame.reference_semi_major_axis_km);
        let collide = CollisionState::from_config(config, &field)?;
        let p = &init.particles;
        let mut state: Vec<State> = (0..p.len())
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
            .collect();
        // D1-CPU: a co-moving arc. Particles made all round the ring are
        // folded into the arc first, as `GpuBackend` does.
        let arc = ArcGeometry::from_domain_in_field(
            &config.domain,
            &field,
            config.frame.reference_semi_major_axis_km,
        )?;
        if let Some(g) = &arc {
            if config.moon_radius_rule() {
                return Err(Error::Config(
                    "[loss] moon_radius = \"params\" cannot be combined with the arc (ICR-004)"
                        .to_string(),
                ));
            }
            for (i, s) in state.iter_mut().enumerate() {
                let w = g
                    .fold(
                        Vec3::new(s[0], s[1], s[2]),
                        Vec3::new(s[3], s[4], s[5]),
                        init.time_s,
                    )
                    .ok_or_else(|| Error::Input(format!("particle {i} has no finite position")))?;
                *s = [
                    w.pos_km.x,
                    w.pos_km.y,
                    w.pos_km.z,
                    w.vel_km_s.x,
                    w.vel_km_s.y,
                    w.vel_km_s.z,
                ];
            }
        }
        let energy0 = if eph.tracks.is_empty() {
            (0..p.len())
                .map(|i| particle_energy(&field, p, i))
                .collect()
        } else {
            Vec::new()
        };
        let chaos = config
            .diagnostics
            .as_ref()
            .and_then(|d| d.chaos)
            .map(|cfg| {
                let shadow = state
                    .iter()
                    .map(|s: &State| {
                        let d = shadow_offset_km(Vec3::new(s[0], s[1], s[2]), cfg.offset_km);
                        [s[0] + d.x, s[1] + d.y, s[2] + d.z, s[3], s[4], s[5]]
                    })
                    .collect();
                ChaosState {
                    cfg,
                    shadow,
                    acc: vec![ChaosAccum::default(); p.len()],
                }
            });
        let ext = if config.loss_or_substep_active() {
            if eph.tracks.len() > MAX_MOONS {
                return Err(Error::Config(format!(
                    "[loss] and [substep] support at most {MAX_MOONS} moons, found {}",
                    eph.tracks.len()
                )));
            }
            Some(ExtState {
                freeze: config.freeze_unbound(),
                radius_rule: config.moon_radius_rule(),
                substep: config.substep.map(|s| (s.accuracy, s.max_level)),
                radii_km: None,
                reason: vec![0; p.len()],
                drop_step: vec![u64::MAX; p.len()],
                level: vec![LEVEL_UNKNOWN; p.len()],
            })
        } else {
            None
        };
        Ok(Self {
            field,
            table: KickTable::for_ephemeris(eph),
            eph: eph.clone(),
            step_s: config.time.step_s,
            start_s: init.time_s,
            steps_total: 0,
            ids: p.id.clone(),
            state,
            snapshot: p.clone(),
            snapshot_stale: arc.is_some(),
            energy0,
            max_rel_energy_error: 0.0,
            chaos,
            ext,
            arc,
            collide,
        })
    }

    /// Advances all particles. If a particle has an unbound or invalid orbit,
    /// the call returns `Error::Backend` with the lowest such id; the state
    /// of the backend is then unspecified and the run must stop.
    fn step(&mut self, n_steps: u64) -> Result<()> {
        let mut done = 0;
        while done < n_steps {
            let first_step = self.steps_total + done;
            let mut block = (n_steps - done).min(BLOCK_STEPS);
            if let Some(c) = &self.chaos {
                // Blocks end at multiples of K, so the renormalisation steps
                // do not depend on how the caller splits the run.
                let k = u64::from(c.cfg.renorm_steps);
                block = block.min(k - first_step % k);
            }
            if self.arc.is_some() {
                // The wrap follows every step (D1-CPU): `MAX_WRAPS` limits
                // the distance of one step, and the time of the wrap is a
                // synchronised time.
                block = 1;
            }
            if let Some(c) = &self.collide {
                // Blocks end at multiples of `every_steps`, so the collision
                // steps do not depend on how the caller splits the run.
                block = block.min(c.every - first_step % c.every);
            }
            if self.ext.is_some() {
                self.step_block_ext(first_step, block)?;
                self.collide_after(first_step + block)?;
                done += block;
                continue;
            }
            let moons = self.has_moons();
            if moons {
                self.table.fill(
                    &self.eph,
                    self.start_s,
                    self.step_s,
                    first_step,
                    block as usize,
                )?;
            }
            let (field, h) = (&self.field, self.step_s);
            let table = moons.then_some(&self.table);
            let mut failed = advance_all(field, table, &mut self.state, h, block);
            if let Some(c) = &mut self.chaos {
                let f2 = advance_all(field, table, &mut c.shadow, h, block);
                failed = failed.or(f2);
            }
            self.snapshot_stale = true;
            if let Some(index) = failed {
                return Err(Error::Backend(format!(
                    "particle {} (index {index}) has an unbound or invalid orbit",
                    self.ids[index]
                )));
            }
            if let Some(arc) = &self.arc {
                let step = first_step + block;
                let t_s = self.start_s + step as f64 * self.step_s;
                let shadow = self.chaos.as_mut().map(|c| c.shadow.as_mut_slice());
                wrap_states(arc, t_s, step, &self.ids, &mut self.state, shadow, None)?;
            }
            if let Some(c) = &mut self.chaos
                && (first_step + block).is_multiple_of(u64::from(c.cfg.renorm_steps))
            {
                let d0 = c.cfg.offset_km;
                self.state
                    .par_iter()
                    .zip(c.shadow.par_iter_mut())
                    .zip(c.acc.par_iter_mut())
                    .for_each(|((m, sh), a)| a.add(renormalise(m, sh, d0)));
            }
            self.collide_after(first_step + block)?;
            done += block;
        }
        self.steps_total += n_steps;
        Ok(())
    }

    fn time_s(&self) -> f64 {
        self.start_s + self.steps_total as f64 * self.step_s
    }

    /// Moon states from the ephemeris at the current time (empty without
    /// moons, or if the time is past the end of the ephemeris).
    fn moons(&self) -> MoonState {
        let t = self.time_s();
        interpolate_moons(&self.eph, t).unwrap_or(MoonState {
            time_s: t,
            moons: Vec::new(),
        })
    }

    fn particles(&mut self) -> Result<ParticleView<'_>> {
        self.sync_snapshot();
        Ok(ParticleView {
            time_s: self.time_s(),
            particles: &self.snapshot,
        })
    }

    fn chaos(&mut self) -> Result<Option<ChaosReport>> {
        let Some(c) = &self.chaos else {
            return Ok(None);
        };
        let interval_s = f64::from(c.cfg.renorm_steps) * self.step_s;
        Ok(Some(ChaosReport {
            time_s: self.time_s(),
            n_renorm: c.acc.first().map_or(0, |a| a.j),
            ftle_per_s: c.acc.iter().map(|a| a.ftle_per_s(interval_s)).collect(),
            log_growth: c.acc.iter().map(|a| a.l).collect(),
            megno_mean: c.acc.iter().map(|a| a.megno_mean()).collect(),
            flags: c
                .acc
                .iter()
                .enumerate()
                .map(|(i, a)| match &self.ext {
                    // Bit 1: dropped (ICR-004).
                    Some(e) if e.reason[i] != 0 => a.flags | 2,
                    _ => a.flags,
                })
                .collect(),
        }))
    }

    fn dropped(&mut self) -> Result<Option<DropReport>> {
        let Some(e) = self.ext.as_ref().filter(|e| e.freeze) else {
            return Ok(None);
        };
        let count = |code: u8| e.reason.iter().filter(|&&r| r == code).count() as u64;
        Ok(Some(DropReport {
            time_s: self.time_s(),
            reason: e.reason.clone(),
            drop_step: e.drop_step.clone(),
            n_unbound: count(fring_model::DROP_UNBOUND),
            n_inside_moon: (0..self.eph.tracks.len())
                .map(|k| count(fring_model::drop_inside_moon(k)))
                .collect(),
        }))
    }

    fn stats(&mut self) -> Result<StepStats> {
        // Computed here, not at every step, to keep the step loop cheap.
        let max_rel_energy_error = if self.has_moons() || self.collide.is_some() {
            None
        } else {
            self.sync_snapshot();
            for (i, &e0) in self.energy0.iter().enumerate() {
                let e = particle_energy(&self.field, &self.snapshot, i);
                self.max_rel_energy_error = self.max_rel_energy_error.max(((e - e0) / e0).abs());
            }
            Some(self.max_rel_energy_error)
        };
        // Collision sums since the previous call (ICR-009).
        let tally = self
            .collide
            .as_mut()
            .map(|c| std::mem::take(&mut c.tally))
            .unwrap_or_default();
        Ok(StepStats {
            time_s: self.time_s(),
            steps_total: self.steps_total,
            max_rel_energy_error,
            collisions_since_last: tally.collisions,
            collision: tally.stats,
        })
    }
}

/// Advances all states by one block in parallel chunks. Returns the index of
/// the lowest particle with an unbound or invalid orbit.
fn advance_all(
    field: &SaturnField<f64>,
    table: Option<&KickTable<f64>>,
    state: &mut [State],
    h: f64,
    block: u64,
) -> Option<usize> {
    let failed: Vec<Option<usize>> = state
        .par_chunks_mut(CHUNK)
        .map(|chunk| advance_chunk(field, table, chunk, h, block))
        .collect();
    failed
        .iter()
        .enumerate()
        .find_map(|(c, f)| f.map(|i| c * CHUNK + i))
}
