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

pub mod integrate;

use fring_model::Vec3;
use fring_model::config::ChaosConfig;
use fring_model::physics::chaos::{ChaosAccum, renormalise, shadow_offset_km};
use fring_model::physics::{KickTable, SaturnField, interpolate_moons};
use fring_model::{
    Backend, ChaosReport, Error, InitialState, MoonEphemeris, MoonState, ParticleView, Particles,
    Result, RunConfig, StepStats, System,
};
use integrate::{State, advance_chunk, particle_energy};
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
}

impl CpuBackend {
    fn has_moons(&self) -> bool {
        !self.eph.tracks.is_empty()
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
        let field = SaturnField::<f64>::from_params(&system.saturn);
        let p = &init.particles;
        let state: Vec<State> = (0..p.len())
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
            snapshot_stale: false,
            energy0,
            max_rel_energy_error: 0.0,
            chaos,
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
            if let Some(c) = &mut self.chaos {
                if (first_step + block) % u64::from(c.cfg.renorm_steps) == 0 {
                    let d0 = c.cfg.offset_km;
                    self.state
                        .par_iter()
                        .zip(c.shadow.par_iter_mut())
                        .zip(c.acc.par_iter_mut())
                        .for_each(|((m, sh), a)| a.add(renormalise(m, sh, d0)));
                }
            }
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
            flags: c.acc.iter().map(|a| a.flags).collect(),
        }))
    }

    fn stats(&mut self) -> Result<StepStats> {
        // Computed here, not at every step, to keep the step loop cheap.
        let max_rel_energy_error = if self.has_moons() {
            None
        } else {
            self.sync_snapshot();
            for (i, &e0) in self.energy0.iter().enumerate() {
                let e = particle_energy(&self.field, &self.snapshot, i);
                self.max_rel_energy_error = self.max_rel_energy_error.max(((e - e0) / e0).abs());
            }
            Some(self.max_rel_energy_error)
        };
        Ok(StepStats {
            time_s: self.time_s(),
            steps_total: self.steps_total,
            max_rel_energy_error,
            ..Default::default()
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
