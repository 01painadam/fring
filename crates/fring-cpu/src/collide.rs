//! K2-CPU: the stochastic collision step of `fring-collide` on the CPU state,
//! in parallel and deterministic.
//!
//! The step acts on the synchronised state at the end of a block, every
//! `every_steps` integrator steps (`SPEC.md` 7). The result does not depend on
//! the number of threads or on how the caller splits a run:
//!
//! - the random numbers depend only on the smallest id of a cell and on the
//!   global step (`SPEC.md` 5.5), not on the cell order;
//! - the cells are processed in parallel, but the velocity changes are added
//!   in cell order, and each particle is in one cell only;
//! - the tallies are summed over fixed chunks of [`CELL_CHUNK`] cells in index
//!   order, so the f64 energy sum does not depend on the thread count.

use std::time::Instant;

use crate::integrate::State;
use fring_collide::stochastic::{CellList, CellScratch, Domain, StochasticModel};
use fring_collide::tally::Tally;
use fring_model::config::Collisions;
use fring_model::physics::SaturnField;
use fring_model::physics::frames::reference_mean_motion;
use fring_model::{Error, Result, RunConfig, Vec3};
use rayon::prelude::*;

/// Tally and velocity changes `(particle index, Δv)` of a chunk of cells.
type ChunkResult = (Tally, Vec<(u32, Vec3<f64>)>);

/// Cells per task. The tally sum is done per chunk, so this number is part of
/// the result (the f64 rounding), and must not depend on the thread count.
pub const CELL_CHUNK: usize = 256;

/// Wall time of the stages of one collision step, in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// Cell keys of all active particles (parallel).
    pub keys_s: f64,
    /// `CellList::group` (serial sort).
    pub group_s: f64,
    /// Pair selection and collisions in the cells (parallel).
    pub cells_s: f64,
    /// Adding the velocity changes (serial).
    pub apply_s: f64,
}

/// Collision model, domain and the tally since the last `stats()` call.
pub struct CollisionState {
    model: StochasticModel,
    domain: Domain,
    /// Collision step every this many integrator steps.
    pub every: u64,
    /// Sums since the previous `Backend::stats` call.
    pub tally: Tally,
    /// Stage times of the last collision step.
    pub last_times: StageTimes,
}

impl CollisionState {
    /// `None` for `model = "none"`. `Error::Config` for the models and
    /// combinations that the CPU does not support yet.
    pub fn from_config(config: &RunConfig, field: &SaturnField<f64>) -> Result<Option<Self>> {
        let every = match config.collisions {
            Collisions::None => return Ok(None),
            Collisions::HardSphere { .. } => {
                return Err(Error::Config(
                    "collisions.model = \"hard_sphere\" is not implemented on the CPU backend \
                     yet (task K3-CPU)"
                        .to_string(),
                ));
            }
            Collisions::Stochastic { every_steps, .. } => every_steps,
        };
        if matches!(config.domain, fring_model::config::Domain::Arc { .. }) {
            return Err(Error::Config(
                "collisions with domain.kind = \"arc\" are not implemented on the CPU backend \
                 yet (task D1-CPU)"
                    .to_string(),
            ));
        }
        if config
            .diagnostics
            .as_ref()
            .is_some_and(|d| d.chaos.is_some())
        {
            return Err(Error::Config(
                "[diagnostics.chaos] cannot be used with collisions.model = \"stochastic\": \
                 a shadow particle has no meaning under random velocity kicks, because the \
                 main particle and its shadow would get different kicks and the separation \
                 (and so MEGNO) would measure the noise (A0 decision, 2026-11-08)"
                    .to_string(),
            ));
        }
        let a0 = config.frame.reference_semi_major_axis_km;
        let n0 = reference_mean_motion(field, a0);
        let bad = || Error::Config("collisions: invalid cell sizes or reference orbit".into());
        let domain = Domain::full_ring_from_config(&config.collisions, a0, n0).ok_or_else(bad)?;
        let model =
            StochasticModel::from_config(&config.collisions, config.run.seed).ok_or_else(bad)?;
        Ok(Some(Self {
            model,
            domain,
            every,
            tally: Tally::default(),
            last_times: StageTimes::default(),
        }))
    }

    /// One collision step on the synchronised `state` at time `t_s` (seconds
    /// since the epoch) after global step `step`. `reason[i] != 0` (dropped or
    /// frozen) leaves particle `i` out; `None` means all take part.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        field: &SaturnField<f64>,
        t_s: f64,
        step: u64,
        step_s: f64,
        ids: &[u32],
        reason: Option<&[u8]>,
        state: &mut [State],
    ) -> Result<()> {
        let dt_s = self.every as f64 * step_s;
        let mut times = StageTimes::default();

        let t0 = Instant::now();
        let items: Vec<_> = (0..state.len())
            .into_par_iter()
            .filter(|&i| reason.is_none_or(|r| r[i] == 0))
            .map(|i| {
                let s = &state[i];
                (
                    self.domain.cell_key(Vec3::new(s[0], s[1], s[2]), t_s),
                    ids[i],
                    i as u32,
                )
            })
            .collect();
        times.keys_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let cells = CellList::group(items);
        times.group_s = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let volume = self.domain.grid.volume_km3();
        let model = &self.model;
        let st: &[State] = state;
        let n_chunks = cells.len().div_ceil(CELL_CHUNK);
        let results: Vec<Result<ChunkResult>> = (0..n_chunks)
            .into_par_iter()
            .map(|ch| {
                let mut scratch = CellScratch::default();
                let mut tally = Tally::default();
                let mut deltas = Vec::new();
                let end = ((ch + 1) * CELL_CHUNK).min(cells.len());
                for c in ch * CELL_CHUNK..end {
                    let members = cells.members(c);
                    tally += model
                        .process_cell_states(
                            field,
                            volume,
                            dt_s,
                            step,
                            ids,
                            st,
                            members,
                            &mut scratch,
                        )
                        .map_err(|e| Error::Backend(format!("collision step {step}: {e}")))?;
                    deltas.extend(
                        members
                            .iter()
                            .copied()
                            .zip(scratch.delta_km_s.iter().copied()),
                    );
                }
                Ok((tally, deltas))
            })
            .collect();
        times.cells_s = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let mut total = Tally::default();
        for r in results {
            let (tally, deltas) = r?;
            total += tally;
            for (m, d) in deltas {
                let s = &mut state[m as usize];
                s[3] += d.x;
                s[4] += d.y;
                s[5] += d.z;
            }
        }
        times.apply_s = t3.elapsed().as_secs_f64();
        self.tally += total;
        self.last_times = times;
        Ok(())
    }
}
