//! K2-CPU and K3-CPU: the collision steps of `fring-collide` on the CPU state.
//!
//! K3 (hard spheres) runs at every integrator step, serially (see
//! [`CollisionState::step`]); it has no random numbers, so the result is the
//! same for any thread count and any split of a run. The rest of this text is
//! about K2 (stochastic), which is parallel and deterministic.
//!
//! The K2 step acts on the synchronised state at the end of a block, every
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
use fring_collide::hard_sphere::HardSphereModel;
use fring_collide::stochastic::{CellList, CellScratch, Domain, StochasticModel};
use fring_collide::tally::Tally;
use fring_model::config::Collisions;
use fring_model::physics::ArcGeometry;
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
    /// Hard spheres: the mask of active particles (parallel).
    pub mask_s: f64,
    /// Hard spheres: `find_pairs` (cell list and neighbour search, serial).
    pub find_s: f64,
    /// Hard spheres: `resolve_pairs` (serial).
    pub resolve_s: f64,
    /// Hard spheres: copying velocities in and out for the pair members.
    pub write_s: f64,
    /// Cell keys of all active particles (parallel).
    pub keys_s: f64,
    /// `CellList::group` (serial sort).
    pub group_s: f64,
    /// Pair selection and collisions in the cells (parallel).
    pub cells_s: f64,
    /// Adding the velocity changes (serial).
    pub apply_s: f64,
}

/// The model with its domain (and, for hard spheres, work space).
enum Kind {
    Stochastic {
        model: StochasticModel,
        domain: Domain,
    },
    HardSphere {
        model: HardSphereModel,
        domain: Domain,
        /// `reason == 0` for each particle.
        active: Vec<bool>,
        /// Velocities of the pair members (other entries are not used).
        vel: Vec<Vec3<f64>>,
    },
}

/// Collision model, domain and the tally since the last `stats()` call.
pub struct CollisionState {
    kind: Kind,
    /// Collision step every this many integrator steps (always 1 for hard
    /// spheres).
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
            Collisions::HardSphere { .. } => 1,
            Collisions::Stochastic { every_steps, .. } => every_steps,
        };
        let name = match config.collisions {
            Collisions::HardSphere { .. } => "hard_sphere",
            _ => "stochastic",
        };
        if config
            .diagnostics
            .as_ref()
            .is_some_and(|d| d.chaos.is_some())
        {
            return Err(Error::Config(format!(
                "[diagnostics.chaos] cannot be used with collisions.model = \"{name}\": \
                 a shadow particle has no meaning under collisions, because the main \
                 particle and its shadow would get different velocity changes and the \
                 separation (and so MEGNO) would measure the collisions (A0 decision, \
                 2026-11-08)"
            )));
        }
        let a0 = config.frame.reference_semi_major_axis_km;
        let n0 = reference_mean_motion(field, a0);
        let bad = || Error::Config("collisions: invalid cell sizes or reference orbit".into());
        let arc = ArcGeometry::from_domain(&config.domain, n0)?;
        let kind = match config.collisions {
            Collisions::HardSphere { .. } => {
                let model = HardSphereModel::from_config(&config.collisions).ok_or_else(bad)?;
                // The cell grid of K3 counts the cells along the ring in a u32:
                // 2 pi a0 / (2 s) must stay below 2^32 (s of about 10 cm at the
                // F ring). Smaller particles need the arc domain (D1-CPU).
                let domain = match arc {
                    Some(g) => model.arc_domain(a0, g).ok_or_else(bad)?,
                    None => model.full_ring_domain(a0, n0).ok_or_else(|| {
                        Error::Config(format!(
                            "collisions.particle_radius_km is too small for the full ring: the \
                         cell grid needs 2 pi a0 / (2 s) < 2^32 cells along the ring \
                         (s >= {:.2e} km); use domain.kind = \"arc\"",
                            std::f64::consts::TAU * a0 / (2.0 * f64::from(u32::MAX))
                        ))
                    })?,
                };
                Kind::HardSphere {
                    model,
                    domain,
                    active: Vec::new(),
                    vel: Vec::new(),
                }
            }
            _ => {
                let domain = match (arc, &config.collisions) {
                    (
                        Some(g),
                        Collisions::Stochastic {
                            cell_radial_km,
                            cell_azimuthal_km,
                            cell_vertical_km,
                            ..
                        },
                    ) => Domain::arc_domain(
                        *cell_radial_km,
                        *cell_azimuthal_km,
                        *cell_vertical_km,
                        a0,
                        g,
                    ),
                    _ => Domain::full_ring_from_config(&config.collisions, a0, n0),
                }
                .ok_or_else(bad)?;
                let model = StochasticModel::from_config(&config.collisions, config.run.seed)
                    .ok_or_else(bad)?;
                Kind::Stochastic { model, domain }
            }
        };
        Ok(Some(Self {
            kind,
            every,
            tally: Tally::default(),
            last_times: StageTimes::default(),
        }))
    }

    /// One collision step on the synchronised `state` at time `t_s` (seconds
    /// since the epoch) after global step `step`. `reason[i] != 0` (dropped or
    /// frozen) leaves particle `i` out; `None` means all take part.
    ///
    /// Hard spheres (K3): pairs are found with `find_pairs` and resolved one
    /// after the other with `resolve_pairs`, at every step. Both are serial
    /// (the pairs share particles and the order matters), and use no random
    /// numbers.
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
        let (model, domain) = match &mut self.kind {
            Kind::Stochastic { model, domain } => (model as &StochasticModel, domain as &Domain),
            Kind::HardSphere { .. } => {
                let (tally, times) = self.step_hard_sphere(t_s, step_s, ids, reason, state);
                self.tally += tally;
                self.last_times = times;
                return Ok(());
            }
        };
        let dt_s = self.every as f64 * step_s;
        let mut times = StageTimes::default();

        let t0 = Instant::now();
        let items: Vec<_> = (0..state.len())
            .into_par_iter()
            .filter(|&i| reason.is_none_or(|r| r[i] == 0))
            .map(|i| {
                let s = &state[i];
                (
                    domain.cell_key(Vec3::new(s[0], s[1], s[2]), t_s),
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
        let volume = domain.grid.volume_km3();
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

    /// The hard-sphere step on the synchronised state at time `t_s`.
    fn step_hard_sphere(
        &mut self,
        t_s: f64,
        step_s: f64,
        ids: &[u32],
        reason: Option<&[u8]>,
        state: &mut [State],
    ) -> (Tally, StageTimes) {
        let Kind::HardSphere {
            model,
            domain,
            active,
            vel,
        } = &mut self.kind
        else {
            unreachable!("step_hard_sphere is only called for hard spheres");
        };
        let mut times = StageTimes::default();

        let t0 = Instant::now();
        active.resize(state.len(), true);
        vel.resize(state.len(), Vec3::new(0.0, 0.0, 0.0));
        match reason {
            Some(r) => active
                .par_iter_mut()
                .zip(r.par_iter())
                .for_each(|(a, &r)| *a = r == 0),
            None => active.fill(true),
        }
        times.mask_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let st: &[State] = state;
        let pos = |i: usize| Vec3::new(st[i][0], st[i][1], st[i][2]);
        let pairs = model.find_pairs(domain, t_s, ids, Some(active), pos);
        times.find_s = t1.elapsed().as_secs_f64();
        if pairs.is_empty() {
            return (Tally::default(), times);
        }

        let t2 = Instant::now();
        for p in &pairs {
            for k in [p.i as usize, p.j as usize] {
                vel[k] = Vec3::new(st[k][3], st[k][4], st[k][5]);
            }
        }
        times.write_s = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        // In an arc the partner of a seam pair is turned by `shift` widths.
        let arc = domain.arc.map(|a| (a.width_cos_sin(), a.width_rad()));
        let separation = |i: usize, j: usize, shift: i8| match arc {
            Some(((cw, sw), _)) if shift != 0 => {
                let (c, s) = (cw, sw * f64::from(shift));
                let q = pos(j);
                pos(i) - Vec3::new(c * q.x - s * q.y, s * q.x + c * q.y, q.z)
            }
            _ => pos(i) - pos(j),
        };
        let tally = model.resolve_pairs(&pairs, step_s, arc.map(|(_, w)| w), separation, vel);
        times.resolve_s = t3.elapsed().as_secs_f64();

        let t4 = Instant::now();
        for p in &pairs {
            for k in [p.i as usize, p.j as usize] {
                state[k][3] = vel[k].x;
                state[k][4] = vel[k].y;
                state[k][5] = vel[k].z;
            }
        }
        times.write_s += t4.elapsed().as_secs_f64();
        (tally, times)
    }
}
