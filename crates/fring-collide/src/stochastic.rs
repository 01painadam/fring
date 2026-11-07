//! The stochastic superparticle model (`SPEC.md` section 5), K2.
//!
//! A step has three parts, and each part is public so that a backend can run
//! the cells in parallel:
//!
//! 1. [`Domain::build_cells`]: sort the active particles into cells. Each cell
//!    lists its members in order of particle id.
//! 2. [`StochasticModel::process_cell_states`]: pick and collide pairs in one
//!    cell. It reads the state and writes the velocity changes to a
//!    [`CellScratch`]; it changes nothing else.
//! 3. The caller adds the velocity changes to the state.
//!
//! [`StochasticModel::collide_step`] does all three, one cell after the other.
//!
//! The random numbers of a cell depend only on the seed, the step and the
//! smallest particle id of the cell (`SPEC.md` 5.5). So the result does not
//! depend on the order in which the cells are processed.
//!
//! (Rust note: a function that takes `&self` and a scratch buffer, and returns
//! its result in the buffer, can run for many cells at the same time. Only
//! the final step, which writes to the state, needs `&mut`.)

use std::f64::consts::TAU;
use std::fmt;

use fring_model::Vec3;
use fring_model::config::{Collisions, Restitution};
use fring_model::physics::frames::{inertial_to_rotating, rotating_to_local};
use fring_model::physics::gravity::SaturnField;
use fring_model::rng::{Counter, Philox, Stream, mulhilo, u01_f64};

use crate::frame::{CellGrid, CellKey, peculiar_velocity};
use crate::pair;
use crate::tally::Tally;

/// Candidates in one cell and step, at most (`SPEC.md` 5.5: the `draw` field
/// of the Philox counter has 16 bits and a candidate uses two draws).
pub const MAX_CANDIDATES: u64 = 32_767;

/// A state of one particle: Saturn-centred inertial position (km) and
/// velocity (km/s), `[x, y, z, vx, vy, vz]`. The type of `fring-cpu`.
pub type State = [f64; 6];

/// A cell needs more candidates than the random numbers allow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollideError {
    /// Smallest particle id in the cell.
    pub smallest_id: u32,
    /// Members of the cell.
    pub members: usize,
    /// Expected number of candidates, M of `SPEC.md` 5.3.
    pub expected_candidates: f64,
}

impl fmt::Display for CollideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "collision cell of particle {} has {} members and needs about {:.0} candidates \
             in one step (limit {MAX_CANDIDATES}): the cells are too large",
            self.smallest_id, self.members, self.expected_candidates
        )
    }
}

impl std::error::Error for CollideError {}

/// Cells with their members: particle indices in order of cell, then of
/// particle id.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CellList {
    order: Vec<u32>,
    starts: Vec<usize>,
}

impl CellList {
    /// Groups items `(cell key, particle id, particle index)`. The cells are
    /// in the order of their keys and the members in the order of their ids.
    pub fn group<K: Ord + Copy>(mut items: Vec<(K, u32, u32)>) -> Self {
        items.sort_unstable_by_key(|&(k, id, _)| (k, id));
        let mut order = Vec::with_capacity(items.len());
        let mut starts = Vec::new();
        for (n, &(k, _, index)) in items.iter().enumerate() {
            if n == 0 || items[n - 1].0 != k {
                starts.push(n);
            }
            order.push(index);
        }
        starts.push(items.len());
        Self { order, starts }
    }

    /// Number of non-empty cells.
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// True if no particle is in a cell.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indices (into the caller's arrays) of the members of cell `c`, in order
    /// of particle id.
    pub fn members(&self, c: usize) -> &[u32] {
        &self.order[self.starts[c]..self.starts[c + 1]]
    }
}

/// The region that the cells fill, in the rotating frame (`SPEC.md` 5.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    /// The cell sizes. Its `domain_length_km` is D.
    pub grid: CellGrid,
    /// Radius a₀ of the reference circle in km.
    pub a0_km: f64,
    /// Mean motion n₀ of the rotating frame in rad/s
    /// (`physics::frames::reference_mean_motion`).
    pub n0_rad_s: f64,
    /// Longitude φ₀ of the start of the domain in the rotating frame, rad:
    /// 0 for the full ring (`SPEC.md` 5.1).
    pub phi0_rad: f64,
}

impl Domain {
    /// The full ring: φ₀ = 0 and D = 2π a₀. `None` if a size is not finite
    /// and > 0.
    pub fn full_ring(
        radial_km: f64,
        azimuthal_km: f64,
        vertical_km: f64,
        a0_km: f64,
        n0_rad_s: f64,
    ) -> Option<Self> {
        let ok = |v: f64| v.is_finite() && v > 0.0;
        if !(ok(a0_km) && ok(n0_rad_s)) {
            return None;
        }
        Some(Self {
            grid: CellGrid::new(radial_km, azimuthal_km, vertical_km, TAU * a0_km)?,
            a0_km,
            n0_rad_s,
            phi0_rad: 0.0,
        })
    }

    /// The full ring with the cell sizes of `[collisions]`. `None` for
    /// another collision model or for invalid numbers.
    pub fn full_ring_from_config(c: &Collisions, a0_km: f64, n0_rad_s: f64) -> Option<Self> {
        match *c {
            Collisions::Stochastic {
                cell_radial_km,
                cell_azimuthal_km,
                cell_vertical_km,
                ..
            } => Self::full_ring(
                cell_radial_km,
                cell_azimuthal_km,
                cell_vertical_km,
                a0_km,
                n0_rad_s,
            ),
            _ => None,
        }
    }

    /// The cell of an inertial position at time `t_s` (seconds since the
    /// epoch): to the rotating frame with θ = n₀ t reduced to [0, 2π), to local
    /// coordinates, then `y` counted from φ₀ (`SPEC.md` 5.1).
    pub fn cell_key(&self, pos_km: Vec3<f64>, t_s: f64) -> CellKey {
        // Reduce θ in f64 before the trigonometry: n₀ t is about 10⁴ rad after
        // a year, and the reduction keeps the rounding of θ small.
        let theta = (self.n0_rad_s * t_s).rem_euclid(TAU);
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let (p_rot, _) = inertial_to_rotating(pos_km, zero, self.n0_rad_s, theta / self.n0_rad_s);
        let l = rotating_to_local(p_rot, zero, self.a0_km);
        self.grid
            .key(l.x_km, l.y_km - self.a0_km * self.phi0_rad, l.z_km)
    }

    /// Sorts the particles into cells. `ids[i]` is the id of particle `i`;
    /// `active[i]` false (dropped or frozen, `SPEC.md` 5.1) leaves it out;
    /// `None` means all are active. `position(i)` is its inertial position.
    pub fn build_cells(
        &self,
        t_s: f64,
        ids: &[u32],
        active: Option<&[bool]>,
        position: impl Fn(usize) -> Vec3<f64>,
    ) -> CellList {
        let items = (0..ids.len())
            .filter(|&i| active.is_none_or(|a| a[i]))
            .map(|i| (self.cell_key(position(i), t_s), ids[i], i as u32))
            .collect();
        CellList::group(items)
    }
}

/// Work space of [`StochasticModel::process_cell_states`]. One per thread; the
/// result of the last call is in `delta_km_s`.
#[derive(Clone, Debug, Default)]
pub struct CellScratch {
    /// Peculiar velocities of the members (changed by the collisions).
    pub u_km_s: Vec<Vec3<f64>>,
    /// Velocity change of each member, in the order of the member list.
    pub delta_km_s: Vec<Vec3<f64>>,
    ids: Vec<u32>,
}

/// The model parameters (`SPEC.md` 3, 6).
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticModel {
    particle_radius_km: f64,
    weight: f64,
    restitution: Restitution,
    rng: Philox,
}

impl StochasticModel {
    /// A model with the radius s of a real particle (km), the weight W, the
    /// restitution law and the run seed.
    pub fn new(particle_radius_km: f64, weight: f64, restitution: Restitution, seed: u64) -> Self {
        Self {
            particle_radius_km,
            weight,
            restitution,
            rng: Philox::from_seed(seed),
        }
    }

    /// The model of `[collisions]`. `None` for another collision model.
    pub fn from_config(c: &Collisions, seed: u64) -> Option<Self> {
        match *c {
            Collisions::Stochastic {
                particle_radius_km,
                weight,
                restitution,
                ..
            } => Some(Self::new(particle_radius_km, weight, restitution, seed)),
            _ => None,
        }
    }

    /// σ = π (2s)² in km².
    pub fn cross_section_km2(&self) -> f64 {
        std::f64::consts::PI * (2.0 * self.particle_radius_km).powi(2)
    }

    /// Pair selection and collisions in one cell (`SPEC.md` 5.3 and 5.4).
    ///
    /// `ids` are the particle ids of the members in ascending order; `u` their
    /// peculiar velocities, changed in place; `delta` receives the velocity
    /// change of each member (added to what it holds). `step` is the step
    /// count of the run after the integrator step (`SPEC.md` 5.5).
    ///
    /// Returns the sums for this cell. Fails if the cell needs more than
    /// [`MAX_CANDIDATES`] candidates.
    pub fn process_cell(
        &self,
        volume_km3: f64,
        dt_s: f64,
        step: u64,
        ids: &[u32],
        u: &mut [Vec3<f64>],
        delta: &mut [Vec3<f64>],
    ) -> Result<Tally, CollideError> {
        let n = ids.len();
        debug_assert!(u.len() == n && delta.len() == n);
        debug_assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids must ascend");
        let mut tally = Tally::default();
        if n == 0 {
            return Ok(tally);
        }
        let s = &mut tally.stats;
        s.occupied_cells = 1;
        s.occupancy_sum = n as u64;
        s.occupancy_sq_sum = (n * n) as u64;
        if n < 2 {
            s.lonely_particles = n as u64;
            return Ok(tally);
        }
        // g_max = 2 max |u − ū| (SPEC 5.3).
        let mut mean = Vec3::new(0.0, 0.0, 0.0);
        for &v in u.iter() {
            mean += v;
        }
        mean = mean * (1.0 / n as f64);
        let g_max = 2.0 * u.iter().map(|&v| (v - mean).norm()).fold(0.0, f64::max);
        if g_max <= 0.0 {
            return Ok(tally);
        }
        let per_pair = self.weight * self.cross_section_km2() * g_max * dt_s / volume_km3;
        if per_pair > 1.0 {
            s.overfull_cells = 1;
        }
        let m = 0.5 * (n * (n - 1)) as f64 * per_pair;
        let smallest_id = ids[0];
        let too_full = CollideError {
            smallest_id,
            members: n,
            expected_candidates: m,
        };
        if m >= MAX_CANDIDATES as f64 {
            return Err(too_full);
        }
        let block = |draw: u16| {
            self.rng.block(Counter {
                index: smallest_id,
                step,
                stream: Stream::COLLISIONS,
                draw,
            })
        };
        let w = block(0);
        let whole = m.floor();
        let mut candidates = whole as u64;
        if m - whole > u01_f64(w[0], w[1]) {
            candidates += 1;
        }
        if candidates > MAX_CANDIDATES {
            return Err(too_full);
        }
        for k in 0..candidates as u16 {
            let w = block(1 + 2 * k);
            let i = mulhilo(w[0], n as u32).0 as usize;
            let j0 = mulhilo(w[1], (n - 1) as u32).0 as usize;
            let j = if j0 >= i { j0 + 1 } else { j0 };
            let g = u[i] - u[j];
            let g_norm = g.norm();
            tally.stats.candidates += 1;
            if g_norm <= 0.0 {
                continue;
            }
            if g_norm > g_max {
                tally.stats.gmax_exceeded += 1;
            } else if u01_f64(w[2], w[3]) >= g_norm / g_max {
                continue;
            }
            let w = block(2 + 2 * k);
            let n_hat = pair::impact_normal(g, u01_f64(w[0], w[1]), u01_f64(w[2], w[3]));
            if let Some(imp) = pair::collide(g, n_hat, &self.restitution) {
                let (mut ui, mut uj) = (u[i], u[j]);
                imp.apply(&mut ui, &mut uj);
                (u[i], u[j]) = (ui, uj);
                delta[i] += imp.delta_ui_km_s;
                delta[j] -= imp.delta_ui_km_s;
                tally.add_collision(imp.energy_loss_km2_s2);
            }
        }
        Ok(tally)
    }

    /// Collides the members of one cell of a [`CellList`] in the full state:
    /// peculiar velocities from the state (`SPEC.md` 5.2), then
    /// [`Self::process_cell`]. The velocity changes (inertial, to be added to
    /// `vx, vy, vz`) are in `scratch.delta_km_s`, in the order of `members`.
    /// The state is not changed, so many cells can run at the same time.
    #[allow(clippy::too_many_arguments)]
    pub fn process_cell_states(
        &self,
        field: &SaturnField<f64>,
        volume_km3: f64,
        dt_s: f64,
        step: u64,
        ids: &[u32],
        state: &[State],
        members: &[u32],
        scratch: &mut CellScratch,
    ) -> Result<Tally, CollideError> {
        scratch.u_km_s.clear();
        scratch.delta_km_s.clear();
        scratch.ids.clear();
        for &m in members {
            let s = &state[m as usize];
            let pos = Vec3::new(s[0], s[1], s[2]);
            let vel = Vec3::new(s[3], s[4], s[5]);
            scratch.u_km_s.push(peculiar_velocity(field, pos, vel));
            scratch.delta_km_s.push(Vec3::new(0.0, 0.0, 0.0));
            scratch.ids.push(ids[m as usize]);
        }
        self.process_cell(
            volume_km3,
            dt_s,
            step,
            &scratch.ids,
            &mut scratch.u_km_s,
            &mut scratch.delta_km_s,
        )
    }

    /// One collision step on the synchronised state (`SPEC.md` 7), one cell
    /// after the other.
    ///
    /// - `t_s`: time of the state, seconds since the epoch; `step`: the step
    ///   count of the run after the integrator step that ends at `t_s`.
    /// - `dt_s`: Δt_c = `every_steps` × h.
    /// - `ids[i]`, `active[i]` (`None`: all), `state[i]`: particle `i`.
    ///
    /// Velocities are changed in place; positions are not. Returns the sums of
    /// the step.
    #[allow(clippy::too_many_arguments)]
    pub fn collide_step(
        &self,
        domain: &Domain,
        field: &SaturnField<f64>,
        t_s: f64,
        step: u64,
        dt_s: f64,
        ids: &[u32],
        active: Option<&[bool]>,
        state: &mut [State],
    ) -> Result<Tally, CollideError> {
        let cells = domain.build_cells(t_s, ids, active, |i| {
            Vec3::new(state[i][0], state[i][1], state[i][2])
        });
        let volume = domain.grid.volume_km3();
        let mut scratch = CellScratch::default();
        let mut total = Tally::default();
        for c in 0..cells.len() {
            let members = cells.members(c);
            total += self.process_cell_states(
                field,
                volume,
                dt_s,
                step,
                ids,
                state,
                members,
                &mut scratch,
            )?;
            for (&m, d) in members.iter().zip(&scratch.delta_km_s) {
                let s = &mut state[m as usize];
                s[3] += d.x;
                s[4] += d.y;
                s[5] += d.z;
            }
        }
        Ok(total)
    }
}

#[cfg(feature = "testkit")]
mod box_model {
    use super::*;
    use crate::testkit::{BoxModel, BoxSpec, ModelParams};

    impl StochasticModel {
        /// The model of a box test (K1).
        pub fn from_params(p: ModelParams) -> Self {
            Self::new(p.particle_radius_km, p.weight, p.restitution, p.seed)
        }
    }

    impl BoxModel for StochasticModel {
        fn collide(
            &mut self,
            cells: &BoxSpec,
            pos_km: &[Vec3<f64>],
            vel_km_s: &mut [Vec3<f64>],
            step: u64,
            dt_s: f64,
        ) -> Tally {
            let items = pos_km
                .iter()
                .enumerate()
                .map(|(i, &p)| (cells.cell_of(p), i as u32, i as u32))
                .collect();
            let list = CellList::group(items);
            let volume = cells.cell_volume_km3();
            let mut total = Tally::default();
            let (mut ids, mut u, mut delta) = (Vec::new(), Vec::new(), Vec::new());
            for c in 0..list.len() {
                let members = list.members(c);
                ids.clear();
                u.clear();
                delta.clear();
                for &m in members {
                    ids.push(m);
                    u.push(vel_km_s[m as usize]);
                    delta.push(Vec3::new(0.0, 0.0, 0.0));
                }
                total += self
                    .process_cell(volume, dt_s, step, &ids, &mut u, &mut delta)
                    .expect("cell too full in a box test");
                for (&m, &v) in members.iter().zip(&u) {
                    vel_km_s[m as usize] = v;
                }
            }
            total
        }
    }
}
