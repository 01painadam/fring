//! What a collision step reports (`SPEC.md` section 9).

use std::ops::AddAssign;

use fring_model::CollisionStats;

/// The sums of one or more collision steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    /// Accepted collisions. The count is not in [`CollisionStats`]: it goes to
    /// `StepStats::collisions_since_last` (ICR-009, amendment A1).
    pub collisions: u64,
    /// The diagnostics of ICR-009, including the energy loss.
    pub stats: CollisionStats,
    /// Σ of the squares of the single energy losses in km⁴ s⁻⁴. The tests
    /// use it for the standard error; a run does not report it.
    pub energy_loss_sq_km4_s4: f64,
}

impl Tally {
    /// Adds one collision with its energy loss.
    pub fn add_collision(&mut self, energy_loss_km2_s2: f64) {
        self.collisions += 1;
        self.stats.energy_loss_km2_s2 += energy_loss_km2_s2;
        self.energy_loss_sq_km4_s4 += energy_loss_km2_s2 * energy_loss_km2_s2;
    }
}

impl AddAssign for Tally {
    fn add_assign(&mut self, o: Self) {
        self.collisions += o.collisions;
        self.energy_loss_sq_km4_s4 += o.energy_loss_sq_km4_s4;
        let (a, b) = (&mut self.stats, o.stats);
        a.energy_loss_km2_s2 += b.energy_loss_km2_s2;
        a.candidates += b.candidates;
        a.occupied_cells += b.occupied_cells;
        a.occupancy_sum += b.occupancy_sum;
        a.occupancy_sq_sum += b.occupancy_sq_sum;
        a.lonely_particles += b.lonely_particles;
        a.overfull_cells += b.overfull_cells;
        a.gmax_exceeded += b.gmax_exceeded;
        a.deep_collisions += b.deep_collisions;
    }
}
