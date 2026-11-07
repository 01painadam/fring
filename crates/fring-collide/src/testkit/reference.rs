//! Two slow reference models that follow `SPEC.md` directly. Test use only.
//!
//! - [`AllPairs`]: every pair of a cell collides with probability
//!   P_ij = W σ |g_ij| Δt / V_c. This is the rate that the no-time-counter
//!   method of `SPEC.md` 5.3 must give, without the method itself.
//! - [`HardSphere`]: real spheres in the periodic box, found with a grid at the
//!   end of each step, with the contact normal of `SPEC.md` section 8.

use fring_model::Vec3;
use fring_model::rng::{Counter, Philox, Stream, u01_f64};

use super::{BoxModel, BoxSpec, ModelParams, Tally};
use crate::pair;

const EMPTY: u32 = u32::MAX;

/// A grid of a periodic cube with a linked list of particles in each cell.
#[derive(Clone, Debug)]
pub struct Grid {
    k: usize,
    cell_km: f64,
    head: Vec<u32>,
    next: Vec<u32>,
}

impl Grid {
    /// Cells with an edge of at least `min_cell_km` (at most 32 per side), for
    /// particle indices below `capacity`.
    pub fn new(cells: &BoxSpec, min_cell_km: f64, capacity: usize) -> Self {
        let k = ((cells.side_km / min_cell_km).floor() as usize).clamp(1, 32);
        Self {
            k,
            cell_km: cells.side_km / k as f64,
            head: vec![EMPTY; k * k * k],
            next: vec![EMPTY; capacity],
        }
    }

    fn coords(&self, p: Vec3<f64>) -> [usize; 3] {
        let k = self.k as i64;
        let f = |x: f64| ((x / self.cell_km).floor() as i64).clamp(0, k - 1) as usize;
        [f(p.x), f(p.y), f(p.z)]
    }

    /// Adds particle `i` at `p`.
    pub fn insert(&mut self, i: u32, p: Vec3<f64>) {
        let [x, y, z] = self.coords(p);
        let c = (x * self.k + y) * self.k + z;
        self.next[i as usize] = self.head[c];
        self.head[c] = i;
    }

    /// Calls `f` for each particle in the cell of `p` and its periodic
    /// neighbours (each cell once).
    pub fn for_neighbours(&self, p: Vec3<f64>, mut f: impl FnMut(u32)) {
        let k = self.k;
        let offsets: &[usize] = match k {
            1 => &[0],
            2 => &[0, 1],
            _ => &[0, 1, 2],
        };
        let [x, y, z] = self.coords(p);
        // For k ≥ 3, offset o means a shift of o − 1; for k ≤ 2 the offsets cover all cells.
        let shift = |c: usize, o: usize| {
            if k >= 3 {
                (c + k + o - 1) % k
            } else {
                (c + o) % k
            }
        };
        for &ox in offsets {
            for &oy in offsets {
                for &oz in offsets {
                    let c = (shift(x, ox) * k + shift(y, oy)) * k + shift(z, oz);
                    let mut j = self.head[c];
                    while j != EMPTY {
                        f(j);
                        j = self.next[j as usize];
                    }
                }
            }
        }
    }
}

/// Stochastic reference: each pair of a cell, in id order, collides with
/// probability W σ |g| Δt / V_c and the impact geometry of `SPEC.md` 5.4.
#[derive(Clone, Debug)]
pub struct AllPairs {
    params: ModelParams,
    rng: Philox,
}

impl AllPairs {
    /// A model with these parameters.
    pub fn new(params: ModelParams) -> Self {
        Self {
            params,
            rng: Philox::from_seed(params.seed),
        }
    }
}

impl BoxModel for AllPairs {
    fn collide(
        &mut self,
        cells: &BoxSpec,
        pos_km: &[Vec3<f64>],
        vel_km_s: &mut [Vec3<f64>],
        step: u64,
        dt_s: f64,
    ) -> Tally {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); cells.cell_count()];
        for (i, &p) in pos_km.iter().enumerate() {
            members[cells.cell_of(p)].push(i as u32);
        }
        let factor =
            self.params.weight * self.params.cross_section_km2() * dt_s / cells.cell_volume_km3();
        let mut tally = Tally::default();
        for m in &members {
            assert!(m.len() < 32_768, "cell too full for the draw counter");
            for a in 0..m.len() {
                for b in a + 1..m.len() {
                    let (i, j) = (m[a] as usize, m[b] as usize);
                    let g = vel_km_s[i] - vel_km_s[j];
                    let p_pair = factor * g.norm();
                    if p_pair > 1.0 {
                        tally.stats.overfull_cells += 1;
                    }
                    // Counter: the first particle of the pair and the place of the second.
                    let block = |draw: u16| {
                        self.rng.block(Counter {
                            index: m[a],
                            step,
                            stream: Stream::COLLISIONS,
                            draw,
                        })
                    };
                    tally.stats.candidates += 1;
                    let w = block(2 * b as u16);
                    if u01_f64(w[0], w[1]) >= p_pair {
                        continue;
                    }
                    let w = block(2 * b as u16 + 1);
                    let n_hat = pair::impact_normal(g, u01_f64(w[0], w[1]), u01_f64(w[2], w[3]));
                    if let Some(imp) = pair::collide(g, n_hat, &self.params.restitution) {
                        let (mut ui, mut uj) = (vel_km_s[i], vel_km_s[j]);
                        imp.apply(&mut ui, &mut uj);
                        vel_km_s[i] = ui;
                        vel_km_s[j] = uj;
                        tally.add_collision(imp.energy_loss_km2_s2);
                    }
                }
            }
        }
        tally
    }
}

/// Hard-sphere reference in the periodic box (`SPEC.md` section 8).
#[derive(Clone, Debug)]
pub struct HardSphere {
    params: ModelParams,
}

impl HardSphere {
    /// A model with these parameters (`weight` must be 1).
    pub fn new(params: ModelParams) -> Self {
        assert_eq!(params.weight, 1.0, "hard spheres are real particles");
        Self { params }
    }
}

impl BoxModel for HardSphere {
    fn collide(
        &mut self,
        cells: &BoxSpec,
        pos_km: &[Vec3<f64>],
        vel_km_s: &mut [Vec3<f64>],
        _step: u64,
        dt_s: f64,
    ) -> Tally {
        let contact = 2.0 * self.params.particle_radius_km;
        let mut grid = Grid::new(cells, contact, pos_km.len());
        for (i, &p) in pos_km.iter().enumerate() {
            grid.insert(i as u32, p);
        }
        let mut pairs = Vec::new();
        for (i, &p) in pos_km.iter().enumerate() {
            grid.for_neighbours(p, |j| {
                if j as usize > i
                    && cells.min_image(p - pos_km[j as usize]).norm_squared() < contact * contact
                {
                    pairs.push((i as u32, j));
                }
            });
        }
        pairs.sort_unstable();
        let mut tally = Tally::default();
        for (i, j) in pairs {
            let (i, j) = (i as usize, j as usize);
            let d = cells.min_image(pos_km[i] - pos_km[j]);
            let g = vel_km_s[i] - vel_km_s[j];
            let b = d.dot(g);
            if b >= 0.0 {
                continue;
            }
            // Back to the contact along the straight relative path: |d − g τ| = 2s, τ > 0.
            // τ = (b + √disc) / |g|², written without the cancellation of b + √disc (b < 0).
            let gg = g.norm_squared();
            let inside = contact * contact - d.norm_squared();
            let tau = inside / ((b * b + gg * inside).sqrt() - b);
            let dc = if tau <= dt_s { d - g * tau } else { d };
            let n_hat = dc / dc.norm();
            if let Some(imp) = pair::collide(g, n_hat, &self.params.restitution) {
                let (mut ui, mut uj) = (vel_km_s[i], vel_km_s[j]);
                imp.apply(&mut ui, &mut uj);
                vel_km_s[i] = ui;
                vel_km_s[j] = uj;
                tally.add_collision(imp.energy_loss_km2_s2);
                if contact - d.norm() > 0.1 * contact {
                    tally.stats.deep_collisions += 1;
                }
            }
        }
        tally
    }
}
