//! The hard-sphere model (`SPEC.md` section 8), K3.
//!
//! Real particles (W = 1) of radius s. At the end of each integrator step the
//! model finds every pair with |rᵢ − rⱼ| < 2s with a cell list, then resolves
//! the pairs one after the other.
//!
//! The model has no randomness. The order of the pairs is fixed by the particle
//! ids, and the contact normal comes from the geometry. So the same input gives
//! the same output bit for bit, and the result does not depend on the order of
//! the cells or of the arrays.
//!
//! A step has two public parts, so that a backend can find the pairs in
//! parallel (the cells are independent) while the resolution stays sequential
//! (the pairs share particles, and the order matters):
//!
//! 1. [`HardSphereModel::find_pairs`]: read-only, returns the pairs sorted by
//!    `(id_i, id_j)`.
//! 2. [`HardSphereModel::resolve_pairs`]: changes the velocities.
//!
//! [`HardSphereModel::collide_step`] does both for the CPU state.
//!
//! The cells are the cells of K2 (`stochastic::Domain`): boxes in the local
//! coordinates of the rotating frame. Here they are only a search structure,
//! so the distance of a pair is always the inertial distance of the positions.

use fring_model::Vec3;
use fring_model::config::{Collisions, Restitution};

use crate::frame::{CellGrid, CellKey};
use crate::pair;
use crate::stochastic::{CellList, Domain, State};
use crate::tally::Tally;

/// Cell edges are this factor larger than the contact distance, so that the
/// rounding of the coordinates (about 10⁻¹¹ km at 1.4 × 10⁵ km) cannot put two
/// touching particles two cells apart.
const EDGE_SLACK: f64 = 1.0 + 1.0e-6;

/// Overlap depth above which a collision counts as deep, as a fraction of the
/// contact distance 2s (`SPEC.md` 8: 0.2 s).
const DEEP_FRACTION: f64 = 0.1;

/// Which cell indices wrap, for the neighbour search.
#[derive(Clone, Copy, Debug)]
struct Wrap {
    /// Radial cells, if they wrap (the periodic box).
    radial: Option<i64>,
    /// Azimuthal cells (they always wrap).
    azimuthal: u32,
    /// Vertical cells, if they wrap (the periodic box).
    vertical: Option<i64>,
}

impl Wrap {
    /// The cell and its 26 neighbours, each once, in key order.
    fn neighbours(&self, key: CellKey) -> Vec<CellKey> {
        let wrap = |i: i64, n: Option<i64>| n.map_or(i, |n| i.rem_euclid(n));
        let ny = i64::from(self.azimuthal);
        let mut out = Vec::with_capacity(27);
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    out.push(CellKey {
                        radial: wrap(key.radial + dx, self.radial),
                        azimuthal: (i64::from(key.azimuthal) + dy).rem_euclid(ny) as u32,
                        vertical: wrap(key.vertical + dz, self.vertical),
                    });
                }
            }
        }
        // With fewer than three cells along an axis, neighbours repeat.
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Pairs `(i, j)` of particle indices with `ids[i] < ids[j]` and `close(i, j)`,
/// sorted by `(ids[i], ids[j])`. Each pair once.
fn pairs_in_cells(
    cells: &CellList,
    keys: &[CellKey],
    wrap: Wrap,
    ids: &[u32],
    close: impl Fn(usize, usize) -> bool,
) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for (c, &key) in keys.iter().enumerate() {
        for nb in wrap.neighbours(key) {
            let Ok(c2) = keys.binary_search(&nb) else {
                continue;
            };
            for &i in cells.members(c) {
                for &j in cells.members(c2) {
                    // A pair in two cells A ≠ B is seen from A and from B, but
                    // only one of the two orders has ids[i] < ids[j].
                    if ids[i as usize] < ids[j as usize] && close(i as usize, j as usize) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable_by_key(|&(i, j)| (ids[i as usize], ids[j as usize]));
    pairs
}

/// The model parameters (`SPEC.md` 3, 6).
#[derive(Clone, Debug, PartialEq)]
pub struct HardSphereModel {
    particle_radius_km: f64,
    restitution: Restitution,
}

impl HardSphereModel {
    /// A model with the radius s (km) and the restitution law.
    pub fn new(particle_radius_km: f64, restitution: Restitution) -> Self {
        Self {
            particle_radius_km,
            restitution,
        }
    }

    /// The model of `[collisions]`. `None` for another collision model.
    pub fn from_config(c: &Collisions) -> Option<Self> {
        match *c {
            Collisions::HardSphere {
                particle_radius_km,
                restitution,
            } => Some(Self::new(particle_radius_km, restitution)),
            _ => None,
        }
    }

    /// The contact distance 2s in km.
    pub fn contact_km(&self) -> f64 {
        2.0 * self.particle_radius_km
    }

    /// The full ring as a [`Domain`] (φ₀ = 0, D = 2π a₀), for [`Self::find_pairs`].
    /// The cell sizes inside are placeholders: the model makes its own grid at
    /// each step, from 2s and the radius range of the particles. `None` if a
    /// number is not finite and > 0.
    pub fn full_ring_domain(&self, a0_km: f64, n0_rad_s: f64) -> Option<Domain> {
        let e = self.contact_km() * EDGE_SLACK;
        Domain::full_ring(e, e, e, a0_km, n0_rad_s)
    }

    /// All pairs with |rᵢ − rⱼ| < 2s (`SPEC.md` 8), as indices into the
    /// caller's arrays, each pair once with `ids[i] < ids[j]`, sorted by
    /// `(ids[i], ids[j])`.
    ///
    /// `position(i)` is the inertial position of particle `i` at time `t_s`;
    /// `active[i]` false leaves it out (`None`: all). Uses only `a0`, `n0`, φ₀
    /// and D of `domain`.
    ///
    /// Cell edges: L_x = L_z = 2s, and L_y = D / n_y with n_y the largest
    /// integer for which L_y ≥ 2s a₀ / ρ_min. The factor a₀ / ρ_min is the
    /// curvature: an arc length y = a₀ φ is longer than the distance at a
    /// radius ρ < a₀. So a pair closer than 2s is in the same cell or in
    /// neighbouring cells, also across the wrap.
    pub fn find_pairs(
        &self,
        domain: &Domain,
        t_s: f64,
        ids: &[u32],
        active: Option<&[bool]>,
        position: impl Fn(usize) -> Vec3<f64>,
    ) -> Vec<(u32, u32)> {
        let rho_min = (0..ids.len())
            .filter(|&i| active.is_none_or(|a| a[i]))
            .map(|i| {
                let p = position(i);
                p.x.hypot(p.y)
            })
            .fold(f64::INFINITY, f64::min);
        if !rho_min.is_finite() {
            return Vec::new();
        }
        let edge = self.contact_km() * EDGE_SLACK;
        let d_km = domain.grid.domain_length_km;
        let rho_min = rho_min.max(1.0e-9 * domain.a0_km);
        let n_y = (d_km * rho_min / (edge * domain.a0_km)).floor().max(1.0);
        let grid = CellGrid::new(edge, d_km / n_y, edge, d_km).expect("finite positive sizes");
        let local = Domain { grid, ..*domain };
        let (cells, keys) = local.build_keyed(t_s, ids, active, &position);
        let wrap = Wrap {
            radial: None,
            azimuthal: grid.azimuthal_cells,
            vertical: None,
        };
        let c2 = self.contact_km().powi(2);
        pairs_in_cells(&cells, &keys, wrap, ids, |i, j| {
            (position(i) - position(j)).norm_squared() < c2
        })
    }

    /// Resolves the pairs of [`Self::find_pairs`], one after the other in the
    /// given order (`SPEC.md` 8).
    ///
    /// `separation(i, j)` is d = rᵢ − rⱼ; `vel` the velocities (inertial, or
    /// any frame that does not change gₙ), changed in place; `h_s` the
    /// integrator step. For each pair the current velocities decide: a pair
    /// with d · g ≥ 0 is left alone. The contact normal is found by going back
    /// along the straight relative path (`SPEC.md` 8).
    pub fn resolve_pairs(
        &self,
        pairs: &[(u32, u32)],
        h_s: f64,
        separation: impl Fn(usize, usize) -> Vec3<f64>,
        vel: &mut [Vec3<f64>],
    ) -> Tally {
        let contact = self.contact_km();
        let mut tally = Tally::default();
        for &(i, j) in pairs {
            let (i, j) = (i as usize, j as usize);
            let d = separation(i, j);
            let g = vel[i] - vel[j];
            let b = d.dot(g);
            if b >= 0.0 {
                continue;
            }
            // Back to the contact: |d − g τ| = 2s with τ > 0. τ = (b + √disc) / |g|²
            // with disc = b² + |g|² (4s² − |d|²), written without the
            // cancellation of b + √disc (b < 0).
            let gg = g.norm_squared();
            let inside = contact * contact - d.norm_squared();
            let tau = inside / ((b * b + gg * inside).sqrt() - b);
            let dc = if tau <= h_s { d - g * tau } else { d };
            let n_hat = dc / dc.norm();
            if let Some(imp) = pair::collide(g, n_hat, &self.restitution) {
                vel[i] += imp.delta_ui_km_s;
                vel[j] -= imp.delta_ui_km_s;
                tally.add_collision(imp.energy_loss_km2_s2);
                if contact - d.norm() > DEEP_FRACTION * contact {
                    tally.stats.deep_collisions += 1;
                }
            }
        }
        tally
    }

    /// One collision step on the synchronised state (`SPEC.md` 8): every
    /// integrator step, `h_s` long. `t_s` is the time of the state (seconds
    /// since the epoch); `ids[i]`, `active[i]` (`None`: all) and `state[i]`
    /// describe particle `i`. Velocities change in place; positions do not.
    pub fn collide_step(
        &self,
        domain: &Domain,
        t_s: f64,
        h_s: f64,
        ids: &[u32],
        active: Option<&[bool]>,
        state: &mut [State],
    ) -> Tally {
        let pos = |s: &[State], i: usize| Vec3::new(s[i][0], s[i][1], s[i][2]);
        let pairs = self.find_pairs(domain, t_s, ids, active, |i| pos(state, i));
        if pairs.is_empty() {
            return Tally::default();
        }
        let mut vel: Vec<Vec3<f64>> = state.iter().map(|s| Vec3::new(s[3], s[4], s[5])).collect();
        let tally = self.resolve_pairs(&pairs, h_s, |i, j| pos(state, i) - pos(state, j), &mut vel);
        for &(i, j) in &pairs {
            for k in [i as usize, j as usize] {
                state[k][3] = vel[k].x;
                state[k][4] = vel[k].y;
                state[k][5] = vel[k].z;
            }
        }
        tally
    }
}

#[cfg(feature = "testkit")]
mod box_model {
    use super::*;
    use crate::testkit::{BoxModel, BoxSpec, ModelParams};

    impl HardSphereModel {
        /// The model of a box test (K1). `weight` must be 1.
        pub fn from_params(p: ModelParams) -> Self {
            assert_eq!(p.weight, 1.0, "hard spheres are real particles");
            Self::new(p.particle_radius_km, p.restitution)
        }
    }

    impl BoxModel for HardSphereModel {
        fn collide(
            &mut self,
            cells: &BoxSpec,
            pos_km: &[Vec3<f64>],
            vel_km_s: &mut [Vec3<f64>],
            _step: u64,
            dt_s: f64,
        ) -> Tally {
            // k cells per side, each at least 2s wide; all three axes wrap.
            let k = ((cells.side_km / self.contact_km()).floor() as i64).clamp(1, 1 << 20);
            let edge = cells.side_km / k as f64;
            let index = |x: f64| ((x / edge).floor() as i64).clamp(0, k - 1);
            let items = pos_km
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let key = CellKey {
                        radial: index(p.x),
                        azimuthal: index(p.y) as u32,
                        vertical: index(p.z),
                    };
                    (key, i as u32, i as u32)
                })
                .collect();
            let (list, keys) = CellList::group_with_keys(items);
            let wrap = Wrap {
                radial: Some(k),
                azimuthal: k as u32,
                vertical: Some(k),
            };
            let ids: Vec<u32> = (0..pos_km.len() as u32).collect();
            let c2 = self.contact_km().powi(2);
            let pairs = pairs_in_cells(&list, &keys, wrap, &ids, |i, j| {
                cells.min_image(pos_km[i] - pos_km[j]).norm_squared() < c2
            });
            self.resolve_pairs(
                &pairs,
                dt_s,
                |i, j| cells.min_image(pos_km[i] - pos_km[j]),
                vel_km_s,
            )
        }
    }
}
