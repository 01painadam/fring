//! Equations of motion of the moons (ADR 0004).
//!
//! Saturn-centred, with Saturn's own motion eliminated:
//!
//! ```text
//! r_i'' = -(GM + GM_i) r_i/|r_i|³ + a_J(r_i)
//!         + sum_{j != i} GM_j [ (r_j - r_i)/|r_j - r_i|³ - r_j/|r_j|³ ]
//! ```
//!
//! `a_J` is the J2, J4, J6 part of [`SaturnField`]. The last term has the
//! direct pull of moon j and the indirect term (Saturn's acceleration toward
//! moon j). Saturn's reaction to the J force is neglected (ADR 0004). This is
//! exactly the system that the REBOUND reference of task V2 integrates.

use super::gravity::SaturnField;
use crate::{SaturnParams, Vec3};

/// Saturn's field and the GM of each moon.
#[derive(Clone, Debug, PartialEq)]
pub struct MoonSystem {
    /// Saturn's field (point mass, J2, J4, J6).
    pub field: SaturnField<f64>,
    /// GM of each moon in km³ s⁻².
    pub gm_moons_km3_s2: Vec<f64>,
}

impl MoonSystem {
    /// Makes the system from Saturn's parameters and the moons' GM values.
    pub fn new(saturn: &SaturnParams, gm_moons_km3_s2: Vec<f64>) -> Self {
        Self {
            field: SaturnField::from_params(saturn),
            gm_moons_km3_s2,
        }
    }

    /// Number of moons.
    pub fn len(&self) -> usize {
        self.gm_moons_km3_s2.len()
    }

    /// True if there are no moons.
    pub fn is_empty(&self) -> bool {
        self.gm_moons_km3_s2.is_empty()
    }

    /// Accelerations in km s⁻² of all moons. `pos_km` and `out_km_s2` are flat
    /// arrays of length `3 × len()`: `x0, y0, z0, x1, y1, z1, ...`.
    ///
    /// # Panics
    /// If a slice has the wrong length.
    pub fn accelerations(&self, pos_km: &[f64], out_km_s2: &mut [f64]) {
        let n = self.len();
        assert!(pos_km.len() == 3 * n && out_km_s2.len() == 3 * n);
        let p = |i: usize| Vec3::new(pos_km[3 * i], pos_km[3 * i + 1], pos_km[3 * i + 2]);
        let gm = self.field.gm_km3_s2;
        for i in 0..n {
            let ri = p(i);
            let r2 = ri.norm_squared();
            let mut a = ri * (-(gm + self.gm_moons_km3_s2[i]) / (r2 * r2.sqrt()))
                + self.field.acceleration_without_point_mass(ri);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let rj = p(j);
                let d = rj - ri;
                let d2 = d.norm_squared();
                let rj2 = rj.norm_squared();
                a += (d / (d2 * d2.sqrt()) - rj / (rj2 * rj2.sqrt())) * self.gm_moons_km3_s2[j];
            }
            out_km_s2[3 * i] = a.x;
            out_km_s2[3 * i + 1] = a.y;
            out_km_s2[3 * i + 2] = a.z;
        }
    }
}
