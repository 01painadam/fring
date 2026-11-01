//! Circular equatorial orbits in the field of section 7.1: speed and the
//! frequencies of section 7.2.

use super::gravity::SaturnField;
use crate::{Real, Vec3};

/// The three frequencies of a circular equatorial orbit, in rad s⁻¹.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircularFrequencies<T: Real> {
    /// Orbital angular frequency n.
    pub n_rad_s: T,
    /// Radial (epicyclic) frequency kappa.
    pub kappa_rad_s: T,
    /// Vertical frequency nu.
    pub nu_rad_s: T,
}

/// Speed in km/s of the circular equatorial orbit at radius `r_km`:
/// `v = sqrt(r |a_r|)`, with the J2, J4 and J6 terms included.
pub fn circular_speed<T: Real>(field: &SaturnField<T>, r_km: T) -> T {
    let a = field.acceleration(Vec3::new(r_km, T::ZERO, T::ZERO));
    (-a.x * r_km).sqrt()
}

/// n, kappa and nu at radius `r_km` (`docs/PLAN.md` section 7.2; exact for
/// this potential).
pub fn frequencies<T: Real>(field: &SaturnField<T>, r_km: T) -> CircularFrequencies<T> {
    let c = |x: f64| T::from_f64(x);
    let q = field.reference_radius_km / r_km;
    let q2 = q * q;
    let q4 = q2 * q2;
    let q6 = q4 * q2;
    let (j2, j4, j6) = (field.j2 * q2, field.j4 * q4, field.j6 * q6);
    let k2 = field.gm_km3_s2 / (r_km * r_km * r_km);
    let n2 = k2 * (T::ONE + c(1.5) * j2 - c(15.0 / 8.0) * j4 + c(35.0 / 16.0) * j6);
    let kappa2 = k2 * (T::ONE - c(1.5) * j2 + c(45.0 / 8.0) * j4 - c(175.0 / 16.0) * j6);
    let nu2 = k2 * (T::ONE + c(4.5) * j2 - c(75.0 / 8.0) * j4 + c(245.0 / 16.0) * j6);
    CircularFrequencies {
        n_rad_s: n2.sqrt(),
        kappa_rad_s: kappa2.sqrt(),
        nu_rad_s: nu2.sqrt(),
    }
}
