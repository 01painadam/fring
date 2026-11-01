//! Wisdom-Holman split, drift-kick-drift, fixed step (ADR 0003).
//!
//! The drift is the exact Kepler motion about Saturn's point mass. The kick
//! uses the acceleration without the point-mass term (J2, J4, J6; later the
//! moons). Compared with the plain leapfrog this keeps the precession rates
//! n - kappa and nu - n accurate at 150 steps per orbit.

use super::gravity::SaturnField;
use super::kepler::kepler_drift;
use crate::{Real, Vec3};

/// One Wisdom-Holman step of `h_s` seconds for one particle: Kepler drift
/// for `h/2`, kick, Kepler drift for `h/2`.
///
/// Returns `None` (and leaves the state unchanged) if the orbit is not
/// bound at some point of the step; see [`kepler_drift`].
#[inline]
pub fn wh_step<T: Real>(
    field: &SaturnField<T>,
    pos_km: &mut Vec3<T>,
    vel_km_s: &mut Vec3<T>,
    h_s: T,
) -> Option<()> {
    let half = T::from_f64(0.5) * h_s;
    let (p, v) = kepler_drift(field.gm_km3_s2, *pos_km, *vel_km_s, half)?;
    let a = field.acceleration_without_point_mass(p);
    let v = Vec3::new(v.x + a.x * h_s, v.y + a.y * h_s, v.z + a.z * h_s);
    let (p, v) = kepler_drift(field.gm_km3_s2, p, v, half)?;
    *pos_km = p;
    *vel_km_s = v;
    Some(())
}

/// `n_steps` Wisdom-Holman steps in a row. The half drifts between two steps
/// are merged into one full drift, which saves one Kepler solve per step.
/// The state at the end is the same as after `n_steps` calls of [`wh_step`]
/// (up to rounding). Returns `None` and leaves the state unchanged if an
/// orbit is not bound.
pub fn wh_steps<T: Real>(
    field: &SaturnField<T>,
    pos_km: &mut Vec3<T>,
    vel_km_s: &mut Vec3<T>,
    h_s: T,
    n_steps: u64,
) -> Option<()> {
    if n_steps == 0 {
        return Some(());
    }
    let half = T::from_f64(0.5) * h_s;
    let (mut p, mut v) = kepler_drift(field.gm_km3_s2, *pos_km, *vel_km_s, half)?;
    for k in 0..n_steps {
        let a = field.acceleration_without_point_mass(p);
        v = Vec3::new(v.x + a.x * h_s, v.y + a.y * h_s, v.z + a.z * h_s);
        let dt = if k + 1 == n_steps { half } else { h_s };
        (p, v) = kepler_drift(field.gm_km3_s2, p, v, dt)?;
    }
    *pos_km = p;
    *vel_km_s = v;
    Some(())
}
