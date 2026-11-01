//! Drift-kick-drift leapfrog, fixed step (`docs/PLAN.md` section 7.3).

use super::circular::circular_speed;
use super::gravity::SaturnField;
use crate::{Real, Vec3};

/// One drift-kick-drift step of size `h_s` seconds for one particle in
/// Saturn's field (no moons): half drift, kick with the acceleration at the
/// middle position, half drift. Second order and symplectic.
#[inline]
pub fn dkd_step<T: Real>(
    field: &SaturnField<T>,
    pos_km: &mut Vec3<T>,
    vel_km_s: &mut Vec3<T>,
    h_s: T,
) {
    let half = T::from_f64(0.5) * h_s;
    *pos_km = add_scaled(*pos_km, *vel_km_s, half);
    let a = field.acceleration(*pos_km);
    *vel_km_s = add_scaled(*vel_km_s, a, h_s);
    *pos_km = add_scaled(*pos_km, *vel_km_s, half);
}

#[inline]
fn add_scaled<T: Real>(p: Vec3<T>, v: Vec3<T>, s: T) -> Vec3<T> {
    Vec3::new(p.x + v.x * s, p.y + v.y * s, p.z + v.z * s)
}

/// The speed `v_h` (km/s) at radius `r_km` for which the leapfrog has an
/// exact circular orbit.
///
/// The step map commutes with rotations about z. So for the start state
/// `pos = (r, 0, 0)`, `vel = (0, v, 0)` there is a `v_h` such that one step
/// ends at the same radius, turned by an angle theta. Every later step is then
/// the same rotation, and the energy is constant up to rounding. `v_h` differs
/// from [`circular_speed`] by a relative amount of order `(n h)²`.
///
/// Found by bisection on `|pos_after_step| = r` in `[0.9, 1.1] × circular_speed`.
/// Returns `None` if `h_s` is so large that the root is not in that bracket.
pub fn leapfrog_circular_speed<T: Real>(field: &SaturnField<T>, r_km: T, h_s: T) -> Option<T> {
    let g = |v: T| {
        let mut p = Vec3::new(r_km, T::ZERO, T::ZERO);
        let mut w = Vec3::new(T::ZERO, v, T::ZERO);
        dkd_step(field, &mut p, &mut w, h_s);
        p.norm() - r_km
    };
    let vc = circular_speed(field, r_km);
    let mut lo = T::from_f64(0.9) * vc;
    let mut hi = T::from_f64(1.1) * vc;
    if g(lo) > T::ZERO || g(hi) < T::ZERO {
        return None;
    }
    for _ in 0..200 {
        let mid = T::from_f64(0.5) * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if g(mid) < T::ZERO {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(T::from_f64(0.5) * (lo + hi))
}
