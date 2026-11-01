//! Exact Kepler motion around Saturn's point mass: the drift of the
//! Wisdom-Holman split (ADR 0003).

use crate::{Real, Vec3};

/// Moves a particle on its Kepler orbit (point mass GM only) for `dt_s`
/// seconds, with the f and g functions and a Newton solve for the change
/// `dE` of the eccentric anomaly. Returns the new `(position_km, velocity_km_s)`.
///
/// Returns `None` if the orbit is not bound (`alpha = 2/r - v²/GM <= 0`, or
/// a value is not finite): the elliptic formulas do not apply. The caller
/// must decide what to do (the CPU backend reports an error).
///
/// Newton's iteration stops when the step is below `4 * T::EPSILON * max(1, |dE|)`
/// or after 40 iterations. In f32 the rounding noise can keep the step above
/// that limit, so after 40 iterations the last value is used. For orbits that
/// are not close to a parabola (any `e` below about 0.9) and `dt_s` of at most
/// a fraction of the period, 3 or 4 iterations are enough.
///
/// `dt_s` may be negative. A time of several periods is allowed.
pub fn kepler_drift<T: Real>(
    gm_km3_s2: T,
    pos_km: Vec3<T>,
    vel_km_s: Vec3<T>,
    dt_s: T,
) -> Option<(Vec3<T>, Vec3<T>)> {
    let two = T::from_f64(2.0);
    let r0 = pos_km.norm();
    let v2 = vel_km_s.norm_squared();
    let alpha = two / r0 - v2 / gm_km3_s2;
    // False for NaN, infinity, zero and negative alpha.
    let bound = alpha.to_f64().is_finite() && alpha > T::ZERO;
    if !bound {
        return None;
    }
    let a = T::ONE / alpha;
    let n = (gm_km3_s2 / (a * a * a)).sqrt();
    let sqrt_gma = (gm_km3_s2 * a).sqrt();
    // e cos E0 and e sin E0 at the start.
    let ecos0 = T::ONE - r0 / a;
    let esin0 = pos_km.dot(vel_km_s) / sqrt_gma;
    let m = n * dt_s;

    // Solve F(x) = x - ecos0 sin x + esin0 (1 - cos x) - m = 0. F' = r/a > 0.
    let mut x = m;
    for _ in 0..40 {
        let (s, c) = (x.sin(), x.cos());
        let f = x - ecos0 * s + esin0 * (T::ONE - c) - m;
        let fp = T::ONE - ecos0 * c + esin0 * s;
        let dx = f / fp;
        x -= dx;
        if dx.abs() <= T::from_f64(4.0) * T::EPSILON * T::ONE.max(x.abs()) {
            break;
        }
    }
    if !x.to_f64().is_finite() {
        return None;
    }
    let (s, c) = (x.sin(), x.cos());
    // 1 - cos x without cancellation for small x.
    let sh = (T::from_f64(0.5) * x).sin();
    let one_minus_c = two * sh * sh;
    let r = a * (T::ONE - ecos0 * c + esin0 * s);
    let f = T::ONE - (a / r0) * one_minus_c;
    let g = dt_s - (x - s) / n;
    let fdot = -(sqrt_gma / (r * r0)) * s;
    let gdot = T::ONE - (a / r) * one_minus_c;
    let pos = Vec3::new(
        f * pos_km.x + g * vel_km_s.x,
        f * pos_km.y + g * vel_km_s.y,
        f * pos_km.z + g * vel_km_s.z,
    );
    let vel = Vec3::new(
        fdot * pos_km.x + gdot * vel_km_s.x,
        fdot * pos_km.y + gdot * vel_km_s.y,
        fdot * pos_km.z + gdot * vel_km_s.z,
    );
    Some((pos, vel))
}
