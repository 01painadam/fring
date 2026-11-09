//! Frame conversions (`docs/PLAN.md` section 4.2, `docs/snapshot-format.md`
//! section 6): inertial, rotating and local (guiding-centre) coordinates.
//!
//! - **Inertial:** Saturn-centred, equatorial.
//! - **Rotating:** turns about +z at `n0` and coincides with the inertial
//!   frame at time 0. Velocities are relative to the rotating frame:
//!   `v_rot = R(-theta) v_in - n0 z_hat x r_rot`, with `theta = n0 t`.
//! - **Local:** curvilinear, around a reference circle of radius `a0` in the
//!   rotating frame. With `rho = hypot(x, y)` and `phi = atan2(y, x)` (in
//!   (-pi, pi]) of the rotating frame: `x = rho - a0`, `y = a0 phi`, `z = z`;
//!   velocities `vx = d rho/dt`, `vy = a0 d phi/dt`, `vz = dz/dt`. The map is
//!   exactly invertible except on the axis (`rho = 0`). In f32 the value
//!   of `y` is large for the full ring (up to pi a0 = 4.4e5 km, resolution
//!   0.03 km), which suits arcs better.

use super::circular::frequencies;
use super::gravity::SaturnField;
use crate::real::sin_cos;
use crate::{Real, Vec3};

/// A state in local coordinates (km and km/s).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LocalState<T: Real> {
    /// Radial offset `rho - a0`, km.
    pub x_km: T,
    /// Azimuthal arc length `a0 phi`, km.
    pub y_km: T,
    /// Height, km.
    pub z_km: T,
    /// Radial velocity, km/s.
    pub vx_km_s: T,
    /// Azimuthal velocity `a0 d phi/dt`, km/s.
    pub vy_km_s: T,
    /// Vertical velocity, km/s.
    pub vz_km_s: T,
}

/// The mean motion n0 in rad/s of the reference circular orbit at `a0_km`
/// (with J2, J4 and J6; `docs/PLAN.md` section 7.2). The rotating frame turns at n0.
pub fn reference_mean_motion<T: Real>(field: &SaturnField<T>, a0_km: T) -> T {
    frequencies(field, a0_km).n_rad_s
}

/// Rotates `(x, y)` by the angle with the given cosine and sine.
#[inline]
fn rot<T: Real>(v: Vec3<T>, cos: T, sin: T) -> Vec3<T> {
    Vec3::new(cos * v.x - sin * v.y, sin * v.x + cos * v.y, v.z)
}

/// Inertial to rotating frame at time `t_s` (seconds since the epoch).
/// Returns `(position_km, velocity_km_s)` in the rotating frame.
pub fn inertial_to_rotating<T: Real>(
    pos_km: Vec3<T>,
    vel_km_s: Vec3<T>,
    n0_rad_s: T,
    t_s: T,
) -> (Vec3<T>, Vec3<T>) {
    let theta = n0_rad_s * t_s;
    let (s, c) = sin_cos(theta);
    let p = rot(pos_km, c, -s);
    let v = rot(vel_km_s, c, -s);
    // minus n0 z_hat x r_rot = + n0 (y, -x, 0)
    (
        p,
        Vec3::new(v.x + n0_rad_s * p.y, v.y - n0_rad_s * p.x, v.z),
    )
}

/// Rotating to inertial frame at time `t_s`. The inverse of [`inertial_to_rotating`].
pub fn rotating_to_inertial<T: Real>(
    pos_km: Vec3<T>,
    vel_km_s: Vec3<T>,
    n0_rad_s: T,
    t_s: T,
) -> (Vec3<T>, Vec3<T>) {
    let theta = n0_rad_s * t_s;
    let (s, c) = sin_cos(theta);
    // add n0 z_hat x r_rot = n0 (-y, x, 0)
    let v = Vec3::new(
        vel_km_s.x - n0_rad_s * pos_km.y,
        vel_km_s.y + n0_rad_s * pos_km.x,
        vel_km_s.z,
    );
    (rot(pos_km, c, s), rot(v, c, s))
}

/// Rotating frame to local coordinates around the circle of radius `a0_km`.
pub fn rotating_to_local<T: Real>(pos_km: Vec3<T>, vel_km_s: Vec3<T>, a0_km: T) -> LocalState<T> {
    let rho2 = pos_km.x * pos_km.x + pos_km.y * pos_km.y;
    let rho = rho2.sqrt();
    let phi = pos_km.y.atan2(pos_km.x);
    let rho_dot = (pos_km.x * vel_km_s.x + pos_km.y * vel_km_s.y) / rho;
    let phi_dot = (pos_km.x * vel_km_s.y - pos_km.y * vel_km_s.x) / rho2;
    LocalState {
        x_km: rho - a0_km,
        y_km: a0_km * phi,
        z_km: pos_km.z,
        vx_km_s: rho_dot,
        vy_km_s: a0_km * phi_dot,
        vz_km_s: vel_km_s.z,
    }
}

/// Local coordinates back to the rotating frame. The inverse of [`rotating_to_local`].
pub fn local_to_rotating<T: Real>(l: &LocalState<T>, a0_km: T) -> (Vec3<T>, Vec3<T>) {
    let rho = l.x_km + a0_km;
    let phi = l.y_km / a0_km;
    let (s, c) = sin_cos(phi);
    let phi_dot = l.vy_km_s / a0_km;
    (
        Vec3::new(rho * c, rho * s, l.z_km),
        Vec3::new(
            l.vx_km_s * c - rho * phi_dot * s,
            l.vx_km_s * s + rho * phi_dot * c,
            l.vz_km_s,
        ),
    )
}
