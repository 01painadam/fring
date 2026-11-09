//! Saturn's gravity field: point mass plus J2, J4 and J6 (`docs/PLAN.md`
//! section 7.1).

use crate::{Real, SaturnParams, Vec3};

/// The field constants, converted to the float type `T`.
///
/// Potential per unit mass, with r the distance from Saturn's centre,
/// u = z/r and q = R/r:
/// `Phi = -(GM/r) [1 - J2 q^2 P2(u) - J4 q^4 P4(u) - J6 q^6 P6(u)]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaturnField<T: Real> {
    /// GM of Saturn in km³ s⁻².
    pub gm_km3_s2: T,
    /// Reference radius R in km.
    pub reference_radius_km: T,
    /// J2 (no unit).
    pub j2: T,
    /// J4 (no unit).
    pub j4: T,
    /// J6 (no unit).
    pub j6: T,
    /// GM of the Kepler drift of the Wisdom-Holman split, in km³ s⁻²
    /// (ICR-011). Equal to `gm_km3_s2` unless [`with_drift_radius`] was
    /// called.
    ///
    /// [`with_drift_radius`]: SaturnField::with_drift_radius
    pub drift_gm_km3_s2: T,
    /// `drift_gm_km3_s2 / gm_km3_s2 - 1` (no unit), computed in f64 and then
    /// converted, so that an f32 field keeps it accurate. Zero for
    /// [`from_params`](SaturnField::from_params).
    pub drift_gm_excess: T,
}

impl<T: Real> SaturnField<T> {
    /// Converts the f64 parameters to `T` (rounded to nearest).
    pub fn from_params(p: &SaturnParams) -> Self {
        Self {
            gm_km3_s2: T::from_f64(p.gm_km3_s2),
            reference_radius_km: T::from_f64(p.reference_radius_km),
            j2: T::from_f64(p.j2),
            j4: T::from_f64(p.j4),
            j6: T::from_f64(p.j6),
            drift_gm_km3_s2: T::from_f64(p.gm_km3_s2),
            drift_gm_excess: T::ZERO,
        }
    }

    /// The same field with the Wisdom-Holman drift about
    /// `mu_eff = n(a0)^2 a0^3` (ICR-011), where n(a0) is the exact in-plane
    /// circular rate of the J2, J4, J6 field:
    /// `mu_eff = GM (1 + 3/2 J2 q^2 - 15/8 J4 q^4 + 35/16 J6 q^6)`, q = R/a0.
    /// `drift_gm_km3_s2` and `drift_gm_excess` come from one f64 computation.
    /// The total force does not change.
    pub fn with_drift_radius(self, a0_km: T) -> Self {
        let q = self.reference_radius_km.to_f64() / a0_km.to_f64();
        let q2 = q * q;
        let q4 = q2 * q2;
        let q6 = q4 * q2;
        let excess = 1.5 * self.j2.to_f64() * q2 - 15.0 / 8.0 * self.j4.to_f64() * q4
            + 35.0 / 16.0 * self.j6.to_f64() * q6;
        Self {
            drift_gm_km3_s2: T::from_f64(self.gm_km3_s2.to_f64() * (1.0 + excess)),
            drift_gm_excess: T::from_f64(excess),
            ..self
        }
    }

    /// Potential per unit mass in km² s⁻² at `pos_km` (Saturn-centred,
    /// equatorial frame). Zero at infinity.
    pub fn potential(&self, pos_km: Vec3<T>) -> T {
        let r2 = pos_km.norm_squared();
        let r = r2.sqrt();
        let u = pos_km.z / r;
        let u2 = u * u;
        let q = self.reference_radius_km / r;
        let q2 = q * q;
        let p2 = (c::<T>(3.0) * u2 - T::ONE) * c(0.5);
        let p4 = ((c::<T>(35.0) * u2 - c(30.0)) * u2 + c(3.0)) * c(0.125);
        let p6 = (((c::<T>(231.0) * u2 - c(315.0)) * u2 + c(105.0)) * u2 - c(5.0)) * c(0.0625);
        let q4 = q2 * q2;
        let s = self.j2 * q2 * p2 + self.j4 * q4 * p4 + self.j6 * q4 * q2 * p6;
        -(self.gm_km3_s2 / r) * (T::ONE - s)
    }

    /// Acceleration in km s⁻² at `pos_km`: `-grad(Phi)`, in closed form.
    ///
    /// With `A = sum (n+1) Jn q^n Pn(u)` and `B = sum Jn q^n Pn'(u)`:
    /// `a = -(GM/r²) [ (1 - A - u B) pos/r + B z_hat ]`.
    pub fn acceleration(&self, pos_km: Vec3<T>) -> Vec3<T> {
        self.accel(pos_km, true)
    }

    /// Acceleration in km s⁻² without the point-mass term: only the J2, J4
    /// and J6 parts. The Wisdom-Holman kick uses this; the Kepler drift
    /// carries the point mass.
    pub fn acceleration_without_point_mass(&self, pos_km: Vec3<T>) -> Vec3<T> {
        self.accel(pos_km, false)
    }

    /// The kick acceleration of the Saturn field for a Wisdom-Holman drift
    /// about `drift_gm_km3_s2` (ICR-011): the J terms plus
    /// `(drift_gm - GM) r / |r|^3`. For [`from_params`](Self::from_params)
    /// it is [`acceleration_without_point_mass`](Self::acceleration_without_point_mass),
    /// bit for bit.
    pub fn kick_acceleration_saturn(&self, pos_km: Vec3<T>) -> Vec3<T> {
        // point = -excess: the point-mass factor of the full force is 1 and
        // the drift carries 1 + excess of it.
        self.accel_with(pos_km, T::ZERO - self.drift_gm_excess)
    }

    #[inline]
    fn accel(&self, pos_km: Vec3<T>, with_point_mass: bool) -> Vec3<T> {
        self.accel_with(pos_km, if with_point_mass { T::ONE } else { T::ZERO })
    }

    /// `point` multiplies the point-mass term: 1 for the full force, 0 for
    /// the J terms only.
    #[inline]
    fn accel_with(&self, pos_km: Vec3<T>, point: T) -> Vec3<T> {
        let r2 = pos_km.norm_squared();
        let r = r2.sqrt();
        let u = pos_km.z / r;
        let u2 = u * u;
        let q = self.reference_radius_km / r;
        let q2 = q * q;
        let q4 = q2 * q2;
        let q6 = q4 * q2;
        let p2 = (c::<T>(3.0) * u2 - T::ONE) * c(0.5);
        let p4 = ((c::<T>(35.0) * u2 - c(30.0)) * u2 + c(3.0)) * c(0.125);
        let p6 = (((c::<T>(231.0) * u2 - c(315.0)) * u2 + c(105.0)) * u2 - c(5.0)) * c(0.0625);
        // Derivatives dPn/du.
        let d2 = c::<T>(3.0) * u;
        let d4 = ((c::<T>(35.0) * u2 - c(15.0)) * u) * c(0.5);
        let d6 = (((c::<T>(1386.0) * u2 - c(1260.0)) * u2 + c(210.0)) * u) * c(0.0625);
        let j2q = self.j2 * q2;
        let j4q = self.j4 * q4;
        let j6q = self.j6 * q6;
        let a = c::<T>(3.0) * j2q * p2 + c::<T>(5.0) * j4q * p4 + c::<T>(7.0) * j6q * p6;
        let b = j2q * d2 + j4q * d4 + j6q * d6;
        let radial = point - a - u * b;
        let k = -self.gm_km3_s2 / (r2 * r);
        Vec3::new(
            k * radial * pos_km.x,
            k * radial * pos_km.y,
            k * (radial * pos_km.z + b * r),
        )
    }

    /// Energy per unit mass in km² s⁻²: `|v|²/2 + Phi`.
    pub fn specific_energy(&self, pos_km: Vec3<T>, vel_km_s: Vec3<T>) -> T {
        c::<T>(0.5) * vel_km_s.norm_squared() + self.potential(pos_km)
    }
}

/// A constant of type `T`.
#[inline]
fn c<T: Real>(x: f64) -> T {
    T::from_f64(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::saturn_from_toml_str;
    use crate::physics::frequencies;

    const A0: f64 = 140_220.0;

    fn field() -> SaturnField<f64> {
        let s = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
        SaturnField::from_params(&s)
    }

    #[test]
    fn from_params_keeps_the_point_mass_drift() {
        let f = field();
        assert_eq!(f.drift_gm_km3_s2, f.gm_km3_s2);
        assert_eq!(f.drift_gm_excess.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn mu_eff_matches_the_circular_rate_and_the_icr_number() {
        let f = field().with_drift_radius(A0);
        let n = frequencies(&f, A0).n_rad_s;
        let mu = n * n * A0 * A0 * A0;
        assert!(((f.drift_gm_km3_s2 - mu) / mu).abs() < 1e-14, "{mu}");
        let rel =
            (f.gm_km3_s2 * (1.0 + f.drift_gm_excess) - f.drift_gm_km3_s2).abs() / f.drift_gm_km3_s2;
        assert!(rel < 1e-15, "{rel:e}");
        assert!((f.drift_gm_km3_s2 - 38_105_113.13).abs() < 0.01);
        assert!((f.drift_gm_excess - 4.5848e-3).abs() < 1e-7);
        // The total force does not change.
        let p = Vec3::new(140_000.0, 3.0e4, 50.0);
        assert_eq!(f.acceleration(p), field().acceleration(p));
    }

    /// For `from_params` the kick is the old J-only acceleration, bit for bit
    /// (`point` is +0.0 in both).
    #[test]
    fn kick_is_bit_identical_for_from_params() {
        let f = field();
        for i in 0..200 {
            let t = f64::from(i) * 0.37;
            let r = 60_000.0 + 1_000.0 * f64::from(i);
            let p = Vec3::new(r * t.cos(), r * t.sin(), 0.05 * r * (2.1 * t).sin());
            let (a, b) = (
                f.kick_acceleration_saturn(p),
                f.acceleration_without_point_mass(p),
            );
            for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    /// kick + drift point mass = the full force; on the circle at a0 the
    /// kick radial part is zero.
    #[test]
    fn kick_plus_drift_is_the_full_force() {
        let f = field().with_drift_radius(A0);
        let p = Vec3::new(139_800.0, 2.0e4, 300.0);
        let r = p.norm();
        let sum = f.kick_acceleration_saturn(p) + p * (-f.drift_gm_km3_s2 / (r * r * r));
        let full = f.acceleration(p);
        assert!((sum - full).norm() / full.norm() < 1e-14);
        let kick = f.kick_acceleration_saturn(Vec3::new(A0, 0.0, 0.0));
        let g = f.gm_km3_s2 / (A0 * A0);
        assert!(kick.x.abs() < 1e-12 * g, "{:e}", kick.x / g);
    }
}
