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
        let radial = T::ONE - a - u * b;
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
