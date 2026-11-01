//! Hermite interpolation of a moon track between two table nodes.
//!
//! One interval runs from node 0 to node 1, `h_s` seconds long. The
//! fraction `s = (t - t0) / h_s` is in [0, 1]. The functions are generic over
//! [`Real`] so that the GPU shader has a Rust twin in f32.
//!
//! - [`hermite5`] is the quintic polynomial that matches position, velocity
//!   and acceleration at both nodes (ICR-002). Its error is about
//!   `f''''''(t) (h/2)^6 / 720` at mid-interval. For a moon on a circular orbit
//!   (radius r, mean motion n) that is `r (n h)^6 / 46080`.
//! - [`hermite3`] is the cubic polynomial from positions and velocities only.
//!   Its mid-interval error is `r (n h)^4 / 384`. It is kept for comparison
//!   tests: the cubic needs 300 s for 1e-3 km, the quintic 1800 s.
//!
//! The position is written as `p0 + (change)`, so no digits are lost to the
//! subtraction of two large numbers (f32 positions near 1.4e5 km have a
//! resolution of 0.016 km).

use crate::{Real, Vec3};

#[inline]
fn c<T: Real>(x: f64) -> T {
    T::from_f64(x)
}

/// Quintic Hermite interpolation. Returns `(position, velocity)` at the
/// fraction `s` of an interval of `h_s` seconds.
///
/// Units follow the inputs: positions in km, velocities in km/s,
/// accelerations in km/s². `s` outside [0, 1] extrapolates.
#[allow(clippy::too_many_arguments)]
pub fn hermite5<T: Real>(
    p0_km: Vec3<T>,
    v0_km_s: Vec3<T>,
    a0_km_s2: Vec3<T>,
    p1_km: Vec3<T>,
    v1_km_s: Vec3<T>,
    a1_km_s2: Vec3<T>,
    h_s: T,
    s: T,
) -> (Vec3<T>, Vec3<T>) {
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s2 * s2;
    let s5 = s4 * s;
    // Basis functions of the data at node 0 (h1, h2) and node 1 (h3, h4, h5);
    // h0 = 1 - h3 is folded into the "p0 +" form.
    let h1 = s - c::<T>(6.0) * s3 + c::<T>(8.0) * s4 - c::<T>(3.0) * s5;
    let h2 = c::<T>(0.5) * s2 - c::<T>(1.5) * s3 + c::<T>(1.5) * s4 - c::<T>(0.5) * s5;
    let h3 = c::<T>(10.0) * s3 - c::<T>(15.0) * s4 + c::<T>(6.0) * s5;
    let h4 = -c::<T>(4.0) * s3 + c::<T>(7.0) * s4 - c::<T>(3.0) * s5;
    let h5 = c::<T>(0.5) * s3 - s4 + c::<T>(0.5) * s5;
    // Derivatives with respect to s.
    let d1 = T::ONE - c::<T>(18.0) * s2 + c::<T>(32.0) * s3 - c::<T>(15.0) * s4;
    let d2 = s - c::<T>(4.5) * s2 + c::<T>(6.0) * s3 - c::<T>(2.5) * s4;
    let d3 = c::<T>(30.0) * s2 - c::<T>(60.0) * s3 + c::<T>(30.0) * s4;
    let d4 = -c::<T>(12.0) * s2 + c::<T>(28.0) * s3 - c::<T>(15.0) * s4;
    let d5 = c::<T>(1.5) * s2 - c::<T>(4.0) * s3 + c::<T>(2.5) * s4;

    let dp = p1_km - p0_km;
    let h2s = h_s * h_s;
    let pos = p0_km
        + dp * h3
        + v0_km_s * (h_s * h1)
        + v1_km_s * (h_s * h4)
        + a0_km_s2 * (h2s * h2)
        + a1_km_s2 * (h2s * h5);
    let vel = dp * (d3 / h_s)
        + v0_km_s * d1
        + v1_km_s * d4
        + a0_km_s2 * (h_s * d2)
        + a1_km_s2 * (h_s * d5);
    (pos, vel)
}

/// Cubic Hermite interpolation from positions and velocities. Returns
/// `(position, velocity)` at the fraction `s` of an interval of `h_s` seconds.
pub fn hermite3<T: Real>(
    p0_km: Vec3<T>,
    v0_km_s: Vec3<T>,
    p1_km: Vec3<T>,
    v1_km_s: Vec3<T>,
    h_s: T,
    s: T,
) -> (Vec3<T>, Vec3<T>) {
    let dp = p1_km - p0_km;
    let hv0 = v0_km_s * h_s;
    let hv1 = v1_km_s * h_s;
    // p(s) = p0 + hv0 s + c2 s² + c3 s³ (in units of km).
    let c2 = dp * c::<T>(3.0) - hv0 * c::<T>(2.0) - hv1;
    let c3 = hv0 + hv1 - dp * c::<T>(2.0);
    let pos = p0_km + hv0 * s + c2 * (s * s) + c3 * (s * s * s);
    let vel = (hv0 + c2 * (c::<T>(2.0) * s) + c3 * (c::<T>(3.0) * s * s)) / h_s;
    (pos, vel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A degree-5 polynomial in time, per component, with its derivatives.
    fn poly(t: f64) -> (Vec3<f64>, Vec3<f64>, Vec3<f64>) {
        let co = [
            [1.0e5, 16.0, 3.0e-3, -2.0e-6, 1.0e-9, -4.0e-13],
            [-3.0e4, 2.0, -1.0e-3, 5.0e-7, 3.0e-10, 2.0e-13],
            [10.0, 0.1, 1.0e-4, 1.0e-7, -1.0e-10, 1.0e-14],
        ];
        let f = |c: &[f64; 6]| {
            let p = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
            let v = c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * (4.0 * c[4] + t * 5.0 * c[5])));
            let a = 2.0 * c[2] + t * (6.0 * c[3] + t * (12.0 * c[4] + t * 20.0 * c[5]));
            (p, v, a)
        };
        let r: Vec<_> = co.iter().map(f).collect();
        (
            Vec3::new(r[0].0, r[1].0, r[2].0),
            Vec3::new(r[0].1, r[1].1, r[2].1),
            Vec3::new(r[0].2, r[1].2, r[2].2),
        )
    }

    #[test]
    fn quintic_reproduces_a_quintic() {
        let (t0, t1) = (100.0, 1900.0);
        let (p0, v0, a0) = poly(t0);
        let (p1, v1, a1) = poly(t1);
        for s in [0.0, 0.1, 0.5, 0.77, 1.0] {
            let (p, v) = hermite5(p0, v0, a0, p1, v1, a1, t1 - t0, s);
            let (pe, ve, _) = poly(t0 + s * (t1 - t0));
            assert!((p - pe).norm() < 1e-9, "s={s}: {:e}", (p - pe).norm());
            assert!((v - ve).norm() < 1e-12, "s={s}: {:e}", (v - ve).norm());
        }
    }

    #[test]
    fn cubic_reproduces_a_cubic() {
        let p = |t: f64| Vec3::new(1.0e5 + 3.0 * t - 1e-3 * t * t + 2e-7 * t * t * t, 0.0, 1.0);
        let v = |t: f64| Vec3::new(3.0 - 2e-3 * t + 6e-7 * t * t, 0.0, 0.0);
        let (p_i, v_i) = hermite3(p(0.0), v(0.0), p(600.0), v(600.0), 600.0, 0.3);
        assert!((p_i - p(180.0)).norm() < 1e-9);
        assert!((v_i - v(180.0)).norm() < 1e-12);
    }

    #[test]
    fn matches_the_nodes() {
        let (p0, v0, a0) = poly(0.0);
        let (p1, v1, a1) = poly(1800.0);
        let (p, v) = hermite5(p0, v0, a0, p1, v1, a1, 1800.0, 0.0);
        assert_eq!((p, v), (p0, v0));
        let (p, v) = hermite5(p0, v0, a0, p1, v1, a1, 1800.0, 1.0);
        assert!((p - p1).norm() < 1e-9 && (v - v1).norm() < 1e-12);
    }

    #[test]
    fn f32_twin_agrees_with_f64() {
        let (p0, v0, a0) = poly(0.0);
        let (p1, v1, a1) = poly(1800.0);
        let (pd, _) = hermite5(p0, v0, a0, p1, v1, a1, 1800.0, 0.37);
        let (pf, _) = hermite5::<f32>(
            p0.cast(),
            v0.cast(),
            a0.cast(),
            p1.cast(),
            v1.cast(),
            a1.cast(),
            1800.0,
            0.37,
        );
        // f32 resolution at 1e5 km is 0.0078 km.
        assert!((pf.cast::<f64>() - pd).norm() < 0.05);
    }
}
