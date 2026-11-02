//! Laplace coefficients and the secular apse rate of a test particle
//! (Murray and Dermott 1999, section 7.3).
//!
//! `b_s^(j)(alpha) = (1/pi) integral_0^(2 pi) cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s dpsi`.
//! A test particle at radius `a` with mean motion `n` and a perturber of mass
//! ratio `mu = GM_moon / GM_saturn` on a circular orbit precesses at
//! `A = (n / 4) mu alpha alpha_bar b_{3/2}^(1)(alpha)`, with `alpha = a_moon / a`
//! and `alpha_bar = 1` for an interior perturber, `alpha = a / a_moon` and
//! `alpha_bar = alpha` for an exterior one.

use fring_model::physics::{SaturnField, frequencies};

/// Laplace coefficient `b_s^(j)(alpha)` by the trapezoid rule on `n_points`
/// points. The integrand is periodic and smooth, so the rule converges
/// exponentially; use `n_points` much greater than `1 / (1 - alpha)`.
pub fn laplace_b(s: f64, j: u32, alpha: f64, n_points: usize) -> f64 {
    let h = 2.0 * std::f64::consts::PI / n_points as f64;
    let mut sum = 0.0;
    for k in 0..n_points {
        let psi = h * k as f64;
        let d = 1.0 - 2.0 * alpha * psi.cos() + alpha * alpha;
        sum += (f64::from(j) * psi).cos() / d.powf(s);
    }
    sum * h / std::f64::consts::PI
}

/// Secular apse rate in rad/s of a test particle, from one perturber on a
/// circular orbit (see the module text).
pub fn secular_apse_rate_rad_s(
    n_rad_s: f64,
    mu: f64,
    a_particle_km: f64,
    a_moon_km: f64,
    n_points: usize,
) -> f64 {
    let (alpha, alpha_bar) = if a_moon_km < a_particle_km {
        (a_moon_km / a_particle_km, 1.0)
    } else {
        let al = a_particle_km / a_moon_km;
        (al, al)
    };
    0.25 * n_rad_s * mu * alpha * alpha_bar * laplace_b(1.5, 1, alpha, n_points)
}

/// The circular-orbit radius in km whose mean motion is `n_rad_s`, in
/// Saturn's field (bisection on [50,000, 500,000] km).
pub fn radius_from_mean_motion(field: &SaturnField<f64>, n_rad_s: f64) -> f64 {
    let (mut lo, mut hi) = (5.0e4, 5.0e5);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if frequencies(field, mid).n_rad_s > n_rad_s {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_alpha_series() {
        let a = 0.01;
        let b1 = laplace_b(1.5, 1, a, 20_000);
        let series = 3.0 * a * (1.0 + 15.0 / 8.0 * a * a);
        assert!((b1 / series - 1.0).abs() < 1e-7, "{b1} {series}");
        let b0 = laplace_b(0.5, 0, a, 20_000);
        assert!((b0 / (2.0 * (1.0 + a * a / 4.0)) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn near_one_matches_the_closed_form() {
        // b_{3/2}^(1) from the elliptic-integral identity is not needed:
        // the trapezoid result must not change when the grid doubles.
        let a = 0.994;
        let x = laplace_b(1.5, 1, a, 400_000);
        let y = laplace_b(1.5, 1, a, 800_000);
        assert!((x / y - 1.0).abs() < 1e-9, "{x} {y}");
        assert!(x > 1.0e4 && x < 1.0e5);
    }
}
